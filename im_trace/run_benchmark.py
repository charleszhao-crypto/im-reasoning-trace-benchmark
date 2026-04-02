#!/usr/bin/env python3
"""
IM-TRACE End-to-End Benchmark Runner

Orchestrates the full evaluation pipeline:
  1. Load cases from JSONL
  2. Load or generate model responses
  3. Score each response with mock annotations (absolute scoring)
  4. Run pairwise comparisons between models
  5. Fit Bradley-Terry ratings
  6. Compute bootstrap confidence intervals
  7. Export leaderboard JSON

Usage:
  python -m im_trace.run_benchmark \
    --cases im_trace/cases/raw/sample_cases.jsonl \
    --responses im_trace/cases/processed/mock_responses.jsonl \
    --output im_trace/results/runs/

This script uses the replay adapter (pre-generated responses) by default.
For live model evaluation, use the appropriate adapter.
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from im_trace.cases.schema.models import (
    ClinicalCase, ModelResponse, PairwiseComparisonRecord,
    RubricPreference, HumanAnnotation, R1Score, R2Score, R3Score, R4Score,
    R4SubscaleScore, SafetyItemScore, EffectivenessItemScore,
    OrdinalScore, ConfidenceLevel, JudgeType, BenchmarkRunSummary,
)
from im_trace.evaluators.absolute.scorer import compute_total, format_scorecard
from im_trace.evaluators.aggregation.aggregate import (
    bootstrap_ci, aggregate_naive, fit_bradley_terry, bt_to_elo, rank_models,
    identify_hard_cases,
)


# ── Data Loading ────────────────────────────────────────────────────────────

def load_cases(path: Path) -> list[ClinicalCase]:
    """Load clinical cases from JSONL."""
    cases = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(ClinicalCase.model_validate_json(line))
    return cases


def load_responses(path: Path) -> list[ModelResponse]:
    """Load model responses from JSONL."""
    responses = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                responses.append(ModelResponse.model_validate_json(line))
    return responses


# ── Mock Annotation Generator ───────────────────────────────────────────────

SAFETY_CRITERIA = [
    "contraindication_check", "drug_interaction_screening", "red_flag_recognition",
    "dosing_safety", "allergy_cross_reactivity", "harm_escalation_awareness",
]
EFFECTIVENESS_CRITERIA = [
    "diagnostic_accuracy", "workup_appropriateness", "treatment_guideline_adherence",
    "monitoring_plan", "patient_education", "disposition_appropriateness",
]
R4_SUBSCALES = [
    "ddx_construction", "pretest_probability_calibration", "evidence_integration",
    "diagnostic_closure", "epistemic_humility",
]


def generate_mock_annotation(
    case: ClinicalCase,
    response: ModelResponse,
    seed: int = 42,
) -> HumanAnnotation:
    """
    Generate a deterministic mock annotation for testing the pipeline.

    Scoring heuristic:
      - Check if response mentions must_consider_diagnoses → R4 ddx score
      - Check if response mentions red_flags → R3 safety score
      - Check if response contains unacceptable_errors → R3 safety = 0
      - Check abstention_expected vs. whether model abstained

    This is NOT a real evaluator — it's a pipeline test fixture.
    """
    rng = random.Random(seed)
    text = response.response_text.lower()

    # R1: check if response contains obvious factual errors
    has_unacceptable = any(err.lower() in text for err in case.unacceptable_errors)
    r1_score = OrdinalScore.INADEQUATE if has_unacceptable else OrdinalScore(rng.choice([1, 2]))

    # R2: check if response addresses the chief complaint
    r2_score = OrdinalScore(rng.choice([1, 2]))

    # R3: check safety items
    safety_items = []
    for criterion in SAFETY_CRITERIA:
        if criterion == "red_flag_recognition":
            # Check if any red flags are mentioned
            flags_mentioned = sum(1 for rf in case.red_flags if rf.lower() in text)
            score = OrdinalScore.COMPLETE if flags_mentioned > 0 else OrdinalScore(rng.choice([0, 1]))
        elif has_unacceptable:
            score = OrdinalScore.INADEQUATE
        else:
            score = OrdinalScore(rng.choice([1, 2]))
        safety_items.append(SafetyItemScore(criterion=criterion, score=score))

    eff_items = []
    for criterion in EFFECTIVENESS_CRITERIA:
        score = OrdinalScore(rng.choice([1, 2]))
        eff_items.append(EffectivenessItemScore(criterion=criterion, score=score))

    r3 = R3Score(safety_items=safety_items, effectiveness_items=eff_items)

    # R4: check differential diagnosis quality
    ddx_mentioned = sum(1 for dx in case.must_consider_diagnoses if dx.lower() in text)
    ddx_ratio = ddx_mentioned / max(len(case.must_consider_diagnoses), 1)

    ddx_score = OrdinalScore.COMPLETE if ddx_ratio >= 0.7 else (OrdinalScore.PARTIAL if ddx_ratio >= 0.3 else OrdinalScore.INADEQUATE)

    # Abstention check
    abstention_words = ["uncertain", "more information", "unclear", "cannot determine", "insufficient"]
    model_abstained = any(w in text for w in abstention_words)
    humility_score = OrdinalScore.COMPLETE if (case.abstention_expected and model_abstained) else OrdinalScore(rng.choice([0, 1, 2]))

    r4 = R4Score(
        ddx_construction=R4SubscaleScore(subscale="ddx_construction", score=ddx_score, rationale="mock"),
        pretest_probability_calibration=R4SubscaleScore(subscale="pretest_probability_calibration", score=OrdinalScore(rng.choice([1, 2])), rationale="mock"),
        evidence_integration=R4SubscaleScore(subscale="evidence_integration", score=OrdinalScore(rng.choice([1, 2])), rationale="mock"),
        diagnostic_closure=R4SubscaleScore(subscale="diagnostic_closure", score=OrdinalScore(rng.choice([1, 2])), rationale="mock"),
        epistemic_humility=R4SubscaleScore(subscale="epistemic_humility", score=humility_score, rationale="mock"),
    )

    return HumanAnnotation(
        case_id=case.case_id,
        response_id=response.response_id,
        model_id=response.model_id,
        evaluator_id="mock-physician",
        r1=R1Score(score=r1_score, rationale="mock"),
        r2=R2Score(score=r2_score, rationale="mock"),
        r3=r3,
        r4=r4,
        overall_notes="Mock annotation for pipeline testing",
    )


# ── Pairwise Comparison (simplified for pipeline test) ──────────────────────

def generate_mock_pairwise(
    case_id: str,
    response_a: ModelResponse,
    response_b: ModelResponse,
    score_a: float,
    score_b: float,
    seed: int = 42,
) -> PairwiseComparisonRecord:
    """Generate a mock pairwise comparison based on absolute scores."""
    rng = random.Random(seed)
    order = "a_first" if rng.random() > 0.5 else "b_first"

    # Determine winner per rubric based on absolute scores
    diff = score_a - score_b
    if abs(diff) < 0.5:
        overall = "tie"
    elif diff > 0:
        overall = "a"
    else:
        overall = "b"

    prefs = [
        RubricPreference(rubric="r1", winner=rng.choice(["a", "b", "tie"])),
        RubricPreference(rubric="r2", winner=rng.choice(["a", "b", "tie"])),
        RubricPreference(rubric="r3", winner=rng.choice(["a", "b", "tie"])),
        RubricPreference(rubric="r4", winner=overall, rationale="Based on absolute score difference"),
        RubricPreference(rubric="overall", winner=overall, rationale=f"Score diff: {diff:.2f}"),
    ]

    return PairwiseComparisonRecord(
        case_id=case_id,
        response_a_id=response_a.response_id,
        response_b_id=response_b.response_id,
        model_a_id=response_a.model_id,
        model_b_id=response_b.model_id,
        presented_order=order,
        seed=seed,
        preferences=prefs,
        judge_type=JudgeType.PHYSICIAN,
        judge_id="mock-physician",
    )


# ── Main Pipeline ───────────────────────────────────────────────────────────

def run_pipeline(
    cases_path: Path,
    responses_path: Path,
    output_dir: Path,
    seed: int = 42,
) -> BenchmarkRunSummary:
    """Execute the full IM-TRACE benchmark pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(" IM-TRACE Benchmark Pipeline")
    print("=" * 60)

    # 1. Load data
    print("\n[1/7] Loading cases and responses...")
    cases = load_cases(cases_path)
    responses = load_responses(responses_path)
    print(f"  Loaded {len(cases)} cases, {len(responses)} responses")

    # Index responses by case_id and model_id
    resp_index: dict[tuple[str, str], ModelResponse] = {}
    for r in responses:
        resp_index[(r.case_id, r.model_id)] = r

    models = sorted(set(r.model_id for r in responses))
    print(f"  Models: {models}")

    # 2. Absolute scoring
    print("\n[2/7] Running absolute scoring...")
    all_scores: dict[str, list[dict]] = defaultdict(list)  # model_id -> [score_dicts]
    annotations_jsonl = output_dir / "annotations.jsonl"

    with open(annotations_jsonl, 'w') as ann_f:
        for case in cases:
            for model_id in models:
                key = (case.case_id, model_id)
                if key not in resp_index:
                    continue
                response = resp_index[key]

                annotation = generate_mock_annotation(case, response, seed)
                score_result = compute_total(annotation)
                score_result["case_id"] = case.case_id
                score_result["model_id"] = model_id
                score_result["judge_type"] = "physician"
                all_scores[model_id].append(score_result)

                # Append annotation to JSONL
                ann_f.write(annotation.model_dump_json() + "\n")

                print(f"  {case.case_id} / {model_id}: {score_result['total']:.2f}/9.0"
                      f" {'⚠ SAFETY CAP' if score_result['safety_capped'] else ''}")

    # 3. Per-model aggregation with bootstrap CI
    print("\n[3/7] Aggregating scores with bootstrap CI...")
    model_summaries = []
    for model_id in models:
        scores = all_scores[model_id]
        totals = [s["total"] for s in scores]
        mean, ci_lo, ci_hi = bootstrap_ci(totals, seed=seed)

        r1s = [s["r1"] for s in scores]
        r2s = [s["r2"] for s in scores]
        r3s = [s["r3_composite"] for s in scores]
        r4s = [s["r4_composite"] for s in scores]

        summary = {
            "model_id": model_id,
            "n_cases": len(scores),
            "total_mean": round(mean, 3),
            "total_ci_lower": round(ci_lo, 3),
            "total_ci_upper": round(ci_hi, 3),
            "r1_mean": round(sum(r1s) / len(r1s), 3) if r1s else 0,
            "r2_mean": round(sum(r2s) / len(r2s), 3) if r2s else 0,
            "r3_mean": round(sum(r3s) / len(r3s), 3) if r3s else 0,
            "r4_mean": round(sum(r4s) / len(r4s), 3) if r4s else 0,
            "safety_violations": sum(1 for s in scores if s["safety_capped"]),
        }
        model_summaries.append(summary)
        print(f"  {model_id}: {mean:.2f} [{ci_lo:.2f}, {ci_hi:.2f}]"
              f" | R1={summary['r1_mean']:.1f} R2={summary['r2_mean']:.1f}"
              f" R3={summary['r3_mean']:.2f} R4={summary['r4_mean']:.2f}")

    # 4. Pairwise comparisons
    print("\n[4/7] Running pairwise comparisons...")
    pairwise_records = []
    comparisons_jsonl = output_dir / "pairwise_comparisons.jsonl"

    with open(comparisons_jsonl, 'w') as pw_f:
        for case in cases:
            for i, model_a in enumerate(models):
                for model_b in models[i + 1:]:
                    key_a = (case.case_id, model_a)
                    key_b = (case.case_id, model_b)
                    if key_a not in resp_index or key_b not in resp_index:
                        continue

                    resp_a = resp_index[key_a]
                    resp_b = resp_index[key_b]

                    # Get absolute scores for comparison basis
                    score_a = next((s["total"] for s in all_scores[model_a] if s["case_id"] == case.case_id), 0)
                    score_b = next((s["total"] for s in all_scores[model_b] if s["case_id"] == case.case_id), 0)

                    record = generate_mock_pairwise(case.case_id, resp_a, resp_b, score_a, score_b, seed)
                    pairwise_records.append(record)
                    pw_f.write(record.model_dump_json() + "\n")

    print(f"  Generated {len(pairwise_records)} pairwise comparisons")

    # 5. Bradley-Terry fitting
    print("\n[5/7] Fitting Bradley-Terry model...")
    bt_input = []
    for record in pairwise_records:
        overall = record.overall_winner
        if overall == "a":
            bt_input.append({"winner": record.model_a_id, "loser": record.model_b_id})
        elif overall == "b":
            bt_input.append({"winner": record.model_b_id, "loser": record.model_a_id})
        # Ties: skip for BT (could split 0.5/0.5 but simpler to skip)

    if bt_input:
        strengths = fit_bradley_terry(bt_input)
        rankings = rank_models(strengths)
        for r in rankings:
            print(f"  #{r['rank']} {r['model_id']}: Elo {r['elo_rating']:.0f} (BT {r['bt_strength']:.3f})")
    else:
        rankings = []
        print("  No non-tie comparisons; skipping BT")

    # 6. Hard cases analysis
    print("\n[6/7] Identifying hard cases...")
    case_annotations: dict[str, list[dict]] = defaultdict(list)
    for model_scores in all_scores.values():
        for s in model_scores:
            case_annotations[s["case_id"]].append(s)
    hard_cases = identify_hard_cases(dict(case_annotations))
    print(f"  Found {len(hard_cases)} hard cases")
    for hc in hard_cases:
        print(f"    {hc['case_id']}: {hc['reason']}")

    # 7. Export leaderboard
    print("\n[7/7] Exporting leaderboard...")
    leaderboard = {
        "generated_at": datetime.now().isoformat(),
        "rubric_version": "1.0",
        "n_cases": len(cases),
        "n_models": len(models),
        "model_scores": model_summaries,
        "pairwise_rankings": rankings,
        "hard_cases": hard_cases,
    }

    leaderboard_path = output_dir / "leaderboard.json"
    with open(leaderboard_path, 'w') as f:
        json.dump(leaderboard, f, indent=2)
    print(f"  Saved to {leaderboard_path}")

    # Build run summary
    run_summary = BenchmarkRunSummary(
        rubric_version="1.0",
        prompt_version="1.0",
        n_cases=len(cases),
        n_models=len(models),
        models_evaluated=models,
        n_human_annotations=sum(len(v) for v in all_scores.values()),
        n_llm_annotations=0,
        model_scores=model_summaries,
        pairwise_rankings=rankings,
        n_hard_cases=len(hard_cases),
        models_with_safety_violations=[
            s["model_id"] for s in model_summaries if s["safety_violations"] > 0
        ],
    )

    summary_path = output_dir / "run_summary.json"
    with open(summary_path, 'w') as f:
        f.write(run_summary.model_dump_json(indent=2))

    # Print leaderboard table
    print(f"\n{'='*60}")
    print(" LEADERBOARD")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Total':>6} {'95% CI':>14} {'R1':>5} {'R2':>5} {'R3':>5} {'R4':>5} {'Safety':>7}")
    print(f"{'-'*60}")
    for s in sorted(model_summaries, key=lambda x: x["total_mean"], reverse=True):
        ci = f"[{s['total_ci_lower']:.1f}, {s['total_ci_upper']:.1f}]"
        safety = f"{s['safety_violations']} viol" if s['safety_violations'] else "clean"
        print(f"{s['model_id']:<20} {s['total_mean']:>6.2f} {ci:>14} {s['r1_mean']:>5.1f} {s['r2_mean']:>5.1f} {s['r3_mean']:>5.2f} {s['r4_mean']:>5.2f} {safety:>7}")

    if rankings:
        print(f"\n{'='*60}")
        print(" PAIRWISE RANKINGS (Bradley-Terry)")
        print(f"{'='*60}")
        for r in rankings:
            print(f"  #{r['rank']} {r['model_id']}: Elo {r['elo_rating']:.0f}")

    print(f"\n{'='*60}")
    print(f" Run complete. Results in {output_dir}")
    print(f"{'='*60}")

    return run_summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="IM-TRACE Benchmark Runner")
    parser.add_argument("--cases", type=Path, default=Path("im_trace/cases/raw/sample_cases.jsonl"))
    parser.add_argument("--responses", type=Path, default=Path("im_trace/cases/processed/mock_responses.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("im_trace/results/runs/demo"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_pipeline(args.cases, args.responses, args.output, args.seed)
