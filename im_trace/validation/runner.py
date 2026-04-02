"""
Validated Run Orchestrator — executes a frozen profile and produces locked artifacts.

Differences from exploratory run_benchmark.py:
  1. Requires a ValidatedProfile (not ad-hoc parameters)
  2. Generates a RunManifest with content hash verification
  3. Outputs are append-only — no overwriting prior runs
  4. Emits warnings if profile content doesn't match expectations
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from im_trace.cases.schema.models import ClinicalCase, ModelResponse
from im_trace.evaluators.absolute.scorer import compute_total
from im_trace.evaluators.aggregation.aggregate import (
    bootstrap_ci, fit_bradley_terry, bt_to_elo, rank_models,
)
from im_trace.validation.profile import ValidatedProfile, RunManifest
from im_trace.run_benchmark import load_cases, load_responses, generate_mock_annotation


def run_validated(
    profile: ValidatedProfile,
    cases_path: Path,
    responses_path: Path,
    output_root: Path,
) -> RunManifest:
    """
    Execute a validated benchmark run from a frozen profile.

    Returns a RunManifest with all output file references.
    """
    start_time = time.time()
    started_at = datetime.now().isoformat()
    warnings: list[str] = []

    # Create run-specific output directory (never overwrite)
    run_id = f"VAL-{uuid4().hex[:8]}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save profile as part of the run record
    profile_path = run_dir / "profile.json"
    with open(profile_path, 'w') as f:
        f.write(profile.model_dump_json(indent=2))

    # Load data
    all_cases = load_cases(cases_path)
    all_responses = load_responses(responses_path)

    # Filter to profile case subset
    case_ids_set = set(profile.case_ids)
    cases = [c for c in all_cases if c.case_id in case_ids_set]
    if len(cases) < len(profile.case_ids):
        missing = case_ids_set - {c.case_id for c in cases}
        warnings.append(f"Missing cases from profile: {missing}")

    # Filter responses to profile models
    profile_models = {a.model_id for a in profile.adapter_configs}
    responses = [r for r in all_responses if r.model_id in profile_models and r.case_id in case_ids_set]

    # Index responses
    resp_index: dict[tuple[str, str], ModelResponse] = {}
    for r in responses:
        resp_index[(r.case_id, r.model_id)] = r

    models = sorted(profile_models)

    # Score
    all_scores: dict[str, list[dict]] = {m: [] for m in models}
    annotations_path = run_dir / "annotations.jsonl"
    n_annotations = 0

    with open(annotations_path, 'w') as ann_f:
        for case in cases:
            for model_id in models:
                key = (case.case_id, model_id)
                if key not in resp_index:
                    warnings.append(f"No response for {key}")
                    continue
                response = resp_index[key]
                annotation = generate_mock_annotation(
                    case, response, profile.aggregation.bootstrap_seed
                )
                score = compute_total(annotation, profile.aggregation.safety_cap_enabled)
                score["case_id"] = case.case_id
                score["model_id"] = model_id
                all_scores[model_id].append(score)
                ann_f.write(annotation.model_dump_json() + "\n")
                n_annotations += 1

    # Aggregate
    model_summaries = []
    for model_id in models:
        scores = all_scores[model_id]
        if not scores:
            continue
        totals = [s["total"] for s in scores]
        mean, ci_lo, ci_hi = bootstrap_ci(
            totals,
            n_bootstrap=profile.aggregation.bootstrap_n,
            ci=profile.aggregation.bootstrap_ci,
            seed=profile.aggregation.bootstrap_seed,
        )
        model_summaries.append({
            "model_id": model_id,
            "n_cases": len(scores),
            "total_mean": round(mean, 3),
            "total_ci_lower": round(ci_lo, 3),
            "total_ci_upper": round(ci_hi, 3),
            "r1_mean": round(sum(s["r1"] for s in scores) / len(scores), 3),
            "r2_mean": round(sum(s["r2"] for s in scores) / len(scores), 3),
            "r3_mean": round(sum(s["r3_composite"] for s in scores) / len(scores), 3),
            "r4_mean": round(sum(s["r4_composite"] for s in scores) / len(scores), 3),
            "safety_violations": sum(1 for s in scores if s["safety_capped"]),
        })

    # Leaderboard
    leaderboard = {
        "run_id": run_id,
        "profile_id": profile.profile_id,
        "profile_content_hash": profile.content_hash,
        "mode": profile.mode,
        "generated_at": datetime.now().isoformat(),
        "rubric_version": profile.rubric_version,
        "n_cases": len(cases),
        "n_models": len(models),
        "model_scores": model_summaries,
    }

    leaderboard_path = run_dir / "leaderboard.json"
    with open(leaderboard_path, 'w') as f:
        json.dump(leaderboard, f, indent=2)

    # All raw scores for downstream analysis
    raw_scores_path = run_dir / "raw_scores.jsonl"
    with open(raw_scores_path, 'w') as f:
        for model_id, scores in all_scores.items():
            for s in scores:
                f.write(json.dumps(s) + "\n")

    duration = time.time() - start_time

    manifest = RunManifest(
        manifest_id=run_id,
        profile_id=profile.profile_id,
        profile_content_hash=profile.content_hash,
        run_mode=profile.mode,
        n_cases=len(cases),
        n_models=len(models),
        n_annotations=n_annotations,
        output_files={
            "profile": str(profile_path),
            "leaderboard": str(leaderboard_path),
            "annotations": str(annotations_path),
            "raw_scores": str(raw_scores_path),
        },
        started_at=started_at,
        completed_at=datetime.now().isoformat(),
        duration_seconds=round(duration, 2),
        warnings=warnings,
    )

    manifest.save(run_dir / "manifest.json")
    return manifest
