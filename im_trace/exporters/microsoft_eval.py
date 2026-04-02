"""
IM-TRACE → Microsoft Healthcare AI Model Evaluator Export.

Consumes IM-TRACE artifacts (annotations JSONL, pairwise JSONL) and emits
JSON structures compatible with Microsoft's Healthcare AI Model Evaluator
expectations:

  - Per-case evaluation records (model_id, scores, rationales, reviewers)
  - Multi-reviewer support (human + LLM mapped as separate reviewers)
  - Custom metric definitions (R1-R4 mapped as custom metrics)
  - Arena comparison records (pairwise comparisons)

No Microsoft SDK dependencies. All output is plain JSON.

References for expected schema shapes:
  https://github.com/Azure/azure-health-ai-model-evaluator (public repo)
  Healthcare AI Model Evaluator v2 schema (as of 2024-Q3)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# ── Custom Metric Definitions ─────────────────────────────────────────────────

# IM-TRACE rubric → Microsoft custom metric definition
_RUBRIC_DEFINITIONS = [
    {
        "metric_id":   "im_trace_r1_factuality",
        "display_name": "R1 Factuality",
        "description": (
            "Accuracy of medical facts in the model response. "
            "Scores 0 (major factual error), 1 (partial), 2 (complete/correct). "
            "References verified against clinical guidelines."
        ),
        "scale": {"type": "ordinal", "min": 0, "max": 2, "labels": {
            "0": "Inadequate — major factual deficiency",
            "1": "Partial — notable gaps or minor errors",
            "2": "Complete — clinically sound and accurate",
        }},
        "source":  "IM-TRACE rubric v1.0",
        "rubric":  "R1",
        "weight":  1.0,
    },
    {
        "metric_id":   "im_trace_r2_clinical_relevance",
        "display_name": "R2 Clinical Relevance",
        "description": (
            "Applicability of the response to the specific clinical scenario. "
            "Addresses chief complaint, scenario-specific factors, actionability."
        ),
        "scale": {"type": "ordinal", "min": 0, "max": 2, "labels": {
            "0": "Inadequate — does not address the clinical scenario",
            "1": "Partial — partially relevant with notable gaps",
            "2": "Complete — fully relevant and actionable",
        }},
        "source":  "IM-TRACE rubric v1.0",
        "rubric":  "R2",
        "weight":  1.0,
    },
    {
        "metric_id":   "im_trace_r3_safety_effectiveness",
        "display_name": "R3 Safety & Effectiveness",
        "description": (
            "CSEDB-adapted safety and clinical effectiveness composite. "
            "Safety items weighted 2x. Any safety score of 0 triggers a cap "
            "on the overall aggregate score."
        ),
        "scale": {"type": "continuous", "min": 0.0, "max": 2.0},
        "source":  "IM-TRACE rubric v1.0 (CSEDB-adapted)",
        "rubric":  "R3",
        "weight":  1.5,
        "safety_weighted": True,
    },
    {
        "metric_id":   "im_trace_r4_reasoning_quality",
        "display_name": "R4 Reasoning Quality",
        "description": (
            "Clinical reasoning trace evaluation across 5 subscales: "
            "DDx construction, pretest probability calibration, evidence integration, "
            "diagnostic closure, epistemic humility. "
            "Aligned with FDA criterion 4: independent review of recommendation basis."
        ),
        "scale": {"type": "continuous", "min": 0.0, "max": 2.0},
        "source":  "IM-TRACE rubric v1.0 (FDA criterion 4)",
        "rubric":  "R4",
        "weight":  1.0,
        "subscales": [
            "ddx_construction",
            "pretest_probability_calibration",
            "evidence_integration",
            "diagnostic_closure",
            "epistemic_humility",
        ],
    },
    {
        "metric_id":   "im_trace_total",
        "display_name": "IM-TRACE Total Score",
        "description": (
            "Composite score: R1 + R2 + (R3_composite * 1.5) + R4_composite. "
            "Max 9.0. Safety cap applies when any R3 safety item scores 0."
        ),
        "scale": {"type": "continuous", "min": 0.0, "max": 9.0},
        "source":  "IM-TRACE rubric v1.0",
        "rubric":  "composite",
        "weight":  None,
    },
]


def export_custom_metrics() -> dict:
    """
    Return IM-TRACE rubric definitions in Microsoft custom evaluator format.

    Compatible with Healthcare AI Model Evaluator custom metric registration.
    """
    return {
        "schema_version":      "2.0",
        "evaluator_name":      "IM-TRACE",
        "evaluator_version":   "1.0",
        "evaluator_type":      "custom",
        "description": (
            "Internal Medicine Taxonomy for Reasoning Assessment and Clinical Evaluation (IM-TRACE). "
            "Four-rubric benchmark for evaluating clinical AI models on internal medicine cases."
        ),
        "metrics": _RUBRIC_DEFINITIONS,
        "scoring_formula": "R1 + R2 + (R3_composite * 1.5) + R4_composite",
        "max_score":   9.0,
        "score_range": [0.0, 9.0],
        "citation": "IM-TRACE v1.0 — ai-physician benchmark",
    }


# ── Annotation → Evaluation Record Mapper ────────────────────────────────────

def _map_reviewer(ann: dict) -> dict:
    """Map an annotation dict to a Microsoft reviewer object."""
    judge_type = ann.get("judge_type", "unknown")
    reviewer_id = (
        ann.get("evaluator_id")
        or ann.get("judge_id")
        or ann.get("judge_model_id")
        or "unknown"
    )

    if judge_type in ("physician", "PHYSICIAN"):
        reviewer_type = "human"
        reviewer_role = "physician_expert"
    elif judge_type in ("llm_judge", "LLM_JUDGE"):
        reviewer_type = "automated"
        reviewer_role = "llm_judge"
        reviewer_id   = ann.get("judge_model_id", reviewer_id)
    elif judge_type in ("adjudicated", "ADJUDICATED"):
        reviewer_type = "adjudicated"
        reviewer_role = "human_reviewed_llm"
    else:
        reviewer_type = "unknown"
        reviewer_role = "unknown"

    return {
        "reviewer_id":   reviewer_id,
        "reviewer_type": reviewer_type,
        "reviewer_role": reviewer_role,
    }


def _extract_scores(ann: dict) -> dict:
    """Extract rubric scores from an annotation dict into a flat metrics dict."""
    scores = {}

    # R1
    r1 = ann.get("r1") or {}
    if isinstance(r1, dict) and "score" in r1:
        scores["im_trace_r1_factuality"] = {
            "value":      r1["score"],
            "rationale":  r1.get("rationale", ""),
            "confidence": r1.get("judge_confidence", "medium"),
        }

    # R2
    r2 = ann.get("r2") or {}
    if isinstance(r2, dict) and "score" in r2:
        scores["im_trace_r2_clinical_relevance"] = {
            "value":      r2["score"],
            "rationale":  r2.get("rationale", ""),
            "confidence": r2.get("judge_confidence", "medium"),
        }

    # R3 composite
    r3 = ann.get("r3") or {}
    if isinstance(r3, dict):
        r3_composite = ann.get("r3_composite")
        if r3_composite is None:
            safety_items = r3.get("safety_items") or []
            effectiveness_items = r3.get("effectiveness_items") or []
            safety_scores = [i.get("score", 0) for i in safety_items if isinstance(i, dict)]
            eff_scores    = [i.get("score", 0) for i in effectiveness_items if isinstance(i, dict)]
            s_mean = sum(safety_scores) / len(safety_scores) if safety_scores else 0.0
            e_mean = sum(eff_scores) / len(eff_scores) if eff_scores else 0.0
            r3_composite = (s_mean * 2 + e_mean) / 3

        scores["im_trace_r3_safety_effectiveness"] = {
            "value":        round(r3_composite, 4),
            "rationale":    r3.get("rationale", ""),
            "confidence":   r3.get("judge_confidence", "medium"),
            "safety_capped": ann.get("safety_capped", False),
        }

    # R4 composite + subscales
    r4 = ann.get("r4") or {}
    if isinstance(r4, dict):
        r4_composite = ann.get("r4_composite")
        subscale_scores = {}
        subscale_names = [
            "ddx_construction", "pretest_probability_calibration",
            "evidence_integration", "diagnostic_closure", "epistemic_humility",
        ]
        raw_sub = []
        for sub in subscale_names:
            sub_data = r4.get(sub) or {}
            if isinstance(sub_data, dict) and "score" in sub_data:
                subscale_scores[sub] = sub_data["score"]
                raw_sub.append(sub_data["score"])

        if r4_composite is None and raw_sub:
            r4_composite = sum(raw_sub) / len(raw_sub)

        if r4_composite is not None:
            scores["im_trace_r4_reasoning_quality"] = {
                "value":      round(r4_composite, 4),
                "subscales":  subscale_scores,
                "confidence": "medium",
            }

    # Total
    total = ann.get("total") or ann.get("total_score")
    if total is not None:
        scores["im_trace_total"] = {
            "value": round(float(total), 4),
        }

    return scores


def _annotation_to_evaluation_record(ann: dict) -> dict:
    """Convert one annotation dict to a Microsoft evaluation record."""
    return {
        "evaluation_id":   ann.get("annotation_id", "unknown"),
        "case_id":         ann.get("case_id", "unknown"),
        "model_id":        ann.get("model_id", "unknown"),
        "response_id":     ann.get("response_id", "unknown"),
        "reviewer":        _map_reviewer(ann),
        "metrics":         _extract_scores(ann),
        "overall_notes":   ann.get("overall_notes", ""),
        "rubric_version":  ann.get("rubric_version", "1.0"),
        "created_at":      ann.get("created_at", ""),
        "schema_version":  "microsoft_healthcare_eval_v2",
        "source_system":   "IM-TRACE",
    }


# ── Export Evaluations ────────────────────────────────────────────────────────

def export_evaluations(annotations_path: Path, output_path: Path) -> Path:
    """
    Read IM-TRACE annotations JSONL and emit Microsoft-compatible evaluation records.

    Input:  annotations_path — JSONL file, one annotation dict per line
    Output: output_path — JSON file with list of evaluation records

    Returns: output_path (for chaining)
    """
    annotations_path = Path(annotations_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    errors = []

    with open(annotations_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                ann = json.loads(line)
                records.append(_annotation_to_evaluation_record(ann))
            except json.JSONDecodeError as e:
                errors.append({"line": line_num, "error": str(e)})

    output = {
        "schema_version":    "microsoft_healthcare_eval_v2",
        "source_system":     "IM-TRACE",
        "evaluator_name":    "IM-TRACE",
        "n_records":         len(records),
        "n_parse_errors":    len(errors),
        "parse_errors":      errors,
        "custom_metrics":    export_custom_metrics(),
        "evaluations":       records,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    return output_path


# ── Export Arena Comparisons ──────────────────────────────────────────────────

def _pairwise_to_arena_record(cmp: dict) -> dict:
    """
    Convert one IM-TRACE PairwiseComparisonRecord dict to a Microsoft arena
    comparison record.

    Microsoft arena format expects:
      - comparison_id
      - case_id
      - model_a / model_b
      - winner ("a" | "b" | "tie")
      - per-rubric preferences
      - judge metadata
    """
    # Extract overall winner from preferences list
    overall_winner = "tie"
    rubric_preferences = []

    preferences = cmp.get("preferences") or []
    for pref in preferences:
        if isinstance(pref, dict):
            rubric = pref.get("rubric", "unknown")
            winner = pref.get("winner", "tie")
            rubric_preferences.append({
                "rubric":          rubric,
                "subscale":        pref.get("subscale"),
                "winner":          winner,
                "rationale":       pref.get("rationale", ""),
                "judge_confidence": pref.get("judge_confidence", "medium"),
            })
            if rubric == "overall":
                overall_winner = winner

    judge_type = cmp.get("judge_type", "unknown")
    judge_id   = cmp.get("judge_id", "unknown")

    return {
        "schema_version":    "microsoft_arena_v1",
        "comparison_id":     cmp.get("comparison_id", "unknown"),
        "case_id":           cmp.get("case_id", "unknown"),
        "model_a":           cmp.get("model_a_id") or cmp.get("model_a", "unknown"),
        "model_b":           cmp.get("model_b_id") or cmp.get("model_b", "unknown"),
        "response_a_id":     cmp.get("response_a_id", "unknown"),
        "response_b_id":     cmp.get("response_b_id", "unknown"),
        "presented_order":   cmp.get("presented_order", "unknown"),
        "overall_winner":    overall_winner,
        "rubric_preferences": rubric_preferences,
        "judge": {
            "judge_type": judge_type,
            "judge_id":   judge_id,
        },
        "rubric_version": cmp.get("rubric_version", "1.0"),
        "created_at":     cmp.get("created_at", ""),
        "source_system":  "IM-TRACE",
    }


def export_arena_comparisons(pairwise_path: Path, output_path: Path) -> Path:
    """
    Read IM-TRACE pairwise JSONL and emit arena-compatible comparison records.

    Input:  pairwise_path — JSONL file, one PairwiseComparisonRecord dict per line
    Output: output_path — JSON file with list of arena comparison records

    Returns: output_path (for chaining)
    """
    pairwise_path = Path(pairwise_path)
    output_path   = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arena_records = []
    errors = []

    with open(pairwise_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                cmp = json.loads(line)
                arena_records.append(_pairwise_to_arena_record(cmp))
            except json.JSONDecodeError as e:
                errors.append({"line": line_num, "error": str(e)})

    # Aggregate win stats per model pair
    win_stats: dict[tuple[str, str], dict[str, int]] = {}
    for rec in arena_records:
        a = rec["model_a"]
        b = rec["model_b"]
        key = (min(a, b), max(a, b))
        if key not in win_stats:
            win_stats[key] = {"model_a_wins": 0, "model_b_wins": 0, "ties": 0, "n_comparisons": 0}
        stats = win_stats[key]
        stats["n_comparisons"] += 1
        winner = rec["overall_winner"]
        if winner == "a":
            # "a" in the original record is model_a
            if a == key[0]:
                stats["model_a_wins"] += 1
            else:
                stats["model_b_wins"] += 1
        elif winner == "b":
            if b == key[1]:
                stats["model_b_wins"] += 1
            else:
                stats["model_a_wins"] += 1
        else:
            stats["ties"] += 1

    matchup_summary = [
        {
            "model_a":         pair[0],
            "model_b":         pair[1],
            "n_comparisons":   stats["n_comparisons"],
            "model_a_wins":    stats["model_a_wins"],
            "model_b_wins":    stats["model_b_wins"],
            "ties":            stats["ties"],
            "model_a_win_rate": round(
                stats["model_a_wins"] / max(stats["n_comparisons"], 1), 4
            ),
        }
        for pair, stats in sorted(win_stats.items())
    ]

    output = {
        "schema_version":   "microsoft_arena_v1",
        "source_system":    "IM-TRACE",
        "n_comparisons":    len(arena_records),
        "n_parse_errors":   len(errors),
        "parse_errors":     errors,
        "matchup_summary":  matchup_summary,
        "comparisons":      arena_records,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    return output_path
