"""
IM-TRACE Hard-Case Detection — 5-Criterion System.

A case enters the hard set not because scores are LOW, but because it reveals
INSTABILITY, DISAGREEMENT, or SAFETY-CRITICAL weaknesses.

Five criteria (a case is "hard" if ANY fires):

  1. human_vs_llm_disagreement
     Total score difference > threshold between human and LLM annotations.

  2. low_judge_confidence
     Any annotation has judge_confidence == "low" on 2+ rubrics.

  3. unstable_pairwise_ordering
     The case changes which model wins when compared to different opponents.

  4. high_ranking_leverage
     Removing this case from the set changes the top-ranked model (LOO analysis).

  5. safety_critical_flag
     Different annotators disagree on whether a safety item is 0 vs. >0.

Each case record carries:
  - case_id
  - criteria_fired: list of criterion names
  - details: per-criterion detail dict
  - severity_score: count of criteria fired (0-5)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from im_trace.cases.schema.models import JudgeType, ConfidenceLevel


# ── Criterion 1: Human vs. LLM Disagreement ──────────────────────────────────

def _check_human_vs_llm_disagreement(
    annotations: list[dict],
    threshold: float,
) -> tuple[bool, dict]:
    """
    Fire if the absolute difference between mean human and mean LLM total scores
    exceeds `threshold`.
    """
    human = [
        a for a in annotations
        if a.get("judge_type") in (JudgeType.PHYSICIAN, JudgeType.PHYSICIAN.value, "physician")
    ]
    llm = [
        a for a in annotations
        if a.get("judge_type") in (JudgeType.LLM_JUDGE, JudgeType.LLM_JUDGE.value, "llm_judge")
    ]

    if not human or not llm:
        return False, {"reason": "insufficient_annotations", "human_n": len(human), "llm_n": len(llm)}

    human_scores = [a.get("total", a.get("total_score", 0.0)) for a in human]
    llm_scores   = [a.get("total", a.get("total_score", 0.0)) for a in llm]

    human_mean = sum(human_scores) / len(human_scores)
    llm_mean   = sum(llm_scores)   / len(llm_scores)
    delta = abs(human_mean - llm_mean)

    fired = delta > threshold
    return fired, {
        "human_mean":  round(human_mean, 3),
        "llm_mean":    round(llm_mean, 3),
        "delta":       round(delta, 3),
        "threshold":   threshold,
        "human_n":     len(human),
        "llm_n":       len(llm),
    }


# ── Criterion 2: Low Judge Confidence ────────────────────────────────────────

def _check_low_judge_confidence(annotations: list[dict]) -> tuple[bool, dict]:
    """
    Fire if any single annotation has judge_confidence == "low" on 2+ rubrics.

    Rubrics checked: r1, r2, r3, and all r4 subscales.
    """
    flagged_annotations = []

    for ann in annotations:
        low_rubrics = []

        # R1
        r1 = ann.get("r1") or {}
        if isinstance(r1, dict):
            if r1.get("judge_confidence") in (ConfidenceLevel.LOW, ConfidenceLevel.LOW.value, "low"):
                low_rubrics.append("r1")

        # R2
        r2 = ann.get("r2") or {}
        if isinstance(r2, dict):
            if r2.get("judge_confidence") in (ConfidenceLevel.LOW, ConfidenceLevel.LOW.value, "low"):
                low_rubrics.append("r2")

        # R3
        r3 = ann.get("r3") or {}
        if isinstance(r3, dict):
            if r3.get("judge_confidence") in (ConfidenceLevel.LOW, ConfidenceLevel.LOW.value, "low"):
                low_rubrics.append("r3")

        # R4 subscales
        r4 = ann.get("r4") or {}
        if isinstance(r4, dict):
            for subscale in ("ddx_construction", "pretest_probability_calibration",
                             "evidence_integration", "diagnostic_closure", "epistemic_humility"):
                subscale_data = r4.get(subscale) or {}
                if isinstance(subscale_data, dict):
                    if subscale_data.get("judge_confidence") in (
                        ConfidenceLevel.LOW, ConfidenceLevel.LOW.value, "low"
                    ):
                        low_rubrics.append(f"r4.{subscale}")

        if len(low_rubrics) >= 2:
            flagged_annotations.append({
                "annotation_id": ann.get("annotation_id", "unknown"),
                "evaluator_id":  ann.get("evaluator_id") or ann.get("judge_id", "unknown"),
                "low_rubrics":   low_rubrics,
                "low_count":     len(low_rubrics),
            })

    fired = len(flagged_annotations) > 0
    return fired, {
        "flagged_annotations": flagged_annotations,
        "total_flagged":       len(flagged_annotations),
    }


# ── Criterion 3: Unstable Pairwise Ordering ───────────────────────────────────

def _check_unstable_pairwise_ordering(
    case_id: str,
    pairwise_records: list[dict],
) -> tuple[bool, dict]:
    """
    Fire if a model's win/loss outcome on this case changes when compared to
    different opponents — i.e., the case produces non-transitive or opponent-
    dependent preference orderings.

    A model is "unstable" on this case if it beats opponent A but loses to opponent B,
    AND it also beats opponent B in a different comparison involving a third model.

    Simplified check: for each (model, case_id) pair, collect all opponents it
    beats and loses to. If the set of beaten opponents ∩ set of lost opponents
    is non-empty (the same opponent appears as both win and loss for the same focal
    model), that is definitionally unstable. Otherwise, test for ordering
    inconsistency via cycle detection across all case-level comparisons.
    """
    relevant = [p for p in (pairwise_records or []) if p.get("case_id") == case_id]

    if len(relevant) < 2:
        return False, {"reason": "fewer than 2 pairwise records for this case", "n_relevant": len(relevant)}

    # Build per-model win/loss sets
    wins  = defaultdict(set)  # wins[m] = set of models m beat
    losses= defaultdict(set)  # losses[m] = set of models m lost to

    for rec in relevant:
        a = rec.get("model_a_id") or rec.get("model_a")
        b = rec.get("model_b_id") or rec.get("model_b")
        winner = rec.get("overall_winner") or rec.get("winner")

        if not a or not b or not winner:
            continue

        if winner == "a":
            wins[a].add(b)
            losses[b].add(a)
        elif winner == "b":
            wins[b].add(a)
            losses[a].add(b)
        # ties ignored for ordering instability

    # Detect cycles: A beats B, B beats C, C beats A — implies this case
    # generates intransitive preference among the models tested on it.
    all_models = sorted({m for rec in relevant
                         for m in [rec.get("model_a_id") or rec.get("model_a"),
                                   rec.get("model_b_id") or rec.get("model_b")] if m})

    def has_cycle(adj: dict) -> list:
        """Return a cycle as list of nodes, or empty list."""
        visited = {}  # node -> color: 0=unvisited, 1=in-stack, 2=done
        stack = []
        for start in adj:
            if visited.get(start, 0) != 0:
                continue
            path = []
            stack_nodes = [(start, iter(adj.get(start, set())))]
            visited[start] = 1
            path.append(start)
            while stack_nodes:
                node, children = stack_nodes[-1]
                try:
                    child = next(children)
                    if visited.get(child, 0) == 0:
                        visited[child] = 1
                        path.append(child)
                        stack_nodes.append((child, iter(adj.get(child, set()))))
                    elif visited.get(child, 0) == 1:
                        # Found cycle
                        idx = path.index(child)
                        return path[idx:]
                except StopIteration:
                    visited[node] = 2
                    path.pop()
                    stack_nodes.pop()
        return []

    cycle = has_cycle(dict(wins))
    fired = len(cycle) > 0

    return fired, {
        "n_relevant_comparisons": len(relevant),
        "models_in_comparisons":  all_models,
        "cycle_detected":         cycle,
        "wins_summary":           {m: sorted(v) for m, v in wins.items()},
    }


# ── Criterion 4: High Ranking Leverage ────────────────────────────────────────

def _check_high_ranking_leverage(
    case_id: str,
    loo_results: list[dict],
) -> tuple[bool, dict]:
    """
    Fire if removing this case from the evaluation set changes the top-ranked model.

    loo_results: pre-computed leave-one-out results. Each entry:
      {
        "left_out_case_id": str,
        "top_model_without": str,   # top-ranked model when this case is excluded
        "top_model_with":    str,   # top-ranked model in full set
      }

    If not provided, this criterion cannot fire (returns False with explanation).
    """
    if not loo_results:
        return False, {"reason": "loo_results not provided — criterion not evaluated"}

    for entry in loo_results:
        if entry.get("left_out_case_id") == case_id:
            top_with    = entry.get("top_model_with")
            top_without = entry.get("top_model_without")
            fired = top_with != top_without and top_without is not None
            return fired, {
                "top_model_with_case":    top_with,
                "top_model_without_case": top_without,
                "ranking_changed":        fired,
            }

    return False, {"reason": f"case_id '{case_id}' not found in loo_results"}


# ── Criterion 5: Safety Critical Flag ────────────────────────────────────────

def _check_safety_critical_flag(annotations: list[dict]) -> tuple[bool, dict]:
    """
    Fire if different annotators disagree on whether any safety item scores 0 vs. >0.

    Checks r3.safety_items for each annotation. If any item has score=0 from
    one annotator and score>0 from another annotator (for the same item criterion),
    that is a safety-critical disagreement.
    """
    # Collect per-criterion safety scores across annotations
    criterion_scores: dict[str, list[int]] = defaultdict(list)

    for ann in annotations:
        r3 = ann.get("r3") or {}
        safety_items = r3.get("safety_items") if isinstance(r3, dict) else []
        if not safety_items:
            continue
        for item in (safety_items or []):
            if isinstance(item, dict):
                criterion = item.get("criterion", "unknown_criterion")
                score = item.get("score")
                if score is not None:
                    try:
                        criterion_scores[criterion].append(int(score))
                    except (TypeError, ValueError):
                        pass

    flagged_criteria = []
    for criterion, scores in criterion_scores.items():
        if len(scores) < 2:
            continue
        has_zero    = any(s == 0 for s in scores)
        has_nonzero = any(s > 0  for s in scores)
        if has_zero and has_nonzero:
            flagged_criteria.append({
                "criterion": criterion,
                "scores":    scores,
                "min_score": min(scores),
                "max_score": max(scores),
            })

    fired = len(flagged_criteria) > 0
    return fired, {
        "flagged_criteria": flagged_criteria,
        "n_flagged":        len(flagged_criteria),
        "n_criteria_checked": len(criterion_scores),
    }


# ── Main Detection Function ───────────────────────────────────────────────────

def detect_hard_cases(
    case_annotations: dict[str, list[dict]],
    pairwise_records: list[dict] | None = None,
    loo_results: list[dict] | None = None,
    threshold: float = 1.5,
) -> list[dict]:
    """
    Run all 5 criteria against each case.

    Args:
        case_annotations:  {case_id: list of annotation dicts}
        pairwise_records:  list of pairwise comparison dicts (for criterion 3)
        loo_results:       pre-computed LOO ranking results (for criterion 4)
        threshold:         score-delta threshold for human_vs_llm_disagreement

    Returns:
        List of hard case records (only cases where at least one criterion fires):
          {
            case_id:         str,
            criteria_fired:  list[str],
            severity_score:  int (0-5),
            details: {
              human_vs_llm_disagreement:    dict or None,
              low_judge_confidence:         dict or None,
              unstable_pairwise_ordering:   dict or None,
              high_ranking_leverage:        dict or None,
              safety_critical_flag:         dict or None,
            }
          }
    """
    hard_cases = []

    for case_id, annotations in case_annotations.items():
        criteria_fired = []
        details = {}

        # Criterion 1
        fired, det = _check_human_vs_llm_disagreement(annotations, threshold)
        details["human_vs_llm_disagreement"] = det
        if fired:
            criteria_fired.append("human_vs_llm_disagreement")

        # Criterion 2
        fired, det = _check_low_judge_confidence(annotations)
        details["low_judge_confidence"] = det
        if fired:
            criteria_fired.append("low_judge_confidence")

        # Criterion 3
        fired, det = _check_unstable_pairwise_ordering(case_id, pairwise_records or [])
        details["unstable_pairwise_ordering"] = det
        if fired:
            criteria_fired.append("unstable_pairwise_ordering")

        # Criterion 4
        fired, det = _check_high_ranking_leverage(case_id, loo_results or [])
        details["high_ranking_leverage"] = det
        if fired:
            criteria_fired.append("high_ranking_leverage")

        # Criterion 5
        fired, det = _check_safety_critical_flag(annotations)
        details["safety_critical_flag"] = det
        if fired:
            criteria_fired.append("safety_critical_flag")

        if criteria_fired:
            hard_cases.append({
                "case_id":        case_id,
                "criteria_fired": criteria_fired,
                "severity_score": len(criteria_fired),
                "details":        details,
            })

    # Sort by severity descending, then case_id for determinism
    hard_cases.sort(key=lambda x: (-x["severity_score"], x["case_id"]))
    return hard_cases


# ── Export ────────────────────────────────────────────────────────────────────

def export_hard_cases(hard_cases: list[dict], output_dir: Path) -> None:
    """
    Write hard_cases.jsonl and hard_case_summary.md to output_dir.

    hard_cases.jsonl: one JSON object per line, each a full hard case record.
    hard_case_summary.md: human-readable markdown table + per-case details.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSONL
    jsonl_path = output_dir / "hard_cases.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for case in hard_cases:
            f.write(json.dumps(case, default=str) + "\n")

    # Markdown summary
    md_path = output_dir / "hard_case_summary.md"
    lines = [
        "# IM-TRACE Hard Case Summary",
        "",
        f"**Total hard cases identified:** {len(hard_cases)}",
        "",
        "## Criteria",
        "",
        "| # | Criterion | Description |",
        "|---|-----------|-------------|",
        "| 1 | `human_vs_llm_disagreement` | Human vs. LLM mean total score delta > threshold |",
        "| 2 | `low_judge_confidence` | Annotation has low confidence on 2+ rubrics |",
        "| 3 | `unstable_pairwise_ordering` | Case produces intransitive win ordering across models |",
        "| 4 | `high_ranking_leverage` | Removing case changes the top-ranked model (LOO) |",
        "| 5 | `safety_critical_flag` | Annotators disagree on safety item 0 vs. >0 |",
        "",
        "## Hard Case Table",
        "",
        "| Case ID | Severity | Criteria Fired |",
        "|---------|----------|----------------|",
    ]

    for case in hard_cases:
        criteria_str = ", ".join(f"`{c}`" for c in case["criteria_fired"])
        lines.append(f"| {case['case_id']} | {case['severity_score']}/5 | {criteria_str} |")

    lines += ["", "## Per-Case Details", ""]

    for case in hard_cases:
        lines.append(f"### {case['case_id']}  (severity {case['severity_score']}/5)")
        lines.append("")
        lines.append(f"**Criteria fired:** {', '.join(case['criteria_fired'])}")
        lines.append("")
        for criterion, detail in case["details"].items():
            if detail and criterion in case["criteria_fired"]:
                lines.append(f"**{criterion}:**")
                lines.append("```json")
                lines.append(json.dumps(detail, indent=2, default=str))
                lines.append("```")
                lines.append("")
        lines.append("---")
        lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
