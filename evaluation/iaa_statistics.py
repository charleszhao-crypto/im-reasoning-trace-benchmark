#!/usr/bin/env python3
"""
IM-TRACE Inter-Annotator Agreement (IAA) Statistics.

Computes Krippendorff's alpha, Cohen's kappa, and percentage agreement
across multiple physician annotators scoring the same cases.

Usage:
  python evaluation/iaa_statistics.py --annotations annotations/
  python evaluation/iaa_statistics.py --annotations annotations/ --rubric r4
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional


def load_annotations(annotations_dir: Path) -> list[dict]:
    """Load all annotation JSON files from a directory."""
    results = []
    for f in sorted(annotations_dir.glob("*.json")):
        with open(f) as fh:
            results.append(json.load(fh))
    return results


def group_by_case(annotations: list[dict]) -> dict[str, list[dict]]:
    """Group annotations by case_id for IAA computation."""
    grouped = defaultdict(list)
    for ann in annotations:
        grouped[ann.get("case_id", "unknown")].append(ann)
    return dict(grouped)


def percentage_agreement(scores_a: list[int], scores_b: list[int]) -> float:
    """Simple percentage agreement between two annotators."""
    if len(scores_a) != len(scores_b) or not scores_a:
        return 0.0
    agree = sum(1 for a, b in zip(scores_a, scores_b) if a == b)
    return agree / len(scores_a) * 100


def cohens_kappa(scores_a: list[int], scores_b: list[int], k: int = 3) -> float:
    """
    Cohen's kappa for ordinal data.
    k = number of categories (default 3 for 0/1/2 scale).
    """
    n = len(scores_a)
    if n == 0:
        return 0.0

    # Observed agreement
    po = sum(1 for a, b in zip(scores_a, scores_b) if a == b) / n

    # Expected agreement by chance
    freq_a = [scores_a.count(i) / n for i in range(k)]
    freq_b = [scores_b.count(i) / n for i in range(k)]
    pe = sum(fa * fb for fa, fb in zip(freq_a, freq_b))

    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0

    return (po - pe) / (1 - pe)


def krippendorff_alpha_ordinal(reliability_data: list[list[Optional[int]]]) -> float:
    """
    Krippendorff's alpha for ordinal data.

    reliability_data: list of annotator vectors. Each vector has one value
    per case (or None if that annotator didn't score that case).

    Uses the ordinal distance metric: d(c,k) = sum of frequencies between c and k.
    """
    n_annotators = len(reliability_data)
    if n_annotators < 2:
        return 0.0

    n_cases = len(reliability_data[0])

    # Collect all non-None values per case
    case_values = []
    for j in range(n_cases):
        vals = [reliability_data[i][j] for i in range(n_annotators)
                if reliability_data[i][j] is not None]
        if len(vals) >= 2:
            case_values.append(vals)

    if not case_values:
        return 0.0

    # All unique values
    all_vals = sorted(set(v for vals in case_values for v in vals))
    val_to_idx = {v: i for i, v in enumerate(all_vals)}

    # Overall value frequency
    total_count = sum(len(vals) for vals in case_values)
    freq = defaultdict(int)
    for vals in case_values:
        for v in vals:
            freq[v] += 1

    # Ordinal distance metric
    def ordinal_dist_sq(c: int, k: int) -> float:
        if c == k:
            return 0.0
        lo, hi = min(c, k), max(c, k)
        # Sum of frequencies from lo to hi (inclusive)
        n_between = sum(freq.get(v, 0) for v in all_vals if lo <= v <= hi)
        # Subtract half of endpoints
        n_between -= (freq.get(lo, 0) + freq.get(hi, 0)) / 2
        return n_between ** 2

    # Observed disagreement
    do = 0.0
    n_pairs_total = 0
    for vals in case_values:
        m = len(vals)
        for i in range(m):
            for j in range(i + 1, m):
                do += ordinal_dist_sq(vals[i], vals[j])
                n_pairs_total += 1

    if n_pairs_total == 0:
        return 0.0
    do /= n_pairs_total

    # Expected disagreement
    de = 0.0
    n_total_pairs = total_count * (total_count - 1) / 2
    for c in all_vals:
        for k in all_vals:
            if c < k:
                de += freq[c] * freq[k] * ordinal_dist_sq(c, k)
    if n_total_pairs > 0:
        de /= n_total_pairs

    if de == 0:
        return 1.0 if do == 0 else 0.0

    return 1.0 - (do / de)


def extract_rubric_scores(annotations: list[dict], rubric: str = "total") -> dict[str, list[float]]:
    """
    Extract scores for a specific rubric from annotations, grouped by evaluator.

    rubric: "r1", "r2", "r3", "r4", "total"
    Returns: {evaluator_id: [score_for_case_0, score_for_case_1, ...]}
    """
    score_key = {
        "r1": "r1_factuality",
        "r2": "r2_relevance",
        "r3": "r3_safety_effectiveness",
        "r4": "r4_reasoning_trace",
        "total": "total",
    }.get(rubric, "total")

    by_evaluator = defaultdict(dict)
    for ann in annotations:
        evaluator = ann.get("evaluator_id", "unknown")
        case_id = ann.get("case_id", "unknown")
        score = ann.get(score_key, 0)
        by_evaluator[evaluator][case_id] = score

    return dict(by_evaluator)


def compute_iaa_report(annotations: list[dict], rubric: str = "total") -> dict:
    """Compute full IAA report for a set of annotations."""
    by_evaluator = extract_rubric_scores(annotations, rubric)
    evaluators = sorted(by_evaluator.keys())

    if len(evaluators) < 2:
        return {"error": "Need at least 2 evaluators for IAA", "n_evaluators": len(evaluators)}

    # Get all case IDs in order
    all_cases = sorted(set(
        case_id for ev_scores in by_evaluator.values() for case_id in ev_scores
    ))

    # Build reliability matrix
    reliability_data = []
    for ev in evaluators:
        row = [by_evaluator[ev].get(case_id) for case_id in all_cases]
        # Round to nearest int for ordinal metrics
        row = [round(v) if v is not None else None for row_v in [row] for v in row_v]
        reliability_data.append(row)

    # Krippendorff's alpha
    alpha = krippendorff_alpha_ordinal(reliability_data)

    # Pairwise Cohen's kappa and agreement
    pairwise = []
    for i in range(len(evaluators)):
        for j in range(i + 1, len(evaluators)):
            # Get common cases
            common_cases = [
                c for c in all_cases
                if reliability_data[i][all_cases.index(c)] is not None
                and reliability_data[j][all_cases.index(c)] is not None
            ]
            if not common_cases:
                continue

            scores_i = [reliability_data[i][all_cases.index(c)] for c in common_cases]
            scores_j = [reliability_data[j][all_cases.index(c)] for c in common_cases]

            kappa = cohens_kappa(scores_i, scores_j)
            pct = percentage_agreement(scores_i, scores_j)

            pairwise.append({
                "evaluator_a": evaluators[i],
                "evaluator_b": evaluators[j],
                "n_common_cases": len(common_cases),
                "cohens_kappa": round(kappa, 3),
                "pct_agreement": round(pct, 1),
            })

    return {
        "rubric": rubric,
        "n_evaluators": len(evaluators),
        "n_cases": len(all_cases),
        "evaluators": evaluators,
        "krippendorff_alpha": round(alpha, 3),
        "alpha_interpretation": interpret_alpha(alpha),
        "pairwise": pairwise,
    }


def interpret_alpha(alpha: float) -> str:
    """Interpret Krippendorff's alpha."""
    if alpha >= 0.8:
        return "Good reliability — suitable for drawing conclusions"
    elif alpha >= 0.667:
        return "Acceptable — suitable for tentative conclusions"
    elif alpha >= 0.4:
        return "Low — insufficient for reliable conclusions"
    else:
        return "Poor — near-chance agreement"


def main():
    parser = argparse.ArgumentParser(description="IM-TRACE IAA Statistics")
    parser.add_argument("--annotations", type=Path, required=True,
                        help="Directory of annotation JSON files")
    parser.add_argument("--rubric", default="total",
                        choices=["r1", "r2", "r3", "r4", "total"],
                        help="Which rubric to compute IAA for")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    annotations = load_annotations(args.annotations)
    if not annotations:
        print(f"No annotation files found in {args.annotations}")
        sys.exit(1)

    report = compute_iaa_report(annotations, args.rubric)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"\n=== IAA Report: {report['rubric'].upper()} ===")
        print(f"Evaluators: {report['n_evaluators']} | Cases: {report['n_cases']}")
        print(f"Krippendorff's alpha: {report['krippendorff_alpha']}")
        print(f"Interpretation: {report['alpha_interpretation']}")
        if report.get('pairwise'):
            print("\nPairwise:")
            for p in report['pairwise']:
                print(f"  {p['evaluator_a']} vs {p['evaluator_b']}: "
                      f"kappa={p['cohens_kappa']}, agree={p['pct_agreement']}%, "
                      f"n={p['n_common_cases']}")


if __name__ == "__main__":
    main()
