"""
Active Pairwise Scheduling — select the next most informative comparisons.

Given existing pairwise records, current ranking estimates, and a budget,
produces a prioritized list of candidate comparisons ordered by expected
information gain.

Current implementation: heuristic acquisition function.
Future: TODO hooks for Bayesian/mutual-information-based criteria.

Design principle: not all comparisons are equally informative. Spend
comparisons near the decision boundary, not where ordering is obvious.
This is the benchmark equivalent of adaptive trial design.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from itertools import combinations
from typing import Optional

from im_trace.analysis.ranking.interface import RankingResult


def compute_acquisition_scores(
    ranking_result: RankingResult,
    existing_comparisons: list[dict],
    case_ids: list[str],
    safety_critical_cases: Optional[set[str]] = None,
    hard_case_ids: Optional[set[str]] = None,
    budget: int = 10,
    annotation_cost: float = 1.0,
    weights: Optional[dict[str, float]] = None,
) -> list[dict]:
    """
    Score all candidate pairwise comparisons by expected information value.

    Acquisition score = weighted sum of:
      - uncertainty_score: how uncertain is the relative ordering?
      - leverage_score: how much would this comparison affect rankings?
      - safety_score: is this a safety-critical case?
      - hard_case_score: is this case already flagged as hard?

    Args:
        ranking_result: current ranking estimates from any backend
        existing_comparisons: list of {"winner", "loser", "case_id"} already done
        case_ids: all available case IDs
        safety_critical_cases: set of case_ids flagged as safety-critical
        hard_case_ids: set of case_ids flagged as hard cases
        budget: maximum number of comparisons to recommend
        annotation_cost: per-comparison cost (for cost-aware selection)
        weights: override default component weights

    Returns:
        List of candidate comparisons sorted by acquisition score (descending),
        truncated to budget. Each entry:
          {
            "model_a": str,
            "model_b": str,
            "case_id": str,
            "acquisition_score": float,
            "components": {uncertainty, leverage, safety, hard_case},
            "rationale": str,
          }
    """
    w = weights or {
        "uncertainty": 0.4,
        "leverage": 0.3,
        "safety": 0.2,
        "hard_case": 0.1,
    }

    safety_set = safety_critical_cases or set()
    hard_set = hard_case_ids or set()
    models = sorted(ranking_result.latent_scores.keys())

    if len(models) < 2:
        return []

    # Count existing comparisons per (model_a, model_b, case_id)
    existing_coverage: dict[tuple[str, str, str], int] = defaultdict(int)
    for c in existing_comparisons:
        key = tuple(sorted([c.get("winner", ""), c.get("loser", "")])) + (c.get("case_id", ""),)
        existing_coverage[key] += 1

    # Compute per-pair uncertainty from superiority matrix
    pair_uncertainty: dict[tuple[str, str], float] = {}
    for a in models:
        for b in models:
            if a >= b:
                continue
            p_ab = ranking_result.pairwise_superiority.get(a, {}).get(b, 0.5)
            # Uncertainty is highest when P(A>B) ≈ 0.5 (maximum entropy)
            # Shannon entropy of Bernoulli(p): -p*log(p) - (1-p)*log(1-p)
            p = max(min(p_ab, 0.999), 0.001)
            entropy = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
            pair_uncertainty[(a, b)] = entropy  # 0 to 1

    # Compute per-pair leverage: how close are their scores?
    score_range = max(ranking_result.latent_scores.values()) - min(ranking_result.latent_scores.values())
    if score_range < 1e-10:
        score_range = 1.0

    pair_leverage: dict[tuple[str, str], float] = {}
    for a in models:
        for b in models:
            if a >= b:
                continue
            score_diff = abs(ranking_result.latent_scores[a] - ranking_result.latent_scores[b])
            # Leverage is high when scores are close (small diff = high leverage)
            leverage = 1.0 - min(score_diff / score_range, 1.0)
            pair_leverage[(a, b)] = leverage

    # Generate all candidates
    candidates = []
    for case_id in case_ids:
        for a, b in combinations(models, 2):
            pair = (a, b)
            key = (a, b, case_id)

            # Skip if already done (diminishing returns after 2nd comparison)
            n_existing = existing_coverage.get(key, 0)
            if n_existing >= 2:
                continue

            # Novelty discount for already-compared pairs
            novelty = 1.0 / (1 + n_existing)

            # Component scores
            uncertainty = pair_uncertainty.get(pair, 0.5)
            leverage = pair_leverage.get(pair, 0.5)
            safety = 1.0 if case_id in safety_set else 0.0
            hard_case = 1.0 if case_id in hard_set else 0.0

            acquisition = (
                w["uncertainty"] * uncertainty +
                w["leverage"] * leverage +
                w["safety"] * safety +
                w["hard_case"] * hard_case
            ) * novelty / max(annotation_cost, 0.01)

            rationale_parts = []
            if uncertainty > 0.8:
                rationale_parts.append("high uncertainty")
            if leverage > 0.7:
                rationale_parts.append("close scores")
            if safety:
                rationale_parts.append("safety-critical")
            if hard_case:
                rationale_parts.append("hard case")
            if n_existing == 0:
                rationale_parts.append("never compared on this case")

            candidates.append({
                "model_a": a,
                "model_b": b,
                "case_id": case_id,
                "acquisition_score": round(acquisition, 4),
                "components": {
                    "uncertainty": round(uncertainty, 3),
                    "leverage": round(leverage, 3),
                    "safety": safety,
                    "hard_case": hard_case,
                    "novelty": round(novelty, 3),
                },
                "rationale": "; ".join(rationale_parts) if rationale_parts else "routine",
                "existing_comparisons": n_existing,
            })

    # Sort by acquisition score descending, take top budget
    candidates.sort(key=lambda x: x["acquisition_score"], reverse=True)
    return candidates[:budget]


def suggest_next_comparisons(
    ranking_result: RankingResult,
    existing_comparisons: list[dict],
    case_ids: list[str],
    safety_critical_cases: Optional[set[str]] = None,
    hard_case_ids: Optional[set[str]] = None,
    budget: int = 10,
) -> dict:
    """
    High-level API: suggest the next batch of pairwise comparisons.

    Returns:
      {
        "candidates": [...],  # prioritized list
        "total_candidates_scored": int,
        "budget": int,
        "selection_method": "heuristic_acquisition",
        "note": "...",
      }
    """
    candidates = compute_acquisition_scores(
        ranking_result=ranking_result,
        existing_comparisons=existing_comparisons,
        case_ids=case_ids,
        safety_critical_cases=safety_critical_cases,
        hard_case_ids=hard_case_ids,
        budget=budget,
    )

    return {
        "candidates": candidates,
        "total_candidates_scored": len(candidates),
        "budget": budget,
        "selection_method": "heuristic_acquisition",
        "weights": {"uncertainty": 0.4, "leverage": 0.3, "safety": 0.2, "hard_case": 0.1},
        "note": (
            "Heuristic acquisition function. Prioritizes high uncertainty, "
            "close scores, safety-critical cases, and hard cases. "
            "TODO: upgrade to mutual-information or posterior-risk-based criteria "
            "when Bayesian posterior is available."
        ),
    }
