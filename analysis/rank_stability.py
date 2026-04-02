"""
Rank Stability Diagnostics — bootstrap rank distributions, superiority matrix,
leave-one-out sensitivity, and fragility analysis.

Outputs rank_stability.json with per-model rank histograms, pairwise superiority
probabilities, and fragility indicators. Consumed by leaderboard renderer or
analysis notebooks.

Key principle: rank is not the same as confidence in rank.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from im_trace.evaluators.aggregation.aggregate import (
    bootstrap_ci, fit_bradley_terry, bt_to_elo, rank_models,
)


def bootstrap_rank_distribution(
    all_scores: dict[str, list[float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, dict[int, float]]:
    """
    Compute the distribution of ranks for each model via bootstrap resampling.

    For each bootstrap sample: resample cases with replacement, recompute means,
    rank models, record each model's rank.

    Returns: {model_id: {rank: frequency}} where frequency sums to 1.0.
    """
    rng = random.Random(seed)
    models = sorted(all_scores.keys())
    n_cases = max(len(v) for v in all_scores.values()) if all_scores else 0

    if n_cases == 0 or not models:
        return {}

    rank_counts: dict[str, Counter] = {m: Counter() for m in models}

    # Build case-indexed arrays for synchronized resampling
    case_indices = list(range(n_cases))

    for _ in range(n_bootstrap):
        sampled_indices = [rng.choice(case_indices) for _ in range(n_cases)]

        means = {}
        for model in models:
            scores = all_scores[model]
            sampled = [scores[i] for i in sampled_indices if i < len(scores)]
            means[model] = sum(sampled) / len(sampled) if sampled else 0

        # Rank by mean (descending)
        ranked = sorted(means.items(), key=lambda x: x[1], reverse=True)
        for rank_idx, (model, _) in enumerate(ranked):
            rank_counts[model][rank_idx + 1] += 1

    # Normalize to frequencies
    return {
        model: {rank: count / n_bootstrap for rank, count in sorted(counts.items())}
        for model, counts in rank_counts.items()
    }


def top_k_frequency(
    rank_dist: dict[str, dict[int, float]],
    k: int = 1,
) -> dict[str, float]:
    """Probability of each model being in top-k across bootstrap samples."""
    return {
        model: sum(freq for rank, freq in dist.items() if rank <= k)
        for model, dist in rank_dist.items()
    }


def pairwise_superiority_matrix(
    all_scores: dict[str, list[float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """
    Compute P(model_a > model_b) for all model pairs via bootstrap.

    Returns: {model_a: {model_b: probability}} where probability is
    the fraction of bootstrap samples where model_a's mean exceeds model_b's.
    """
    rng = random.Random(seed)
    models = sorted(all_scores.keys())
    n_cases = max(len(v) for v in all_scores.values()) if all_scores else 0

    if n_cases == 0 or len(models) < 2:
        return {}

    win_counts: dict[str, dict[str, int]] = {
        m: {o: 0 for o in models if o != m} for m in models
    }
    case_indices = list(range(n_cases))

    for _ in range(n_bootstrap):
        sampled_indices = [rng.choice(case_indices) for _ in range(n_cases)]

        means = {}
        for model in models:
            scores = all_scores[model]
            sampled = [scores[i] for i in sampled_indices if i < len(scores)]
            means[model] = sum(sampled) / len(sampled) if sampled else 0

        for i, a in enumerate(models):
            for b in models[i + 1:]:
                if means[a] > means[b]:
                    win_counts[a][b] += 1
                elif means[b] > means[a]:
                    win_counts[b][a] += 1
                else:
                    # Tie: split
                    win_counts[a][b] += 0.5
                    win_counts[b][a] += 0.5

    return {
        a: {b: round(count / n_bootstrap, 3) for b, count in others.items()}
        for a, others in win_counts.items()
    }


def leave_one_case_out(
    all_scores: dict[str, list[dict]],
) -> list[dict]:
    """
    Leave-one-case-out sensitivity analysis.

    For each case: remove it, recompute means and rankings.
    Report which cases, when removed, change the top-ranked model.

    Input: {model_id: [score_dicts]} where each dict has "case_id" and "total"
    Returns: list of {removed_case_id, rankings_changed, new_top_model, original_top_model}
    """
    models = sorted(all_scores.keys())
    if not models:
        return []

    # Original rankings
    orig_means = {
        m: sum(s["total"] for s in scores) / len(scores)
        for m, scores in all_scores.items() if scores
    }
    orig_top = max(orig_means, key=orig_means.get) if orig_means else None

    # All case IDs
    all_case_ids = set()
    for scores in all_scores.values():
        for s in scores:
            all_case_ids.add(s["case_id"])

    results = []
    for removed_case in sorted(all_case_ids):
        reduced_means = {}
        for m, scores in all_scores.items():
            remaining = [s["total"] for s in scores if s["case_id"] != removed_case]
            if remaining:
                reduced_means[m] = sum(remaining) / len(remaining)

        new_top = max(reduced_means, key=reduced_means.get) if reduced_means else None
        rankings_changed = new_top != orig_top

        results.append({
            "removed_case_id": removed_case,
            "rankings_changed": rankings_changed,
            "original_top_model": orig_top,
            "new_top_model": new_top,
            "mean_deltas": {
                m: round(reduced_means.get(m, 0) - orig_means.get(m, 0), 3)
                for m in models
            },
        })

    return results


def compute_fragility_summary(
    rank_dist: dict[str, dict[int, float]],
    superiority_matrix: dict[str, dict[str, float]],
    loo_results: list[dict],
) -> dict:
    """
    Compute a fragility summary for the leaderboard.

    Fragility indicators:
      - rank_entropy: how spread out is each model's rank distribution (high = unstable)
      - top1_confidence: P(current #1 stays #1) across bootstrap
      - loo_sensitivity: fraction of cases whose removal changes top-1
      - min_superiority: minimum P(a>b) for the top-ranked model vs any other
    """
    import math

    models = sorted(rank_dist.keys())
    if not models:
        return {"fragile": True, "reason": "no models"}

    # Rank entropy per model
    rank_entropy = {}
    for model, dist in rank_dist.items():
        entropy = 0.0
        for rank, freq in dist.items():
            if freq > 0:
                entropy -= freq * math.log2(freq)
        rank_entropy[model] = round(entropy, 3)

    # Top-1 confidence
    top1_freq = top_k_frequency(rank_dist, k=1)
    current_top = max(top1_freq, key=top1_freq.get) if top1_freq else None
    top1_confidence = top1_freq.get(current_top, 0) if current_top else 0

    # LOO sensitivity
    n_loo_changes = sum(1 for r in loo_results if r["rankings_changed"])
    loo_sensitivity = n_loo_changes / len(loo_results) if loo_results else 0

    # Min superiority for top model
    min_superiority = 1.0
    if current_top and current_top in superiority_matrix:
        for other, prob in superiority_matrix[current_top].items():
            min_superiority = min(min_superiority, prob)

    fragile = (
        top1_confidence < 0.7
        or loo_sensitivity > 0.3
        or min_superiority < 0.6
    )

    return {
        "current_top_model": current_top,
        "top1_confidence": round(top1_confidence, 3),
        "min_superiority_vs_any": round(min_superiority, 3),
        "loo_sensitivity": round(loo_sensitivity, 3),
        "n_loo_rank_changes": n_loo_changes,
        "n_loo_total": len(loo_results),
        "rank_entropy": rank_entropy,
        "fragile": fragile,
        "fragility_reasons": [
            r for r in [
                "top1_confidence < 0.7" if top1_confidence < 0.7 else None,
                "loo_sensitivity > 0.3" if loo_sensitivity > 0.3 else None,
                "min_superiority < 0.6" if min_superiority < 0.6 else None,
            ] if r
        ],
    }


def run_rank_stability_analysis(
    all_scores: dict[str, list[dict]],
    n_bootstrap: int = 1000,
    seed: int = 42,
    output_path: Optional[Path] = None,
) -> dict:
    """
    Full rank stability analysis. Produces rank_stability.json.

    Input: {model_id: [score_dicts]} where each dict has "case_id" and "total"
    """
    # Extract totals for bootstrap
    totals_by_model = {
        m: [s["total"] for s in scores]
        for m, scores in all_scores.items()
    }

    rank_dist = bootstrap_rank_distribution(totals_by_model, n_bootstrap, seed)
    top1 = top_k_frequency(rank_dist, k=1)
    sup_matrix = pairwise_superiority_matrix(totals_by_model, n_bootstrap, seed)
    loo = leave_one_case_out(all_scores)
    fragility = compute_fragility_summary(rank_dist, sup_matrix, loo)

    # Decision tiers
    models = sorted(rank_dist.keys())
    top3 = top_k_frequency(rank_dist, k=3)
    decision_tiers = _assign_decision_tiers(rank_dist, sup_matrix, fragility)

    result = {
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "rank_distributions": rank_dist,
        "top1_frequency": top1,
        "top3_frequency": top3,
        "pairwise_superiority_matrix": sup_matrix,
        "leave_one_case_out": loo,
        "fragility_summary": fragility,
        "decision_tiers": decision_tiers,
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

    return result


# ── Decision Tiers ──────────────────────────────────────────────────────────

def _assign_decision_tiers(
    rank_dist: dict[str, dict[int, float]],
    superiority_matrix: dict[str, dict[str, float]],
    fragility: dict,
) -> dict[str, dict]:
    """
    Assign decision-oriented tiers to each model.

    Tiers:
      - "leader": P(top-1) > 0.7 AND min P(A>B) > 0.6
      - "co_leader": P(top-1) > 0.3 AND not clearly dominated
      - "top_cluster": P(top-3) > 0.7 but not a leader
      - "mid_pack": not clearly in top cluster or bottom
      - "below_frontier": P(top-3) < 0.2
    """
    models = sorted(rank_dist.keys())
    top1 = top_k_frequency(rank_dist, k=1)
    top3 = top_k_frequency(rank_dist, k=3)

    tiers = {}
    for m in models:
        p_top1 = top1.get(m, 0)
        p_top3 = top3.get(m, 0)

        # Check minimum superiority vs all others
        min_sup = 1.0
        if m in superiority_matrix:
            for other, prob in superiority_matrix[m].items():
                min_sup = min(min_sup, prob)

        if p_top1 > 0.7 and min_sup > 0.6:
            tier = "leader"
        elif p_top1 > 0.3:
            tier = "co_leader"
        elif p_top3 > 0.7:
            tier = "top_cluster"
        elif p_top3 < 0.2:
            tier = "below_frontier"
        else:
            tier = "mid_pack"

        tiers[m] = {
            "tier": tier,
            "top1_probability": round(p_top1, 3),
            "top3_probability": round(p_top3, 3),
            "min_superiority": round(min_sup, 3),
        }

    return tiers
