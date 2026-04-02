"""
IM-TRACE Aggregation — Bootstrap CI, weighted aggregation, Bradley-Terry stubs.

Implements:
  1. Naive mean aggregation
  2. Weighted aggregation with safety cap/penalty
  3. Bootstrap confidence intervals
  4. High-disagreement case detection
  5. Hard-cases export
  6. Stub: many-facet Rasch (not safely implemented)
  7. Stub: Dawid-Skene latent consensus (not safely implemented)
  8. Bradley-Terry model fitting for pairwise results
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from typing import Optional

from im_trace.cases.schema.models import AggregateCaseScore, SubscaleProfile


# ── Bootstrap Confidence Intervals ──────────────────────────────────────────

def bootstrap_ci(
    scores: list[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a set of scores.

    Returns: (mean, ci_lower, ci_upper)
    """
    if not scores:
        return (0.0, 0.0, 0.0)
    if len(scores) == 1:
        return (scores[0], scores[0], scores[0])

    rng = random.Random(seed)
    n = len(scores)
    means = []

    for _ in range(n_bootstrap):
        sample = [rng.choice(scores) for _ in range(n)]
        means.append(sum(sample) / n)

    means.sort()
    alpha = (1 - ci) / 2
    lower_idx = int(alpha * n_bootstrap)
    upper_idx = int((1 - alpha) * n_bootstrap) - 1

    return (
        sum(scores) / n,
        means[max(0, lower_idx)],
        means[min(len(means) - 1, upper_idx)],
    )


# ── Naive Mean Aggregation ──────────────────────────────────────────────────

def aggregate_naive(
    annotations: list[dict],
) -> dict:
    """
    Simple mean aggregation across annotations for the same case+model.

    Input: list of score dicts (from absolute.scorer.compute_total)
    Returns: mean scores with bootstrap CIs
    """
    if not annotations:
        return {}

    fields = ["r1", "r2", "r3_composite", "r4_composite", "total"]
    result = {}

    for field in fields:
        values = [a[field] for a in annotations if field in a]
        mean, ci_lower, ci_upper = bootstrap_ci(values)
        result[field] = {
            "mean": round(mean, 3),
            "ci_lower": round(ci_lower, 3),
            "ci_upper": round(ci_upper, 3),
            "n": len(values),
        }

    return result


# ── Weighted Aggregation with Safety Penalty ────────────────────────────────

def aggregate_weighted(
    annotations: list[dict],
    safety_weight: float = 2.0,
) -> dict:
    """
    Weighted aggregation where safety-capped annotations carry extra weight
    in pulling down the aggregate.

    If ANY annotation has safety_capped=True, the aggregate total is
    penalized by 15% (adjustable).
    """
    naive = aggregate_naive(annotations)
    if not naive:
        return naive

    any_safety_violation = any(a.get("safety_capped", False) for a in annotations)
    if any_safety_violation:
        penalty = 0.85  # 15% penalty
        naive["total"]["mean"] = round(naive["total"]["mean"] * penalty, 3)
        naive["total"]["safety_penalty_applied"] = True
        naive["total"]["penalty_factor"] = penalty

    return naive


# ── Disagreement Detection ──────────────────────────────────────────────────

def detect_disagreement(
    annotations: list[dict],
    threshold: float = 1.5,
) -> dict:
    """
    Detect high-disagreement cases.

    Returns disagreement metrics and flags cases where the range
    of total scores exceeds the threshold.
    """
    totals = [a["total"] for a in annotations]
    if len(totals) < 2:
        return {"n": len(totals), "is_high_disagreement": False}

    score_range = max(totals) - min(totals)
    variance = sum((t - sum(totals)/len(totals))**2 for t in totals) / len(totals)

    return {
        "n": len(totals),
        "range": round(score_range, 3),
        "variance": round(variance, 3),
        "min": round(min(totals), 3),
        "max": round(max(totals), 3),
        "is_high_disagreement": score_range >= threshold,
        "disagreement_type": _classify_disagreement(annotations) if score_range >= threshold else None,
    }


def _classify_disagreement(annotations: list[dict]) -> Optional[str]:
    """Classify the type of disagreement for hard-cases export."""
    # Check if disagreement is mainly on safety
    safety_scores = [a.get("r3_composite", 0) for a in annotations]
    if max(safety_scores) - min(safety_scores) > 0.5:
        return "safety_critical_conflict"

    # Check if disagreement is mainly on R4
    r4_scores = [a.get("r4_composite", 0) for a in annotations]
    if max(r4_scores) - min(r4_scores) > 0.5:
        return "reasoning_quality_disagreement"

    return "general_disagreement"


# ── Hard Cases Export ────────────────────────────────────────────────────────

def identify_hard_cases(
    case_annotations: dict[str, list[dict]],
    disagreement_threshold: float = 1.5,
) -> list[dict]:
    """
    Identify hard cases across a benchmark run.

    Hard cases are those with:
      - High human-vs-human disagreement
      - High human-vs-LLM disagreement
      - Safety-critical conflicts
      - Unstable pairwise ordering

    Returns list of hard case records for the hard_cases export.
    """
    hard_cases = []

    for case_id, annotations in case_annotations.items():
        # Split by judge type
        human = [a for a in annotations if a.get("judge_type") == "physician"]
        llm = [a for a in annotations if a.get("judge_type") == "llm_judge"]

        # Human-vs-human disagreement
        if len(human) >= 2:
            h_disagree = detect_disagreement(human, disagreement_threshold)
            if h_disagree["is_high_disagreement"]:
                hard_cases.append({
                    "case_id": case_id,
                    "reason": "human_vs_human_disagreement",
                    "details": h_disagree,
                })

        # Human-vs-LLM disagreement
        if human and llm:
            h_mean = sum(a["total"] for a in human) / len(human)
            l_mean = sum(a["total"] for a in llm) / len(llm)
            if abs(h_mean - l_mean) >= disagreement_threshold:
                hard_cases.append({
                    "case_id": case_id,
                    "reason": "human_vs_llm_disagreement",
                    "details": {
                        "human_mean": round(h_mean, 3),
                        "llm_mean": round(l_mean, 3),
                        "delta": round(abs(h_mean - l_mean), 3),
                    },
                })

        # Safety-critical conflicts
        any_safe = any(not a.get("safety_capped", False) for a in annotations)
        any_unsafe = any(a.get("safety_capped", False) for a in annotations)
        if any_safe and any_unsafe:
            hard_cases.append({
                "case_id": case_id,
                "reason": "safety_critical_conflict",
                "details": {
                    "n_safe": sum(1 for a in annotations if not a.get("safety_capped", False)),
                    "n_unsafe": sum(1 for a in annotations if a.get("safety_capped", False)),
                },
            })

    return hard_cases


# ── Bradley-Terry Model (Pairwise → Ratings) ────────────────────────────────

def fit_bradley_terry(
    comparisons: list[dict],
    max_iter: int = 100,
    tol: float = 1e-6,
) -> dict[str, float]:
    """
    Fit Bradley-Terry model to pairwise comparison data.

    Input: list of {"winner": model_id, "loser": model_id}
    Returns: {model_id: strength_parameter}

    The strength parameters can be converted to Elo-like ratings:
      elo = 400 * log10(strength) + 1500
    """
    if not comparisons:
        return {}

    # Collect all models
    models = set()
    for c in comparisons:
        models.add(c["winner"])
        models.add(c["loser"])
    models = sorted(models)

    # Initialize strengths
    strength = {m: 1.0 for m in models}

    # Count wins
    wins = defaultdict(int)
    matchups = defaultdict(int)
    for c in comparisons:
        wins[c["winner"]] += 1
        matchups[(c["winner"], c["loser"])] += 1
        matchups[(c["loser"], c["winner"])] += 1

    # Iterative MLE
    for iteration in range(max_iter):
        old_strength = dict(strength)

        for m in models:
            numerator = sum(
                1 for c in comparisons if c["winner"] == m
            )
            denominator = 0.0
            for other in models:
                if other == m:
                    continue
                n_games = matchups.get((m, other), 0)
                if n_games > 0:
                    denominator += n_games / (strength[m] + strength[other])

            if denominator > 0:
                strength[m] = numerator / denominator

        # Normalize (fix one model's strength to prevent drift)
        total = sum(strength.values())
        for m in models:
            strength[m] /= total / len(models)

        # Check convergence
        max_change = max(
            abs(strength[m] - old_strength[m]) for m in models
        )
        if max_change < tol:
            break

    return strength


def bt_to_elo(strengths: dict[str, float], base: float = 1500, scale: float = 400) -> dict[str, float]:
    """Convert Bradley-Terry strengths to Elo-like ratings."""
    return {
        model: round(base + scale * math.log10(max(s, 1e-10)), 1)
        for model, s in strengths.items()
    }


def rank_models(strengths: dict[str, float]) -> list[dict]:
    """Rank models by Bradley-Terry strength."""
    elo = bt_to_elo(strengths)
    ranked = sorted(elo.items(), key=lambda x: x[1], reverse=True)
    return [
        {"rank": i + 1, "model_id": model, "elo_rating": rating, "bt_strength": round(strengths[model], 4)}
        for i, (model, rating) in enumerate(ranked)
    ]


# ── Stubs for Advanced Psychometric Models ──────────────────────────────────

def many_facet_rasch_stub(annotations: list[dict]) -> dict:
    """
    STUB: Many-Facet Rasch Model for rater severity adjustment.

    TODO: Implement using a dedicated psychometric library (e.g., facets,
    pyirt, or custom JMLE). This would model:
      - Item difficulty (case hardness)
      - Rater severity (annotator leniency/harshness)
      - Model ability (LLM performance)

    Not implemented because incorrect implementation is worse than
    not implementing. Use external psychometric software for now.
    """
    return {
        "status": "stub",
        "message": "Many-Facet Rasch model not yet implemented. Use external psychometric software.",
        "recommended_tools": ["facets (Linacre)", "TAM R package", "pyirt"],
    }


def dawid_skene_stub(annotations: list[dict]) -> dict:
    """
    STUB: Dawid-Skene latent consensus model.

    TODO: Implement EM algorithm for estimating true labels from
    multiple noisy annotators. This would model:
      - True latent score for each case
      - Each annotator's confusion matrix
      - Posterior probability of each score

    Not implemented because the EM convergence properties with
    ordinal clinical scores need careful validation.
    """
    return {
        "status": "stub",
        "message": "Dawid-Skene consensus model not yet implemented.",
        "recommended_tools": ["dawid-skene Python package", "MACE", "custom EM"],
    }
