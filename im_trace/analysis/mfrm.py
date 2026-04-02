"""
IM-TRACE Many-Facet Rasch Model (MFRM) — Analysis Layer.

JMLE (Joint Maximum Likelihood Estimation) for three-facet rating scale model:
  - Model ability   (theta_j)
  - Case difficulty (beta_i)
  - Rater severity  (alpha_r)

Log-odds formulation (Partial Credit variant reduced to Rating Scale):
  log P(X >= k) / P(X < k) = theta_j - beta_i - alpha_r - tau_k

Where tau_k are category thresholds shared across all items (Rating Scale Model).

IMPORTANT: This is an ANALYSIS TOOL for research purposes.
  - Use only when N >= 30 observations per facet level
  - Treat estimates as directional, not definitive
  - Never report standard errors — they require more data than typical pilot runs
  - Compare raw vs. adjusted scores; large divergences warrant manual review
  - All outputs carry analysis_only: true

Use standard library only (math, collections, warnings).
"""

from __future__ import annotations

import math
import warnings
from collections import defaultdict
from pathlib import Path


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_log(x: float, fallback: float = -10.0) -> float:
    if x <= 0:
        return fallback
    return math.log(x)


def _logistic(x: float) -> float:
    """Numerically stable logistic."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


# ── Rating Scale Model Probability ──────────────────────────────────────────

def _rsm_category_probs(theta: float, beta: float, alpha: float, thresholds: list[float]) -> list[float]:
    """
    Compute category probabilities under the Rating Scale Model.

    For K categories (0 .. K-1), K-1 thresholds tau_1 .. tau_{K-1}.
    P(X=k) ∝ exp( sum_{m=1}^{k} (theta - beta - alpha - tau_m) ), with sum_0 = 0.

    Returns list of K probabilities summing to 1.
    """
    K = len(thresholds) + 1  # number of categories
    log_numerators = []
    cumsum = 0.0
    log_numerators.append(0.0)  # category 0 baseline
    for k in range(1, K):
        cumsum += (theta - beta - alpha - thresholds[k - 1])
        log_numerators.append(cumsum)

    # Subtract max for numerical stability before exp
    max_log = max(log_numerators)
    exps = [math.exp(ln - max_log) for ln in log_numerators]
    total = sum(exps)
    return [e / total for e in exps]


def _rsm_expected(theta: float, beta: float, alpha: float, thresholds: list[float]) -> float:
    """Expected score under Rating Scale Model."""
    probs = _rsm_category_probs(theta, beta, alpha, thresholds)
    return sum(k * p for k, p in enumerate(probs))


# ── Data Preparation ─────────────────────────────────────────────────────────

def prepare_mfrm_data(annotations: list[dict]) -> dict:
    """
    Extract structured tuples from raw annotation dicts.

    Expected keys in each dict:
      model_id   — the model being evaluated
      case_id    — clinical case identifier
      evaluator_id OR judge_id — rater identifier
      total      — numeric score (0, 1, or 2 ordinal)

    Returns:
      {
        "records":        list of (model_id, case_id, rater_id, score) tuples,
        "model_ids":      sorted list of unique model IDs,
        "case_ids":       sorted list of unique case IDs,
        "rater_ids":      sorted list of unique rater IDs,
        "score_range":    (min_score, max_score),
        "n_records":      int,
        "warnings":       list[str],
      }
    """
    records = []
    parse_warnings = []

    for i, ann in enumerate(annotations):
        model_id = ann.get("model_id")
        case_id = ann.get("case_id")
        rater_id = ann.get("evaluator_id") or ann.get("judge_id")
        score = ann.get("total")

        missing = [k for k, v in [("model_id", model_id), ("case_id", case_id),
                                   ("evaluator_id/judge_id", rater_id), ("total", score)]
                   if v is None]
        if missing:
            parse_warnings.append(
                f"Annotation index {i} missing fields: {missing}. Skipping."
            )
            continue

        try:
            score_float = float(score)
        except (TypeError, ValueError):
            parse_warnings.append(
                f"Annotation index {i} has non-numeric total={score!r}. Skipping."
            )
            continue

        records.append((str(model_id), str(case_id), str(rater_id), score_float))

    all_models = sorted({r[0] for r in records})
    all_cases  = sorted({r[1] for r in records})
    all_raters = sorted({r[2] for r in records})
    scores     = [r[3] for r in records]

    return {
        "records":     records,
        "model_ids":   all_models,
        "case_ids":    all_cases,
        "rater_ids":   all_raters,
        "score_range": (min(scores), max(scores)) if scores else (0.0, 0.0),
        "n_records":   len(records),
        "warnings":    parse_warnings,
    }


# ── JMLE Fitting ─────────────────────────────────────────────────────────────

def fit_mfrm(
    data: dict,
    max_iter: int = 100,
    lr: float = 0.1,
    min_observations: int = 3,
) -> dict:
    """
    JMLE fit of the Rating Scale Model with three facets.

    Convergence criterion: max parameter change < 1e-4.

    Args:
        data:             Output of prepare_mfrm_data()
        max_iter:         Maximum JMLE iterations
        lr:               Learning rate for parameter updates (gradient step size)
        min_observations: Minimum observations for a facet level to be estimated.
                          Levels below threshold are omitted with a warning.

    Returns dict with keys:
        model_ability:   {model_id: theta}
        case_difficulty: {case_id: beta}
        rater_severity:  {rater_id: alpha}
        thresholds:      [tau_1, tau_2] (for 0/1/2 ordinal, 2 thresholds)
        convergence:     bool
        iterations:      int
        warnings:        list[str]
        analysis_only:   True (always)
    """
    records   = data.get("records", [])
    fit_warns = list(data.get("warnings", []))

    if not records:
        fit_warns.append("No records to fit.")
        return {
            "model_ability": {}, "case_difficulty": {}, "rater_severity": {},
            "thresholds": [0.0, 1.0], "convergence": False, "iterations": 0,
            "warnings": fit_warns, "analysis_only": True,
        }

    # Count observations per facet level
    model_obs  = defaultdict(int)
    case_obs   = defaultdict(int)
    rater_obs  = defaultdict(int)
    for m, c, r, _ in records:
        model_obs[m] += 1
        case_obs[c]  += 1
        rater_obs[r] += 1

    def _check_sparse(obs_dict: dict, label: str) -> set[str]:
        sparse = set()
        for fid, count in obs_dict.items():
            if count < min_observations:
                fit_warns.append(
                    f"{label} '{fid}' has only {count} observation(s) "
                    f"(< min_observations={min_observations}). Excluding from estimation."
                )
                sparse.add(fid)
        return sparse

    sparse_models = _check_sparse(model_obs,  "model_id")
    sparse_cases  = _check_sparse(case_obs,   "case_id")
    sparse_raters = _check_sparse(rater_obs,  "rater_id")

    # Filter records: only keep those where all three facets are estimable
    filtered = [
        (m, c, r, s) for m, c, r, s in records
        if m not in sparse_models and c not in sparse_cases and r not in sparse_raters
    ]

    if not filtered:
        fit_warns.append(
            "No records remain after sparse-facet exclusion. Cannot fit model."
        )
        return {
            "model_ability": {}, "case_difficulty": {}, "rater_severity": {},
            "thresholds": [0.0, 1.0], "convergence": False, "iterations": 0,
            "warnings": fit_warns, "analysis_only": True,
        }

    all_models = sorted({r[0] for r in filtered})
    all_cases  = sorted({r[1] for r in filtered})
    all_raters = sorted({r[2] for r in filtered})

    # Initialize parameters at zero
    theta = {m: 0.0 for m in all_models}
    beta  = {c: 0.0 for c in all_cases}
    alpha = {r: 0.0 for r in all_raters}
    # For 0/1/2 scoring: 2 thresholds. Initialise slightly apart.
    tau = [-0.5, 0.5]  # tau[0] = tau_1, tau[1] = tau_2

    converged = False
    iterations = 0

    for it in range(max_iter):
        max_change = 0.0

        # ── Update model ability (theta) ─────────────────────────────────
        for m in all_models:
            obs_m = [(c, r, s) for mm, c, r, s in filtered if mm == m]
            residual = 0.0
            for c, r, s in obs_m:
                expected = _rsm_expected(theta[m], beta[c], alpha[r], tau)
                residual += s - expected
            step = lr * residual / max(len(obs_m), 1)
            change = abs(step)
            theta[m] += step
            max_change = max(max_change, change)

        # ── Update case difficulty (beta) ────────────────────────────────
        for c in all_cases:
            obs_c = [(m, r, s) for mm, cc, r, s in filtered if cc == c]
            residual = 0.0
            for m, r, s in obs_c:
                expected = _rsm_expected(theta[m], beta[c], alpha[r], tau)
                residual += s - expected
            step = -lr * residual / max(len(obs_c), 1)  # beta is subtracted
            change = abs(step)
            beta[c] += step
            max_change = max(max_change, change)

        # ── Update rater severity (alpha) ────────────────────────────────
        for r in all_raters:
            obs_r = [(m, c, s) for mm, c, rr, s in filtered if rr == r]
            residual = 0.0
            for m, c, s in obs_r:
                expected = _rsm_expected(theta[m], beta[c], alpha[r], tau)
                residual += s - expected
            step = -lr * residual / max(len(obs_r), 1)  # alpha is subtracted
            change = abs(step)
            alpha[r] += step
            max_change = max(max_change, change)

        # ── Update thresholds (tau) ──────────────────────────────────────
        for k_idx in range(len(tau)):
            k = k_idx + 1  # threshold index (1-based)
            residual = 0.0
            for m, c, r, s in filtered:
                probs = _rsm_category_probs(theta[m], beta[c], alpha[r], tau)
                # Partial derivative of log-likelihood w.r.t. tau_k:
                # dL/d(tau_k) = - sum_{x >= k} P(X >= k) + I(x >= k)
                prob_ge_k = sum(probs[kk] for kk in range(k, len(probs)))
                indicator = 1.0 if s >= k else 0.0
                residual += indicator - prob_ge_k
            step = -lr * residual / max(len(filtered), 1)
            change = abs(step)
            tau[k_idx] += step
            max_change = max(max_change, change)

        # ── Centering constraints ────────────────────────────────────────
        # Anchor: mean theta = 0, mean beta = 0, mean alpha = 0
        theta_mean = sum(theta.values()) / max(len(theta), 1)
        beta_mean  = sum(beta.values())  / max(len(beta), 1)
        alpha_mean = sum(alpha.values()) / max(len(alpha), 1)
        for m in all_models: theta[m] -= theta_mean
        for c in all_cases:  beta[c]  -= beta_mean
        for r in all_raters: alpha[r] -= alpha_mean

        iterations = it + 1
        if max_change < 1e-4:
            converged = True
            break

    if not converged:
        fit_warns.append(
            f"JMLE did not converge in {max_iter} iterations. "
            "Estimates may be unstable. Increase max_iter or check data sparsity."
        )

    return {
        "model_ability":   {m: round(v, 4) for m, v in theta.items()},
        "case_difficulty": {c: round(v, 4) for c, v in beta.items()},
        "rater_severity":  {r: round(v, 4) for r, v in alpha.items()},
        "thresholds":      [round(t, 4) for t in tau],
        "convergence":     converged,
        "iterations":      iterations,
        "warnings":        fit_warns,
        "analysis_only":   True,
    }


# ── Diagnostics ───────────────────────────────────────────────────────────────

def mfrm_diagnostics(fit_result: dict) -> dict:
    """
    Compute rater-level diagnostics from a fitted MFRM result.

    Metrics per rater:
      centrality  — signed distance from mean severity (positive = harsher)
      extremity   — fraction of extreme scores (0 or max) in that rater's records
                    NOTE: requires raw data; computed as abs(alpha - mean_alpha)
                    as a proxy when raw data unavailable.
      is_outlier  — |severity| > 2 SD from mean severity

    Returns:
      {
        "rater_stats":      {rater_id: {centrality, extremity_proxy, is_outlier}},
        "mean_severity":    float,
        "sd_severity":      float,
        "outlier_raters":   list[str],
        "analysis_only":    True,
        "warnings":         list[str],
      }
    """
    diag_warns = []
    severities = fit_result.get("rater_severity", {})

    if not severities:
        return {
            "rater_stats": {}, "mean_severity": 0.0, "sd_severity": 0.0,
            "outlier_raters": [], "analysis_only": True,
            "warnings": ["No rater severity estimates found in fit_result."],
        }

    vals = list(severities.values())
    mean_sev = sum(vals) / len(vals)
    variance = sum((v - mean_sev) ** 2 for v in vals) / max(len(vals) - 1, 1)
    sd_sev   = math.sqrt(variance) if variance > 0 else 0.0

    if sd_sev == 0.0:
        diag_warns.append(
            "All rater severity estimates are identical (SD=0). "
            "Check for model non-identification or data sparsity."
        )

    rater_stats = {}
    outlier_raters = []

    for rater_id, sev in severities.items():
        centrality = round(sev - mean_sev, 4)
        # Extremity proxy: |deviation| / (SD + epsilon) — how far from the pack
        extremity_proxy = round(abs(centrality) / (sd_sev + 1e-9), 4)
        is_outlier = abs(centrality) > 2.0 * sd_sev if sd_sev > 0 else False

        rater_stats[rater_id] = {
            "severity":        round(sev, 4),
            "centrality":      centrality,
            "extremity_proxy": extremity_proxy,
            "is_outlier":      is_outlier,
        }
        if is_outlier:
            outlier_raters.append(rater_id)

    if outlier_raters:
        diag_warns.append(
            f"Outlier raters detected (|severity| > 2 SD): {outlier_raters}. "
            "Review their annotations manually before including in analysis."
        )

    return {
        "rater_stats":    rater_stats,
        "mean_severity":  round(mean_sev, 4),
        "sd_severity":    round(sd_sev, 4),
        "outlier_raters": outlier_raters,
        "analysis_only":  True,
        "warnings":       diag_warns,
    }


# ── Raw vs. Adjusted Comparison ───────────────────────────────────────────────

def compare_raw_vs_adjusted(
    raw_scores: dict[str, float],
    fit_result: dict,
) -> list[dict]:
    """
    Side-by-side comparison of raw mean scores vs. MFRM-adjusted latent ability.

    Args:
        raw_scores:  {model_id: raw_mean_total_score}
        fit_result:  Output of fit_mfrm()

    Returns:
        List of dicts, one per model, sorted descending by adjusted theta:
          {
            model_id:            str,
            raw_mean:            float,
            adjusted_theta:      float or None,
            raw_rank:            int,
            adjusted_rank:       int,
            rank_change:         int (positive = improved rank after adjustment),
            analysis_only:       True,
          }
    """
    abilities = fit_result.get("model_ability", {})
    comp_warns = []

    all_model_ids = sorted(set(list(raw_scores.keys()) + list(abilities.keys())))

    records = []
    for m in all_model_ids:
        raw  = raw_scores.get(m)
        adj  = abilities.get(m)
        if raw is None:
            comp_warns.append(f"Model '{m}' has MFRM estimate but no raw score. Skipping.")
            continue
        records.append({"model_id": m, "raw_mean": round(raw, 4), "adjusted_theta": adj})

    # Rank by raw
    raw_sorted = sorted(records, key=lambda x: x["raw_mean"], reverse=True)
    for rank, rec in enumerate(raw_sorted, 1):
        rec["raw_rank"] = rank

    # Rank by adjusted (None → push to end)
    adj_sorted = sorted(
        records,
        key=lambda x: (x["adjusted_theta"] is None, -(x["adjusted_theta"] or -999)),
    )
    for rank, rec in enumerate(adj_sorted, 1):
        rec["adjusted_rank"] = rank

    # Compute rank change (positive = better rank after adjustment)
    for rec in records:
        if rec.get("adjusted_theta") is not None:
            rec["rank_change"] = rec["raw_rank"] - rec["adjusted_rank"]
        else:
            rec["rank_change"] = None
        rec["analysis_only"] = True

    # Final sort: by adjusted_theta descending
    records.sort(
        key=lambda x: (x["adjusted_theta"] is None, -(x["adjusted_theta"] or -999))
    )

    if comp_warns:
        for w in comp_warns:
            warnings.warn(w, stacklevel=2)

    return records
