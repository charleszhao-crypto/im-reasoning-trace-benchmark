# Adaptive Ranking and Frontier-Set Calibration

*IM-TRACE v1.2 — Statistical efficiency, ranking robustness, anti-gaming durability*

---

## Ranking Backend Abstraction

IM-TRACE v1.2 introduces a pluggable ranking backend interface. All backends accept pairwise comparison records and return a standardized `RankingResult` with:

- Latent quality scores
- Elo-like ratings
- Pairwise superiority matrix P(A > B) for all model pairs
- Top-k confidence (probability of being in top 1, 2, 3...)
- Backend-specific diagnostics

### Official Backend: Bradley-Terry

Bradley-Terry with Laplace smoothing remains the **official** ranking backend. Results from this backend are used in validated leaderboards and publications. BT is well-understood, has clean probabilistic interpretation, and is the standard in the LLM evaluation community (Chatbot Arena, MT-Bench).

### Analysis Backend: Spectral Ranking

A spectral ranking backend (Fiedler vector of the comparison graph Laplacian) provides an independent ranking estimate for cross-validation. **This backend is analysis-only — it is NOT used for official rankings.**

The diagnostic value: if BT and spectral rankings agree, the ranking signal is likely stable. If they diverge significantly, that indicates data sparsity or inconsistency is dominating the result — the rankings should be treated with lower confidence regardless of which backend is used.

**Limitations of the spectral backend:**
- Power iteration eigenvector computation (no scipy dependency)
- Bootstrap CI uses net-win-rate proxy, not full spectral refit per sample
- Less principled uncertainty quantification than BT
- Sensitive to graph connectivity

---

## Active Pairwise Scheduling

Not all comparisons are equally informative. The active selection module recommends the next most valuable pairwise comparisons given a limited budget.

### Acquisition Function (Heuristic)

Each candidate comparison (model pair × case) receives an acquisition score:

```
acquisition = (
    0.4 × uncertainty_score +    # P(A>B) near 0.5 → high entropy
    0.3 × leverage_score +       # close latent scores → high leverage
    0.2 × safety_score +         # safety-critical case flag
    0.1 × hard_case_score        # hard-case flag
) × novelty_discount / annotation_cost
```

**Uncertainty** is measured by the Shannon entropy of the Bernoulli(P(A>B)) distribution — maximum at P=0.5 (most uncertain), zero at P=0 or P=1.

**Leverage** is inversely proportional to the normalized score difference between models — close scores mean the comparison has more potential to change rankings.

**Safety** and **hard case** flags add constant bonuses to prioritize clinically important comparisons.

**Novelty** discounts comparisons that have already been done on the same case — the first comparison is most valuable; subsequent comparisons on the same pair have diminishing returns.

### Current Status: Heuristic

This is a practical heuristic, not a formally optimal acquisition function. It works well for budget allocation in early benchmark runs.

**TODO:** Upgrade to mutual-information-based or posterior-risk-based criteria when a Bayesian posterior over latent qualities is available. The interface is designed to make this substitution clean — the `suggest_next_comparisons` API is stable.

---

## Decision-Oriented Rank Reporting

### Decision Tiers

Models are assigned to interpretive tiers based on bootstrap rank distributions:

| Tier | Criteria | Interpretation |
|------|----------|----------------|
| **leader** | P(top-1) > 0.7 AND min P(A>B) > 0.6 | Clear leader; safe to recommend |
| **co_leader** | P(top-1) > 0.3 | Not clearly behind; treat as peer |
| **top_cluster** | P(top-3) > 0.7 | In the competitive set but not leading |
| **mid_pack** | Neither top nor bottom | Insufficiently differentiated |
| **below_frontier** | P(top-3) < 0.2 | Clearly behind the competitive set |

**Why tiers matter:** in most clinical AI deployment decisions, you do not care whether a model is "rank 2" vs. "rank 3" if their superiority probabilities overlap heavily. What you care about is whether one model is reliably safer, more calibrated, or more suitable for a given deployment context.

### Leaderboard JSON Fields (v1.2)

```json
{
  "decision_tiers": {
    "model-alpha": {
      "tier": "co_leader",
      "top1_probability": 0.763,
      "top3_probability": 1.0,
      "min_superiority": 0.729
    }
  },
  "fragility_summary": {
    "fragile": false,
    "top1_confidence": 0.763,
    "min_superiority_vs_any": 0.729,
    "loo_sensitivity": 0.25
  }
}
```

---

## Frontier-Set Governance

Hard cases are no longer just diagnostic byproducts — they are the source for a **renewable frontier set**.

### Governance Fields

Each hard case now carries:

| Field | Type | Meaning |
|-------|------|---------|
| `disagreement_score` | 0-1 | Normalized disagreement intensity |
| `judge_confidence_penalty` | 0-1 | Fraction of annotations with low confidence |
| `ranking_leverage` | 0/1 | Whether removing this case changes top-1 |
| `safety_weight` | 0/1 | Whether safety-critical disagreement exists |
| `perturbation_instability` | null/float | Reserved for perturbation family data |
| `conditional_rank_divergence` | null/float | Reserved for stratified ranking data |
| `recommended_action` | string | Governance action (see below) |

### Recommended Actions

| Action | When | Meaning |
|--------|------|---------|
| `send_to_human` | Safety-critical + high disagreement | Needs physician adjudication before any automated use |
| `needs_rubric_revision` | High judge confidence penalty | The rubric may be ambiguous for this case type |
| `holdout_candidate` | High ranking leverage | Good candidate for a delayed-release holdout set |
| `promote_to_frontier_set` | Default hard case | Add to the evolving frontier for future evaluation |

---

## Statistical Caveats

1. **Benchmark validity ≠ model performance.** IM-TRACE scores measure performance on a specific case set with specific rubrics. They do not measure general clinical reasoning ability. See NIST AI 800-3 on the distinction between benchmark accuracy and generalized accuracy.

2. **Official rankings use Bradley-Terry.** The spectral backend is analysis-only. If BT and spectral disagree, that is a data quality signal, not a reason to switch backends.

3. **Active selection is heuristic.** The acquisition function is practical but not formally optimal. Do not over-interpret acquisition scores as information-theoretic quantities.

4. **Decision tiers are thresholded.** The tier boundaries (0.7, 0.3, 0.2) are empirically reasonable but not derived from a formal decision theory. They may need recalibration as case sets grow.

5. **Frontier-set governance is advisory.** The `recommended_action` field is a suggestion based on heuristic rules, not a binding protocol. Human judgment overrides in all cases.
