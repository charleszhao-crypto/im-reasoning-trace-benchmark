# IM-TRACE Design Rationale Log

*Append-only record of engineering, scientific, technical, and medical decisions for the arXiv paper.*

---

## v1.0 Decisions (April 1, 2026)

### R1. Four Rubrics Instead of One Composite Score
**Decision:** Decompose into R1 Factuality, R2 Clinical Relevance, R3 Safety-Effectiveness, R4 Reasoning Trace.
**Medical:** Diagnostic errors have distinct failure modes (knowledge, contextual, safety, reasoning) — taught separately in medical education per Croskerry's dual-process theory.
**Scientific:** CLEVER showed factuality/relevance are separable; CSEDB showed safety/effectiveness are separable. R4 adds reasoning process as a third independent axis.
**Engineering:** Decomposed scores enable targeted model improvement — developers can identify *which* dimension to optimize.

### R2. R4 as Novel Contribution — Process vs. Output
**Decision:** Create a dedicated rubric for reasoning *process* quality, not just output correctness.
**Scientific:** ARISE 2026 (Stanford-Harvard, 500+ studies): "95% of clinical AI benchmarks measure accuracy alone." MonitorBench (March 2026): CoT monitorability decreases with model capability — surface fluency is unreliable proxy for reasoning quality.
**Medical:** Process-output distinction is foundational in medical education (Norman 2005, Eva 2005). R4's five subscales map to clinical reasoning competencies: DDx construction, pre-test probability calibration, evidence integration, diagnostic closure, epistemic humility.
**Competitive update (v1.1):** MedR-Bench claims "first reasoning process quality analysis." IM-TRACE differentiates as: audit rubric (not performance benchmark), static trace (not interactive), IM-specific, physician-review-optimized.

### R3. Ordinal Scoring (0/1/2)
**Decision:** 3-point ordinal scale at raw annotation level.
**Scientific:** IAA degrades with scale granularity. Physicians reliably distinguish inadequate/partial/complete but not 67 vs 71.
**Engineering:** Ordinal scores compose cleanly into the 9-point formula without normalization. Continuous estimates derivable later via psychometric models (MFRM, IRT).

### R4. Non-Compensatory Safety Cap (4.0/9.0)
**Decision:** Any R3 safety item scoring 0 caps total at 4.0.
**Medical:** A correct diagnosis with a lethal drug interaction is not "almost as good" — it's dangerous. Mirrors clinical pass/fail behavior in residency competency assessment.
**Engineering:** Without cap, models could score "Excellent" (7.5+) with a catastrophic safety failure. Cap places safety-violated responses in "Marginal" band.
**Empirical:** Affects 2-3 of 4 cases in demo runs — empirically meaningful, not theoretical.

### R5. Constraint-Based Gold Standards
**Decision:** Define gold as expected components + forbidden omissions, NOT single correct chain of thought.
**Medical:** Clinical reasoning is legitimately plural (Norman 2005, Eva 2005). Two competent physicians may reason differently to the same correct management.
**Scientific:** Constraint-based assessment preserves pluralism while penalizing genuine errors. Enables LLM-judge scoring (field-by-field verification vs. transcript matching).

### R6. Judge Confidence ≠ Model Confidence
**Decision:** Every annotation carries `judge_confidence` (evaluator certainty about the score) distinct from model confidence.
**Scientific:** Conflating evaluator uncertainty with model uncertainty is a common evaluation error. Low judge confidence on 2+ rubrics signals rubric ambiguity, not evaluator incompetence.

### R7. Both Absolute and Pairwise Scoring
**Decision:** Support absolute (9-point composite) and pairwise (blinded A/B) as complementary tracks.
**Scientific:** MT-Bench and Chatbot Arena show absolute and pairwise rankings can diverge — that divergence is informative. A model winning pairwise but scoring lower on absolute may sound better but fail on safety anchors.
**Engineering:** Raw pairwise events stored append-only JSONL for refitting with different models (BT, Elo, TrueSkill) without re-running comparisons.

---

## v1.1 Decisions (April 2, 2026)

### R8. Exploratory vs. Validated Modes in Code
**Decision:** `ValidatedProfile` with content hash + `RunManifest` with immutable audit trail.
**Engineering:** Clinical analytics increasingly formalizes exploratory/validated distinction (ICH E9, GxP). IM-TRACE applies this to AI evaluation — validated runs produce locked, comparable artifacts.
**Scientific:** NIST AI 800-3 (Feb 2026) requires separating "benchmark accuracy" from "generalized accuracy."

### R9. Rank Stability as Structured Data
**Decision:** Export `rank_stability.json` with rank histograms, superiority probabilities, LOO, fragility — not just prose.
**Scientific:** "Model A ranks #1" is a point estimate. The distribution (76.3% top-1, P(A>B)=0.729, LOO sensitivity 0.25) is more honest and more useful.
**Fragility thresholds:** top1_confidence < 0.7, LOO_sensitivity > 0.3, min_superiority < 0.6.

### R10. MFRM as Diagnostic Instrument, Not Oracle
**Decision:** Analysis-only MFRM with sparse-data warnings. Adjusted scores reported side-by-side, NOT replacing raw scores.
**Scientific:** MFRM (Linacre 1989) requires ~30 observations per facet for stable estimates (Bond & Fox 2015). v1.1's initial profile has far fewer. Implementation emits warnings and skips when data insufficient.
**Medical:** Same "hawk/dove" rater severity problem as OSCE assessment in medical education.

### R11. Hard Cases Defined by Instability, Not Low Scores
**Decision:** Five criteria: human-LLM disagreement, low judge confidence, unstable pairwise ordering, high ranking leverage, safety-critical flag.
**Medical:** "Sentinel events" in clinical QI are not the most common errors — they reveal systemic weakness. Hard cases serve the same function for benchmarks.

### R12. BT Laplace Smoothing
**Decision:** Add virtual 0.5 wins per model-pair to prevent zero-strength degeneracy.
**Technical:** Equivalent to symmetric Dirichlet prior. Same technique as Chatbot Arena BT implementation. Eliminates -2500 Elo artifacts.

### R13. Provider-Agnostic Telemetry
**Decision:** Optional `input_tokens`, `output_tokens`, `estimated_cost_usd` on ModelResponse.
**Engineering:** Cost-aware benchmarking without entangling with provider APIs. Adapters populate when they know pricing.

### R14. Microsoft Export as Adapter, Not Dependency
**Decision:** Clean mapper with zero MS SDK dependencies.
**Engineering:** Moat is rubric ontology, not evaluation plumbing. If MS API changes, only the exporter needs updating.

---

## v1.2 Planned Decisions (Rationale Pre-Registered)

### R15. Active Pairwise Scheduling (Implemented v1.2)
**Decision:** Heuristic acquisition function scoring uncertainty × leverage × safety × hard-case, with novelty discount and cost penalty.
**Scientific rationale:** PARWiS (2026) [B36] demonstrates spectral ranking + disruptive pair selection under limited budgets. Not all comparisons are equally informative. Spending comparisons near the decision boundary maximizes statistical efficiency — the benchmark equivalent of adaptive trial design [B54].
**Engineering rationale:** The acquisition function decomposes into four interpretable components. Each candidate comparison gets a score. The budget parameter truncates to the top-scoring candidates. This is a heuristic — not a formally optimal Bayesian acquisition function — but it provides substantial improvement over exhaustive or random comparison scheduling.
**Technical detail:** Uncertainty is Shannon entropy of Bernoulli(P(A>B)). Leverage is inversely proportional to normalized score gap. Novelty discounts repeat comparisons. The interface is stable for future upgrade to MI-based criteria.

### R16. Ranking Backend Abstraction (Implemented v1.2)
**Decision:** `RankingBackend` ABC with `RankingResult` dataclass. BT = official default. Spectral = analysis-only alternative.
**Scientific rationale:** SyncRank [B37] and spectral MLE [B38] handle sparse, inconsistent comparisons differently from BT. Agreement between backends = stable signal. Divergence = data sparsity or inconsistency warning — a diagnostic on the benchmark itself, not just on the models.
**Engineering rationale:** The `RankingResult` standardizes: latent scores, Elo ratings, ranked list, pairwise superiority matrix, top-k confidence, and diagnostics. Any backend fills the same contract. Adding new backends (SyncRank, contextual BT [B56]) requires only implementing the `fit()` method.
**Medical analogy:** Multiple diagnostic tests on the same patient. If they agree, confidence increases. If they disagree, investigate the discrepancy before acting.

### R17. Frontier-Set Governance (Implemented v1.2)
**Decision:** Hard cases carry governance fields: `disagreement_score`, `judge_confidence_penalty`, `ranking_leverage`, `safety_weight`, and `recommended_action`.
**Scientific rationale:** Public benchmarks decay as models optimize against them [B39]. A rolling frontier reserve — hard cases refreshed by disagreement and leverage — is the benchmark equivalent of mechanism hardening.
**Medical rationale:** Sentinel event methodology in clinical QI. The most valuable cases for improving the benchmark are not the ones where everyone agrees the model failed — they're the ones where evaluation itself is uncertain or where a single case changes the leaderboard.
**Recommended actions:** `send_to_human` (safety-critical + high disagreement), `needs_rubric_revision` (high judge confidence penalty), `holdout_candidate` (high ranking leverage), `promote_to_frontier_set` (default hard case).

### R18. Decision-Tier Leaderboard (Implemented v1.2)
**Decision:** Assign models to interpretive tiers (leader, co_leader, top_cluster, mid_pack, below_frontier) based on bootstrap rank distributions and pairwise superiority.
**Scientific rationale:** "Rank 2 vs rank 3" is meaningless when superiority probabilities overlap heavily [B39, B53]. Decision tiers align with actual deployment choices — clinicians and procurement teams care about "is this model reliably in the competitive set?" not "is it exactly rank 4?"
**Engineering rationale:** Tier boundaries (P(top-1) > 0.7 for leader, P(top-3) > 0.7 for top_cluster, P(top-3) < 0.2 for below_frontier) are empirically reasonable defaults. They may need recalibration as case sets grow.
**Honest caveat:** These are thresholded heuristics, not derived from formal decision theory. The thresholds are documented and can be overridden.
