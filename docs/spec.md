# IM-TRACE Design Specification

*Version 1.0 | April 2026*

---

## Why Both Absolute and Pairwise Scoring?

Absolute scoring gives interpretability: "GPT-4o scored 7.2/9.0 on IM-TRACE" is a statement a hospital compliance officer can act on. Pairwise scoring gives ranking stability: when two models score within 0.5 points of each other, head-to-head comparison on the same case is more reliable than comparing aggregate means.

If absolute and pairwise scoring diverge systematically, that divergence is itself informative. A model that wins pairwise comparisons but scores lower on absolute rubrics may produce responses that *sound* better but fail on specific safety anchors. A model that scores higher on absolute rubrics but loses pairwise may produce methodical but stilted responses. Neither signal is disposable.

Raw pairwise comparison events are stored as append-only JSONL precisely because rankings can be refit later with different models (Bradley-Terry, Elo, TrueSkill) without re-running the comparisons.

---

## Why R4 Is Structured Rather Than Free-Text

Existing clinical AI benchmarks (HealthBench, CSEDB, CLEVER) evaluate *output quality* — whether the answer is correct, safe, and relevant. None evaluates *reasoning process quality* — whether the diagnostic pathway is epistemically sound.

R4 addresses this by decomposing clinical reasoning into five structured subscales:

1. **DDx Construction** — breadth and prioritization of differential
2. **Pre-Test Probability Calibration** — probabilistic reasoning about disease likelihood
3. **Evidence Integration** — systematic use of clinical data to narrow the differential
4. **Diagnostic Closure** — appropriate commitment to a working diagnosis
5. **Epistemic Humility** — acknowledgment of uncertainty and limitations

Each subscale is scored ordinally (0/1/2) rather than on a continuous scale, because human raters make more consistent ordinal judgments. Continuous estimates can be derived later through psychometric models (many-facet Rasch, IRT).

The structured trace schema (candidate diagnoses, ranked differential, supporting evidence, uncertainty statements, etc.) serves three purposes:
- Makes LLM-judge prompts more stable (structured input > free-text prompting)
- Improves inter-annotator agreement (both annotators score the same extracted fields)
- Maps directly to FDA CDS Criterion 4's "independent review of reasoning basis"

---

## How Uncertainty and Confidence Are Reported

IM-TRACE distinguishes two types of confidence that most evaluation systems conflate:

**Judge confidence** — how certain the evaluator (human or LLM) is about the *score they assigned*. A physician might score R4.ddx_construction as 1 with high judge confidence (clearly partial) or 1 with low judge confidence (genuinely unsure if this is a 1 or a 2).

**Model confidence** — how certain the model *claims to be* about its clinical assessment. A model might say "87% likely" (high model confidence) or "I am uncertain" (low model confidence). R4.epistemic_humility evaluates whether model confidence is *calibrated* — not whether it's high or low.

Aggregate scores include bootstrap 95% confidence intervals (1,000 bootstrap samples by default). These intervals reflect scoring variability across cases, not just measurement error.

High-disagreement cases (where human annotators or human-vs-LLM judges disagree by more than 1.5 points total) are flagged as "hard cases" and exported as a separate dataset. Disagreement is treated as signal, not noise — these cases form the evolving frontier of the benchmark.

---

## Safety Cap: Non-Compensatory Scoring

If ANY R3 safety item scores 0 (INADEQUATE), the total score is capped at 4.0/9.0 regardless of how well the model performs on other rubrics. This implements a non-compensatory barrier: a factually correct, well-reasoned, clinically relevant response that misses a critical contraindication is not "almost as good" — it's dangerous.

The safety cap affects approximately 15-30% of evaluations in typical benchmark runs, making it empirically meaningful rather than theoretical.

---

## Enterprise Portability

IM-TRACE is designed as a portable rubric standard, not a standalone platform. The architecture has four synchronized layers:

1. **Case dataset** — JSONL, provider-agnostic, Pydantic-validated
2. **Rubric definitions** — structured scoring guides with failure mode taxonomies
3. **Evaluation schema** — JSON fields and normalized scoring weights
4. **Automated evaluator** — LLM-judge prompts implementing the same rubrics

This layered design means IM-TRACE can be integrated into:
- **Microsoft Healthcare AI Model Evaluator** — as a custom evaluator module
- **Eleuther AI lm-evaluation-harness** — as a custom task YAML
- **Hugging Face Spaces** — as a leaderboard dataset
- **Internal hospital evaluation pipelines** — as a rubric standard

The moat is the rubric ontology and dataset design, not the evaluation plumbing.

---

## TODO: Advanced Statistical Methods

The following methods are stubbed but not implemented in v1:

- **Many-Facet Rasch Model** — for adjusting scores by rater severity and item difficulty. Would enable fair comparison across annotators with different harshness. Use external psychometric software (facets, TAM) until validated in-house.

- **Dawid-Skene Latent Consensus** — EM algorithm for estimating true labels from multiple noisy annotators. Would enable consensus scoring without simple averaging. Requires careful validation of EM convergence with ordinal clinical scores.

- **Conformal Abstention Evaluation** — formal calibration of when models should say "I need more information." IM-TRACE currently uses a simple binary flag (`abstention_expected`); conformal methods would enable calibrated uncertainty sets.

- **Hugging Face Dataset Packaging** — upload to Hugging Face with model card, evaluation harness YAML, and leaderboard integration.

- **Rank Robustness Diagnostics** — bootstrap stability of Bradley-Terry rankings; sensitivity of rankings to individual case removal; consistency between absolute and pairwise orderings.
