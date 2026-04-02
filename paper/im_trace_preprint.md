# IM-TRACE: A Four-Rubric Reasoning Trace Audit Benchmark for Internal Medicine LLM Evaluation

**Charles Zhao, MD**

*Independent Clinical AI Safety Evaluation*

---

## Abstract

Large language models are increasingly deployed in clinical decision support, yet 95% of clinical AI benchmarks measure accuracy alone (ARISE 2026). We present IM-TRACE (Internal Medicine — Trace Reasoning Audit for Clinical Evaluation), a four-rubric benchmark that evaluates clinical reasoning *process* quality, not just output correctness. IM-TRACE decomposes evaluation into Factuality (R1, adapted from CLEVER), Clinical Relevance (R2, adapted from CLEVER), Safety-Effectiveness (R3, adapted from CSEDB), and a novel Reasoning Trace Completeness rubric (R4) with five structured subscales: differential diagnosis construction, pre-test probability calibration, evidence integration, diagnostic closure, and epistemic humility. R4 addresses the gap identified by ARISE 2026 — "claim-level grounding and verification of reasoning traces" — and maps directly to the FDA's January 2026 CDS Criterion 4 requirement that healthcare professionals be able to "independently review the basis for the recommendation." The benchmark uses ordinal scoring (0/1/2), non-compensatory safety capping, dual absolute-plus-pairwise evaluation, constraint-based gold standards that preserve clinical reasoning pluralism, and a validated-profile system for reproducible runs. We provide an open-source implementation with 50 passing tests, Pydantic-validated data contracts, pluggable model adapters, Bradley-Terry ranking with Laplace smoothing, bootstrap rank stability diagnostics, and an analysis-only Many-Facet Rasch module for rater severity adjustment. IM-TRACE is designed as a portable clinical reasoning audit standard, not a standalone platform — compatible with existing evaluation ecosystems including the Microsoft Healthcare AI Model Evaluator.

**Keywords:** clinical AI evaluation, reasoning trace, benchmark, internal medicine, patient safety, LLM evaluation, FDA CDS

---

## 1. Introduction

The deployment of large language models (LLMs) in clinical settings has accelerated rapidly. ChatGPT Health processes 230 million weekly health queries (Forbes, Jan 2026). Hippocratic AI has raised $278M for clinical AI agents. Amazon launched Health AI on its consumer platform in March 2026. Yet the evaluation infrastructure for these systems remains overwhelmingly focused on a single dimension: *accuracy*.

The ARISE Network's 2026 State of Clinical AI Report, a Stanford-Harvard multidisciplinary review of over 500 studies, found that "95% of clinical AI benchmarks measure accuracy alone" and called for "claim-level grounding and verification of reasoning traces — measuring support, not fluency — [to] enable increased user trust" [B1]. This finding is corroborated by an August 2025 arXiv analysis concluding that "existing benchmarks have significant gaps in evaluating clinical reasoning processes" [B18].

The consequences of this evaluation gap are clinically significant. A model that produces the correct diagnosis through pattern matching rather than sound clinical reasoning will fail unpredictably on the next ambiguous case. MonitorBench (March 2026) demonstrated that chain-of-thought monitorability *decreases* as model capability increases — the most capable models produce reasoning traces that appear convincing but are less causally connected to their actual outputs [B16]. This means the models most likely to be deployed in clinical settings are precisely the ones whose reasoning is least trustworthy at face value.

Simultaneously, regulatory frameworks are creating compliance-driven demand for reasoning trace evaluation. The FDA's January 2026 Clinical Decision Support Software Final Guidance requires that qualifying non-device CDS enable healthcare professionals to "independently review the basis for the recommendation" (Criterion 4) [B7]. The CHAI/Joint Commission RUAIH framework (September 2025) establishes governance expectations including "local validation" and ongoing "monitoring" [B11]. The EU AI Act, effective August 2026, classifies healthcare AI as high-risk and requires systematic risk identification and misuse evaluation (Article 9) [B13].

IM-TRACE addresses this convergence of scientific need and regulatory demand by providing a structured, physician-governable audit rubric for clinical reasoning traces in internal medicine. Unlike benchmarks that measure staged task performance (MedR-Bench [B5]) or output quality (HealthBench [B2], CSEDB [B3], CLEVER [B4]), IM-TRACE evaluates the *epistemic process* by which a model arrives at its clinical assessment — the quality of the reasoning pathway itself.

### 1.1 Contributions

1. **R4: Reasoning Trace Completeness** — a novel five-subscale rubric for evaluating clinical reasoning process quality, with 25 enumerated failure modes and structured LLM-judge prompts.

2. **Non-compensatory safety scoring** — a safety cap mechanism that prevents high reasoning quality from offsetting catastrophic safety failures.

3. **Dual absolute-plus-pairwise evaluation** — complementary scoring tracks that provide both interpretable benchmarks and stable rankings, with structured disagreement as signal.

4. **Constraint-based gold standards** — case definitions that specify expected reasoning components and forbidden omissions rather than single correct transcripts, preserving the legitimate pluralism of clinical reasoning.

5. **Validated-profile system** — frozen configurations with content hashes for reproducible benchmark runs, enforcing the distinction between exploratory and validated evaluation modes.

6. **Rank stability diagnostics** — bootstrap rank distributions, pairwise superiority matrices, leave-one-case-out sensitivity, and fragility indicators exported as structured data.

7. **Open-source portable implementation** — Pydantic-validated schemas, pluggable adapters, and portability exports designed for integration into existing evaluation ecosystems.

---

## 2. Related Work

### 2.1 First-Generation Benchmarks: Accuracy

MedQA, PubMedQA, MedMCQA, and MMLU medical subsets evaluate LLM performance on multiple-choice questions derived from medical examinations [see B2 review]. The Hugging Face Open Medical-LLM Leaderboard evaluates models exclusively on these accuracy-based metrics. While valuable for measuring factual recall, these benchmarks evaluate *what* a model knows, not *how* it reasons.

### 2.2 Second-Generation Benchmarks: Output Quality

HealthBench (OpenAI, May 2025) evaluates model responses across seven healthcare themes using 48,562 physician-authored criteria applied to 5,000 multi-turn conversations [B2]. CSEDB (Nature Digital Medicine, December 2025) applies 30 indicators (17 safety, 13 effectiveness) across 26 specialties using 32 clinical experts [B3]. CLEVER (JMIR AI, December 2025) evaluates factuality, clinical relevance, and conciseness via randomized blind physician study [B4]. These benchmarks add multi-dimensional output quality evaluation but do not assess the reasoning process that produced the output.

### 2.3 Reasoning Process Evaluation

MedR-Bench (Nature Communications, 2025) presents "the first quantitative analysis focused on LLM reasoning process quality in clinical scenarios" using 1,453 structured patient cases spanning diagnosis and treatment planning [B5]. Q4Dx evaluates interactive diagnostic questioning efficiency under partial information [B6].

IM-TRACE is distinguished from MedR-Bench by four structural differences: (a) it is an *audit rubric* optimized for physician review, not a performance benchmark; (b) it evaluates *static* reasoning traces rather than interactive task completion; (c) it is *specialty-specific* (internal medicine) rather than pan-specialty; and (d) it explicitly maps to *regulatory auditability* requirements (FDA Criterion 4, CHAI RUAIH, EU AI Act Article 9).

### 2.4 Regulatory Context

The FDA's January 2026 CDS guidance softened Criterion 4 transparency requirements to require a "summary of the general approach at a level appropriate for the intended HCP user" [B7, B8] and moved the time-criticality exclusion from Criterion 3 to Criterion 4 [B10]. This creates a specific, compliance-driven demand for evaluation tools that assess whether a clinical AI system's reasoning is sufficiently transparent and auditable — the function IM-TRACE's R4 rubric directly addresses.

---

## 3. The IM-TRACE Framework

### 3.1 Scoring Architecture

IM-TRACE decomposes clinical LLM evaluation into four orthogonal rubrics:

| Rubric | Source | Dimension | Scale |
|--------|--------|-----------|-------|
| R1: Factuality | CLEVER [B4] | Accuracy of medical facts | 0-2 |
| R2: Clinical Relevance | CLEVER [B4] | Applicability to specific scenario | 0-2 |
| R3: Safety-Effectiveness | CSEDB [B3], IM-adapted | 6 safety + 6 effectiveness criteria | 0-2 composite |
| R4: Reasoning Trace Completeness | **Novel** | Quality of diagnostic reasoning pathway | 0-2 composite |

**Total score:** R1 + R2 + (R3 × 1.5) + R4 = max **9.0**

The 1.5× weight on R3 reflects the primacy of safety in clinical AI evaluation, consistent with CSEDB's safety-weighted methodology [B3].

**Non-compensatory safety cap:** If any R3 safety item scores 0 (inadequate), the total score is capped at 4.0/9.0 regardless of other rubric scores. This implements the clinical principle that a factually correct, well-reasoned response that misses a critical contraindication is not "almost as good" — it is dangerous.

### 3.2 R4: Reasoning Trace Completeness

R4 evaluates five subscales that collectively assess the quality of the clinical reasoning pathway:

| Subscale | What It Measures |
|----------|------------------|
| **DDx Construction** | Breadth and prioritization of differential diagnosis |
| **Pre-Test Probability Calibration** | Reasoning about disease likelihood before testing |
| **Evidence Integration** | Systematic use of clinical data to narrow the differential |
| **Diagnostic Closure** | Appropriate commitment to a working diagnosis |
| **Epistemic Humility** | Acknowledgment of uncertainty and limitations |

Each subscale is scored ordinally (0/1/2) with anchored scoring criteria. The composite R4 score is the mean of the five subscale scores.

**Failure mode taxonomy.** Each subscale has an enumerated set of named failure modes (25 total across 5 subscales), such as `premature_closure`, `anchoring_bias`, `base_rate_neglect`, `cherry_picking`, `false_confidence`, and `hallucinated_certainty`. Annotators identify which failure modes are present, providing structured diagnostic information beyond the ordinal score.

**Structured trace schema.** Model responses are parsed into a `ReasoningTraceSchema` with named fields: `candidate_diagnoses`, `ranked_differential`, `confidence_or_probability_language`, `supporting_evidence`, `missing_or_disconfirming_evidence`, `proposed_tests_or_actions`, `working_diagnosis`, `disposition`, `uncertainty_statements`, and `escalation_statements`. This structured extraction reduces scoring ambiguity, improves inter-annotator agreement, and maps directly to FDA Criterion 4's requirement for independently reviewable reasoning.

### 3.3 Ordinal Scoring Rationale

All raw annotation scores use a 3-point ordinal scale (0=inadequate, 1=partial, 2=complete). This design choice is grounded in the psychometric finding that inter-rater agreement degrades as scale granularity increases [B28, B29]. Physicians can reliably distinguish categorical quality levels but struggle with continuous scales. Continuous latent estimates can be derived from ordinal observations through psychometric models (Many-Facet Rasch [B28], IRT) when multi-rater data are available.

### 3.4 Constraint-Based Gold Standards

Clinical cases define gold standards as expected components (`must_consider_diagnoses`, `red_flags`, `acceptable_next_steps`) and forbidden omissions (`unacceptable_errors`) rather than a single correct chain of thought. This preserves the legitimate pluralism of clinical reasoning — two competent physicians may construct different differentials and reach the same correct management through different reasoning paths [B23, B24].

### 3.5 Judge Confidence vs. Model Confidence

Every annotation carries a `judge_confidence` field indicating how certain the *evaluator* is about the score, distinct from any confidence the *model* expresses. This separation is critical: low judge confidence on multiple rubrics signals rubric ambiguity or case difficulty, contributing to hard-case identification (Section 5.3). Model confidence calibration is independently evaluated through R4's epistemic humility subscale.

---

## 4. Evaluation Methodology

### 4.1 Dual Scoring Tracks

IM-TRACE supports absolute scoring (four-rubric composite out of 9.0) and pairwise comparison (blinded A/B with randomized presentation order). These tracks are complementary:

- **Absolute scoring** provides interpretable benchmarks ("Model X scores 7.2/9.0")
- **Pairwise scoring** provides stable rankings when absolute scores are close

When absolute and pairwise rankings diverge systematically, that divergence is clinically informative — a model that wins pairwise but scores lower on absolute rubrics may produce responses that sound better but fail on specific safety anchors.

Raw pairwise events are stored as append-only records with deterministic seeds for reproducibility. Rankings can be refit with different models (Bradley-Terry, Elo, spectral methods) without re-running comparisons.

### 4.2 Bradley-Terry Ranking with Laplace Smoothing

Pairwise results are aggregated using the Bradley-Terry model with Laplace smoothing (0.5 virtual wins per model-pair) to prevent zero-strength degeneracy when a model has zero observed wins. This regularization is equivalent to a symmetric Dirichlet prior on pairwise probabilities [B41]. Strength parameters are converted to Elo-like ratings for interpretability.

### 4.3 Bootstrap Confidence Intervals

Aggregate scores include bootstrap 95% confidence intervals (1,000 samples, deterministic seed). These intervals reflect scoring variability across cases, not measurement error alone.

---

## 5. Calibration and Analysis

### 5.1 Validated vs. Exploratory Modes

IM-TRACE enforces a structural distinction between exploratory runs (iterating on prompts, cases, and analysis techniques) and validated runs (producing locked, reproducible artifacts). Validated runs require a `ValidatedProfile` that pins rubric version, prompt version, case subset, model configurations, and aggregation recipe. Each validated run generates a `RunManifest` with a content hash for audit. This follows the principle that benchmark reproducibility requires explicit separation of "benchmark accuracy" from "generalized accuracy" [B14, B55].

### 5.2 Rank Stability Diagnostics

Rankings are accompanied by structured stability diagnostics:

- **Bootstrap rank distributions** per model (resampled cases)
- **Top-k frequency** (probability of being top-1, top-3)
- **Pairwise superiority probability matrix** P(A > B) for all model pairs
- **Leave-one-case-out sensitivity** (which case removals change the ranking)
- **Fragility summary** with explicit thresholds: top-1 confidence < 0.7, LOO sensitivity > 0.3, and minimum superiority < 0.6 trigger a "fragile" flag

These diagnostics are exported as `rank_stability.json` for downstream consumption, following the principle that rank is not the same as confidence in rank.

### 5.3 Hard-Case Detection

Cases are flagged as "hard" when they reveal evaluation instability — not merely because scores are low. Five criteria, any of which triggers hard-case status:

1. Human vs. LLM judge disagreement (total score difference > 1.5)
2. Low judge confidence on 2+ rubrics
3. Unstable pairwise ordering (case changes which model wins)
4. High ranking leverage (removing case changes top-1 model)
5. Safety-critical flag (annotators disagree on safety item 0 vs. >0)

Hard cases form the evolving frontier of the benchmark — the cases most worth adding to the next validated profile revision.

### 5.4 Many-Facet Rasch Model

An analysis-only MFRM module (JMLE fitting) separates model ability, case difficulty, and rater severity. The implementation emits explicit warnings when data are too sparse (fewer than 3 observations per facet) and skips affected estimates rather than producing unreliable adjusted scores [B28, B29]. MFRM-adjusted scores are reported side-by-side with raw scores but do not replace them — overstated psychometrics are worse than absent psychometrics [B57, B58].

---

## 6. Regulatory Alignment

IM-TRACE maps to active regulatory frameworks as "necessary but not sufficient" for compliance — it provides evaluation coverage, not compliance certification.

| Framework | Requirement | IM-TRACE Coverage |
|-----------|-------------|-------------------|
| FDA CDS 2026 Criterion 4 | HCP can review reasoning basis | R4 (direct mapping) |
| FDA CDS 2026 Criterion 3 | Non-directive, judgment-preserving | R2, R3 |
| CHAI/JC RUAIH | Local validation | R1+R2+R3+R4 |
| CHAI/JC RUAIH | Monitoring | R3+R4 |
| EU AI Act Article 9(a) | Risk identification | R1, R3 |
| EU AI Act Article 9(b) | Misuse risk evaluation | R4 epistemic humility |

The strongest mapping is R4 to FDA Criterion 4: the structured trace schema with candidate diagnoses, ranked differential, supporting evidence, and uncertainty statements maps directly to the Criterion 4 requirement that CDS provide information enabling "independent review of recommendations in a manner that promotes usability" [B7].

---

## 7. Implementation

### 7.1 Architecture

IM-TRACE is implemented as a Python package with Pydantic-validated data contracts, pluggable model adapters (base, mock, replay), and a modular evaluation pipeline. The architecture has four synchronized layers:

1. **Case dataset** — JSONL, provider-agnostic, Pydantic-validated
2. **Rubric definitions** — structured scoring guides with failure mode taxonomies
3. **Evaluation schema** — JSON fields and normalized scoring weights
4. **Automated evaluator** — LLM-judge prompts implementing the same rubrics

### 7.2 Portability

IM-TRACE is designed as a portable rubric standard, not a standalone platform. An export layer maps internal artifacts to external evaluation frameworks (Microsoft Healthcare AI Model Evaluator [B19]) without introducing platform dependencies. The moat is the rubric ontology and dataset design, not the evaluation plumbing.

### 7.3 Test Coverage

50 tests cover schema validation, scoring formula correctness, safety-cap behavior, pairwise randomization invariants, Bradley-Terry convergence, bootstrap confidence intervals, validated-profile reproducibility, rank stability diagnostics, MFRM sparse-data handling, and end-to-end pipeline integrity.

---

## 8. Limitations

1. **Sample size.** The current validated profile contains 4 adversarial IM cases with mock responses. Publication-grade evaluation requires 50+ cases with 2+ physician annotators and 3+ frontier models.

2. **Single-specialty scope.** IM-TRACE is designed for internal medicine. Extension to other specialties requires specialty-specific R3 safety criteria and R4 subscale anchoring.

3. **LLM-judge agreement.** The R4 rubric is designed for physician annotation. Agreement between physician and LLM-judge scoring on R4 specifically has not yet been validated — this is planned as a standalone study.

4. **Static trace evaluation.** IM-TRACE evaluates static reasoning traces, not interactive clinical dialogue. This is a deliberate design choice (see Section 3.2) but limits applicability to real-time clinical AI assistants.

5. **No bias auditing.** IM-TRACE does not evaluate demographic subgroup performance disparities. An R5 bias audit module is a planned extension for EU AI Act Article 9(a) coverage.

---

## 9. Conclusion

IM-TRACE addresses a convergent gap in clinical AI evaluation: the scientific need for reasoning process assessment (ARISE 2026), the regulatory demand for reasoning auditability (FDA CDS Criterion 4), and the practical requirement for reliable model comparison (rank stability diagnostics). By decomposing evaluation into four orthogonal rubrics with a novel reasoning trace completeness dimension, implementing non-compensatory safety scoring, and providing structured rank stability reporting, IM-TRACE offers a methodologically serious evaluation instrument for internal medicine LLMs.

The benchmark is open-source, portable, and designed for integration into existing evaluation ecosystems. It is not intended to be the definitive clinical AI evaluation tool — it is intended to be a rigorous, physician-governable instrument for the specific and currently unaddressed problem of clinical reasoning process quality assessment.

**Code and data:** https://github.com/charleszhao-crypto/im-reasoning-trace-benchmark

---

## References

[B1] ARISE Network. "State of Clinical AI Report 2026." Stanford-Harvard, Jan 2026.

[B2] OpenAI. "HealthBench." May 2025. cdn.openai.com/pdf/healthbench_paper.pdf

[B3] Chen et al. "CSEDB: Clinical Safety-Effectiveness Dual-Track Benchmark." Nature Digital Medicine, Dec 2025.

[B4] [Authors]. "CLEVER: Clinical LLM Evaluation with Rubrics." JMIR AI, Dec 2025.

[B5] [Authors]. "MedR-Bench." Nature Communications, 2025.

[B6] [Authors]. "Q4Dx." 2025.

[B7] FDA. "Clinical Decision Support Software: Final Guidance." Jan 2026.

[B8] Honigman LLP. "FDA CDS 2026 Analysis." Feb 2026.

[B10] Jones Day. "FDA Updates CDS Guidance." Jan 2026.

[B11] CHAI / Joint Commission. "RUAIH." Sep 2025.

[B13] EU AI Act. Article 9. Effective Aug 2026.

[B14] NIST. "AI 800-3." Feb 2026.

[B16] [Authors]. "MonitorBench." Mar 2026.

[B18] [Authors]. "Rethinking Medical Benchmarks." arXiv, Aug 2025.

[B19] Microsoft. "Healthcare AI Model Evaluator." 2026.

[B23] Norman GR. Medical Education, 2005.

[B24] Eva KW. Medical Education, 2005.

[B25] Croskerry P. Academic Medicine, 2009.

[B28] Linacre JM. "Many-Facet Rasch Measurement." 1989.

[B29] Bond TG, Fox CM. "Applying the Rasch Model." 3rd ed., 2015.

[B41] Chatbot Arena / LMSYS. "Arena Rank." 2026.

[B55] NIST. "AI 800-2." Jan 2026.

[B57] WIDA. "Validating Writing Scoring Scale Using MFRM." 2015.

[B58] [Authors]. "Correcting Human Labels for Rater Effects." arXiv, 2026.
