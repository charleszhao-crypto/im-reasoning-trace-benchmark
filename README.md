# IM-TRACE

**Internal Medicine — Trace Reasoning Audit for Clinical Evaluation**

A four-rubric benchmark for evaluating clinical reasoning quality in large language models. Scores not just *what* a model concludes, but *how* it reasons.

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Tests](https://img.shields.io/badge/tests-34_passing-green.svg)]()

---

## The Problem

95% of clinical AI benchmarks measure accuracy alone (ARISE 2026, Stanford-Harvard). A model can get the right answer with dangerous reasoning — premature closure, missed differentials, false confidence. Current benchmarks can't distinguish between a correct diagnosis reached through sound clinical logic and one reached through pattern matching that will fail on the next patient.

## The Solution

IM-TRACE evaluates clinical reasoning **process**, not just output:

| Rubric | Source | What It Measures |
|--------|--------|------------------|
| **R1** Factuality | CLEVER | Are the medical facts correct? |
| **R2** Clinical Relevance | CLEVER | Does it address this specific case? |
| **R3** Safety-Effectiveness | CSEDB | Is it safe? Does it follow guidelines? |
| **R4** Reasoning Trace | **Novel** | Is the diagnostic reasoning sound? |

### R4: The Novel Contribution

Five subscales that no prior benchmark evaluates together:

- **DDx Construction** — breadth and prioritization of differential diagnosis
- **Pre-Test Probability** — calibrated reasoning about disease likelihood
- **Evidence Integration** — systematic Bayesian updating with clinical data
- **Diagnostic Closure** — appropriate commitment to a working diagnosis
- **Epistemic Humility** — honest acknowledgment of uncertainty

**Total score:** R1 + R2 + (R3 x 1.5) + R4 = max **9.0**

Safety is non-compensatory: any R3 safety item scoring 0 caps the total at 4.0.

---

## Quick Start

```bash
pip install pydantic pyyaml

# Run the demo benchmark (4 cases, 2 mock models)
python -m im_trace.run_benchmark

# Run tests
python -m unittest im_trace.tests.test_pipeline -v
```

**Output:**
```
LEADERBOARD
Model                 Total    95% CI        R1    R2    R3    R4  Safety
model-beta             4.59    [4.2, 4.9]   1.0   1.0  1.22  0.95  1 viol
model-alpha            4.28    [4.0, 4.8]   1.0   1.0  1.11  1.20  3 viol

PAIRWISE RANKINGS (Bradley-Terry)
  #1 model-beta: Elo 1620
  #2 model-alpha: Elo -2500
```

---

## Architecture

```
im_trace/
├── cases/
│   ├── schema/models.py          # Pydantic data contracts (ClinicalCase, ModelResponse, etc.)
│   ├── raw/sample_cases.jsonl    # 4 IM cases (STEMI, premature closure, abstention, polypharmacy)
│   └── processed/                # Stored model responses (JSONL)
├── rubrics/
│   └── r4_reasoning_trace/
│       └── scoring_guide.py      # R4 subscale guides, failure modes, LLM judge prompts
├── evaluators/
│   ├── absolute/scorer.py        # Four-rubric scorer with safety cap
│   ├── pairwise/comparator.py    # Blinded A/B comparisons with randomization
│   └── aggregation/aggregate.py  # Bootstrap CI, Bradley-Terry, hard-case detection
├── adapters/
│   ├── base.py                   # Abstract adapter interface
│   ├── mock.py                   # Deterministic mock for testing
│   └── replay.py                 # Replay stored responses from JSONL
├── tests/test_pipeline.py        # 34 tests covering schemas, scoring, pairwise, E2E
├── docs/spec.md                  # Design rationale and specification
├── run_benchmark.py              # End-to-end pipeline orchestrator
└── results/runs/                 # Benchmark output (leaderboard JSON, annotations JSONL)
```

---

## Design Principles

**Constraint-based gold standard.** Cases define expected components and forbidden omissions — not a single correct chain of thought. Multiple valid reasoning paths are respected.

**Ordinal scoring.** Raw scores are 0/1/2 (inadequate/partial/complete). Human raters make more consistent ordinal judgments than continuous estimates. Continuous scores can be derived later through psychometric models.

**Judge confidence ≠ model confidence.** Every score carries a judge confidence level (high/medium/low) tracking how certain the *evaluator* is — distinct from how confident the *model claims to be*.

**Dual scoring tracks.** Absolute scoring gives interpretable benchmarks ("7.2/9.0"). Pairwise scoring gives stable rankings. When they diverge, that's signal.

**Replayability.** Stored model responses can be rescored without regeneration. Pairwise events are append-only JSONL so rankings can be refit with different models later.

**Safety is non-compensatory.** A zero on any safety item caps the total at 4.0/9.0. Excellent reasoning cannot offset a missed critical contraindication.

---

## Regulatory Alignment

IM-TRACE maps directly to active regulatory requirements:

| Framework | Criterion | IM-TRACE Rubric |
|-----------|-----------|-----------------|
| **FDA CDS 2026** | Criterion 4: HCP can review reasoning basis | R4 (direct mapping) |
| **FDA CDS 2026** | Criterion 3: non-directive output | R2, R3 |
| **CHAI/JC RUAIH** | Local validation | R1+R2+R3+R4 |
| **EU AI Act Art. 9** | Risk identification | R3 |

IM-TRACE provides evaluation coverage that is **necessary but not sufficient** for regulatory compliance. It does not claim to be a compliance certification tool.

---

## Extending IM-TRACE

**Add a new model:** Write an adapter implementing `BaseAdapter.generate()`, or save responses to JSONL and use `ReplayAdapter`.

**Add cases:** Append to `cases/raw/*.jsonl`. Each case is a `ClinicalCase` Pydantic object with constraint-based gold standards.

**LLM-as-judge:** The R4 scoring guide includes structured LLM judge prompts. Use `scoring_guide.make_r4_judge_prompt()` to generate judge prompts, then validate against physician annotations.

**Enterprise integration:** IM-TRACE is designed as a portable rubric standard. See `docs/spec.md` for integration with Microsoft Healthcare AI Model Evaluator, Eleuther AI harness, and Hugging Face.

---

## Related Work

| Benchmark | What It Measures | R4 Coverage |
|-----------|------------------|-------------|
| HealthBench (OpenAI, 2025) | Output quality across 7 themes | None |
| CSEDB (Nature Digital Medicine, 2025) | Safety + effectiveness (30 criteria) | None |
| CLEVER (JMIR AI, 2025) | Factuality, relevance, conciseness | None |
| MedR-Bench (Nature Communications, 2025) | Reasoning process quality | Partial — staged task performance, not audit rubric |
| MedQA / MultiMedQA | Multiple-choice accuracy | None |
| **IM-TRACE** | **Reasoning process audit** | **5-subscale structured rubric** |

IM-TRACE's positioning: an integrated clinical reasoning trace **audit rubric** for internal medicine, optimized for physician review and regulatory-adjacent auditability. Distinct from benchmarks that measure staged task performance (MedR-Bench) or output quality (HealthBench/CSEDB/CLEVER).

---

## Citation

```bibtex
@misc{zhao2026imtrace,
  title={IM-TRACE: A Four-Rubric Reasoning Trace Benchmark for Internal Medicine LLM Evaluation},
  author={Zhao, Charles},
  year={2026},
  note={Available at https://github.com/charleszhao-crypto/im-reasoning-trace-benchmark}
}
```

---

## Author

**Charles Zhao, MD** — Clinical AI Safety Evaluation

*IM-TRACE is an independent physician-led evaluation benchmark. The author has no commercial relationship with any AI system evaluated using this framework.*
