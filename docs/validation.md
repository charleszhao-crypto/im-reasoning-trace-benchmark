# Exploratory vs. Validated Modes

*IM-TRACE v1.1 Calibration Architecture*

---

## Why Two Modes?

Clinical AI evaluation must balance innovation speed with measurement fidelity. Exploratory runs iterate quickly on prompts, case design, and analysis techniques. Validated runs produce locked, reproducible artifacts suitable for publication, regulatory reference, or cross-institutional comparison.

These modes are not just documentation language — they are enforced in code via `ValidatedProfile` and `RunManifest`.

---

## Exploratory Mode

**Use for:** developing new cases, testing prompt variants, prototyping analysis modules, calibrating rubrics.

- Uses `run_benchmark.py` directly
- No frozen profile required
- Output directory may be overwritten
- No manifest generated
- Analysis modules run freely

**Artifacts are marked `mode: "exploratory"` and should NOT be cited as validated results.**

---

## Validated Mode

**Use for:** publication-grade results, regulatory reference, cross-institutional comparison, leaderboard updates.

- Requires a `ValidatedProfile` (frozen config)
- Profile pins: rubric version, prompt version, case subset, model configs, aggregation recipe
- Output directory is append-only (new `VAL-*` directory per run)
- Generates a signed `RunManifest` with content hash
- All outputs traceable back to the profile

**A validated run that cannot be reproduced from its manifest is a failed run.**

---

## ValidatedProfile Structure

```python
ValidatedProfile(
    profile_id="imtrace-v1-core4",
    profile_version="1.0",
    rubric_version="1.0",
    prompt_version="1.0",
    case_ids=["IM-STEMI", "IM-CLOSURE", "IM-ABSTAIN", "IM-SAFETY"],
    adapter_configs=[
        AdapterConfig(model_id="model-alpha", provider="mock"),
        AdapterConfig(model_id="model-beta", provider="mock"),
    ],
    aggregation=AggregationRecipe(
        bootstrap_n=1000,
        bootstrap_seed=42,
        safety_cap=4.0,
    ),
    mode="validated",
)
```

The profile's `content_hash` is a SHA-256 digest of all scoring-relevant parameters. If any parameter changes, the hash changes, and the manifest will reflect a different profile version.

---

## RunManifest

Generated at run completion. Contains:

- `manifest_id`: unique run identifier
- `profile_id` + `profile_content_hash`: which profile produced this run
- `output_files`: paths to all generated artifacts
- `warnings`: any issues encountered during the run
- `duration_seconds`: wall-clock time

The manifest is the audit trail. It answers: "what exact configuration produced these results?"

---

## Rank Stability Reporting

Rank stability diagnostics are produced for both modes but carry different weight:

- **Exploratory:** informational — helps decide if more cases or annotators are needed
- **Validated:** authoritative — included in publication and leaderboard

The `rank_stability.json` artifact contains:
- Per-model bootstrap rank distributions
- Top-1 frequency (P[model is #1])
- Pairwise superiority probability matrix
- Leave-one-case-out sensitivity
- Fragility summary with explicit thresholds

**Fragility thresholds:**
- `top1_confidence < 0.7` → rankings are unstable
- `loo_sensitivity > 0.3` → rankings depend on specific cases
- `min_superiority < 0.6` → top model's lead is not robust

---

## Many-Facet Rasch Model (MFRM)

MFRM is an **analysis instrument**, not a production scorer. It answers:
- How severe is each rater?
- How difficult is each case?
- What is each model's latent ability after adjusting for rater and case effects?

**When to use:** when 2+ raters have scored overlapping cases.

**When NOT to use:** with sparse data (< 3 observations per facet). The implementation emits warnings and skips estimates when data are insufficient.

**Output:** side-by-side raw vs. adjusted scores. The adjusted scores are informational — they do not replace raw scores in the leaderboard unless explicitly approved.

---

## Hard Cases

A case enters the hard set when it reveals **instability, disagreement, or safety-critical weakness** — not merely because scores are low. Five criteria:

1. Human vs. LLM judge disagreement (total > 1.5 points)
2. Low judge confidence on 2+ rubrics
3. Unstable pairwise ordering (case changes which model wins)
4. High ranking leverage (removing case changes top-1)
5. Safety-critical flag (annotators disagree on safety item 0 vs. >0)

Hard cases are exported as `hard_cases.jsonl` and `hard_case_summary.md`. They form the evolving frontier of the benchmark — the cases most worth adding to the next validated profile revision.

---

## Portability

IM-TRACE exports are decoupled from any specific external platform. The `exporters/` module maps internal artifacts to external schemas:

- `microsoft_eval.py`: maps to Microsoft Healthcare AI Model Evaluator format
- Future: `huggingface.py`, `eleuther_harness.py`

These exporters consume IM-TRACE outputs and emit provider-specific JSON. They never modify internal artifacts or depend on external libraries.
