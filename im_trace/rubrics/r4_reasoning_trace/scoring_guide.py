"""
R4 Reasoning Trace Completeness — Scoring Guides + LLM Judge Prompts.

Each of the 5 subscales has:
  1. A human scoring guide (clinical language)
  2. An LLM judge prompt template (structured extraction)
  3. Failure mode taxonomy

Design principles:
  - Ordinal scoring (0/1/2) for human raters
  - Constraint-based: expected components + forbidden omissions
  - Judge confidence tracked separately from model confidence
  - Failure modes enumerated per subscale
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


# ── Failure Mode Taxonomy ────────────────────────────────────────────────────

FAILURE_MODES = {
    "ddx_construction": [
        "single_diagnosis_only",         # Only one diagnosis considered
        "missing_must_not_miss",         # Life-threatening diagnosis absent from DDx
        "no_prioritization",             # List without ranking
        "anchoring_bias",                # First diagnosis dominates despite evidence for others
        "availability_bias",             # Common diagnoses overrepresented
    ],
    "pretest_probability_calibration": [
        "no_probability_reasoning",      # Conditions listed without likelihood assessment
        "base_rate_neglect",             # Ignores prevalence/epidemiology
        "overconfidence",                # Probability too high given evidence
        "underconfidence",               # Probability too low given classic presentation
        "ignores_demographics",          # Age/sex/race-relevant prevalence ignored
    ],
    "evidence_integration": [
        "cherry_picking",                # Only uses evidence supporting favored diagnosis
        "ignores_negative_findings",     # Doesn't account for absent expected findings
        "no_bayesian_updating",          # Doesn't revise DDx as new info arrives
        "misinterprets_test_result",     # Lab/imaging finding misread
        "correlation_as_causation",      # Temporal association treated as diagnostic
    ],
    "diagnostic_closure": [
        "premature_closure",             # Commits to diagnosis before sufficient evidence
        "indefinite_hedging",            # Never commits despite sufficient evidence
        "no_working_diagnosis",          # DDx without synthesis
        "wrong_final_diagnosis",         # Commits to incorrect diagnosis
        "closure_without_justification", # States diagnosis without explaining why
    ],
    "epistemic_humility": [
        "false_confidence",              # No uncertainty despite ambiguous case
        "no_limitations_acknowledged",   # Doesn't flag what it doesn't know
        "no_referral_consideration",     # Doesn't consider when to escalate
        "hallucinated_certainty",        # Invents precision ("87% likely") without basis
        "ignores_alternative_interpretations",  # Doesn't consider other explanations
    ],
}


# ── Human Scoring Guides ────────────────────────────────────────────────────

HUMAN_SCORING_GUIDES = {
    "ddx_construction": {
        "description": "Breadth and prioritization of differential diagnosis",
        "scores": {
            0: (
                "INADEQUATE: Single diagnosis or absent differential. No consideration of alternatives. "
                "OR: misses a must-not-miss diagnosis that any IM physician would include."
            ),
            1: (
                "PARTIAL: Differential present but incomplete. May miss important conditions, "
                "have poor prioritization, or include irrelevant diagnoses. Some breadth but "
                "not systematically organized by probability."
            ),
            2: (
                "COMPLETE: Comprehensive differential appropriate to the presentation. "
                "Well-prioritized by clinical probability. Includes all must-not-miss diagnoses. "
                "Breadth matches what a competent IM attending would generate."
            ),
        },
        "anchor_questions": [
            "Are ALL must-not-miss diagnoses from the gold standard included?",
            "Is the differential ranked by clinical probability (not alphabetically)?",
            "Does breadth match what you'd expect from an IM attending?",
        ],
    },
    "pretest_probability_calibration": {
        "description": "Calibration of disease likelihood before testing",
        "scores": {
            0: (
                "INADEQUATE: No probability reasoning whatsoever. Diagnoses listed without "
                "any indication of relative likelihood. OR: grossly miscalibrated (claims a rare "
                "condition is 'most likely' without supporting evidence)."
            ),
            1: (
                "PARTIAL: Some implicit probability reasoning ('most likely,' 'less likely') "
                "but without clear calibration to the clinical scenario. May ignore base rates "
                "or demographics."
            ),
            2: (
                "COMPLETE: Explicit, well-calibrated probability reasoning. Accounts for "
                "prevalence, demographics, and risk factors. Language like 'given the patient's "
                "age, sex, and risk profile, the pre-test probability of X is high/moderate/low.' "
                "Calibration consistent with clinical epidemiology."
            ),
        },
        "anchor_questions": [
            "Does the model cite prevalence, base rates, or epidemiological data?",
            "Are probability estimates consistent with the clinical scenario?",
            "Would you call the calibration 'well-calibrated' or 'off'?",
        ],
    },
    "evidence_integration": {
        "description": "How history, exam, and test results narrow the differential",
        "scores": {
            0: (
                "INADEQUATE: Evidence from the case is ignored, misapplied, or not connected "
                "to the differential. The model states a diagnosis without using the clinical data "
                "to support or refute it. OR: misinterprets a key finding."
            ),
            1: (
                "PARTIAL: Some evidence integrated but key findings missed or underweighted. "
                "May use evidence to support the favored diagnosis but ignore evidence that "
                "points elsewhere (cherry-picking)."
            ),
            2: (
                "COMPLETE: Systematic integration. Each piece of evidence (positive AND negative) "
                "is connected to the differential. Bayesian updating: the differential narrows "
                "as evidence accumulates. Key negative findings explicitly noted."
            ),
        },
        "anchor_questions": [
            "Does the model use ALL key findings from the case, not just some?",
            "Are negative findings (things NOT present) explicitly addressed?",
            "Does the differential narrow as evidence is considered (Bayesian updating)?",
        ],
    },
    "diagnostic_closure": {
        "description": "Appropriate commitment to a working diagnosis",
        "scores": {
            0: (
                "INADEQUATE: Premature closure on the wrong diagnosis. OR: no working diagnosis "
                "at all — just an open-ended list with no synthesis. OR: states a diagnosis but "
                "provides no justification for why this one over alternatives."
            ),
            1: (
                "PARTIAL: Reaches a reasonable working diagnosis but under-justified. "
                "The reasoning connecting evidence to conclusion is incomplete. OR: appropriate "
                "hedging but could be more decisive given available evidence."
            ),
            2: (
                "COMPLETE: Well-justified diagnostic conclusion with explicit reasoning chain. "
                "'The most likely diagnosis is X because of findings A, B, C, and the absence "
                "of D and E which would suggest Y.' Appropriate level of commitment for the "
                "available evidence."
            ),
        },
        "anchor_questions": [
            "Is there a clear working diagnosis (not just a DDx list)?",
            "Is the commitment level appropriate for the available evidence?",
            "Is the reasoning from evidence to conclusion explicit?",
        ],
    },
    "epistemic_humility": {
        "description": "Acknowledgment of uncertainty, limitations, and alternatives",
        "scores": {
            0: (
                "INADEQUATE: False confidence. No acknowledgment of uncertainty despite ambiguous "
                "case. Claims certainty where none exists. OR: invents false precision "
                "('87.3% likely') without epidemiological basis."
            ),
            1: (
                "PARTIAL: Some hedging ('possibly,' 'consider') but without structured uncertainty "
                "communication. Doesn't explicitly flag what additional information would change "
                "the assessment."
            ),
            2: (
                "COMPLETE: Explicit uncertainty quantification. Flags what it doesn't know. "
                "States what additional data would change the assessment. Considers when to "
                "escalate or refer. Language like 'I am uncertain about X; obtaining Y would "
                "help distinguish between Z and W.'"
            ),
        },
        "anchor_questions": [
            "Does the model flag what it DOESN'T know?",
            "Does it state what additional information would change its assessment?",
            "Does it consider when to escalate, refer, or ask for help?",
        ],
    },
}


# ── LLM Judge Prompt Templates ──────────────────────────────────────────────

LLM_JUDGE_SYSTEM_PROMPT = """You are a clinical AI evaluation judge. You score model responses using the IM-TRACE R4 Reasoning Trace Completeness rubric.

IMPORTANT: You are scoring the REASONING PROCESS, not just the final answer. A correct final diagnosis with poor reasoning scores lower than an incorrect diagnosis with excellent reasoning.

Score each subscale on a 0-2 ordinal scale:
  0 = INADEQUATE: Major deficiency; clinically unacceptable
  1 = PARTIAL: Partially meets criteria; notable gaps
  2 = COMPLETE: Fully meets criteria; clinically sound

You must also report:
  - rationale: Why you gave this score (2-3 sentences)
  - detected_failure_modes: List of specific failure patterns observed (from the taxonomy)
  - judge_confidence: How certain YOU are about this score (high/medium/low)

Respond ONLY with valid JSON."""


def make_r4_judge_prompt(
    case_text: str,
    response_text: str,
    reasoning_trace: Optional[dict] = None,
) -> str:
    """Generate the LLM judge prompt for R4 scoring."""

    trace_section = ""
    if reasoning_trace:
        trace_section = f"""
## Structured Reasoning Trace (extracted)
```json
{reasoning_trace}
```
"""

    return f"""## Clinical Case
{case_text}

## Model Response Being Evaluated
{response_text}
{trace_section}
## Scoring Task: R4 Reasoning Trace Completeness

Score EACH of the five subscales below. For each, provide:
- score: 0, 1, or 2
- rationale: 2-3 sentence justification
- detected_failure_modes: list of specific failures from the taxonomy below
- judge_confidence: "high", "medium", or "low"

### Subscale 1: DDx Construction
Does the model generate a comprehensive, well-prioritized differential?
Failure modes: single_diagnosis_only, missing_must_not_miss, no_prioritization, anchoring_bias, availability_bias

### Subscale 2: Pre-Test Probability Calibration
Does the model reason about disease likelihood with appropriate calibration?
Failure modes: no_probability_reasoning, base_rate_neglect, overconfidence, underconfidence, ignores_demographics

### Subscale 3: Evidence Integration
Does the model systematically use all clinical evidence to narrow the differential?
Failure modes: cherry_picking, ignores_negative_findings, no_bayesian_updating, misinterprets_test_result, correlation_as_causation

### Subscale 4: Diagnostic Closure
Does the model reach an appropriately justified working diagnosis?
Failure modes: premature_closure, indefinite_hedging, no_working_diagnosis, wrong_final_diagnosis, closure_without_justification

### Subscale 5: Epistemic Humility
Does the model acknowledge uncertainty, limitations, and escalation needs?
Failure modes: false_confidence, no_limitations_acknowledged, no_referral_consideration, hallucinated_certainty, ignores_alternative_interpretations

Return EXACTLY this JSON structure:
{{
  "ddx_construction": {{
    "score": <0|1|2>,
    "rationale": "<2-3 sentences>",
    "detected_failure_modes": ["<mode1>", ...],
    "judge_confidence": "<high|medium|low>"
  }},
  "pretest_probability_calibration": {{
    "score": <0|1|2>,
    "rationale": "<2-3 sentences>",
    "detected_failure_modes": ["<mode1>", ...],
    "judge_confidence": "<high|medium|low>"
  }},
  "evidence_integration": {{
    "score": <0|1|2>,
    "rationale": "<2-3 sentences>",
    "detected_failure_modes": ["<mode1>", ...],
    "judge_confidence": "<high|medium|low>"
  }},
  "diagnostic_closure": {{
    "score": <0|1|2>,
    "rationale": "<2-3 sentences>",
    "detected_failure_modes": ["<mode1>", ...],
    "judge_confidence": "<high|medium|low>"
  }},
  "epistemic_humility": {{
    "score": <0|1|2>,
    "rationale": "<2-3 sentences>",
    "detected_failure_modes": ["<mode1>", ...],
    "judge_confidence": "<high|medium|low>"
  }}
}}"""


# ── Trace Extraction Prompt ─────────────────────────────────────────────────

TRACE_EXTRACTION_SYSTEM_PROMPT = """You are a clinical reasoning trace extractor. Given a model's free-text response to a clinical case, extract the structured reasoning components into the IM-TRACE schema.

Extract what is PRESENT in the response. Do not infer what is absent. If a field has no corresponding content in the response, return an empty list or null.

Respond ONLY with valid JSON."""


def make_trace_extraction_prompt(case_text: str, response_text: str) -> str:
    """Generate prompt to extract structured reasoning trace from free-text."""
    return f"""## Clinical Case
{case_text}

## Model Response
{response_text}

## Extraction Task

Extract the clinical reasoning components from the model response into this exact JSON schema:

{{
  "candidate_diagnoses": ["<list of all diagnoses mentioned>"],
  "ranked_differential": [
    {{"diagnosis": "<name>", "rank": <int>, "reasoning": "<why this rank>"}}
  ],
  "confidence_or_probability_language": ["<any statements about likelihood, probability, or confidence>"],
  "supporting_evidence": ["<clinical findings cited in support of diagnoses>"],
  "missing_or_disconfirming_evidence": ["<findings acknowledged as absent or contradictory>"],
  "proposed_tests_or_actions": ["<recommended labs, imaging, procedures, treatments>"],
  "working_diagnosis": "<final/primary diagnosis if stated, or null>",
  "disposition": "<admit/discharge/ICU/refer/other, or null>",
  "uncertainty_statements": ["<any statements acknowledging uncertainty or limitations>"],
  "escalation_statements": ["<any statements about escalation, referral, or getting help>"]
}}"""
