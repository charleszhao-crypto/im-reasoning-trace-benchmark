"""
IM-TRACE Core Data Schemas — Pydantic models for clinical AI evaluation.

Design principles:
  - Clinical gold reasoning is constraint-based, not transcript-based
  - R4 is ordinal (0/1/2) at raw annotation, mapped to continuous later
  - Judge confidence ≠ model confidence (separate variables)
  - Perturbation families supported for robustness testing
  - Everything versioned by rubric version and prompt version
"""

from __future__ import annotations

from datetime import datetime
from enum import IntEnum, Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────────────

class Specialty(str, Enum):
    CARDIOLOGY = "cardiology"
    PULMONOLOGY = "pulmonology"
    GASTROENTEROLOGY = "gastroenterology"
    NEPHROLOGY = "nephrology"
    ENDOCRINOLOGY = "endocrinology"
    RHEUMATOLOGY = "rheumatology"
    INFECTIOUS_DISEASE = "infectious_disease"
    HEMATOLOGY_ONCOLOGY = "hematology_oncology"
    NEUROLOGY = "neurology"
    GENERAL_IM = "general_im"
    EMERGENCY = "emergency"
    CRITICAL_CARE = "critical_care"


class Difficulty(str, Enum):
    STANDARD = "standard"       # Clear presentation, single primary diagnosis
    COMPLEX = "complex"         # Multi-morbidity, polypharmacy, diagnostic ambiguity
    ADVERSARIAL = "adversarial" # Designed to probe specific failure modes


class AmbiguityClass(str, Enum):
    """Reasoning challenge type — more actionable than specialty alone."""
    CLEAR_CUT = "clear_cut"                   # Obvious diagnosis, test clinical knowledge
    BROAD_DDX = "broad_ddx"                   # Wide differential, test breadth
    SPARSE_DATA = "sparse_data"               # Incomplete history, test reasoning under uncertainty
    CONFLICTING_EVIDENCE = "conflicting_evidence"  # Data points that point in different directions
    FALSE_REASSURANCE = "false_reassurance"    # Looks benign, is dangerous
    MUST_NOT_MISS = "must_not_miss"            # Red flag recognition under time pressure
    PREMATURE_CLOSURE_RISK = "premature_closure"  # First diagnosis is plausible but wrong
    ABSTENTION_APPROPRIATE = "abstention_appropriate"  # Should say "I need more info" or "refer"
    POLYPHARMACY = "polypharmacy"              # Drug interaction / dosing complexity
    ATYPICAL_PRESENTATION = "atypical_presentation"  # Classic disease, unusual presentation


class OrdinalScore(IntEnum):
    """Raw annotation score. Ordinal, not continuous.
    Human raters are better at consistent 0/1/2 than distinguishing 67 from 71.
    Map to continuous via psychometric models later."""
    INADEQUATE = 0   # Major deficiency; clinically unacceptable
    PARTIAL = 1      # Partially meets criteria; notable gaps
    COMPLETE = 2     # Fully meets criteria; clinically sound


class JudgeType(str, Enum):
    PHYSICIAN = "physician"
    LLM_JUDGE = "llm_judge"
    ADJUDICATED = "adjudicated"  # Human-reviewed LLM judgment


class ConfidenceLevel(str, Enum):
    """Judge confidence — how certain the EVALUATOR is about the score.
    Distinct from model confidence (how certain the MODEL claims to be)."""
    HIGH = "high"       # Clear-cut scoring decision
    MEDIUM = "medium"   # Some ambiguity in how to score
    LOW = "low"         # Genuinely uncertain about correct score


# ── Clinical Case ────────────────────────────────────────────────────────────

class ClinicalCase(BaseModel):
    """A clinical vignette designed for LLM evaluation.

    Gold reasoning is constraint-based: expected components, necessary
    considerations, and forbidden omissions — NOT a single "correct" chain
    of thought. This preserves pluralism in legitimate clinical reasoning.
    """
    case_id: str = Field(default_factory=lambda: f"IM-{uuid4().hex[:6].upper()}")
    title: str
    specialty: Specialty
    difficulty: Difficulty
    ambiguity_class: AmbiguityClass
    case_text: str                              # The clinical vignette presented to the model

    # Constraint-based gold standard (not a transcript)
    structured_findings: dict = Field(default_factory=dict)  # Key clinical data points
    gold_facts: list[str] = Field(default_factory=list)      # Facts that MUST be correct in response
    must_consider_diagnoses: list[str] = Field(default_factory=list)  # DDx items that must appear
    red_flags: list[str] = Field(default_factory=list)       # Safety-critical items to recognize
    acceptable_next_steps: list[str] = Field(default_factory=list)   # Valid management options
    unacceptable_errors: list[str] = Field(default_factory=list)     # Errors that score 0 on safety
    abstention_expected: bool = False            # Should the model say "I need more info"?

    # Perturbation support
    parent_case_id: Optional[str] = None         # If this is a perturbation of another case
    perturbation_type: Optional[str] = None      # "removed_key_datum", "added_distractor", "changed_prevalence", "altered_time_pressure"

    metadata: dict = Field(default_factory=dict)
    rubric_version: str = "1.0"
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


# ── Model Response ───────────────────────────────────────────────────────────

class ModelResponse(BaseModel):
    """A model's response to a clinical case, with optional reasoning trace."""
    response_id: str = Field(default_factory=lambda: f"RESP-{uuid4().hex[:8]}")
    case_id: str
    model_id: str                               # e.g., "gpt-4o", "claude-sonnet-4-6"
    model_provider: str                         # "openai", "anthropic", "google", "local"
    response_text: str                          # Raw model output
    reasoning_trace_raw: Optional[str] = None   # CoT if available
    response_time_ms: Optional[int] = None
    prompt_version: str = "1.0"
    temperature: float = 0.3
    max_tokens: int = 4000
    system_prompt_hash: Optional[str] = None    # For reproducibility
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


# ── Reasoning Trace Schema (Structured R4 Input) ────────────────────────────

class ReasoningTraceSchema(BaseModel):
    """Structured extraction of clinical reasoning from a model response.

    This is the PARSED form of the reasoning trace — either extracted
    by a trace extraction layer from free-text, or natively produced by
    models that output structured reasoning.

    Fields map directly to FDA Criterion 4: "independently review the
    basis for the recommendation."
    """
    candidate_diagnoses: list[str] = Field(default_factory=list)
    ranked_differential: list[dict] = Field(default_factory=list)  # [{"diagnosis": str, "rank": int, "reasoning": str}]
    confidence_or_probability_language: list[str] = Field(default_factory=list)  # Explicit probability/confidence statements
    supporting_evidence: list[str] = Field(default_factory=list)   # Findings cited in support
    missing_or_disconfirming_evidence: list[str] = Field(default_factory=list)  # Evidence acknowledged as absent or contradictory
    proposed_tests_or_actions: list[str] = Field(default_factory=list)
    working_diagnosis: Optional[str] = None
    disposition: Optional[str] = None            # admit/discharge/ICU/refer/etc.
    uncertainty_statements: list[str] = Field(default_factory=list)  # "I am uncertain about..."
    escalation_statements: list[str] = Field(default_factory=list)  # "This requires..."

    extraction_method: str = "manual"            # "manual", "llm_extraction", "native_structured"
    extraction_model: Optional[str] = None       # If LLM-extracted, which model
    extraction_prompt_version: Optional[str] = None


# ── R4 Subscale Scores ──────────────────────────────────────────────────────

class R4SubscaleScore(BaseModel):
    """Score for a single R4 subscale dimension.

    Ordinal at raw level (0/1/2). Judge confidence tracked separately
    from model confidence. Detected failure modes are enumerated.
    """
    subscale: str                               # "ddx_construction", "pretest_probability_calibration", etc.
    score: OrdinalScore
    rationale: str                              # Why this score
    detected_failure_modes: list[str] = Field(default_factory=list)  # e.g., "premature_closure", "anchoring_bias"
    judge_confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM  # How certain the JUDGE is


class R4Score(BaseModel):
    """Complete R4 evaluation with all 5 subscales."""
    ddx_construction: R4SubscaleScore
    pretest_probability_calibration: R4SubscaleScore
    evidence_integration: R4SubscaleScore
    diagnostic_closure: R4SubscaleScore
    epistemic_humility: R4SubscaleScore

    @property
    def composite(self) -> float:
        """Mean of 5 subscale scores. Returns 0.0-2.0."""
        scores = [
            self.ddx_construction.score,
            self.pretest_probability_calibration.score,
            self.evidence_integration.score,
            self.diagnostic_closure.score,
            self.epistemic_humility.score,
        ]
        return sum(scores) / len(scores)

    @property
    def all_subscales(self) -> list[R4SubscaleScore]:
        return [
            self.ddx_construction,
            self.pretest_probability_calibration,
            self.evidence_integration,
            self.diagnostic_closure,
            self.epistemic_humility,
        ]


# ── Rubric Scores (R1-R3) ──────────────────────────────────────────────────

class R1Score(BaseModel):
    """R1 Factuality — accuracy of medical facts."""
    score: OrdinalScore
    factual_errors: list[str] = Field(default_factory=list)
    correct_claims_verified: list[str] = Field(default_factory=list)
    guidelines_referenced: list[str] = Field(default_factory=list)
    judge_confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    rationale: str = ""


class R2Score(BaseModel):
    """R2 Clinical Relevance — applicability to specific scenario."""
    score: OrdinalScore
    addresses_chief_complaint: bool = True
    addresses_specific_scenario: bool = True
    actionability: str = "medium"               # "high", "medium", "low"
    judge_confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    rationale: str = ""


class SafetyItemScore(BaseModel):
    criterion: str
    score: OrdinalScore
    notes: str = ""

class EffectivenessItemScore(BaseModel):
    criterion: str
    score: OrdinalScore
    notes: str = ""

class R3Score(BaseModel):
    """R3 Safety-Effectiveness — CSEDB-adapted, safety weighted 2x."""
    safety_items: list[SafetyItemScore]
    effectiveness_items: list[EffectivenessItemScore]
    judge_confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    rationale: str = ""

    @property
    def safety_mean(self) -> float:
        if not self.safety_items:
            return 0.0
        return sum(i.score for i in self.safety_items) / len(self.safety_items)

    @property
    def effectiveness_mean(self) -> float:
        if not self.effectiveness_items:
            return 0.0
        return sum(i.score for i in self.effectiveness_items) / len(self.effectiveness_items)

    @property
    def composite(self) -> float:
        """Safety 2x weighted per CSEDB. Returns 0-2 scale."""
        return (self.safety_mean * 2 + self.effectiveness_mean) / 3


# ── Annotation Records ──────────────────────────────────────────────────────

class HumanAnnotation(BaseModel):
    """A physician's complete scoring of a model response."""
    annotation_id: str = Field(default_factory=lambda: f"ANN-{uuid4().hex[:8]}")
    case_id: str
    response_id: str
    model_id: str
    evaluator_id: str
    judge_type: JudgeType = JudgeType.PHYSICIAN

    r1: R1Score
    r2: R2Score
    r3: R3Score
    r4: R4Score

    reasoning_trace: Optional[ReasoningTraceSchema] = None  # Structured extraction used for R4

    overall_notes: str = ""
    annotation_duration_minutes: Optional[float] = None

    rubric_version: str = "1.0"
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    @property
    def total_score(self) -> float:
        """R1 + R2 + (R3 * 1.5) + R4. Max 9.0."""
        return float(self.r1.score) + float(self.r2.score) + (self.r3.composite * 1.5) + self.r4.composite


class LLMJudgeAnnotation(BaseModel):
    """An LLM judge's scoring of a model response — same schema as human."""
    annotation_id: str = Field(default_factory=lambda: f"LLM-{uuid4().hex[:8]}")
    case_id: str
    response_id: str
    model_id: str                               # Model being evaluated
    judge_model_id: str                         # Model doing the judging
    judge_type: JudgeType = JudgeType.LLM_JUDGE

    r1: R1Score
    r2: R2Score
    r3: R3Score
    r4: R4Score

    reasoning_trace: Optional[ReasoningTraceSchema] = None
    judge_prompt_version: str = "1.0"
    judge_raw_output: Optional[str] = None       # Full judge response for audit

    rubric_version: str = "1.0"
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    @property
    def total_score(self) -> float:
        return float(self.r1.score) + float(self.r2.score) + (self.r3.composite * 1.5) + self.r4.composite


# ── Pairwise Comparison ─────────────────────────────────────────────────────

class RubricPreference(BaseModel):
    """A single rubric-level or subscale-level pairwise preference."""
    rubric: str                                 # "r1", "r2", "r3", "r4", "overall"
    subscale: Optional[str] = None              # e.g., "ddx_construction" for R4 subscales
    winner: str                                 # "a", "b", "tie"
    rationale: str = ""
    judge_confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM


class PairwiseComparisonRecord(BaseModel):
    """A/B comparison of two model responses on the same case.

    Response order is randomized. Provider/model names are blinded
    where possible. Pairwise and absolute scoring are complementary:
    absolute for interpretability, pairwise for ranking stability.
    If they diverge systematically, that divergence is informative.
    """
    comparison_id: str = Field(default_factory=lambda: f"CMP-{uuid4().hex[:8]}")
    case_id: str
    response_a_id: str
    response_b_id: str
    model_a_id: str                             # Revealed only in results, not during judging
    model_b_id: str
    presented_order: str                        # "a_first" or "b_first" (randomized)
    seed: int = 42                              # Deterministic seed used for randomization

    preferences: list[RubricPreference]         # Per-rubric and per-subscale preferences

    judge_type: JudgeType
    judge_id: str                               # evaluator_id or judge_model_id

    rubric_version: str = "1.0"
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    @property
    def overall_winner(self) -> str:
        for p in self.preferences:
            if p.rubric == "overall":
                return p.winner
        return "tie"


# ── Aggregate Scores ────────────────────────────────────────────────────────

class SubscaleProfile(BaseModel):
    """Per-subscale aggregate for a model across cases."""
    subscale: str
    mean: float
    ci_lower: float                             # Bootstrap 95% CI
    ci_upper: float
    n_cases: int


class AggregateCaseScore(BaseModel):
    """Aggregate score for one model on one case (multi-annotator)."""
    case_id: str
    model_id: str
    n_annotations: int

    r1_mean: float
    r2_mean: float
    r3_mean: float
    r4_mean: float
    total_mean: float

    r4_profile: list[SubscaleProfile]           # Per-subscale breakdown

    human_llm_agreement: Optional[float] = None  # Krippendorff's alpha if both exist
    is_hard_case: bool = False                   # Flagged by disagreement analysis
    hard_case_reason: Optional[str] = None


class BenchmarkRunSummary(BaseModel):
    """Summary of a complete benchmark run across models and cases."""
    run_id: str = Field(default_factory=lambda: f"RUN-{uuid4().hex[:8]}")
    run_date: str = Field(default_factory=lambda: datetime.now().isoformat())
    rubric_version: str
    prompt_version: str
    n_cases: int
    n_models: int
    models_evaluated: list[str]
    n_human_annotations: int
    n_llm_annotations: int

    # Per-model summary
    model_scores: list[dict]                    # [{model_id, total_mean, total_ci_lower, total_ci_upper, r1-r4 means, r4 profile}]

    # Cross-model pairwise rankings (Bradley-Terry)
    pairwise_rankings: Optional[list[dict]] = None  # [{model_id, elo_rating, bt_score, rank}]

    # Disagreement analysis
    n_hard_cases: int = 0
    human_llm_overall_agreement: Optional[float] = None

    # Safety cap analysis
    models_with_safety_violations: list[str] = Field(default_factory=list)  # Models scoring 0 on any R3 safety item

    metadata: dict = Field(default_factory=dict)
