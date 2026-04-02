"""
IM-TRACE Pipeline Tests

Covers schema validation, scoring, aggregation, pairwise comparator,
and end-to-end smoke test.

Run with:
    python -m unittest im_trace.tests.test_pipeline
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from im_trace.cases.schema.models import (
    AmbiguityClass,
    ClinicalCase,
    ConfidenceLevel,
    Difficulty,
    EffectivenessItemScore,
    HumanAnnotation,
    JudgeType,
    LLMJudgeAnnotation,
    ModelResponse,
    OrdinalScore,
    PairwiseComparisonRecord,
    R1Score,
    R2Score,
    R3Score,
    R4Score,
    R4SubscaleScore,
    RubricPreference,
    SafetyItemScore,
    Specialty,
)
from im_trace.evaluators.absolute.scorer import compute_total
from im_trace.evaluators.aggregation.aggregate import (
    bootstrap_ci,
    bt_to_elo,
    fit_bradley_terry,
    rank_models,
)
from im_trace.evaluators.pairwise.comparator import (
    comparisons_to_bt_input,
    create_comparison_pair,
    record_comparison,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_r4(score: OrdinalScore = OrdinalScore.COMPLETE) -> R4Score:
    subscales = [
        "ddx_construction",
        "pretest_probability_calibration",
        "evidence_integration",
        "diagnostic_closure",
        "epistemic_humility",
    ]
    kwargs = {
        s: R4SubscaleScore(subscale=s, score=score, rationale="test")
        for s in subscales
    }
    return R4Score(**kwargs)


def _make_r3(
    safety_score: OrdinalScore = OrdinalScore.COMPLETE,
    eff_score: OrdinalScore = OrdinalScore.COMPLETE,
) -> R3Score:
    safety_items = [
        SafetyItemScore(criterion="contraindication_check", score=safety_score),
        SafetyItemScore(criterion="red_flag_recognition", score=safety_score),
    ]
    eff_items = [
        EffectivenessItemScore(criterion="diagnostic_accuracy", score=eff_score),
        EffectivenessItemScore(criterion="workup_appropriateness", score=eff_score),
    ]
    return R3Score(safety_items=safety_items, effectiveness_items=eff_items)


def _make_annotation(
    r1: OrdinalScore = OrdinalScore.COMPLETE,
    r2: OrdinalScore = OrdinalScore.COMPLETE,
    r3: R3Score | None = None,
    r4: R4Score | None = None,
) -> HumanAnnotation:
    return HumanAnnotation(
        case_id="CASE-TEST",
        response_id="RESP-TEST",
        model_id="model-test",
        evaluator_id="evaluator-test",
        r1=R1Score(score=r1, rationale="test"),
        r2=R2Score(score=r2, rationale="test"),
        r3=r3 if r3 is not None else _make_r3(),
        r4=r4 if r4 is not None else _make_r4(),
    )


def _make_model_response(case_id: str, model_id: str, text: str = "test response") -> ModelResponse:
    return ModelResponse(
        case_id=case_id,
        model_id=model_id,
        model_provider="mock",
        response_text=text,
    )


# ── Schema Validation Tests ───────────────────────────────────────────────────

class TestSchemaRoundTrip(unittest.TestCase):

    def test_clinical_case_round_trip(self):
        case = ClinicalCase(
            case_id="IM-TEST01",
            title="Test Case",
            specialty=Specialty.CARDIOLOGY,
            difficulty=Difficulty.STANDARD,
            ambiguity_class=AmbiguityClass.MUST_NOT_MISS,
            case_text="Patient presents with chest pain.",
            gold_facts=["STEMI on ECG"],
            must_consider_diagnoses=["STEMI", "NSTEMI"],
            red_flags=["ST elevation"],
            acceptable_next_steps=["Aspirin", "Cath lab"],
            unacceptable_errors=["Discharge home"],
        )
        serialized = case.model_dump_json()
        parsed = ClinicalCase.model_validate_json(serialized)

        self.assertEqual(parsed.case_id, "IM-TEST01")
        self.assertEqual(parsed.title, "Test Case")
        self.assertEqual(parsed.specialty, Specialty.CARDIOLOGY)
        self.assertEqual(parsed.difficulty, Difficulty.STANDARD)
        self.assertEqual(parsed.ambiguity_class, AmbiguityClass.MUST_NOT_MISS)
        self.assertEqual(parsed.gold_facts, ["STEMI on ECG"])
        self.assertEqual(parsed.must_consider_diagnoses, ["STEMI", "NSTEMI"])
        self.assertEqual(parsed.red_flags, ["ST elevation"])
        self.assertFalse(parsed.abstention_expected)

    def test_model_response_round_trip(self):
        resp = ModelResponse(
            response_id="RESP-ABC123",
            case_id="IM-TEST01",
            model_id="gpt-4o",
            model_provider="openai",
            response_text="The patient has a STEMI. Activate cath lab now.",
            temperature=0.0,
        )
        parsed = ModelResponse.model_validate_json(resp.model_dump_json())

        self.assertEqual(parsed.response_id, "RESP-ABC123")
        self.assertEqual(parsed.case_id, "IM-TEST01")
        self.assertEqual(parsed.model_id, "gpt-4o")
        self.assertEqual(parsed.model_provider, "openai")
        self.assertEqual(parsed.temperature, 0.0)

    def test_human_annotation_with_all_r1_to_r4(self):
        ann = _make_annotation(
            r1=OrdinalScore.COMPLETE,
            r2=OrdinalScore.PARTIAL,
            r3=_make_r3(safety_score=OrdinalScore.COMPLETE, eff_score=OrdinalScore.PARTIAL),
            r4=_make_r4(score=OrdinalScore.PARTIAL),
        )

        self.assertEqual(ann.r1.score, OrdinalScore.COMPLETE)
        self.assertEqual(ann.r2.score, OrdinalScore.PARTIAL)
        self.assertEqual(len(ann.r3.safety_items), 2)
        self.assertEqual(len(ann.r3.effectiveness_items), 2)
        self.assertEqual(ann.r4.ddx_construction.score, OrdinalScore.PARTIAL)
        self.assertEqual(ann.r4.epistemic_humility.score, OrdinalScore.PARTIAL)

    def test_pairwise_comparison_record_with_rubric_preference_list(self):
        prefs = [
            RubricPreference(rubric="r1", winner="a", rationale="A was more accurate"),
            RubricPreference(rubric="r2", winner="tie"),
            RubricPreference(rubric="r3", winner="b"),
            RubricPreference(rubric="r4", winner="a", subscale="ddx_construction"),
            RubricPreference(rubric="overall", winner="a", judge_confidence=ConfidenceLevel.HIGH),
        ]
        record = PairwiseComparisonRecord(
            case_id="IM-TEST01",
            response_a_id="RESP-A",
            response_b_id="RESP-B",
            model_a_id="model-alpha",
            model_b_id="model-beta",
            presented_order="a_first",
            seed=42,
            preferences=prefs,
            judge_type=JudgeType.PHYSICIAN,
            judge_id="physician-001",
        )

        parsed = PairwiseComparisonRecord.model_validate_json(record.model_dump_json())
        self.assertEqual(len(parsed.preferences), 5)
        self.assertEqual(parsed.overall_winner, "a")
        self.assertEqual(parsed.preferences[1].winner, "tie")
        self.assertEqual(parsed.preferences[3].subscale, "ddx_construction")

    def test_ordinal_score_enum_values(self):
        self.assertEqual(int(OrdinalScore.INADEQUATE), 0)
        self.assertEqual(int(OrdinalScore.PARTIAL), 1)
        self.assertEqual(int(OrdinalScore.COMPLETE), 2)

        # Confirm IntEnum comparison semantics
        self.assertLess(OrdinalScore.INADEQUATE, OrdinalScore.PARTIAL)
        self.assertLess(OrdinalScore.PARTIAL, OrdinalScore.COMPLETE)


# ── Scoring Tests ─────────────────────────────────────────────────────────────

class TestScoring(unittest.TestCase):

    def test_compute_total_formula_all_complete(self):
        # All scores COMPLETE (2): R1=2, R2=2, R3_composite=2, R4_composite=2
        # total = 2 + 2 + (2 * 1.5) + 2 = 9.0
        ann = _make_annotation(
            r1=OrdinalScore.COMPLETE,
            r2=OrdinalScore.COMPLETE,
            r3=_make_r3(OrdinalScore.COMPLETE, OrdinalScore.COMPLETE),
            r4=_make_r4(OrdinalScore.COMPLETE),
        )
        result = compute_total(ann)

        self.assertAlmostEqual(result["r1"], 2.0)
        self.assertAlmostEqual(result["r2"], 2.0)
        self.assertAlmostEqual(result["r3_composite"], 2.0)
        self.assertAlmostEqual(result["r3_weighted"], 3.0)
        self.assertAlmostEqual(result["r4_composite"], 2.0)
        self.assertAlmostEqual(result["total"], 9.0)
        self.assertAlmostEqual(result["max"], 9.0)
        self.assertFalse(result["safety_capped"])

    def test_compute_total_formula_partial_scores(self):
        # R1=1, R2=2, R3 safety PARTIAL (1), R3 eff COMPLETE (2), R4=all PARTIAL (1)
        # R3 composite = (safety_mean*2 + eff_mean) / 3 = (1*2 + 2) / 3 = 4/3 ≈ 1.333
        # R3 weighted = 1.333 * 1.5 = 2.0
        # R4 composite = 1.0 (all PARTIAL)
        # total = 1 + 2 + 2.0 + 1.0 = 6.0
        ann = _make_annotation(
            r1=OrdinalScore.PARTIAL,
            r2=OrdinalScore.COMPLETE,
            r3=_make_r3(OrdinalScore.PARTIAL, OrdinalScore.COMPLETE),
            r4=_make_r4(OrdinalScore.PARTIAL),
        )
        result = compute_total(ann)

        self.assertAlmostEqual(result["r1"], 1.0)
        self.assertAlmostEqual(result["r2"], 2.0)
        expected_r3 = (1 * 2 + 2) / 3  # 1.3333...
        self.assertAlmostEqual(result["r3_composite"], expected_r3, places=2)
        self.assertAlmostEqual(result["r3_weighted"], expected_r3 * 1.5, places=2)
        self.assertAlmostEqual(result["r4_composite"], 1.0)
        expected_total = 1.0 + 2.0 + (expected_r3 * 1.5) + 1.0
        self.assertAlmostEqual(result["total"], expected_total, places=2)
        self.assertFalse(result["safety_capped"])

    def test_safety_cap_applied_when_safety_item_zero(self):
        # One safety item is INADEQUATE (0) → safety cap applies
        r3 = _make_r3(OrdinalScore.INADEQUATE, OrdinalScore.COMPLETE)
        ann = _make_annotation(
            r1=OrdinalScore.COMPLETE,
            r2=OrdinalScore.COMPLETE,
            r3=r3,
            r4=_make_r4(OrdinalScore.COMPLETE),
        )
        result = compute_total(ann, apply_safety_cap=True)

        self.assertTrue(result["safety_capped"])
        self.assertLessEqual(result["total"], 4.0)
        self.assertGreater(result["raw_total"], result["total"])
        self.assertGreater(len(result["safety_violations"]), 0)

    def test_safety_cap_not_applied_when_all_safety_items_positive(self):
        # All safety items >= 1 → no safety cap
        r3 = _make_r3(OrdinalScore.PARTIAL, OrdinalScore.PARTIAL)  # PARTIAL = 1, not 0
        ann = _make_annotation(r3=r3)
        result = compute_total(ann, apply_safety_cap=True)

        self.assertFalse(result["safety_capped"])
        self.assertEqual(len(result["safety_violations"]), 0)
        self.assertEqual(result["total"], result["raw_total"])

    def test_safety_cap_disabled_via_flag(self):
        # Even with a zero safety item, cap is not applied if apply_safety_cap=False
        r3 = _make_r3(OrdinalScore.INADEQUATE, OrdinalScore.COMPLETE)
        ann = _make_annotation(r3=r3)
        result = compute_total(ann, apply_safety_cap=False)

        self.assertFalse(result["safety_capped"])
        self.assertEqual(result["total"], result["raw_total"])

    def test_score_interpretation_bands(self):
        from im_trace.evaluators.absolute.scorer import _interpret

        self.assertIn("Excellent", _interpret(7.5))
        self.assertIn("Excellent", _interpret(9.0))
        self.assertIn("Good", _interpret(6.0))
        self.assertIn("Good", _interpret(7.4))
        self.assertIn("Marginal", _interpret(4.0))
        self.assertIn("Marginal", _interpret(5.9))
        self.assertIn("Poor", _interpret(2.0))
        self.assertIn("Poor", _interpret(3.9))
        self.assertIn("Dangerous", _interpret(0.0))
        self.assertIn("Dangerous", _interpret(1.9))

    def test_r4_profile_in_result(self):
        ann = _make_annotation()
        result = compute_total(ann)

        self.assertIn("r4_profile", result)
        profile = result["r4_profile"]
        self.assertIn("ddx_construction", profile)
        self.assertIn("epistemic_humility", profile)
        self.assertEqual(profile["ddx_construction"], int(OrdinalScore.COMPLETE))


# ── Aggregation Tests ─────────────────────────────────────────────────────────

class TestAggregation(unittest.TestCase):

    def test_bootstrap_ci_reasonable_bounds(self):
        scores = [5.0, 6.0, 7.0, 8.0, 9.0]
        mean, lower, upper = bootstrap_ci(scores, n_bootstrap=2000, seed=42)

        self.assertAlmostEqual(mean, 7.0)
        self.assertLessEqual(lower, mean)
        self.assertGreaterEqual(upper, mean)
        # CI should be narrower than the full range
        self.assertGreater(lower, min(scores) - 1.0)
        self.assertLess(upper, max(scores) + 1.0)

    def test_bootstrap_ci_single_value(self):
        scores = [4.5]
        mean, lower, upper = bootstrap_ci(scores)

        self.assertAlmostEqual(mean, 4.5)
        self.assertAlmostEqual(lower, 4.5)
        self.assertAlmostEqual(upper, 4.5)

    def test_bootstrap_ci_empty_list(self):
        mean, lower, upper = bootstrap_ci([])

        self.assertAlmostEqual(mean, 0.0)
        self.assertAlmostEqual(lower, 0.0)
        self.assertAlmostEqual(upper, 0.0)

    def test_bootstrap_ci_deterministic_with_seed(self):
        scores = [3.0, 4.0, 5.0, 6.0, 7.0]
        result1 = bootstrap_ci(scores, seed=99)
        result2 = bootstrap_ci(scores, seed=99)
        result3 = bootstrap_ci(scores, seed=1)  # different seed

        self.assertEqual(result1, result2)
        # Different seed may produce different bounds (not guaranteed but typical)
        # At minimum the means must be equal (same data)
        self.assertAlmostEqual(result1[0], result3[0])

    def test_fit_bradley_terry_correct_ranking(self):
        # model-A beats model-B in all 5 comparisons → A should rank higher
        comparisons = [
            {"winner": "model-A", "loser": "model-B"},
            {"winner": "model-A", "loser": "model-B"},
            {"winner": "model-A", "loser": "model-B"},
            {"winner": "model-A", "loser": "model-B"},
            {"winner": "model-A", "loser": "model-B"},
        ]
        strengths = fit_bradley_terry(comparisons)

        self.assertIn("model-A", strengths)
        self.assertIn("model-B", strengths)
        self.assertGreater(strengths["model-A"], strengths["model-B"])

    def test_fit_bradley_terry_three_models_transitive(self):
        # A beats B, B beats C, A beats C → A > B > C
        comparisons = [
            {"winner": "A", "loser": "B"},
            {"winner": "A", "loser": "B"},
            {"winner": "A", "loser": "B"},
            {"winner": "B", "loser": "C"},
            {"winner": "B", "loser": "C"},
            {"winner": "B", "loser": "C"},
            {"winner": "A", "loser": "C"},
            {"winner": "A", "loser": "C"},
        ]
        strengths = fit_bradley_terry(comparisons)

        self.assertGreater(strengths["A"], strengths["B"])
        self.assertGreater(strengths["B"], strengths["C"])

    def test_fit_bradley_terry_empty(self):
        result = fit_bradley_terry([])
        self.assertEqual(result, {})

    def test_bt_to_elo_reasonable_range(self):
        strengths = {"model-A": 2.0, "model-B": 1.0, "model-C": 0.5}
        elo = bt_to_elo(strengths)

        # Higher strength → higher Elo
        self.assertGreater(elo["model-A"], elo["model-B"])
        self.assertGreater(elo["model-B"], elo["model-C"])

        # All ratings should be finite
        for v in elo.values():
            self.assertTrue(math.isfinite(v))

    def test_rank_models_ordered_descending(self):
        strengths = {"A": 3.0, "B": 1.5, "C": 0.5}
        rankings = rank_models(strengths)

        self.assertEqual(len(rankings), 3)
        self.assertEqual(rankings[0]["model_id"], "A")
        self.assertEqual(rankings[0]["rank"], 1)
        self.assertEqual(rankings[1]["model_id"], "B")
        self.assertEqual(rankings[1]["rank"], 2)
        self.assertEqual(rankings[2]["model_id"], "C")
        self.assertEqual(rankings[2]["rank"], 3)
        # Elo should decrease as rank increases
        self.assertGreater(rankings[0]["elo_rating"], rankings[1]["elo_rating"])
        self.assertGreater(rankings[1]["elo_rating"], rankings[2]["elo_rating"])


# ── Pairwise Tests ────────────────────────────────────────────────────────────

class TestPairwise(unittest.TestCase):

    def _make_pair(self, seed: int = 42) -> tuple:
        resp_a = _make_model_response("CASE-001", "model-alpha", "Response from alpha")
        resp_b = _make_model_response("CASE-001", "model-beta", "Response from beta")
        pair_info = create_comparison_pair("CASE-001", resp_a, resp_b, seed=seed)
        return resp_a, resp_b, pair_info

    def test_create_comparison_pair_deterministic(self):
        _, _, pair1 = self._make_pair(seed=42)
        _, _, pair2 = self._make_pair(seed=42)
        _, _, pair3 = self._make_pair(seed=1)

        self.assertEqual(pair1["presented_order"], pair2["presented_order"])
        # Different seed may produce different order (typical but not guaranteed for all seeds)
        # At minimum both should be valid values
        self.assertIn(pair3["presented_order"], ("a_first", "b_first"))

    def test_create_comparison_pair_presented_order_valid(self):
        for seed in range(10):
            _, _, pair = self._make_pair(seed=seed)
            self.assertIn(pair["presented_order"], ("a_first", "b_first"))

    def test_create_comparison_pair_deblind_map_present(self):
        resp_a, resp_b, pair_info = self._make_pair(seed=42)

        self.assertIn("_deblind_map", pair_info)
        self.assertIn("Response A", pair_info["_deblind_map"])
        self.assertIn("Response B", pair_info["_deblind_map"])
        self.assertIn("_response_a_id", pair_info)
        self.assertIn("_response_b_id", pair_info)
        self.assertEqual(pair_info["_model_a_id"], "model-alpha")
        self.assertEqual(pair_info["_model_b_id"], "model-beta")

    def test_create_comparison_pair_display_has_both_responses(self):
        resp_a, resp_b, pair_info = self._make_pair(seed=42)

        display = pair_info["display"]
        self.assertIn("Response A", display)
        self.assertIn("Response B", display)

        all_ids = {display["Response A"]["response_id"], display["Response B"]["response_id"]}
        self.assertIn(resp_a.response_id, all_ids)
        self.assertIn(resp_b.response_id, all_ids)

    def test_record_comparison_produces_valid_record(self):
        resp_a, resp_b, pair_info = self._make_pair(seed=42)

        prefs = [
            RubricPreference(rubric="overall", winner="a", rationale="Alpha was clearer"),
        ]
        record = record_comparison(
            pair_info=pair_info,
            preferences=prefs,
            judge_type=JudgeType.PHYSICIAN,
            judge_id="physician-test",
        )

        self.assertIsInstance(record, PairwiseComparisonRecord)
        self.assertEqual(record.case_id, "CASE-001")
        self.assertEqual(record.model_a_id, "model-alpha")
        self.assertEqual(record.model_b_id, "model-beta")
        self.assertEqual(record.seed, 42)
        self.assertEqual(record.overall_winner, "a")

    def test_comparisons_to_bt_input_win(self):
        resp_a, resp_b, pair_info = self._make_pair(seed=42)
        prefs = [RubricPreference(rubric="overall", winner="a")]
        record = record_comparison(pair_info, prefs, JudgeType.PHYSICIAN, "test")

        bt = comparisons_to_bt_input([record], rubric="overall")
        self.assertEqual(len(bt), 1)
        self.assertEqual(bt[0]["winner"], "model-alpha")
        self.assertEqual(bt[0]["loser"], "model-beta")

    def test_comparisons_to_bt_input_tie_splits(self):
        resp_a, resp_b, pair_info = self._make_pair(seed=42)
        prefs = [RubricPreference(rubric="overall", winner="tie")]
        record = record_comparison(pair_info, prefs, JudgeType.PHYSICIAN, "test")

        bt = comparisons_to_bt_input([record], rubric="overall")
        # Tie splits into 2 entries, one for each direction
        self.assertEqual(len(bt), 2)
        winners = {e["winner"] for e in bt}
        self.assertIn("model-alpha", winners)
        self.assertIn("model-beta", winners)

    def test_comparisons_to_bt_input_no_matching_rubric(self):
        resp_a, resp_b, pair_info = self._make_pair(seed=42)
        prefs = [RubricPreference(rubric="r1", winner="a")]
        record = record_comparison(pair_info, prefs, JudgeType.PHYSICIAN, "test")

        # Ask for "overall" but record only has "r1" → skip
        bt = comparisons_to_bt_input([record], rubric="overall")
        self.assertEqual(len(bt), 0)

    def test_seed_42_order_is_deterministic_known_value(self):
        # Verify the specific output of seed=42 is stable across runs
        resp_a = _make_model_response("CASE-001", "model-alpha", "alpha text")
        resp_b = _make_model_response("CASE-001", "model-beta", "beta text")
        pair1 = create_comparison_pair("CASE-001", resp_a, resp_b, seed=42)
        pair2 = create_comparison_pair("CASE-001", resp_a, resp_b, seed=42)

        self.assertEqual(pair1["presented_order"], pair2["presented_order"])
        self.assertEqual(pair1["seed"], 42)


# ── End-to-End Smoke Test ─────────────────────────────────────────────────────

class TestEndToEndPipeline(unittest.TestCase):

    # Resolve paths relative to this file so tests work from any cwd
    _BASE = Path(__file__).parent.parent  # im_trace/
    CASES_PATH = _BASE / "cases" / "raw" / "sample_cases.jsonl"
    RESPONSES_PATH = _BASE / "cases" / "processed" / "mock_responses.jsonl"

    def test_load_cases(self):
        from im_trace.run_benchmark import load_cases

        self.assertTrue(self.CASES_PATH.exists(), f"sample_cases.jsonl not found at {self.CASES_PATH}")
        cases = load_cases(self.CASES_PATH)

        self.assertGreater(len(cases), 0)
        for case in cases:
            self.assertIsInstance(case, ClinicalCase)
            self.assertTrue(case.case_id)
            self.assertTrue(case.title)

    def test_load_responses(self):
        from im_trace.run_benchmark import load_responses

        self.assertTrue(self.RESPONSES_PATH.exists(), f"mock_responses.jsonl not found at {self.RESPONSES_PATH}")
        responses = load_responses(self.RESPONSES_PATH)

        self.assertGreater(len(responses), 0)
        for resp in responses:
            self.assertIsInstance(resp, ModelResponse)
            self.assertTrue(resp.case_id)
            self.assertTrue(resp.model_id)

    def test_pipeline_produces_leaderboard(self):
        from im_trace.run_benchmark import run_pipeline

        if not self.CASES_PATH.exists() or not self.RESPONSES_PATH.exists():
            self.skipTest("Sample data files not found; skipping end-to-end test")

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "test_run"
            summary = run_pipeline(
                cases_path=self.CASES_PATH,
                responses_path=self.RESPONSES_PATH,
                output_dir=output_dir,
                seed=42,
            )

            # leaderboard.json must exist
            leaderboard_path = output_dir / "leaderboard.json"
            self.assertTrue(leaderboard_path.exists(), "leaderboard.json was not produced")

            with open(leaderboard_path) as f:
                leaderboard = json.load(f)

            # Top-level keys
            self.assertIn("model_scores", leaderboard)
            self.assertIn("n_cases", leaderboard)
            self.assertIn("n_models", leaderboard)
            self.assertGreater(leaderboard["n_cases"], 0)
            self.assertGreater(leaderboard["n_models"], 0)

            # Each model entry has required fields
            for entry in leaderboard["model_scores"]:
                self.assertIn("model_id", entry)
                self.assertIn("total_mean", entry)
                self.assertIn("total_ci_lower", entry)
                self.assertIn("total_ci_upper", entry)
                # Score must be within valid range
                self.assertGreaterEqual(entry["total_mean"], 0.0)
                self.assertLessEqual(entry["total_mean"], 9.0)

            # run_summary.json must exist
            summary_path = output_dir / "run_summary.json"
            self.assertTrue(summary_path.exists(), "run_summary.json was not produced")

            # annotations.jsonl must exist and be non-empty
            ann_path = output_dir / "annotations.jsonl"
            self.assertTrue(ann_path.exists(), "annotations.jsonl was not produced")
            with open(ann_path) as f:
                lines = [l for l in f if l.strip()]
            self.assertGreater(len(lines), 0)

            # BenchmarkRunSummary fields
            self.assertGreater(summary.n_cases, 0)
            self.assertGreater(summary.n_models, 0)
            self.assertEqual(len(summary.models_evaluated), summary.n_models)

    def test_generate_mock_annotation_deterministic(self):
        from im_trace.run_benchmark import generate_mock_annotation, load_cases, load_responses

        cases = load_cases(self.CASES_PATH)
        responses = load_responses(self.RESPONSES_PATH)

        # Find a matching case/response pair
        case = cases[0]
        resp = next((r for r in responses if r.case_id == case.case_id), None)
        if resp is None:
            self.skipTest("No response found for first case")

        ann1 = generate_mock_annotation(case, resp, seed=42)
        ann2 = generate_mock_annotation(case, resp, seed=42)

        # Same seed → same scores
        self.assertEqual(int(ann1.r1.score), int(ann2.r1.score))
        self.assertEqual(int(ann1.r2.score), int(ann2.r2.score))


if __name__ == "__main__":
    unittest.main()
