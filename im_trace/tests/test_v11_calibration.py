"""
Tests for IM-TRACE v1.1 calibration modules:
  - Validated profile + runner
  - Rank stability
  - MFRM (sparse data warnings)
  - BT Laplace smoothing fix
  - Telemetry fields
"""

import json
import math
import tempfile
import unittest
from pathlib import Path

from im_trace.cases.schema.models import (
    ClinicalCase, ModelResponse, OrdinalScore, Specialty, Difficulty, AmbiguityClass,
)
from im_trace.validation.profile import ValidatedProfile, AdapterConfig, AggregationRecipe, RunManifest
from im_trace.validation.runner import run_validated
from im_trace.analysis.rank_stability import (
    bootstrap_rank_distribution, top_k_frequency, pairwise_superiority_matrix,
    leave_one_case_out, compute_fragility_summary, run_rank_stability_analysis,
)
from im_trace.analysis.mfrm import prepare_mfrm_data, fit_mfrm, mfrm_diagnostics
from im_trace.evaluators.aggregation.aggregate import fit_bradley_terry, bt_to_elo


class TestBTLaplaceSmoothing(unittest.TestCase):
    """Test that BT fitting no longer produces zero-strength models."""

    def test_bt_no_zero_strength(self):
        """Model with zero observed wins should still get a positive strength."""
        comparisons = [
            {"winner": "strong", "loser": "weak"},
            {"winner": "strong", "loser": "weak"},
            {"winner": "strong", "loser": "weak"},
        ]
        strengths = fit_bradley_terry(comparisons)
        self.assertGreater(strengths["weak"], 0.0, "Laplace smoothing should prevent zero strength")
        self.assertGreater(strengths["strong"], strengths["weak"])

    def test_bt_elo_no_negative_infinity(self):
        """Elo ratings should be finite for all models."""
        comparisons = [{"winner": "a", "loser": "b"}]
        strengths = fit_bradley_terry(comparisons)
        elo = bt_to_elo(strengths)
        for model, rating in elo.items():
            self.assertTrue(math.isfinite(rating), f"{model} has non-finite Elo: {rating}")
            self.assertGreater(rating, -2000, f"{model} Elo too low: {rating}")


class TestValidatedProfile(unittest.TestCase):

    def test_profile_content_hash_deterministic(self):
        p1 = ValidatedProfile(
            profile_id="test", case_ids=["A", "B"],
            adapter_configs=[AdapterConfig(model_id="m1", provider="mock")],
        )
        p2 = ValidatedProfile(
            profile_id="test", case_ids=["A", "B"],
            adapter_configs=[AdapterConfig(model_id="m1", provider="mock")],
        )
        self.assertEqual(p1.content_hash, p2.content_hash)

    def test_profile_hash_changes_with_cases(self):
        p1 = ValidatedProfile(
            profile_id="test", case_ids=["A", "B"],
            adapter_configs=[AdapterConfig(model_id="m1", provider="mock")],
        )
        p2 = ValidatedProfile(
            profile_id="test", case_ids=["A", "B", "C"],
            adapter_configs=[AdapterConfig(model_id="m1", provider="mock")],
        )
        self.assertNotEqual(p1.content_hash, p2.content_hash)

    def test_manifest_round_trip(self):
        m = RunManifest(
            manifest_id="TEST-1", profile_id="p1", profile_content_hash="abc123",
            run_mode="validated", n_cases=4, n_models=2, n_annotations=8,
            output_files={"leaderboard": "/tmp/lb.json"},
            started_at="2026-01-01T00:00:00", completed_at="2026-01-01T00:00:01",
            duration_seconds=1.0,
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            m.save(Path(f.name))
            loaded = RunManifest.load(Path(f.name))
        self.assertEqual(loaded.manifest_id, "TEST-1")
        self.assertEqual(loaded.profile_content_hash, "abc123")


class TestValidatedRunner(unittest.TestCase):

    def test_validated_run_produces_manifest(self):
        profile = ValidatedProfile(
            profile_id="test-run",
            case_ids=["IM-STEMI", "IM-CLOSURE", "IM-ABSTAIN", "IM-SAFETY"],
            adapter_configs=[
                AdapterConfig(model_id="model-alpha", provider="mock"),
                AdapterConfig(model_id="model-beta", provider="mock"),
            ],
            mode="validated",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = run_validated(
                profile=profile,
                cases_path=Path("im_trace/cases/raw/sample_cases.jsonl"),
                responses_path=Path("im_trace/cases/processed/mock_responses.jsonl"),
                output_root=Path(tmpdir),
            )
            self.assertTrue(manifest.manifest_id.startswith("VAL-"))
            self.assertEqual(manifest.n_cases, 4)
            self.assertEqual(manifest.n_models, 2)
            self.assertEqual(manifest.n_annotations, 8)
            self.assertEqual(manifest.run_mode, "validated")
            # Verify output files exist
            for name, path in manifest.output_files.items():
                self.assertTrue(Path(path).exists(), f"{name} not found at {path}")


class TestRankStability(unittest.TestCase):

    def test_bootstrap_rank_distribution_sums_to_one(self):
        scores = {"a": [7.0, 6.5, 7.2, 6.8], "b": [6.0, 6.5, 5.8, 6.2]}
        dist = bootstrap_rank_distribution(scores, n_bootstrap=500, seed=42)
        for model, ranks in dist.items():
            total = sum(ranks.values())
            self.assertAlmostEqual(total, 1.0, places=2, msg=f"{model} rank freqs don't sum to 1")

    def test_top_k_frequency(self):
        dist = {"a": {1: 0.8, 2: 0.2}, "b": {1: 0.2, 2: 0.8}}
        top1 = top_k_frequency(dist, k=1)
        self.assertEqual(top1["a"], 0.8)
        self.assertEqual(top1["b"], 0.2)

    def test_superiority_matrix_symmetric(self):
        scores = {"a": [7.0, 6.5], "b": [6.0, 6.5]}
        matrix = pairwise_superiority_matrix(scores, n_bootstrap=500, seed=42)
        # P(a>b) + P(b>a) should approximately equal 1
        p_ab = matrix.get("a", {}).get("b", 0)
        p_ba = matrix.get("b", {}).get("a", 0)
        self.assertAlmostEqual(p_ab + p_ba, 1.0, places=1)

    def test_leave_one_case_out(self):
        scores = {
            "a": [{"case_id": "c1", "total": 7.0}, {"case_id": "c2", "total": 5.0}],
            "b": [{"case_id": "c1", "total": 6.0}, {"case_id": "c2", "total": 6.5}],
        }
        loo = leave_one_case_out(scores)
        self.assertEqual(len(loo), 2)
        for r in loo:
            self.assertIn("removed_case_id", r)
            self.assertIn("rankings_changed", r)

    def test_fragility_summary_structure(self):
        dist = {"a": {1: 0.9, 2: 0.1}, "b": {1: 0.1, 2: 0.9}}
        matrix = {"a": {"b": 0.9}, "b": {"a": 0.1}}
        loo = [{"rankings_changed": False}, {"rankings_changed": False}]
        frag = compute_fragility_summary(dist, matrix, loo)
        self.assertIn("fragile", frag)
        self.assertIn("top1_confidence", frag)
        self.assertFalse(frag["fragile"])  # 0.9 confidence should not be fragile

    def test_full_rank_stability_analysis(self):
        scores = {
            "a": [{"case_id": "c1", "total": 7.0}, {"case_id": "c2", "total": 6.5}],
            "b": [{"case_id": "c1", "total": 6.0}, {"case_id": "c2", "total": 6.0}],
        }
        result = run_rank_stability_analysis(scores, n_bootstrap=100, seed=42)
        self.assertIn("rank_distributions", result)
        self.assertIn("fragility_summary", result)


class TestMFRM(unittest.TestCase):

    def test_sparse_data_emits_warnings(self):
        """MFRM with 1 rater should warn about sparse data."""
        annotations = [
            {"model_id": "a", "case_id": "c1", "evaluator_id": "r1", "total": 7.0},
            {"model_id": "b", "case_id": "c1", "evaluator_id": "r1", "total": 6.0},
        ]
        data = prepare_mfrm_data(annotations)
        fit = fit_mfrm(data)
        self.assertTrue(len(fit["warnings"]) > 0, "Should warn about sparse data")

    def test_mfrm_output_marked_analysis_only(self):
        annotations = [
            {"model_id": "a", "case_id": "c1", "evaluator_id": "r1", "total": 7.0},
        ]
        data = prepare_mfrm_data(annotations)
        fit = fit_mfrm(data)
        diag = mfrm_diagnostics(fit)
        self.assertTrue(diag.get("analysis_only", False))


class TestTelemetryFields(unittest.TestCase):

    def test_model_response_has_telemetry(self):
        resp = ModelResponse(
            case_id="c1", model_id="gpt-4o", model_provider="openai",
            response_text="test",
            input_tokens=150, output_tokens=300, total_tokens=450,
            estimated_cost_usd=0.0045,
        )
        self.assertEqual(resp.input_tokens, 150)
        self.assertEqual(resp.total_tokens, 450)
        self.assertEqual(resp.estimated_cost_usd, 0.0045)

    def test_telemetry_optional(self):
        resp = ModelResponse(
            case_id="c1", model_id="m1", model_provider="mock",
            response_text="test",
        )
        self.assertIsNone(resp.input_tokens)
        self.assertIsNone(resp.estimated_cost_usd)


if __name__ == "__main__":
    unittest.main()
