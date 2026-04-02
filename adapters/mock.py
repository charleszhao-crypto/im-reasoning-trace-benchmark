"""
Mock adapter for deterministic testing without live API calls.

Returns pre-configured responses keyed by case_id. Useful for:
  - Unit tests that need stable, reproducible outputs
  - Local development without API credentials
  - Verifying evaluator/scorer logic in isolation

Usage:
    responses = {
        "IM-STEMI": "This patient has inferior STEMI based on ST elevations...",
        "IM-CLOSURE": "Likely GERD given epigastric discomfort...",
    }
    adapter = MockAdapter(responses=responses)
    resp = adapter.generate(case)
"""

from __future__ import annotations

from im_trace.cases.schema.models import ClinicalCase, ModelResponse
from im_trace.adapters.base import BaseAdapter

_GENERIC_RESPONSE = (
    "Based on the clinical presentation, this case warrants a broad differential "
    "diagnosis. I would prioritize obtaining further history, relevant laboratory "
    "studies, and imaging as indicated. The most urgent concern is ruling out "
    "life-threatening etiologies before settling on a working diagnosis. "
    "A systematic approach to management would include stabilization, diagnostic "
    "workup, and specialist consultation as appropriate."
)


class MockAdapter(BaseAdapter):
    """Deterministic mock adapter for testing.

    Args:
        responses: Mapping of case_id to response text. Cases not in the
            dict receive a generic fallback response.
        model_id: Model identifier reported in ModelResponse. Defaults to
            'mock-model-alpha'.
    """

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        model_id: str = "mock-model-alpha",
    ) -> None:
        self._responses: dict[str, str] = responses or {}
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def provider(self) -> str:
        return "mock"

    def generate(self, case: ClinicalCase, system_prompt: str = "") -> ModelResponse:
        """Return the pre-configured response for case.case_id, or a generic fallback."""
        response_text = self._responses.get(case.case_id, _GENERIC_RESPONSE)
        return ModelResponse(
            case_id=case.case_id,
            model_id=self._model_id,
            model_provider=self.provider,
            response_text=response_text,
            response_time_ms=0,
        )
