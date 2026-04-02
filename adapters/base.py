"""
Abstract base adapter for IM-TRACE model integrations.

All provider adapters (Anthropic, OpenAI, Google, mock, replay) must
implement this interface. The adapter is responsible for:
  - Accepting a ClinicalCase and optional system prompt
  - Returning a fully-populated ModelResponse
  - Exposing stable model_id and provider identifiers

Design note: Adapters are intentionally thin. They do not implement
scoring, caching, or retry logic — those concerns live in the evaluator
and harness layers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from im_trace.cases.schema.models import ClinicalCase, ModelResponse


class BaseAdapter(ABC):
    """Abstract base class for all IM-TRACE model adapters."""

    @abstractmethod
    def generate(self, case: ClinicalCase, system_prompt: str = "") -> ModelResponse:
        """Generate a model response for the given clinical case.

        Args:
            case: The ClinicalCase to present to the model.
            system_prompt: Optional system-level instructions. Defaults to
                empty string (adapter may apply its own default).

        Returns:
            A ModelResponse populated with at minimum: case_id, model_id,
            model_provider, and response_text.
        """
        ...

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Stable model identifier, e.g. 'claude-sonnet-4-6' or 'gpt-4o'."""
        ...

    @property
    @abstractmethod
    def provider(self) -> str:
        """Provider name, e.g. 'anthropic', 'openai', 'google', 'mock'."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_id={self.model_id!r}, provider={self.provider!r})"
