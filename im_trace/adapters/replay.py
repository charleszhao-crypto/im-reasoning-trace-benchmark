"""
Replay adapter for deterministic evaluation against stored responses.

Loads ModelResponse objects from a JSONL file at initialization and
returns the matching stored response by case_id. Useful for:
  - Reproducible benchmark runs without re-querying APIs
  - Debugging evaluator/scorer logic against known inputs
  - Cost-free re-scoring after rubric updates

JSONL format: one JSON-serialized ModelResponse per line.

Usage:
    adapter = ReplayAdapter("/path/to/stored_responses.jsonl")
    resp = adapter.generate(case)  # Returns stored response for case.case_id

Raises:
    FileNotFoundError: If the JSONL path does not exist.
    KeyError: If no stored response exists for case.case_id.
    ValueError: If a line in the JSONL file cannot be parsed as ModelResponse.
"""

from __future__ import annotations

import json
from pathlib import Path

from im_trace.cases.schema.models import ClinicalCase, ModelResponse
from im_trace.adapters.base import BaseAdapter


class ReplayAdapter(BaseAdapter):
    """Adapter that replays stored ModelResponse objects from a JSONL file.

    The model_id and provider reported by this adapter reflect the first
    response loaded from the file. If the file contains responses from
    multiple models, consider constructing one ReplayAdapter per model
    by pre-filtering the JSONL.

    Args:
        jsonl_path: Path to the JSONL file containing stored ModelResponse objects.

    Raises:
        FileNotFoundError: If jsonl_path does not exist.
        ValueError: If any line in the file is not a valid ModelResponse.
    """

    def __init__(self, jsonl_path: str | Path) -> None:
        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"Replay JSONL not found: {path}")

        self._responses: dict[str, ModelResponse] = {}
        self._model_id_inferred: str = "replay-unknown"
        self._provider_inferred: str = "replay"

        with path.open("r", encoding="utf-8") as f:
            for lineno, raw in enumerate(f, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                    resp = ModelResponse(**data)
                except Exception as exc:
                    raise ValueError(
                        f"Failed to parse ModelResponse at {path}:{lineno}: {exc}"
                    ) from exc

                self._responses[resp.case_id] = resp

                # Use the first response to set adapter-level identifiers
                if len(self._responses) == 1:
                    self._model_id_inferred = resp.model_id
                    self._provider_inferred = resp.model_provider

    @property
    def model_id(self) -> str:
        return self._model_id_inferred

    @property
    def provider(self) -> str:
        return self._provider_inferred

    def generate(self, case: ClinicalCase, system_prompt: str = "") -> ModelResponse:
        """Return the stored ModelResponse for case.case_id.

        Args:
            case: The ClinicalCase to look up.
            system_prompt: Ignored for replay — the stored response is returned as-is.

        Raises:
            KeyError: If no stored response exists for case.case_id.
        """
        if case.case_id not in self._responses:
            raise KeyError(
                f"No stored response for case_id={case.case_id!r}. "
                f"Available: {sorted(self._responses)}"
            )
        return self._responses[case.case_id]

    @property
    def loaded_case_ids(self) -> list[str]:
        """Sorted list of case_ids with stored responses."""
        return sorted(self._responses)
