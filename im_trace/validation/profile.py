"""
Validated Profile — frozen configuration for reproducible benchmark runs.

A validated profile pins:
  - rubric version
  - prompt version
  - case subset (by case_id list)
  - aggregation recipe (scoring formula, safety cap threshold, bootstrap params)
  - adapter config (model IDs, temperature, max_tokens)

Exploratory runs use draft prompts and evolving case subsets.
Validated runs use locked profiles and generate deterministic manifests.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class AggregationRecipe(BaseModel):
    """Frozen aggregation parameters."""
    scoring_formula: str = "R1 + R2 + (R3 * 1.5) + R4"
    safety_cap: float = 4.0
    safety_cap_enabled: bool = True
    bootstrap_n: int = 1000
    bootstrap_ci: float = 0.95
    bootstrap_seed: int = 42


class AdapterConfig(BaseModel):
    """Frozen adapter configuration for one model."""
    model_id: str
    provider: str
    temperature: float = 0.3
    max_tokens: int = 4000
    system_prompt_hash: Optional[str] = None


class ValidatedProfile(BaseModel):
    """
    Frozen benchmark configuration.

    Once a profile is created and used for a validated run,
    it should NOT be modified. Create a new version instead.
    """
    profile_id: str
    profile_version: str = "1.0"
    description: str = ""

    rubric_version: str = "1.0"
    prompt_version: str = "1.0"

    case_ids: list[str]                         # Frozen case subset
    adapter_configs: list[AdapterConfig]         # Frozen model configs
    aggregation: AggregationRecipe = Field(default_factory=AggregationRecipe)

    mode: str = "validated"                     # "validated" or "exploratory"
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    @property
    def content_hash(self) -> str:
        """Deterministic hash of the profile's scoring-relevant content."""
        payload = json.dumps({
            "rubric_version": self.rubric_version,
            "prompt_version": self.prompt_version,
            "case_ids": sorted(self.case_ids),
            "models": sorted(a.model_id for a in self.adapter_configs),
            "aggregation": self.aggregation.model_dump(),
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


class RunManifest(BaseModel):
    """
    Immutable record of a validated run.

    Generated at run completion. References the profile used,
    content hash, and output file locations.
    """
    manifest_id: str
    profile_id: str
    profile_content_hash: str
    run_mode: str                               # "validated" or "exploratory"

    n_cases: int
    n_models: int
    n_annotations: int

    output_files: dict[str, str]                # {"leaderboard": "path", "annotations": "path", ...}

    started_at: str
    completed_at: str
    duration_seconds: float

    warnings: list[str] = Field(default_factory=list)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path) -> "RunManifest":
        with open(path) as f:
            return cls.model_validate_json(f.read())
