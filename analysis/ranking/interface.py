"""
Ranking Backend Interface — pluggable pairwise ranking methods.

All backends accept pairwise comparison records and return a RankingResult
with latent scores, superiority matrix, uncertainty, and top-k confidence.

Bradley-Terry remains the default. Additional backends (spectral, SyncRank)
are analysis-only alternatives for cross-validation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RankingResult:
    """Standardized output from any ranking backend."""
    backend_name: str
    latent_scores: dict[str, float]              # {model_id: score} — higher is better
    elo_ratings: dict[str, float]                 # {model_id: Elo-like rating}
    rankings: list[dict]                          # [{rank, model_id, score, elo}] sorted
    pairwise_superiority: dict[str, dict[str, float]]  # P(A > B) for all pairs
    top_k_confidence: dict[int, dict[str, float]]  # {k: {model_id: P(in top k)}}
    diagnostics: dict = field(default_factory=dict)  # Backend-specific diagnostics
    converged: bool = True
    iterations: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def top_model(self) -> Optional[str]:
        return self.rankings[0]["model_id"] if self.rankings else None


class RankingBackend(ABC):
    """Abstract base class for ranking backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier, e.g. 'bradley_terry', 'spectral'."""
        ...

    @abstractmethod
    def fit(
        self,
        comparisons: list[dict],
        n_bootstrap: int = 1000,
        seed: int = 42,
    ) -> RankingResult:
        """
        Fit ranking model to pairwise comparison data.

        Args:
            comparisons: list of {"winner": model_id, "loser": model_id}
            n_bootstrap: number of bootstrap samples for CI/superiority
            seed: deterministic seed

        Returns:
            RankingResult with all fields populated.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
