"""
Bradley-Terry Backend — wraps existing BT implementation behind the ranking interface.

This is the default (official) ranking backend. Results from this backend
are used in validated leaderboards. Alternative backends are analysis-only.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict

from im_trace.analysis.ranking.interface import RankingBackend, RankingResult
from im_trace.evaluators.aggregation.aggregate import (
    fit_bradley_terry, bt_to_elo,
)


class BradleyTerryBackend(RankingBackend):

    @property
    def name(self) -> str:
        return "bradley_terry"

    def fit(
        self,
        comparisons: list[dict],
        n_bootstrap: int = 1000,
        seed: int = 42,
    ) -> RankingResult:
        if not comparisons:
            return RankingResult(
                backend_name=self.name, latent_scores={}, elo_ratings={},
                rankings=[], pairwise_superiority={}, top_k_confidence={},
            )

        # Fit BT on full data
        strengths = fit_bradley_terry(comparisons)
        elo = bt_to_elo(strengths)
        models = sorted(strengths.keys())

        rankings = sorted(
            [{"rank": 0, "model_id": m, "score": round(strengths[m], 4), "elo": elo[m]}
             for m in models],
            key=lambda x: x["score"], reverse=True,
        )
        for i, r in enumerate(rankings):
            r["rank"] = i + 1

        # Bootstrap for superiority matrix and top-k confidence
        rng = random.Random(seed)
        n = len(comparisons)
        rank_counts = {m: defaultdict(int) for m in models}
        win_counts = {m: {o: 0 for o in models if o != m} for m in models}

        for _ in range(n_bootstrap):
            sample = [rng.choice(comparisons) for _ in range(n)]
            bt = fit_bradley_terry(sample)

            # Record ranks
            ranked = sorted(bt.items(), key=lambda x: x[1], reverse=True)
            for rank_idx, (model, _) in enumerate(ranked):
                rank_counts[model][rank_idx + 1] += 1

            # Record pairwise wins
            for i_m, a in enumerate(models):
                for b in models[i_m + 1:]:
                    if bt.get(a, 0) > bt.get(b, 0):
                        win_counts[a][b] += 1
                    elif bt.get(b, 0) > bt.get(a, 0):
                        win_counts[b][a] += 1
                    else:
                        win_counts[a][b] += 0.5
                        win_counts[b][a] += 0.5

        # Pairwise superiority matrix
        superiority = {
            a: {b: round(win_counts[a][b] / n_bootstrap, 3) for b in models if b != a}
            for a in models
        }

        # Top-k confidence
        top_k = {}
        for k in range(1, min(len(models) + 1, 6)):  # top-1 through top-5
            top_k[k] = {
                m: round(sum(freq for rank, freq in rank_counts[m].items() if rank <= k) / n_bootstrap, 3)
                for m in models
            }

        return RankingResult(
            backend_name=self.name,
            latent_scores=strengths,
            elo_ratings=elo,
            rankings=rankings,
            pairwise_superiority=superiority,
            top_k_confidence=top_k,
            converged=True,
            diagnostics={"n_comparisons": len(comparisons), "n_bootstrap": n_bootstrap},
        )
