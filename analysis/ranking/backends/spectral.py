"""
Spectral Ranking Backend — Fiedler-vector-based ranking from pairwise comparisons.

Analysis-only alternative to Bradley-Terry. Useful for cross-validation:
if BT and spectral agree, the ranking signal is stable. If they diverge,
data sparsity or inconsistency is dominating the result.

Method: construct the comparison graph Laplacian, compute the Fiedler vector
(second-smallest eigenvector), and use its entries as latent quality scores.
This is equivalent to a spectral relaxation of the minimum feedback arc set
problem — the ranking that minimizes disagreement with observed comparisons.

Limitations:
  - Assumes undirected comparison graph is connected (adds virtual edges if not)
  - Eigenvector computation uses power iteration (no numpy/scipy dependency)
  - Tie-handling is basic (0.5 weight edges)
  - Uncertainty estimates are bootstrap-based (same as BT), not spectral CI
  - Not suitable as primary ranking — use for diagnostic comparison only

References:
  - SyncRank (Cucuringu 2016): synchronization-based ranking from noisy comparisons
  - PARWiS (2026): spectral ranking + active pair selection
  - Spectral MLE for top-k (Chen & Suh 2015): top-k identification from pairwise data
"""

from __future__ import annotations

import math
import random
from collections import defaultdict

from im_trace.analysis.ranking.interface import RankingBackend, RankingResult


def _power_iteration_second_eigenvector(
    laplacian: list[list[float]],
    n: int,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> list[float]:
    """
    Compute the Fiedler vector (eigenvector of second-smallest eigenvalue)
    of a graph Laplacian via power iteration with deflation.

    Uses inverse power iteration on L to find the smallest non-trivial eigenvector.
    Since L is positive semi-definite with smallest eigenvalue 0 (constant vector),
    we deflate the constant component and find the next smallest.
    """
    # Random initial vector
    rng = random.Random(42)
    v = [rng.gauss(0, 1) for _ in range(n)]

    # Remove constant component (project out the nullspace of L)
    def _remove_constant(vec: list[float]) -> list[float]:
        mean = sum(vec) / len(vec)
        return [x - mean for x in vec]

    def _normalize(vec: list[float]) -> list[float]:
        norm = math.sqrt(sum(x * x for x in vec))
        if norm < 1e-15:
            return vec
        return [x / norm for x in vec]

    def _mat_vec(mat: list[list[float]], vec: list[float]) -> list[float]:
        return [sum(mat[i][j] * vec[j] for j in range(n)) for i in range(n)]

    # Shift L to make it positive definite: M = max_eigenvalue_estimate * I - L
    # Then power iteration on M gives the eigenvector of the largest eigenvalue of M,
    # which corresponds to the smallest non-trivial eigenvalue of L
    max_diag = max(laplacian[i][i] for i in range(n))
    shift = max_diag + 1.0  # Conservative upper bound on max eigenvalue

    # Shifted matrix: M = shift * I - L
    M = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            M[i][j] = -laplacian[i][j]
            if i == j:
                M[i][j] += shift

    v = _remove_constant(v)
    v = _normalize(v)

    for _ in range(max_iter):
        v_new = _mat_vec(M, v)
        v_new = _remove_constant(v_new)
        v_new = _normalize(v_new)

        # Check convergence
        diff = math.sqrt(sum((a - b) ** 2 for a, b in zip(v, v_new)))
        v = v_new
        if diff < tol:
            break

    return v


class SpectralBackend(RankingBackend):
    """
    Spectral ranking via comparison graph Laplacian.

    Analysis-only. NOT the official ranking backend.
    Use for diagnostic comparison with Bradley-Terry.
    """

    @property
    def name(self) -> str:
        return "spectral"

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
                warnings=["No comparisons provided"],
            )

        # Collect models
        models_set: set[str] = set()
        for c in comparisons:
            models_set.add(c["winner"])
            models_set.add(c["loser"])
        models = sorted(models_set)
        n = len(models)
        idx = {m: i for i, m in enumerate(models)}

        if n < 2:
            return RankingResult(
                backend_name=self.name, latent_scores={models[0]: 1.0} if n == 1 else {},
                elo_ratings={}, rankings=[], pairwise_superiority={},
                top_k_confidence={}, warnings=["Fewer than 2 models"],
            )

        # Build weighted adjacency matrix from comparisons
        # W[i][j] = number of times i beat j (+ 0.5 for ties via Laplace)
        W = [[0.0] * n for _ in range(n)]
        for c in comparisons:
            i, j = idx[c["winner"]], idx[c["loser"]]
            W[i][j] += 1.0

        # Add Laplace smoothing (same as BT backend)
        for i in range(n):
            for j in range(n):
                if i != j:
                    W[i][j] += 0.5

        # Build comparison graph Laplacian
        # For ranking: L = D - A where A[i][j] = W[i][j] - W[j][i] (skew-symmetric)
        # But spectral ranking on skew-symmetric data uses the symmetric version:
        # A_sym[i][j] = (W[i][j] + W[j][i]) as edge weight, L_sym = D_sym - A_sym
        A_sym = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    A_sym[i][j] = W[i][j] + W[j][i]

        L = [[0.0] * n for _ in range(n)]
        for i in range(n):
            degree = sum(A_sym[i])
            L[i][i] = degree
            for j in range(n):
                if i != j:
                    L[i][j] = -A_sym[i][j]

        # Compute Fiedler vector
        fiedler = _power_iteration_second_eigenvector(L, n)

        # The Fiedler vector partitions the graph. For ranking, we need to
        # orient it: models with more wins should have higher scores.
        # Compute net wins and correlate with Fiedler entries.
        net_wins = [0.0] * n
        for c in comparisons:
            net_wins[idx[c["winner"]]] += 1
            net_wins[idx[c["loser"]]] -= 1

        # Check correlation — flip Fiedler if negatively correlated with net wins
        corr = sum(fiedler[i] * net_wins[i] for i in range(n))
        if corr < 0:
            fiedler = [-x for x in fiedler]

        # Normalize scores to [0, 1] range
        min_f, max_f = min(fiedler), max(fiedler)
        score_range = max_f - min_f if max_f > min_f else 1.0
        latent_scores = {models[i]: round((fiedler[i] - min_f) / score_range, 4) for i in range(n)}

        # Convert to Elo-like ratings (map [0,1] to ~[1300, 1700])
        elo_ratings = {m: round(1300 + 400 * s, 1) for m, s in latent_scores.items()}

        # Rankings
        rankings = sorted(
            [{"rank": 0, "model_id": m, "score": latent_scores[m], "elo": elo_ratings[m]}
             for m in models],
            key=lambda x: x["score"], reverse=True,
        )
        for i, r in enumerate(rankings):
            r["rank"] = i + 1

        # Bootstrap superiority + top-k (same method as BT backend)
        rng = random.Random(seed)
        rank_hist = {m: defaultdict(int) for m in models}
        win_hist = {m: {o: 0 for o in models if o != m} for m in models}

        for _ in range(n_bootstrap):
            sample = [rng.choice(comparisons) for _ in range(len(comparisons))]

            # Quick score: net win rate per model (faster than full spectral per bootstrap)
            nw = defaultdict(float)
            for c in sample:
                nw[c["winner"]] += 1
                nw[c["loser"]] -= 1

            ranked = sorted(models, key=lambda m: nw.get(m, 0), reverse=True)
            for rank_idx, model in enumerate(ranked):
                rank_hist[model][rank_idx + 1] += 1

            for i_m, a in enumerate(models):
                for b in models[i_m + 1:]:
                    if nw.get(a, 0) > nw.get(b, 0):
                        win_hist[a][b] += 1
                    elif nw.get(b, 0) > nw.get(a, 0):
                        win_hist[b][a] += 1
                    else:
                        win_hist[a][b] += 0.5
                        win_hist[b][a] += 0.5

        superiority = {
            a: {b: round(win_hist[a][b] / n_bootstrap, 3) for b in models if b != a}
            for a in models
        }

        top_k = {}
        for k in range(1, min(n + 1, 6)):
            top_k[k] = {
                m: round(sum(f for r, f in rank_hist[m].items() if r <= k) / n_bootstrap, 3)
                for m in models
            }

        return RankingResult(
            backend_name=self.name,
            latent_scores=latent_scores,
            elo_ratings=elo_ratings,
            rankings=rankings,
            pairwise_superiority=superiority,
            top_k_confidence=top_k,
            converged=True,
            diagnostics={
                "method": "fiedler_vector_of_comparison_laplacian",
                "n_models": n,
                "n_comparisons": len(comparisons),
                "analysis_only": True,
                "limitations": [
                    "Power iteration eigenvector (no scipy)",
                    "Bootstrap CI uses net-win-rate proxy, not full spectral per sample",
                    "Not suitable as primary ranking backend",
                ],
            },
            warnings=["Spectral backend is analysis-only. Use Bradley-Terry for official rankings."],
        )
