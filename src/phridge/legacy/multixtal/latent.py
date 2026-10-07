"""Phase-1 ``LatentPosterior``: point mass at the GLS score estimate."""

from __future__ import annotations

from typing import Any

import numpy as np


class PointMassGLS:
    """``z_d`` by generalized least squares from dataset ``d``'s own residuals."""

    def estimate(self, residual: Any, weights: Any, loadings: Any, ridge: float = 1e-6) -> np.ndarray:
        r = np.asarray(residual, dtype=np.float64).reshape(-1)
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        l = np.asarray(loadings, dtype=np.float64)
        if l.ndim == 1:
            l = l.reshape(1, -1)
        rank = l.shape[0]
        if rank == 0:
            return np.zeros(0, dtype=np.float64)
        # z = (L W L^T + λI)^{-1} L W r
        lw = l * w[None, :]
        gram = lw @ l.T
        gram = gram + float(ridge) * np.eye(rank)
        rhs = lw @ r
        try:
            return np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(gram, rhs, rcond=None)[0]
