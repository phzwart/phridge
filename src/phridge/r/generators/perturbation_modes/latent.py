"""Per-dataset real scores from complex residuals, with the Gram covariance."""

from __future__ import annotations

from typing import Any

import numpy as np


class ComplexGLS:
    """``LatentPosterior`` for ``Re[Δ̄ Ω Δᵀ] z = Re[Δ̄ Ω (R - Δ0)]``.

    ``estimate`` returns the point ``z``. ``covariance`` is the inverse of the
    real Gram matrix (optionally ridged).
    """

    def __init__(self, delta0: Any = None) -> None:
        self.delta0 = None if delta0 is None else np.asarray(delta0, dtype=np.complex128).reshape(-1)

    def _gram(
        self,
        residual: np.ndarray,
        weights: np.ndarray,
        loadings: np.ndarray,
        ridge: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        resid = np.asarray(residual, dtype=np.complex128).reshape(-1)
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        modes = np.asarray(loadings, dtype=np.complex128)
        if modes.ndim == 1:
            modes = modes.reshape(1, -1)
        if self.delta0 is not None:
            resid = resid - self.delta0
        rank = int(modes.shape[0])
        gram = np.zeros((rank, rank), dtype=np.float64)
        rhs = np.zeros(rank, dtype=np.float64)
        if rank == 0:
            return gram, rhs, resid
        weighted = np.conj(modes) * w[None, :]
        gram = np.real(weighted @ modes.T)
        rhs = np.real(weighted @ resid)
        if ridge:
            gram = gram + float(ridge) * np.eye(rank)
        return gram, rhs, resid

    def estimate(self, residual: Any, weights: Any, loadings: Any, ridge: float = 1e-6) -> np.ndarray:
        gram, rhs, _resid = self._gram(residual, weights, loadings, ridge)
        if gram.shape[0] == 0:
            return np.zeros(0, dtype=np.float64)
        try:
            return np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(gram, rhs, rcond=None)[0]

    def covariance(self, residual: Any, weights: Any, loadings: Any, ridge: float = 0.0) -> np.ndarray:
        gram, _rhs, _resid = self._gram(residual, weights, loadings, ridge)
        if gram.shape[0] == 0:
            return np.zeros((0, 0), dtype=np.float64)
        return np.linalg.inv(gram)
