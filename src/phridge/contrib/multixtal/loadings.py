"""Phase-1 ``LoadingParameterization``: free per-reflection in-phase coefficients."""

from __future__ import annotations

from typing import Any

import numpy as np


class FreeInPhaseLoadings:
    def apply(self, loadings: Any, scores: Any) -> np.ndarray:
        l = np.asarray(loadings, dtype=np.float64)
        z = np.asarray(scores, dtype=np.float64)
        if l.size == 0 or z.size == 0:
            n_h = l.shape[-1] if l.ndim == 2 else 0
            n_d = z.shape[0] if z.ndim == 2 else 0
            return np.zeros((n_d, n_h), dtype=np.float64)
        return z @ l
