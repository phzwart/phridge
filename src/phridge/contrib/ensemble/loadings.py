"""Full complex loadings: a real score times a complex mode per reflection."""

from __future__ import annotations

from typing import Any

import numpy as np


class FullComplexLoadings:
    """``LoadingParameterization`` with complex Δ_k and real z_nk.

    ``apply`` returns ``scores @ loadings`` with shape ``(N, H)``.
    """

    def apply(self, loadings: Any, scores: Any) -> np.ndarray:
        modes = np.asarray(loadings, dtype=np.complex128)
        z = np.asarray(scores, dtype=np.float64)
        if modes.ndim == 1:
            modes = modes.reshape(1, -1)
        if z.ndim == 1:
            z = z.reshape(1, -1)
        if modes.size == 0 or z.size == 0 or modes.shape[0] == 0:
            n_h = modes.shape[-1] if modes.ndim == 2 else 0
            n_data = z.shape[0] if z.ndim == 2 else 0
            return np.zeros((n_data, n_h), dtype=np.complex128)
        return z.astype(np.complex128) @ modes
