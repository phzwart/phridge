"""Cubic B-spline in s^2, log-parameterized for positivity.

``free_beta`` is piecewise-constant shells plus a second-difference penalty, not a
spline. This module is new; it only reuses the free-beta convention that beta is
independent of ``1 - sigma_A^2`` and stays strictly positive.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def open_uniform_knots(x_min: float, x_max: float, n_coef: int, degree: int = 3) -> np.ndarray:
    """Open uniform knot vector of length ``n_coef + degree + 1``."""
    n_coef = int(n_coef)
    degree = int(degree)
    if n_coef < degree + 1:
        raise ValueError("n_coef must be >= degree + 1")
    n_interior = n_coef - degree
    lo = float(x_min)
    hi = float(x_max)
    if hi <= lo:
        hi = lo + 1e-6
    interior = np.linspace(lo, hi, n_interior + 1, dtype=np.float64)
    knots = np.empty(n_coef + degree + 1, dtype=np.float64)
    knots[: degree + 1] = lo
    knots[degree : degree + n_interior + 1] = interior
    knots[n_coef:] = hi
    return knots


def bspline_design_numpy(x: np.ndarray, knots: np.ndarray, degree: int = 3) -> np.ndarray:
    """Cox-de Boor design matrix, shape ``(len(x), n_coef)``."""
    t = np.asarray(x, dtype=np.float64).reshape(-1)
    knots = np.asarray(knots, dtype=np.float64).reshape(-1)
    lo, hi = float(knots[degree]), float(knots[-degree - 1])
    if hi > lo:
        t = np.clip(t, lo, np.nextafter(hi, lo))
    n_coef = int(knots.size) - degree - 1
    n = t.size
    basis = np.zeros((n, n_coef), dtype=np.float64)
    # Degree 0.
    for i in range(n_coef + degree):
        left = knots[i]
        right = knots[i + 1]
        if i == n_coef + degree - 1:
            sel = (t >= left) & (t <= right)
        else:
            sel = (t >= left) & (t < right)
        if i < n_coef:
            basis[sel, i] = 1.0
        # store extra columns temporarily in a padded array for the recursion
    padded = np.zeros((n, n_coef + degree), dtype=np.float64)
    for i in range(n_coef + degree):
        left = knots[i]
        right = knots[i + 1]
        if i == n_coef + degree - 1:
            sel = (t >= left) & (t <= right)
        else:
            sel = (t >= left) & (t < right)
        padded[sel, i] = 1.0
    for k in range(1, degree + 1):
        nxt = np.zeros_like(padded)
        for i in range(n_coef + degree - k):
            denom_l = knots[i + k] - knots[i]
            denom_r = knots[i + k + 1] - knots[i + 1]
            left = 0.0 if denom_l == 0.0 else ((t - knots[i]) / denom_l) * padded[:, i]
            right = 0.0 if denom_r == 0.0 else ((knots[i + k + 1] - t) / denom_r) * padded[:, i + 1]
            nxt[:, i] = left + right
        padded = nxt
    basis = padded[:, :n_coef]
    row_sum = basis.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum > 0, row_sum, 1.0)
    return basis / row_sum


def bspline_design(x: Any, knots: Any, degree: int = 3) -> Any:
    """Design matrix; numpy or torch depending on ``x``."""
    is_torch = hasattr(x, "clamp") and hasattr(x, "device") and type(x).__module__.startswith("torch")
    if is_torch:
        import torch

        knots_t = torch.as_tensor(knots, dtype=x.dtype, device=x.device)
        t = x.reshape(-1)
        lo, hi = knots_t[degree], knots_t[-degree - 1]
        if float(hi) > float(lo):
            t = t.clamp(min=lo, max=hi - 1e-12)
        n_coef = int(knots_t.numel()) - degree - 1
        n = int(t.numel())
        padded = torch.zeros((n, n_coef + degree), dtype=x.dtype, device=x.device)
        for i in range(n_coef + degree):
            left = knots_t[i]
            right = knots_t[i + 1]
            if i == n_coef + degree - 1:
                sel = (t >= left) & (t <= right)
            else:
                sel = (t >= left) & (t < right)
            padded[sel, i] = 1.0
        for k in range(1, degree + 1):
            nxt = torch.zeros_like(padded)
            for i in range(n_coef + degree - k):
                denom_l = knots_t[i + k] - knots_t[i]
                denom_r = knots_t[i + k + 1] - knots_t[i + 1]
                left = torch.zeros(n, dtype=x.dtype, device=x.device)
                right = torch.zeros(n, dtype=x.dtype, device=x.device)
                if float(denom_l) != 0.0:
                    left = ((t - knots_t[i]) / denom_l) * padded[:, i]
                if float(denom_r) != 0.0:
                    right = ((knots_t[i + k + 1] - t) / denom_r) * padded[:, i + 1]
                nxt[:, i] = left + right
            padded = nxt
        basis = padded[:, :n_coef]
        row_sum = basis.sum(dim=1, keepdim=True).clamp(min=1e-12)
        return basis / row_sum
    return bspline_design_numpy(np.asarray(x, dtype=np.float64), np.asarray(knots, dtype=np.float64), degree)


def eval_log_spline(x: Any, knots: Any, coef: Any, degree: int = 3) -> Any:
    """``exp(B(x) @ coef)``, strictly positive."""
    design = bspline_design(x, knots, degree)
    log_y = design @ coef
    if hasattr(log_y, "exp"):
        return log_y.exp()
    return np.exp(np.asarray(log_y, dtype=np.float64))
