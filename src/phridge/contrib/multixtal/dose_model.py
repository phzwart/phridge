"""Dose profiles ``g_k(ρ D)``: linear, exponential, and learned."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from phridge.contrib.multixtal.spline import bspline_design_numpy, open_uniform_knots


@dataclass
class LinearDose:
    def profiles(self, dose: Any, rho: Any = None) -> np.ndarray:
        d = _effective(dose, rho)
        return d.reshape(-1, 1)


@dataclass
class ExpDose:
    d_c: np.ndarray

    def profiles(self, dose: Any, rho: Any = None) -> np.ndarray:
        d = _effective(dose, rho)
        out = np.stack([1.0 - np.exp(-d / max(float(dc), 1e-8)) for dc in self.d_c], axis=1)
        return out


@dataclass
class LearnedDose:
    knots: np.ndarray
    coef: np.ndarray  # (n_knots, K)
    bands: Optional[np.ndarray] = None

    def profiles(self, dose: Any, rho: Any = None) -> np.ndarray:
        d = _effective(dose, rho)
        design = bspline_design_numpy(d, self.knots)
        return design @ self.coef


def _effective(dose: Any, rho: Any) -> np.ndarray:
    d = np.asarray(dose, dtype=np.float64).reshape(-1)
    if rho is None:
        return d
    r = np.asarray(rho, dtype=np.float64)
    if r.size == 1:
        return float(r.reshape(-1)[0]) * d
    if r.size == d.size:
        return r.reshape(-1) * d
    raise ValueError("rho must be scalar or aligned with dose")


def build_dose_model(
    kind: str,
    *,
    n_exp_modes: int = 1,
    d_c: Optional[np.ndarray] = None,
    learned: Optional[LearnedDose] = None,
) -> Any:
    if kind == "linear":
        return LinearDose()
    if kind == "exp":
        if d_c is None:
            d_c = np.full(int(n_exp_modes), 1.0)
        return ExpDose(d_c=np.asarray(d_c, dtype=np.float64).reshape(-1))
    if kind == "learned":
        if learned is None:
            raise ValueError("learned dose model requires fitted coefficients")
        return learned
    raise ValueError(f"unknown dose model {kind!r}")


def fit_exp_time_constants(
    y: np.ndarray,
    dose: np.ndarray,
    weights: np.ndarray,
    n_modes: int = 1,
) -> np.ndarray:
    """Rough grid search for ``D_c`` from pooled residuals vs dose."""
    d = np.asarray(dose, dtype=np.float64).reshape(-1)
    r = np.asarray(y, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    grid = np.geomspace(max(float(np.nanmax(d)) * 0.05, 1e-3), max(float(np.nanmax(d)) * 2.0, 1.0), 16)
    best = np.full(int(n_modes), float(np.nanmedian(d)) if d.size else 1.0)
    best_loss = np.inf
    for dc in grid:
        g = 1.0 - np.exp(-d / dc)
        if float(np.std(g)) < 1e-8:
            continue
        # Simple projection.
        num = float(np.sum(w * r * g))
        den = float(np.sum(w * g * g))
        pred = (num / den if den > 0 else 0.0) * g
        loss = float(np.sum(w * (r - pred) ** 2))
        if loss < best_loss:
            best_loss = loss
            best[:] = dc
    if n_modes > 1:
        best[1:] = best[0] * np.linspace(0.4, 2.5, n_modes)[1:]
    return best


def fit_learned_profiles(
    bin_means: np.ndarray,
    bin_centers: np.ndarray,
    weights: np.ndarray,
    rank: int,
    n_knots: int = 6,
    n_boot: int = 20,
    rng: Optional[np.random.Generator] = None,
) -> LearnedDose:
    """Low-rank smooth scores vs dose-bin centre. ``bin_means`` is (n_bins, H) or (n_bins,)."""
    rng = rng or np.random.default_rng(0)
    y = np.asarray(bin_means, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    n_bins = y.shape[0]
    rank = max(1, min(int(rank), n_bins))
    xc = np.asarray(bin_centers, dtype=np.float64).reshape(-1)
    knots = open_uniform_knots(float(xc.min()), float(xc.max()), n_knots)
    design = bspline_design_numpy(xc, knots)
    # SVD of weighted bin × reflection matrix, then project scores onto the spline.
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim == 1:
        w = w.reshape(-1, 1)
    mat = y * np.sqrt(np.maximum(w, 0.0))
    u, s, vt = np.linalg.svd(mat, full_matrices=False)
    z = u[:, :rank] * s[:rank]
    # Least squares: design @ coef ≈ z
    coef, *_ = np.linalg.lstsq(design, z, rcond=None)
    bands = np.zeros((n_boot, n_knots, rank), dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, y.shape[1], size=y.shape[1])
        mat_b = y[:, idx] * np.sqrt(np.maximum(w[:, idx] if w.shape[1] == y.shape[1] else w, 0.0))
        ub, sb, _ = np.linalg.svd(mat_b, full_matrices=False)
        zb = ub[:, :rank] * sb[:rank]
        cb, *_ = np.linalg.lstsq(design, zb, rcond=None)
        bands[b] = cb
    return LearnedDose(knots=knots, coef=coef, bands=bands)


def permute_dose_bins_within_reflection(
    bins: np.ndarray,
    obs_h: np.ndarray,
    phi: Optional[np.ndarray],
    rng: np.random.Generator,
    angle_tol: float = 0.35,
) -> np.ndarray:
    """Permute dose-bin labels among observations of the same reflection.

    When ``phi`` is given, only shuffle among similar rotation angles (D6).
    """
    out = np.array(bins, copy=True)
    h = np.asarray(obs_h, dtype=np.int64)
    for hi in np.unique(h):
        idx = np.flatnonzero(h == hi)
        if idx.size < 2:
            continue
        if phi is None:
            out[idx] = bins[rng.permutation(idx)]
            continue
        ang = np.asarray(phi, dtype=np.float64)[idx]
        used = np.zeros(idx.size, dtype=bool)
        for j, i0 in enumerate(idx):
            if used[j]:
                continue
            close = np.abs(((ang - ang[j] + np.pi) % (2 * np.pi)) - np.pi) <= angle_tol
            group = idx[close & ~used]
            if group.size >= 2:
                out[group] = bins[rng.permutation(group)]
                used[close] = True
    return out
