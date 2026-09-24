"""Observation-level systematics: batch scale/B, absorption, optional partiality.

These nuisances are smooth in reciprocal space and in batch. They cannot
represent a reflection-specific pattern — that is the job of the damage modes.
Absorption is a low-order real spherical-harmonic surface in the incident and
diffracted beam directions (AIMLESS/DIALS-style), common to the crystal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from phridge.contrib.multixtal.spline import bspline_design_numpy, open_uniform_knots


def real_spherical_harmonics(dirs: np.ndarray, order: int) -> np.ndarray:
    """Real SH basis up to ``order`` on unit directions ``(N, 3)``. Column 0 is 1."""
    u = np.asarray(dirs, dtype=np.float64)
    nrm = np.linalg.norm(u, axis=1, keepdims=True)
    nrm = np.where(nrm > 0, nrm, 1.0)
    x, y, z = (u / nrm).T
    cols = [np.ones(x.size)]
    if order >= 1:
        cols.extend([x, y, z])
    if order >= 2:
        cols.extend([x * y, y * z, z * x, x * x - y * y, 3.0 * z * z - 1.0])
    if order >= 3:
        cols.extend([x * y * z, z * (x * x - y * y), x * (x * x - 3.0 * y * y), y * (3.0 * x * x - y * y)])
    if order >= 4:
        cols.extend([x * x * y * y, y * y * z * z, z * z * x * x])
    return np.stack(cols, axis=1)


@dataclass
class FittedSystematics:
    log_scale_coef: np.ndarray
    b_coef: np.ndarray
    knots: np.ndarray
    sh_inc: np.ndarray
    sh_dif: np.ndarray
    wedge_coef: np.ndarray
    order: int

    def scale(
        self,
        s_sq: Any,
        s_inc: Any,
        s_dif: Any,
        batch: Any,
        wedge: Any = None,
        dose: Any = None,
    ) -> np.ndarray:
        x = np.asarray(dose if dose is not None else batch, dtype=np.float64).reshape(-1)
        ss = np.asarray(s_sq, dtype=np.float64).reshape(-1)
        design = bspline_design_numpy(x, self.knots)
        log_k = design @ self.log_scale_coef
        b = design @ self.b_coef
        abs_inc = real_spherical_harmonics(np.asarray(s_inc, dtype=np.float64), self.order) @ self.sh_inc
        abs_dif = real_spherical_harmonics(np.asarray(s_dif, dtype=np.float64), self.order) @ self.sh_dif
        part = 0.0
        if wedge is not None and self.wedge_coef.size:
            w = np.asarray(wedge, dtype=np.float64).reshape(-1)
            part = self.wedge_coef[0] * (w - 0.5) + (self.wedge_coef[1] * (w - 0.5) ** 2 if self.wedge_coef.size > 1 else 0.0)
        return np.exp(log_k - b * ss / 4.0 + abs_inc + abs_dif + part)


class SmoothSystematics:
    """Phase-1 ``SystematicsModel``: spline batch scale/B + SH absorption."""

    def __init__(self, fitted: FittedSystematics) -> None:
        self.fitted = fitted

    def scale(self, s_sq: Any, s_inc: Any, s_dif: Any, batch: Any, wedge: Any = None) -> np.ndarray:
        return self.fitted.scale(s_sq, s_inc, s_dif, batch, wedge=wedge)


def fit_systematics(
    i: np.ndarray,
    sig: np.ndarray,
    s_sq: np.ndarray,
    s_inc: np.ndarray,
    s_dif: np.ndarray,
    batch: np.ndarray,
    expected_i: np.ndarray,
    *,
    dose: Optional[np.ndarray] = None,
    wedge: Optional[np.ndarray] = None,
    order: int = 4,
    n_knots: int = 6,
) -> FittedSystematics:
    """Weighted log-linear fit of A_sys so ``log(I / E[I]) ≈ log A_sys``.

    The design is only batch-smooth and angular-smooth — no per-hkl columns.
    """
    y = np.log(np.maximum(np.asarray(i, dtype=np.float64), 1e-8)) - np.log(np.maximum(np.asarray(expected_i, dtype=np.float64), 1e-8))
    y = np.clip(np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0), -20.0, 20.0)
    rel = np.asarray(sig, dtype=np.float64) ** 2 / np.maximum(np.asarray(i, dtype=np.float64) ** 2, 1e-8)
    w = 1.0 / np.maximum(rel, 1e-6)
    w = np.nan_to_num(w, nan=1.0, posinf=1.0, neginf=0.0)
    mean_w = float(np.mean(w))
    w = np.ones_like(w) if (not np.isfinite(mean_w) or mean_w <= 0) else w / mean_w
    w = np.clip(w, 0.0, 50.0)
    x = np.asarray(dose if dose is not None else batch, dtype=np.float64).reshape(-1)
    ss = np.asarray(s_sq, dtype=np.float64).reshape(-1)
    knots = open_uniform_knots(float(np.nanmin(x)), float(np.nanmax(x)), n_knots)
    b_scale = bspline_design_numpy(x, knots)
    sh_i = real_spherical_harmonics(s_inc, order)
    sh_d = real_spherical_harmonics(s_dif, order)
    # Absorption SH: drop the constant (already in the spline intercept).
    sh_i = sh_i[:, 1:] if sh_i.shape[1] > 1 else sh_i
    sh_d = sh_d[:, 1:] if sh_d.shape[1] > 1 else sh_d
    blocks = [b_scale, -ss[:, None] / 4.0 * b_scale, sh_i, sh_d]
    if wedge is not None:
        wv = np.asarray(wedge, dtype=np.float64).reshape(-1) - 0.5
        blocks.append(np.stack([wv, wv**2], axis=1))
    xmat = np.clip(np.nan_to_num(np.concatenate(blocks, axis=1), nan=0.0, posinf=0.0, neginf=0.0), -1.0e3, 1.0e3)
    xw = np.nan_to_num(xmat * w[:, None], nan=0.0, posinf=0.0, neginf=0.0)
    with np.errstate(all="ignore"):
        gram = xw.T @ xmat
        rhs = xw.T @ y
    # Second-difference penalty on the two spline blocks.
    pen = np.zeros_like(gram)
    for start, n in ((0, n_knots), (n_knots, n_knots)):
        for j in range(start + 1, start + n - 1):
            pen[j, j] += 2.0
            pen[j - 1, j] -= 1.0
            pen[j, j - 1] -= 1.0
            pen[j + 1, j] -= 1.0
            pen[j, j + 1] -= 1.0
    gram = np.nan_to_num(gram, nan=0.0, posinf=0.0, neginf=0.0)
    rhs = np.nan_to_num(rhs, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        coef = np.linalg.solve(gram + 1e-4 * np.eye(gram.shape[0]) + 0.1 * pen, rhs)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(xmat * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
    coef = np.nan_to_num(coef, nan=0.0, posinf=0.0, neginf=0.0)
    n_sh = sh_i.shape[1]
    log_c = coef[:n_knots]
    b_c = coef[n_knots : 2 * n_knots]
    sh_ic = coef[2 * n_knots : 2 * n_knots + n_sh]
    sh_dc = coef[2 * n_knots + n_sh : 2 * n_knots + 2 * n_sh]
    # Restore the dropped l=0 column as zero.
    sh_inc = np.concatenate([[0.0], sh_ic]) if sh_ic.size else np.zeros(1)
    sh_dif = np.concatenate([[0.0], sh_dc]) if sh_dc.size else np.zeros(1)
    rest = coef[2 * n_knots + 2 * n_sh :]
    return FittedSystematics(
        log_scale_coef=log_c,
        b_coef=b_c,
        knots=knots,
        sh_inc=sh_inc,
        sh_dif=sh_dif,
        wedge_coef=rest if rest.size else np.zeros(0),
        order=int(order),
    )
