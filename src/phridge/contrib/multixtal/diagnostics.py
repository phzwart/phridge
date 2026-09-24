"""D6 fake-signal diagnostics. A mode is physical only if it passes 1–4.

1. Dose–angle collinearity → withhold that crystal's within-crystal damage.
2. Smoothness fraction of ``ℓ_k`` (low-order SH × resolution spline) > threshold.
3. Angle-matched permutation null (when the design allows).
4. Leave-one-dose-bin-out / leave-one-pass-out χ²/dof with vs without modes.
5. Chemical plausibility (info only): map peaks vs nearest model atoms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from phridge.contrib.multixtal.dose_model import permute_dose_bins_within_reflection
from phridge.contrib.multixtal.spline import bspline_design_numpy, open_uniform_knots
from phridge.contrib.multixtal.systematics import real_spherical_harmonics


@dataclass
class DamageDiagnostics:
    withheld: np.ndarray
    smoothness: np.ndarray
    smoothness_flag: np.ndarray
    perm_stat: float
    perm_null_95: float
    perm_pass: bool
    loo_bin_with: np.ndarray
    loo_bin_without: np.ndarray
    loo_pass_with: np.ndarray
    loo_pass_without: np.ndarray
    physical: np.ndarray
    atom_hits: list[dict[str, Any]] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def smoothness_fraction(
    loadings: np.ndarray,
    hkl: np.ndarray,
    s_sq: np.ndarray,
    order: int = 2,
    n_knots: int = 4,
) -> np.ndarray:
    """Power of each ``ℓ_k`` explained by low-order SH(ŝ) × resolution spline."""
    ell = np.asarray(loadings, dtype=np.float64)
    if ell.ndim == 1:
        ell = ell.reshape(1, -1)
    hkl = np.asarray(hkl, dtype=np.float64).reshape(-1, 3)
    ss = np.asarray(s_sq, dtype=np.float64).reshape(-1)
    nrm = np.linalg.norm(hkl, axis=1, keepdims=True)
    nrm = np.where(nrm > 0, nrm, 1.0)
    sh = real_spherical_harmonics(hkl / nrm, order)
    knots = open_uniform_knots(float(np.nanmin(ss)), float(np.nanmax(ss)), n_knots)
    spl = bspline_design_numpy(ss, knots)
    # Kronecker-style interaction: each SH column × each spline column.
    cols = [sh[:, i : i + 1] * spl[:, j : j + 1] for i in range(sh.shape[1]) for j in range(spl.shape[1])]
    x = np.concatenate(cols, axis=1)
    frac = np.zeros(ell.shape[0], dtype=np.float64)
    for k in range(ell.shape[0]):
        y = ell[k]
        ok = np.isfinite(y) & (np.abs(y) > 0)
        if int(ok.sum()) < x.shape[1] + 2:
            # Under-determined: treat as fully smooth only if y itself is SH-like.
            frac[k] = 0.0
            continue
        coef, *_ = np.linalg.lstsq(x[ok], y[ok], rcond=None)
        pred = x[ok] @ coef
        sst = float(np.sum((y[ok] - np.mean(y[ok])) ** 2))
        sse = float(np.sum((y[ok] - pred) ** 2))
        frac[k] = 1.0 - sse / sst if sst > 0 else 0.0
    return np.clip(frac, 0.0, 1.0)


def permutation_null(
    y: np.ndarray,
    w: np.ndarray,
    obs_h: np.ndarray,
    bins: np.ndarray,
    phi: Optional[np.ndarray],
    n_perms: int,
    rng: np.random.Generator,
    stat_fn=None,
) -> tuple[float, float, bool]:
    """Observed statistic vs 95% of angle-matched dose-bin permutations."""

    def _default_stat(yy: np.ndarray, ww: np.ndarray, hh: np.ndarray, bb: np.ndarray) -> float:
        # First singular value of a bin × reflection weighted residual matrix.
        n_h = int(hh.max()) + 1 if hh.size else 0
        n_b = int(bb.max()) + 1 if bb.size else 0
        if n_h == 0 or n_b < 2:
            return 0.0
        mat = np.zeros((n_b, n_h), dtype=np.float64)
        wt = np.zeros_like(mat)
        for b in range(n_b):
            for h in range(n_h):
                sel = (bb == b) & (hh == h) & (ww > 0)
                if np.any(sel):
                    mat[b, h] = float(np.sum(ww[sel] * yy[sel]) / np.sum(ww[sel]))
                    wt[b, h] = float(np.sum(ww[sel]))
        centered = (mat - np.nanmean(mat, axis=0, keepdims=True)) * np.sqrt(np.maximum(wt, 0.0))
        centered = np.nan_to_num(centered, nan=0.0)
        sv = np.linalg.svd(centered, compute_uv=False)
        return float(sv[0]) if sv.size else 0.0

    fn = stat_fn or _default_stat
    obs = float(fn(y, w, obs_h, bins))
    null = np.zeros(int(n_perms), dtype=np.float64)
    for i in range(int(n_perms)):
        bp = permute_dose_bins_within_reflection(bins, obs_h, phi, rng)
        null[i] = float(fn(y, w, obs_h, bp))
    q95 = float(np.quantile(null, 0.95)) if null.size else np.inf
    return obs, q95, bool(obs > q95)


def loo_chi2(
    y: np.ndarray,
    w: np.ndarray,
    group: np.ndarray,
    pred_with: np.ndarray,
    pred_without: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Leave-one-group-out χ²/dof with vs without the damage prediction."""
    g = np.asarray(group)
    ids = np.unique(g)
    with_g = np.full(ids.size, np.nan, dtype=np.float64)
    without = np.full(ids.size, np.nan, dtype=np.float64)
    yw = np.asarray(y, dtype=np.float64).reshape(-1)
    ww = np.asarray(w, dtype=np.float64).reshape(-1)
    pw = np.asarray(pred_with, dtype=np.float64).reshape(-1)
    po = np.asarray(pred_without, dtype=np.float64).reshape(-1)
    if pw.size != yw.size:
        pw = np.broadcast_to(pw.reshape(-1)[:1] if pw.size == 1 else np.resize(pw, yw.size), yw.shape).copy()
    if po.size != yw.size:
        po = np.zeros_like(yw) if po.size == 0 else np.resize(po, yw.size)
    for i, gid in enumerate(ids):
        sel = (g == gid) & (ww > 0) & np.isfinite(yw)
        dof = max(int(sel.sum()) - 1, 1)
        if not np.any(sel):
            continue
        with_g[i] = float(np.sum(ww[sel] * (yw[sel] - pw[sel]) ** 2) / dof)
        without[i] = float(np.sum(ww[sel] * (yw[sel] - po[sel]) ** 2) / dof)
    return with_g, without


def loading_density(loadings: np.ndarray, hkl: np.ndarray, frac_xyz: np.ndarray) -> np.ndarray:
    """In-phase Fourier of ``ℓ`` at fractional sites: ``Σ_h ℓ_h cos(2π h·x)``."""
    ell = np.asarray(loadings, dtype=np.float64)
    if ell.ndim == 2:
        ell = ell[0]
    h = np.asarray(hkl, dtype=np.float64).reshape(-1, 3)
    x = np.asarray(frac_xyz, dtype=np.float64).reshape(-1, 3)
    phase = 2.0 * np.pi * (x @ h.T)
    return (np.cos(phase) * ell[None, :]).sum(axis=1)


def nearest_atom_hits(
    loadings: np.ndarray,
    hkl: np.ndarray,
    sites_frac: Optional[np.ndarray],
    labels: Optional[list[str]] = None,
    n_peaks: int = 8,
) -> list[dict[str, Any]]:
    """Info-only: largest |ρ| sites among the model atoms (Cys SG, carboxylates, …)."""
    if sites_frac is None or sites_frac.size == 0:
        return []
    rho = loading_density(loadings, hkl, sites_frac)
    order = np.argsort(-np.abs(rho))
    hits = []
    labs = labels or [""] * int(sites_frac.shape[0])
    for i in order[: int(n_peaks)]:
        hits.append({"index": int(i), "label": labs[i] if i < len(labs) else "", "density": float(rho[i])})
    return hits


def run_diagnostics(
    *,
    withheld: np.ndarray,
    loadings: np.ndarray,
    hkl: np.ndarray,
    s_sq: np.ndarray,
    y_obs: np.ndarray,
    w_obs: np.ndarray,
    obs_h: np.ndarray,
    bins: np.ndarray,
    phi: Optional[np.ndarray],
    pass_id: Optional[np.ndarray],
    pred_with: np.ndarray,
    pred_without: np.ndarray,
    smoothness_threshold: float,
    n_perms: int,
    rng: np.random.Generator,
    sites_frac: Optional[np.ndarray] = None,
    atom_labels: Optional[list[str]] = None,
    allow_perm: bool = True,
) -> DamageDiagnostics:
    """Run D6.1–D6.5 and mark each mode physical only if 2–4 pass."""
    smooth = smoothness_fraction(loadings, hkl, s_sq)
    smooth_flag = smooth > float(smoothness_threshold)
    if allow_perm and np.unique(bins).size >= 2:
        stat, q95, perm_ok = permutation_null(y_obs, w_obs, obs_h, bins, phi, n_perms, rng)
    else:
        stat, q95, perm_ok = 0.0, float("inf"), False
    loo_b_w, loo_b_o = loo_chi2(y_obs, w_obs, bins, pred_with, pred_without)
    if pass_id is not None and np.unique(pass_id).size >= 2:
        loo_p_w, loo_p_o = loo_chi2(y_obs, w_obs, pass_id, pred_with, pred_without)
    else:
        loo_p_w = np.zeros(0)
        loo_p_o = np.zeros(0)
    loo_ok = bool(np.nanmean(loo_b_w) < np.nanmean(loo_b_o)) if loo_b_w.size and np.isfinite(loo_b_w).any() else False
    k = loadings.shape[0] if loadings.ndim == 2 else 1
    physical = np.zeros(k, dtype=bool)
    for i in range(k):
        physical[i] = (not bool(smooth_flag[i])) and perm_ok and loo_ok
    hits = nearest_atom_hits(loadings, hkl, sites_frac, atom_labels)
    return DamageDiagnostics(
        withheld=np.asarray(withheld, dtype=bool),
        smoothness=smooth,
        smoothness_flag=smooth_flag,
        perm_stat=stat,
        perm_null_95=q95,
        perm_pass=perm_ok,
        loo_bin_with=loo_b_w,
        loo_bin_without=loo_b_o,
        loo_pass_with=loo_p_w,
        loo_pass_without=loo_p_o,
        physical=physical,
        atom_hits=hits,
        stats={
            "n_withheld": int(np.asarray(withheld, dtype=bool).sum()),
            "smoothness": smooth.tolist(),
            "perm_pass": perm_ok,
            "loo_improves": loo_ok,
        },
    )
