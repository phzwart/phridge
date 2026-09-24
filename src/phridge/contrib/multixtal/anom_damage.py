"""Anomalous-site occupancy vs dose: ``q_s(D) = q_{s0} exp(−D / D_s)``.

Reuse ``fit_site_q`` per crystal × dose bin on pairs assigned by mean dose,
after removing the D3 dose-difference term from the observation residuals.
Per-crystal ``q_{s0}``, shared half-dose ``D_s``. Linear fallback when the
exponential is unidentified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from phridge.contrib.multixtal.anomalous import fit_site_q


@dataclass
class AnomDamageFit:
    q0: np.ndarray
    d_s: float
    half_dose: float
    q_bin: np.ndarray
    d_bin: np.ndarray
    q_se_bin: np.ndarray
    linear_slope: float
    used_exp: bool
    stats: dict = field(default_factory=dict)


def _bin_pairs(
    d_i: np.ndarray,
    i_bar: np.ndarray,
    both: np.ndarray,
    mean_dose: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack crystal×bin slices as extra 'datasets' for ``fit_site_q``."""
    n_data, n_h = d_i.shape
    # Value-quantile bins (not rank bins): a constant dose stays in one bin.
    flat = np.asarray(mean_dose, dtype=np.float64)
    finite = flat[np.isfinite(flat)]
    if finite.size:
        edges = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, int(n_bins) + 1)))
        if edges.size == 1:
            bins = np.zeros_like(flat, dtype=np.int64)
        else:
            bins = np.clip(np.digitize(flat, edges[1:-1], right=True), 0, int(n_bins) - 1)
    else:
        bins = np.zeros_like(flat, dtype=np.int64)
    rows_di: list[np.ndarray] = []
    rows_ib: list[np.ndarray] = []
    rows_both: list[np.ndarray] = []
    rows_d: list[float] = []
    rows_crystal: list[int] = []
    for d in range(n_data):
        for b in range(n_bins):
            sel = bins[d] == b
            if int(sel.sum()) < 4:
                continue
            di = np.where(sel, d_i[d], 0.0)
            ib = np.where(sel, i_bar[d], 0.0)
            bt = both[d] & sel
            if int(bt.sum()) < 4:
                continue
            rows_di.append(di)
            rows_ib.append(ib)
            rows_both.append(bt)
            rows_d.append(float(np.nanmean(mean_dose[d, sel])))
            rows_crystal.append(d)
    if not rows_di:
        empty = np.zeros((0, n_h))
        return empty, empty, np.zeros((0, n_h), dtype=bool), np.zeros(0), np.zeros(0, dtype=np.int64)
    return (
        np.stack(rows_di, axis=0),
        np.stack(rows_ib, axis=0),
        np.stack(rows_both, axis=0),
        np.asarray(rows_d, dtype=np.float64),
        np.asarray(rows_crystal, dtype=np.int64),
    )


def _fit_exp_decay(q: np.ndarray, dose: np.ndarray, crystal: np.ndarray, n_crystal: int) -> tuple[np.ndarray, float, bool]:
    """``q ≈ q0[d] * exp(−D / Ds)``. Returns ``(q0, Ds, used_exp)``."""
    q = np.asarray(q, dtype=np.float64)
    d = np.asarray(dose, dtype=np.float64)
    c = np.asarray(crystal, dtype=np.int64)
    q0 = np.zeros(n_crystal, dtype=np.float64)
    ok = np.isfinite(q) & np.isfinite(d) & (np.abs(q) > 1e-12)
    counts = np.array([int(np.sum((c == i) & ok)) for i in range(n_crystal)])
    if int(ok.sum()) < 3:
        for i in range(n_crystal):
            sel = (c == i) & np.isfinite(q)
            q0[i] = float(np.nanmean(q[sel])) if np.any(sel) else 0.0
        return q0, float("inf"), False
    d_max = max(float(np.nanmax(d[ok])), 1e-3)
    grid = np.geomspace(0.05 * d_max, 4.0 * d_max, 24)
    # Per-crystal q0 is unidentified when each crystal has a single dose; share q0.
    share_q0 = bool(np.max(counts) < 2)
    best_ds = d_max
    best_loss = np.inf
    best_q0 = q0.copy()
    for ds in grid:
        g_all = np.exp(-d[ok] / ds)
        if share_q0:
            num = float(np.sum(q[ok] * g_all))
            den = float(np.sum(g_all * g_all))
            shared = num / den if den > 0 else 0.0
            cand = np.full(n_crystal, shared)
            loss = float(np.sum((q[ok] - shared * g_all) ** 2))
        else:
            cand = np.zeros(n_crystal, dtype=np.float64)
            loss = 0.0
            for i in range(n_crystal):
                sel = (c == i) & ok
                if not np.any(sel):
                    continue
                g = np.exp(-d[sel] / ds)
                num = float(np.sum(q[sel] * g))
                den = float(np.sum(g * g))
                cand[i] = num / den if den > 0 else 0.0
                loss += float(np.sum((q[sel] - cand[i] * g) ** 2))
        if loss < best_loss:
            best_loss = loss
            best_ds = float(ds)
            best_q0 = cand
    # Intercept-only baseline (no decay). Shared or per-crystal.
    if share_q0:
        lin_q0 = np.full(n_crystal, float(np.mean(q[ok])))
        lin_loss = float(np.sum((q[ok] - lin_q0[0]) ** 2))
    else:
        lin_loss = 0.0
        lin_q0 = np.zeros(n_crystal, dtype=np.float64)
        for i in range(n_crystal):
            sel = (c == i) & np.isfinite(q)
            if not np.any(sel):
                continue
            lin_q0[i] = float(np.nanmean(q[sel]))
            lin_loss += float(np.sum((q[sel] - lin_q0[i]) ** 2))
    if best_loss > 0.95 * lin_loss:
        return lin_q0, float("inf"), False
    return best_q0, best_ds, True


def fit_anom_site_decay(
    d_i: np.ndarray,
    i_bar: np.ndarray,
    both: np.ndarray,
    signature: np.ndarray,
    mean_dose: np.ndarray,
    var_ratio: np.ndarray,
    sigma_a: np.ndarray,
    strong: np.ndarray,
    n_bins: int = 6,
    n_crystal: Optional[int] = None,
) -> AnomDamageFit:
    """Fit ``q_s(D)`` after D3 has been stripped from the Bijvoet differences."""
    di_b, ib_b, both_b, d_bin, crystal = _bin_pairs(d_i, i_bar, both, mean_dose, n_bins)
    n_c = int(n_crystal if n_crystal is not None else d_i.shape[0])
    if di_b.shape[0] == 0:
        return AnomDamageFit(
            q0=np.zeros(n_c),
            d_s=float("inf"),
            half_dose=float("inf"),
            q_bin=np.zeros(0),
            d_bin=np.zeros(0),
            q_se_bin=np.zeros(0),
            linear_slope=0.0,
            used_exp=False,
            stats={"n_bins_used": 0},
        )
    sa = np.asarray(sigma_a, dtype=np.float64)
    if sa.ndim == 2:
        sa_use = np.stack([sa[min(int(c), sa.shape[0] - 1)] for c in crystal], axis=0)
    else:
        sa_use = np.broadcast_to(sa, (di_b.shape[0], di_b.shape[1])).copy()
    vr = np.asarray(var_ratio, dtype=np.float64)
    if vr.ndim == 2:
        vr_use = np.stack([vr[min(int(c), vr.shape[0] - 1)] for c in crystal], axis=0)
    else:
        vr_use = np.broadcast_to(vr, di_b.shape).copy()
    q_bin, q_se = fit_site_q(di_b, ib_b, both_b, signature, vr_use, sa_use, strong)
    q0, d_s, used_exp = _fit_exp_decay(q_bin, d_bin, crystal, n_c)
    half = float(d_s * np.log(2.0)) if np.isfinite(d_s) else float("inf")
    slope = 0.0
    ok = np.isfinite(q_bin) & np.isfinite(d_bin)
    if int(ok.sum()) >= 2:
        slope = float(np.polyfit(d_bin[ok], q_bin[ok], 1)[0])
    return AnomDamageFit(
        q0=q0,
        d_s=float(d_s),
        half_dose=half,
        q_bin=q_bin,
        d_bin=d_bin,
        q_se_bin=q_se,
        linear_slope=slope,
        used_exp=used_exp,
        stats={"n_bins_used": int(q_bin.size), "n_crystal": n_c},
    )


def corrected_bijvoet(
    d_e2: np.ndarray,
    c_h: np.ndarray,
    damage_delta: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Anomalous intercept after removing the D3 dose-difference leak."""
    out = np.asarray(d_e2, dtype=np.float64) - np.asarray(c_h, dtype=np.float64)[None, :]
    if damage_delta is not None:
        out = out - np.asarray(damage_delta, dtype=np.float64)
    return out
