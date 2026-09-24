"""Spectral taper and explained-power entropy (spec eqs 15–16).

Both terms are minimised by the uniform field, so regularisation pulls
toward Read's σ_A (Section 5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from phridge.contrib.spatial_sigmaa_v2.fields import FieldBasis
from phridge.contrib.spatial_sigmaa_v2.moments import form_factors, model_temperature, resolution_s2
from phridge.sfcalc.engine.engine import ScatteringModel


@dataclass
class RegulariserResult:
    """Value of the two extra terms in eq. 15 and their field-coefficient grads."""

    taper: float
    entropy: float
    d_lambda_c: np.ndarray
    d_kappa_c: np.ndarray

    @property
    def value(self) -> float:
        return float(self.taper + self.entropy)


def spectral_taper(
    lambda_c: np.ndarray,
    kappa_c: np.ndarray,
    v_k: np.ndarray,
    *,
    scale: float = 1.0,
) -> tuple[float, np.ndarray, np.ndarray]:
    """½ Σ_k |c_k|² / (scale v_k). Coefficients with ``v_k <= 0`` are skipped."""
    lam = np.asarray(lambda_c, dtype=np.float64).reshape(-1)
    kap = np.asarray(kappa_c, dtype=np.float64).reshape(-1)
    v = np.asarray(v_k, dtype=np.float64).reshape(-1) * float(scale)
    d_lam = np.zeros_like(lam)
    d_kap = np.zeros_like(kap)
    if v.size == 0:
        return 0.0, d_lam, d_kap
    ok = v > 0.0
    if not np.any(ok):
        return 0.0, d_lam, d_kap
    value = 0.5 * float(np.sum((lam[ok] ** 2 + kap[ok] ** 2) / v[ok]))
    d_lam[ok] = lam[ok] / v[ok]
    d_kap[ok] = kap[ok] / v[ok]
    return value, d_lam, d_kap


def explained_power_distribution(
    model: ScatteringModel,
    hkl: np.ndarray,
    d_jh: np.ndarray,
    shell_index: np.ndarray,
    n_shells: int,
) -> tuple[np.ndarray, np.ndarray]:
    """p_j(s) of eq. 16 and reflection counts n_s.

    ``p`` has shape (N, S). Empty shells are left as a uniform 1/N.
    """
    s2 = resolution_s2(model.unit_cell, hkl)
    fj = form_factors(model, s2)
    tj = model_temperature(model, hkl, s2)
    d2 = np.asarray(d_jh, dtype=np.float64) ** 2
    # Spec eq. 16 uses d² f² T². Occupancy / multiplicity are absorbed in f
    # when the caller has already folded them in; here f is the table value.
    power_h = d2 * (fj * tj) ** 2  # (N, H)
    n = model.n_scatterers
    p = np.full((n, n_shells), 1.0 / max(n, 1), dtype=np.float64)
    counts = np.zeros(n_shells, dtype=np.float64)
    idx = np.asarray(shell_index, dtype=np.int64).reshape(-1)
    for s in range(n_shells):
        sel = idx == s
        counts[s] = float(np.count_nonzero(sel))
        if not np.any(sel):
            continue
        num = np.mean(power_h[:, sel], axis=1)
        tot = float(np.sum(num))
        if tot <= 1e-30:
            continue
        p[:, s] = num / tot
    return p, counts


def entropy_kl(p: np.ndarray, n_s: np.ndarray, *, weight: float) -> tuple[float, np.ndarray]:
    """α Σ_s n_s KL(p(s) ∥ uniform) and ∂(that)/∂p_j(s)."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-30, 1.0)
    n = p.shape[0]
    log_np = np.log(n * p)
    kl = np.sum(p * log_np, axis=0)  # (S,)
    ns = np.asarray(n_s, dtype=np.float64).reshape(-1)
    value = float(weight) * float(np.sum(ns * kl))
    # ∂KL/∂p_j = log(N p_j) + 1; the simplex constraint is imposed by the
    # caller through n_j → p_j, not here.
    d_p = float(weight) * ns[None, :] * (log_np + 1.0)
    return value, d_p


def _shell_index(s2: np.ndarray, edges: np.ndarray) -> np.ndarray:
    ed = np.asarray(edges, dtype=np.float64).reshape(-1)
    ss = np.asarray(s2, dtype=np.float64).reshape(-1)
    if ed.size < 2:
        return np.zeros(ss.shape[0], dtype=np.int64)
    return np.clip(np.searchsorted(ed, ss, side="right") - 1, 0, ed.size - 2)


def regulariser(
    model: ScatteringModel,
    hkl: np.ndarray,
    d_jh: np.ndarray,
    lambda_c: np.ndarray,
    kappa_c: np.ndarray,
    v_k: np.ndarray,
    *,
    basis: Optional[FieldBasis] = None,
    sites_frac: Optional[np.ndarray] = None,
    shell_s2_edges: Optional[np.ndarray] = None,
    entropy_weight: float = 0.0,
    spectral_taper_scale: float = 1.0,
) -> RegulariserResult:
    """Full extra objective of eq. 15 and grads w.r.t. real field coefficients."""
    taper, d_lam_c, d_kap_c = spectral_taper(
        lambda_c, kappa_c, v_k, scale=spectral_taper_scale
    )
    ent = 0.0
    if entropy_weight > 0.0 and shell_s2_edges is not None and basis is not None:
        s2 = resolution_s2(model.unit_cell, hkl)
        edges = np.asarray(shell_s2_edges, dtype=np.float64).reshape(-1)
        n_shells = max(int(edges.size) - 1, 0)
        if n_shells > 0:
            idx = _shell_index(s2, edges)
            p, counts = explained_power_distribution(model, hkl, d_jh, idx, n_shells)
            ent, d_p = entropy_kl(p, counts, weight=entropy_weight)
            # n_j = mean_h d² f² T², p = n / Σ n. Chain through λ, κ then M.
            sites = model.sites_frac if sites_frac is None else sites_frac
            fj = form_factors(model, s2)
            tj = model_temperature(model, hkl, s2)
            d2 = np.asarray(d_jh, dtype=np.float64) ** 2
            power_h = d2 * (fj * tj) ** 2
            n_at = model.n_scatterers
            d_lambda = np.zeros(n_at, dtype=np.float64)
            d_kappa = np.zeros(n_at, dtype=np.float64)
            for s in range(n_shells):
                sel = idx == s
                if not np.any(sel):
                    continue
                num = np.mean(power_h[:, sel], axis=1)
                tot = float(np.sum(num))
                if tot <= 1e-30:
                    continue
                # ∂KL/∂n_i = (log(N p_i) − KL) / S, then × α n_s from d_p via p.
                # L = α n_s KL, ∂L/∂p_j = d_p[:, s], ∂p_j/∂n_i = (δ_ji − p_j)/S
                dp = d_p[:, s]
                p_s = p[:, s]
                d_n = (dp - np.dot(dp, p_s)) / tot
                d_lambda += d_n * (2.0 * num)
                s2_s = float(np.mean(s2[sel]))
                d_kappa += d_n * (2.0 * num) * (-s2_s / 4.0)
            m = basis.design_matrix(sites)
            d_lam_c = d_lam_c + m.T @ d_lambda
            d_kap_c = d_kap_c + m.T @ d_kappa
    return RegulariserResult(
        taper=taper,
        entropy=ent,
        d_lambda_c=d_lam_c,
        d_kappa_c=d_kap_c,
    )
