"""DamageTerm: within- and between-crystal WLS for dose loadings ``ℓ_{k,h}``.

Per strong ``h`` the observation model is

    y_o = μ_h + z_{d(o)}ᵀ L_h + Σ_k g_k(ρ_d D_o) ℓ_{k,h}
        + s_o (c_h + z_{d(o)}ᵀ B_h) + ε_o

Two estimators are always computed:

* **Within-crystal** — crystal×reflection intercept ``a_{dh}`` absorbs the
  conformational mean; ``ℓ`` comes from dose contrasts inside a crystal.
  Crystals that fail the dose–angle confound contribute only an intercept.
* **Between-crystal** — crystal-mean ``y`` is regressed on ``mean_o g_k(D_o)``.

Inverse-covariance combination is used only when a per-reflection χ² test of
the difference agrees; otherwise the within-crystal estimate is kept (or the
between-crystal estimate when within-crystal has no dose contrast).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from phridge.contrib.multixtal.residuals import normalized_residuals


@dataclass
class DamageFit:
    loadings: np.ndarray
    loadings_within: np.ndarray
    loadings_between: np.ndarray
    se: np.ndarray
    se_within: np.ndarray
    se_between: np.ndarray
    intercept_adh: np.ndarray
    c_h: np.ndarray
    b_h: np.ndarray
    chi2_agree: np.ndarray
    used_combine: np.ndarray
    withheld: np.ndarray
    g_mean: np.ndarray
    zero_dose_i: np.ndarray
    zero_dose_se: np.ndarray
    rho: np.ndarray
    profiles: np.ndarray
    y_obs: np.ndarray
    w_obs: np.ndarray
    stats: dict = field(default_factory=dict)

    def apply(self, profiles: np.ndarray, loadings: Optional[np.ndarray] = None) -> np.ndarray:
        ell = self.loadings if loadings is None else loadings
        return np.asarray(profiles, dtype=np.float64) @ np.asarray(ell, dtype=np.float64)


class LinearDamage:
    """Phase-1 ``DamageTerm``: ``profiles @ loadings``."""

    def apply(self, profiles: np.ndarray, loadings: np.ndarray) -> np.ndarray:
        return np.asarray(profiles, dtype=np.float64) @ np.asarray(loadings, dtype=np.float64)


def observation_residuals(
    i: np.ndarray,
    sig: np.ndarray,
    h: np.ndarray,
    crystal: np.ndarray,
    a_sys: np.ndarray,
    nuisances: list,
    epsilon: np.ndarray,
    centric: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Observation-level ``(y, W)`` in the same units as the crystal-mean residuals."""
    i = np.asarray(i, dtype=np.float64).reshape(-1)
    sig = np.asarray(sig, dtype=np.float64).reshape(-1)
    h = np.asarray(h, dtype=np.int64).reshape(-1)
    crystal = np.asarray(crystal, dtype=np.int64).reshape(-1)
    a_sys = np.asarray(a_sys, dtype=np.float64).reshape(-1)
    n = i.size
    y = np.zeros(n, dtype=np.float64)
    w = np.zeros(n, dtype=np.float64)
    for d, item in enumerate(nuisances):
        sel = crystal == d
        if not np.any(sel):
            continue
        hi = h[sel]
        n_h = int(item.scale_a.shape[0])
        ok = (hi >= 0) & (hi < n_h)
        if not np.any(ok):
            continue
        idx = np.flatnonzero(sel)[ok]
        hi = h[idx]
        i_d = i[idx][None, :]
        sig2 = (sig[idx] ** 2)[None, :]
        obs = np.ones((1, idx.size), dtype=bool)
        yd, wm, _wme, _e2, _sa = normalized_residuals(
            i_d,
            sig2,
            obs,
            item.f_c_abs2[hi][None, :],
            epsilon[hi][None, :],
            centric[hi],
            (item.scale_a[hi] * a_sys[idx])[None, :],
            item.alpha[hi][None, :],
            item.beta[hi][None, :],
            item.sigma_p[hi][None, :],
        )
        y[idx] = yd[0]
        w[idx] = wm[0]
    return y, w


def _wls(x: np.ndarray, y: np.ndarray, w: np.ndarray, ridge: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(coef, var_diag)`` of a weighted least-squares fit."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    w = np.maximum(np.asarray(w, dtype=np.float64).reshape(-1), 0.0)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.shape[0] == 0 or x.shape[1] == 0:
        return np.zeros(x.shape[1] if x.ndim == 2 else 0), np.full(x.shape[1] if x.ndim == 2 else 0, np.inf)
    xw = x * w[:, None]
    gram = xw.T @ x + float(ridge) * np.eye(x.shape[1])
    rhs = xw.T @ y
    try:
        coef = np.linalg.solve(gram, rhs)
        cov = np.linalg.inv(gram)
    except np.linalg.LinAlgError:
        coef, *_ = np.linalg.lstsq(x * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)
        try:
            cov = np.linalg.pinv(gram)
        except np.linalg.LinAlgError:
            cov = np.eye(x.shape[1]) * np.inf
    var = np.maximum(np.diag(cov), 0.0)
    return coef, var


def fit_damage_within(
    y: np.ndarray,
    w: np.ndarray,
    obs_h: np.ndarray,
    crystal: np.ndarray,
    profiles: np.ndarray,
    sign: np.ndarray,
    scores: np.ndarray,
    strong: np.ndarray,
    withheld: np.ndarray,
    n_h: int,
    ridge: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Within-crystal WLS. Returns ``(ℓ, se, a_dh, c_h, B_h)``."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    obs_h = np.asarray(obs_h, dtype=np.int64).reshape(-1)
    crystal = np.asarray(crystal, dtype=np.int64).reshape(-1)
    g = np.asarray(profiles, dtype=np.float64)
    if g.ndim == 1:
        g = g.reshape(-1, 1)
    sign = np.asarray(sign, dtype=np.float64).reshape(-1)
    z = np.asarray(scores, dtype=np.float64)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    n_c = int(z.shape[0]) if z.size else (int(crystal.max()) + 1 if crystal.size else 0)
    rank = int(z.shape[1]) if z.ndim == 2 else 0
    k = int(g.shape[1])
    ell = np.zeros((k, n_h), dtype=np.float64)
    se = np.full((k, n_h), np.inf, dtype=np.float64)
    a_dh = np.zeros((max(n_c, 1), n_h), dtype=np.float64)
    c_h = np.zeros(n_h, dtype=np.float64)
    b_h = np.zeros((rank, n_h), dtype=np.float64)
    withheld = np.asarray(withheld, dtype=bool)
    if withheld.size < n_c:
        withheld = np.resize(withheld, n_c)
    for h in range(n_h):
        if not bool(strong[h]):
            continue
        sel = (obs_h == h) & (w > 0) & np.isfinite(y)
        if int(sel.sum()) < 2:
            continue
        d_ids = crystal[sel]
        uniq, inv = np.unique(d_ids, return_inverse=True)
        n_dum = int(uniq.size)
        g_h = np.array(g[sel], copy=True)
        for j, d in enumerate(uniq):
            if 0 <= int(d) < withheld.size and withheld[int(d)]:
                g_h[inv == j] = 0.0
        blocks = [np.eye(n_dum, dtype=np.float64)[inv], g_h]
        s_h = sign[sel]
        use_sign = np.any(s_h != 0) and (np.unique(np.sign(s_h[s_h != 0])).size > 1)
        if use_sign:
            blocks.append(s_h.reshape(-1, 1))
            if rank:
                z_h = z[np.clip(d_ids, 0, n_c - 1)]
                blocks.append(s_h[:, None] * z_h)
        x = np.concatenate(blocks, axis=1)
        coef, var = _wls(x, y[sel], w[sel], ridge=ridge)
        a_dh[uniq, h] = coef[:n_dum]
        ell[:, h] = coef[n_dum : n_dum + k]
        se[:, h] = np.sqrt(np.maximum(var[n_dum : n_dum + k], 0.0))
        cursor = n_dum + k
        if use_sign:
            c_h[h] = float(coef[cursor])
            cursor += 1
            if rank:
                b_h[:, h] = coef[cursor : cursor + rank]
    return ell, se, a_dh, c_h, b_h


def fit_damage_between(
    y_crystal: np.ndarray,
    w_crystal: np.ndarray,
    scores: np.ndarray,
    loadings: np.ndarray,
    mu: np.ndarray,
    g_mean: np.ndarray,
    strong: np.ndarray,
    ridge: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Between-crystal WLS of ``y − μ − z L`` on ``mean_o g_k``."""
    y = np.asarray(y_crystal, dtype=np.float64)
    w = np.asarray(w_crystal, dtype=np.float64)
    z = np.asarray(scores, dtype=np.float64)
    l = np.asarray(loadings, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    g = np.asarray(g_mean, dtype=np.float64)
    if g.ndim == 1:
        g = g.reshape(-1, 1)
    n_data, n_h = y.shape
    k = int(g.shape[1])
    ell = np.zeros((k, n_h), dtype=np.float64)
    se = np.full((k, n_h), np.inf, dtype=np.float64)
    resid = y - mu[None, :]
    if z.size and l.size:
        resid = resid - z @ l
    for h in range(n_h):
        if not bool(strong[h]):
            continue
        sel = (w[:, h] > 0) & np.isfinite(resid[:, h])
        if int(sel.sum()) < 2:
            continue
        coef, var = _wls(g[sel], resid[sel, h], w[sel, h], ridge=ridge)
        ell[:, h] = coef[:k]
        se[:, h] = np.sqrt(np.maximum(var[:k], 0.0))
    return ell, se


def combine_loadings(
    ell_w: np.ndarray,
    se_w: np.ndarray,
    ell_b: np.ndarray,
    se_b: np.ndarray,
    chi2_cut: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Inverse-covariance combine when the per-reflection χ² agrees.

    Returns ``(ell, se, chi2, used_combine)``. Falls back to within-crystal,
    or to between-crystal when within-crystal has infinite variance.
    """
    ew = np.asarray(ell_w, dtype=np.float64)
    eb = np.asarray(ell_b, dtype=np.float64)
    vw = np.asarray(se_w, dtype=np.float64) ** 2
    vb = np.asarray(se_b, dtype=np.float64) ** 2
    k, n_h = ew.shape
    out = np.array(ew, copy=True)
    se = np.array(se_w, copy=True)
    chi2 = np.full(n_h, np.nan, dtype=np.float64)
    used = np.zeros(n_h, dtype=bool)
    for h in range(n_h):
        finite_w = np.isfinite(vw[:, h]) & (vw[:, h] < 1e8)
        finite_b = np.isfinite(vb[:, h]) & (vb[:, h] < 1e8)
        if not np.any(finite_w) and np.any(finite_b):
            out[:, h] = eb[:, h]
            se[:, h] = se_b[:, h]
            continue
        if not np.any(finite_w) or not np.any(finite_b):
            continue
        diff = ew[:, h] - eb[:, h]
        var = vw[:, h] + vb[:, h]
        var = np.where(var > 0, var, np.inf)
        chi2[h] = float(np.sum(diff**2 / var))
        if chi2[h] <= chi2_cut * max(k, 1) and np.isfinite(chi2[h]):
            prec_w = np.where(vw[:, h] > 0, 1.0 / vw[:, h], 0.0)
            prec_b = np.where(vb[:, h] > 0, 1.0 / vb[:, h], 0.0)
            prec = prec_w + prec_b
            out[:, h] = np.where(prec > 0, (ew[:, h] * prec_w + eb[:, h] * prec_b) / prec, ew[:, h])
            se[:, h] = np.where(prec > 0, 1.0 / np.sqrt(prec), se_w[:, h])
            used[h] = True
    return out, se, chi2, used


def crystal_mean_profiles(
    profiles: np.ndarray,
    crystal: np.ndarray,
    obs_h: Optional[np.ndarray],
    n_crystal: int,
    n_h: Optional[int] = None,
) -> np.ndarray:
    """``mean_o g_k`` per crystal (and optionally per crystal×reflection)."""
    g = np.asarray(profiles, dtype=np.float64)
    if g.ndim == 1:
        g = g.reshape(-1, 1)
    d = np.asarray(crystal, dtype=np.int64).reshape(-1)
    k = g.shape[1]
    if obs_h is None or n_h is None:
        out = np.zeros((n_crystal, k), dtype=np.float64)
        for i in range(n_crystal):
            sel = d == i
            if np.any(sel):
                out[i] = np.mean(g[sel], axis=0)
        return out
    out = np.zeros((n_crystal, int(n_h), k), dtype=np.float64)
    h = np.asarray(obs_h, dtype=np.int64).reshape(-1)
    for i in range(n_crystal):
        for hi in np.unique(h[d == i]):
            if 0 <= int(hi) < int(n_h):
                sel = (d == i) & (h == hi)
                out[i, int(hi)] = np.mean(g[sel], axis=0)
    return out


def zero_dose_intensity(
    a_dh: np.ndarray,
    nuisances: list,
    epsilon: np.ndarray,
    se_adh: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert crystal×reflection intercepts at ``D = 0`` back to intensity."""
    n_c, n_h = a_dh.shape
    i0 = np.zeros((n_c, n_h), dtype=np.float64)
    se = np.full((n_c, n_h), np.nan, dtype=np.float64)
    eps = np.asarray(epsilon, dtype=np.float64).reshape(-1)
    for d, item in enumerate(nuisances):
        if d >= n_c:
            break
        from phridge.contrib.multixtal.reflection_model import sigma_a_from_alpha_beta

        sa = np.maximum(sigma_a_from_alpha_beta(item.alpha, item.beta, item.sigma_p), 1e-6)
        den = item.scale_a * eps * (item.alpha**2 * item.sigma_p + item.beta)
        den = np.maximum(den, 1e-12)
        e_c2 = item.f_c_abs2 / np.maximum(item.sigma_p, 1e-12)
        e2_mean = (sa**2) * e_c2 + (1.0 - sa**2)
        e2_0 = a_dh[d] * sa + e2_mean
        i0[d] = e2_0 * den
        if se_adh is not None:
            se[d] = np.abs(se_adh[d] * sa * den)
    return i0, se


def fit_damage(
    y_obs: np.ndarray,
    w_obs: np.ndarray,
    obs_h: np.ndarray,
    crystal: np.ndarray,
    profiles: np.ndarray,
    sign: np.ndarray,
    scores: np.ndarray,
    loadings: np.ndarray,
    mu: np.ndarray,
    y_crystal: np.ndarray,
    w_crystal: np.ndarray,
    strong: np.ndarray,
    withheld: np.ndarray,
    nuisances: list,
    epsilon: np.ndarray,
    rho: np.ndarray,
    ridge: float = 1e-6,
) -> DamageFit:
    """Fit both estimators, combine, and report zero-dose intensities."""
    n_h = int(y_crystal.shape[1])
    n_c = int(y_crystal.shape[0])
    ell_w, se_w, a_dh, c_h, b_h = fit_damage_within(
        y_obs, w_obs, obs_h, crystal, profiles, sign, scores, strong, withheld, n_h, ridge=ridge
    )
    g_mean = crystal_mean_profiles(profiles, crystal, None, n_c)
    ell_b, se_b = fit_damage_between(y_crystal, w_crystal, scores, loadings, mu, g_mean, strong, ridge=ridge)
    ell, se, chi2, used = combine_loadings(ell_w, se_w, ell_b, se_b)
    i0, i0_se = zero_dose_intensity(a_dh, nuisances, epsilon)
    within_contrast = float(np.nanmean(np.std(profiles, axis=0))) if profiles.size else 0.0
    return DamageFit(
        loadings=ell,
        loadings_within=ell_w,
        loadings_between=ell_b,
        se=se,
        se_within=se_w,
        se_between=se_b,
        intercept_adh=a_dh,
        c_h=c_h,
        b_h=b_h,
        chi2_agree=chi2,
        used_combine=used,
        withheld=np.asarray(withheld, dtype=bool),
        g_mean=g_mean,
        zero_dose_i=i0,
        zero_dose_se=i0_se,
        rho=np.asarray(rho, dtype=np.float64).reshape(-1),
        profiles=np.asarray(profiles, dtype=np.float64),
        y_obs=np.asarray(y_obs, dtype=np.float64),
        w_obs=np.asarray(w_obs, dtype=np.float64),
        stats={
            "n_agree": int(used.sum()),
            "median_chi2_agree": float(np.nanmedian(chi2)) if np.isfinite(chi2).any() else float("nan"),
            "within_contrast": within_contrast,
        },
    )
