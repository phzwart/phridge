"""Model-normalized residuals on the strong set."""

from __future__ import annotations

from typing import Any

import numpy as np

from phridge.contrib.multixtal.reflection_model import sigma_a_from_alpha_beta


def mate_mean(
    i_plus: np.ndarray,
    i_minus: np.ndarray,
    sig_plus: np.ndarray,
    sig_minus: np.ndarray,
    mask_plus: np.ndarray,
    mask_minus: np.ndarray,
    centric: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``Ibar`` = mean of available mates, measurement variance, and observation mask.

    Centrics use the plus mate only (the unique intensity).
    """
    ip = np.asarray(i_plus, dtype=np.float64)
    im = np.asarray(i_minus, dtype=np.float64)
    sp = np.asarray(sig_plus, dtype=np.float64)
    sm = np.asarray(sig_minus, dtype=np.float64)
    mp = np.asarray(mask_plus, dtype=bool)
    mm = np.asarray(mask_minus, dtype=bool)
    cen = np.asarray(centric, dtype=bool)
    n_data, n_h = ip.shape
    i_bar = np.full((n_data, n_h), np.nan, dtype=np.float64)
    sig2 = np.full((n_data, n_h), np.nan, dtype=np.float64)
    both = mp & mm & ~cen[None, :]
    only_p = mp & (~mm | cen[None, :])
    only_m = mm & ~mp & ~cen[None, :]
    i_bar[both] = 0.5 * (ip[both] + im[both])
    sig2[both] = 0.25 * (sp[both] ** 2 + sm[both] ** 2)
    i_bar[only_p] = ip[only_p]
    sig2[only_p] = sp[only_p] ** 2
    i_bar[only_m] = im[only_m]
    sig2[only_m] = sm[only_m] ** 2
    observed = both | only_p | only_m
    sig2 = np.maximum(sig2, 1e-12)
    return i_bar, sig2, observed


def sigma_p_shell(f_c_abs2: np.ndarray, s_sq: np.ndarray, n_shells: int = 10) -> np.ndarray:
    """Shell-smoothed mean of ``|Fc|²`` (``Σ_P``), interpolated back onto reflections."""
    fc2 = np.asarray(f_c_abs2, dtype=np.float64)
    ss = np.asarray(s_sq, dtype=np.float64)
    n = fc2.size
    if n == 0:
        return fc2
    n_shells = max(1, min(int(n_shells), n))
    order = np.argsort(ss, kind="stable")
    bins = np.empty(n, dtype=np.int64)
    bins[order] = (np.arange(n) * n_shells) // n
    means = np.zeros(n_shells, dtype=np.float64)
    for k in range(n_shells):
        sel = bins == k
        vals = fc2[sel]
        vals = vals[np.isfinite(vals) & (vals >= 0)]
        means[k] = float(np.mean(vals)) if vals.size else 1.0
    means = np.maximum(means, 1e-12)
    return means[bins]


def normalized_residuals(
    i_bar: np.ndarray,
    sig2: np.ndarray,
    observed: np.ndarray,
    f_c_abs2: np.ndarray,
    epsilon: np.ndarray,
    centric: np.ndarray,
    scale_a: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    sigma_p: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(y, W_model, W_meas, e2_o, sigma_a)``.

    ``E²_o = Ibar / [A ε (α² Σ_P + β)]``
    ``E[E²|model] = σ_A² E_c² + (1 − σ_A²)``
    ``y = (E²_o − E[E²|model]) / σ_A``
    """
    i = np.asarray(i_bar, dtype=np.float64)
    v_meas = np.maximum(np.asarray(sig2, dtype=np.float64), 1e-12)
    obs = np.asarray(observed, dtype=bool)
    fc2 = np.asarray(f_c_abs2, dtype=np.float64)
    eps = np.asarray(epsilon, dtype=np.float64)
    cen = np.asarray(centric, dtype=bool)
    a_d = np.asarray(scale_a, dtype=np.float64)
    al = np.asarray(alpha, dtype=np.float64)
    be = np.asarray(beta, dtype=np.float64)
    sp = np.maximum(np.asarray(sigma_p, dtype=np.float64), 1e-12)
    sa = sigma_a_from_alpha_beta(al, be, sp)
    sa = np.maximum(sa, 1e-6)
    den = a_d * eps * (al * al * sp + be)
    den = np.maximum(den, 1e-12)
    e2_o = i / den
    e_c2 = fc2 / sp
    e2_mean = (sa**2) * e_c2 + (1.0 - sa**2)
    y = (e2_o - e2_mean) / sa
    # Variance of Ibar in E² units: Var(E²_o) = Var(I) / den²
    # Rice + measurement variance of I, then transform.
    rice_unit = be * be + 2.0 * (al * al) * fc2 * be
    rice_var_i = (a_d * eps) ** 2 * rice_unit
    two = np.where(cen, 2.0, 1.0)
    var_i = two * rice_var_i + v_meas
    var_e2 = var_i / (den**2)
    var_y = var_e2 / (sa**2)
    var_y_meas = v_meas / (den**2 * sa**2)
    w_model = np.where(obs, 1.0 / np.maximum(var_y, 1e-12), 0.0)
    w_meas = np.where(obs, 1.0 / np.maximum(var_y_meas, 1e-12), 0.0)
    y = np.where(obs, y, 0.0)
    return y, w_model, w_meas, e2_o, sa
