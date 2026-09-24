"""Closed-form Rice / Woolfson NLL on (F_eff, Σ_Δ). Spec eqs 7–9.

This is not ``ml_i``: there is no experimental intensity noise. Gradients
are the per-reflection factors in eq. 24, ``∂L/∂F_eff*`` and ``∂L/∂Σ_Δ``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Abramowitz & Stegun 9.8.1–9.8.4 (e^{-x} I_0, e^{-x} I_1).
_I0_SMALL = (1.0, 3.5156229, 3.0899424, 1.2067492, 0.2659732, 0.0360768, 0.0045813)
_I0_LARGE = (
    0.39894228,
    0.01328592,
    0.00225319,
    -0.00157565,
    0.00916281,
    -0.02057706,
    0.02635537,
    -0.01647633,
    0.00392377,
)
_I1_SMALL = (0.5, 0.87890594, 0.51498869, 0.15084934, 0.02658733, 0.00301532, 0.00032411)
_I1_LARGE = (
    0.39894228,
    -0.03988024,
    -0.00362018,
    0.00163801,
    -0.01031555,
    0.02282967,
    -0.02895312,
    0.01787654,
    -0.00420059,
)
_AS_CUT = 3.75
_LOG2 = math.log(2.0)
_LOG_PI = math.log(math.pi)


def _horner(t: np.ndarray, coeffs: tuple[float, ...]) -> np.ndarray:
    s = np.zeros_like(t, dtype=np.float64)
    for c in reversed(coeffs):
        s = s * t + c
    return s


def log_i0(x: np.ndarray) -> np.ndarray:
    """log I_0(x), stable for large |x|."""
    ax = np.abs(np.asarray(x, dtype=np.float64))
    small = ax <= _AS_CUT
    t_s = ax / _AS_CUT
    i0 = _horner(t_s * t_s, _I0_SMALL)
    t_l = np.divide(_AS_CUT, ax, out=np.ones_like(ax), where=~small)
    i0e = _horner(t_l, _I0_LARGE) / np.sqrt(np.maximum(ax, 1e-300))
    out = np.empty_like(ax)
    out[small] = np.log(np.maximum(i0[small], 1e-300))
    out[~small] = ax[~small] + np.log(np.maximum(i0e[~small], 1e-300))
    return out


def bessel_i1_over_i0(x: np.ndarray) -> np.ndarray:
    """I_1(x)/I_0(x) via exponentially scaled polynomials. 0 at x=0."""
    ax = np.abs(np.asarray(x, dtype=np.float64))
    small = ax <= _AS_CUT
    t_s = ax / _AS_CUT
    t2 = t_s * t_s
    i0 = _horner(t2, _I0_SMALL)
    i1 = ax * _horner(t2, _I1_SMALL)
    t_l = np.divide(_AS_CUT, ax, out=np.ones_like(ax), where=~small)
    i0e = _horner(t_l, _I0_LARGE)
    i1e = _horner(t_l, _I1_LARGE)
    out = np.empty_like(ax)
    out[small] = np.divide(i1[small], np.maximum(i0[small], 1e-300))
    out[~small] = np.divide(i1e[~small], np.maximum(i0e[~small], 1e-300))
    return np.copysign(out, x)


def log_cosh(x: np.ndarray) -> np.ndarray:
    ax = np.abs(np.asarray(x, dtype=np.float64))
    return ax + np.log1p(np.exp(-2.0 * ax)) - _LOG2


@dataclass
class RiceResult:
    """Per-reflection intensity NLL (eq. 9) and the eq. 24 factors."""

    nll: np.ndarray
    d_f_eff_star: np.ndarray
    d_sigma: np.ndarray

    @property
    def value(self) -> float:
        return float(np.sum(self.nll))


def rice_nll_intensity(
    intensity: np.ndarray,
    f_eff: np.ndarray,
    sigma_delta: np.ndarray,
    *,
    epsilon: Optional[np.ndarray] = None,
    centric: Optional[np.ndarray] = None,
    sigma_floor: float = 1e-12,
) -> RiceResult:
    """−log p(I; F_eff, Σ_Δ) from eqs 7–8, converted to intensities as in §4.

    ``d_f_eff_star`` is G_h = (∂L/∂|F_eff|) F_eff/|F_eff| so that
    dL = Σ_h Re[conj(G_h) dF_eff] (cctbx ``d_target_d_f_calc`` convention).
    """
    i_obs = np.asarray(intensity, dtype=np.float64).reshape(-1)
    fe = np.asarray(f_eff, dtype=np.complex128).reshape(-1)
    sig = np.maximum(np.asarray(sigma_delta, dtype=np.float64).reshape(-1), sigma_floor)
    n = i_obs.shape[0]
    if fe.shape[0] != n or sig.shape[0] != n:
        raise ValueError("intensity, F_eff, Σ_Δ length mismatch")
    eps = np.ones(n, dtype=np.float64) if epsilon is None else np.asarray(epsilon, dtype=np.float64).reshape(-1)
    cen = np.zeros(n, dtype=bool) if centric is None else np.asarray(centric, dtype=bool).reshape(-1)
    fo = np.sqrt(np.maximum(i_obs, 0.0))
    fe_abs = np.abs(fe)
    v = eps * sig  # ε Σ_Δ
    nll = np.empty(n, dtype=np.float64)
    d_abs = np.empty(n, dtype=np.float64)
    d_sig = np.empty(n, dtype=np.float64)

    ac = ~cen
    if np.any(ac):
        # p(I) = p(|F|)/(2|F|) from (7): L = log(v) + (I+|Fe|²)/v − log I0(2 Fo Fe / v)
        va, foa, fea, ia = v[ac], fo[ac], fe_abs[ac], i_obs[ac]
        x = 2.0 * foa * fea / va
        r = bessel_i1_over_i0(x)
        nll[ac] = np.log(va) + (ia + fea * fea) / va - log_i0(x)
        d_abs[ac] = 2.0 * (fea - foa * r) / va
        # dL/dΣ = ε [1/v − (I+Fe²)/v² + r x / v]
        d_sig[ac] = eps[ac] * (1.0 / va - (ia + fea * fea) / (va * va) + r * x / va)

    if np.any(cen):
        # p(|F|) from (8); p(I)=p(|F|)/(2|F|).
        vc, foc, fec, ic = v[cen], fo[cen], fe_abs[cen], i_obs[cen]
        z = foc * fec / vc
        th = np.tanh(z)
        nll[cen] = (
            0.5 * (_LOG_PI + np.log(vc) - _LOG2)
            + (ic + fec * fec) / (2.0 * vc)
            - log_cosh(z)
            + np.log(np.maximum(2.0 * foc, 1e-300))
        )
        d_abs[cen] = (fec - foc * th) / vc
        d_sig[cen] = eps[cen] * (
            0.5 / vc - (ic + fec * fec) / (2.0 * vc * vc) + th * foc * fec / (vc * vc)
        )

    phase = np.divide(fe, fe_abs, out=np.zeros_like(fe), where=fe_abs > 1e-15)
    return RiceResult(nll=nll, d_f_eff_star=d_abs * phase, d_sigma=d_sig)


def least_squares_intensity_nll(
    intensity: np.ndarray,
    f_eff: np.ndarray,
    sigma_delta: np.ndarray,
    *,
    epsilon: Optional[np.ndarray] = None,
    sigma_floor: float = 1e-12,
) -> np.ndarray:
    """Weighted LS on intensities, eq. 21 (quadratic term only).

    Valid when |F_eff|² ≫ ε Σ_Δ (good-model / strong-data limit).
    """
    i_obs = np.asarray(intensity, dtype=np.float64).reshape(-1)
    fe2 = np.abs(np.asarray(f_eff)) ** 2
    sig = np.maximum(np.asarray(sigma_delta, dtype=np.float64).reshape(-1), sigma_floor)
    eps = np.ones_like(i_obs) if epsilon is None else np.asarray(epsilon, dtype=np.float64).reshape(-1)
    denom = 4.0 * np.maximum(fe2, sigma_floor) * eps * sig
    return (i_obs - fe2) ** 2 / denom


def intensity_variance_limit(
    f_eff: np.ndarray,
    sigma_delta: np.ndarray,
    *,
    epsilon: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Var(I) ≈ 2 |F_eff|² ε Σ_Δ. Spec eq. 20."""
    fe2 = np.abs(np.asarray(f_eff)) ** 2
    sig = np.asarray(sigma_delta, dtype=np.float64).reshape(-1)
    eps = np.ones_like(sig) if epsilon is None else np.asarray(epsilon, dtype=np.float64).reshape(-1)
    return 2.0 * fe2 * eps * sig


def draw_circular_normal(
    f_eff: np.ndarray,
    sigma_delta: np.ndarray,
    rng: np.random.Generator,
    *,
    epsilon: Optional[np.ndarray] = None,
    centric: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Draw F from eqs 5–6 (A3, A4). Acentric: CN(μ, εΣ); centric: N(μ, εΣ) real."""
    mu = np.asarray(f_eff, dtype=np.complex128).reshape(-1)
    sig = np.asarray(sigma_delta, dtype=np.float64).reshape(-1)
    n = mu.shape[0]
    eps = np.ones(n) if epsilon is None else np.asarray(epsilon, dtype=np.float64).reshape(-1)
    cen = np.zeros(n, dtype=bool) if centric is None else np.asarray(centric, dtype=bool).reshape(-1)
    var = np.maximum(eps * sig, 0.0)
    out = np.zeros(n, dtype=np.complex128)
    ac = ~cen
    if np.any(ac):
        sd = np.sqrt(var[ac] / 2.0)
        out[ac] = (
            mu[ac]
            + rng.normal(scale=sd)
            + 1j * rng.normal(scale=sd)
        )
    if np.any(cen):
        # Centric F is real in a phase-restricted frame; take the phase of F_eff.
        phase = np.divide(mu[cen], np.abs(mu[cen]), out=np.ones_like(mu[cen]), where=np.abs(mu[cen]) > 1e-15)
        mag = rng.normal(loc=np.abs(mu[cen]), scale=np.sqrt(var[cen]))
        out[cen] = mag * phase
    return out
