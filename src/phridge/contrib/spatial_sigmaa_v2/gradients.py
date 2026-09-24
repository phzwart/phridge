"""Per-atom and field-coefficient gradients (spec §10, eqs 24–29).

Mean term: Agarwal–Ten Eyck on the modified model (eq. 25), coefficients
``D_0(s) · ∂L/∂F_eff*``. Variance term: phase-free sums (eqs 26–28);
anisotropic errors are stubbed. Field coefficients: design-matrix chain
rule (eq. 29).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from phridge.contrib.spatial_sigmaa_v2.fields import FieldBasis
from phridge.contrib.spatial_sigmaa_v2.moments import form_factors, model_temperature, resolution_s2
from phridge.contrib.spatial_sigmaa_v2.per_atom import (
    EIGHT_PI2,
    luzzati_d_iso,
    modified_model,
)
from phridge.sfcalc.engine.cell import sym6_to_mat
from phridge.sfcalc.engine.engine import EngineParams, ScatteringModel, StructureFactorEngine

TWO_PI = 2.0 * math.pi
TWO_PI2 = 2.0 * math.pi * math.pi
FOUR_PI2 = 4.0 * math.pi * math.pi


@dataclass
class AtomGrads:
    """Per-atom derivatives of L_data. ``site_frac`` is fractional (engine frame)."""

    site_frac: np.ndarray  # (N, 3)
    w: np.ndarray  # (N,)
    u_err_iso: np.ndarray  # (N,)


def _atom_power(model: ScatteringModel, hkl: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``mult (occ f T)²``, form factors, temperature. Shapes (N, H)."""
    s2 = resolution_s2(model.unit_cell, hkl)
    fj = form_factors(model, s2)
    tj = model_temperature(model, hkl, s2)
    occ = np.asarray(model.occupancy, dtype=np.float64).reshape(-1, 1)
    mult = np.asarray(model.multiplicity, dtype=np.float64).reshape(-1, 1)
    return mult * (occ * fj * tj) ** 2, fj, tj


def mean_gradients_direct(
    model: ScatteringModel,
    hkl: np.ndarray,
    d_jh: np.ndarray,
    d_f_eff_star: np.ndarray,
    w: np.ndarray,
) -> AtomGrads:
    """Numpy VJP of eq. 3 (small-N reference). ``d_f_eff_star`` is G_h of eq. 24."""
    h = np.asarray(hkl, dtype=np.float64).reshape(-1, 3)
    s2 = resolution_s2(model.unit_cell, hkl)
    fj = form_factors(model, s2)
    g = np.asarray(d_f_eff_star, dtype=np.complex128).reshape(-1)
    d_jh = np.asarray(d_jh, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    occ = np.asarray(model.occupancy, dtype=np.float64)
    mult = np.asarray(model.multiplicity, dtype=np.float64)
    n_sym = float(model.n_sym)
    sites = np.asarray(model.sites_frac, dtype=np.float64)
    rot = np.asarray(model.rot, dtype=np.float64)
    trans = np.asarray(model.trans, dtype=np.float64)
    aniso = np.asarray(model.anisotropic, dtype=bool)
    n = model.n_scatterers
    d_site = np.zeros((n, 3), dtype=np.float64)
    d_w = np.zeros(n, dtype=np.float64)
    d_u = np.zeros(n, dtype=np.float64)
    for j in range(n):
        weight = occ[j] * mult[j] / n_sym
        inv_w = 1.0 / w[j] if w[j] > 1e-15 else 0.0
        for si in range(rot.shape[0]):
            x = rot[si] @ sites[j] + trans[si]
            if aniso[j]:
                u = sym6_to_mat(model.u_star[j : j + 1])[0]
                u_sym = rot[si] @ u @ rot[si].T
                dw = np.exp(-TWO_PI2 * np.einsum("hi,ij,hj->h", h, u_sym, h))
            else:
                dw = np.exp(-TWO_PI2 * float(model.u_iso[j]) * s2)
            contrib = (weight * d_jh[j] * fj[j] * dw) * np.exp(1j * TWO_PI * (h @ x))
            h_rot = h @ rot[si]
            d_site[j] += np.real(np.conj(g)[:, None] * contrib[:, None] * (1j * TWO_PI * h_rot)).sum(axis=0)
            d_w[j] += float(np.real(np.conj(g) * contrib * inv_w).sum())
            d_u[j] += float(np.real(np.conj(g) * contrib * (-TWO_PI2 * s2)).sum())
    return AtomGrads(site_frac=d_site, w=d_w, u_err_iso=d_u)


def mean_gradients_fft(
    model: ScatteringModel,
    hkl: np.ndarray,
    w: np.ndarray,
    u_err_iso: np.ndarray,
    d0: np.ndarray,
    d_f_eff_star: np.ndarray,
    *,
    engine_params: Optional[EngineParams] = None,
    device: str = "cpu",
) -> AtomGrads:
    """Agarwal–Ten Eyck on the modified model. Spec eq. 25, steps (a)–(c).

    Map coefficients are ``D_0(s) · ∂L/∂F_eff*``. Occupancy grads of the
    modified structure are ``∂L/∂(occ w)``; we convert to ``∂L/∂w``.
    """
    modified = modified_model(model, w, u_err_iso)
    s2 = resolution_s2(model.unit_cell, hkl)
    d_min = float(1.0 / np.sqrt(np.max(s2))) if s2.size else 2.0
    eng = StructureFactorEngine(
        modified,
        np.asarray(hkl, dtype=np.int64),
        engine_params or EngineParams(d_min=d_min, quality_factor=1000.0),
        device=device,
    )
    g_fc = np.asarray(d0, dtype=np.float64).reshape(-1) * np.asarray(d_f_eff_star, dtype=np.complex128).reshape(-1)
    raw = eng.gradients(g_fc)
    occ = np.asarray(model.occupancy, dtype=np.float64)
    return AtomGrads(
        site_frac=np.asarray(raw["site_frac"], dtype=np.float64),
        w=occ * np.asarray(raw["occupancy"], dtype=np.float64),
        u_err_iso=np.asarray(raw["u_iso"], dtype=np.float64),
    )


def variance_gradients_iso(
    model: ScatteringModel,
    hkl: np.ndarray,
    w: np.ndarray,
    u_err_iso: np.ndarray,
    d_sigma: np.ndarray,
) -> AtomGrads:
    """Phase-free variance grads, isotropic U_j. Spec eqs 26–28.

    ``site_frac`` is identically zero (eq. 26). ``u_err_iso`` is the scalar
    ``∂L/∂U_j`` obtained by contracting eq. 28 with ``I`` (trace / 3 is not
    used; for U = u I one has ``hᵀ (∂/∂u) h = s²``).
    """
    s2 = resolution_s2(model.unit_cell, hkl)
    power, _, _ = _atom_power(model, hkl)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    d_sig = np.asarray(d_sigma, dtype=np.float64).reshape(1, -1)
    d_iso = luzzati_d_iso(u_err_iso, s2)
    d2 = d_iso**2
    # eq. 27: ∂Σ/∂w = power (1 − 2 w D²)
    d_w = np.sum(d_sig * power * (1.0 - 2.0 * w[:, None] * d2), axis=1)
    # eq. 28 contracted: ∂Σ/∂u = power w² D² 4π² s²
    d_u = np.sum(d_sig * power * (w[:, None] ** 2) * d2 * (FOUR_PI2 * s2[None, :]), axis=1)
    return AtomGrads(
        site_frac=np.zeros((model.n_scatterers, 3), dtype=np.float64),
        w=d_w,
        u_err_iso=d_u,
    )


def variance_gradients_aniso(*_args: object, **_kwargs: object) -> AtomGrads:
    """Self-Patterson variance-gradient map. Spec §10; Phase 3 stub."""
    raise NotImplementedError(
        "anisotropic variance-gradient map (self-Patterson at the origin, spec §10)"
    )


def add_atom_grads(mean: AtomGrads, variance: AtomGrads) -> AtomGrads:
    """Eq. 24: mean (F_eff) plus variance (Σ_Δ)."""
    return AtomGrads(
        site_frac=mean.site_frac + variance.site_frac,
        w=mean.w + variance.w,
        u_err_iso=mean.u_err_iso + variance.u_err_iso,
    )


def atom_to_field_params(grads: AtomGrads, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """∂L/∂λ_j and ∂L/∂κ_j from ∂L/∂w and ∂L/∂U. Spec eq. 13, text after (29)."""
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    d_lambda = grads.w * w
    d_kappa = grads.u_err_iso / EIGHT_PI2
    return d_lambda, d_kappa


def field_gradients(
    basis: FieldBasis,
    sites_frac: np.ndarray,
    d_lambda: np.ndarray,
    d_kappa: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """∂L/∂c_k for λ and κ. Spec eq. 29 (real design-matrix form)."""
    m = basis.design_matrix(sites_frac)
    return m.T @ np.asarray(d_lambda, dtype=np.float64).reshape(-1), m.T @ np.asarray(
        d_kappa, dtype=np.float64
    ).reshape(-1)


def d_d0_and_sigma_miss(
    model: ScatteringModel,
    hkl: np.ndarray,
    w: np.ndarray,
    u_err_iso: np.ndarray,
    d0: np.ndarray,
    d_f_eff_star: np.ndarray,
    d_sigma: np.ndarray,
    f_eff: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-reflection ∂L/∂D_0 and ∂L/∂Σ_miss."""
    d0 = np.asarray(d0, dtype=np.float64).reshape(-1)
    g = np.asarray(d_f_eff_star, dtype=np.complex128).reshape(-1)
    fe = np.asarray(f_eff, dtype=np.complex128).reshape(-1)
    # F_eff = D0 F_c ⇒ ∂L/∂D0|_mean = Re[conj(G) F_c] = Re[conj(G) F_eff / D0]
    d_d0_mean = np.real(np.conj(g) * np.divide(fe, d0, out=np.zeros_like(fe), where=d0 > 1e-15))
    s2 = resolution_s2(model.unit_cell, hkl)
    power, _, _ = _atom_power(model, hkl)
    d_j = d0[None, :] * np.asarray(w, dtype=np.float64).reshape(-1, 1) * luzzati_d_iso(u_err_iso, s2)
    # ∂Σ/∂D0 = −2 Σ_j power d_j² / D0
    d_sig = np.asarray(d_sigma, dtype=np.float64).reshape(-1)
    d_d0_var = d_sig * np.sum(power * (-2.0 * d_j**2) / np.maximum(d0[None, :], 1e-15), axis=0)
    return d_d0_mean + d_d0_var, d_sig.copy()


