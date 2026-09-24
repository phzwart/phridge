"""First and second moments (F_eff, Σ_Δ). Spec eqs 3–4.

F_eff is the FFT structure factor of the modified model (eq. 25) times the
atom-independent shell scale D_0(s). Σ_Δ is phase-free (no FFT).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from phridge.sfcalc.engine.cell import reciprocal_cartesian, sym6_to_mat
from phridge.sfcalc.engine.engine import EngineParams, ScatteringModel, StructureFactorEngine

TWO_PI2 = 2.0 * math.pi * math.pi
TWO_PI = 2.0 * math.pi


def resolution_s2(unit_cell: tuple, hkl: np.ndarray) -> np.ndarray:
    """s² = |d*|² = 1/d² for each Miller index."""
    return np.sum(reciprocal_cartesian(unit_cell, np.asarray(hkl, dtype=np.int64)) ** 2, axis=1)


def form_factors(model: ScatteringModel, s2: np.ndarray) -> np.ndarray:
    """f_j(s) from the Gaussian table, shape (N, H). fp is added; fdp ignored (phase-free)."""
    stol2 = np.asarray(s2, dtype=np.float64).reshape(-1) / 4.0
    ti = np.asarray(model.type_index)
    a = np.asarray(model.gauss_a, dtype=np.float64)[ti]  # (N, K)
    b = np.asarray(model.gauss_b, dtype=np.float64)[ti]
    c = np.asarray(model.gauss_c, dtype=np.float64)[ti] + np.asarray(model.fp, dtype=np.float64)
    # f = Σ_k a_k exp(-b_k stol²) + c
    expo = np.exp(-b[:, :, None] * stol2[None, None, :])  # (N, K, H)
    return c[:, None] + np.sum(a[:, :, None] * expo, axis=1)


def model_temperature(model: ScatteringModel, hkl: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """T_j(h) from the modelled ADP only (not the error U_j). Shape (N, H)."""
    h = np.asarray(hkl, dtype=np.float64).reshape(-1, 3)
    ss = np.asarray(s2, dtype=np.float64).reshape(-1)
    n = model.n_scatterers
    t = np.empty((n, h.shape[0]), dtype=np.float64)
    aniso = np.asarray(model.anisotropic, dtype=bool)
    t[~aniso] = np.exp(-TWO_PI2 * np.asarray(model.u_iso)[~aniso, None] * ss[None, :])
    if np.any(aniso):
        u = sym6_to_mat(np.asarray(model.u_star, dtype=np.float64)[aniso])
        # h^T U* h
        huh = np.einsum("hi,nij,hj->nh", h, u, h)
        t[aniso] = np.exp(-TWO_PI2 * huh)
    return t


def sigma_delta(
    model: ScatteringModel,
    hkl: np.ndarray,
    w: np.ndarray,
    d_jh: np.ndarray,
    sigma_miss: np.ndarray,
) -> np.ndarray:
    """Σ_Δ(h) = Σ_j mult_j (occ f T)² (w_j − d_j²) + Σ_miss(s). Spec eq. 4.

    Independence of ξ_j, Δ_j, y_m is approximation (A1). Unmodelled scatterers
    are uniform (A5) and contribute only Σ_miss.
    """
    s2 = resolution_s2(model.unit_cell, hkl)
    fj = form_factors(model, s2)
    tj = model_temperature(model, hkl, s2)
    occ = np.asarray(model.occupancy, dtype=np.float64).reshape(-1, 1)
    mult = np.asarray(model.multiplicity, dtype=np.float64).reshape(-1, 1)
    w = np.asarray(w, dtype=np.float64).reshape(-1, 1)
    d2 = np.asarray(d_jh, dtype=np.float64) ** 2
    atom = (occ * fj * tj) ** 2 * (w - d2)
    return np.sum(mult * atom, axis=0) + np.asarray(sigma_miss, dtype=np.float64).reshape(-1)


def sigma_p(model: ScatteringModel, hkl: np.ndarray) -> np.ndarray:
    """Σ_P(h) = Σ_j mult_j (occ f T)², the model scattering power in eq. 11."""
    s2 = resolution_s2(model.unit_cell, hkl)
    fj = form_factors(model, s2)
    tj = model_temperature(model, hkl, s2)
    occ = np.asarray(model.occupancy, dtype=np.float64).reshape(-1, 1)
    mult = np.asarray(model.multiplicity, dtype=np.float64).reshape(-1, 1)
    return np.sum(mult * (occ * fj * tj) ** 2, axis=0)


def f_eff_direct(
    model: ScatteringModel,
    hkl: np.ndarray,
    d_jh: np.ndarray,
) -> np.ndarray:
    """F_eff by direct summation of eq. 3 (small-N reference; includes symops)."""
    h = np.asarray(hkl, dtype=np.float64).reshape(-1, 3)
    s2 = resolution_s2(model.unit_cell, hkl)
    fj = form_factors(model, s2)
    occ = np.asarray(model.occupancy, dtype=np.float64)
    mult = np.asarray(model.multiplicity, dtype=np.float64)
    n_sym = float(model.n_sym)
    d_jh = np.asarray(d_jh, dtype=np.float64)
    f = np.zeros(h.shape[0], dtype=np.complex128)
    sites = np.asarray(model.sites_frac, dtype=np.float64)
    rot = np.asarray(model.rot, dtype=np.float64)
    trans = np.asarray(model.trans, dtype=np.float64)
    aniso = np.asarray(model.anisotropic, dtype=bool)
    s2 = np.asarray(s2, dtype=np.float64)
    for j in range(model.n_scatterers):
        weight = occ[j] * mult[j] / n_sym
        for si in range(rot.shape[0]):
            x = rot[si] @ sites[j] + trans[si]
            if aniso[j]:
                u = sym6_to_mat(model.u_star[j : j + 1])[0]
                u_sym = rot[si] @ u @ rot[si].T
                dw = np.exp(-TWO_PI2 * np.einsum("hi,ij,hj->h", h, u_sym, h))
            else:
                dw = np.exp(-TWO_PI2 * float(model.u_iso[j]) * s2)
            phase = np.exp(1j * TWO_PI * (h @ x))
            f += (weight * d_jh[j] * fj[j] * dw) * phase
    return f


def f_eff_fft(
    modified: ScatteringModel,
    hkl: np.ndarray,
    d0: np.ndarray,
    params: Optional[EngineParams] = None,
    device: str = "cpu",
) -> np.ndarray:
    """F_eff = D_0(s) F_c(modified model) via the existing FFT engine. Spec eq. 3, 25."""
    s2 = resolution_s2(modified.unit_cell, hkl)
    d_min = float(1.0 / np.sqrt(np.max(s2))) if s2.size else 2.0
    eng = StructureFactorEngine(
        modified,
        np.asarray(hkl, dtype=np.int64),
        params or EngineParams(d_min=d_min, quality_factor=1000.0),
        device=device,
    )
    fc = eng.f_calc_numpy()
    return np.asarray(d0, dtype=np.float64).reshape(-1) * fc


def moments(
    model: ScatteringModel,
    hkl: np.ndarray,
    w: np.ndarray,
    u_err_iso: np.ndarray,
    d0: np.ndarray,
    sigma_miss: np.ndarray,
    *,
    use_fft: bool = False,
    engine_params: Optional[EngineParams] = None,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Return (F_eff, Σ_Δ).

    ``d0`` and ``sigma_miss`` are per-reflection. ``w`` and ``u_err_iso`` are
    per-atom. When ``use_fft`` is true, F_eff goes through
    :class:`StructureFactorEngine` on the modified model; otherwise a direct sum.
    """
    from phridge.contrib.spatial_sigmaa_v2.per_atom import modified_model

    s2 = resolution_s2(model.unit_cell, hkl)
    # d_j from eq. 14: need λ, κ. Callers who already have d_jh should use
    # sigma_delta / f_eff_* directly. Here reconstruct d_j = D0 w D(U_j).
    from phridge.contrib.spatial_sigmaa_v2.per_atom import luzzati_d_iso

    d_j = np.asarray(d0, dtype=np.float64).reshape(1, -1) * np.asarray(w, dtype=np.float64).reshape(-1, 1) * luzzati_d_iso(
        u_err_iso, s2
    )
    sig = sigma_delta(model, hkl, w, d_j, sigma_miss)
    if use_fft:
        modified = modified_model(model, w, u_err_iso)
        fe = f_eff_fft(modified, hkl, d0, params=engine_params, device=device)
    else:
        fe = f_eff_direct(model, hkl, d_j)
    return fe, sig
