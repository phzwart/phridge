"""Fisher closure for per-atom U_j (spec §9, eqs 21–23).

The 3×3 Gauss–Newton site blocks from the intensity LS target (eq. 21)
are the observed Fisher information for the coordinates. They arrive in
the engine's fractional frame; spec U_j is Cartesian Å², so invert after
the frame change. Off-diagonal (neighbour) blocks are omitted — the usual
sparse-inverse approximation, labelled as such.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from phridge.contrib.spatial_sigmaa_v2.moments import (
    f_eff_direct,
    form_factors,
    resolution_s2,
    sigma_delta,
)
from phridge.contrib.spatial_sigmaa_v2.per_atom import luzzati_d_iso
from phridge.contrib.spatial_sigmaa_v2.target import intensity_variance_limit
from phridge.sfcalc.engine.cell import orthogonalization_matrix, sym6_to_mat
from phridge.sfcalc.engine.engine import ScatteringModel

TWO_PI = 2.0 * np.pi
TWO_PI2 = 2.0 * np.pi * np.pi


def ls_curvatures(
    f_eff: np.ndarray,
    sigma_delta: np.ndarray,
    *,
    epsilon: Optional[np.ndarray] = None,
    sigma_floor: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """``(c_r, c_t)`` so engine GN reproduces eq. 22.

    ``∂|F|²/∂θ = 2 |F| Re[e^{-iφ} ∂F/∂θ]``. Then
    ``M = Σ (∂|F|²)(∂|F|²)ᵀ / Var(I)`` with ``Var(I) = 2 |F|² ε Σ``
    is the radial-only Hessian with ``c_r = 2 / (ε Σ)``, ``c_t = 0``.
    """
    sig = np.maximum(np.asarray(sigma_delta, dtype=np.float64).reshape(-1), sigma_floor)
    eps = np.ones_like(sig) if epsilon is None else np.asarray(epsilon, dtype=np.float64).reshape(-1)
    c_r = 2.0 / (eps * sig)
    return c_r, np.zeros_like(c_r)


def frac_blocks_to_cartesian(site_frac: np.ndarray, unit_cell: tuple) -> np.ndarray:
    """H_cart = O^{-T} H_frac O^{-1}. ``site_frac`` is (N, 3, 3)."""
    h = np.asarray(site_frac, dtype=np.float64).reshape(-1, 3, 3)
    o = orthogonalization_matrix(unit_cell)
    o_inv = np.linalg.inv(o)
    # H_frac = O^T H_cart O  ⇒  H_cart = O^{-T} H_frac O^{-1}
    return np.einsum("ia,nab,bj->nij", o_inv.T, h, o_inv)


def cartesian_covariance(site_frac: np.ndarray, unit_cell: tuple, *, ridge: float = 1e-8) -> np.ndarray:
    """U_j = [M^{-1}]_{x_j x_j} in Cartesian Å². Spec eq. 23.

    Diagonal-block inverse only (off-diagonal sparse inverse omitted).
    """
    h = np.asarray(site_frac, dtype=np.float64).reshape(-1, 3, 3)
    o = orthogonalization_matrix(unit_cell)
    eye = np.eye(3, dtype=np.float64)
    out = np.zeros_like(h)
    for j in range(h.shape[0]):
        cov_frac = np.linalg.inv(h[j] + ridge * eye)
        out[j] = o @ cov_frac @ o.T
    return out


def u_iso_from_cart(u_cart: np.ndarray) -> np.ndarray:
    """Isotropic part tr(U)/3."""
    u = np.asarray(u_cart, dtype=np.float64).reshape(-1, 3, 3)
    return np.trace(u, axis1=1, axis2=2) / 3.0


def intensity_ls_site_blocks(
    model: ScatteringModel,
    hkl: np.ndarray,
    d_jh: np.ndarray,
    f_eff: np.ndarray,
    sigma_delta: np.ndarray,
    *,
    epsilon: Optional[np.ndarray] = None,
    sigma_floor: float = 1e-12,
) -> np.ndarray:
    """Fractional 3×3 M_j of eq. 22 from a direct Jacobian (numpy, no torch).

    Diagonal blocks only.
    """
    h = np.asarray(hkl, dtype=np.float64).reshape(-1, 3)
    s2 = resolution_s2(model.unit_cell, hkl)
    fj = form_factors(model, s2)
    fe = np.asarray(f_eff, dtype=np.complex128).reshape(-1)
    var_i = np.maximum(intensity_variance_limit(fe, sigma_delta, epsilon=epsilon), sigma_floor)
    wgt = 1.0 / var_i
    occ = np.asarray(model.occupancy, dtype=np.float64)
    mult = np.asarray(model.multiplicity, dtype=np.float64)
    n_sym = float(model.n_sym)
    sites = np.asarray(model.sites_frac, dtype=np.float64)
    rot = np.asarray(model.rot, dtype=np.float64)
    trans = np.asarray(model.trans, dtype=np.float64)
    aniso = np.asarray(model.anisotropic, dtype=bool)
    d_jh = np.asarray(d_jh, dtype=np.float64)
    n = model.n_scatterers
    blocks = np.zeros((n, 3, 3), dtype=np.float64)
    for j in range(n):
        weight = occ[j] * mult[j] / n_sym
        dF = np.zeros((3, h.shape[0]), dtype=np.complex128)
        for si in range(rot.shape[0]):
            x = rot[si] @ sites[j] + trans[si]
            if aniso[j]:
                u = sym6_to_mat(model.u_star[j : j + 1])[0]
                u_sym = rot[si] @ u @ rot[si].T
                dw = np.exp(-TWO_PI2 * np.einsum("hi,ij,hj->h", h, u_sym, h))
            else:
                dw = np.exp(-TWO_PI2 * float(model.u_iso[j]) * s2)
            with np.errstate(all="ignore"):
                contrib = (weight * d_jh[j] * fj[j] * dw) * np.exp(1j * TWO_PI * (h @ x))
            h_rot = h @ rot[si]
            dF += contrib[None, :] * (1j * TWO_PI * h_rot.T)
        # J = ∂|F|²/∂x_frac = 2 Re[conj(F) ∂F/∂x]
        jac = 2.0 * np.real(np.conj(fe)[None, :] * dF)
        blocks[j] = (jac * wgt[None, :]) @ jac.T
    return blocks


def iterate_fisher_closure(
    model: ScatteringModel,
    hkl: np.ndarray,
    u_err_iso: np.ndarray,
    *,
    w: Optional[np.ndarray] = None,
    d0: Optional[np.ndarray] = None,
    sigma_miss: Optional[np.ndarray] = None,
    epsilon: Optional[np.ndarray] = None,
    n_iter: int = 6,
    damp: float = 0.5,
    u_min: float = 1e-6,
    u_max: float = 2.0,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Fixed-point iteration of eqs 21–23.

    ``U ← (1−damp) U + damp [M⁻¹]_{xx}`` (isotropic part). ``damp=1`` is
    the raw map; the default half-step is the usual damped closure.
    """
    n = model.n_scatterers
    u = np.clip(np.asarray(u_err_iso, dtype=np.float64).reshape(-1), u_min, u_max)
    if u.shape[0] != n:
        raise ValueError("u_err_iso length must match n_scatterers")
    ww = np.ones(n) if w is None else np.asarray(w, dtype=np.float64).reshape(-1)
    n_h = np.asarray(hkl).reshape(-1, 3).shape[0]
    d0_h = np.ones(n_h) if d0 is None else np.asarray(d0, dtype=np.float64).reshape(-1)
    miss = np.zeros(n_h) if sigma_miss is None else np.asarray(sigma_miss, dtype=np.float64).reshape(-1)
    s2 = resolution_s2(model.unit_cell, hkl)
    history = [u.copy()]
    for _ in range(int(n_iter)):
        d_j = d0_h.reshape(1, -1) * ww.reshape(-1, 1) * luzzati_d_iso(u, s2)
        fe = f_eff_direct(model, hkl, d_j)
        sig = sigma_delta(model, hkl, ww, d_j, miss)
        m_frac = intensity_ls_site_blocks(model, hkl, d_j, fe, sig, epsilon=epsilon)
        u_cart = cartesian_covariance(m_frac, model.unit_cell)
        u_new = np.clip(u_iso_from_cart(u_cart), u_min, u_max)
        u = (1.0 - float(damp)) * u + float(damp) * u_new
        history.append(u.copy())
    return u, history
