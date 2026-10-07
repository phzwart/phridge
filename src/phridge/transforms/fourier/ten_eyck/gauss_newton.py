"""Gauss-Newton block helpers for the Ten Eyck engine."""

from __future__ import annotations

import math

import numpy as np

from phridge.transforms.fourier.ten_eyck.grids import TWO_PI2


def _gn_adp_row(j, aniso_j, s, h, x_exp, a_exp, rot, u_star, u_iso, dstar2, wm, wp, e_m2iphi):
    """Per-atom ADP Gauss-Newton row (anisotropic block or isotropic diagonal)."""
    n_h = h.shape[0]
    if aniso_j:
        dF_u = np.zeros((6, n_h), dtype=np.complex128)
        for si in range(s):
            e = j * s + si
            eph = np.exp(2j * math.pi * (h @ x_exp[e]))
            hh = np.stack(
                [h[:, 0] ** 2, h[:, 1] ** 2, h[:, 2] ** 2, 2 * h[:, 0] * h[:, 1], 2 * h[:, 0] * h[:, 2], 2 * h[:, 1] * h[:, 2]],
                axis=0,
            )
            gtmx = _u_star_gradient_transform(rot[si])
            kappa = -TWO_PI2 * (gtmx @ hh)
            dF_u += kappa * (a_exp[e] * eph)[None, :]
        return _gn_block_from_df(dF_u, wm, wp, e_m2iphi), 0.0
    dF_uiso = np.zeros(n_h, dtype=np.complex128)
    for si in range(s):
        e = j * s + si
        eph = np.exp(2j * math.pi * (h @ x_exp[e]))
        dF_uiso += a_exp[e] * eph * (-TWO_PI2 * dstar2)
    return np.zeros((6, 6), dtype=np.float64), _gn_scalar_from_df(dF_uiso, wm, wp, e_m2iphi)


def _gn_block_from_df(dF, wm, wp, e_m2iphi):
    """GN block from dF[param, refl] complex Jacobians."""
    n = dF.shape[0]
    out = np.zeros((n, n), dtype=np.float64)
    for a in range(n):
        for b in range(a, n):
            cprod = np.conj(dF[a]) * dF[b]
            sprod = dF[a] * dF[b] * e_m2iphi
            val = float(np.sum(wp * cprod.real + wm * sprod.real))
            out[a, b] = out[b, a] = val
    return out


def _gn_scalar_from_df(dF, wm, wp, e_m2iphi):
    cprod = np.conj(dF) * dF
    sprod = dF * dF * e_m2iphi
    return float(np.sum(wp * cprod.real + wm * sprod.real))


def _u_star_gradient_transform(R):
    """6x6 matrix mapping d(huh)/dU*_exp contributions back to d/dU*_asu.

    U*_exp = R U*_asu R^T; returns G such that kappa_asu = G @ kappa_exp_coeffs
    where kappa_exp_coeffs = (h'h', k'k', l'l', 2h'k', 2h'l', 2k'l') for h' = R^T h
    ... actually we use cctbx tensor_rank_2::gradient_transform_matrix convention:
    d_target/dU*_asu = G^T @ d_target/dU*_exp, so kappa_asu = G @ kappa_exp when
    dF ∝ kappa · (a e^{iφ}).
    """
    # Build G where vec(U_exp) related; for sym_mat3 packed (6,):
    # huh = h^T U_exp h = h^T R U R^T h = (R^T h)^T U (R^T h)
    # so kappa_asu_μ = -2π² * m_μ(R^T h) with m = (hh,kk,ll,2hk,2hl,2kl)
    # Equivalently kappa_asu = G @ (-2π² m(h_exp)) with h_exp = h (same miller in
    # crystal frame) and U_exp = R U R^T means m(h) on U_exp = m(R^T h) on U_asu.
    # Easiest: express m_asu(h) = m(R^T h) via the 6x6 that maps m(h_exp)->m_asu.
    # We pass hh computed from crystal-frame h for U_exp; need ∂huh/∂U_asu.
    # huh = h^T R U R^T h = (R^T h)^T U (R^T h). Let p = R^T h.
    # ∂huh/∂U_ij packed = m(p). So kappa_asu = -2π² m(R^T h) = G @ (-2π² m(h))
    # with G mapping m(h) -> m(R^T h)... no that's not a linear map on m(h) alone
    # independent of h. Compute G as gradient_transform: d/dU_asu = G^T d/dU_exp
    # where U_exp = R U_asu R^T.
    G = np.zeros((6, 6), dtype=np.float64)
    # Finite basis: for each ASU basis matrix E_μ, U_exp = R E_μ R^T, read packed
    basis = [
        np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64),
        np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float64),
        np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=np.float64),
        np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.float64),
        np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=np.float64),
        np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float64),
    ]
    for mu, E in enumerate(basis):
        Uexp = R @ E @ R.T
        # pack (11,22,33,12,13,23); off-diagonals stored as full U_ij (not doubled)
        packed = np.array([Uexp[0, 0], Uexp[1, 1], Uexp[2, 2], Uexp[0, 1], Uexp[0, 2], Uexp[1, 2]])
        G[:, mu] = packed  # U_exp_packed = G @ U_asu_packed ... check off-diag
    # For off-diagonal basis E_3 has U_01=U_10=1, packed_asu[3]=1 meaning U_01=1.
    # U_exp = R E R^T, packed_exp = G @ e_mu. Then huh = m(h)·U_exp_packed with
    # m = (hh,kk,ll,2hk,2hl,2kl). And huh = m_asu · U_asu with m_asu = G.T @ m.
    # So kappa_asu = -2π² G.T @ m(h) when kappa_exp = -2π² m(h).
    # Thus we want gtmx such that kappa_asu = gtmx @ hh_as_m, i.e. gtmx = G.T
    return G.T
