"""Unit-cell algebra in numpy/torch, cctbx conventions.

Orthogonalization matrix ``O`` (a along x, b in the xy plane) so that
``x_cart = O @ x_frac`` and ``u_cart = O @ u_star @ O.T`` (adptbx).
Reciprocal-space cartesian vector ``d* = O^{-T} @ h`` with ``|d*| = 1/d``.
"""

from __future__ import annotations

import math

import numpy as np


def orthogonalization_matrix(unit_cell) -> np.ndarray:
    a, b, c, alpha, beta, gamma = (float(x) for x in unit_cell)
    ca, cb, cg = (math.cos(math.radians(x)) for x in (alpha, beta, gamma))
    sg = math.sin(math.radians(gamma))
    v = a * b * c * math.sqrt(1 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg)
    return np.array(
        [
            [a, b * cg, c * cb],
            [0.0, b * sg, c * (ca - cb * cg) / sg],
            [0.0, 0.0, v / (a * b * sg)],
        ],
        dtype=np.float64,
    )


def cell_volume(unit_cell) -> float:
    return float(np.linalg.det(orthogonalization_matrix(unit_cell)))


def reciprocal_cartesian(unit_cell, hkl: np.ndarray) -> np.ndarray:
    """Cartesian reciprocal vectors d* for integer hkl, shape (N, 3)."""
    o_inv_t = np.linalg.inv(orthogonalization_matrix(unit_cell)).T
    return np.asarray(hkl, dtype=np.float64) @ o_inv_t.T


def u_star_to_cart(unit_cell, u_star6: np.ndarray) -> np.ndarray:
    """u_star (N,6 cctbx order 11,22,33,12,13,23) -> u_cart (N,3,3)."""
    o = orthogonalization_matrix(unit_cell)
    u = sym6_to_mat(np.asarray(u_star6, dtype=np.float64))
    return np.einsum("ij,njk,lk->nil", o, u, o)


def sym6_to_mat(u6):
    """(N,6) in cctbx order (11,22,33,12,13,23) -> (N,3,3). numpy or torch."""
    if hasattr(u6, "new_zeros"):
        import torch

        n = u6.shape[0]
        m = torch.stack(
            [
                torch.stack([u6[:, 0], u6[:, 3], u6[:, 4]], dim=1),
                torch.stack([u6[:, 3], u6[:, 1], u6[:, 5]], dim=1),
                torch.stack([u6[:, 4], u6[:, 5], u6[:, 2]], dim=1),
            ],
            dim=1,
        )
        return m.reshape(n, 3, 3)
    u6 = np.asarray(u6, dtype=np.float64)
    m = np.zeros((u6.shape[0], 3, 3), dtype=np.float64)
    m[:, 0, 0] = u6[:, 0]
    m[:, 1, 1] = u6[:, 1]
    m[:, 2, 2] = u6[:, 2]
    m[:, 0, 1] = m[:, 1, 0] = u6[:, 3]
    m[:, 0, 2] = m[:, 2, 0] = u6[:, 4]
    m[:, 1, 2] = m[:, 2, 1] = u6[:, 5]
    return m


def mat_to_sym6(m):
    """(N,3,3) symmetric -> (N,6) cctbx order. numpy or torch."""
    if hasattr(m, "new_zeros"):
        import torch

        return torch.stack(
            [m[:, 0, 0], m[:, 1, 1], m[:, 2, 2], m[:, 0, 1], m[:, 0, 2], m[:, 1, 2]], dim=1
        )
    return np.stack(
        [m[:, 0, 0], m[:, 1, 1], m[:, 2, 2], m[:, 0, 1], m[:, 0, 2], m[:, 1, 2]], axis=1
    )


def u_base(d_min: float, grid_resolution_factor: float, quality_factor: float = 100.0, max_u_base: float = 12.665147955292221) -> float:
    """cctbx xray.calc_u_base: extra isotropic U added before sampling.

    Reproduces cctbx (sampling_base.h): u = d_min^2 log10(q) f^2 / (2 pi^2 (1 - 2 f)).
    """
    f = grid_resolution_factor
    value = d_min * d_min * math.log10(quality_factor) * f * f / (2 * math.pi * math.pi * (1 - 2 * f))
    return min(value, max_u_base)
