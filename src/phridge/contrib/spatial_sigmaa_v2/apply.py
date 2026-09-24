"""Apply frozen (w_j, U_j, D_0) to the coordinate-cycle structure factor.

λ(x) is not re-evaluated: the weights come from the last field / Fisher
step (spec §10). Occupancy gradients of the modified model are ∂L/∂(occ w);
callers convert back with ``occupancy_grads_to_model``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from phridge.contrib.spatial_sigmaa_v2.moments import resolution_s2
from phridge.contrib.spatial_sigmaa_v2.packing import (
    PackedSpatialSigmaAV2,
    PackedSpatialSigmaAV2Result,
    _to_numpy,
)
from phridge.contrib.spatial_sigmaa_v2.per_atom import interpolate_shell, iso_u_star
from phridge.packing_xtal import PackedXray


def frozen_from_result(result: PackedSpatialSigmaAV2Result) -> tuple[np.ndarray, np.ndarray]:
    """``(w, U_iso)`` stored on the wire result. U_iso = tr(U)/3."""
    w = np.asarray(_to_numpy(result.w), dtype=np.float64).reshape(-1)
    u = np.asarray(_to_numpy(result.tr_U), dtype=np.float64).reshape(-1) / 3.0
    return w, u


def d0_from_block(block: PackedSpatialSigmaAV2, unit_cell: tuple, hkl: np.ndarray) -> np.ndarray:
    """Per-reflection D_0(s). Empty table → 1."""
    s2 = resolution_s2(tuple(float(x) for x in unit_cell), np.asarray(_to_numpy(hkl), dtype=np.int64))
    return interpolate_shell(
        s2,
        np.asarray(_to_numpy(block.shell_s2_edges), dtype=np.float64),
        np.asarray(_to_numpy(block.D0), dtype=np.float64),
        1.0,
    )


def xray_with_frozen_errors(xray: PackedXray, result: PackedSpatialSigmaAV2Result) -> PackedXray:
    """Occupancy w_j and ADP U_ADP + U_j. Spec eq. 25, atoms' x unchanged."""
    w, u_err = frozen_from_result(result)
    n = xray.sites_frac.shape[0]
    if w.shape[0] != n:
        raise ValueError("SpatialSigmaAV2Result n_scatterers does not match XrayStructure")
    aniso = np.array([bool(s.anisotropic) for s in xray.meta.scatterers], dtype=bool)
    occ = np.asarray(_to_numpy(xray.occupancy), dtype=np.float64) * w
    u_iso = np.asarray(_to_numpy(xray.u_iso), dtype=np.float64).copy()
    u_star = np.asarray(_to_numpy(xray.u_star), dtype=np.float64).copy()
    u_iso[~aniso] = u_iso[~aniso] + u_err[~aniso]
    if np.any(aniso):
        cell = tuple(float(x) for x in xray.meta.crystal.unit_cell)
        u_star[aniso] = u_star[aniso] + iso_u_star(cell, u_err[aniso])
    return PackedXray(
        crystal=xray.meta.crystal,
        sites_frac=np.asarray(_to_numpy(xray.sites_frac), dtype=np.float64),
        occupancy=occ,
        u_iso=u_iso,
        scatterers=list(xray.meta.scatterers),
        u_star=u_star,
    )


def occupancy_grads_to_model(d_occ_modified: np.ndarray, w: np.ndarray) -> np.ndarray:
    """∂L/∂occ = w · ∂L/∂(occ w)."""
    return (
        np.asarray(_to_numpy(d_occ_modified), dtype=np.float64).reshape(-1)
        * np.asarray(_to_numpy(w), dtype=np.float64).reshape(-1)
    )


def should_apply(block: Any, result: Any) -> bool:
    """True when a fitted v2 result can be folded into F_eff."""
    if block is None or result is None:
        return False
    if not isinstance(block, PackedSpatialSigmaAV2) or not isinstance(result, PackedSpatialSigmaAV2Result):
        return False
    if not block.meta.enabled:
        return False
    return int(result.w.shape[0]) > 0
