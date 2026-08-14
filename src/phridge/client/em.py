"""EM map helpers (iotbx.map_manager and packed EmMap)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from phridge.client.convert_xtal import density_at_sites, em_map_from_map_manager, em_map_to_map_manager
from phridge.packing import PackedMap
from phridge.packing_xtal import PackedEmMap, PackedHierarchy

__all__ = [
    "density_at_hierarchy",
    "density_at_sites",
    "em_map_as_real_map",
    "em_map_from_map_manager",
    "em_map_from_numpy",
    "em_map_to_map_manager",
]


def em_map_from_numpy(
    data: np.ndarray,
    crystal: Any,
    *,
    wrapping: bool = False,
    origin: Optional[list[int]] = None,
    origin_cart: Optional[list[float]] = None,
    pixel_sizes: Optional[list[float]] = None,
    experiment_type: str = "cryo_em",
    is_mask: bool = False,
    resolution: Optional[float] = None,
    label: Optional[str] = None,
) -> PackedEmMap:
    from phridge.client.convert import crystal_from_cctbx
    from phridge.models import CrystalSymmetry

    if not isinstance(crystal, CrystalSymmetry):
        crystal = crystal_from_cctbx(crystal)
    array = np.asarray(data)
    if array.ndim != 3:
        raise ValueError(f"expected 3-D map, got shape {array.shape}")
    return PackedEmMap(
        crystal=crystal,
        data=array,
        origin=list(origin or [0, 0, 0]),
        n_real=list(array.shape),
        experiment_type=experiment_type,
        wrapping=wrapping,
        is_mask=is_mask,
        pixel_sizes=pixel_sizes,
        origin_cart=origin_cart,
        resolution=resolution,
        label=label,
    )


def em_map_as_real_map(packed: PackedEmMap) -> PackedMap:
    """Drop EM-only metadata; same grid for workers that consume RealMap."""
    return PackedMap(crystal=packed.meta.crystal, data=packed.data, origin=list(packed.meta.origin))


def density_at_hierarchy(packed_em: PackedEmMap, hierarchy: PackedHierarchy) -> np.ndarray:
    """Sample the EM map at Hierarchy Cartesian sites (same i_seq order)."""
    return density_at_sites(packed_em, hierarchy.xyz)
