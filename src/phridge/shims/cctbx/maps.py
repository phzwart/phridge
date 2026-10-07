"""Real, complex, and EM map adapter."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from phridge.contracts.models import CrystalGridding, CrystalSymmetry, MapSpace, ObservationType
from phridge.contracts.packing import PackedComplexMap, PackedEmMap, PackedHierarchy, PackedMap, PackedMapCoefficients
from phridge.shims.cctbx._flex import double, grid as flex_grid, vec3_double
from phridge.shims.cctbx.miller import _label_of, miller_from_cctbx, miller_to_cctbx
from phridge.shims.cctbx.symmetry import crystal_from_cctbx, crystal_to_cctbx


from phridge.contracts.manifest import PluginManifest, Slot, Space, Tier

MANIFEST = PluginManifest(
    name='cctbx_maps',
    slot=Slot.io_adapter,
    space=Space.D,
    tier=Tier.shim,
)

def map_from_cctbx(
    data: Any,
    crystal: Any,
    origin: Optional[list[int]] = None,
) -> PackedMap:
    array = _as_numpy_3d(data)
    if hasattr(crystal, "unit_cell"):
        crystal = crystal_from_cctbx(crystal)
    if not isinstance(crystal, CrystalSymmetry):
        raise TypeError("crystal must be CrystalSymmetry or cctbx.crystal.symmetry")
    return PackedMap(crystal=crystal, data=array, origin=origin)


def map_to_cctbx(packed: PackedMap) -> Any:
    
    flat = np.ascontiguousarray(packed.data.reshape(-1), dtype=np.float64)
    grid = flex_grid(tuple(int(n) for n in packed.meta.n_real))
    try:
        result = double(flat)
    except Exception:
        result = double(list(flat))
    result.reshape(grid)
    return result

def _as_numpy_3d(data: Any) -> np.ndarray:
    if isinstance(data, np.ndarray):
        array = data
    elif hasattr(data, "real_map_unpadded"):
        return _as_numpy_3d(data.real_map_unpadded())
    elif hasattr(data, "as_numpy_array"):
        array = np.asarray(data.as_numpy_array())
    else:
        array = np.asarray(list(data))
    if array.ndim != 3:
        raise ValueError(f"expected 3-D map, got shape {array.shape}")
    return array

def map_coefficients_from_cctbx(array: Any) -> PackedMapCoefficients:
    if not array.is_complex_array():
        raise TypeError("map coefficients require a complex miller array")
    packed = miller_from_cctbx(array)
    packed.meta.label = _label_of(array)
    packed.meta.observation_type = ObservationType.complex
    return PackedMapCoefficients(packed, label=packed.meta.label)


def map_coefficients_to_cctbx(packed: PackedMapCoefficients) -> Any:
    return miller_to_cctbx(packed.miller)


def complex_map_from_cctbx(
    data: Any,
    crystal: Any,
    origin: Optional[list[int]] = None,
) -> PackedComplexMap:
    array = np.asarray(data if isinstance(data, np.ndarray) else list(data))
    if array.ndim != 3:
        raise ValueError(f"expected 3-D complex map, got shape {array.shape}")
    if hasattr(crystal, "unit_cell"):
        crystal = crystal_from_cctbx(crystal)
    return PackedComplexMap(crystal=crystal, data=array, origin=origin, space=MapSpace.p1)


def complex_map_to_numpy(packed: PackedComplexMap) -> np.ndarray:
    return packed.data


def gridding_from_cctbx(grid: Any) -> CrystalGridding:
    crystal = crystal_from_cctbx(grid.crystal_symmetry())
    n_real = [int(x) for x in grid.n_real()]
    d_min = float(grid.d_min()) if grid.d_min() is not None else None
    resolution_factor = float(grid.resolution_factor()) if grid.resolution_factor() is not None else None
    return CrystalGridding(
        crystal=crystal,
        n_real=n_real,
        d_min=d_min,
        resolution_factor=resolution_factor,
        space=MapSpace.p1,
    )

def em_map_from_map_manager(mm: Any, label: Optional[str] = None) -> Any:
    from phridge.contracts.packing import PackedEmMap

    data = mm.map_data()
    array = np.asarray(data.as_numpy_array()) if hasattr(data, "as_numpy_array") else np.asarray(list(data)).reshape(tuple(data.all()))
    origin = [int(x) for x in data.origin()] if hasattr(data, "origin") else [0, 0, 0]
    n_real = [int(x) for x in (mm.map_data().all() if hasattr(mm.map_data(), "all") else array.shape)]
    crystal = crystal_from_cctbx(mm.crystal_symmetry())
    pixels = None
    if hasattr(mm, "pixel_sizes"):
        pixels = [float(x) for x in mm.pixel_sizes()]
    origin_cart = None
    if hasattr(mm, "shift_cart"):
        origin_cart = [float(x) for x in mm.shift_cart()]
    exp = mm.experiment_type() if hasattr(mm, "experiment_type") else "cryo_em"
    if exp in (None, ""):
        exp = "other"
    wrapping = bool(mm.wrapping()) if hasattr(mm, "wrapping") else False
    resolution = None
    if hasattr(mm, "resolution") and mm.resolution() not in (None, 0, -1):
        try:
            resolution = float(mm.resolution())
        except Exception:
            resolution = None
    is_mask = bool(mm.is_mask()) if hasattr(mm, "is_mask") else False
    return PackedEmMap(
        crystal=crystal,
        data=array,
        origin=origin,
        n_real=list(array.shape),
        experiment_type=str(exp),
        wrapping=wrapping,
        is_mask=is_mask,
        pixel_sizes=pixels,
        origin_cart=origin_cart,
        resolution=resolution,
        label=label,
    )


def em_map_to_map_manager(packed: Any) -> Any:
    from iotbx.map_manager import map_manager
    
    grid = flex_grid(tuple(int(n) for n in packed.meta.n_real))
    flat = np.ascontiguousarray(packed.data.reshape(-1), dtype=np.float64)
    try:
        data = double(flat)
    except Exception:
        data = double(list(flat))
    data.reshape(grid)
    mm = map_manager(
        map_data=data,
        unit_cell_grid=tuple(int(n) for n in packed.meta.n_real),
        unit_cell_crystal_symmetry=crystal_to_cctbx(packed.meta.crystal),
        wrapping=packed.meta.wrapping,
    )
    if packed.meta.experiment_type:
        try:
            mm.set_experiment_type(packed.meta.experiment_type.value)
        except Exception:
            pass
    if packed.meta.resolution is not None:
        try:
            mm.set_resolution(packed.meta.resolution)
        except Exception:
            pass
    return mm


def density_at_sites(packed_em: Any, sites_cart: np.ndarray) -> np.ndarray:
    """Sample an EM/real map at Cartesian coordinates (Å)."""
    
    mm = em_map_to_map_manager(packed_em)
    sites = vec3_double([tuple(float(x) for x in row) for row in np.asarray(sites_cart)])
    return np.asarray(list(mm.density_at_sites_cart(sites)), dtype=np.float64)

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
    from phridge.shims.cctbx.symmetry import crystal_from_cctbx
    from phridge.contracts.models import CrystalSymmetry

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
