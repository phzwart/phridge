"""Single cctbx adapter. Import this package, not cctbx."""

from __future__ import annotations

from typing import Any

import numpy as np

from phridge.contracts.models import (
    CrystalGridding,
    CrystalSymmetry,
    ModelGeometry,
    SfEngineParams,
)
from phridge.contracts.packing import (
    PackedCartesian,
    PackedComplexMap,
    PackedEmMap,
    PackedFractional,
    PackedHL,
    PackedHierarchy,
    PackedMap,
    PackedMapCoefficients,
    PackedMiller,
    PackedReflections,
    PackedRestraints,
    PackedXray,
)
from phridge.q.error_models.spatial.packing import PackedSpatialSigmaAV2, PackedSpatialSigmaAV2Result
from phridge.sfcalc.packing import PackedScatteringTable, PackedSfCurvatures, PackedSfGradients, PackedTargetResult
from phridge.shims.cctbx._adptbx import grad_u_star_as_u_cart
from phridge.shims.cctbx._flex import is_numeric_flex
from phridge.shims.cctbx.hierarchy import (
    hierarchy_from_cctbx,
    hierarchy_to_cctbx,
    sites_cart_from_cctbx,
    sites_cart_to_cctbx,
    sites_frac_from_cctbx,
    sites_frac_to_cctbx,
)
from phridge.shims.cctbx.maps import (
    complex_map_from_cctbx,
    complex_map_to_numpy,
    density_at_hierarchy,
    density_at_sites,
    em_map_as_real_map,
    em_map_from_map_manager,
    em_map_from_numpy,
    em_map_to_map_manager,
    gridding_from_cctbx,
    map_coefficients_from_cctbx,
    map_coefficients_to_cctbx,
    map_from_cctbx,
    map_to_cctbx,
)
from phridge.shims.cctbx.miller import (
    hl_from_cctbx,
    hl_to_cctbx,
    miller_from_cctbx,
    miller_or_hl_from_cctbx,
    miller_to_cctbx,
    reflections_from_mtz,
    reflections_to_mtz,
)
from phridge.shims.cctbx.restraints import (
    ADPPriorOptions,
    build_adp_restraints,
    fit_adp_hyperparameters,
    model_geometry,
    restraints_from_cctbx,
    restraints_to_proxies,
)
from phridge.shims.cctbx.scattering import scattering_table_from_cctbx
from phridge.shims.cctbx.symmetry import crystal_from_cctbx, crystal_to_cctbx
from phridge.shims.cctbx.xray_structure import (
    from_robject,
    packed_order,
    random_xray_structure,
    to_robject,
    xray_from_cctbx,
    xray_structure_from_pdb,
    xray_to_cctbx,
)

__all__ = [
    "ADPPriorOptions",
    "build_adp_restraints",
    "complex_map_from_cctbx",
    "complex_map_to_numpy",
    "crystal_from_cctbx",
    "crystal_to_cctbx",
    "density_at_hierarchy",
    "density_at_sites",
    "em_map_as_real_map",
    "em_map_from_map_manager",
    "em_map_from_numpy",
    "em_map_to_map_manager",
    "fit_adp_hyperparameters",
    "from_canonical",
    "from_robject",
    "grad_u_star_as_u_cart",
    "gridding_from_cctbx",
    "has_cctbx",
    "hierarchy_from_cctbx",
    "hierarchy_to_cctbx",
    "hl_from_cctbx",
    "hl_to_cctbx",
    "is_cctbx_object",
    "map_coefficients_from_cctbx",
    "map_coefficients_to_cctbx",
    "map_from_cctbx",
    "map_to_cctbx",
    "miller_from_cctbx",
    "miller_or_hl_from_cctbx",
    "miller_to_cctbx",
    "model_geometry",
    "packed_order",
    "random_xray_structure",
    "reflections_from_mtz",
    "reflections_to_mtz",
    "restraints_from_cctbx",
    "restraints_to_proxies",
    "scattering_table_from_cctbx",
    "sites_cart_from_cctbx",
    "sites_cart_to_cctbx",
    "sites_frac_from_cctbx",
    "sites_frac_to_cctbx",
    "to_canonical",
    "to_robject",
    "xray_from_cctbx",
    "xray_structure_from_pdb",
    "xray_to_cctbx",
]


def has_cctbx() -> bool:
    try:
        import cctbx  # noqa: F401
    except ImportError:
        return False
    return True


def is_cctbx_object(value: Any) -> bool:
    if not has_cctbx():
        return False
    mod = type(value).__module__
    return mod.startswith("cctbx") or mod.startswith("iotbx") or mod.startswith("scitbx")


def to_canonical(value: Any) -> Any:
    """Convert a client value to a codec-encodable object."""
    packed = (
        PackedMiller,
        PackedMap,
        PackedHL,
        PackedReflections,
        PackedComplexMap,
        PackedMapCoefficients,
        PackedCartesian,
        PackedFractional,
        PackedHierarchy,
        PackedXray,
        PackedEmMap,
        PackedRestraints,
        PackedScatteringTable,
        PackedSfGradients,
        PackedSfCurvatures,
        PackedTargetResult,
        PackedSpatialSigmaAV2,
        PackedSpatialSigmaAV2Result,
        ModelGeometry,
        CrystalGridding,
        CrystalSymmetry,
        SfEngineParams,
        np.ndarray,
    )
    if isinstance(value, packed):
        return value
    if isinstance(value, dict) and "unit_cell" in value and "space_group_hall" in value:
        return CrystalSymmetry.model_validate(value)
    if is_cctbx_object(value):
        return _cctbx_to_canonical(value)
    return value


def from_canonical(value: Any, *, prefer_cctbx: bool) -> Any:
    if not prefer_cctbx or not has_cctbx():
        return value
    if isinstance(value, CrystalSymmetry):
        return crystal_to_cctbx(value)
    if isinstance(value, PackedMiller):
        return miller_to_cctbx(value)
    if isinstance(value, PackedMap):
        return map_to_cctbx(value)
    if isinstance(value, PackedHL):
        return hl_to_cctbx(value)
    if isinstance(value, PackedReflections):
        return reflections_to_mtz(value)
    if isinstance(value, PackedMapCoefficients):
        return map_coefficients_to_cctbx(value)
    if isinstance(value, PackedHierarchy):
        return hierarchy_to_cctbx(value)
    if isinstance(value, PackedXray):
        return xray_to_cctbx(value)
    if isinstance(value, PackedEmMap):
        return em_map_to_map_manager(value)
    if isinstance(value, PackedRestraints):
        return restraints_to_proxies(value)
    if isinstance(value, ModelGeometry):
        return value
    if isinstance(value, PackedCartesian):
        return sites_cart_to_cctbx(value)
    if isinstance(value, PackedFractional):
        return sites_frac_to_cctbx(value)
    if isinstance(value, CrystalGridding):
        return value
    return value


def _cctbx_to_canonical(value: Any) -> Any:
    from cctbx import crystal, maptbx, miller, xray
    from cctbx.geometry_restraints.manager import manager as restraints_manager
    from iotbx import mtz, pdb
    from iotbx.map_manager import map_manager

    if isinstance(value, xray.structure):  # subclass of crystal.symmetry: test first
        return xray_from_cctbx(value)
    if isinstance(value, miller.array):
        return miller_or_hl_from_cctbx(value)
    if isinstance(value, crystal.symmetry):
        return crystal_from_cctbx(value)
    if isinstance(value, miller.fft_map):
        return map_from_cctbx(value, value.crystal_symmetry())
    if isinstance(value, mtz.object):
        return reflections_from_mtz(value)
    if isinstance(value, pdb.hierarchy.root):
        return hierarchy_from_cctbx(value)
    if isinstance(value, maptbx.crystal_gridding):
        return gridding_from_cctbx(value)
    if isinstance(value, map_manager):
        return em_map_from_map_manager(value)
    if isinstance(value, restraints_manager):
        return restraints_from_cctbx(value)
    if is_numeric_flex(value):
        if getattr(value, "nd", 1) == 3:
            raise TypeError("3-D flex maps need crystal symmetry; use map_from_cctbx(data, crystal)")
        return np.asarray(value)
    raise TypeError(f"no cctbx converter for {type(value)!r}")


def require_cctbx_stack() -> None:
    """Import cctbx and mmtbx so a missing install fails at worker startup."""
    import cctbx  # noqa: F401
    import mmtbx  # noqa: F401
