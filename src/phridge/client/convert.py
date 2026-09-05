"""Client-side converters: cctbx/numpy <-> canonical packed types."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from phridge.models import (
    AnomalousLayout,
    CrystalGridding,
    CrystalSymmetry,
    ModelGeometry,
    ObservationType,
    SfEngineParams,
    SymOp,
)
from phridge.packing import PackedMap, PackedMiller
from phridge.packing_xtal import (
    PackedCartesian,
    PackedComplexMap,
    PackedEmMap,
    PackedFractional,
    PackedHL,
    PackedHierarchy,
    PackedMapCoefficients,
    PackedReflections,
    PackedXray,
)
from phridge.packing_geometry import PackedRestraints
from phridge.packing_scattering import PackedScatteringTable, PackedSfCurvatures, PackedSfGradients, PackedTargetResult

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
    from phridge.client import convert_xtal as xtal

    if isinstance(value, PackedHL):
        return xtal.hl_to_cctbx(value)
    if isinstance(value, PackedReflections):
        return xtal.reflections_to_mtz(value)
    if isinstance(value, PackedMapCoefficients):
        return xtal.map_coefficients_to_cctbx(value)
    if isinstance(value, PackedHierarchy):
        return xtal.hierarchy_to_cctbx(value)
    if isinstance(value, PackedXray):
        return xtal.xray_to_cctbx(value)
    if isinstance(value, PackedEmMap):
        return xtal.em_map_to_map_manager(value)
    if isinstance(value, PackedRestraints):
        from phridge.client.convert_geometry import restraints_to_proxies

        return restraints_to_proxies(value)
    if isinstance(value, ModelGeometry):
        return value
    if isinstance(value, PackedCartesian):
        return xtal.sites_cart_to_cctbx(value)
    if isinstance(value, PackedFractional):
        return xtal.sites_frac_to_cctbx(value)
    if isinstance(value, CrystalGridding):
        return value
    return value


def _cctbx_to_canonical(value: Any) -> Any:
    from cctbx import crystal, miller, xray, maptbx
    from iotbx import mtz, pdb
    from scitbx.array_family import flex

    from phridge.client import convert_xtal as xtal

    from cctbx.geometry_restraints.manager import manager as restraints_manager
    from iotbx.map_manager import map_manager

    if isinstance(value, xray.structure):  # subclass of crystal.symmetry: test first
        return xtal.xray_from_cctbx(value)
    if isinstance(value, crystal.symmetry):
        return crystal_from_cctbx(value)
    if isinstance(value, miller.array):
        return xtal.miller_or_hl_from_cctbx(value)
    if isinstance(value, miller.fft_map):
        return map_from_cctbx(value, value.crystal_symmetry())
    if isinstance(value, mtz.object):
        return xtal.reflections_from_mtz(value)
    if isinstance(value, pdb.hierarchy.root):
        return xtal.hierarchy_from_cctbx(value)
    if isinstance(value, xray.structure):
        return xtal.xray_from_cctbx(value)
    if isinstance(value, maptbx.crystal_gridding):
        return xtal.gridding_from_cctbx(value)
    if isinstance(value, map_manager):
        return xtal.em_map_from_map_manager(value)
    if isinstance(value, restraints_manager):
        from phridge.client.convert_geometry import restraints_from_cctbx

        return restraints_from_cctbx(value)
    if isinstance(value, flex.double) and getattr(value, "nd", 1) == 3:
        raise TypeError("3-D flex maps need crystal symmetry; use map_from_cctbx(data, crystal)")
    raise TypeError(f"no cctbx converter for {type(value)!r}")


def crystal_from_cctbx(sym: Any, *, with_symops: bool = True) -> CrystalSymmetry:
    info = sym.space_group_info()
    hall = str(info.type().hall_symbol())
    symops = None
    if with_symops:
        symops = [
            SymOp(r=[float(x) for x in op.r().as_double()], t=[float(x) for x in op.t().as_double()])
            for op in sym.space_group()
        ]
    return CrystalSymmetry(
        unit_cell=[float(x) for x in sym.unit_cell().parameters()],
        space_group_hall=hall,
        space_group_number=int(info.type().number()),
        symops=symops,
    )


def crystal_to_cctbx(cs: CrystalSymmetry) -> Any:
    from cctbx import crystal, sgtbx

    try:
        symbols = sgtbx.space_group_symbols(cs.space_group_hall)
        group = sgtbx.space_group(symbols.hall())
    except Exception:
        if cs.space_group_number is None:
            raise ValueError(
                f"cannot reconstruct crystal.symmetry from Hall {cs.space_group_hall!r}"
            ) from None
        try:
            symbols = sgtbx.space_group_symbols(cs.space_group_number)
            group = sgtbx.space_group(symbols.hall())
        except Exception as exc:
            raise ValueError(
                f"cannot reconstruct crystal.symmetry from Hall {cs.space_group_hall!r} "
                f"or number {cs.space_group_number}"
            ) from exc
    return crystal.symmetry(unit_cell=tuple(cs.unit_cell), space_group=group)


def miller_from_cctbx(
    array: Any,
    *,
    anomalous_layout: Optional[AnomalousLayout] = None,
) -> PackedMiller:
    anomalous = bool(array.anomalous_flag())
    if anomalous_layout is None:
        anomalous_layout = (
            AnomalousLayout.both_hemispheres if anomalous else AnomalousLayout.asu
        )
    hkl = np.asarray(list(array.indices()), dtype=np.int32)
    data = np.asarray(list(array.data()))
    sigmas = None
    if array.sigmas() is not None:
        sigmas = np.asarray(list(array.sigmas()))
    label = None
    info = array.info()
    if info is not None and getattr(info, "labels", None):
        label = str(info.labels[0])
    return PackedMiller(
        crystal=crystal_from_cctbx(array.crystal_symmetry()),
        hkl=hkl,
        data=data,
        sigmas=sigmas,
        anomalous=anomalous,
        observation_type=_observation_type(array),
        anomalous_layout=anomalous_layout,
        label=label,
    )


def miller_to_cctbx(packed: PackedMiller) -> Any:
    from cctbx import miller
    from cctbx.array_family import flex as cctbx_flex
    from scitbx.array_family import flex

    crystal = crystal_to_cctbx(packed.meta.crystal)
    indices = cctbx_flex.miller_index([tuple(int(x) for x in row) for row in packed.hkl])
    miller_set = miller.set(crystal, indices, anomalous_flag=packed.meta.anomalous)
    if np.iscomplexobj(packed.data):
        data = flex.complex_double(list(packed.data))
    else:
        data = flex.double(list(packed.data))
    result = miller.array(miller_set=miller_set, data=data)
    if packed.sigmas is not None:
        result = result.customized_copy(sigmas=flex.double(list(packed.sigmas)))
    return result


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
    from scitbx.array_family import flex

    flat = packed.data.reshape(-1)
    grid = flex.grid(tuple(int(n) for n in packed.meta.n_real))
    result = flex.double(list(flat))
    result.reshape(grid)
    return result


def _observation_type(array: Any) -> ObservationType:
    if array.is_complex_array():
        return ObservationType.complex
    if array.is_hendrickson_lattman_array():
        return ObservationType.hl
    if array.is_xray_intensity_array():
        return ObservationType.iobs
    if array.is_xray_amplitude_array():
        return ObservationType.fobs
    if array.is_real_array():
        return ObservationType.amplitude
    return ObservationType.other


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
