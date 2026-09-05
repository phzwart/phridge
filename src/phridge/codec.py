"""Encode and decode canonical objects to Redis ObjectRefs."""

from __future__ import annotations

from typing import Any, Callable

from phridge.models import (
    CCTBX_TYPES,
    ArrayMeta,
    CrystalGridding,
    CrystalSymmetry,
    ModelGeometry,
    ObjectKind,
    ObjectRef,
    SfEngineParams,
)
from phridge.packing import (
    PackedMap,
    PackedMiller,
    pack_json,
    pack_npy,
    unpack_json,
    unpack_map,
    unpack_miller,
    unpack_npy,
)
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
    unpack_cartesian,
    unpack_complex_map,
    unpack_em_map,
    unpack_fractional,
    unpack_hierarchy,
    unpack_hl,
    unpack_map_coefficients,
    unpack_reflections,
    unpack_xray,
)
from phridge.packing_geometry import PackedRestraints, unpack_restraints
from phridge.packing_scattering import (
    PackedScatteringTable,
    PackedSfCurvatures,
    PackedSfGradients,
    PackedTargetResult,
    unpack_scattering_table,
    unpack_sf_curvatures,
    unpack_sf_gradients,
    unpack_target_result,
)
from phridge.redis_store import RedisStore

_PACKED_TYPES: list[tuple[type, str]] = [
    (PackedMiller, "MillerArray"),
    (PackedMap, "RealMap"),
    (PackedHL, "HendricksonLattman"),
    (PackedReflections, "ReflectionFile"),
    (PackedComplexMap, "ComplexMap"),
    (PackedMapCoefficients, "MapCoefficients"),
    (PackedCartesian, "CartesianSites"),
    (PackedFractional, "FractionalSites"),
    (PackedHierarchy, "Hierarchy"),
    (PackedXray, "XrayStructure"),
    (PackedEmMap, "EmMap"),
    (PackedRestraints, "GeometryRestraints"),
    (PackedScatteringTable, "ScatteringTable"),
    (PackedSfGradients, "SfGradients"),
    (PackedSfCurvatures, "SfCurvatures"),
    (PackedTargetResult, "TargetResult"),
]

_JSON_ONLY = {
    CrystalSymmetry: "CrystalSymmetry",
    CrystalGridding: "CrystalGridding",
    ModelGeometry: "ModelGeometry",
    SfEngineParams: "SfEngineParams",
}

_UNPACK: dict[str, Callable[[bytes, Any], Any]] = {
    "MillerArray": unpack_miller,
    "RealMap": unpack_map,
    "HendricksonLattman": unpack_hl,
    "ReflectionFile": unpack_reflections,
    "ComplexMap": unpack_complex_map,
    "MapCoefficients": unpack_map_coefficients,
    "CartesianSites": unpack_cartesian,
    "FractionalSites": unpack_fractional,
    "Hierarchy": unpack_hierarchy,
    "XrayStructure": unpack_xray,
    "EmMap": unpack_em_map,
    "GeometryRestraints": unpack_restraints,
    "ScatteringTable": unpack_scattering_table,
    "SfGradients": unpack_sf_gradients,
    "SfCurvatures": unpack_sf_curvatures,
    "TargetResult": unpack_target_result,
}


def encode_value(store: RedisStore, job_id: str, name: str, value: Any) -> ObjectRef:
    for cls, cctbx_type in _PACKED_TYPES:
        if isinstance(value, cls):
            key = store.obj_key(job_id, name)
            store.put_bytes(key, value.pack())
            return ObjectRef(
                kind=ObjectKind.cctbx,
                key=key,
                cctbx_type=cctbx_type,
                meta=value.meta.model_dump(mode="json"),
            )
    for cls, cctbx_type in _JSON_ONLY.items():
        if isinstance(value, cls):
            return ObjectRef(
                kind=ObjectKind.cctbx,
                cctbx_type=cctbx_type,
                meta=value.model_dump(mode="json"),
            )
    if type(value).__name__ in CCTBX_TYPES and not hasattr(value, "pack"):
        raise TypeError(f"{type(value).__name__} metadata needs buffers; pass the Packed* type")
    if _is_numpy_array(value):
        import numpy as np

        array = np.ascontiguousarray(value)
        key = store.obj_key(job_id, name)
        store.put_bytes(key, pack_npy(array))
        meta = ArrayMeta(dtype=str(array.dtype), shape=[int(n) for n in array.shape])
        return ObjectRef(
            kind=ObjectKind.array,
            key=key,
            dtype=meta.dtype,
            shape=meta.shape,
            meta=meta.model_dump(),
        )
    key = store.obj_key(job_id, name)
    store.put_bytes(key, pack_json(value))
    return ObjectRef(kind=ObjectKind.json, key=key)


def decode_ref(store: RedisStore, ref: ObjectRef) -> Any:
    if ref.kind == ObjectKind.cctbx:
        if not ref.cctbx_type:
            raise ValueError("cctbx ObjectRef missing cctbx_type")
        cls = CCTBX_TYPES[ref.cctbx_type]
        if ref.meta is None:
            raise ValueError("cctbx ObjectRef missing meta")
        meta = cls.model_validate(ref.meta)
        if ref.cctbx_type in _JSON_ONLY.values():
            return meta
        if ref.key is None:
            raise ValueError(f"{ref.cctbx_type} requires a packed blob key")
        unpack = _UNPACK.get(ref.cctbx_type)
        if unpack is None:
            raise TypeError(f"{ref.cctbx_type} has no unpacker")
        return unpack(store.get_bytes(ref.key), meta)
    if ref.kind == ObjectKind.array:
        if ref.key is None:
            raise ValueError("array ObjectRef missing key")
        return unpack_npy(store.get_bytes(ref.key))
    if ref.kind == ObjectKind.json:
        if ref.key is None:
            raise ValueError("json ObjectRef missing key")
        return unpack_json(store.get_bytes(ref.key))
    if ref.kind == ObjectKind.blob:
        if ref.key is None:
            raise ValueError("blob ObjectRef missing key")
        return store.get_bytes(ref.key)
    raise ValueError(f"unknown kind: {ref.kind}")


def _is_numpy_array(value: Any) -> bool:
    try:
        import numpy as np
    except ImportError:
        return False
    return isinstance(value, np.ndarray)
