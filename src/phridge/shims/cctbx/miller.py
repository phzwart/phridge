"""Miller array, Hendrickson-Lattman, and MTZ adapter."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from phridge.contracts.models import AnomalousLayout, MtzColumnType, ObservationType, ReflectionColumn
from phridge.contracts.packing import PackedHL, PackedMiller, PackedReflections
from phridge.shims.cctbx._flex import complex_double, double, hendrickson_lattman, miller_index
from phridge.shims.cctbx.symmetry import crystal_from_cctbx, crystal_to_cctbx


from phridge.contracts.manifest import PluginManifest, Slot, Space, Tier

MANIFEST = PluginManifest(
    name='cctbx_miller',
    slot=Slot.io_adapter,
    space=Space.D,
    tier=Tier.shim,
)

_MTZ_TYPE = {item.value: item for item in MtzColumnType}

_MTZ_OBS = {
    "F": ObservationType.fobs,
    "G": ObservationType.fobs,
    "K": ObservationType.fobs,
    "J": ObservationType.iobs,
    "L": ObservationType.iobs,
    "M": ObservationType.iobs,
    "A": ObservationType.hl,
    "B": ObservationType.hl,
    "C": ObservationType.hl,
}

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
        
    crystal = crystal_to_cctbx(packed.meta.crystal)
    try:
        hkl_arr = np.ascontiguousarray(packed.hkl, dtype=np.int32)
        indices = miller_index(hkl_arr)
    except Exception:
        indices = miller_index([tuple(int(x) for x in row) for row in packed.hkl])
    miller_set = miller.set(crystal, indices, anomalous_flag=packed.meta.anomalous)
    if np.iscomplexobj(packed.data):
        cdata = np.ascontiguousarray(packed.data, dtype=np.complex128)
        try:
            data = complex_double(cdata)
        except Exception:
            data = complex_double(list(cdata))
    else:
        rdata = np.ascontiguousarray(packed.data, dtype=np.float64)
        try:
            data = double(rdata)
        except Exception:
            data = double(list(rdata))
    result = miller.array(miller_set=miller_set, data=data)
    if packed.sigmas is not None:
        sdata = np.ascontiguousarray(packed.sigmas, dtype=np.float64)
        try:
            sigmas_flex = double(sdata)
        except Exception:
            sigmas_flex = double(list(sdata))
        result = result.customized_copy(sigmas=sigmas_flex)
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

def _label_of(array: Any) -> Optional[str]:
    info = array.info()
    if info is None or not info.labels:
        return None
    return str(info.labels[0])


def miller_or_hl_from_cctbx(array: Any, **kwargs: Any) -> PackedMiller | PackedHL:
    if array.is_hendrickson_lattman_array():
        return hl_from_cctbx(array, **kwargs)
    packed = miller_from_cctbx(array, **kwargs)
    return packed


def hl_from_cctbx(array: Any, *, anomalous_layout: Optional[AnomalousLayout] = None) -> PackedHL:
    anomalous = bool(array.anomalous_flag())
    if anomalous_layout is None:
        anomalous_layout = (
            AnomalousLayout.both_hemispheres if anomalous else AnomalousLayout.asu
        )
    rows = [tuple(float(x) for x in row) for row in array.data()]
    a, b, c, d = zip(*rows) if rows else ([], [], [], [])
    return PackedHL(
        crystal=crystal_from_cctbx(array.crystal_symmetry()),
        hkl=np.asarray(list(array.indices()), dtype=np.int32),
        a=np.asarray(a),
        b=np.asarray(b),
        c=np.asarray(c),
        d=np.asarray(d),
        anomalous=anomalous,
        anomalous_layout=anomalous_layout,
        label=_label_of(array),
    )


def hl_to_cctbx(packed: PackedHL) -> Any:
    from cctbx import miller
    
    crystal = crystal_to_cctbx(packed.meta.crystal)
    indices = miller_index([tuple(int(x) for x in row) for row in packed.hkl])
    miller_set = miller.set(crystal, indices, anomalous_flag=packed.meta.anomalous)
    hl = hendrickson_lattman(
        list(zip(packed.a.tolist(), packed.b.tolist(), packed.c.tolist(), packed.d.tolist()))
    )
    return miller.array(miller_set=miller_set, data=hl)


def reflections_from_mtz(obj: Any) -> PackedReflections:
    labels = list(obj.column_labels())
    types = list(obj.column_types())
    hkl = np.asarray(list(obj.extract_miller_indices()), dtype=np.int32)
    crystals = list(obj.crystals())
    crystal = crystal_from_cctbx(crystals[-1].crystal_symmetry())
    wavelength = None
    for cryst in crystals:
        for dataset in cryst.datasets():
            wave = float(dataset.wavelength())
            if wave > 0:
                wavelength = wave
    history = [str(line) for line in obj.history()]
    columns: list[ReflectionColumn] = []
    data: dict[str, np.ndarray] = {}
    sigmas: dict[str, np.ndarray] = {}
    last_data_name: Optional[str] = None
    anomalous_layout = AnomalousLayout.asu
    for label, raw in zip(labels, types):
        raw = str(raw)
        if raw == "H":
            continue
        values = np.asarray(list(obj.get_column(label).extract_values()), dtype=np.float64)
        if raw == "Q" and last_data_name is not None:
            sigmas[last_data_name] = values
            columns[-1].has_sigmas = True
            continue
        mtz_type = _MTZ_TYPE.get(raw, MtzColumnType.other)
        if raw in {"G", "K", "L", "M"}:
            anomalous_layout = AnomalousLayout.both_hemispheres
        col = ReflectionColumn(
            label=str(label),
            mtz_type=mtz_type,
            mtz_type_raw=None if mtz_type is not MtzColumnType.other else raw,
            observation_type=_MTZ_OBS.get(raw, ObservationType.other),
            anomalous=raw in {"G", "K", "L", "M", "D"},
            has_sigmas=False,
            npz_name=str(label),
            data_dtype="float64",
        )
        columns.append(col)
        data[col.npz_name] = values
        last_data_name = col.npz_name
    if not columns:
        raise ValueError("mtz object has no data columns")
    return PackedReflections(
        crystal=crystal,
        hkl=hkl,
        columns=columns,
        data=data,
        sigmas=sigmas,
        wavelength=wavelength,
        history=history,
        anomalous_layout=anomalous_layout,
    )


def reflections_to_mtz(packed: PackedReflections) -> Any:
    arrays = []
    for col in packed.meta.columns:
        miller = PackedMiller(
            crystal=packed.meta.crystal,
            hkl=packed.hkl,
            data=packed.data[col.npz_name],
            sigmas=packed.sigmas.get(col.npz_name),
            anomalous=col.anomalous,
            observation_type=col.observation_type,
            anomalous_layout=packed.meta.anomalous_layout,
            label=col.label,
        )
        array = miller_to_cctbx(miller)
        if col.observation_type is ObservationType.fobs:
            array = array.set_observation_type_xray_amplitude()
        elif col.observation_type is ObservationType.iobs:
            array = array.set_observation_type_xray_intensity()
        arrays.append((col.label, array))
    root, first = arrays[0]
    dataset = first.as_mtz_dataset(column_root_label=root)
    for label, array in arrays[1:]:
        dataset.add_miller_array(array, column_root_label=label)
    return dataset.mtz_object()
