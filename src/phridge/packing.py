"""Canonical byte packing. Binary stays out of LinkML.

MillerArray and RealMap use one npz per object (see schema/cctbx.yaml).
Generic arrays use a .npy buffer. JSON is UTF-8.
"""

from __future__ import annotations

import io
import json
from typing import Any, Optional

import numpy as np

from phridge.models import (
    AnomalousLayout,
    CrystalSymmetry,
    MapSpace,
    MillerArray,
    ObservationType,
    RealMap,
)

CANONICAL_FLOAT = np.dtype("<f8")
CANONICAL_INT = np.dtype("<i4")
CANONICAL_COMPLEX = np.dtype("<c16")


def as_canonical_float(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array, dtype=CANONICAL_FLOAT)


def as_canonical_int(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array, dtype=CANONICAL_INT)


def as_canonical_complex(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array, dtype=CANONICAL_COMPLEX)


def canonical_data_array(array: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(array):
        return as_canonical_complex(array)
    return as_canonical_float(array)


def pack_npy(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(array), allow_pickle=False)
    return buf.getvalue()


def unpack_npy(blob: bytes) -> np.ndarray:
    buf = io.BytesIO(blob)
    return np.load(buf, allow_pickle=False)


def pack_json(value: Any) -> bytes:
    return json.dumps(value, default=_json_default).encode("utf-8")


def unpack_json(blob: bytes) -> Any:
    return json.loads(blob.decode("utf-8"))


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


class PackedMiller:
    """Numpy-facing miller array plus LinkML metadata."""

    def __init__(
        self,
        crystal: CrystalSymmetry,
        hkl: np.ndarray,
        data: np.ndarray,
        sigmas: Optional[np.ndarray] = None,
        *,
        anomalous: bool = False,
        observation_type: ObservationType = ObservationType.other,
        anomalous_layout: AnomalousLayout = AnomalousLayout.asu,
        label: Optional[str] = None,
    ) -> None:
        self.hkl = as_canonical_int(np.asarray(hkl))
        if self.hkl.ndim != 2 or self.hkl.shape[1] != 3:
            raise ValueError("hkl must have shape (N, 3)")
        self.data = canonical_data_array(np.asarray(data))
        if self.data.shape[0] != self.hkl.shape[0]:
            raise ValueError("hkl and data length mismatch")
        self.sigmas = None if sigmas is None else as_canonical_float(np.asarray(sigmas))
        if self.sigmas is not None and self.sigmas.shape[0] != self.hkl.shape[0]:
            raise ValueError("sigmas length mismatch")
        self.meta = MillerArray(
            label=label,
            crystal=crystal,
            anomalous=anomalous,
            observation_type=observation_type,
            anomalous_layout=anomalous_layout,
            n_refl=int(self.hkl.shape[0]),
            has_sigmas=self.sigmas is not None,
            index_dtype="int32",
            data_dtype="complex128" if np.iscomplexobj(self.data) else "float64",
        )

    def pack(self) -> bytes:
        payload: dict[str, np.ndarray] = {"hkl": self.hkl, "data": self.data}
        if self.sigmas is not None:
            payload["sigmas"] = self.sigmas
        buf = io.BytesIO()
        np.savez(buf, **payload)
        return buf.getvalue()


def unpack_miller(blob: bytes, meta: MillerArray) -> PackedMiller:
    with np.load(io.BytesIO(blob), allow_pickle=False) as zf:
        hkl = zf["hkl"]
        data = zf["data"]
        sigmas = zf["sigmas"] if "sigmas" in zf.files else None
    packed = PackedMiller(
        crystal=meta.crystal,
        hkl=hkl,
        data=data,
        sigmas=sigmas,
        anomalous=meta.anomalous,
        observation_type=meta.observation_type,
        anomalous_layout=meta.anomalous_layout,
        label=meta.label,
    )
    if packed.meta.n_refl != meta.n_refl:
        raise ValueError("n_refl does not match packed hkl")
    if packed.meta.has_sigmas != meta.has_sigmas:
        raise ValueError("has_sigmas does not match packed npz")
    return packed


class PackedMap:
    """Numpy-facing real map plus LinkML metadata."""

    def __init__(
        self,
        crystal: CrystalSymmetry,
        data: np.ndarray,
        origin: Optional[list[int]] = None,
        n_real: Optional[list[int]] = None,
    ) -> None:
        self.data = as_canonical_float(np.asarray(data))
        if self.data.ndim != 3:
            raise ValueError("real map data must be 3-D")
        # data is stored C-order (z, y, x) == shape (n_real[2], n_real[1], n_real[0])
        # when n_real is crystallographic (nx, ny, nz). If n_real is omitted,
        # treat data.shape as (nx, ny, nz) for the numpy-facing API.
        if n_real is None:
            n_real = [int(n) for n in self.data.shape]
        if list(self.data.shape) != list(n_real):
            raise ValueError(
                f"data.shape {self.data.shape} must equal n_real {n_real} "
                "(numpy-facing maps use shape (nx, ny, nz))"
            )
        self.meta = RealMap(
            crystal=crystal,
            origin=[0, 0, 0] if origin is None else [int(x) for x in origin],
            n_real=[int(x) for x in n_real],
            space=MapSpace.p1,
            dtype="float64",
        )

    def pack(self) -> bytes:
        buf = io.BytesIO()
        np.savez(buf, data=self.data)
        return buf.getvalue()


def unpack_map(blob: bytes, meta: RealMap) -> PackedMap:
    with np.load(io.BytesIO(blob), allow_pickle=False) as zf:
        data = zf["data"]
    return PackedMap(
        crystal=meta.crystal,
        data=data,
        origin=meta.origin,
        n_real=meta.n_real,
    )
