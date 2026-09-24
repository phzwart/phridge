"""Worker-side helpers: canonical numpy <-> torch. Device is worker config."""

from __future__ import annotations

from typing import Any

import numpy as np

from phridge.packing import PackedMap, PackedMiller


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def to_torch(value: Any, device: str, compute_dtype: Any = None) -> Any:
    try:
        import torch
    except ImportError:
        return value

    if compute_dtype is None:
        compute_dtype = torch.float32
    if isinstance(value, PackedMiller):
        value.data = _ndarray_to_torch(value.data, device, compute_dtype)
        if value.sigmas is not None:
            value.sigmas = _ndarray_to_torch(value.sigmas, device, compute_dtype)
        return value
    if isinstance(value, PackedMap):
        value.data = _ndarray_to_torch(value.data, device, compute_dtype)
        return value
    if isinstance(value, np.ndarray):
        return _ndarray_to_torch(value, device, compute_dtype)
    # Spatial σ_A v2 is a numpy field / Fisher step. Do not lift its arrays
    # onto the compute device — apply.py / alternate.py read them on the host.
    if _is_host_spatial_sigma_a_v2(value):
        return value
    return value


def from_torch(value: Any) -> Any:
    if isinstance(value, PackedMiller):
        if _is_torch(value.data):
            value.data = _torch_to_canonical(value.data)
        if value.sigmas is not None and _is_torch(value.sigmas):
            value.sigmas = _torch_to_canonical(value.sigmas)
        value.meta.data_dtype = (
            "complex128" if np.iscomplexobj(value.data) else "float64"
        )
        return value
    if isinstance(value, PackedMap):
        if _is_torch(value.data):
            value.data = _torch_to_canonical(value.data)
        return value
    hosted = _host_spatial_sigma_a_v2(value)
    if hosted is not None:
        return hosted
    if _is_torch(value):
        return _torch_to_canonical(value)
    if isinstance(value, dict):
        return {k: from_torch(v) for k, v in value.items()}
    if isinstance(value, list):
        return [from_torch(v) for v in value]
    if isinstance(value, tuple):
        return tuple(from_torch(v) for v in value)
    return value


def _spatial_sigma_a_v2_types() -> tuple[type, ...]:
    try:
        from phridge.contrib.spatial_sigmaa_v2.packing import (
            PackedSpatialSigmaAV2,
            PackedSpatialSigmaAV2Result,
        )
    except ImportError:
        return ()
    return (PackedSpatialSigmaAV2, PackedSpatialSigmaAV2Result)


def _is_host_spatial_sigma_a_v2(value: Any) -> bool:
    return bool(_spatial_sigma_a_v2_types()) and isinstance(value, _spatial_sigma_a_v2_types())


def _host_spatial_sigma_a_v2(value: Any) -> Any:
    types = _spatial_sigma_a_v2_types()
    if not types or not isinstance(value, types):
        return None
    for name in getattr(value, "FIELDS", ()):
        arr = getattr(value, name, None)
        if _is_torch(arr):
            setattr(value, name, _torch_to_canonical(arr))
    return value


def _is_torch(value: Any) -> bool:
    try:
        import torch
    except ImportError:
        return False
    return isinstance(value, torch.Tensor)


def _ndarray_to_torch(array: np.ndarray, device: str, compute_dtype: Any) -> Any:
    import torch

    tensor = torch.from_numpy(np.ascontiguousarray(array))
    if device.startswith("mps"):
        if tensor.is_floating_point():
            tensor = tensor.to(dtype=torch.float32)
        elif tensor.is_complex():
            tensor = tensor.to(dtype=torch.complex64)
    elif tensor.is_floating_point():
        if compute_dtype is not None:
            tensor = tensor.to(dtype=compute_dtype)
    return tensor.to(device)


def _torch_to_canonical(tensor: Any) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    if np.iscomplexobj(array):
        return np.ascontiguousarray(array, dtype=np.complex128)
    if np.issubdtype(array.dtype, np.floating):
        return np.ascontiguousarray(array, dtype=np.float64)
    if np.issubdtype(array.dtype, np.integer):
        return np.ascontiguousarray(array, dtype=array.dtype)
    return np.ascontiguousarray(array)
