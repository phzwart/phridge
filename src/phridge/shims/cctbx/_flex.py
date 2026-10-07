"""The only module that imports flex arrays."""

from __future__ import annotations

from typing import Any

import numpy as np


def _flex():
    from scitbx.array_family import flex

    return flex


def _cctbx_flex():
    from cctbx.array_family import flex as cctbx_flex

    return cctbx_flex


def flex_to_numpy(value: Any) -> np.ndarray:
    return np.asarray(value)


def is_numeric_flex(value: Any) -> bool:
    flex = _flex()
    return isinstance(value, (flex.double, flex.bool, flex.int, flex.size_t, flex.complex_double, flex.float))


def double(values: Any) -> Any:
    return _flex().double(values)


def complex_double(values: Any) -> Any:
    return _flex().complex_double(values)


def vec3_double(rows: Any) -> Any:
    return _flex().vec3_double(rows)


def sym_mat3_double(rows: Any) -> Any:
    return _flex().sym_mat3_double(rows)


def grid(shape: tuple[int, ...]) -> Any:
    return _flex().grid(shape)


def miller_index(values: Any) -> Any:
    return _cctbx_flex().miller_index(values)


def hendrickson_lattman(rows: Any) -> Any:
    return _cctbx_flex().hendrickson_lattman(rows)


def xray_scatterer(items: Any) -> Any:
    return _cctbx_flex().xray_scatterer(items)


def numpy_to_flex(values: Any, *, kind: str = "double") -> Any:
    if kind == "double":
        return double(values)
    if kind == "complex_double":
        return complex_double(values)
    if kind == "vec3_double":
        return vec3_double(values)
    if kind == "sym_mat3_double":
        return sym_mat3_double(values)
    if kind == "miller_index":
        return miller_index(values)
    if kind == "hendrickson_lattman":
        return hendrickson_lattman(values)
    if kind == "xray_scatterer":
        return xray_scatterer(values)
    raise ValueError(f"unknown flex kind {kind!r}")
