"""Worker op ``ml_i_maps``: map coefficients from the intensity likelihood posterior.

Inputs mirror ``target_eval`` (same ``f_obs`` / per-reflection arrays / target spec) plus a
``maps`` JSON dict validated as :class:`IntensityMapOptions`. Outputs are complex
``MillerArray`` coefficient sets on the ``f_obs`` hkl list and per-reflection weight arrays.
The impl imports torch lazily so a Phenix client can import this module.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from phridge.ops import register_op
from phridge.packing import PackedMiller

OP_NAME = "ml_i_maps"

_INPUTS = {
    "f_calc": "MillerArray",
    "f_obs": "MillerArray",
    "target": "json",
    "weights": "array",
    "r_free": "array",
    "alpha": "array",
    "beta": "array",
    "epsilon": "array",
    "centric": "array",
    "maps": "json",
}
_OUTPUTS = {
    "difference": "MillerArray",
    "model": "MillerArray",
    "gradient": "MillerArray",
    "newton": "MillerArray",
    "fom": "array",
    "robust_weight": "array",
    "curvature": "array",
    "f_post": "array",
    "d_loglik_d_nu": "array",
    "stats": "json",
}


def ml_i_maps(
    f_calc: PackedMiller,
    f_obs: PackedMiller,
    target: dict,
    weights: Optional[Any] = None,
    r_free: Optional[Any] = None,
    alpha: Optional[Any] = None,
    beta: Optional[Any] = None,
    epsilon: Optional[Any] = None,
    centric: Optional[Any] = None,
    maps: Optional[dict] = None,
) -> dict[str, Any]:
    """Worker implementation (torch side)."""
    import torch

    from phridge.contrib.intensity_ll.maps import IntensityMapOptions, intensity_map_coefficients
    from phridge.contrib.intensity_ll.target import IntensityLogLikelihood
    from phridge.sfcalc.ops import _DEVICE, _miller_like, _np, _observations
    from phridge.sfcalc.targets import build_target

    if _np(f_calc.hkl).shape != _np(f_obs.hkl).shape or not np.array_equal(_np(f_calc.hkl), _np(f_obs.hkl)):
        raise ValueError("f_calc and f_obs must be on identical hkl lists")
    spec = dict(target)
    if spec.get("name") != "ml_i":
        raise ValueError(f"{OP_NAME} needs an ml_i target spec, got {spec.get('name')!r}")
    tgt = build_target(spec)
    assert isinstance(tgt, IntensityLogLikelihood)
    opts = IntensityMapOptions.model_validate(maps or {})
    obs = _observations(f_obs, weights, r_free, alpha, beta, epsilon, centric)
    fc = torch.as_tensor(_np(f_calc.data, np.complex128), dtype=torch.complex128, device=_DEVICE["device"])
    out = intensity_map_coefficients(tgt, fc, obs, opts)
    n = fc.shape[0]
    return {
        "difference": _miller_like(f_obs, out.difference, "ML_I_DIFF"),
        "model": _miller_like(f_obs, out.model, "ML_I_MODEL"),
        "gradient": _miller_like(f_obs, out.gradient, "ML_I_GRAD"),
        "newton": _miller_like(f_obs, out.newton, "ML_I_NEWTON"),
        "fom": out.fom,
        "robust_weight": out.robust_weight,
        "curvature": out.curvature,
        "f_post": out.f_post,
        "d_loglik_d_nu": np.zeros(n, dtype=np.float64) if out.d_loglik_d_nu is None else out.d_loglik_d_nu,
        "stats": out.stats,
    }


ml_i_maps.compute_dtype = "float64"  # type: ignore[attr-defined]


def register_ops() -> None:
    """Register ``ml_i_maps`` in the shared op catalog with its worker impl (idempotent)."""
    register_op(OP_NAME, ml_i_maps, inputs=dict(_INPUTS), outputs=dict(_OUTPUTS))


register_ops()
