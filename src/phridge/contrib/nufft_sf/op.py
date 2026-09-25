"""Worker ops that build ``NufftStructureFactorEngine`` from packed xtal inputs."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from phridge.contrib.nufft_sf.options import NufftEngineOptions
from phridge.ops import register_op
from phridge.packing import PackedMiller
from phridge.packing_xtal import PackedXray
from phridge.sfcalc.ops import _DEVICE, _bound_engine, _miller_like, _np, _packed_grads, scattering_model
from phridge.sfcalc.packing import PackedScatteringTable, PackedSfGradients

BIND_OP = "nufft_sf_bind"
CALC_OP = "nufft_sf_calc"
GRAD_OP = "nufft_sf_gradients"

_CALC_INPUTS = {
    "xray": "XrayStructure",
    "table": "ScatteringTable",
    "hkl": "MillerArray",
    "params": "json",
    "handle": "json",
}
_CALC_OUTPUTS = {"f_calc": "MillerArray"}
_BIND_OUTPUTS = {"handle": "json"}
_GRAD_INPUTS = {
    "xray": "XrayStructure",
    "table": "ScatteringTable",
    "d_target_d_f_calc": "MillerArray",
    "params": "json",
    "handle": "json",
}
_GRAD_OUTPUTS = {"gradients": "SfGradients"}


def _params(params: Any) -> NufftEngineOptions:
    if isinstance(params, NufftEngineOptions):
        return params
    if isinstance(params, dict):
        return NufftEngineOptions.model_validate(params)
    raise TypeError("params must be NufftEngineOptions or dict")


def _engine(xray: PackedXray, table: PackedScatteringTable, hkl: np.ndarray, params: Any):
    from phridge.sfcalc.engine.nufft_engine import NufftEngineParams, NufftStructureFactorEngine

    opts = _params(params)
    dtype = opts.dtype
    if str(_DEVICE["device"]).startswith("mps"):
        dtype = "float32"
    nparams = NufftEngineParams(
        d_min=opts.d_min,
        tau=opts.tau,
        n_max=opts.n_max,
        eps=opts.eps,
        dtype=dtype,
        symmetry=opts.symmetry,
        t_chunk=opts.t_chunk,
    )
    return NufftStructureFactorEngine(scattering_model(xray, table), hkl, nparams, device=_DEVICE["device"])


def nufft_sf_bind(
    xray: PackedXray,
    table: PackedScatteringTable,
    hkl: PackedMiller,
    params: Any,
) -> str:
    """Build the NUFFT engine once; later calc/grad pass ``handle``."""
    from phridge.sfcalc.sessions import stash_engine

    eng = _engine(xray, table, _np(hkl.hkl), params)
    return stash_engine(eng, hkl)


def nufft_sf_calc(
    xray: Optional[PackedXray] = None,
    table: Optional[PackedScatteringTable] = None,
    hkl: Optional[PackedMiller] = None,
    params: Any = None,
    handle: Any = None,
) -> PackedMiller:
    """F_calc on the indices of ``hkl`` (its data is ignored)."""
    eng, tmpl = _bound_engine(handle, xray)
    if eng is None:
        if xray is None or table is None or hkl is None or params is None:
            raise TypeError("nufft_sf_calc needs xray, table, hkl, params (or a bound handle)")
        eng = _engine(xray, table, _np(hkl.hkl), params)
        tmpl = hkl
    out_hkl = hkl if hkl is not None else tmpl
    return _miller_like(out_hkl, eng.f_calc_numpy(), "F_calc")


def nufft_sf_gradients(
    xray: Optional[PackedXray] = None,
    table: Optional[PackedScatteringTable] = None,
    d_target_d_f_calc: Optional[PackedMiller] = None,
    params: Any = None,
    handle: Any = None,
) -> PackedSfGradients:
    """d target / d scatterer parameters (cctbx ``d_target_d_f_calc`` convention)."""
    if d_target_d_f_calc is None:
        raise TypeError("nufft_sf_gradients needs d_target_d_f_calc")
    eng, _tmpl = _bound_engine(handle, xray)
    if eng is None:
        if xray is None or table is None or params is None:
            raise TypeError("nufft_sf_gradients needs xray, table, params (or a bound handle)")
        eng = _engine(xray, table, _np(d_target_d_f_calc.hkl), params)
    g = eng.gradients(_np(d_target_d_f_calc.data, np.complex128))
    return _packed_grads(g)


def register_ops() -> None:
    register_op(
        BIND_OP,
        nufft_sf_bind,
        inputs={k: v for k, v in _CALC_INPUTS.items() if k != "handle"},
        outputs=dict(_BIND_OUTPUTS),
    )
    register_op(CALC_OP, nufft_sf_calc, inputs=dict(_CALC_INPUTS), outputs=dict(_CALC_OUTPUTS))
    register_op(GRAD_OP, nufft_sf_gradients, inputs=dict(_GRAD_INPUTS), outputs=dict(_GRAD_OUTPUTS))
