"""Worker ops that build ``NufftStructureFactorEngine`` from packed xtal inputs."""

from __future__ import annotations

from typing import Any

import numpy as np

from phridge.contrib.nufft_sf.options import NufftEngineOptions
from phridge.ops import register_op
from phridge.packing import PackedMiller
from phridge.packing_xtal import PackedXray
from phridge.sfcalc.ops import _DEVICE, _miller_like, _np, scattering_model
from phridge.sfcalc.packing import PackedScatteringTable, PackedSfGradients

CALC_OP = "nufft_sf_calc"
GRAD_OP = "nufft_sf_gradients"

_CALC_INPUTS = {
    "xray": "XrayStructure",
    "table": "ScatteringTable",
    "hkl": "MillerArray",
    "params": "json",
}
_CALC_OUTPUTS = {"f_calc": "MillerArray"}
_GRAD_INPUTS = {
    "xray": "XrayStructure",
    "table": "ScatteringTable",
    "d_target_d_f_calc": "MillerArray",
    "params": "json",
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


def nufft_sf_calc(
    xray: PackedXray,
    table: PackedScatteringTable,
    hkl: PackedMiller,
    params: Any,
) -> PackedMiller:
    """F_calc on the indices of ``hkl`` (its data is ignored)."""
    eng = _engine(xray, table, _np(hkl.hkl), params)
    return _miller_like(hkl, eng.f_calc_numpy(), "F_calc")


def nufft_sf_gradients(
    xray: PackedXray,
    table: PackedScatteringTable,
    d_target_d_f_calc: PackedMiller,
    params: Any,
) -> PackedSfGradients:
    """d target / d scatterer parameters (cctbx ``d_target_d_f_calc`` convention)."""
    eng = _engine(xray, table, _np(d_target_d_f_calc.hkl), params)
    g = eng.gradients(_np(d_target_d_f_calc.data, np.complex128))
    return PackedSfGradients(
        d_site_frac=g["site_frac"],
        d_occupancy=g["occupancy"],
        d_u_iso=g["u_iso"],
        d_u_star=g["u_star"],
        d_fp=g["fp"],
        d_fdp=g["fdp"],
    )


def register_ops() -> None:
    register_op(CALC_OP, nufft_sf_calc, inputs=dict(_CALC_INPUTS), outputs=dict(_CALC_OUTPUTS))
    register_op(GRAD_OP, nufft_sf_gradients, inputs=dict(_GRAD_INPUTS), outputs=dict(_GRAD_OUTPUTS))
