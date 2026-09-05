"""Worker ops: FFT structure factors, gradients, and refinement targets.

Inputs arrive as Packed* objects (numpy or torch buffers) plus JSON;
outputs are Packed* objects the codec knows how to store.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from phridge.models import ObservationType, SfEngineParams
from phridge.packing import PackedMiller
from phridge.packing_scattering import PackedScatteringTable, PackedSfCurvatures, PackedSfGradients, PackedTargetResult
from phridge.packing_xtal import PackedXray
from phridge.worker.targets import Observations, TargetEval, build_target
from phridge.worker.xtal.engine import EngineParams, ScatteringModel, StructureFactorEngine

_DEVICE = {"device": "cpu"}


def set_device(device: str) -> None:
    _DEVICE["device"] = device


def _np(value: Any, dtype=None) -> Optional[np.ndarray]:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    return arr if dtype is None else arr.astype(dtype)


def _engine_params(params: Any) -> EngineParams:
    if isinstance(params, SfEngineParams):
        p = params
    elif isinstance(params, dict):
        p = SfEngineParams.model_validate(params)
    else:
        raise TypeError("params must be SfEngineParams or dict")
    return EngineParams(
        d_min=p.d_min,
        grid_resolution_factor=p.grid_resolution_factor,
        quality_factor=p.quality_factor,
        wing_cutoff=p.wing_cutoff,
        u_extra=p.u_extra,
        n_real=None if p.n_real is None else tuple(p.n_real),
        dtype=p.dtype,
    )


def scattering_model(xray: PackedXray, table: PackedScatteringTable) -> ScatteringModel:
    """Assemble the engine's model from the wire types."""
    crystal = xray.meta.crystal
    if not crystal.symops:
        raise ValueError("XrayStructure.crystal has no symops; export with crystal_from_cctbx(with_symops=True)")
    rot = np.array([np.asarray(op.r, dtype=np.float64).reshape(3, 3) for op in crystal.symops])
    trans = np.array([np.asarray(op.t, dtype=np.float64) for op in crystal.symops])
    scatterers = xray.meta.scatterers
    n = len(scatterers)
    fp = np.array([0.0 if s.fp is None else s.fp for s in scatterers], dtype=np.float64)
    fdp = np.array([0.0 if s.fdp is None else s.fdp for s in scatterers], dtype=np.float64)
    aniso = np.array([bool(s.anisotropic) for s in scatterers], dtype=bool)
    type_index = table.row_index([s.scattering_type for s in scatterers])
    return ScatteringModel(
        unit_cell=tuple(float(x) for x in crystal.unit_cell),
        sites_frac=_np(xray.sites_frac, np.float64).reshape(n, 3),
        occupancy=_np(xray.occupancy, np.float64),
        u_iso=_np(xray.u_iso, np.float64),
        u_star=_np(xray.u_star, np.float64).reshape(n, 6),
        anisotropic=aniso,
        fp=fp,
        fdp=fdp,
        type_index=type_index,
        gauss_a=_np(table.gauss_a, np.float64),
        gauss_b=_np(table.gauss_b, np.float64),
        gauss_c=_np(table.gauss_c, np.float64),
        rot=rot,
        trans=trans,
    )


def _engine(xray: PackedXray, table: PackedScatteringTable, hkl: np.ndarray, params: Any) -> StructureFactorEngine:
    return StructureFactorEngine(scattering_model(xray, table), hkl, _engine_params(params), device=_DEVICE["device"])


def _miller_like(template: PackedMiller, data: np.ndarray, label: str) -> PackedMiller:
    return PackedMiller(
        crystal=template.meta.crystal,
        hkl=_np(template.hkl),
        data=data,
        anomalous=template.meta.anomalous,
        observation_type=ObservationType.complex,
        anomalous_layout=template.meta.anomalous_layout,
        label=label,
    )


# --------------------------------------------------------------------- ops
def sf_calc(xray: PackedXray, table: PackedScatteringTable, hkl: PackedMiller, params: Any) -> PackedMiller:
    """F_calc on the indices of ``hkl`` (its data is ignored)."""
    eng = _engine(xray, table, _np(hkl.hkl), params)
    return _miller_like(hkl, eng.f_calc_numpy(), "F_calc")


sf_calc.compute_dtype = "float64"


def sf_gradients(
    xray: PackedXray,
    table: PackedScatteringTable,
    d_target_d_f_calc: PackedMiller,
    params: Any,
) -> PackedSfGradients:
    """d target / d scatterer parameters given d_target_d_f_calc (cctbx convention)."""
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


sf_gradients.compute_dtype = "float64"


def _observations(
    f_obs: PackedMiller,
    weights=None,
    r_free=None,
    alpha=None,
    beta=None,
    epsilon=None,
    centric=None,
) -> Observations:
    import torch

    return Observations.from_numpy(
        device=_DEVICE["device"],
        dtype=torch.float64,
        data=_np(f_obs.data, np.float64),
        sigmas=None if f_obs.sigmas is None else _np(f_obs.sigmas, np.float64),
        weights=_np(weights),
        r_free=_np(r_free),
        alpha=_np(alpha),
        beta=_np(beta),
        epsilon=_np(epsilon),
        centric=_np(centric),
    )


def _target_result(name: str, ev: TargetEval) -> PackedTargetResult:
    return PackedTargetResult(
        name=name,
        value=ev.value,
        per_reflection=ev.per_reflection,
        d_target_d_f_calc=ev.d_target_d_f_calc,
        curv_radial=ev.curv_radial,
        curv_tangential=ev.curv_tangential,
        value_test=ev.value_test,
        scale_factor=ev.scale_factor,
    )


def target_eval(
    f_calc: PackedMiller,
    f_obs: PackedMiller,
    target: dict,
    weights=None,
    r_free=None,
    alpha=None,
    beta=None,
    epsilon=None,
    centric=None,
    compute_curvature: bool = True,
) -> PackedTargetResult:
    """Evaluate a registered target on given F_calc; returns value, dQ/dF, curvature."""
    import torch

    if _np(f_calc.hkl).shape != _np(f_obs.hkl).shape or not np.array_equal(_np(f_calc.hkl), _np(f_obs.hkl)):
        raise ValueError("f_calc and f_obs must be on identical hkl lists")
    tgt = build_target(dict(target))
    obs = _observations(f_obs, weights, r_free, alpha, beta, epsilon, centric)
    fc = torch.as_tensor(_np(f_calc.data, np.complex128), dtype=torch.complex128, device=_DEVICE["device"])
    ev = tgt.evaluate(fc, obs, compute_curvature=bool(compute_curvature))
    return _target_result(tgt.name, ev)


target_eval.compute_dtype = "float64"


def refine_gradients(
    xray: PackedXray,
    table: PackedScatteringTable,
    f_obs: PackedMiller,
    params: Any,
    target: dict,
    weights=None,
    r_free=None,
    alpha=None,
    beta=None,
    epsilon=None,
    centric=None,
    compute_curvature: bool = True,
) -> dict:
    """Whole chain: F_calc -> target -> dQ/dF -> dQ/d(scatterer params)."""
    import torch

    eng = _engine(xray, table, _np(f_obs.hkl), params)
    tgt = build_target(dict(target))
    obs = _observations(f_obs, weights, r_free, alpha, beta, epsilon, centric)
    p = eng.tensors(requires_grad=True)
    fc = eng.f_calc(*p)
    ev = tgt.evaluate(fc.detach(), obs, compute_curvature=bool(compute_curvature))
    g = torch.as_tensor(ev.d_target_d_f_calc, dtype=fc.dtype, device=fc.device)
    q = (fc * g.conj()).real.sum()
    grads = torch.autograd.grad(q, p, allow_unused=True)
    arrays = [(torch.zeros_like(t) if gr is None else gr).detach().cpu().numpy().astype(np.float64) for t, gr in zip(p, grads)]
    gradients = PackedSfGradients(
        d_site_frac=arrays[0],
        d_occupancy=arrays[1],
        d_u_iso=arrays[2],
        d_u_star=arrays[3],
        d_fp=arrays[4],
        d_fdp=arrays[5],
        target=ev.value,
    )
    return {
        "f_calc": _miller_like(f_obs, fc.detach().cpu().numpy().astype(np.complex128), "F_calc"),
        "target": _target_result(tgt.name, ev),
        "gradients": gradients,
    }


refine_gradients.compute_dtype = "float64"


def gauss_newton_hvp(
    xray: PackedXray,
    table: PackedScatteringTable,
    target: PackedTargetResult,
    hkl: PackedMiller,
    v: PackedSfGradients,
    params: Any,
) -> PackedSfGradients:
    """(J^T H J) v using the curvatures stored in a TargetResult.

    ``v`` uses the SfGradients layout as a parameter-space vector;
    ``hkl`` supplies the reflection list the target was evaluated on.
    """
    if target.curv_radial is None:
        raise ValueError("TargetResult has no curvatures; evaluate with compute_curvature=True")
    eng = _engine(xray, table, _np(hkl.hkl), params)
    tangents = tuple(_np(getattr(v, k), np.float64) for k in PackedSfGradients.FIELDS)
    hv = eng.gauss_newton_hvp(tangents, _np(target.curv_radial), _np(target.curv_tangential))
    return PackedSfGradients(
        d_site_frac=hv["site_frac"],
        d_occupancy=hv["occupancy"],
        d_u_iso=hv["u_iso"],
        d_u_star=hv["u_star"],
        d_fp=hv["fp"],
        d_fdp=hv["fdp"],
    )


gauss_newton_hvp.compute_dtype = "float64"


def gauss_newton_diagonal(
    xray: PackedXray,
    table: PackedScatteringTable,
    target: PackedTargetResult,
    hkl: PackedMiller,
    params: Any,
    n_probes: int = 8,
    seed: int = 0,
) -> PackedSfGradients:
    """Hutchinson estimate of diag(J^T H J) in SfGradients layout."""
    if target.curv_radial is None:
        raise ValueError("TargetResult has no curvatures; evaluate with compute_curvature=True")
    eng = _engine(xray, table, _np(hkl.hkl), params)
    diag = eng.gauss_newton_diagonal(
        _np(target.curv_radial),
        _np(target.curv_tangential),
        n_probes=int(n_probes),
        seed=int(seed),
    )
    return PackedSfGradients(
        d_site_frac=diag["site_frac"],
        d_occupancy=diag["occupancy"],
        d_u_iso=diag["u_iso"],
        d_u_star=diag["u_star"],
        d_fp=diag["fp"],
        d_fdp=diag["fdp"],
    )


gauss_newton_diagonal.compute_dtype = "float64"


def gauss_newton_blocks(
    xray: PackedXray,
    table: PackedScatteringTable,
    target: PackedTargetResult,
    hkl: PackedMiller,
    params: Any,
) -> PackedSfCurvatures:
    """Exact per-atom Gauss-Newton blocks (Tronrud D+S) as SfCurvatures."""
    if target.curv_radial is None:
        raise ValueError("TargetResult has no curvatures; evaluate with compute_curvature=True")
    eng = _engine(xray, table, _np(hkl.hkl), params)
    blocks = eng.gauss_newton_blocks(_np(target.curv_radial), _np(target.curv_tangential))
    return PackedSfCurvatures(
        site_frac=blocks["site_frac"],
        occupancy=blocks["occupancy"],
        u_iso=blocks["u_iso"],
        u_star=blocks["u_star"],
        fp=blocks["fp"],
        fdp=blocks["fdp"],
    )


gauss_newton_blocks.compute_dtype = "float64"
