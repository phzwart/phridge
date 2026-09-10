"""Worker op ``ml_i_omit_windows``: windowed omit map coefficients."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from phridge.contrib.intensity_ll.omit_windows import (
    OMIT_OP_NAME,
    OmitWindowOptions,
    assign_box_windows,
    assign_residue_block_windows,
    run_omit_windows_core,
)
from phridge.packing import PackedMiller

_OMIT_INPUTS = {
    "f_model": "MillerArray",
    "f_obs": "MillerArray",
    "xray": "XrayStructure",
    "table": "ScatteringTable",
    "params": "SfEngineParams",
    "target": "json",
    "weights": "array",
    "r_free": "array",
    "alpha": "array",
    "beta": "array",
    "epsilon": "array",
    "centric": "array",
    "nu": "array",
    "k_scale": "array",
    "omit": "json",
    "residue_ids": "json",
}

_OMIT_OUTPUTS = {
    "hkl": "array",
    "coef_model": "array",
    "coef_difference": "array",
    "beta_adjust": "array",
    "window_ids": "array",
    "atom_to_window": "array",
    "window_meta": "json",
    "stats": "json",
}


def ml_i_omit_windows(
    f_model: PackedMiller,
    f_obs: PackedMiller,
    xray: Any,
    table: Any,
    params: Any,
    target: dict,
    weights: Optional[Any] = None,
    r_free: Optional[Any] = None,
    alpha: Optional[Any] = None,
    beta: Optional[Any] = None,
    epsilon: Optional[Any] = None,
    centric: Optional[Any] = None,
    nu: Optional[Any] = None,
    k_scale: Optional[Any] = None,
    omit: Optional[dict] = None,
    residue_ids: Optional[Any] = None,
) -> dict[str, Any]:
    """Compute per-window ``ml_i`` map coefficients with linear SF omit + residual-β adjust.

    Solvent and scales stay those of the full ``f_model``. See
    :mod:`phridge.contrib.intensity_ll.omit_windows` for the math.
    """
    from phridge.contrib.intensity_ll.target import IntensityLogLikelihood
    from phridge.sfcalc.ops import _engine, _np, _observations
    from phridge.sfcalc.targets import build_target

    opts = OmitWindowOptions.model_validate(omit or {})
    if _np(f_model.hkl).shape != _np(f_obs.hkl).shape or not np.array_equal(_np(f_model.hkl), _np(f_obs.hkl)):
        raise ValueError("f_model and f_obs must share identical hkl")

    spec = dict(target)
    if spec.get("name") != "ml_i":
        raise ValueError(f"{OMIT_OP_NAME} needs an ml_i target spec, got {spec.get('name')!r}")
    tgt = build_target(spec)
    assert isinstance(tgt, IntensityLogLikelihood)

    hkl = _np(f_obs.hkl, np.int64)
    n = len(f_obs.data)
    fm = _np(f_model.data, np.complex128)
    sa = _np(alpha, np.float64) if alpha is not None else np.full(n, float(tgt.sigma_a or 0.8))
    sw = _np(beta, np.float64) if beta is not None else np.full(n, float(tgt.sigma_wilson or 1.0))
    eps = _np(epsilon, np.float64) if epsilon is not None else np.ones(n, dtype=np.float64)
    k_sc = _np(k_scale, np.float64) if k_scale is not None else np.ones(n, dtype=np.float64)

    eng = _engine(xray, table, hkl, params)
    sites = np.asarray(eng.model.sites_frac, dtype=np.float64)

    if opts.mode == "residue_blocks":
        if residue_ids is None:
            raise ValueError("residue_blocks mode requires residue_ids=[(chain, resseq), ...]")
        partition = assign_residue_block_windows(residue_ids, block_size=opts.block_size)
    else:
        partition = assign_box_windows(
            sites,
            eng.model.unit_cell,
            box_size=opts.box_size,
            shift=opts.shift,
        )

    obs = _observations(f_obs, weights, r_free, alpha, beta, epsilon, centric, nu=nu)
    store = run_omit_windows_core(
        eng=eng,
        f_model=fm,
        k_scale=k_sc,
        obs_target=tgt,
        obs=obs,
        sigma_a=sa,
        sigma_wilson=sw,
        epsilon=eps,
        partition=partition,
        options=opts,
        extra_meta={"op": OMIT_OP_NAME},
    )
    return {
        "hkl": store.hkl,
        "coef_model": store.coef_model if store.coef_model is not None else np.zeros((0, n), dtype=np.complex64),
        "coef_difference": store.coef_difference
        if store.coef_difference is not None
        else np.zeros((0, n), dtype=np.complex64),
        "beta_adjust": store.beta_adjust,
        "window_ids": store.window_ids,
        "atom_to_window": store.atom_to_window,
        "window_meta": store.window_meta,
        "stats": {
            "n_windows_computed": int(len(store.window_ids)),
            "n_windows_total": len(partition.windows),
            "n_empty": len(partition.empty_window_ids),
            "kappa": store.window_meta.get("kappa"),
        },
    }


ml_i_omit_windows.compute_dtype = "float64"  # type: ignore[attr-defined]
