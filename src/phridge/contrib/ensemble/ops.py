"""Worker ops ``ensemble_fit``, ``ensemble_scores``, and ``ensemble_maps``."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from phridge.models import CrystalSymmetry, ObservationType
from phridge.ops import register_op
from phridge.packing import PackedMiller

OP_NAME_FIT = "ensemble_fit"
OP_NAME_SCORES = "ensemble_scores"
OP_NAME_MAPS = "ensemble_maps"

_FIT_INPUTS = {
    "hkl": "array",
    "centric": "array",
    "epsilon": "array",
    "i_obs": "array",
    "sig_i": "array",
    "mask": "array",
    "options": "json",
    "cells": "array",
    "f_mu": "array",
    "f_var": "array",
    "f_mask": "array",
    "f_phase": "array",
    "roles": "array",
    "hl": "array",
    "xray": "XrayStructure",
    "table": "ScatteringTable",
    "params": "SfEngineParams",
    "sites_frac": "array",
}
_FIT_OUTPUTS = {
    "delta0": "array",
    "modes": "array",
    "scores": "array",
    "score_cov": "array",
    "sigma2": "array",
    "d": "array",
    "shell_index": "array",
    "loglik_history": "array",
    "heldout": "json",
    "fom": "array",
    "stats": "json",
}
_SCORES_INPUTS = {
    "i_obs": "array",
    "sig_i": "array",
    "mask": "array",
    "f_mu": "array",
    "delta0": "array",
    "modes": "array",
    "d": "array",
    "sigma2": "array",
    "shell_index": "array",
    "epsilon": "array",
    "centric": "array",
    "options": "json",
    "hl": "array",
    "f_var": "array",
    "roles": "array",
}
_SCORES_OUTPUTS = {
    "scores": "array",
    "score_cov": "array",
    "loglik": "json",
    "fom": "array",
    "stats": "json",
}
_MAPS_INPUTS = {
    "hkl": "array",
    "delta0": "array",
    "modes": "array",
    "cells": "array",
    "residuals": "array",
}
_MAPS_OUTPUTS = {
    "delta0": "MillerArray",
    "modes": "array",
    "residuals": "array",
}


def _np(value: Any, dtype: Any = None) -> Optional[np.ndarray]:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    return arr if dtype is None else arr.astype(dtype, copy=False)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _reference_f(
    hkl: np.ndarray,
    cells: Optional[np.ndarray],
    xray: Any,
    table: Any,
    params: Any,
    sites_frac: Any,
    f_mu: Optional[np.ndarray],
) -> np.ndarray:
    if f_mu is not None:
        return np.asarray(f_mu, dtype=np.complex128).reshape(-1)
    if xray is None or sites_frac is None:
        raise ValueError("ensemble_fit needs f_mu or an xray structure and sites_frac")
    from phridge.contrib.multixtal.structure import f_calc_s_g, model_with_cell
    from phridge.sfcalc.ops import _engine_params, scattering_model

    model = scattering_model(xray, table)
    sites = np.asarray(sites_frac, dtype=np.float64)
    cell_rows = np.asarray(cells, dtype=np.float64) if cells is not None else None
    if sites.ndim == 3:
        # Per-dataset placement, matching run_multixtal_core. F_mu is dataset 0
        # in that dataset's cell; the loop still evaluates every dataset so a
        # later per-dataset phase can reuse the same engine path.
        out = []
        for d in range(sites.shape[0]):
            cell = cell_rows[d] if cell_rows is not None else model.unit_cell
            placed = model_with_cell(model, cell, sites[d])
            f_full, _s, _g = f_calc_s_g(placed, hkl, params=_engine_params(params) if params is not None else None)
            out.append(f_full)
        return np.asarray(out[0], dtype=np.complex128)
    cell = cell_rows[0] if cell_rows is not None and cell_rows.ndim == 2 else model.unit_cell
    placed = model_with_cell(model, cell, sites)
    f_full, _s, _g = f_calc_s_g(placed, hkl, params=_engine_params(params) if params is not None else None)
    return np.asarray(f_full, dtype=np.complex128)


def _shell_index(hkl: np.ndarray, cells: Optional[np.ndarray], bin_size: int) -> np.ndarray:
    import torch

    from phridge.contrib.intensity_ll.rint import equal_count_shell_indices
    from phridge.contrib.multixtal.structure import resolution_s2

    cell = (30.0, 30.0, 40.0, 90.0, 90.0, 90.0)
    if cells is not None:
        rows = np.asarray(cells, dtype=np.float64)
        cell = tuple(float(x) for x in (rows[0] if rows.ndim == 2 else rows))
    s2 = resolution_s2(cell, np.asarray(hkl))
    d = 1.0 / np.sqrt(np.maximum(s2, 1e-12))
    return equal_count_shell_indices(torch.as_tensor(d, dtype=torch.float64), int(bin_size)).cpu().numpy().astype(np.int32)


def _hl_from_phase(
    f_phase: np.ndarray,
    f_mu: np.ndarray,
    i_obs: np.ndarray,
    epsilon: np.ndarray,
    sigma_wilson: np.ndarray,
    sigma_a: np.ndarray,
    weight: float,
) -> np.ndarray:
    """Acentric HL vector. ``X = 2 σ_A |E_o| |E_c| / (1-σ_A²)``, then ``(A+iB) = X e^{iφ}``."""
    scale = np.sqrt(np.maximum(epsilon * sigma_wilson, 1e-12))
    ec = np.abs(f_phase) / scale
    eo = np.sqrt(np.maximum(i_obs, 0.0)) / scale
    sa = np.clip(sigma_a, 1e-3, 1.0 - 1e-3)
    x = 2.0 * sa * eo * ec / (1.0 - sa**2)
    phase = np.exp(1j * np.angle(f_phase))
    return (float(weight) * x * phase).astype(np.complex128)


def _result_dict(result: Any, heldout: dict[str, Any]) -> dict[str, Any]:
    stats = dict(result.stats)
    if result.loglik_unprojected is not None:
        stats["loglik_unprojected"] = result.loglik_unprojected
        stats["loglik_projected"] = result.loglik_projected
        stats["projected_and_unprojected_not_comparable"] = True
    return {
        "delta0": np.asarray(result.delta0, dtype=np.complex128),
        "modes": np.asarray(result.modes, dtype=np.complex128),
        "scores": np.asarray(result.scores, dtype=np.float64),
        "score_cov": np.asarray(result.score_cov, dtype=np.float64),
        "sigma2": np.asarray(result.sigma2, dtype=np.float64),
        "d": np.asarray(result.d, dtype=np.float64),
        "shell_index": np.asarray(result.shell_index, dtype=np.int32),
        "loglik_history": np.asarray(result.loglik_history, dtype=np.float64),
        "heldout": _jsonable(heldout),
        "fom": np.asarray(result.fom, dtype=np.float64),
        "stats": _jsonable(stats),
    }


def ensemble_fit(
    hkl: Any,
    centric: Any,
    epsilon: Any,
    i_obs: Any,
    sig_i: Any,
    mask: Any,
    options: Optional[dict] = None,
    cells: Optional[Any] = None,
    f_mu: Optional[Any] = None,
    f_var: Optional[Any] = None,
    f_mask: Optional[Any] = None,
    f_phase: Optional[Any] = None,
    roles: Optional[Any] = None,
    hl: Optional[Any] = None,
    xray: Optional[Any] = None,
    table: Optional[Any] = None,
    params: Optional[Any] = None,
    sites_frac: Optional[Any] = None,
) -> dict[str, Any]:
    """Shared Δ0, modes, and per-dataset scores."""
    from phridge.contrib.ensemble.cloud import latent_cloud
    from phridge.legacy.ensemble_fit import em_fit, fit_known_f, phased_observed
    from phridge.contrib.ensemble.holdout import crystal_folds, half_split, k_scan
    from phridge.contrib.ensemble.mstep import expand_shells
    from phridge.contrib.ensemble.options import EnsembleOptions

    opts = EnsembleOptions.model_validate(options or {})
    hkl_a = _np(hkl, np.int32)
    i_obs_a = _np(i_obs, np.float64)
    sig_a = _np(sig_i, np.float64)
    mask_a = _np(mask, bool)
    n_data, n_h = i_obs_a.shape
    eps = _np(epsilon, np.float64).reshape(-1)
    cen = _np(centric, bool).reshape(-1)
    f_mu_a = _reference_f(hkl_a, _np(cells) if cells is not None else None, xray, table, params, _np(sites_frac) if sites_frac is not None else None, _np(f_mu) if f_mu is not None else None)
    if f_mu_a.ndim == 2:
        f_mu_vec = f_mu_a[0]
    else:
        f_mu_vec = f_mu_a.reshape(-1)
    f_var_a = np.zeros((n_data, n_h), dtype=np.complex128) if f_var is None else _np(f_var, np.complex128)
    if f_mask is not None and opts.likelihood == "intensity":
        from phridge.contrib.multixtal.structure import apply_bulk_solvent, resolution_s2

        cell = (30.0, 30.0, 40.0, 90.0, 90.0, 90.0)
        if cells is not None:
            rows = _np(cells, np.float64)
            cell = tuple(float(x) for x in rows[0])
        s2 = resolution_s2(cell, hkl_a)
        solvent = apply_bulk_solvent(
            np.zeros_like(f_var_a),
            _np(f_mask, np.complex128),
            s2,
            0.35,
            50.0,
        )
        f_var_a = f_var_a + solvent
    shells = _shell_index(hkl_a, _np(cells) if cells is not None else None, opts.bin_size)
    role_a = None if roles is None else _np(roles, np.int8)
    if opts.likelihood == "known_f":
        phase_src = f_mu_vec if f_phase is None else _np(f_phase, np.complex128)
        if np.asarray(phase_src).ndim == 1:
            phase = np.angle(phase_src)
            phase = np.broadcast_to(phase, (n_data, n_h))
        else:
            phase = np.angle(phase_src)
        f_obs = phased_observed(i_obs_a, phase, mask_a)
        result = fit_known_f(
            f_obs,
            f_mu_vec,
            mask_a,
            f_var=f_var_a,
            rank=opts.k,
            n_sweeps=opts.n_sweeps,
            ridge=opts.ridge,
            roles=role_a,
            shell_index=shells,
        )
    else:
        wilson = np.maximum(np.mean(np.where(mask_a, np.maximum(i_obs_a, 0.0), np.nan), axis=1), 1.0)
        hl_a = None if hl is None else _np(hl, np.complex128)
        weight = float(opts.phase_injection.hl_weight) if opts.phase_injection.enabled else 0.0
        if weight > 0.0 and hl_a is None and f_phase is not None:
            phase_f = _np(f_phase, np.complex128)
            sa = np.full((n_data, n_h), float(opts.sigma_a0))
            sw = np.broadcast_to(wilson[:, None], (n_data, n_h))
            eps_b = np.broadcast_to(eps, (n_data, n_h))
            hl_a = _hl_from_phase(phase_f, f_mu_vec, i_obs_a, eps_b, sw, sa, 1.0)
        if opts.fit_nuisance:
            from phridge.contrib.ensemble.nuisance import fit_one_dataset

            cell = (30.0, 30.0, 40.0, 90.0, 90.0, 90.0)
            if cells is not None:
                cell = tuple(float(x) for x in _np(cells, np.float64)[0])
            fitted = fit_one_dataset(hkl_a, i_obs_a[0], sig_a[0], f_mu_vec, eps, cen, cell, tune_mask=mask_a[0])
            wilson = np.broadcast_to(np.mean(fitted["sigma_wilson"]) * np.ones(n_data), (n_data,))
        result = em_fit(
            i_obs_a,
            sig_a,
            mask_a,
            f_mu_vec,
            eps,
            cen,
            wilson,
            f_var=f_var_a,
            rank=opts.k,
            n_sweeps=opts.n_sweeps,
            ridge=opts.ridge,
            shell_index=shells,
            roles=role_a,
            hl=hl_a,
            hl_weight=weight,
            nu=opts.nu,
            sigma_a0=opts.sigma_a0,
            dof_correct=opts.dof_correct,
            hkl=hkl_a,
            cell=None if cells is None else tuple(float(x) for x in _np(cells, np.float64)[0]),
            projection=opts.projection.model_dump() if opts.projection.enabled else None,
        )
        if opts.direct:
            from phridge.legacy.ensemble_fit import direct_fit

            result = direct_fit(
                i_obs_a,
                sig_a,
                mask_a,
                f_mu_vec,
                eps,
                cen,
                wilson,
                f_var=f_var_a,
                rank=opts.k,
                max_iter=opts.max_lbfgs_iter,
                shell_index=shells,
                nu=opts.nu,
                sigma_a0=opts.sigma_a0,
                ridge=opts.ridge,
            )
    heldout: dict[str, Any] = {}
    if opts.holdout.enabled and opts.likelihood == "known_f":
        from phridge.legacy.ensemble_fit import known_f_target

        phase = np.angle(f_mu_vec)
        f_obs = phased_observed(i_obs_a, np.broadcast_to(phase, (n_data, n_h)), mask_a)
        target = known_f_target(f_obs, f_mu_vec, f_var_a, np.ones((n_data, n_h)))
        weights = mask_a.astype(np.float64)
        if role_a is not None:
            weights = weights * (role_a == 0)
            free_w = mask_a.astype(np.float64) * (role_a == 2)
        else:
            free_w = weights
        k_values = opts.holdout.k_values or [opts.k]
        heldout["k_scan"] = k_scan(target, weights, k_values, n_folds=min(opts.holdout.n_folds, n_data), n_sweeps=opts.n_sweeps, seed=opts.holdout.seed, free_weights=free_w)
        if opts.holdout.half_split:
            heldout["half_split"] = half_split(target, weights, shells, rank=opts.k, n_sweeps=opts.n_sweeps, seed=opts.holdout.seed)
        heldout["crystal_folds"] = crystal_folds(target, weights, rank=opts.k, n_folds=min(opts.holdout.n_folds, n_data), n_sweeps=opts.n_sweeps, seed=opts.holdout.seed, free_weights=free_w)
    cloud = latent_cloud(result.scores, result.score_cov)
    result.stats["quadratic_r2"] = cloud["quadratic_r2"]
    result.stats["deconvolved_covariance"] = np.asarray(cloud["deconvolved_covariance"]).tolist()
    _ = expand_shells
    return _result_dict(result, heldout)


def ensemble_scores(
    i_obs: Any,
    sig_i: Any,
    mask: Any,
    f_mu: Any,
    delta0: Any,
    modes: Any,
    d: Any,
    sigma2: Any,
    shell_index: Any,
    epsilon: Any,
    centric: Any,
    options: Optional[dict] = None,
    hl: Optional[Any] = None,
    f_var: Optional[Any] = None,
    roles: Optional[Any] = None,
) -> dict[str, Any]:
    """E-step and score update for one or more datasets with modes held fixed."""
    from phridge.legacy.ensemble_fit import _e_step, _mu_field
    from phridge.contrib.ensemble.mstep import expand_shells, solve_scores_torch, work_weights
    from phridge.contrib.ensemble.options import EnsembleOptions

    opts = EnsembleOptions.model_validate(options or {})
    i_obs_a = _np(i_obs, np.float64)
    if i_obs_a.ndim == 1:
        i_obs_a = i_obs_a.reshape(1, -1)
    n_data, n_h = i_obs_a.shape
    sig_a = _np(sig_i, np.float64).reshape(n_data, n_h)
    mask_a = _np(mask, bool).reshape(n_data, n_h)
    f_mu_v = _np(f_mu, np.complex128).reshape(-1)
    delta = _np(delta0, np.complex128).reshape(-1)
    modes_a = _np(modes, np.complex128)
    if modes_a.ndim == 1:
        modes_a = modes_a.reshape(1, -1)
    shells = _np(shell_index, np.int32).reshape(-1)
    d_refl = expand_shells(_np(d, np.float64), shells)
    if d_refl.ndim == 1:
        d_refl = np.broadcast_to(d_refl, (n_data, n_h)).copy()
    sigma_refl = expand_shells(_np(sigma2, np.float64), shells)
    if sigma_refl.ndim == 1:
        sigma_refl = np.broadcast_to(sigma_refl, (n_data, n_h)).copy()
    eps = _np(epsilon, np.float64).reshape(-1)
    cen = _np(centric, bool).reshape(-1)
    eps_b = np.broadcast_to(eps, (n_data, n_h)).copy()
    cen_b = np.broadcast_to(cen, (n_data, n_h)).copy()
    wilson = np.maximum(i_obs_a, 1.0)
    var = np.zeros((n_data, n_h), dtype=np.complex128) if f_var is None else _np(f_var, np.complex128)
    scores = np.zeros((n_data, modes_a.shape[0]), dtype=np.float64)
    mu = _mu_field(f_mu_v, var, delta, modes_a, scores, d_refl)
    role_a = None if roles is None else _np(roles, np.int8)
    weights = work_weights(mask_a, d_refl, sigma_refl, role_a, include_calibration=False)
    target = (np.asarray(f_mu_v) * 0.0)  # placeholder to keep the line short
    _ = target
    # One E-step at z = 0, then the closed-form score update. Modes stay fixed.
    moments = _e_step(mu, sigma_refl, i_obs_a, sig_a, eps_b, wilson, cen_b, mask_a, opts.nu, None if hl is None else _np(hl), opts.phase_injection.hl_weight if opts.phase_injection.enabled else 0.0)
    y = (moments["mean_f"] - var) / np.maximum(d_refl, 1e-6) - f_mu_v
    y = np.where(weights > 0, y, 0.0)
    scores, cov = solve_scores_torch(y, delta, modes_a, weights, opts.ridge)
    return {
        "scores": scores,
        "score_cov": cov,
        "loglik": float(np.sum(moments["log_lik"])),
        "fom": moments["fom"],
        "stats": {"k": int(modes_a.shape[0]), "n_data": int(n_data)},
    }


def ensemble_maps(
    hkl: Any,
    delta0: Any,
    modes: Any,
    cells: Any,
    residuals: Optional[Any] = None,
) -> dict[str, Any]:
    """Δ0 as a Miller array. Modes stay a stacked complex array (leading mode axis)."""
    hkl_a = _np(hkl, np.int32)
    cell_rows = _np(cells, np.float64)
    cell = [float(x) for x in (cell_rows[0] if cell_rows.ndim == 2 else cell_rows)]
    crystal = CrystalSymmetry(unit_cell=cell, space_group_hall="P 1", space_group_number=1)
    delta = _np(delta0, np.complex128).reshape(-1)
    packed = PackedMiller(
        crystal,
        hkl_a,
        delta,
        anomalous=False,
        observation_type=ObservationType.complex,
        label="delta0",
    )
    modes_a = _np(modes, np.complex128)
    resid = np.zeros((0, hkl_a.shape[0]), dtype=np.complex128) if residuals is None else _np(residuals, np.complex128)
    return {"delta0": packed, "modes": modes_a, "residuals": resid}


ensemble_fit.compute_dtype = "float64"
ensemble_scores.compute_dtype = "float64"
ensemble_maps.compute_dtype = "float64"


def register_ops() -> None:
    register_op(OP_NAME_FIT, ensemble_fit, inputs=dict(_FIT_INPUTS), outputs=dict(_FIT_OUTPUTS))
    register_op(OP_NAME_SCORES, ensemble_scores, inputs=dict(_SCORES_INPUTS), outputs=dict(_SCORES_OUTPUTS))
    register_op(OP_NAME_MAPS, ensemble_maps, inputs=dict(_MAPS_INPUTS), outputs=dict(_MAPS_OUTPUTS))
