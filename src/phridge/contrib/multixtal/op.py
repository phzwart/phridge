"""Worker op ``multixtal_fit``: unpack existing packed types, run the core, pack."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from phridge.ops import register_op
from phridge.packing_xtal import PackedXray

OP_NAME = "multixtal_fit"

_INPUTS = {
    "xray": "XrayStructure",
    "table": "ScatteringTable",
    "params": "SfEngineParams",
    "sites_frac": "array",
    "cells": "array",
    "wavelengths": "array",
    "hkl": "array",
    "centric": "array",
    "epsilon": "array",
    "i_plus": "array",
    "sig_plus": "array",
    "i_minus": "array",
    "sig_minus": "array",
    "mask_plus": "array",
    "mask_minus": "array",
    "f_mask": "array",
    "options": "json",
    "g_site": "array",
    "obs_i": "array",
    "obs_sig": "array",
    "obs_h": "array",
    "obs_crystal": "array",
    "obs_batch": "array",
    "obs_dose": "array",
    "obs_phi": "array",
    "obs_sign": "array",
    "obs_s_inc": "array",
    "obs_s_dif": "array",
    "obs_wedge": "array",
    "obs_pass": "array",
}
_OUTPUTS = {
    "stats": "json",
    "y": "array",
    "mu": "array",
    "scores": "array",
    "loadings": "array",
    "strong": "array",
    "sigma_a": "array",
    "k_sol": "array",
    "b_sol": "array",
    "chi2_model": "array",
    "chi2_mean": "array",
    "chi2_in_sample": "array",
    "chi2_loo": "array",
    "flagged": "array",
    "f_explained": "array",
    "anom_intercept": "array",
    "anom_slopes": "array",
    "q": "array",
    "q_se": "array",
    "mean_anomalous": "array",
    "factor_anomalous": "array",
    "factor_ordinary": "array",
    "influence": "json",
    "s": "array",
    "g": "array",
    "damage_loadings": "array",
    "damage_g": "array",
    "rho": "array",
    "zero_dose_i": "array",
    "damage_maps": "array",
    "anom_corrected": "array",
    "dose_influence": "json",
}


def _np(value: Any, dtype: Any = None) -> Optional[np.ndarray]:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    return arr if dtype is None else arr.astype(dtype)


def multixtal_fit(
    xray: PackedXray,
    table: Any,
    params: Any,
    sites_frac: Any,
    cells: Any,
    hkl: Any,
    centric: Any,
    epsilon: Any,
    i_plus: Any,
    sig_plus: Any,
    i_minus: Any,
    sig_minus: Any,
    mask_plus: Any,
    mask_minus: Any,
    f_mask: Any,
    options: Optional[dict] = None,
    wavelengths: Optional[Any] = None,
    g_site: Optional[Any] = None,
    obs_i: Optional[Any] = None,
    obs_sig: Optional[Any] = None,
    obs_h: Optional[Any] = None,
    obs_crystal: Optional[Any] = None,
    obs_batch: Optional[Any] = None,
    obs_dose: Optional[Any] = None,
    obs_phi: Optional[Any] = None,
    obs_sign: Optional[Any] = None,
    obs_s_inc: Optional[Any] = None,
    obs_s_dif: Optional[Any] = None,
    obs_wedge: Optional[Any] = None,
    obs_pass: Optional[Any] = None,
) -> dict[str, Any]:
    """Run the phase-1 multi-dataset covariance pipeline."""
    from phridge.contrib.multixtal.options import MultixtalOptions
    from phridge.contrib.multixtal.pipeline import run_multixtal_core
    from phridge.sfcalc.ops import _engine_params, scattering_model

    opts = MultixtalOptions.model_validate(options or {})
    model = scattering_model(xray, table)
    obs = None
    if obs_i is not None and obs_dose is not None:
        from phridge.contrib.multixtal.unmerged import ObservationTable

        oi = _np(obs_i, np.float64).reshape(-1)
        n_o = oi.size
        zeros = np.zeros(n_o, dtype=np.float64)
        ones = np.ones(n_o, dtype=np.int64)
        s0 = np.zeros((n_o, 3), dtype=np.float64)
        s0[:, 2] = 1.0
        obs = ObservationTable(
            i=oi,
            sig=_np(obs_sig, np.float64).reshape(-1) if obs_sig is not None else np.ones(n_o),
            h=_np(obs_h, np.int64).reshape(-1) if obs_h is not None else np.zeros(n_o, dtype=np.int64),
            crystal=_np(obs_crystal, np.int64).reshape(-1) if obs_crystal is not None else np.zeros(n_o, dtype=np.int64),
            batch=_np(obs_batch, np.int64).reshape(-1) if obs_batch is not None else np.zeros(n_o, dtype=np.int64),
            dose=_np(obs_dose, np.float64).reshape(-1),
            phi=_np(obs_phi, np.float64).reshape(-1) if obs_phi is not None else zeros,
            sign=_np(obs_sign, np.int64).reshape(-1) if obs_sign is not None else ones,
            s_inc=_np(obs_s_inc, np.float64).reshape(-1, 3) if obs_s_inc is not None else s0,
            s_dif=_np(obs_s_dif, np.float64).reshape(-1, 3) if obs_s_dif is not None else s0,
            wedge=_np(obs_wedge, np.float64).reshape(-1) if obs_wedge is not None else zeros,
            pass_id=_np(obs_pass, np.int64).reshape(-1) if obs_pass is not None else np.zeros(n_o, dtype=np.int64),
        )
    result = run_multixtal_core(
        model=model,
        hkl=_np(hkl, np.int64),
        cells=_np(cells, np.float64),
        sites_frac=_np(sites_frac, np.float64),
        i_plus=_np(i_plus, np.float64),
        i_minus=_np(i_minus, np.float64),
        sig_plus=_np(sig_plus, np.float64),
        sig_minus=_np(sig_minus, np.float64),
        mask_plus=_np(mask_plus, bool),
        mask_minus=_np(mask_minus, bool),
        f_mask=_np(f_mask, np.complex128),
        epsilon=_np(epsilon, np.float64).reshape(-1),
        centric=_np(centric, bool).reshape(-1),
        crystal=xray.meta.crystal,
        options=opts,
        engine_params=_engine_params(params),
        g_site=None if g_site is None else _np(g_site, np.complex128),
        obs=obs,
    )
    inf = [
        {
            "quantity": item.quantity,
            "shares": item.shares.tolist(),
            "effective_n": item.effective_n,
            "cooks": item.cooks.tolist(),
            "principal_angles": item.principal_angles.tolist(),
            "cooks_loadings": item.cooks_loadings.tolist(),
            "consistency": item.consistency.tolist(),
        }
        for item in result.influence
    ]
    n_data = result.y.shape[0]
    empty_c = np.zeros(result.y.shape[1], dtype=np.complex128)
    anom = result.anomalous
    maps = result.maps
    return {
        "stats": result.stats,
        "y": result.y,
        "mu": result.factors.mu,
        "scores": result.factors.scores,
        "loadings": result.factors.loadings,
        "strong": result.strong.astype(np.float64),
        "sigma_a": np.stack([item.sigma_a for item in result.nuisances], axis=0),
        "k_sol": np.array([item.k_sol for item in result.nuisances], dtype=np.float64),
        "b_sol": np.array([item.b_sol for item in result.nuisances], dtype=np.float64),
        "chi2_model": result.explanation.chi2_model,
        "chi2_mean": result.explanation.chi2_mean,
        "chi2_in_sample": result.explanation.chi2_in_sample,
        "chi2_loo": result.explanation.chi2_loo,
        "flagged": result.explanation.flagged.astype(np.float64),
        "f_explained": result.explanation.f_explained,
        "anom_intercept": empty_c.real if anom is None else anom.intercept,
        "anom_slopes": np.zeros((0, result.y.shape[1])) if anom is None else anom.slopes,
        "q": np.zeros(n_data) if anom is None else anom.q,
        "q_se": np.full(n_data, np.nan) if anom is None else anom.q_se,
        "mean_anomalous": empty_c if maps is None else maps.mean_anomalous,
        "factor_anomalous": np.zeros((0, result.y.shape[1]), dtype=np.complex128)
        if maps is None
        else maps.factor_anomalous,
        "factor_ordinary": np.zeros((0, result.y.shape[1]), dtype=np.complex128)
        if maps is None
        else maps.factor_ordinary,
        "influence": inf,
        "s": result.s,
        "g": result.g,
        "damage_loadings": np.zeros((0, result.y.shape[1])) if result.damage is None else result.damage.loadings,
        "damage_g": np.zeros((0, 1)) if result.damage is None else result.damage.profiles,
        "rho": np.ones(result.y.shape[0]) if result.damage is None else result.damage.rho,
        "zero_dose_i": np.zeros((result.y.shape[0], result.y.shape[1]))
        if result.damage is None
        else result.damage.zero_dose_i,
        "damage_maps": np.zeros((0, result.y.shape[1]), dtype=np.complex128)
        if result.damage_maps is None
        else result.damage_maps,
        "anom_corrected": empty_c if result.anom_corrected is None else result.anom_corrected,
        "dose_influence": None
        if result.dose_influence is None
        else {
            "quantity": result.dose_influence.quantity,
            "shares": result.dose_influence.shares.tolist(),
            "effective_n": result.dose_influence.effective_n,
            "cooks": result.dose_influence.cooks.tolist(),
            "consistency": result.dose_influence.consistency.tolist(),
        },
    }


multixtal_fit.compute_dtype = "float64"


def register_ops() -> None:
    register_op(OP_NAME, multixtal_fit, inputs=dict(_INPUTS), outputs=dict(_OUTPUTS))
