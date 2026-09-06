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
NUISANCE_FIT_OP_NAME = "ml_i_nuisance_fit"

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

_NUISANCE_INPUTS = {
    "f_calc": "MillerArray",
    "f_obs": "MillerArray",
    "tune_mask": "array",
    "s_sq": "array",
    "epsilon": "array",
    "centric": "array",
    "nu_bounds": "json",
    "nu": "json",
    "fit_nu": "json",
    "fit_scale": "json",
}
_NUISANCE_OUTPUTS = {
    "sigma_a": "array",
    "sigma_wilson": "array",
    "nu": "json",
    "nu_se": "json",
    "scale_k": "json",
    "p_theta": "json",
    "tune_nll": "json",
    "sigma_a_params": "json",
    "sigma_wilson_params": "json",
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
    dev = _DEVICE["device"]
    is_mps = dev.startswith("mps")
    cdtype = torch.complex64 if is_mps else torch.complex128
    np_cdtype = np.complex64 if is_mps else np.complex128
    fc = torch.as_tensor(_np(f_calc.data, np_cdtype), dtype=cdtype, device=dev)
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


def _compute_s_sq(unit_cell: Any, hkl: np.ndarray) -> np.ndarray:
    a, b, c, alpha_deg, beta_deg, gamma_deg = [float(x) for x in unit_cell]
    ar = np.radians(alpha_deg)
    br = np.radians(beta_deg)
    gr = np.radians(gamma_deg)
    ca, cb, cg = np.cos(ar), np.cos(br), np.cos(gr)
    sa, sb, sg = np.sin(ar), np.sin(br), np.sin(gr)
    val = 1.0 - ca**2 - cb**2 - cg**2 + 2.0 * ca * cb * cg
    val = max(val, 1e-12)
    V = a * b * c * np.sqrt(val)
    ar_star = b * c * sa / V
    br_star = a * c * sb / V
    cr_star = a * b * sg / V
    ca_star = (cb * cg - ca) / max(sb * sg, 1e-12)
    cb_star = (ca * cg - cb) / max(sa * sg, 1e-12)
    cg_star = (ca * cb - cg) / max(sa * sb, 1e-12)
    G_star = np.array(
        [
            [ar_star**2, ar_star * br_star * cg_star, ar_star * cr_star * cb_star],
            [ar_star * br_star * cg_star, br_star**2, br_star * cr_star * ca_star],
            [ar_star * cr_star * cb_star, br_star * cr_star * ca_star, cr_star**2],
        ],
        dtype=np.float64,
    )
    hkl_d = hkl.astype(np.float64)
    return np.einsum("ni,ij,nj->n", hkl_d, G_star, hkl_d)


def ml_i_nuisance_fit(
    f_calc: PackedMiller,
    f_obs: PackedMiller,
    tune_mask: Any,
    s_sq: Optional[Any] = None,
    epsilon: Optional[Any] = None,
    centric: Optional[Any] = None,
    nu_bounds: Optional[Any] = None,
    nu: Optional[Any] = None,
    fit_nu: Optional[Any] = True,
    fit_scale: Optional[Any] = False,
) -> dict[str, Any]:
    """Worker implementation for held-out nuisance parameter fitting (theta).

    Fits:
      - Sigma_N(s): Wilson-fit exponential curve Sigma_0 exp(-0.5 B_W s^2)
      - sigma_A(s): Read-style smooth curve sqrt(k) exp(-0.25 B_delta s^2)
      - nu: Student-t degrees of freedom in nu_bounds (global, with observed Fisher SE)
      - scale_k: optional scale factor on F_c
    All fitted strictly on the tune set (tune_mask == True).
    """
    import torch
    from scipy.optimize import minimize_scalar

    from phridge.contrib.intensity_ll.maps import posterior_moments
    from phridge.contrib.intensity_ll.mli import log_likelihood_normal, log_likelihood_t, normalize
    from phridge.sfcalc.ops import _DEVICE, _np

    if _np(f_calc.hkl).shape != _np(f_obs.hkl).shape or not np.array_equal(_np(f_calc.hkl), _np(f_obs.hkl)):
        raise ValueError("f_calc and f_obs must be on identical hkl lists")

    n = len(f_obs.data)
    tune = np.asarray(tune_mask, dtype=bool)
    if not np.any(tune):
        raise ValueError("tune_mask contains 0 reflections")

    io = _np(f_obs.data, np.float64)
    sig = _np(f_obs.sigmas, np.float64) if f_obs.sigmas is not None else np.ones(n, dtype=np.float64)
    fc_raw = np.abs(_np(f_calc.data, np.complex128))
    eps_np = np.asarray(epsilon, dtype=np.float64) if epsilon is not None else np.ones(n, dtype=np.float64)
    cen_np = np.asarray(centric, dtype=bool) if centric is not None else np.zeros(n, dtype=bool)

    if s_sq is not None:
        s_sq_np = np.asarray(s_sq, dtype=np.float64)
    else:
        s_sq_np = _compute_s_sq(f_obs.crystal.unit_cell, _np(f_obs.hkl))

    # 1. Fit Wilson curve Sigma_N(s) = Sigma_0 * exp(-0.5 * B_wilson * s^2) on tune reflections
    valid_tune = tune & (sig > 0) & (eps_np > 0)
    if not np.any(valid_tune):
        valid_tune = tune

    y_tune = io[valid_tune] / np.maximum(eps_np[valid_tune], 1e-12)
    x_tune = s_sq_np[valid_tune]

    # Divide into resolution shells to robustly fit slope and intercept
    n_valid = int(valid_tune.sum())
    n_shells = min(10, max(2, n_valid // 15))
    if n_shells >= 2 and (x_tune.max() - x_tune.min()) > 1e-6:
        bins = np.linspace(x_tune.min(), x_tune.max(), n_shells + 1)
        bin_idx = np.digitize(x_tune, bins[:-1]) - 1
        bx, by = [], []
        pos_mean = float(np.mean(y_tune[y_tune > 0])) if np.any(y_tune > 0) else 100.0
        for bi in range(n_shells):
            m = bin_idx == bi
            if np.any(m):
                bx.append(float(np.mean(x_tune[m])))
                mean_y = float(np.mean(y_tune[m]))
                by.append(max(mean_y, 0.05 * pos_mean))
        if len(bx) >= 2:
            slope, intercept = np.polyfit(bx, np.log(by), 1)
            sigma_0 = float(np.exp(intercept))
            b_wilson = float(max(-2.0 * slope, 0.0))
        else:
            sigma_0 = float(np.median(y_tune[y_tune > 0])) if np.any(y_tune > 0) else 100.0
            b_wilson = 0.0
    else:
        sigma_0 = float(np.median(y_tune[y_tune > 0])) if np.any(y_tune > 0) else 100.0
        b_wilson = 0.0

    sigma_w_full = sigma_0 * np.exp(-0.5 * b_wilson * s_sq_np)

    # 2. PyTorch setup for sigma_A, nu, and optional scale
    dev = _DEVICE["device"]
    is_mps = dev.startswith("mps")
    dtype = torch.float32 if is_mps else torch.float64

    tune_t = torch.as_tensor(tune, device=dev)
    fc_t = torch.as_tensor(fc_raw[tune], dtype=dtype, device=dev)
    io_t = torch.as_tensor(io[tune], dtype=dtype, device=dev)
    si_t = torch.as_tensor(sig[tune], dtype=dtype, device=dev)
    eps_t = torch.as_tensor(eps_np[tune], dtype=dtype, device=dev)
    cen_t = torch.as_tensor(cen_np[tune], device=dev)
    sw_t = torch.as_tensor(sigma_w_full[tune], dtype=dtype, device=dev)
    s_sq_t = torch.as_tensor(s_sq_np[tune], dtype=dtype, device=dev)

    bounds = [2.5, 200.0] if nu_bounds is None else [float(nu_bounds[0]), float(nu_bounds[1])]
    do_fit_nu = bool(fit_nu) and (nu is None or float(nu) < 199.0)
    do_fit_scale = bool(fit_scale)

    # Free parameters: uk (k = 0.999 * sigmoid(uk) + 1e-4), ub (b = softplus(ub)), us (scale = exp(us))
    uk = torch.tensor(1.5, dtype=dtype, device=dev, requires_grad=True)
    ub = torch.tensor(0.5, dtype=dtype, device=dev, requires_grad=True)
    us = torch.tensor(0.0, dtype=dtype, device=dev, requires_grad=True) if do_fit_scale else None

    params = [uk, ub] + ([us] if do_fit_scale else [])

    def compute_nll(nu_val: Optional[float]):
        k_t = 0.999 * torch.sigmoid(uk) + 1e-4
        b_t = torch.nn.functional.softplus(ub)
        sa_t = torch.clamp(torch.sqrt(k_t) * torch.exp(-0.25 * b_t * s_sq_t), 1e-4, 0.9999)
        fc_scaled = fc_t * torch.exp(us) if do_fit_scale else fc_t
        Ec, sA_n, Zo, sZ = normalize(fc_scaled, io_t, si_t, eps_t, sw_t, sa_t)
        if nu_val is not None and nu_val < 199.0:
            ll = log_likelihood_t(Ec, sA_n, Zo, sZ, cen_t, nu=float(nu_val), n_u=10)
        else:
            ll = log_likelihood_normal(Ec, sA_n, Zo, sZ, cen_t)
        return -ll.sum()

    current_nu: Optional[float] = float(nu) if (nu is not None and float(nu) < 199.0) else (7.0 if do_fit_nu else None)

    # Initial L-BFGS step on sigma_A (and scale)
    opt = torch.optim.LBFGS(params, max_iter=25, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = compute_nll(current_nu)
        loss.backward()
        return loss

    try:
        opt.step(closure)
    except Exception:
        pass

    # Nu refinement
    best_nu: Optional[float] = current_nu
    nu_se: Optional[float] = None

    if do_fit_nu:
        def nu_obj(nu_candidate: float) -> float:
            with torch.no_grad():
                val = compute_nll(float(nu_candidate)).item()
                return float(val)

        res_nu = minimize_scalar(nu_obj, bounds=(bounds[0], bounds[1]), method="bounded")
        best_nu = float(res_nu.x)
        current_nu = best_nu

        # Re-tune sigma_A at best nu
        try:
            opt.step(closure)
        except Exception:
            pass

        # Fisher information & standard error of nu from observed score and curvature
        with torch.no_grad():
            k_t = 0.999 * torch.sigmoid(uk) + 1e-4
            b_t = torch.nn.functional.softplus(ub)
            sa_t = torch.clamp(torch.sqrt(k_t) * torch.exp(-0.25 * b_t * s_sq_t), 1e-4, 0.9999)
            fc_scaled = fc_t * torch.exp(us) if do_fit_scale else fc_t
            Ec, sA_n, Zo, sZ = normalize(fc_scaled, io_t, si_t, eps_t, sw_t, sa_t)
            post = posterior_moments(Ec, sA_n, Zo, sZ, cen_t, nu=best_nu, n_u=10)
            if post.d_loglik_d_nu is not None:
                fisher_info = float((post.d_loglik_d_nu ** 2).sum().item())
                # Curvature via finite difference of score
                delta = 0.1
                post_p = posterior_moments(Ec, sA_n, Zo, sZ, cen_t, nu=best_nu + delta, n_u=10)
                post_m = posterior_moments(Ec, sA_n, Zo, sZ, cen_t, nu=max(best_nu - delta, 2.05), n_u=10)
                if post_p.d_loglik_d_nu is not None and post_m.d_loglik_d_nu is not None:
                    h_val = float((post_p.d_loglik_d_nu.sum() - post_m.d_loglik_d_nu.sum()).item() / (best_nu + delta - max(best_nu - delta, 2.05)))
                    curv_info = max(-h_val, 0.0)
                else:
                    curv_info = 0.0
                eff_info = max(fisher_info, curv_info, 1e-6)
                nu_se = float(1.0 / np.sqrt(eff_info))
            else:
                nu_se = None
    elif nu is not None and float(nu) < 199.0:
        best_nu = float(nu)
        nu_se = None
    else:
        best_nu = None
        nu_se = None

    # Evaluate final values
    k_final = float(0.999 * torch.sigmoid(uk).item() + 1e-4)
    b_final = float(torch.nn.functional.softplus(ub).item())
    scale_final = float(torch.exp(us).item()) if do_fit_scale else 1.0

    sigma_a_full = np.clip(np.sqrt(k_final) * np.exp(-0.25 * b_final * s_sq_np), 1e-4, 0.9999)

    with torch.no_grad():
        final_tune_nll = float(compute_nll(best_nu).item() / max(float(tune.sum()), 1.0))

    p_theta = 2 + 2 + (1 if best_nu is not None else 0) + (1 if do_fit_scale else 0)

    return {
        "sigma_a": sigma_a_full,
        "sigma_wilson": sigma_w_full,
        "nu": best_nu,
        "nu_se": nu_se,
        "scale_k": scale_final,
        "p_theta": int(p_theta),
        "tune_nll": final_tune_nll,
        "sigma_a_params": {"k": k_final, "b_delta": b_final},
        "sigma_wilson_params": {"sigma_0": sigma_0, "b_wilson": b_wilson},
    }


ml_i_nuisance_fit.compute_dtype = "float64"  # type: ignore[attr-defined]


def register_ops() -> None:
    """Register ``ml_i_maps`` and ``ml_i_nuisance_fit`` in the shared op catalog (idempotent)."""
    register_op(OP_NAME, ml_i_maps, inputs=dict(_INPUTS), outputs=dict(_OUTPUTS))
    register_op(NUISANCE_FIT_OP_NAME, ml_i_nuisance_fit, inputs=dict(_NUISANCE_INPUTS), outputs=dict(_NUISANCE_OUTPUTS))


register_ops()
