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
MAPS_OP_NAME = OP_NAME
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
    "f_mode": "array",
    "r_values": "json",
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
    "sigma_a_mode": "json",
    "n_sigma_a_bins": "json",
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

    fc_amp = np.abs(_np(f_calc.data, np.complex128))
    f_post_arr = out.f_post if out.f_post is not None else np.zeros(n, dtype=np.float64)
    f_mode_arr = out.f_mode if out.f_mode is not None else np.zeros(n, dtype=np.float64)
    io_arr = _np(f_obs.data, np.float64)
    i_calc = fc_amp**2

    def _calc_cc(x: np.ndarray, y: np.ndarray) -> float:
        if len(x) > 1 and np.std(x) > 1e-12 and np.std(y) > 1e-12:
            c = np.corrcoef(x, y)[0, 1]
            return float(c) if np.isfinite(c) else float("nan")
        return float("nan")

    r_val_dict: dict[str, Any] = {}
    has_post = f_post_arr is not None and np.any(f_post_arr > 0)
    has_mode = f_mode_arr is not None and np.any(f_mode_arr > 0)

    if has_post or has_mode:
        # 1. Posterior mean estimate <F> R-values and CC
        den_post_all = max(float(np.sum(f_post_arr)), 1e-12) if has_post else 1.0
        r_post_all = float(np.sum(np.abs(f_post_arr - fc_amp)) / den_post_all) if has_post else float("nan")
        r_val_dict["r_post_all"] = r_post_all
        r_val_dict["cc_post_all"] = _calc_cc(f_post_arr, fc_amp) if has_post else float("nan")

        # 2. Posterior mode estimate F_mode R-values and CC
        den_mode_all = max(float(np.sum(f_mode_arr)), 1e-12) if has_mode else 1.0
        r_mode_all = float(np.sum(np.abs(f_mode_arr - fc_amp)) / den_mode_all) if has_mode else float("nan")
        r_val_dict["r_mode_all"] = r_mode_all
        r_val_dict["cc_mode_all"] = _calc_cc(f_mode_arr, fc_amp) if has_mode else float("nan")

        # Primary r_all: default to posterior estimate <F> (or mode if post not available)
        r_val_dict["r_all"] = r_post_all if has_post else r_mode_all
        r_val_dict["cc_all"] = r_val_dict["cc_post_all"] if has_post else r_val_dict["cc_mode_all"]

        # 3. Direct Observed Intensity R-values (I_obs vs I_calc)
        den_i_all = max(float(np.sum(np.abs(io_arr))), 1e-12)
        r_val_dict["r_intensity_all"] = float(np.sum(np.abs(io_arr - i_calc)) / den_i_all)
        r_val_dict["cc_intensity_all"] = _calc_cc(io_arr, i_calc)

        rf_arr = _np(r_free)
        if rf_arr is not None:
            rf_bool = rf_arr.astype(bool)
            wk_bool = ~rf_bool
            if np.any(wk_bool):
                # Work reflections
                if has_post:
                    den_post_wk = max(float(np.sum(f_post_arr[wk_bool])), 1e-12)
                    r_val_dict["r_post_work"] = float(np.sum(np.abs(f_post_arr[wk_bool] - fc_amp[wk_bool])) / den_post_wk)
                    r_val_dict["cc_post_work"] = _calc_cc(f_post_arr[wk_bool], fc_amp[wk_bool])
                if has_mode:
                    den_mode_wk = max(float(np.sum(f_mode_arr[wk_bool])), 1e-12)
                    r_val_dict["r_mode_work"] = float(np.sum(np.abs(f_mode_arr[wk_bool] - fc_amp[wk_bool])) / den_mode_wk)
                    r_val_dict["cc_mode_work"] = _calc_cc(f_mode_arr[wk_bool], fc_amp[wk_bool])

                r_val_dict["r_work"] = r_val_dict.get("r_post_work", r_val_dict.get("r_mode_work", float("nan")))
                r_val_dict["cc_work"] = r_val_dict.get("cc_post_work", r_val_dict.get("cc_mode_work", float("nan")))

                den_i_wk = max(float(np.sum(np.abs(io_arr[wk_bool]))), 1e-12)
                r_val_dict["r_intensity_work"] = float(np.sum(np.abs(io_arr[wk_bool] - i_calc[wk_bool])) / den_i_wk)
                r_val_dict["cc_intensity_work"] = _calc_cc(io_arr[wk_bool], i_calc[wk_bool])

            if np.any(rf_bool):
                # Free reflections
                if has_post:
                    den_post_fr = max(float(np.sum(f_post_arr[rf_bool])), 1e-12)
                    r_val_dict["r_post_free"] = float(np.sum(np.abs(f_post_arr[rf_bool] - fc_amp[rf_bool])) / den_post_fr)
                    r_val_dict["cc_post_free"] = _calc_cc(f_post_arr[rf_bool], fc_amp[rf_bool])
                if has_mode:
                    den_mode_fr = max(float(np.sum(f_mode_arr[rf_bool])), 1e-12)
                    r_val_dict["r_mode_free"] = float(np.sum(np.abs(f_mode_arr[rf_bool] - fc_amp[rf_bool])) / den_mode_fr)
                    r_val_dict["cc_mode_free"] = _calc_cc(f_mode_arr[rf_bool], fc_amp[rf_bool])

                r_val_dict["r_free"] = r_val_dict.get("r_post_free", r_val_dict.get("r_mode_free", float("nan")))
                r_val_dict["cc_free"] = r_val_dict.get("cc_post_free", r_val_dict.get("cc_mode_free", float("nan")))

                den_i_fr = max(float(np.sum(np.abs(io_arr[rf_bool]))), 1e-12)
                r_val_dict["r_intensity_free"] = float(np.sum(np.abs(io_arr[rf_bool] - i_calc[rf_bool])) / den_i_fr)
                r_val_dict["cc_intensity_free"] = _calc_cc(io_arr[rf_bool], i_calc[rf_bool])
        else:
            r_val_dict["r_post_work"] = r_val_dict.get("r_post_all", float("nan"))
            r_val_dict["r_mode_work"] = r_val_dict.get("r_mode_all", float("nan"))
            r_val_dict["r_work"] = r_val_dict["r_all"]
            r_val_dict["cc_post_work"] = r_val_dict.get("cc_post_all", float("nan"))
            r_val_dict["cc_mode_work"] = r_val_dict.get("cc_mode_all", float("nan"))
            r_val_dict["cc_work"] = r_val_dict["cc_all"]
            r_val_dict["r_intensity_work"] = r_val_dict["r_intensity_all"]
            r_val_dict["cc_intensity_work"] = r_val_dict.get("cc_intensity_all", float("nan"))

    return {
        "difference": _miller_like(f_obs, out.difference, "ML_I_DIFF"),
        "model": _miller_like(f_obs, out.model, "ML_I_MODEL"),
        "gradient": _miller_like(f_obs, out.gradient, "ML_I_GRAD"),
        "newton": _miller_like(f_obs, out.newton, "ML_I_NEWTON"),
        "fom": out.fom,
        "robust_weight": out.robust_weight,
        "curvature": out.curvature,
        "f_post": out.f_post,
        "f_mode": f_mode_arr,
        "r_values": r_val_dict,
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
    sigma_a_mode: Optional[Any] = "bins",
    n_sigma_a_bins: Optional[Any] = None,
) -> dict[str, Any]:
    """Worker implementation for held-out nuisance parameter fitting (theta).

    Fits:
      - Sigma_N(s): Wilson-fit exponential curve Sigma_0 exp(-0.5 B_W s^2)
      - sigma_A(s): per-resolution-bin values (default, monotone in s²) or optional
        Read-style sqrt(k) exp(-0.25 B_delta s^2)
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
    tune = _np(tune_mask, bool)
    if not np.any(tune):
        raise ValueError("tune_mask contains 0 reflections")

    io = _np(f_obs.data, np.float64)
    sig = _np(f_obs.sigmas, np.float64) if f_obs.sigmas is not None else np.ones(n, dtype=np.float64)
    fc_raw = np.abs(_np(f_calc.data, np.complex128))
    eps_np = _np(epsilon, np.float64) if epsilon is not None else np.ones(n, dtype=np.float64)
    cen_np = _np(centric, bool) if centric is not None else np.zeros(n, dtype=bool)

    if s_sq is not None:
        s_sq_np = _np(s_sq, np.float64)
    else:
        uc = f_obs.meta.crystal.unit_cell if hasattr(f_obs, "meta") else f_obs.crystal.unit_cell
        s_sq_np = _compute_s_sq(uc, _np(f_obs.hkl))

    mode = str(sigma_a_mode or "bins").strip().lower()
    if mode not in ("bins", "read", "bin", "shell", "shells"):
        mode = "bins"
    if mode in ("bin", "shell", "shells"):
        mode = "bins"

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
    s_sq_tune_np = s_sq_np[tune]

    bounds = [2.5, 200.0] if nu_bounds is None else [float(nu_bounds[0]), float(nu_bounds[1])]
    do_fit_nu = bool(fit_nu) and (nu is None or float(nu) < 199.0)
    do_fit_scale = bool(fit_scale)

    us = torch.tensor(0.0, dtype=dtype, device=dev, requires_grad=True) if do_fit_scale else None

    # --- sigma_A parameterization ---
    sa_bin_centers: Optional[np.ndarray] = None
    sa_bin_values: Optional[np.ndarray] = None
    if mode == "bins":
        n_tune = int(tune.sum())
        if n_sigma_a_bins is not None:
            n_sa = max(3, int(n_sigma_a_bins))
        else:
            # Enough shells to resolve outer-shell decay; not so many that NLL overfits.
            n_sa = int(min(20, max(6, n_tune // 250)))
        s_lo = float(s_sq_tune_np.min())
        s_hi = float(s_sq_tune_np.max())
        if s_hi - s_lo < 1e-8:
            edges = np.array([s_lo - 1e-6, s_hi + 1e-6], dtype=np.float64)
            n_sa = 1
        else:
            edges = np.linspace(s_lo, s_hi, n_sa + 1)
        # digitize: bin 0 = lowest s² (lowest resolution)
        shell_idx_tune = np.clip(np.digitize(s_sq_tune_np, edges[1:-1], right=False), 0, n_sa - 1)
        shell_idx_t = torch.as_tensor(shell_idx_tune, dtype=torch.long, device=dev)
        # Monotone decreasing in s²: sa[0] >= sa[1] >= ... via softplus drops
        u0 = torch.tensor(2.0, dtype=dtype, device=dev, requires_grad=True)
        if n_sa > 1:
            raw_deltas = torch.full((n_sa - 1,), 0.3, dtype=dtype, device=dev, requires_grad=True)
            sa_params = [u0, raw_deltas] + ([us] if do_fit_scale else [])
        else:
            raw_deltas = None
            sa_params = [u0] + ([us] if do_fit_scale else [])

        def _sa_bins_from_params():
            if n_sa == 1 or raw_deltas is None:
                logits = u0.reshape(1)
            else:
                drops = torch.nn.functional.softplus(raw_deltas)
                cum = torch.cat(
                    [torch.zeros(1, dtype=dtype, device=dev), torch.cumsum(drops, dim=0)]
                )
                logits = u0 - cum
            return 0.01 + 0.989 * torch.sigmoid(logits)

        def _sa_tune_from_params():
            sa_bins = _sa_bins_from_params()
            return sa_bins[shell_idx_t]

        centers = 0.5 * (edges[:-1] + edges[1:])
        sa_bin_centers = centers
    else:
        # Read-style: σ_A = √k exp(-0.25 BΔ s²)
        uk = torch.tensor(1.5, dtype=dtype, device=dev, requires_grad=True)
        # Start with a nontrivial BΔ so the fit is not pinned at a constant.
        ub = torch.tensor(2.0, dtype=dtype, device=dev, requires_grad=True)
        sa_params = [uk, ub] + ([us] if do_fit_scale else [])

        def _sa_tune_from_params():
            k_t = 0.999 * torch.sigmoid(uk) + 1e-4
            b_t = torch.nn.functional.softplus(ub)
            return torch.clamp(torch.sqrt(k_t) * torch.exp(-0.25 * b_t * s_sq_t), 1e-4, 0.9999)

    params = sa_params

    def compute_nll(nu_val: Optional[float]):
        sa_t = _sa_tune_from_params()
        fc_scaled = fc_t * torch.exp(us) if do_fit_scale else fc_t
        Ec, sA_n, Zo, sZ = normalize(fc_scaled, io_t, si_t, eps_t, sw_t, sa_t)
        if nu_val is not None and nu_val < 199.0:
            ll = log_likelihood_t(Ec, sA_n, Zo, sZ, cen_t, nu=float(nu_val), n_u=10)
        else:
            ll = log_likelihood_normal(Ec, sA_n, Zo, sZ, cen_t)
        return -ll.sum()

    current_nu: Optional[float] = float(nu) if (nu is not None and float(nu) < 199.0) else (7.0 if do_fit_nu else None)

    opt = torch.optim.LBFGS(params, max_iter=40, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = compute_nll(current_nu)
        loss.backward()
        return loss

    try:
        opt.step(closure)
    except Exception:
        pass

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

        try:
            opt.step(closure)
        except Exception:
            pass

        with torch.no_grad():
            sa_t = _sa_tune_from_params()
            fc_scaled = fc_t * torch.exp(us) if do_fit_scale else fc_t
            Ec, sA_n, Zo, sZ = normalize(fc_scaled, io_t, si_t, eps_t, sw_t, sa_t)
            post = posterior_moments(Ec, sA_n, Zo, sZ, cen_t, nu=best_nu, n_u=10)
            if post.d_loglik_d_nu is not None:
                fisher_info = float((post.d_loglik_d_nu ** 2).sum().item())
                delta = 0.1
                post_p = posterior_moments(Ec, sA_n, Zo, sZ, cen_t, nu=best_nu + delta, n_u=10)
                post_m = posterior_moments(Ec, sA_n, Zo, sZ, cen_t, nu=max(best_nu - delta, 2.05), n_u=10)
                if post_p.d_loglik_d_nu is not None and post_m.d_loglik_d_nu is not None:
                    h_val = float(
                        (post_p.d_loglik_d_nu.sum() - post_m.d_loglik_d_nu.sum()).item()
                        / (best_nu + delta - max(best_nu - delta, 2.05))
                    )
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

    scale_final = float(torch.exp(us).item()) if do_fit_scale else 1.0

    with torch.no_grad():
        if mode == "bins":
            sa_bins_t = _sa_bins_from_params()
            sa_bin_values = sa_bins_t.detach().cpu().numpy().astype(np.float64)
            # Assign every reflection by s² shell (same edges as tune fit).
            if n_sa == 1:
                sigma_a_full = np.full(n, float(sa_bin_values[0]), dtype=np.float64)
            else:
                shell_all = np.clip(np.digitize(s_sq_np, edges[1:-1], right=False), 0, n_sa - 1)
                sigma_a_full = sa_bin_values[shell_all]
            # Piecewise-linear interpolation in s² for smoother reporting / grads.
            if n_sa >= 2 and sa_bin_centers is not None:
                sigma_a_full = np.interp(
                    s_sq_np,
                    sa_bin_centers,
                    sa_bin_values,
                    left=float(sa_bin_values[0]),
                    right=float(sa_bin_values[-1]),
                )
            sigma_a_full = np.clip(sigma_a_full, 1e-4, 0.9999)
            k_final = float(sa_bin_values[0] ** 2) if sa_bin_values is not None else float("nan")
            # Effective BΔ from end-points (for header only).
            if sa_bin_centers is not None and sa_bin_values is not None and n_sa >= 2:
                ds = float(sa_bin_centers[-1] - sa_bin_centers[0])
                if ds > 1e-12 and sa_bin_values[0] > 1e-6:
                    ratio = max(float(sa_bin_values[-1] / sa_bin_values[0]), 1e-6)
                    b_final = float(max(-4.0 * np.log(ratio) / ds, 0.0))
                else:
                    b_final = 0.0
            else:
                b_final = 0.0
            sigma_a_params = {
                "mode": "bins",
                "n_bins": int(n_sa),
                "k": k_final,
                "b_delta": b_final,
                "bin_centers_s2": sa_bin_centers.tolist() if sa_bin_centers is not None else [],
                "bin_sigma_a": sa_bin_values.tolist() if sa_bin_values is not None else [],
            }
        else:
            k_final = float(0.999 * torch.sigmoid(uk).item() + 1e-4)
            b_final = float(torch.nn.functional.softplus(ub).item())
            sigma_a_full = np.clip(
                np.sqrt(k_final) * np.exp(-0.25 * b_final * s_sq_np), 1e-4, 0.9999
            )
            sigma_a_params = {"mode": "read", "k": k_final, "b_delta": b_final}

        final_tune_nll = float(compute_nll(best_nu).item() / max(float(tune.sum()), 1.0))

    p_theta = (int(n_sa) if mode == "bins" else 2) + (1 if best_nu is not None else 0) + (1 if do_fit_scale else 0)
    # Wilson contributes 2 more params in the accounting used by benchmarks
    p_theta += 2

    return {
        "sigma_a": sigma_a_full,
        "sigma_wilson": sigma_w_full,
        "nu": best_nu,
        "nu_se": nu_se,
        "scale_k": scale_final,
        "p_theta": int(p_theta),
        "tune_nll": final_tune_nll,
        "sigma_a_params": sigma_a_params,
        "sigma_wilson_params": {"sigma_0": sigma_0, "b_wilson": b_wilson},
    }


ml_i_nuisance_fit.compute_dtype = "float64"  # type: ignore[attr-defined]


TARGET_AND_GRADIENTS_OP_NAME = "ml_i_target_and_gradients"

_TARGET_AND_GRADIENTS_INPUTS = {
    "xray": "XrayStructure",
    "table": "ScatteringTable",
    "f_obs": "MillerArray",
    "params": "SfEngineParams",
    "target": "json",
    "weights": "array",
    "r_free": "array",
    "alpha": "array",
    "beta": "array",
    "epsilon": "array",
    "centric": "array",
    "precondition": "json",
    "damping": "json",
    "scale_factor": "json",
    "f_bulk": "array",
}

_TARGET_AND_GRADIENTS_OUTPUTS = {
    "f_calc": "MillerArray",
    "target": "TargetResult",
    "gradients": "SfGradients",
    "preconditioned_gradients": "SfGradients",
    "curvatures": "SfCurvatures",
}


def _ortho_fract_matrices(unit_cell: Any) -> tuple[np.ndarray, np.ndarray]:
    """Compute orthogonalization matrix O (frac -> cart) and fractionalization matrix F (cart -> frac)."""
    a, b, c, alpha_deg, beta_deg, gamma_deg = [float(x) for x in unit_cell]
    ar, br, gr = np.radians(alpha_deg), np.radians(beta_deg), np.radians(gamma_deg)
    ca, cb, cg = np.cos(ar), np.cos(br), np.cos(gr)
    sa, sb, sg = np.sin(ar), np.sin(br), np.sin(gr)
    val = 1.0 - ca**2 - cb**2 - cg**2 + 2.0 * ca * cb * cg
    V = a * b * c * np.sqrt(max(val, 1e-12))
    O = np.array(
        [
            [a, b * cg, c * cb],
            [0.0, b * sg, c * (ca - cb * cg) / max(sg, 1e-12)],
            [0.0, 0.0, V / max(a * b * sg, 1e-12)],
        ],
        dtype=np.float64,
    )
    F = np.linalg.inv(O)
    return O, F


def ml_i_target_and_gradients(
    xray: Any,
    table: Any,
    f_obs: Any,
    params: Any,
    target: dict,
    weights: Optional[Any] = None,
    r_free: Optional[Any] = None,
    alpha: Optional[Any] = None,
    beta: Optional[Any] = None,
    epsilon: Optional[Any] = None,
    centric: Optional[Any] = None,
    precondition: Optional[Any] = False,
    damping: Optional[Any] = 0.05,
    scale_factor: Optional[Any] = 1.0,
    f_bulk: Optional[Any] = None,
) -> dict[str, Any]:
    """Whole chain for intensity likelihood: F_calc -> ml_i target -> dQ/dF -> dQ/d(scatterer params).

    Optionally computes exact Gauss-Newton diagonal preconditioners for Cartesian
    sites, occupancy, isotropic U, and anisotropic U* in a single server pass.
    """
    import torch

    from phridge.contrib.intensity_ll.target import IntensityLogLikelihood
    from phridge.sfcalc.ops import (
        _DEVICE,
        _engine,
        _miller_like,
        _np,
        _observations,
        _target_result,
    )
    from phridge.sfcalc.packing import PackedSfCurvatures, PackedSfGradients
    from phridge.sfcalc.targets import build_target

    do_precond = bool(precondition)
    damp_val = float(damping) if damping is not None else 0.05
    k_scale = float(scale_factor) if scale_factor is not None else 1.0

    eng = _engine(xray, table, _np(f_obs.hkl), params)
    spec = dict(target)
    if spec.get("name") != "ml_i":
        spec["name"] = "ml_i"
    tgt = build_target(spec)
    obs = _observations(f_obs, weights, r_free, alpha, beta, epsilon, centric)

    p = eng.tensors(requires_grad=True)
    fc = eng.f_calc(*p)
    if f_bulk is not None:
        fb_np = _np(f_bulk, np.complex128)
        fb_t = torch.as_tensor(fb_np, dtype=fc.dtype, device=fc.device)
        f_model = k_scale * (fc + fb_t)
    else:
        f_model = k_scale * fc

    ev = tgt.evaluate(f_model.detach(), obs, compute_curvature=do_precond)
    g = torch.as_tensor(ev.d_target_d_f_calc, dtype=fc.dtype, device=fc.device)
    # Scale derivative w.r.t fc: dQ/dfc = k_scale * dQ/df_model
    q = (k_scale * fc * g.conj()).real.sum()
    grads = torch.autograd.grad(q, p, allow_unused=True)
    arrays = [
        (torch.zeros_like(t) if gr is None else gr).detach().cpu().numpy().astype(np.float64)
        for t, gr in zip(p, grads)
    ]

    gradients = PackedSfGradients(
        d_site_frac=arrays[0],
        d_occupancy=arrays[1],
        d_u_iso=arrays[2],
        d_u_star=arrays[3],
        d_fp=arrays[4],
        d_fdp=arrays[5],
        target=ev.value,
    )

    n_sc = len(xray.meta.scatterers)
    if do_precond:
        # Scale radial/tangential curvatures by k_scale^2
        curv_r = _np(ev.curv_radial) * (k_scale**2)
        curv_t = _np(ev.curv_tangential) * (k_scale**2)
        blocks = eng.gauss_newton_blocks(curv_r, curv_t)
        curvatures = PackedSfCurvatures(
            site_frac=blocks["site_frac"],
            occupancy=blocks["occupancy"],
            u_iso=blocks["u_iso"],
            u_star=blocks["u_star"],
            fp=blocks["fp"],
            fdp=blocks["fdp"],
        )

        def _scale_by_diag(H_diag: np.ndarray, g: np.ndarray) -> np.ndarray:
            """Levenberg-damped Jacobi preconditioning: g / (max(H,0) + λ)."""
            H_diag = np.asarray(H_diag, dtype=np.float64)
            g = np.asarray(g, dtype=np.float64)
            pos = H_diag[H_diag > 0]
            med = float(np.median(pos)) if pos.size > 0 else 1.0
            lam = max(1e-6, med * damp_val)
            return g / (np.maximum(H_diag, 0.0) + lam)

        # 1. Site preconditioning in Cartesian frame
        unit_cell = [float(x) for x in xray.meta.crystal.unit_cell]
        O_mat, F_mat = _ortho_fract_matrices(unit_cell)
        H_frac = _np(blocks["site_frac"], np.float64)  # (N, 3, 3)
        H_cart = np.einsum("ia,nab,bj->nij", F_mat.T, H_frac, F_mat)
        diag_H_cart = np.diagonal(H_cart, axis1=1, axis2=2)  # (N, 3)
        g_cart = arrays[0] @ F_mat
        g_cart_pre = _scale_by_diag(diag_H_cart, g_cart)
        g_frac_pre = g_cart_pre @ O_mat

        # 2. Occupancy
        g_occ_pre = _scale_by_diag(_np(blocks["occupancy"], np.float64), arrays[1])

        # 3. Isotropic B/U
        g_u_pre = _scale_by_diag(_np(blocks["u_iso"], np.float64), arrays[2])

        # 4. Anisotropic U* (per-component diagonal of the 6×6 GN block)
        H_ustar = _np(blocks["u_star"], np.float64)  # (N, 6, 6)
        diag_ustar = np.diagonal(H_ustar, axis1=1, axis2=2)  # (N, 6)
        g_ustar_pre = _scale_by_diag(diag_ustar, arrays[3])

        preconditioned_gradients = PackedSfGradients(
            d_site_frac=g_frac_pre,
            d_occupancy=g_occ_pre,
            d_u_iso=g_u_pre,
            d_u_star=g_ustar_pre,
            d_fp=arrays[4],
            d_fdp=arrays[5],
            target=ev.value,
        )
    else:
        curvatures = PackedSfCurvatures(
            site_frac=np.zeros((n_sc, 3, 3), dtype=np.float64),
            occupancy=np.zeros(n_sc, dtype=np.float64),
            u_iso=np.zeros(n_sc, dtype=np.float64),
            u_star=np.zeros((n_sc, 6, 6), dtype=np.float64),
            fp=np.zeros(n_sc, dtype=np.float64),
            fdp=np.zeros(n_sc, dtype=np.float64),
        )
        preconditioned_gradients = gradients

    fc_ret = fc.detach().cpu().numpy().astype(np.complex128)
    return {
        "f_calc": _miller_like(f_obs, fc_ret, "F_calc"),
        "target": _target_result(tgt.name, ev),
        "gradients": preconditioned_gradients if do_precond else gradients,
        "preconditioned_gradients": preconditioned_gradients,
        "curvatures": curvatures,
    }


ml_i_target_and_gradients.compute_dtype = "float64"  # type: ignore[attr-defined]


def register_ops() -> None:
    """Register ``ml_i_maps`` and ``ml_i_nuisance_fit`` in the shared op catalog (idempotent)."""
    register_op(OP_NAME, ml_i_maps, inputs=dict(_INPUTS), outputs=dict(_OUTPUTS))
    register_op(NUISANCE_FIT_OP_NAME, ml_i_nuisance_fit, inputs=dict(_NUISANCE_INPUTS), outputs=dict(_NUISANCE_OUTPUTS))
    register_op(
        TARGET_AND_GRADIENTS_OP_NAME,
        ml_i_target_and_gradients,
        inputs=dict(_TARGET_AND_GRADIENTS_INPUTS),
        outputs=dict(_TARGET_AND_GRADIENTS_OUTPUTS),
    )


register_ops()
