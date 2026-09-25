"""Worker op ``ml_i_maps``: map coefficients from the intensity likelihood posterior.

Inputs mirror ``target_eval`` (same ``f_obs`` / per-reflection arrays / target spec) plus a
``maps`` JSON dict validated as :class:`IntensityMapOptions`. Outputs are complex
``MillerArray`` coefficient sets on the ``f_obs`` hkl list and per-reflection weight arrays.
The impl imports torch lazily so a Phenix client can import this module.
"""

from __future__ import annotations

import math
import time
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
    "nu": "array",
    "beta_residual": "array",
    "maps": "json",
    "d_spacings": "array",
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
    "nu_mode": "json",
    "nu_grid": "json",
    "fit_scale": "json",
    "sigma_a_mode": "json",
    "n_sigma_a_bins": "json",
    "tv_norm": "json",
    "fit_sigma_wilson": "json",
    "wilson_model": "json",
    "wilson_max_iter": "json",
    "wilson_bins": "array",
    "beta_mode": "json",
    "sigma_a_shape": "json",
    "smooth_sigma_a": "json",
    "smooth_beta": "json",
    "beta_consistency_prior": "json",
    "sigma_a_tensor": "json",
    "sphericity": "json",
    "f_atoms": "array",
    "f_bulk": "array",
    "local_sigma_a": "json",
    "spatial_sigma_a_v2": "SpatialSigmaAV2",
}
_NUISANCE_OUTPUTS = {
    "sigma_a": "array",
    "sigma_wilson": "array",
    "beta": "array",
    "beta_residual": "array",
    "nu": "json",
    "nu_per_refl": "array",
    "nu_se": "json",
    "nu_params": "json",
    "scale_k": "json",
    "p_theta": "json",
    "tune_nll": "json",
    "sigma_a_params": "json",
    "sigma_wilson_params": "json",
}


def _score_test_enabled() -> bool:
    """``PHRIDGE_SCORE_TEST`` gate; off by default because the reference variance
    costs a dozen extra posterior solves per reflection."""
    import os

    return os.environ.get("PHRIDGE_SCORE_TEST", "").strip().lower() in {"1", "true", "yes", "on"}


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
    nu: Optional[Any] = None,
    beta_residual: Optional[Any] = None,
    maps: Optional[dict] = None,
    d_spacings: Optional[Any] = None,
) -> dict[str, Any]:
    """Worker implementation (torch side)."""
    import torch

    from phridge.contrib.intensity_ll.maps import IntensityMapOptions, intensity_map_coefficients
    from phridge.contrib.intensity_ll.mli import normalize
    from phridge.contrib.intensity_ll.rint import integrated_s_report
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
    obs = _observations(
        f_obs, weights, r_free, alpha, beta, epsilon, centric, nu=nu,
        beta_residual=beta_residual,
    )
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

    # S_post / S_prior (reporting only; denser GL quadrature than maps)
    try:
        fo = obs.data
        eps = obs.epsilon if obs.epsilon is not None else torch.ones_like(fo)
        centric_t = obs.centric if obs.centric is not None else torch.zeros_like(fo, dtype=torch.bool)
        sA = tgt._sigma_a(obs)
        sW = tgt._sigma_wilson(obs)
        sig = tgt._sigma(obs)
        fc_abs = fc.abs()
        ok = (sA > 0) & (sA < 1.0 - 1e-6) & (sW > 0) & (sig > 0) & (eps > 0) & (fc_abs > 0)
        sA_s = torch.where(ok, sA, torch.full_like(sA, 0.5))
        sW_s = torch.where(ok, sW, torch.ones_like(sW))
        sig_s = torch.where(ok, sig, torch.ones_like(sig))
        eps_s = torch.where(ok, eps, torch.ones_like(eps))
        fc_s = torch.where(ok, fc_abs, torch.ones_like(fc_abs))
        fo_s = torch.where(ok, fo, torch.zeros_like(fo))
        # Through the target so the reported posterior statistics describe the same prior
        # the refinement target used; otherwise S_post would audit a different model.
        Ec, sA_n, Zo, sZ, _ = tgt.normalized(fc_s, fo_s, sig_s, eps_s, sW_s, sA_s, obs)
        # Keep Rice width a=1-σ_A² away from 0 for stable reporting quadrature
        sA_n = sA_n.clamp(1e-4, 1.0 - 1e-4)
        # float64 on CPU for reporting accuracy (MPS maps may be float32).
        # Move to CPU *before* widening: a fused ``.to(dtype=float64, device="cpu")``
        # on an MPS tensor raises for small inputs but silently returns
        # uninitialized memory above ~1k elements (MPS has no float64).
        Ec64 = Ec.detach().cpu().double()
        sA64 = sA_n.detach().cpu().double()
        Zo64 = Zo.detach().cpu().double()
        sZ64 = sZ.detach().cpu().double()
        # Preserve negative Z_o (same as maps). Only scrub non-finite.
        Zo64 = torch.nan_to_num(Zo64, nan=0.0, posinf=1e6, neginf=-1e6)
        sZ64 = torch.nan_to_num(sZ64, nan=1.0, posinf=1e6, neginf=1.0).clamp_min(1e-12)
        Ec64 = torch.nan_to_num(Ec64, nan=0.0, posinf=1e3, neginf=0.0).clamp_min(0.0)
        cen64 = centric_t.detach().to(device="cpu")
        work = ok.detach().to(device="cpu")
        free_mask_arg: Optional[Any] = None
        rf_np = _np(r_free)
        if rf_np is not None and np.any(np.asarray(rf_np).astype(bool)):
            free = torch.as_tensor(np.asarray(rf_np).astype(bool), device="cpu") & work
            work = work & ~free
            free_mask_arg = free

        bin_size = opts.bin_size
        if bin_size is None:
            import os

            raw_bs = os.environ.get("PHRIDGE_STATS_BIN_SIZE", "").strip()
            bin_size = int(raw_bs) if raw_bs.isdigit() else 500
        d_np = _np(d_spacings)
        if d_np is None:
            # Derive d from f_obs miller indices (no extra client input required)
            try:
                uc = f_obs.meta.crystal.unit_cell
                s_sq = _compute_s_sq(uc, _np(f_obs.hkl))
                d_np = 1.0 / np.sqrt(np.maximum(s_sq, 1e-300))
            except Exception:
                d_np = None
                bin_size = None
        d_t = (
            torch.as_tensor(np.asarray(d_np, dtype=np.float64), dtype=torch.float64)
            if d_np is not None
            else None
        )

        nu_arg: Any = None
        if obs.nu is not None:
            nu_t = obs.nu.detach().cpu().double()
            nu_t = torch.nan_to_num(nu_t, nan=200.0, posinf=200.0, neginf=2.5).clamp(2.05, 500.0)
            nu_arg = nu_t
        elif tgt.nu is not None and np.isfinite(float(tgt.nu)):
            nu_arg = float(tgt.nu)

        # Match map-coefficient quadrature exactly (same posterior as r_post / f_post)
        rint = integrated_s_report(
            Ec64,
            sA64,
            Zo64,
            sZ64,
            cen64,
            work_mask=work,
            free_mask=free_mask_arg,
            nu=nu_arg,
            d_spacings=d_t,
            bin_size=bin_size,
            n_u=int(getattr(tgt, "n_u", 12) or 12),
            snr_strong=float(getattr(tgt, "snr_strong", 5.0) or 5.0),
            n_hermite=int(getattr(tgt, "n_hermite", 7) or 7),
            n_legendre=int(getattr(tgt, "n_legendre", 24) or 24),
            k_window=float(getattr(tgt, "k_window", 8.0) or 8.0),
            chunk_size=4096,
            score_test=_score_test_enabled(),
        )
        r_val_dict.update(rint.as_s_values())
        r_val_dict["s_report_version"] = 9

        # E_C = |F_c|/sqrt(εΣ) must be O(1) by construction. When it is not, the
        # Wilson Σ (obs.beta) is broken and every normalized-unit statistic is
        # meaningless — report that instead of a plausible-looking R.
        ec_med = float(r_val_dict.get("ec_median", float("nan")))
        if not (np.isfinite(ec_med) and 0.01 <= ec_med <= 100.0):
            for _k in (
                "s_post_work", "s_post_free", "s_post_all",
                "s_prior_work", "s_prior_free", "s_prior_all",
            ):
                r_val_dict[_k] = float("nan")
            for _k in ("s_post_work_bins", "s_post_free_bins", "s_prior_work_bins", "s_prior_free_bins"):
                if _k in r_val_dict:
                    r_val_dict[_k] = [float("nan")] * len(r_val_dict[_k])
            r_val_dict["s_report_error"] = (
                f"normalization broken: median E_C={ec_med:.4g} (expected ~1). "
                "sigma_wilson (obs.beta) is not a valid Wilson Sigma."
            )
        s_post_w = float(r_val_dict.get("s_post_work", float("nan")))
        k_s_w = float(r_val_dict.get("k_s_work", float("nan")))
        s_vis_w = float(r_val_dict.get("s_vis_work", float("nan")))
        need_dbg = (
            (not np.isfinite(s_post_w))
            or (not np.isfinite(k_s_w))
            or (np.isfinite(s_post_w) and s_post_w > 0.5)
            or int(r_val_dict.get("n_ec_outliers", 0)) > 0
            or (
                np.isfinite(s_post_w)
                and np.isfinite(s_vis_w)
                and s_vis_w > 1e-6
                and s_post_w > 3.0 * s_vis_w
            )
        )
        if need_dbg:
            n_ok = int(work.sum().item()) if hasattr(work, "sum") else -1

            def _mmm(t: Any) -> Any:
                """min / median / max of a raw input, unmasked and unclamped."""
                if t is None:
                    return None
                tt = t.detach().cpu().double().reshape(-1)
                tt = tt[torch.isfinite(tt)]
                if tt.numel() == 0:
                    return "all non-finite"
                return [float(tt.min()), float(tt.median()), float(tt.max())]

            ec_ok = Ec64[work] if n_ok > 0 else Ec64[:0]
            r_val_dict["s_report_debug"] = {
                "n_ok_work": n_ok,
                "n_total": int(Ec64.numel()),
                "k_s_work": k_s_w,
                "k_s_prior_work": float(r_val_dict.get("k_s_prior_work", float("nan"))),
                "s_post_work": s_post_w,
                "s_vis_work": s_vis_w,
                "n_ec_outliers": int(r_val_dict.get("n_ec_outliers", 0)),
                "Ec_used_min_med_max": _mmm(ec_ok),
                # Raw nuisance inputs exactly as the client sent them (alpha=σ_A, beta=Σ_W)
                "RAW_sigma_a(alpha)": _mmm(sA),
                "RAW_sigma_wilson(beta)": _mmm(sW),
                "RAW_fcalc": _mmm(fc_abs),
                "RAW_iobs": _mmm(fo),
                "RAW_sigma_i": _mmm(sig),
                "RAW_epsilon": _mmm(eps),
                "n_sigma_a_outside_0_1": int(((sA <= 0) | (sA >= 1.0)).sum().item()),
                "n_sigma_wilson_nonpos": int((sW <= 0).sum().item()),
                "dtype_device": f"{sA.dtype}/{sA.device}",
                "nu_kind": (
                    "per_refl"
                    if isinstance(nu_arg, torch.Tensor) and nu_arg.numel() > 1
                    else ("scalar" if nu_arg is not None else "none")
                ),
            }
    except Exception as exc:
        import traceback
        import sys

        r_val_dict.setdefault("s_post_work", float("nan"))
        r_val_dict.setdefault("s_post_free", float("nan"))
        r_val_dict.setdefault("s_post_all", float("nan"))
        r_val_dict.setdefault("s_prior_work", float("nan"))
        r_val_dict.setdefault("s_prior_free", float("nan"))
        r_val_dict.setdefault("s_prior_all", float("nan"))
        r_val_dict["s_report_error"] = f"{type(exc).__name__}: {exc}"
        r_val_dict["s_post_traceback"] = traceback.format_exc(limit=6)
        try:
            print(f"[ml_i_maps] S_post failed: {r_val_dict['s_report_error']}", file=sys.stderr, flush=True)
        except Exception:
            pass

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


def _shell_standard_errors(
    *,
    torch: Any,
    fb: Any,
    normalize: Any,
    log_likelihood_normal: Any,
    log_likelihood_t: Any,
    sigma_a: np.ndarray,
    beta: np.ndarray,
    free_beta: bool,
    shell_idx: Any,
    fc_t: Any,
    io_t: Any,
    si_t: Any,
    eps_t: Any,
    sw_t: Any,
    cen_t: Any,
    nu_tune: Any,
    nu_scalar: Optional[float],
    dtype: Any,
    device: Any,
) -> Any:
    """Standard errors of per-shell ``(σ_A, β)`` from the Hessian of the tune NLL.

    Every reflection belongs to exactly one shell, so the NLL is a sum of independent
    per-shell terms and its Hessian is **block diagonal** with 2x2 blocks. That is worth
    exploiting: two Hessian-vector products recover every block, where a dense Hessian
    would cost one backward pass per parameter (40 for 20 shells). Probing with a vector
    of ones in the σ_A slots picks out each block's σ_A column precisely *because* the
    off-diagonal blocks vanish.

    The Hessian is of the unpenalized NLL: these say how well the data pin each shell, and
    the smoothness penalty is not data. Evaluated in a fresh free logit parameterization at
    the optimum, so a monotone fit still gets honest per-shell errors.

    In constrained mode β is not free; its error follows from σ_A by the delta method on
    ``β = 1 - σ_A²``, which is where the exact ``-1`` correlation comes from.
    """
    k = int(len(sigma_a))
    u = torch.as_tensor(fb.logit_from_sigma_a(sigma_a), dtype=dtype, device=device)
    u = u.clone().requires_grad_(True)
    v = torch.as_tensor(fb.logit_from_beta(beta), dtype=dtype, device=device)
    v = v.clone().requires_grad_(True)

    def _nll(u_vec: Any, v_vec: Any) -> Any:
        sa_bins = fb.SIGMA_A_LO + fb.SIGMA_A_SPAN * torch.sigmoid(u_vec)
        sa_t = sa_bins[shell_idx]
        ec, sa_n, zo, sz = normalize(fc_t, io_t, si_t, eps_t, sw_t, sa_t)
        if free_beta:
            b_bins = fb.BETA_LO + fb.BETA_SPAN * torch.sigmoid(v_vec)
            ec, sa_n = fb.rice_inputs(ec, sa_t, b_bins[shell_idx])
        if nu_tune is not None:
            ll = log_likelihood_t(ec, sa_n, zo, sz, cen_t, nu=nu_tune, n_u=10)
        elif nu_scalar is not None and float(nu_scalar) < 199.0:
            ll = log_likelihood_t(ec, sa_n, zo, sz, cen_t, nu=float(nu_scalar), n_u=10)
        else:
            ll = log_likelihood_normal(ec, sa_n, zo, sz, cen_t)
        return -ll.sum()

    params = [u, v] if free_beta else [u]
    grads = torch.autograd.grad(_nll(u, v), params, create_graph=True)

    def _hvp(seeds: list[Any]) -> list[Any]:
        dot = sum((g * s).sum() for g, s in zip(grads, seeds))
        out = torch.autograd.grad(dot, params, retain_graph=True, allow_unused=True)
        return [
            torch.zeros_like(p) if o is None else o.detach() for p, o in zip(params, out)
        ]

    ones = torch.ones(k, dtype=dtype, device=device)
    zeros = torch.zeros(k, dtype=dtype, device=device)
    hess = np.zeros((2 * k, 2 * k), dtype=np.float64)
    if free_beta:
        col_u = _hvp([ones, zeros])
        col_v = _hvp([zeros, ones])
        h_uu = col_u[0].cpu().numpy().astype(np.float64)
        h_vu = col_u[1].cpu().numpy().astype(np.float64)
        h_vv = col_v[1].cpu().numpy().astype(np.float64)
        for i in range(k):
            hess[i, i] = h_uu[i]
            hess[k + i, k + i] = h_vv[i]
            hess[i, k + i] = hess[k + i, i] = h_vu[i]
        return fb.shell_errors_from_hessian(hess, sigma_a, beta)

    h_uu = _hvp([ones])[0].cpu().numpy().astype(np.float64)
    with np.errstate(all="ignore"):
        var_u = np.where(h_uu > 0.0, 1.0 / np.maximum(h_uu, 1e-300), np.nan)
        frac = np.clip((np.asarray(sigma_a) - fb.SIGMA_A_LO) / fb.SIGMA_A_SPAN, 1e-12, 1 - 1e-12)
        se_sa = fb.SIGMA_A_SPAN * frac * (1.0 - frac) * np.sqrt(var_u)
        # β = 1 - σ_A² is a deterministic function of σ_A here, so its error is |dβ/dσ_A|
        # times σ_A's and the two are perfectly anti-correlated by construction.
        se_beta = 2.0 * np.asarray(sigma_a, dtype=np.float64) * se_sa
    corr = np.where(np.isfinite(se_sa), -1.0, np.nan)
    return fb.ShellErrors(se_sa, se_beta, corr, bool(np.all(np.isfinite(se_sa))))


def _data_frac_spread_by_shell(
    sigma_w: np.ndarray,
    sigma_i: np.ndarray,
    epsilon: np.ndarray,
    s_sq: np.ndarray,
    n_shells: int = 8,
) -> dict[str, Any]:
    """Within-shell spread of ``data%`` = σ_Z⁻²/(1+σ_Z⁻²), σ_Z = σ_I/(ε Σ_W).

    Equal-count shells in ``s_sq``, so resolution is held fixed inside a shell and the
    only thing left to spread ``data%`` is direction (plus genuine σ_I variation). Under
    an isotropic Σ_W the anisotropy shows up here as a wide p90−p10; under a correct
    tensor it should collapse. Reported per shell and pooled.
    """
    from phridge.contrib.intensity_ll.wilson import data_frac_spread

    denom = np.maximum(epsilon * sigma_w, 1e-30)
    sz = np.asarray(sigma_i, dtype=np.float64) / denom
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / np.maximum(sz, 1e-30) ** 2
        frac = inv / (1.0 + inv)
    frac = np.where(np.isfinite(frac), frac, np.nan)

    order = np.argsort(np.asarray(s_sq, dtype=np.float64))
    ns = max(1, int(n_shells))
    bounds = np.linspace(0, order.size, ns + 1).astype(int)
    rows: list[dict[str, float]] = []
    for i in range(ns):
        sel = order[bounds[i] : bounds[i + 1]]
        if sel.size == 0:
            continue
        sp = data_frac_spread(frac[sel])
        rows.append(
            {
                "n": int(sel.size),
                "p10": sp.p10,
                "p50": sp.p50,
                "p90": sp.p90,
                "spread": sp.spread,
            }
        )
    pooled = float(np.nanmean([r["spread"] for r in rows])) if rows else float("nan")
    return {"shells": rows, "mean_spread": pooled}


def _wilson_bin_ids(wilson_bins: Any, s_sq: np.ndarray, wilson_mod: Any) -> np.ndarray:
    """Per-reflection bin index for the binned Σ_W, compact and zero-based.

    The caller's bins are used when they cover every reflection, so that the binned
    model with no tensor reproduces the caller's own per-bin-mean Σ exactly. Otherwise
    equal-count bins in s² are built here.
    """
    from phridge.sfcalc.ops import _np

    n = int(np.asarray(s_sq).size)
    if wilson_bins is not None:
        raw = _np(wilson_bins, np.float64).ravel()
        if raw.shape == (n,) and np.all(np.isfinite(raw)) and np.all(raw >= 0):
            _, compact = np.unique(raw.astype(np.int64), return_inverse=True)
            return compact.astype(np.int64)
    return wilson_mod.equal_count_bins(s_sq)


def _fit_binned_wilson(
    *,
    torch: Any,
    normalize: Any,
    log_likelihood_normal: Any,
    io: np.ndarray,
    sig: np.ndarray,
    eps: np.ndarray,
    centric: np.ndarray,
    bin_id: np.ndarray,
    s_cart: Optional[np.ndarray],
    report_mask: np.ndarray,
    max_iter: int,
    dtype: Any,
    device: Any,
    failures: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Σ_W = S_k exp(-0.5 sᵀBs), tr B = 0, with S_k profiled so every bin's mean Z is 1.

    For each trial B the bin scale is the mean of I/(εA) over the bin, so the observed
    mean intensity of every bin is held fixed and the likelihood only decides how that
    intensity is shared between directions. B = 0 is the plain per-bin mean.

    Fitted on every reflection: this stage sees only I_obs and σ_I, never F_calc, so
    using the free set here does not leak the model into it. Returns
    ``(sigma_w, bin_means, b_cart, nll_per_refl_on_report_mask)`` with the NLL being
    −log p(I) under the pure Wilson prior (σ_A → 0).
    """
    n = io.size
    valid = (sig > 0) & (eps > 0) & np.isfinite(io) & np.isfinite(sig)
    n_bins = int(bin_id.max()) + 1
    y_np = io / np.maximum(eps, 1e-12)
    # Floor from the B = 0 means, fixed for the whole fit so it cannot move with B.
    _, means0 = _wilson_binned_means_np(y_np, bin_id, valid, n_bins)
    pos = means0[means0 > 0]
    floor = 1e-3 * float(np.median(pos)) if pos.size else 1e-8

    io_t = torch.as_tensor(io, dtype=dtype, device=device)
    si_t = torch.as_tensor(np.where(valid, sig, 1.0), dtype=dtype, device=device)
    eps_t = torch.as_tensor(np.where(eps > 0, eps, 1.0), dtype=dtype, device=device)
    cen_t = torch.as_tensor(centric, device=device)
    y_t = torch.as_tensor(np.where(valid, y_np, 0.0), dtype=dtype, device=device)
    w_t = torch.as_tensor(valid.astype(np.float64), dtype=dtype, device=device)
    bin_t = torch.as_tensor(bin_id, dtype=torch.long, device=device)
    counts = torch.zeros(n_bins, dtype=dtype, device=device).index_add_(0, bin_t, w_t)
    counts = counts.clamp(min=1.0)
    ones = torch.ones(n, dtype=dtype, device=device)
    sa_w = torch.full((n,), 1e-4, dtype=dtype, device=device)
    fit_tensor = s_cart is not None
    s_t = torch.as_tensor(s_cart, dtype=dtype, device=device) if fit_tensor else None
    p = torch.zeros(6, dtype=dtype, device=device, requires_grad=fit_tensor)
    eye = torch.eye(3, dtype=dtype, device=device)

    def _b() -> Any:
        b = torch.stack(
            [
                torch.stack([p[0], p[3], p[4]]),
                torch.stack([p[3], p[1], p[5]]),
                torch.stack([p[4], p[5], p[2]]),
            ]
        )
        return b - torch.trace(b) / 3.0 * eye

    def _sigma_w() -> tuple[Any, Any]:
        if fit_tensor:
            a = torch.exp(-0.5 * torch.einsum("ni,ij,nj->n", s_t, _b(), s_t))
        else:
            a = ones
        sums = torch.zeros(n_bins, dtype=dtype, device=device).index_add_(0, bin_t, w_t * y_t / a)
        means = (sums / counts).clamp(min=floor)
        return (means[bin_t] * a).clamp(min=1e-12), means

    def _nll_i() -> Any:
        sw, _ = _sigma_w()
        ec, sa_n, zo, sz = normalize(ones, io_t, si_t, eps_t, sw, sa_w)
        ll = log_likelihood_normal(ec, sa_n, zo, sz, cen_t) - torch.log(eps_t * sw)
        return -ll

    n_valid = max(float(valid.sum()), 1.0)
    if fit_tensor:
        opt = torch.optim.LBFGS([p], max_iter=int(max_iter), line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = (_nll_i() * w_t).sum() / n_valid
            loss.backward()
            return loss

        try:
            opt.step(closure)
        except Exception as exc:
            failures.append(f"wilson (binned tensor): {type(exc).__name__}: {exc}")

    with torch.no_grad():
        sw, means = _sigma_w()
        per = _nll_i()
        rep = torch.as_tensor(report_mask & valid, device=device)
        nll = float(per[rep].mean().item()) if bool(rep.any()) else float("nan")
        b_np = _b().detach().cpu().numpy().astype(np.float64) if fit_tensor else np.zeros((3, 3))
        return (
            sw.detach().cpu().numpy().astype(np.float64),
            means.detach().cpu().numpy().astype(np.float64),
            b_np,
            nll,
        )


def _wilson_binned_means_np(
    y: np.ndarray, bin_id: np.ndarray, valid: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    sums = np.bincount(bin_id[valid], weights=y[valid], minlength=n_bins)
    counts = np.bincount(bin_id[valid], minlength=n_bins)
    return counts, sums / np.maximum(counts, 1)


def parse_nu_grid(raw: Any, default: str = "5:50:5") -> list[float]:
    """ν values for ``nu_mode=grid``. ``lo:hi:step`` or an explicit list."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        raw = default
    if isinstance(raw, (list, tuple)):
        out = [float(x) for x in raw]
        return [v for v in out if v == v and v > 0.0]
    text = str(raw).strip()
    if ":" in text:
        parts = [p.strip() for p in text.split(":")]
        lo = float(parts[0])
        hi = float(parts[1]) if len(parts) > 1 else lo
        step = float(parts[2]) if len(parts) > 2 else 5.0
        if step <= 0.0:
            step = 5.0
        n = int(math.floor((hi - lo) / step + 1e-9)) + 1
        return [float(lo + i * step) for i in range(max(n, 1)) if lo + i * step <= hi + 0.5 * step]
    out = [float(x) for x in text.replace(",", " ").split()]
    return [v for v in out if v == v and v > 0.0]


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
    nu_mode: Optional[Any] = "bins",
    nu_grid: Optional[Any] = None,
    fit_scale: Optional[Any] = False,
    sigma_a_mode: Optional[Any] = "bins",
    n_sigma_a_bins: Optional[Any] = None,
    tv_norm: Optional[Any] = 0.0,
    fit_sigma_wilson: Optional[Any] = True,
    wilson_model: Optional[Any] = "binned",
    wilson_max_iter: Optional[Any] = 60,
    wilson_bins: Optional[Any] = None,
    beta_mode: Optional[Any] = "free",
    sigma_a_shape: Optional[Any] = "free",
    smooth_sigma_a: Optional[Any] = 1.0,
    smooth_beta: Optional[Any] = 1.0,
    beta_consistency_prior: Optional[Any] = 0.0,
    sigma_a_tensor: Optional[Any] = True,
    sphericity: Optional[Any] = 1.0,
    f_atoms: Optional[Any] = None,
    f_bulk: Optional[Any] = None,
    local_sigma_a: Optional[Any] = None,
    spatial_sigma_a_v2: Optional[Any] = None,
) -> dict[str, Any]:
    """Worker implementation for held-out nuisance parameter fitting (theta).

    ``spatial_sigma_a_v2`` is accepted so the flag-on payload is a valid
    job; Phase 1 does not consume it (the existing σ_A path is unchanged).

    Two-stage fit on the tune set (no joint Σ₀–σ_A free scale):

      1. **Wilson Σ from intensities alone** (no atomic model): by default binned,
         ``Σ(h) = S_k exp(-0.5 s_cartᵀ B s_cart)`` with ``tr B = 0`` and ``S_k`` the
         mean of ``I/(εA)`` in resolution bin ``k`` (the caller's ``wilson_bins``, or
         equal-count bins), so every bin keeps its observed mean intensity and only
         ``B`` is fitted, on all reflections. ``wilson_model="anisotropic"`` is the
         single-curve tensor ``Σ₀ exp(-0.5 s_cartᵀ B s_cart)`` and ``"isotropic"`` the
         scalar ``Σ₀ exp(-0.5 B_W s²)`` (see :mod:`phridge.contrib.intensity_ll.wilson`).
         The curve forms are initialized by a moment Wilson plot, then refined by
         maximizing the intensity likelihood under a pure Wilson prior
         (``σ_A → 0``) times Gaussian measurement noise — so noisy / negative
         ``I_obs`` are handled properly. No ``F_calc`` enters this stage.

         The anisotropic fit starts from ``B = B_W·I``, which reproduces the
         isotropic Σ exactly for any cell, so it can only improve the likelihood.
         It requires the model-side ``k_anisotropic`` to be constrained to
         isotropic: the two are degenerate and only one may be free.

      2. **σ_A and β (and optional ν) with Σ frozen**: one independent logit per
         resolution shell for each, regularized by a second-difference
         smoothness penalty (``smooth_sigma_a``, ``smooth_beta``) instead of a
         monotonicity constraint. Student-t ``ν``: ``nu_mode=grid`` (default
         when the client asks for a scan) holds a scalar ``ν`` and refits
         σ_A/β at each node of ``nu_grid`` (default 5, 10, …, 50);
         ``grid_bins`` uses that same scan but picks ν independently per
         resolution shell from the per-shell NLL, then refits σ_A/β with
         those values held; ``bins`` puts one ν logit on the same shells as
         σ_A in one LBFGS; ``global`` is a 1-D scalar search after σ_A. A
         Read-style σ_A curve is still available via ``sigma_a_mode="read"``.

         In normalized units ``E[Z_o|E_C] = σ_A² E_C² + β`` exactly, so within a
         shell σ_A² is the slope and β the intercept. Fitting β frees the
         intercept: the classical ``β = 1 - σ_A²`` forces the line through
         (1, 1), which asserts the Wilson normalization is exact, and when it is
         not the constrained fit tilts the slope and launders the normalization
         error into σ_A. ``beta_mode="constrained"`` and
         ``sigma_a_shape="monotone"`` restore the old behaviour exactly.

         β free makes σ_A degenerate with an overall ``F_c`` scale (only
         ``σ_A²k²`` enters the slope), so ``fit_scale`` must be off; the op
         raises rather than reporting an arbitrary split.

         On the free+free path a pair of Laue-class tensors ``M_A``, ``M_β``
         modulates the shell scalars by direction (``ŝᵀ M ŝ``), with a
         sphericity prior. Constrained+monotone does not grow those parameters.

    The returned ``β`` array is in absolute units, ``Σ_W β_shell`` — the
    unexplained intensity variance, the companion to σ_A's correlation.
    """
    _ = spatial_sigma_a_v2
    import torch
    from scipy.optimize import minimize_scalar

    from phridge.contrib.intensity_ll import aniso_rice as _ar
    from phridge.contrib.intensity_ll import free_beta as _fb
    from phridge.contrib.intensity_ll import local_sigma_a as _lsa
    from phridge.contrib.intensity_ll import wilson as _wilson
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
    def _complex_split(value: Any, name: str) -> Optional[np.ndarray]:
        if value is None:
            return None
        data = value.data if hasattr(value, "data") else value
        arr = _np(data, np.complex128)
        if arr is None:
            return None
        arr = np.asarray(arr, dtype=np.complex128).reshape(-1)
        if arr.shape[0] != n:
            raise ValueError(f"{name} must have length {n}, got {arr.shape[0]}")
        return arr

    fa_np = _complex_split(f_atoms, "f_atoms")
    fb_np = _complex_split(f_bulk, "f_bulk")
    eps_np = _np(epsilon, np.float64) if epsilon is not None else np.ones(n, dtype=np.float64)
    cen_np = _np(centric, bool) if centric is not None else np.zeros(n, dtype=bool)

    try:
        uc_for_report: Any = (
            f_obs.meta.crystal.unit_cell if hasattr(f_obs, "meta") else f_obs.crystal.unit_cell
        )
    except Exception:
        uc_for_report = (1.0, 1.0, 1.0, 90.0, 90.0, 90.0)

    if s_sq is not None:
        s_sq_np = _np(s_sq, np.float64)
    else:
        s_sq_np = _compute_s_sq(uc_for_report, _np(f_obs.hkl))

    mode = str(sigma_a_mode or "bins").strip().lower()
    # "monotone" names a *shape* for the binned fit, not a third functional form, but it
    # reads naturally in the sigma_a_mode slot and older callers spell it there.
    if mode in ("monotone", "monotonic"):
        mode, sigma_a_shape = "bins", "monotone"
    if mode not in ("bins", "read", "bin", "shell", "shells"):
        mode = "bins"
    if mode in ("bin", "shell", "shells"):
        mode = "bins"
    n_mode = str(nu_mode or "bins").strip().lower()
    if n_mode in ("bin", "shell", "shells"):
        n_mode = "bins"
    if n_mode in ("scan", "grid-scan", "grid_scan"):
        n_mode = "grid"
    if n_mode in ("grid-bins", "grid_bins", "per-bin", "per_bin", "bins-grid", "bins_grid"):
        n_mode = "grid_bins"
    if n_mode not in ("bins", "global", "grid", "grid_bins"):
        n_mode = "bins"
    # Read-style σ_A has no shells → no per-shell ν. A grid scan is still a scalar.
    if mode != "bins" and n_mode == "bins":
        n_mode = "global"
    if mode != "bins" and n_mode == "grid_bins":
        n_mode = "grid"
    tv_lam_val = 0.0 if tv_norm is None else max(0.0, float(tv_norm))
    do_fit_sw = bool(fit_sigma_wilson)
    wilson_opts = _wilson.WilsonOptions(
        wilson_model=_wilson.normalize_wilson_model(wilson_model),
        max_iter=int(wilson_max_iter) if wilson_max_iter else 60,
    )
    want_aniso = wilson_opts.carries_anisotropy
    wilson_max_iter = wilson_opts.max_iter

    shell_opts = _fb.ShellFitOptions(
        beta_mode=_fb.normalize_beta_mode(beta_mode),
        sigma_a_shape=_fb.normalize_sigma_a_shape(sigma_a_shape),
        lambda_u=0.0 if smooth_sigma_a is None else max(0.0, float(smooth_sigma_a)),
        lambda_v=0.0 if smooth_beta is None else max(0.0, float(smooth_beta)),
        lambda_consistency=(
            0.0 if beta_consistency_prior is None else max(0.0, float(beta_consistency_prior))
        ),
    )
    # β free leaves σ_A and an overall F_c scale exactly degenerate: only σ_A²k² enters
    # the slope of Z_o on E_C². Refining both returns one of infinitely many splits, each
    # fitting equally well, so refuse rather than report an arbitrary one.
    if shell_opts.beta_free and bool(fit_scale):
        raise ValueError(
            "fit_scale=True with beta_mode='free' is not identifiable: σ_A and the overall "
            "F_c scale k enter the Rice first moment only as σ_A²k², so the likelihood "
            "cannot separate them and the reported split would be arbitrary. Either fix the "
            "scale during stage 2 (fit_scale=False, e.g. let bulk-solvent scaling own it) "
            "or use beta_mode='constrained', where β = 1 - σ_A² breaks the degeneracy."
        )
    spatial_opts = _lsa.resolve_options(local_sigma_a)
    if spatial_opts.enabled and shell_opts.beta_free and bool(fit_scale):
        raise ValueError(
            "spatial σ_A with fit_scale=True and beta_mode='free' is not identifiable: "
            "u, the overall F_c scale k and σ_A enter the slope together. Freeze the "
            "scale (fit_scale=False) or use beta_mode='constrained'."
        )
    spatial_fallback: Optional[str] = None
    do_spatial = bool(spatial_opts.enabled)
    if do_spatial:
        if fa_np is None or fb_np is None:
            spatial_fallback = "spatial σ_A needs f_atoms and f_bulk"
            do_spatial = False
        elif not (
            np.all(np.isfinite(fa_np.real))
            and np.all(np.isfinite(fa_np.imag))
            and np.all(np.isfinite(fb_np.real))
            and np.all(np.isfinite(fb_np.imag))
        ):
            spatial_fallback = "f_atoms / f_bulk are not finite"
            do_spatial = False
    # Read-style σ_A is a two-parameter curve with no shells, so there is nowhere to hang
    # an independent per-shell β. Fall back to the constrained form and say so.
    beta_fallback: Optional[str] = None
    if shell_opts.beta_free and mode != "bins":
        beta_fallback = f"sigma_a_mode={mode!r} has no resolution shells to fit β in"
        shell_opts = shell_opts.model_copy(update={"beta_mode": "constrained"})

    def _nu_as_numpy(nu_in: Any) -> Optional[np.ndarray]:
        """Host copy of ``nu``. The worker may have lifted a numpy payload to MPS."""
        if nu_in is None:
            return None
        arr = _np(nu_in, np.float64)
        if arr is None:
            return None
        return np.asarray(arr, dtype=np.float64).ravel()

    def _nu_init_scalar(nu_in: Any) -> Optional[float]:
        arr = _nu_as_numpy(nu_in)
        if arr is None or arr.size == 0:
            return None
        val = float(np.nanmean(arr))
        if not np.isfinite(val) or val >= 199.0:
            return None
        return val

    # ------------------------------------------------------------------
    # Stage 0: moment Wilson plot → (Σ₀, B_W) initialization
    # ------------------------------------------------------------------
    valid_tune = tune & (sig > 0) & (eps_np > 0)
    if not np.any(valid_tune):
        valid_tune = tune

    y_tune = io[valid_tune] / np.maximum(eps_np[valid_tune], 1e-12)
    x_tune = s_sq_np[valid_tune]

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

    sigma_0 = max(float(sigma_0), 1e-8)
    b_wilson = max(float(b_wilson), 0.0)
    sigma_0_init = float(sigma_0)
    b_wilson_init = float(b_wilson)

    # ------------------------------------------------------------------
    # Stage 1: intensity-only ML Wilson (no F_calc, σ_A → 0)
    # ------------------------------------------------------------------
    dev = _DEVICE["device"]
    is_mps = dev.startswith("mps")
    dtype = torch.float32 if is_mps else torch.float64

    io_t = torch.as_tensor(io[tune], dtype=dtype, device=dev)
    si_t = torch.as_tensor(sig[tune], dtype=dtype, device=dev)
    eps_t = torch.as_tensor(eps_np[tune], dtype=dtype, device=dev)
    cen_t = torch.as_tensor(cen_np[tune], device=dev)
    s_sq_t = torch.as_tensor(s_sq_np[tune], dtype=dtype, device=dev)
    s_sq_tune_np = s_sq_np[tune]
    # Dummy |F_c|; with σ_A ≈ 0 the Rice prior is Wilson and independent of E_C.
    fc_wilson = torch.ones_like(io_t)
    sa_wilson = torch.full_like(io_t, 1e-4)

    def _inv_softplus(x: float) -> float:
        x = max(float(x), 1e-8)
        if x > 20.0:
            return x
        return float(math.log(math.expm1(x)))

    # Cartesian reciprocal vectors: Wilson anisotropy and the Rice modulation tensors
    # both need them. If the cell is unreachable, fall back rather than guessing a
    # metric from raw integer indices, which would fit a tensor in the wrong basis.
    s_cart_np: Optional[np.ndarray] = None
    wilson_fallback: Optional[str] = None
    try:
        uc_aniso = (
            f_obs.meta.crystal.unit_cell if hasattr(f_obs, "meta") else f_obs.crystal.unit_cell
        )
        s_cart_np = _wilson.reciprocal_cartesian(uc_aniso, _np(f_obs.hkl))
        if s_cart_np.shape != (n, 3) or not np.all(np.isfinite(s_cart_np)):
            raise ValueError(f"reciprocal vectors have shape {s_cart_np.shape}")
    except Exception as exc:
        if want_aniso:
            wilson_fallback = f"no usable unit cell for the anisotropic fit ({exc})"
            want_aniso = False
        s_cart_np = None

    if want_aniso and s_cart_np is not None:
        # Six components need directions that span the sphere. A line, a plane or a
        # narrow cone leaves the tensor unconstrained in some direction, and an
        # unconstrained tensor reports a confident number that means nothing -- so fall
        # back rather than fit something unidentifiable. Judged on the reflections the
        # tensor is fitted on: all of them for the binned model, the tune set otherwise.
        ident = _wilson.tensor_identifiability(
            s_cart_np if wilson_opts.binned else s_cart_np[tune]
        )
        if ident < _wilson.MIN_TENSOR_IDENTIFIABILITY:
            wilson_fallback = (
                f"reflection directions cannot determine a tensor "
                f"(identifiability {ident:.2e} < {_wilson.MIN_TENSOR_IDENTIFIABILITY:g}); "
                + ("bins only, no tensor" if wilson_opts.binned else "fitted isotropic instead")
            )
            want_aniso = False

    do_binned = wilson_opts.binned
    do_aniso = bool(want_aniso and do_fit_sw and s_cart_np is not None and not do_binned)
    wilson_nll_final: Optional[float] = None
    lbfgs_failures: list[str] = []
    b_cart_final = np.eye(3, dtype=np.float64) * b_wilson_init
    wilson_bin_means: Optional[np.ndarray] = None
    wilson_bin_id: Optional[np.ndarray] = None

    if do_binned:
        wilson_bin_id = _wilson_bin_ids(wilson_bins, s_sq_np, _wilson)
        fit_tensor = bool(want_aniso and do_fit_sw and s_cart_np is not None)
        t_w0 = time.monotonic()
        sigma_w_full, wilson_bin_means, b_cart_final, wilson_nll_final = _fit_binned_wilson(
            torch=torch,
            normalize=normalize,
            log_likelihood_normal=log_likelihood_normal,
            io=io,
            sig=sig,
            eps=eps_np,
            centric=cen_np,
            bin_id=wilson_bin_id,
            s_cart=s_cart_np if fit_tensor else None,
            report_mask=tune,
            max_iter=int(wilson_max_iter),
            dtype=dtype,
            device=dev,
            failures=lbfgs_failures,
        )
        wilson_seconds = time.monotonic() - t_w0
        sigma_0 = float(np.median(wilson_bin_means))
        b_wilson = 0.0
    elif do_fit_sw:
        log_s0 = torch.tensor(math.log(sigma_0_init), dtype=dtype, device=dev, requires_grad=True)
        params_w = [log_s0]
        if do_aniso:
            # B = L Lᵀ in the Cartesian reciprocal frame, log-diagonal: PSD is automatic
            # and LBFGS stays unconstrained. Started at B = B_W·I, which reproduces the
            # isotropic Σ_W exactly for any cell, so the fit can only improve the NLL.
            ld_init, od_init = _wilson.isotropic_cholesky_params(b_wilson_init)
            log_diag = torch.tensor(ld_init, dtype=dtype, device=dev, requires_grad=True)
            off_diag = torch.tensor(od_init, dtype=dtype, device=dev, requires_grad=True)
            s_cart_t = torch.as_tensor(s_cart_np[tune], dtype=dtype, device=dev)
            params_w += [log_diag, off_diag]
        else:
            u_bw = torch.tensor(
                _inv_softplus(b_wilson_init), dtype=dtype, device=dev, requires_grad=True
            )
            params_w.append(u_bw)

        def _b_tensor() -> Any:
            low = torch.zeros(3, 3, dtype=dtype, device=dev)
            d = torch.exp(log_diag)
            low = low + torch.diag(d)
            low = low + torch.stack(
                [
                    torch.zeros(3, dtype=dtype, device=dev),
                    torch.stack([off_diag[0], torch.zeros((), dtype=dtype, device=dev),
                                 torch.zeros((), dtype=dtype, device=dev)]),
                    torch.stack([off_diag[1], off_diag[2],
                                 torch.zeros((), dtype=dtype, device=dev)]),
                ]
            )
            return low @ low.T

        def _sigma_w_now(s0_t: Any) -> Any:
            if do_aniso:
                quad = torch.einsum("ni,ij,nj->n", s_cart_t, _b_tensor(), s_cart_t)
            else:
                quad = torch.nn.functional.softplus(u_bw) * s_sq_t
            return (s0_t * torch.exp(-0.5 * quad)).clamp(min=1e-12)

        def _wilson_nll() -> Any:
            s0_t = torch.exp(log_s0).clamp(min=1e-12)  # Σ₀ > 0
            sw_now = _sigma_w_now(s0_t)
            Ec, sA_n, Zo, sZ = normalize(fc_wilson, io_t, si_t, eps_t, sw_now, sa_wilson)
            # mli returns log-density of Z = I/(εΣ); convert to log-density of I:
            #   p(I) dI = p(Z) dZ  ⇒  log p(I) = log p(Z) - log(εΣ)
            ll_z = log_likelihood_normal(Ec, sA_n, Zo, sZ, cen_t)
            ll_i = ll_z - torch.log((eps_t * sw_now).clamp(min=1e-12))
            return -ll_i.sum()

        opt_w = torch.optim.LBFGS(
            params_w, max_iter=int(wilson_max_iter), line_search_fn="strong_wolfe"
        )

        def _wilson_closure():
            opt_w.zero_grad()
            loss = _wilson_nll()
            loss.backward()
            return loss

        try:
            opt_w.step(_wilson_closure)
        except Exception as exc:
            # A failed line search used to vanish here and leave the moment start
            # in place, which the report then printed as if it were the optimum.
            lbfgs_failures.append(f"wilson: {type(exc).__name__}: {exc}")

        with torch.no_grad():
            sigma_0 = float(torch.exp(log_s0).clamp(min=1e-12).item())
            if do_aniso:
                b_cart_final = _b_tensor().detach().cpu().numpy().astype(np.float64)
                # The scalar B_W keeps a meaning for every isotropic consumer: the
                # gauge-fixed trace/3, which is what the eigen-report calls b_iso.
                b_wilson = float(np.trace(b_cart_final) / 3.0)
            else:
                b_wilson = float(torch.nn.functional.softplus(u_bw).item())
                b_cart_final = np.eye(3, dtype=np.float64) * b_wilson
            wilson_nll_final = float(_wilson_nll().item() / max(float(tune.sum()), 1.0))
    else:
        b_cart_final = np.eye(3, dtype=np.float64) * b_wilson

    if do_binned:
        pass  # sigma_w_full already set by the binned fit
    elif do_aniso:
        sigma_w_full = _wilson.sigma_w_from_b(
            s_cart=s_cart_np, sigma_0=sigma_0, b_cart=b_cart_final
        )
    else:
        sigma_w_full = sigma_0 * np.exp(-0.5 * b_wilson * s_sq_np)
    sw_t = torch.as_tensor(sigma_w_full[tune], dtype=dtype, device=dev)

    # ------------------------------------------------------------------
    # Stage 2: σ_A (+ optional ν, F_c scale) with Σ frozen
    # ------------------------------------------------------------------
    fc_t = torch.as_tensor(fc_raw[tune], dtype=dtype, device=dev)
    u_spatial = None
    fa_t: Optional[Any] = None
    fb_t: Optional[Any] = None
    if do_spatial and fa_np is not None and fb_np is not None:
        cdtype = torch.complex64 if str(dev).startswith("mps") else torch.complex128
        fa_t = torch.as_tensor(fa_np[tune], dtype=cdtype, device=dev)
        fb_t = torch.as_tensor(fb_np[tune], dtype=cdtype, device=dev)
        u_spatial = torch.zeros((), dtype=dtype, device=dev, requires_grad=True)
    bounds = [2.5, 200.0] if nu_bounds is None else [float(nu_bounds[0]), float(nu_bounds[1])]
    nu_held_full: Optional[np.ndarray] = None
    _nu_arr = _nu_as_numpy(nu)
    if _nu_arr is not None and _nu_arr.size == n and np.any(np.isfinite(_nu_arr)):
        nu_held_full = _nu_arr
    nu_init = _nu_init_scalar(nu)
    do_fit_nu = bool(fit_nu) and (nu_init is None or nu_init < 199.0)
    do_fit_scale = bool(fit_scale)
    us = torch.tensor(0.0, dtype=dtype, device=dev, requires_grad=True) if do_fit_scale else None
    do_free_beta = bool(shell_opts.beta_free and mode == "bins")
    do_monotone = bool(shell_opts.monotone)

    sa_bin_centers: Optional[np.ndarray] = None
    sa_bin_values: Optional[np.ndarray] = None
    n_sa = 1
    edges: Optional[np.ndarray] = None
    shell_idx_t: Optional[Any] = None
    if mode == "bins":
        n_tune = int(tune.sum())
        if n_sigma_a_bins is not None:
            n_sa = max(3, int(n_sigma_a_bins))
        else:
            n_sa = int(min(20, max(6, n_tune // 250)))
        s_lo = float(s_sq_tune_np.min())
        s_hi = float(s_sq_tune_np.max())
        if s_hi - s_lo < 1e-8:
            edges = np.array([s_lo - 1e-6, s_hi + 1e-6], dtype=np.float64)
            n_sa = 1
        else:
            edges = np.linspace(s_lo, s_hi, n_sa + 1)
        shell_idx_tune = np.clip(np.digitize(s_sq_tune_np, edges[1:-1], right=False), 0, n_sa - 1)
        shell_idx_t = torch.as_tensor(shell_idx_tune, dtype=torch.long, device=dev)
        if not do_monotone or do_free_beta:
            # Moment start: regress Z_o on E_C² per shell, slope → σ_A², intercept → β.
            # Crude, but it lands near the answer exactly when the normalization is off —
            # the case the constrained start handles worst and the joint fit would
            # otherwise have to walk out of with two strongly correlated parameters.
            denom_i = np.maximum(eps_np[tune] * sigma_w_full[tune], 1e-30)
            sa_init_np, beta_init_np = _fb.moment_init_shells(
                z_obs=io[tune] / denom_i,
                e_c_sq=(fc_raw[tune] ** 2) / denom_i,
                sigma_z=sig[tune] / denom_i,
                shell_idx=shell_idx_tune,
                n_shells=n_sa,
            )
        else:
            sa_init_np = np.full(n_sa, 0.7, dtype=np.float64)
            beta_init_np = np.full(n_sa, 0.5, dtype=np.float64)
        if do_monotone:
            # Legacy cumulative-drop σ_A: logits fall by softplus(δ) per shell, so the
            # profile can never rise. Kept verbatim so old runs reproduce bit-for-bit.
            u0 = torch.tensor(2.0, dtype=dtype, device=dev, requires_grad=True)
            if n_sa > 1:
                raw_deltas = torch.full(
                    (n_sa - 1,), 0.3, dtype=dtype, device=dev, requires_grad=True
                )
                sa_params = [u0, raw_deltas] + ([us] if do_fit_scale else [])
            else:
                raw_deltas = None
                sa_params = [u0] + ([us] if do_fit_scale else [])
            u_sa = None

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

        else:
            # One free logit per shell. σ_A is depressed at low resolution where the
            # solvent model is poor and can dip mid-range (ice rings, detector artifacts);
            # a monotone profile has to push that structure into neighbouring shells.
            # Smoothness (applied in the closure) is the honest replacement: it says the
            # profile has no kinks, not that it never rises.
            u0 = None
            raw_deltas = None
            u_sa = torch.as_tensor(
                _fb.logit_from_sigma_a(sa_init_np), dtype=dtype, device=dev
            ).clone().requires_grad_(True)
            sa_params = [u_sa] + ([us] if do_fit_scale else [])

            def _sa_bins_from_params():
                return _fb.SIGMA_A_LO + _fb.SIGMA_A_SPAN * torch.sigmoid(u_sa)

        if do_free_beta:
            v_beta = torch.as_tensor(
                _fb.logit_from_beta(beta_init_np), dtype=dtype, device=dev
            ).clone().requires_grad_(True)
            sa_params = list(sa_params) + [v_beta]

            def _beta_bins_from_params():
                return _fb.BETA_LO + _fb.BETA_SPAN * torch.sigmoid(v_beta)

        else:
            v_beta = None

            def _beta_bins_from_params():
                sa = _sa_bins_from_params()
                return 1.0 - sa**2

        def _sa_tune_from_params():
            return _sa_bins_from_params()[shell_idx_t]

        def _beta_tune_from_params():
            return _beta_bins_from_params()[shell_idx_t]

        sa_bin_centers = 0.5 * (edges[:-1] + edges[1:])
    else:
        uk = torch.tensor(1.5, dtype=dtype, device=dev, requires_grad=True)
        ub = torch.tensor(2.0, dtype=dtype, device=dev, requires_grad=True)
        sa_params = [uk, ub] + ([us] if do_fit_scale else [])
        raw_deltas = None
        u0 = None
        u_sa = None
        v_beta = None

        def _sa_bins_from_params():
            raise RuntimeError("bin σ_A accessor unavailable in read mode")

        def _beta_bins_from_params():
            raise RuntimeError("bin β accessor unavailable in read mode")

        def _sa_tune_from_params():
            k_t = 0.999 * torch.sigmoid(uk) + 1e-4
            b_t = torch.nn.functional.softplus(ub)
            return torch.clamp(torch.sqrt(k_t) * torch.exp(-0.25 * b_t * s_sq_t), 1e-4, 0.9999)

        def _beta_tune_from_params():
            return 1.0 - _sa_tune_from_params() ** 2

    # Laue-class M_A / M_β: only on the free+free binned path. Constrained+monotone
    # does not grow these parameters (bit-identical). Cubic / unidentifiable → skip.
    crystal = None
    try:
        crystal = f_obs.meta.crystal if hasattr(f_obs, "meta") else f_obs.crystal
    except Exception:
        crystal = None
    laue = _ar.laue_from_crystal(crystal) if crystal is not None else _ar.LAUE_TRICLINIC
    tensor_enabled = True if sigma_a_tensor is None else bool(sigma_a_tensor)
    lam_sph = 0.0 if sphericity is None else max(0.0, float(sphericity))
    ident_rice = (
        float(_wilson.tensor_identifiability(s_cart_np[tune]))
        if s_cart_np is not None
        else 0.0
    )
    do_rice_tensor, rice_tensor_fallback = _ar.modulation_wanted(
        enabled=tensor_enabled,
        beta_free=do_free_beta,
        sigma_a_free=not do_monotone,
        bins_mode=mode == "bins",
        laue=laue,
        s_cart=s_cart_np[tune] if s_cart_np is not None else None,
        identifiability=ident_rice,
    )
    u_m_a = None
    u_m_b = None
    s_hat_tune_t: Optional[Any] = None
    s_hat_np: Optional[np.ndarray] = None
    if do_rice_tensor and s_cart_np is not None and laue.n_free > 0:
        s_hat_np = _ar.unit_directions(s_cart_np)
        s_hat_tune_t = torch.as_tensor(s_hat_np[tune], dtype=dtype, device=dev)
        u_m_a = torch.zeros(laue.n_free, dtype=dtype, device=dev, requires_grad=True)
        u_m_b = torch.zeros(laue.n_free, dtype=dtype, device=dev, requires_grad=True)
        sa_params = list(sa_params) + [u_m_a, u_m_b]
        _sa_shell = _sa_tune_from_params
        _beta_shell = _beta_tune_from_params

        def _quad_from(u_m: Any) -> Any:
            a = _ar.a_from_params_torch(u_m, laue)
            m = torch.eye(3, dtype=dtype, device=dev) + a
            q = _ar.quadratic_form_torch(s_hat_tune_t, m)
            return torch.clamp(q, 1e-6, 1e6)

        def _sa_tune_from_params():
            return torch.clamp(_sa_shell() * torch.sqrt(_quad_from(u_m_a)), 1e-4, 0.9999)

        def _beta_tune_from_params():
            return torch.clamp(
                _beta_shell() * _quad_from(u_m_b),
                _fb.BETA_LO,
                _fb.BETA_LO + _fb.BETA_SPAN,
            )

    do_fit_nu_bins = bool(do_fit_nu and n_mode == "bins" and mode == "bins" and shell_idx_t is not None)
    do_fit_nu_grid_bins = bool(
        do_fit_nu and n_mode == "grid_bins" and mode == "bins" and shell_idx_t is not None
    )
    do_fit_nu_grid = bool(do_fit_nu and (n_mode == "grid" or do_fit_nu_grid_bins))
    do_fit_nu_global = bool(do_fit_nu and n_mode == "global")
    current_nu: Optional[float] = nu_init if nu_init is not None else (7.0 if do_fit_nu else None)
    held_nu_bins_t: Optional[Any] = None
    u_nu = None
    nu_lo, nu_hi = float(bounds[0]), float(bounds[1])
    if do_fit_nu_bins:
        nu0 = float(current_nu if current_nu is not None else 7.0)
        nu0 = min(max(nu0, nu_lo + 0.05), nu_hi - 0.05)
        frac = (nu0 - nu_lo) / max(nu_hi - nu_lo, 1e-6)
        # Start inside the sigmoid's responsive band. At frac→0/1 the slope
        # d(ν)/d(u) collapses, so an initial ν sitting on a bound cannot be
        # optimized away from it and is silently reported as the fitted value.
        frac = min(max(frac, 0.05), 0.95)
        logit0 = math.log(frac / (1.0 - frac))
        u_nu = torch.full((n_sa,), logit0, dtype=dtype, device=dev, requires_grad=True)

        def _nu_bins_from_params():
            return nu_lo + (nu_hi - nu_lo) * torch.sigmoid(u_nu)

        def _nu_tune_from_params():
            return _nu_bins_from_params()[shell_idx_t]

        sa_params = list(sa_params) + [u_nu]

    params = list(sa_params)
    if do_spatial and u_spatial is not None:
        params = list(params) + [u_spatial]

    def _fc_amp() -> Any:
        if do_spatial and u_spatial is not None and fa_t is not None and fb_t is not None:
            w_mol, w_sol = _lsa.channel_weights(u_spatial, spatial_opts.e_bar)
            sid = shell_idx_t if spatial_opts.rms_gauge == "shell" else None
            gauge = spatial_opts.rms_gauge if sid is not None else "global"
            f_eff, _nk = _lsa.mix_f_eff(fa_t, fb_t, w_mol, w_sol, sid, gauge)
            amp = f_eff.abs()
        else:
            amp = fc_t
        return amp * torch.exp(us) if do_fit_scale else amp

    def nll_per_refl(nu_val: Optional[float] = None):
        sa_t = _sa_tune_from_params()
        fc_scaled = _fc_amp()
        Ec, sA_n, Zo, sZ = normalize(fc_scaled, io_t, si_t, eps_t, sw_t, sa_t)
        if do_free_beta:
            # Exact: the integrands use (E_C, σ_A) only through the product σ_A·E_C and
            # through a = 1 - σ_A², so this substitution makes a = β while preserving the
            # product, giving a Rice prior with E[Z_o|E_C] = σ_A²E_C² + β. The quadrature,
            # target and gradients are untouched — see free_beta.rice_inputs. The
            # constrained path deliberately does not route through here, because the
            # round trip sqrt(1-(1-σ_A²)) is not bit-identical to σ_A.
            Ec, sA_n = _fb.rice_inputs(Ec, sa_t, _beta_tune_from_params())
        if do_fit_nu_bins:
            ll = log_likelihood_t(Ec, sA_n, Zo, sZ, cen_t, nu=_nu_tune_from_params(), n_u=10)
        elif held_nu_bins_t is not None and shell_idx_t is not None:
            nu_t = held_nu_bins_t[shell_idx_t]
            gauss = nu_t >= 199.0
            ll = torch.empty_like(Ec)
            if bool((~gauss).any()):
                ll[~gauss] = log_likelihood_t(
                    Ec[~gauss], sA_n[~gauss], Zo[~gauss], sZ[~gauss], cen_t[~gauss],
                    nu=nu_t[~gauss], n_u=10,
                )
            if bool(gauss.any()):
                ll[gauss] = log_likelihood_normal(
                    Ec[gauss], sA_n[gauss], Zo[gauss], sZ[gauss], cen_t[gauss]
                )
        elif nu_val is not None and float(nu_val) < 199.0:
            ll = log_likelihood_t(Ec, sA_n, Zo, sZ, cen_t, nu=float(nu_val), n_u=10)
        else:
            ll = log_likelihood_normal(Ec, sA_n, Zo, sZ, cen_t)
        return -ll

    def compute_nll(nu_val: Optional[float] = None):
        return nll_per_refl(nu_val).sum()

    def _grid_node_snapshot(nu_val: Optional[float]) -> dict[str, Any]:
        """Per-shell NLL and the σ_A/β that produced it, after the node’s LBFGS."""
        with torch.no_grad():
            nll_i = nll_per_refl(nu_val)
            nll_mean = float(nll_i.mean().item())
            sa: list[float] = []
            beta: list[float] = []
            nll_shell: list[float] = []
            n_shell: list[int] = []
            if mode == "bins":
                sa_np = _sa_bins_from_params().detach().cpu().numpy().astype(np.float64)
                beta_np = _beta_bins_from_params().detach().cpu().numpy().astype(np.float64)
                sa = [float(x) for x in sa_np]
                beta = [float(x) for x in beta_np]
                if shell_idx_t is not None and sa_np.size:
                    sid = shell_idx_t.detach().cpu().numpy()
                    nll_np = nll_i.detach().cpu().numpy()
                    for k in range(int(sa_np.size)):
                        m = sid == k
                        n_k = int(m.sum())
                        n_shell.append(n_k)
                        nll_shell.append(float(nll_np[m].mean()) if n_k else float("nan"))
            return {
                "nll": nll_mean,
                "sigma_a": sa,
                "beta": beta,
                "nll_shell": nll_shell,
                "n_shell": n_shell,
            }

    opt = torch.optim.LBFGS(params, max_iter=40, line_search_fn="strong_wolfe")

    def _penalty():
        """Regularization added to the NLL. Never included in reported standard errors."""
        pen = None
        if mode != "bins" or n_sa <= 1:
            return pen
        if tv_lam_val > 0.0:
            sa_bins = _sa_bins_from_params()
            diffs = sa_bins[1:] - sa_bins[:-1]
            pen = tv_lam_val * torch.sqrt(diffs**2 + 1e-6).sum()
            if do_fit_nu_bins:
                nu_bins = _nu_bins_from_params()
                nd = nu_bins[1:] - nu_bins[:-1]
                # Scale TV so ν jumps (~few units) are comparable to σ_A (~0.01–0.1)
                pen = pen + tv_lam_val * torch.sqrt((nd / 10.0) ** 2 + 1e-6).sum()
        # Smoothness on the logits, in place of monotonicity. On the logits rather than on
        # σ_A itself so the penalty is scale-free near the bounds, and second-difference so
        # a straight trend through the shells is free and only kinks are charged.
        if u_sa is not None and shell_opts.lambda_u > 0.0:
            term = shell_opts.lambda_u * _fb.second_difference_penalty(u_sa)
            pen = term if pen is None else pen + term
        if v_beta is not None and shell_opts.lambda_v > 0.0:
            term = shell_opts.lambda_v * _fb.second_difference_penalty(v_beta)
            pen = term if pen is None else pen + term
        if do_free_beta and shell_opts.lambda_consistency > 0.0:
            # Optional pull toward the constrained form, for data too weak to determine
            # both. Off by default: the point of a free β is to let it disagree.
            sa_bins = _sa_bins_from_params()
            gap = _beta_bins_from_params() - (1.0 - sa_bins**2)
            term = shell_opts.lambda_consistency * (gap**2).sum()
            pen = term if pen is None else pen + term
        if do_rice_tensor and lam_sph > 0.0 and u_m_a is not None and u_m_b is not None:
            term = lam_sph * (
                _ar.frobenius_sq(_ar.a_from_params_torch(u_m_a, laue))
                + _ar.frobenius_sq(_ar.a_from_params_torch(u_m_b, laue))
            )
            pen = term if pen is None else pen + term
        if do_spatial and u_spatial is not None and spatial_opts.lambda_u > 0.0:
            term = spatial_opts.lambda_u * (u_spatial ** 2)
            pen = term if pen is None else pen + term
        return pen

    def closure():
        opt.zero_grad()
        loss = compute_nll(current_nu)
        pen = _penalty()
        if pen is not None:
            loss = loss + pen
        loss.backward()
        return loss

    def _param_data() -> list[Any]:
        return [p.detach().clone() for p in params]

    def _load_param_data(saved: list[Any]) -> None:
        with torch.no_grad():
            for p, src in zip(params, saved):
                p.copy_(src)

    grid_nll_per_refl: list[float] = []
    grid_nu_values: list[float] = []
    grid_nll_gaussian: Optional[float] = None
    grid_snapshots: list[dict[str, Any]] = []
    grid_snapshot_gaussian: Optional[dict[str, Any]] = None
    n_tune = max(float(tune.sum()), 1.0)

    if do_fit_nu_grid:
        init_state = _param_data()
        grid_nu_values = parse_nu_grid(nu_grid)
        if not grid_nu_values:
            grid_nu_values = parse_nu_grid(None)
        best_state = init_state
        best_nll_sum = float("inf")
        best_nu_grid = float(grid_nu_values[0])
        for nu_k in grid_nu_values:
            _load_param_data(init_state)
            current_nu = float(nu_k)
            opt_k = torch.optim.LBFGS(params, max_iter=40, line_search_fn="strong_wolfe")
            try:
                opt_k.step(closure)
            except Exception as exc:
                lbfgs_failures.append(f"sigma_a(ν={nu_k:g}): {type(exc).__name__}: {exc}")
            snap = _grid_node_snapshot(current_nu)
            grid_snapshots.append(snap)
            grid_nll_per_refl.append(float(snap["nll"]))
            nll_k = float(snap["nll"]) * n_tune
            if nll_k < best_nll_sum:
                best_nll_sum = nll_k
                best_nu_grid = float(nu_k)
                best_state = _param_data()
        _load_param_data(init_state)
        current_nu = 200.0
        opt_g = torch.optim.LBFGS(params, max_iter=40, line_search_fn="strong_wolfe")
        try:
            opt_g.step(closure)
        except Exception as exc:
            lbfgs_failures.append(f"sigma_a(Gaussian): {type(exc).__name__}: {exc}")
        grid_snapshot_gaussian = _grid_node_snapshot(None)
        grid_nll_gaussian = float(grid_snapshot_gaussian["nll"])
        _load_param_data(best_state)
        current_nu = best_nu_grid
        if do_fit_nu_grid_bins and grid_snapshots:
            n_sh = 0
            for snap in grid_snapshots:
                n_sh = max(n_sh, len(snap.get("nll_shell") or []))
            g_shell = list((grid_snapshot_gaussian or {}).get("nll_shell") or [])
            picked: list[float] = []
            for k in range(n_sh):
                scores: list[tuple[float, float]] = []
                for nu_i, snap in zip(grid_nu_values, grid_snapshots):
                    row = snap.get("nll_shell") or []
                    if k < len(row) and np.isfinite(float(row[k])):
                        scores.append((float(row[k]), float(nu_i)))
                if k < len(g_shell) and np.isfinite(float(g_shell[k])):
                    scores.append((float(g_shell[k]), 200.0))
                if scores:
                    picked.append(min(scores, key=lambda t: t[0])[1])
                else:
                    picked.append(float(best_nu_grid))
            held_nu_bins_t = torch.as_tensor(picked, dtype=dtype, device=dev)
            _load_param_data(init_state)
            current_nu = None
            opt_mix = torch.optim.LBFGS(params, max_iter=40, line_search_fn="strong_wolfe")
            try:
                opt_mix.step(closure)
            except Exception as exc:
                lbfgs_failures.append(f"sigma_a(ν per shell): {type(exc).__name__}: {exc}")
            finite = [v for v in picked if v < 199.0]
            current_nu = float(np.mean(finite)) if finite else 200.0
    else:
        try:
            opt.step(closure)
        except Exception as exc:
            lbfgs_failures.append(f"sigma_a: {type(exc).__name__}: {exc}")

    best_nu: Optional[float] = current_nu
    nu_se: Optional[float] = None
    nu_bin_values: Optional[np.ndarray] = None
    nu_full = np.full(n, np.nan, dtype=np.float64)
    nu_params: dict[str, Any] = {"mode": "none"}

    if do_fit_nu_bins:
        with torch.no_grad():
            nu_bin_values = _nu_bins_from_params().detach().cpu().numpy().astype(np.float64)
            best_nu = float(np.mean(nu_bin_values))
            if n_sa == 1:
                nu_full[:] = float(nu_bin_values[0])
            else:
                assert edges is not None
                # Piecewise-constant ν (not interpolated): log_likelihood_t groups by
                # unique ν values, so continuous interp would explode runtime.
                shell_all = np.clip(np.digitize(s_sq_np, edges[1:-1], right=False), 0, n_sa - 1)
                nu_full = nu_bin_values[shell_all]
            nu_full = np.clip(nu_full, nu_lo, nu_hi)
            nu_full = np.where(np.isfinite(nu_full), nu_full, float(best_nu))
            nu_params = {
                "mode": "bins",
                "n_bins": int(n_sa),
                "tv_norm": tv_lam_val,
                "bounds": [nu_lo, nu_hi],
                "bin_centers_s2": sa_bin_centers.tolist() if sa_bin_centers is not None else [],
                "bin_nu": nu_bin_values.tolist(),
            }
        # Optional SE on mean ν (fixed σ_A)
        try:
            with torch.no_grad():
                sa_t = _sa_tune_from_params()
                fc_scaled = _fc_amp()
                Ec, sA_n, Zo, sZ = normalize(fc_scaled, io_t, si_t, eps_t, sw_t, sa_t)
                post = posterior_moments(Ec, sA_n, Zo, sZ, cen_t, nu=best_nu, n_u=10)
                if post.d_loglik_d_nu is not None:
                    fisher_info = float((post.d_loglik_d_nu ** 2).sum().item())
                    nu_se = float(1.0 / np.sqrt(max(fisher_info, 1e-6)))
        except Exception:
            nu_se = None
    elif do_fit_nu_grid:
        if do_fit_nu_grid_bins and held_nu_bins_t is not None:
            nu_bin_values = held_nu_bins_t.detach().cpu().numpy().astype(np.float64)
            finite = nu_bin_values[np.isfinite(nu_bin_values) & (nu_bin_values < 199.0)]
            best_nu = float(np.mean(finite)) if finite.size else 200.0
            if n_sa == 1:
                nu_full[:] = float(nu_bin_values[0])
            else:
                assert edges is not None
                shell_all = np.clip(np.digitize(s_sq_np, edges[1:-1], right=False), 0, n_sa - 1)
                nu_full = nu_bin_values[shell_all]
            nu_full = np.where(np.isfinite(nu_full), nu_full, float(best_nu))
            nu_params = {
                "mode": "grid_bins",
                "nu": best_nu,
                "n_bins": int(n_sa),
                "bin_nu": [float(x) for x in nu_bin_values],
                "grid": [float(v) for v in grid_nu_values],
                "grid_nll": [float(v) for v in grid_nll_per_refl],
                "grid_nll_gaussian": grid_nll_gaussian,
                "grid_sigma_a": [list(s.get("sigma_a") or []) for s in grid_snapshots],
                "grid_beta": [list(s.get("beta") or []) for s in grid_snapshots],
                "grid_nll_shell": [list(s.get("nll_shell") or []) for s in grid_snapshots],
                "grid_n_shell": list((grid_snapshots[0].get("n_shell") or []) if grid_snapshots else []),
                "gaussian_sigma_a": list((grid_snapshot_gaussian or {}).get("sigma_a") or []),
                "gaussian_beta": list((grid_snapshot_gaussian or {}).get("beta") or []),
                "gaussian_nll_shell": list((grid_snapshot_gaussian or {}).get("nll_shell") or []),
                "bin_centers_s2": sa_bin_centers.tolist() if sa_bin_centers is not None else [],
                "bounds": [nu_lo, nu_hi],
            }
        else:
            best_nu = float(current_nu) if current_nu is not None else float(grid_nu_values[0])
            nu_full[:] = best_nu
            nu_params = {
                "mode": "grid",
                "nu": best_nu,
                "grid": [float(v) for v in grid_nu_values],
                "grid_nll": [float(v) for v in grid_nll_per_refl],
                "grid_nll_gaussian": grid_nll_gaussian,
                "grid_sigma_a": [list(s.get("sigma_a") or []) for s in grid_snapshots],
                "grid_beta": [list(s.get("beta") or []) for s in grid_snapshots],
                "grid_nll_shell": [list(s.get("nll_shell") or []) for s in grid_snapshots],
                "grid_n_shell": list((grid_snapshots[0].get("n_shell") or []) if grid_snapshots else []),
                "gaussian_sigma_a": list((grid_snapshot_gaussian or {}).get("sigma_a") or []),
                "gaussian_beta": list((grid_snapshot_gaussian or {}).get("beta") or []),
                "gaussian_nll_shell": list((grid_snapshot_gaussian or {}).get("nll_shell") or []),
                "bin_centers_s2": sa_bin_centers.tolist() if sa_bin_centers is not None else [],
                "bounds": [nu_lo, nu_hi],
            }
    elif do_fit_nu_global:
        def nu_obj(nu_candidate: float) -> float:
            with torch.no_grad():
                return float(compute_nll(float(nu_candidate)).item())

        res_nu = minimize_scalar(nu_obj, bounds=(bounds[0], bounds[1]), method="bounded")
        best_nu = float(res_nu.x)
        current_nu = best_nu

        try:
            opt.step(closure)
        except Exception as exc:
            lbfgs_failures.append(f"sigma_a (after nu): {type(exc).__name__}: {exc}")

        with torch.no_grad():
            sa_t = _sa_tune_from_params()
            fc_scaled = _fc_amp()
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
        nu_full[:] = float(best_nu)
        nu_params = {"mode": "global", "nu": float(best_nu), "bounds": [nu_lo, nu_hi]}
    elif nu_held_full is not None:
        t_vals = nu_held_full[np.isfinite(nu_held_full) & (nu_held_full < 199.0)]
        best_nu = float(np.mean(t_vals)) if t_vals.size else float(np.nanmean(nu_held_full))
        nu_se = None
        nu_full = np.where(np.isfinite(nu_held_full), nu_held_full, best_nu)
        nu_params = {"mode": "fixed", "nu": best_nu, "held_per_refl": True}
    elif nu_init is not None:
        best_nu = float(nu_init)
        nu_se = None
        nu_full[:] = best_nu
        nu_params = {"mode": "fixed", "nu": best_nu}
    else:
        best_nu = None
        nu_se = None
        nu_params = {"mode": "none"}

    # ν identifiability profile: per-reflection NLL(ν) with σ_A frozen, plus the
    # ν→∞ (Gaussian) limit. A profile that descends monotonically to the Gaussian
    # value means ν is not identified by the data, so the reported ν is set by the
    # bounds rather than by the likelihood.
    if do_fit_nu:
        if do_fit_nu_grid and grid_nu_values and grid_nll_per_refl:
            # Joint (σ_A, β) refit at each node — the number that chose ν.
            nu_params["profile_nu"] = [round(float(g), 3) for g in grid_nu_values]
            nu_params["profile_nll"] = [float(v) for v in grid_nll_per_refl]
            if grid_nll_gaussian is not None:
                nu_params["profile_nll_gaussian"] = float(grid_nll_gaussian)
        else:
            try:
                with torch.no_grad():
                    sa_prof = _sa_tune_from_params()
                    fc_prof = _fc_amp()
                    Ec_p, sA_p, Zo_p, sZ_p = normalize(fc_prof, io_t, si_t, eps_t, sw_t, sa_prof)
                    n_p = max(int(Ec_p.numel()), 1)
                    grid = np.unique(np.geomspace(max(nu_lo, 2.05), nu_hi, 10))
                    nu_params["profile_nu"] = [round(float(g), 3) for g in grid]
                    nu_params["profile_nll"] = [
                        float(
                            -log_likelihood_t(
                                Ec_p, sA_p, Zo_p, sZ_p, cen_t, nu=float(g), n_u=10
                            ).sum().item()
                        )
                        / n_p
                        for g in grid
                    ]
                    nu_params["profile_nll_gaussian"] = (
                        float(-log_likelihood_normal(Ec_p, sA_p, Zo_p, sZ_p, cen_t).sum().item()) / n_p
                    )
            except Exception:
                pass

    scale_final = float(torch.exp(us).item()) if do_fit_scale else 1.0

    with torch.no_grad():
        if mode == "bins":
            sa_bins_t = _sa_bins_from_params()
            sa_bin_values = sa_bins_t.detach().cpu().numpy().astype(np.float64)
            if n_sa == 1:
                sigma_a_full = np.full(n, float(sa_bin_values[0]), dtype=np.float64)
            else:
                assert edges is not None
                shell_all = np.clip(np.digitize(s_sq_np, edges[1:-1], right=False), 0, n_sa - 1)
                sigma_a_full = sa_bin_values[shell_all]
            if n_sa >= 2 and sa_bin_centers is not None:
                sigma_a_full = np.interp(
                    s_sq_np,
                    sa_bin_centers,
                    sa_bin_values,
                    left=float(sa_bin_values[0]),
                    right=float(sa_bin_values[-1]),
                )
            sigma_a_full = np.clip(sigma_a_full, 1e-4, 0.9999)
            k_final = float(sa_bin_values[0] ** 2)
            if sa_bin_centers is not None and n_sa >= 2:
                ds = float(sa_bin_centers[-1] - sa_bin_centers[0])
                if ds > 1e-12 and sa_bin_values[0] > 1e-6:
                    ratio = max(float(sa_bin_values[-1] / sa_bin_values[0]), 1e-6)
                    b_final = float(max(-4.0 * np.log(ratio) / ds, 0.0))
                else:
                    b_final = 0.0
            else:
                b_final = 0.0
            beta_bin_values = (
                _beta_bins_from_params().detach().cpu().numpy().astype(np.float64)
            )
            sigma_a_params = {
                "mode": "bins",
                "n_bins": int(n_sa),
                "k": k_final,
                "b_delta": b_final,
                "tv_norm": tv_lam_val,
                "bin_centers_s2": sa_bin_centers.tolist() if sa_bin_centers is not None else [],
                "bin_sigma_a": sa_bin_values.tolist(),
            }
        else:
            k_final = float(0.999 * torch.sigmoid(uk).item() + 1e-4)
            b_final = float(torch.nn.functional.softplus(ub).item())
            sigma_a_full = np.clip(
                np.sqrt(k_final) * np.exp(-0.25 * b_final * s_sq_np), 1e-4, 0.9999
            )
            sigma_a_params = {"mode": "read", "k": k_final, "b_delta": b_final}
            beta_bin_values = None

        final_tune_nll = float(compute_nll(best_nu).item() / max(float(tune.sum()), 1.0))

    shell_errors = _fb.ShellErrors(
        np.full(max(n_sa, 1), np.nan),
        np.full(max(n_sa, 1), np.nan),
        np.full(max(n_sa, 1), np.nan),
        False,
    )
    if mode == "bins" and sa_bin_values is not None:
        try:
            shell_errors = _shell_standard_errors(
                torch=torch,
                fb=_fb,
                normalize=normalize,
                log_likelihood_normal=log_likelihood_normal,
                log_likelihood_t=log_likelihood_t,
                sigma_a=sa_bin_values,
                beta=(
                    beta_bin_values
                    if beta_bin_values is not None
                    else 1.0 - sa_bin_values**2
                ),
                free_beta=do_free_beta,
                shell_idx=shell_idx_t,
                fc_t=_fc_amp().detach(),
                io_t=io_t,
                si_t=si_t,
                eps_t=eps_t,
                sw_t=sw_t,
                cen_t=cen_t,
                nu_tune=(
                    _nu_tune_from_params().detach() if do_fit_nu_bins else None
                ),
                nu_scalar=best_nu,
                dtype=dtype,
                device=dev,
            )
        except Exception:
            pass  # an error bar must never cost a macro cycle

    # β in absolute units: the unexplained intensity variance. Constrained, this is the
    # cctbx-style Σ(1 - σ_A²); free, the fitted intercept carries the Wilson scale.
    if do_free_beta and beta_bin_values is not None:
        assert edges is not None and sa_bin_centers is not None
        if n_sa == 1:
            beta_norm_full = np.full(n, float(beta_bin_values[0]), dtype=np.float64)
        else:
            # Interpolated on the same bin centres as σ_A, so σ_A² + β stays a meaningful
            # per-reflection consistency check rather than mixing two griddings.
            beta_norm_full = np.interp(
                s_sq_np,
                sa_bin_centers,
                beta_bin_values,
                left=float(beta_bin_values[0]),
                right=float(beta_bin_values[-1]),
            )
        beta_norm_full = np.clip(beta_norm_full, _fb.BETA_MIN, _fb.BETA_MAX)
        beta_full = sigma_w_full * beta_norm_full
    else:
        beta_norm_full = 1.0 - np.clip(sigma_a_full, 0.0, 0.9999) ** 2
        beta_full = sigma_w_full * beta_norm_full

    # Directional modulation is applied to the interpolated shell curves so the
    # target and the maps see σ_A(h), β(h), not just σ_A_k, β_k.
    if do_rice_tensor and u_m_a is not None and u_m_b is not None and s_hat_np is not None:
        with torch.no_grad():
            a_np = _ar.a_from_params(_np(u_m_a), laue)
            b_np = _ar.a_from_params(_np(u_m_b), laue)
        q_a = np.clip(_ar.quadratic_form(s_hat_np, np.eye(3) + a_np), 1e-6, 1e6)
        q_b = np.clip(_ar.quadratic_form(s_hat_np, np.eye(3) + b_np), 1e-6, 1e6)
        sigma_a_full = np.clip(sigma_a_full * np.sqrt(q_a), 1e-4, 0.9999)
        beta_norm_full = np.clip(beta_norm_full * q_b, _fb.BETA_MIN, _fb.BETA_MAX)
        beta_full = sigma_w_full * beta_norm_full

    if mode == "bins" and sa_bin_values is not None and sa_bin_centers is not None:
        shell_table = _fb.describe_shells(
            centers_s_sq=sa_bin_centers,
            sigma_a=sa_bin_values,
            beta=(
                beta_bin_values if beta_bin_values is not None else 1.0 - sa_bin_values**2
            ),
            errors=shell_errors,
            options=shell_opts,
        )
        sigma_a_params.update(shell_table.as_json())
    else:
        # Read mode has no shells, so there is no table -- but the report still has to say
        # which β form was used, and this is the one mode where the fallback fires.
        sigma_a_params["beta_mode"] = shell_opts.beta_mode
        sigma_a_params["sigma_a_shape"] = shell_opts.sigma_a_shape
    if beta_fallback:
        sigma_a_params["beta_fallback"] = beta_fallback

    n_rice_params = 0
    sigma_a_params["sigma_a_tensor"] = bool(do_rice_tensor)
    sigma_a_params["lambda_sphericity"] = float(lam_sph)
    sigma_a_params["laue"] = laue.name
    if rice_tensor_fallback:
        sigma_a_params["sigma_a_tensor_fallback"] = rice_tensor_fallback
    if do_rice_tensor and u_m_a is not None and u_m_b is not None:
        n_rice_params = 2 * int(laue.n_free)
        with torch.no_grad():
            a_rep = _ar.a_from_params(_np(u_m_a), laue)
            b_rep = _ar.a_from_params(_np(u_m_b), laue)
        sigma_a_params["M_A"] = _ar.describe_modulation(
            a_rep, unit_cell=uc_for_report, laue=laue
        ).as_json()
        sigma_a_params["M_beta"] = _ar.describe_modulation(
            b_rep, unit_cell=uc_for_report, laue=laue
        ).as_json()
        sigma_a_params["tensor_identifiability"] = float(ident_rice)
        sigma_a_params["n_tensor_params"] = n_rice_params
    else:
        sigma_a_params["n_tensor_params"] = 0

    u_final: Optional[float] = None
    w_mol_f: Optional[float] = None
    w_sol_f: Optional[float] = None
    n_k_report: Optional[np.ndarray] = None
    n_env = 0
    if do_spatial and u_spatial is not None and fa_np is not None and fb_np is not None:
        u_final = float(u_spatial.detach().cpu().item())
        w_mol_f, w_sol_f = _lsa.channel_weights(u_final, spatial_opts.e_bar)
        if mode == "bins" and edges is not None and n_sa > 1:
            shell_all = np.clip(np.digitize(s_sq_np, edges[1:-1], right=False), 0, n_sa - 1)
        else:
            shell_all = np.zeros(n, dtype=np.int64)
        _fe, n_k_report = _lsa.mix_f_eff(
            fa_np, fb_np, w_mol_f, w_sol_f, shell_all, spatial_opts.rms_gauge
        )
        n_k_report = np.asarray(n_k_report, dtype=np.float64)
        try:
            hkl_np = _np(f_obs.hkl)
            _hk, _c, keep_env = _lsa.envelope_coefficients(
                hkl_np, fb_np, s_sq_np, spatial_opts.d_min, spatial_opts.blur_b
            )
            n_env = int(np.sum(keep_env))
        except Exception:
            n_env = 0
    spatial_json = _lsa.describe_local_sigma_a(
        enabled=bool(do_spatial),
        u=u_final,
        blur_b=spatial_opts.blur_b,
        d_min=spatial_opts.d_min,
        e_bar=spatial_opts.e_bar,
        w_mol=w_mol_f,
        w_sol=w_sol_f,
        n_shell_rms=n_k_report,
        n_envelope_hkl=n_env,
        lambda_u=spatial_opts.lambda_u,
        rms_gauge=spatial_opts.rms_gauge,
        fallback=spatial_fallback,
    )
    if mode == "bins" and edges is not None:
        spatial_json["bin_edges_s2"] = [float(x) for x in edges]
    sigma_a_params["local_sigma_a"] = spatial_json

    n_nu_params = 0
    if best_nu is not None:
        n_nu_params = int(n_sa) if nu_params.get("mode") in ("bins", "grid_bins") else 1
    # Two parameters per shell once β is free, and the report has to say so: p_theta is
    # what the held-out statistics are corrected by.
    n_shell_params = (int(n_sa) * (2 if do_free_beta else 1)) if mode == "bins" else 2
    p_theta = n_shell_params + n_nu_params + n_rice_params + (1 if do_fit_scale else 0)
    p_theta += 1 if do_spatial else 0
    # Wilson: Σ₀ + either the scalar B_W or the 6 tensor components; binned is one mean
    # per bin plus the 5 components of a traceless tensor.
    binned_tensor = bool(do_binned and np.any(b_cart_final != 0.0))
    if do_binned:
        assert wilson_bin_means is not None
        n_wilson_params = int(wilson_bin_means.size) + (5 if binned_tensor else 0)
    else:
        n_wilson_params = 7 if do_aniso else 2
    p_theta += n_wilson_params

    # Report the normalization in its inspectable eigen-form. Raw B components are
    # basis-dependent and not comparable between data sets; eigenvalues and directions
    # are. The isotropic fit goes through the same description so the header never
    # branches and delta_b_aniso is exactly 0 there.
    wilson_tensor = _wilson.describe_wilson_tensor(
        sigma_0=sigma_0,
        b_cart=b_cart_final,
        unit_cell=uc_for_report,
        model="binned" if do_binned else ("anisotropic" if do_aniso else "isotropic"),
        nll_per_refl=float("nan") if wilson_nll_final is None else wilson_nll_final,
        n_params=n_wilson_params,
    )
    if do_binned:
        method = "bin_means+tensor_ml" if binned_tensor else "bin_means"
    else:
        method = "intensity_ml" if do_fit_sw else "moment_plot"
    wilson_params: dict[str, Any] = {
        "sigma_0": float(sigma_0),
        "b_wilson": float(b_wilson),
        "sigma_0_init": sigma_0_init,
        "b_wilson_init": b_wilson_init,
        "fitted": bool(do_fit_sw),
        "method": method,
        "wilson_nll": wilson_nll_final,
    }
    wilson_params.update(wilson_tensor.as_json())
    wilson_params["wilson_model_requested"] = wilson_opts.wilson_model
    if do_binned:
        assert wilson_bin_means is not None and wilson_bin_id is not None
        wilson_params["n_bins"] = int(wilson_bin_means.size)
        wilson_params["bin_means"] = [float(x) for x in wilson_bin_means]
        wilson_params["seconds"] = float(wilson_seconds)
    if s_cart_np is not None:
        wilson_params["tensor_identifiability"] = float(
            _wilson.tensor_identifiability(s_cart_np if do_binned else s_cart_np[tune])
        )
    if wilson_fallback:
        wilson_params["wilson_fallback"] = wilson_fallback
    # data% spread per shell: under an isotropic Σ_W this contains the anisotropy, so a
    # collapse here is the direct evidence that the tensor did its job.
    wilson_params["data_frac_spread"] = _data_frac_spread_by_shell(
        sigma_w_full, sig, eps_np, s_sq_np
    )

    return {
        "sigma_a": sigma_a_full,
        "sigma_wilson": sigma_w_full,
        "beta": beta_full,
        # The same residual variance in normalized units, which is what the target and the
        # maps consume: they divide by Σ_W themselves. None when β is tied, so a consumer
        # that sees None keeps the classical 1 - σ_A² and nothing changes.
        "beta_residual": beta_norm_full if do_free_beta else None,
        "nu": best_nu,
        "nu_per_refl": nu_full if best_nu is not None else np.full(n, np.nan, dtype=np.float64),
        "nu_se": nu_se,
        "nu_params": nu_params,
        "scale_k": scale_final,
        "p_theta": int(p_theta),
        "tune_nll": final_tune_nll,
        "lbfgs_failures": lbfgs_failures,
        "sigma_a_params": sigma_a_params,
        "sigma_wilson_params": wilson_params,
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
    "nu": "array",
    "beta_residual": "array",
    "precondition": "json",
    "damping": "json",
    "scale_factor": "json",
    "f_bulk": "array",
    "k_model": "array",
    "residual_k1": "json",
    "spatial_sigma_a_v2": "SpatialSigmaAV2",
    "spatial_sigma_a_v2_result": "SpatialSigmaAV2Result",
    "handle": "json",
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
    nu: Optional[Any] = None,
    beta_residual: Optional[Any] = None,
    precondition: Optional[Any] = False,
    damping: Optional[Any] = 0.05,
    scale_factor: Optional[Any] = 1.0,
    f_bulk: Optional[Any] = None,
    k_model: Optional[Any] = None,
    residual_k1: Optional[Any] = False,
    spatial_sigma_a_v2: Optional[Any] = None,
    spatial_sigma_a_v2_result: Optional[Any] = None,
    handle: Optional[Any] = None,
) -> dict[str, Any]:
    """Whole chain for intensity likelihood: F_calc -> ml_i target -> dQ/dF -> dQ/d(scatterer params).

    ``F_model = k (F_calc + F_bulk)`` with ``k = scale_factor * k_model``, times the
    least-squares residual scale ``k1`` of ``sqrt|I|`` on ``|F_model|`` when
    ``residual_k1`` is set. That is the client's ``f_model_scaled_with_k1()``: ``k_model``
    is mmtbx's ``k_isotropic_exp * k_isotropic * k_anisotropic``. Without both, the
    target would see a different F_model from the one the nuisance fit and the maps saw.

    Optionally computes exact Gauss-Newton diagonal preconditioners for Cartesian
    sites, occupancy, isotropic U, and anisotropic U* in a single server pass.

    When a fitted ``spatial_sigma_a_v2_result`` is present, ``F_calc`` is the
    modified-model structure factor (eq. 25) times ``D_0(s)``. Occupancy
    gradients are converted back to the modelled occupancy. Absent the result,
    the path is unchanged (flag off / unfitted block).
    """
    import torch

    from phridge.contrib.intensity_ll.target import IntensityLogLikelihood
    from phridge.sfcalc.ops import (
        _DEVICE,
        _bound_engine,
        _engine,
        _miller_like,
        _np,
        _observations,
        _target_result,
        _to_like,
    )
    from phridge.sfcalc.packing import PackedSfCurvatures, PackedSfGradients
    from phridge.sfcalc.targets import build_target

    do_precond = bool(precondition)
    damp_val = float(damping) if damping is not None else 0.05
    k_scale = float(scale_factor) if scale_factor is not None else 1.0

    d0_v2 = None
    w_v2 = None
    from phridge.contrib.spatial_sigmaa_v2.apply import (
        d0_from_block,
        frozen_from_result,
        occupancy_grads_to_model,
        should_apply,
        xray_with_frozen_errors,
    )

    if should_apply(spatial_sigma_a_v2, spatial_sigma_a_v2_result):
        w_v2, _u = frozen_from_result(spatial_sigma_a_v2_result)
        xray = xray_with_frozen_errors(xray, spatial_sigma_a_v2_result)
        cell = tuple(float(x) for x in xray.meta.crystal.unit_cell)
        d0_v2 = d0_from_block(spatial_sigma_a_v2, cell, _np(f_obs.hkl))

    eng, _tmpl = _bound_engine(handle, xray)
    if eng is None:
        eng = _engine(xray, table, _np(f_obs.hkl), params)
    spec = dict(target)
    if spec.get("name") != "ml_i":
        spec["name"] = "ml_i"
    tgt = build_target(spec)
    obs = _observations(
        f_obs, weights, r_free, alpha, beta, epsilon, centric, nu=nu,
        beta_residual=beta_residual,
    )

    p = eng.tensors(requires_grad=True)
    fc = eng.f_calc(*p)
    d0_t = None
    if d0_v2 is not None:
        d0_t = _to_like(d0_v2, fc.real)
        fc = fc * d0_t
    # k as a host array (for the curvature scaling) and as a device tensor
    k_np = np.full(fc.shape[0], k_scale, dtype=np.float64)
    if k_model is not None:
        km = _np(k_model, np.float64).ravel()
        if km.shape != k_np.shape or not np.all(np.isfinite(km)):
            raise ValueError(
                f"k_model must be {k_np.shape[0]} finite values, got shape {km.shape}"
            )
        k_np = k_np * km
    # dtype on the host, then device — never fuse (MPS has no float64)
    k_t = _to_like(k_np, fc.real)
    if f_bulk is not None:
        fb_np = _np(f_bulk, np.complex128)
        fb_t = _to_like(fb_np, fc)
        f_model = k_t * (fc + fb_t)
    else:
        f_model = k_t * fc
    if bool(residual_k1):
        # The client's scale_k1: LS scale of sqrt|I| on |F_model|, held fixed for the
        # gradient exactly as the functor path holds it.
        with torch.no_grad():
            fo_t = _to_like(torch.sqrt(torch.abs(obs.data)), fc.real)
            fm_abs = f_model.detach().abs()
            den = float((fm_abs * fm_abs).sum().item())
            k1 = float((fo_t * fm_abs).sum().item()) / den if den > 0 else 1.0
        if np.isfinite(k1) and k1 > 0:
            k_np = k_np * k1
            k_t = k_t * k1
            f_model = f_model * k1

    obs = obs.to_like(f_model)
    ev = tgt.evaluate(f_model.detach(), obs, compute_curvature=do_precond)
    g = _to_like(ev.d_target_d_f_calc, fc)
    # Scale derivative w.r.t fc: dQ/dfc = k * dQ/df_model
    q = (k_t * fc * g.conj()).real.sum()
    grads = torch.autograd.grad(q, p, allow_unused=True)
    arrays = [
        (torch.zeros_like(t) if gr is None else gr).detach().cpu().numpy().astype(np.float64)
        for t, gr in zip(p, grads)
    ]
    if w_v2 is not None:
        arrays[1] = occupancy_grads_to_model(arrays[1], w_v2)

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
        # Scale radial/tangential curvatures by k^2
        curv_r = _np(ev.curv_radial) * (k_np**2)
        curv_t = _np(ev.curv_tangential) * (k_np**2)
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
    """Register ``ml_i_maps``, ``ml_i_nuisance_fit``, ``ml_i_omit_windows``, ``ml_i_bulk_solvent_fit``, ``ml_i_surrogate_fit``, and target_and_gradients."""
    register_op(OP_NAME, ml_i_maps, inputs=dict(_INPUTS), outputs=dict(_OUTPUTS))
    register_op(NUISANCE_FIT_OP_NAME, ml_i_nuisance_fit, inputs=dict(_NUISANCE_INPUTS), outputs=dict(_NUISANCE_OUTPUTS))
    register_op(
        TARGET_AND_GRADIENTS_OP_NAME,
        ml_i_target_and_gradients,
        inputs=dict(_TARGET_AND_GRADIENTS_INPUTS),
        outputs=dict(_TARGET_AND_GRADIENTS_OUTPUTS),
    )
    from phridge.contrib.intensity_ll.omit_windows import OMIT_OP_NAME
    from phridge.contrib.intensity_ll.omit_windows_op import (
        _OMIT_INPUTS,
        _OMIT_OUTPUTS,
        ml_i_omit_windows,
    )

    register_op(OMIT_OP_NAME, ml_i_omit_windows, inputs=dict(_OMIT_INPUTS), outputs=dict(_OMIT_OUTPUTS))

    from phridge.contrib.intensity_ll.bulk_solvent_op import (
        _BULK_SOLVENT_INPUTS,
        _BULK_SOLVENT_OUTPUTS,
        BULK_SOLVENT_OP_NAME,
        ml_i_bulk_solvent_fit,
    )

    register_op(
        BULK_SOLVENT_OP_NAME,
        ml_i_bulk_solvent_fit,
        inputs=dict(_BULK_SOLVENT_INPUTS),
        outputs=dict(_BULK_SOLVENT_OUTPUTS),
    )

    from phridge.contrib.intensity_ll.surrogate_op import (
        _SURROGATE_INPUTS,
        _SURROGATE_OUTPUTS,
        SURROGATE_FIT_OP_NAME,
        ml_i_surrogate_fit,
    )

    register_op(
        SURROGATE_FIT_OP_NAME,
        ml_i_surrogate_fit,
        inputs=dict(_SURROGATE_INPUTS),
        outputs=dict(_SURROGATE_OUTPUTS),
    )


register_ops()
