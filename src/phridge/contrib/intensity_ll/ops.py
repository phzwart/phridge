"""Worker op ``ml_i_maps``: map coefficients from the intensity likelihood posterior.

Inputs mirror ``target_eval`` (same ``f_obs`` / per-reflection arrays / target spec) plus a
``maps`` JSON dict validated as :class:`IntensityMapOptions`. Outputs are complex
``MillerArray`` coefficient sets on the ``f_obs`` hkl list and per-reflection weight arrays.
The impl imports torch lazily so a Phenix client can import this module.
"""

from __future__ import annotations

import math
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
    "fit_scale": "json",
    "sigma_a_mode": "json",
    "n_sigma_a_bins": "json",
    "tv_norm": "json",
    "fit_sigma_wilson": "json",
}
_NUISANCE_OUTPUTS = {
    "sigma_a": "array",
    "sigma_wilson": "array",
    "beta": "array",
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
    obs = _observations(f_obs, weights, r_free, alpha, beta, epsilon, centric, nu=nu)
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
        Ec, sA_n, Zo, sZ = normalize(fc_s, fo_s, sig_s, eps_s, sW_s, sA_s)
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
        r_val_dict["s_report_version"] = 7

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
    fit_scale: Optional[Any] = False,
    sigma_a_mode: Optional[Any] = "bins",
    n_sigma_a_bins: Optional[Any] = None,
    tv_norm: Optional[Any] = 0.0,
    fit_sigma_wilson: Optional[Any] = True,
) -> dict[str, Any]:
    """Worker implementation for held-out nuisance parameter fitting (theta).

    Two-stage fit on the tune set (no joint Σ₀–σ_A free scale):

      1. **Wilson Σ(s) from intensities alone** (no atomic model):
         ``Σ(s) = Σ₀ exp(-0.5 B_W s²)`` with ``Σ₀ > 0``.
         Initialized by a moment Wilson plot, then (by default) refined by
         maximizing the intensity likelihood under a pure Wilson prior
         (``σ_A → 0``) times Gaussian measurement noise — so noisy / negative
         ``I_obs`` are handled properly. No ``F_calc`` enters this stage.

      2. **σ_A (and optional ν, F_c scale) with Σ frozen**:
         monotone resolution bins (default) or Read-style curve; optional TV
         on adjacent σ_A bins. Student-t ``ν`` defaults to the **same**
         resolution shells with the same TV penalty (``nu_mode=bins``);
         ``nu_mode=global`` keeps a single scalar ``ν``.

    After stage 2 the cctbx-style residual scale
    ``β = Σ (1 - σ_A²)`` is reported alongside ``σ_A`` (the classic split of
    correlation vs unexplained intensity variance).
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
    n_mode = str(nu_mode or "bins").strip().lower()
    if n_mode in ("bin", "shell", "shells"):
        n_mode = "bins"
    if n_mode not in ("bins", "global"):
        n_mode = "bins"
    # Read-style σ_A has no shells → scalar ν only
    if mode != "bins":
        n_mode = "global"
    tv_lam_val = 0.0 if tv_norm is None else max(0.0, float(tv_norm))
    do_fit_sw = bool(fit_sigma_wilson)

    def _nu_init_scalar(nu_in: Any) -> Optional[float]:
        if nu_in is None:
            return None
        arr = np.asarray(nu_in, dtype=np.float64).ravel()
        if arr.size == 0:
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

    wilson_nll_final: Optional[float] = None
    if do_fit_sw:
        log_s0 = torch.tensor(math.log(sigma_0_init), dtype=dtype, device=dev, requires_grad=True)
        u_bw = torch.tensor(_inv_softplus(b_wilson_init), dtype=dtype, device=dev, requires_grad=True)

        def _wilson_nll() -> Any:
            s0_t = torch.exp(log_s0).clamp(min=1e-12)  # Σ₀ > 0
            bw_t = torch.nn.functional.softplus(u_bw)
            sw_now = (s0_t * torch.exp(-0.5 * bw_t * s_sq_t)).clamp(min=1e-12)
            Ec, sA_n, Zo, sZ = normalize(fc_wilson, io_t, si_t, eps_t, sw_now, sa_wilson)
            # mli returns log-density of Z = I/(εΣ); convert to log-density of I:
            #   p(I) dI = p(Z) dZ  ⇒  log p(I) = log p(Z) - log(εΣ)
            ll_z = log_likelihood_normal(Ec, sA_n, Zo, sZ, cen_t)
            ll_i = ll_z - torch.log((eps_t * sw_now).clamp(min=1e-12))
            return -ll_i.sum()

        opt_w = torch.optim.LBFGS([log_s0, u_bw], max_iter=60, line_search_fn="strong_wolfe")

        def _wilson_closure():
            opt_w.zero_grad()
            loss = _wilson_nll()
            loss.backward()
            return loss

        try:
            opt_w.step(_wilson_closure)
        except Exception:
            pass

        with torch.no_grad():
            sigma_0 = float(torch.exp(log_s0).clamp(min=1e-12).item())
            b_wilson = float(torch.nn.functional.softplus(u_bw).item())
            wilson_nll_final = float(_wilson_nll().item() / max(float(tune.sum()), 1.0))

    sigma_w_full = sigma_0 * np.exp(-0.5 * b_wilson * s_sq_np)
    sw_t = torch.as_tensor(sigma_w_full[tune], dtype=dtype, device=dev)

    # ------------------------------------------------------------------
    # Stage 2: σ_A (+ optional ν, F_c scale) with Σ frozen
    # ------------------------------------------------------------------
    fc_t = torch.as_tensor(fc_raw[tune], dtype=dtype, device=dev)
    bounds = [2.5, 200.0] if nu_bounds is None else [float(nu_bounds[0]), float(nu_bounds[1])]
    nu_init = _nu_init_scalar(nu)
    do_fit_nu = bool(fit_nu) and (nu_init is None or nu_init < 199.0)
    do_fit_scale = bool(fit_scale)
    us = torch.tensor(0.0, dtype=dtype, device=dev, requires_grad=True) if do_fit_scale else None

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
            return _sa_bins_from_params()[shell_idx_t]

        sa_bin_centers = 0.5 * (edges[:-1] + edges[1:])
    else:
        uk = torch.tensor(1.5, dtype=dtype, device=dev, requires_grad=True)
        ub = torch.tensor(2.0, dtype=dtype, device=dev, requires_grad=True)
        sa_params = [uk, ub] + ([us] if do_fit_scale else [])
        raw_deltas = None
        u0 = None

        def _sa_bins_from_params():
            raise RuntimeError("bin σ_A accessor unavailable in read mode")

        def _sa_tune_from_params():
            k_t = 0.999 * torch.sigmoid(uk) + 1e-4
            b_t = torch.nn.functional.softplus(ub)
            return torch.clamp(torch.sqrt(k_t) * torch.exp(-0.25 * b_t * s_sq_t), 1e-4, 0.9999)

    do_fit_nu_bins = bool(do_fit_nu and n_mode == "bins" and mode == "bins" and shell_idx_t is not None)
    do_fit_nu_global = bool(do_fit_nu and not do_fit_nu_bins)
    current_nu: Optional[float] = nu_init if nu_init is not None else (7.0 if do_fit_nu else None)
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

    def compute_nll(nu_val: Optional[float] = None):
        sa_t = _sa_tune_from_params()
        fc_scaled = fc_t * torch.exp(us) if do_fit_scale else fc_t
        Ec, sA_n, Zo, sZ = normalize(fc_scaled, io_t, si_t, eps_t, sw_t, sa_t)
        if do_fit_nu_bins:
            ll = log_likelihood_t(Ec, sA_n, Zo, sZ, cen_t, nu=_nu_tune_from_params(), n_u=10)
        elif nu_val is not None and float(nu_val) < 199.0:
            ll = log_likelihood_t(Ec, sA_n, Zo, sZ, cen_t, nu=float(nu_val), n_u=10)
        else:
            ll = log_likelihood_normal(Ec, sA_n, Zo, sZ, cen_t)
        return -ll.sum()

    opt = torch.optim.LBFGS(params, max_iter=40, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = compute_nll(current_nu)
        if tv_lam_val > 0.0 and mode == "bins" and n_sa > 1:
            sa_bins = _sa_bins_from_params()
            diffs = sa_bins[1:] - sa_bins[:-1]
            loss = loss + tv_lam_val * torch.sqrt(diffs**2 + 1e-6).sum()
            if do_fit_nu_bins:
                nu_bins = _nu_bins_from_params()
                nd = nu_bins[1:] - nu_bins[:-1]
                # Scale TV so ν jumps (~few units) are comparable to σ_A (~0.01–0.1)
                loss = loss + tv_lam_val * torch.sqrt((nd / 10.0) ** 2 + 1e-6).sum()
        loss.backward()
        return loss

    try:
        opt.step(closure)
    except Exception:
        pass

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
                fc_scaled = fc_t * torch.exp(us) if do_fit_scale else fc_t
                Ec, sA_n, Zo, sZ = normalize(fc_scaled, io_t, si_t, eps_t, sw_t, sa_t)
                post = posterior_moments(Ec, sA_n, Zo, sZ, cen_t, nu=best_nu, n_u=10)
                if post.d_loglik_d_nu is not None:
                    fisher_info = float((post.d_loglik_d_nu ** 2).sum().item())
                    nu_se = float(1.0 / np.sqrt(max(fisher_info, 1e-6)))
        except Exception:
            nu_se = None
    elif do_fit_nu_global:
        def nu_obj(nu_candidate: float) -> float:
            with torch.no_grad():
                return float(compute_nll(float(nu_candidate)).item())

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
        nu_full[:] = float(best_nu)
        nu_params = {"mode": "global", "nu": float(best_nu), "bounds": [nu_lo, nu_hi]}
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
        try:
            with torch.no_grad():
                sa_prof = _sa_tune_from_params()
                fc_prof = fc_t * torch.exp(us) if do_fit_scale else fc_t
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

        final_tune_nll = float(compute_nll(best_nu).item() / max(float(tune.sum()), 1.0))

    # cctbx-style residual: β = Σ (1 - σ_A²)  (unexplained intensity variance)
    beta_full = sigma_w_full * (1.0 - np.clip(sigma_a_full, 0.0, 0.9999) ** 2)

    n_nu_params = 0
    if best_nu is not None:
        n_nu_params = int(n_sa) if nu_params.get("mode") == "bins" else 1
    p_theta = (int(n_sa) if mode == "bins" else 2) + n_nu_params + (1 if do_fit_scale else 0)
    p_theta += 2  # Wilson Σ₀, B_W

    return {
        "sigma_a": sigma_a_full,
        "sigma_wilson": sigma_w_full,
        "beta": beta_full,
        "nu": best_nu,
        "nu_per_refl": nu_full if best_nu is not None else np.full(n, np.nan, dtype=np.float64),
        "nu_se": nu_se,
        "nu_params": nu_params,
        "scale_k": scale_final,
        "p_theta": int(p_theta),
        "tune_nll": final_tune_nll,
        "sigma_a_params": sigma_a_params,
        "sigma_wilson_params": {
            "sigma_0": float(sigma_0),
            "b_wilson": float(b_wilson),
            "sigma_0_init": sigma_0_init,
            "b_wilson_init": b_wilson_init,
            "fitted": bool(do_fit_sw),
            "method": "intensity_ml" if do_fit_sw else "moment_plot",
            "wilson_nll": wilson_nll_final,
        },
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
    nu: Optional[Any] = None,
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
    obs = _observations(f_obs, weights, r_free, alpha, beta, epsilon, centric, nu=nu)

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
    """Register ``ml_i_maps``, ``ml_i_nuisance_fit``, and target_and_gradients."""
    register_op(OP_NAME, ml_i_maps, inputs=dict(_INPUTS), outputs=dict(_OUTPUTS))
    register_op(NUISANCE_FIT_OP_NAME, ml_i_nuisance_fit, inputs=dict(_NUISANCE_INPUTS), outputs=dict(_NUISANCE_OUTPUTS))
    register_op(
        TARGET_AND_GRADIENTS_OP_NAME,
        ml_i_target_and_gradients,
        inputs=dict(_TARGET_AND_GRADIENTS_INPUTS),
        outputs=dict(_TARGET_AND_GRADIENTS_OUTPUTS),
    )


register_ops()
