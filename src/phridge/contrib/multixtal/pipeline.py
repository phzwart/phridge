"""End-to-end torch/numpy core. Talks only to the six phase-1 interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from phridge.contrib.multixtal.anomalous import (
    AnomalousFit,
    BijvoetRegression,
    bijvoet_delta,
    correct_slopes_for_scale_leakage,
    fit_site_q,
    normalize_delta,
)
from phridge.contrib.multixtal.explain import DatasetExplanation, explain_datasets
from phridge.contrib.multixtal.factors import (
    FactorFit,
    apply_scale_leakage_to_y,
    fit_factors,
    scale_leakage,
)
from phridge.contrib.multixtal.influence import InfluenceResult, analyze_grouped_influence, analyze_influence
from phridge.contrib.multixtal.maps_out import MapCoefficients, map_coefficients
from phridge.contrib.multixtal.nuisance import DatasetNuisance, fit_dataset_nuisance
from phridge.contrib.multixtal.options import MultixtalOptions
from phridge.contrib.multixtal.residuals import mate_mean, normalized_residuals
from phridge.contrib.multixtal.selector import PopulationSelector
from phridge.contrib.multixtal.structure import f_calc_s_g, resolution_s2, s_hat
from phridge.sfcalc.engine.engine import EngineParams, ScatteringModel


@dataclass
class MultixtalResult:
    f_calc: np.ndarray
    s: np.ndarray
    g: np.ndarray
    i_bar: np.ndarray
    sig2: np.ndarray
    observed: np.ndarray
    strong: np.ndarray
    nuisances: list[DatasetNuisance]
    y: np.ndarray
    w_model: np.ndarray
    w_meas: np.ndarray
    factors: FactorFit
    explanation: DatasetExplanation
    anomalous: Optional[AnomalousFit]
    maps: Optional[MapCoefficients]
    influence: list[InfluenceResult]
    stats: dict[str, Any] = field(default_factory=dict)
    damage: Any = None
    systematics: Any = None
    dose_fit: Any = None
    anom_damage: Any = None
    diagnostics: Any = None
    baselines: Any = None
    dose_influence: Optional[InfluenceResult] = None
    damage_maps: Optional[np.ndarray] = None
    anom_corrected: Optional[np.ndarray] = None


def _stack_nuisance(items: list[DatasetNuisance], key: str) -> np.ndarray:
    return np.stack([np.asarray(getattr(item, key), dtype=np.float64) for item in items], axis=0)


def run_multixtal_core(
    model: ScatteringModel,
    hkl: np.ndarray,
    cells: np.ndarray,
    sites_frac: np.ndarray,
    i_plus: np.ndarray,
    i_minus: np.ndarray,
    sig_plus: np.ndarray,
    sig_minus: np.ndarray,
    mask_plus: np.ndarray,
    mask_minus: np.ndarray,
    f_mask: np.ndarray,
    epsilon: np.ndarray,
    centric: np.ndarray,
    crystal: Any,
    options: MultixtalOptions,
    engine_params: Optional[EngineParams] = None,
    g_site: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    obs: Optional[Any] = None,
) -> MultixtalResult:
    """Steps 1–8. ``model`` is the template; cells/sites are per dataset."""
    hkl = np.asarray(hkl, dtype=np.int64).reshape(-1, 3)
    cells = np.asarray(cells, dtype=np.float64).reshape(-1, 6)
    sites_frac = np.asarray(sites_frac, dtype=np.float64)
    n_data = int(cells.shape[0])
    n_h = int(hkl.shape[0])
    if sites_frac.ndim == 2:
        sites_frac = np.broadcast_to(sites_frac[None, :, :], (n_data, sites_frac.shape[0], 3)).copy()

    from phridge.contrib.multixtal.structure import model_with_cell

    f_calc = np.zeros((n_data, n_h), dtype=np.complex128)
    s_arr = np.zeros((n_data, n_h), dtype=np.complex128)
    g_arr = np.zeros((n_data, n_h), dtype=np.complex128)
    s2 = np.zeros((n_data, n_h), dtype=np.float64)
    sh = np.zeros((n_data, n_h, 3), dtype=np.float64)
    for d in range(n_data):
        md = model_with_cell(model, cells[d], sites_frac[d])
        f_full, s, g = f_calc_s_g(md, hkl, params=engine_params)
        f_calc[d] = f_full
        s_arr[d] = s
        g_arr[d] = g
        s2[d] = resolution_s2(cells[d], hkl)
        sh[d] = s_hat(cells[d], hkl)

    i_bar, sig2, observed = mate_mean(i_plus, i_minus, sig_plus, sig_minus, mask_plus, mask_minus, centric)
    sig_bar = np.sqrt(np.maximum(sig2, 1e-12))

    selector = PopulationSelector(
        strong_isig=options.strong_isig,
        strong_min_datasets=options.strong_min_datasets,
        e_calc_floor=options.e_calc_floor,
    )
    e_calc = np.sqrt(np.maximum(np.abs(f_calc) ** 2, 0.0))
    strong = selector.select(i_bar, sig_bar, observed, e_calc=e_calc)

    nuisances: list[DatasetNuisance] = []
    for d in range(n_data):
        nuisances.append(
            fit_dataset_nuisance(
                i_bar[d],
                sig2[d],
                observed[d],
                f_calc[d],
                f_mask[d],
                s2[d],
                sh[d],
                epsilon,
                centric,
                crystal,
                n_knots=options.n_spline_knots,
                max_iter=options.max_lbfgs_iter,
            )
        )

    def _residuals(nuis: list[DatasetNuisance]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        y = np.zeros((n_data, n_h), dtype=np.float64)
        wm = np.zeros_like(y)
        wme = np.zeros_like(y)
        e2 = np.zeros_like(y)
        sa = np.zeros_like(y)
        for d, item in enumerate(nuis):
            yd, wmd, wmed, e2d, sad = normalized_residuals(
                i_bar[d][None, :],
                sig2[d][None, :],
                observed[d][None, :],
                item.f_c_abs2[None, :],
                epsilon[None, :],
                centric,
                item.scale_a[None, :],
                item.alpha[None, :],
                item.beta[None, :],
                item.sigma_p[None, :],
            )
            y[d] = yd[0]
            wm[d] = wmd[0]
            wme[d] = wmed[0]
            e2[d] = e2d[0]
            sa[d] = sad[0]
        # Restrict fitting weights to the strong set.
        wm = np.where(strong[None, :], wm, 0.0)
        return y, wm, wme, e2, sa

    y, w_model, w_meas, e2_o, sa = _residuals(nuisances)
    factors = fit_factors(y, w_model, options.rank, options.rank_perms, ridge=options.ridge, rng=rng)
    beta_before = _stack_nuisance(nuisances, "beta")

    for _outer in range(max(0, options.n_outer - 1)):
        if factors.rank == 0:
            break
        mode = factors.scores @ factors.loadings
        # Convert y-units mode to a mean shift in I via the residual definition (approximate).
        mode_i = mode * sa * (nuisances[0].scale_a[None, :] * epsilon[None, :] * (
            _stack_nuisance(nuisances, "alpha") ** 2 * _stack_nuisance(nuisances, "sigma_p")
            + _stack_nuisance(nuisances, "beta")
        ))
        refit: list[DatasetNuisance] = []
        for d in range(n_data):
            refit.append(
                fit_dataset_nuisance(
                    i_bar[d],
                    sig2[d],
                    observed[d],
                    f_calc[d],
                    f_mask[d],
                    s2[d],
                    sh[d],
                    epsilon,
                    centric,
                    crystal,
                    n_knots=options.n_spline_knots,
                    max_iter=max(8, options.max_lbfgs_iter // 2),
                    mode_mean=mode_i[d],
                )
            )
        nuisances = refit
        y, w_model, w_meas, e2_o, sa = _residuals(nuisances)
        factors = fit_factors(y, w_model, factors.rank, options.rank_perms, ridge=options.ridge, rng=rng)

    beta_after = _stack_nuisance(nuisances, "beta")
    d_beta = float(np.nanmean(beta_after - beta_before))

    log_alpha = np.log(np.maximum(_stack_nuisance(nuisances, "alpha"), 1e-12))
    slopes, significant = scale_leakage(log_alpha, factors.scores, observed & strong[None, :])
    factors.scale_leakage_slopes = slopes
    factors.scale_leakage_significant = significant
    if factors.rank > 0 and np.any(significant):
        y = apply_scale_leakage_to_y(y, e2_o, sa, factors.scores, slopes)
        factors = fit_factors(y, w_model, factors.rank, options.rank_perms, ridge=options.ridge, rng=rng)
        factors.scale_leakage_slopes = slopes
        factors.scale_leakage_significant = significant

    explanation = explain_datasets(
        y, w_meas, factors.mu, factors.scores, factors.loadings, w_model, ridge=options.ridge, flag_z=options.flag_z
    )

    if np.any(explanation.flagged) and n_data - int(explanation.flagged.sum()) >= 2:
        keep = ~explanation.flagged
        y_k, w_k = y[keep], w_model[keep]
        refit_f = fit_factors(y_k, w_k, "auto", options.rank_perms, ridge=options.ridge, rng=rng)
        explanation_refit = explain_datasets(
            y_k, w_meas[keep], refit_f.mu, refit_f.scores, refit_f.loadings, w_k, ridge=options.ridge, flag_z=options.flag_z
        )
        stats_flag = {
            "refit_rank": refit_f.rank,
            "refit_chi2_loo": explanation_refit.chi2_loo.tolist(),
        }
    else:
        stats_flag = {"refit_rank": factors.rank}

    anom_fit: Optional[AnomalousFit] = None
    maps: Optional[MapCoefficients] = None
    anom_w: Optional[np.ndarray] = None
    if options.anomalous:
        d_i, var_i, both = bijvoet_delta(i_plus, i_minus, sig_plus, sig_minus, mask_plus, mask_minus, centric)
        alpha = _stack_nuisance(nuisances, "alpha")
        beta = _stack_nuisance(nuisances, "beta")
        scale = _stack_nuisance(nuisances, "scale_a")
        sp = _stack_nuisance(nuisances, "sigma_p")
        d_e2, var_e2 = normalize_delta(d_i, var_i, scale, alpha, beta, sp, epsilon[None, :])
        w_bij = np.where(both & strong[None, :], 1.0 / np.maximum(var_e2, 1e-12), 0.0)
        intercept, slopes_a = BijvoetRegression().regress(d_e2, factors.scores, w_bij, both & strong[None, :])
        slopes_a = correct_slopes_for_scale_leakage(intercept, slopes_a, factors.scale_leakage_slopes)
        q = np.zeros(n_data, dtype=np.float64)
        q_se = np.full(n_data, np.nan, dtype=np.float64)
        if g_site is not None:
            s_c = np.where(np.abs(s_arr) > 0, s_arr, 1.0)
            # Use mean S over datasets for the model signature.
            s_mean = np.nanmean(np.where(np.abs(s_arr) > 0, s_arr, np.nan), axis=0)
            s_mean = np.where(np.isfinite(s_mean) & (np.abs(s_mean) > 1e-12), s_mean, 1.0)
            signature = -4.0 * np.imag(np.asarray(g_site, dtype=np.complex128) / s_mean)
            i_bar_safe = np.maximum(i_bar, 1e-8)
            var_ratio = var_i / (i_bar_safe**2)
            q, q_se = fit_site_q(d_i, i_bar, both, signature, var_ratio, sa, strong)
        anom_fit = AnomalousFit(intercept=intercept, slopes=slopes_a, q=q, q_se=q_se)
        anom_w = w_bij
        e_c = np.sqrt(np.maximum(_stack_nuisance(nuisances, "f_c_abs2") / np.maximum(sp, 1e-12), 0.0))
        maps = map_coefficients(s_arr, intercept, slopes_a, factors.loadings, e_c, sa, centric, strong)

    influence = analyze_influence(
        y,
        w_model,
        w_meas,
        factors.mu,
        factors.scores,
        factors.loadings,
        explanation.chi2_loo,
        ridge=options.ridge,
        anom_intercept=None if anom_fit is None else anom_fit.intercept,
        anom_weights=anom_w,
    )

    # Strong-set size per resolution shell (from the first dataset's s²).
    ss0 = s2[0]
    n_shells = 8
    order = np.argsort(ss0, kind="stable")
    shells = np.empty(n_h, dtype=np.int64)
    shells[order] = (np.arange(n_h) * n_shells) // max(n_h, 1)
    strong_per_shell = [int(np.sum(strong & (shells == k))) for k in range(n_shells)]

    stats = {
        "n_datasets": n_data,
        "n_reflections": n_h,
        "n_strong": int(strong.sum()),
        "strong_per_shell": strong_per_shell,
        "rank": factors.rank,
        "singular_values": factors.singular_values.tolist(),
        "null_top": factors.null_top.tolist(),
        "delta_beta": d_beta,
        "scale_leakage_slopes": factors.scale_leakage_slopes.tolist(),
        "scale_leakage_significant": factors.scale_leakage_significant.tolist(),
        "flagged": explanation.flagged.tolist(),
        **stats_flag,
    }
    result = MultixtalResult(
        f_calc=f_calc,
        s=s_arr,
        g=g_arr,
        i_bar=i_bar,
        sig2=sig2,
        observed=observed,
        strong=strong,
        nuisances=nuisances,
        y=y,
        w_model=w_model,
        w_meas=w_meas,
        factors=factors,
        explanation=explanation,
        anomalous=anom_fit,
        maps=maps,
        influence=influence,
        stats=stats,
    )
    _apply_damage_layer(
        result,
        obs=obs,
        hkl=hkl,
        s2=s2,
        epsilon=epsilon,
        centric=centric,
        options=options,
        i_plus=i_plus,
        i_minus=i_minus,
        sig_plus=sig_plus,
        sig_minus=sig_minus,
        mask_plus=mask_plus,
        mask_minus=mask_minus,
        g_site=g_site,
        sites_frac=sites_frac,
        rng=rng or np.random.default_rng(0),
    )
    return result


def _obs_dose(obs: Any) -> np.ndarray:
    if obs is None:
        return np.zeros(0)
    if hasattr(obs, "dose"):
        return np.asarray(obs.dose, dtype=np.float64)
    if isinstance(obs, dict):
        return np.asarray(obs.get("dose", []), dtype=np.float64)
    return np.zeros(0)


def _apply_damage_layer(
    result: MultixtalResult,
    *,
    obs: Any,
    hkl: np.ndarray,
    s2: np.ndarray,
    epsilon: np.ndarray,
    centric: np.ndarray,
    options: MultixtalOptions,
    i_plus: np.ndarray,
    i_minus: np.ndarray,
    sig_plus: np.ndarray,
    sig_minus: np.ndarray,
    mask_plus: np.ndarray,
    mask_minus: np.ndarray,
    g_site: Optional[np.ndarray],
    sites_frac: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """Optional observation-level radiation-damage layer. No-ops when ``obs`` is
    absent or every dose is zero — the phase-1 fields stay bit-identical.
    """
    dose = _obs_dose(obs)
    if obs is None or dose.size == 0 or np.allclose(dose, 0.0):
        result.stats["damage_skipped"] = True
        result.stats["damage_skip_reason"] = "no_obs" if obs is None or dose.size == 0 else "D=0"
        return

    from phridge.contrib.multixtal.anom_damage import fit_anom_site_decay
    from phridge.contrib.multixtal.baselines import run_baselines
    from phridge.contrib.multixtal.damage import fit_damage, observation_residuals
    from phridge.contrib.multixtal.diagnostics import run_diagnostics
    from phridge.contrib.multixtal.dose_io import dose_bins, sensitivities
    from phridge.contrib.multixtal.dose_model import (
        build_dose_model,
        fit_exp_time_constants,
        fit_learned_profiles,
    )
    from phridge.contrib.multixtal.maps_out import damage_map_coefficients
    from phridge.contrib.multixtal.systematics import SmoothSystematics, fit_systematics

    def _arr(name: str, default: Any = None) -> np.ndarray:
        if hasattr(obs, name):
            val = getattr(obs, name)
        elif isinstance(obs, dict):
            val = obs.get(name, default)
        else:
            val = default
        if val is None:
            raise KeyError(name)
        return np.asarray(val)

    i_obs = _arr("i")
    sig_obs = _arr("sig")
    obs_h = _arr("h").astype(np.int64)
    crystal = _arr("crystal").astype(np.int64)
    batch = _arr("batch")
    phi = _arr("phi")
    sign = _arr("sign")
    s_inc = _arr("s_inc")
    s_dif = _arr("s_dif")
    try:
        wedge = _arr("wedge")
    except KeyError:
        wedge = None
    try:
        pass_id = _arr("pass_id")
    except KeyError:
        pass_id = None

    n_data = result.y.shape[0]
    n_h = result.y.shape[1]
    reports = list(getattr(obs, "reports", []) or [])
    withheld = np.zeros(n_data, dtype=bool)
    for rep in reports:
        if getattr(rep, "confound", False) and 0 <= int(rep.crystal) < n_data:
            withheld[int(rep.crystal)] = True
    if not reports:
        from phridge.contrib.multixtal.dose_io import report_crystal

        for d in range(n_data):
            sel = crystal == d
            if not np.any(sel):
                continue
            rep = report_crystal(
                d, dose[sel], phi[sel], batch[sel], obs_h[sel], n_h,
                n_bins=options.dose_bins, confound_corr=options.confound_corr,
            )
            reports.append(rep)
            withheld[d] = bool(rep.confound)

    expected = np.zeros(i_obs.size, dtype=np.float64)
    s_sq_obs = np.zeros(i_obs.size, dtype=np.float64)
    for d, item in enumerate(result.nuisances):
        sel = crystal == d
        if not np.any(sel):
            continue
        hi = np.clip(obs_h[sel], 0, n_h - 1)
        expected[sel] = item.scale_a[hi] * epsilon[hi] * (item.alpha[hi] ** 2 * item.f_c_abs2[hi] + item.beta[hi])
        s_sq_obs[sel] = s2[d, hi]

    sys_fits = []
    a_sys = np.ones(i_obs.size, dtype=np.float64)
    for d in range(n_data):
        sel = crystal == d
        if int(sel.sum()) < 8:
            sys_fits.append(None)
            continue
        fitted = fit_systematics(
            i_obs[sel],
            sig_obs[sel],
            s_sq_obs[sel],
            s_inc[sel],
            s_dif[sel],
            batch[sel],
            expected[sel],
            dose=dose[sel],
            wedge=None if wedge is None else wedge[sel],
            order=options.absorption_order,
        )
        sys_fits.append(SmoothSystematics(fitted))
        a_sys[sel] = fitted.scale(s_sq_obs[sel], s_inc[sel], s_dif[sel], batch[sel], wedge=None if wedge is None else wedge[sel], dose=dose[sel])

    y_obs, w_obs = observation_residuals(
        i_obs, sig_obs, obs_h, crystal, a_sys, result.nuisances, epsilon, centric
    )

    rho = sensitivities(n_data)
    kind = str(options.dose_model)
    learned = None
    d_c = None
    bins = dose_bins(dose, options.dose_bins)
    if kind == "exp":
        d_c = fit_exp_time_constants(y_obs, dose, w_obs, n_modes=options.n_exp_modes)
    elif kind == "learned":
        n_bins = int(options.dose_bins)
        bin_means = np.zeros((n_bins, n_h), dtype=np.float64)
        bin_w = np.zeros((n_bins, n_h), dtype=np.float64)
        centers = np.zeros(n_bins, dtype=np.float64)
        for b in range(n_bins):
            sel = bins == b
            centers[b] = float(np.nanmean(dose[sel])) if np.any(sel) else float(b)
            for h in range(n_h):
                sel_h = sel & (obs_h == h) & (w_obs > 0)
                if np.any(sel_h):
                    bin_means[b, h] = float(np.sum(w_obs[sel_h] * y_obs[sel_h]) / np.sum(w_obs[sel_h]))
                    bin_w[b, h] = float(np.sum(w_obs[sel_h]))
        rank_use = options.damage_rank
        if rank_use == "auto":
            from phridge.contrib.multixtal.dose_model import permute_dose_bins_within_reflection

            mat = bin_means * np.sqrt(np.maximum(bin_w, 0.0))
            sv = np.linalg.svd(np.nan_to_num(mat, nan=0.0), compute_uv=False)
            tops = []
            for _ in range(max(8, options.rank_perms)):
                bp = permute_dose_bins_within_reflection(bins, obs_h, phi, rng)
                bm = np.zeros_like(bin_means)
                for b in range(n_bins):
                    for h in range(n_h):
                        sel_h = (bp == b) & (obs_h == h) & (w_obs > 0)
                        if np.any(sel_h):
                            bm[b, h] = float(np.sum(w_obs[sel_h] * y_obs[sel_h]) / np.sum(w_obs[sel_h]))
                svp = np.linalg.svd(np.nan_to_num(bm * np.sqrt(np.maximum(bin_w, 0.0)), nan=0.0), compute_uv=False)
                tops.append(float(svp[0]) if svp.size else 0.0)
            thresh = float(np.quantile(tops, 0.95)) if tops else np.inf
            rank_use = int(np.sum(sv > thresh))
        rank_use = max(1, min(int(rank_use), n_bins))
        learned = fit_learned_profiles(bin_means, centers, bin_w, rank=rank_use, rng=rng)

    dose_fit = build_dose_model(kind, n_exp_modes=options.n_exp_modes, d_c=d_c, learned=learned)
    rho_obs = rho[np.clip(crystal, 0, n_data - 1)]
    profiles = np.asarray(dose_fit.profiles(dose, rho=rho_obs), dtype=np.float64)

    damage = fit_damage(
        y_obs,
        w_obs,
        obs_h,
        crystal,
        profiles,
        sign,
        result.factors.scores,
        result.factors.loadings,
        result.factors.mu,
        result.y,
        result.w_model,
        result.strong,
        withheld,
        result.nuisances,
        epsilon,
        rho,
        ridge=options.ridge,
    )

    # Outer loop: subtract damage from crystal means, refit phase-1 factors, refit D3.
    from phridge.contrib.multixtal.damage import crystal_mean_profiles

    y_work = np.array(result.y, copy=True)
    factors = result.factors
    for _ in range(max(0, options.n_outer - 1)):
        g_mean = crystal_mean_profiles(profiles, crystal, None, n_data)
        y_work = result.y - g_mean @ damage.loadings
        factors = fit_factors(y_work, result.w_model, factors.rank, options.rank_perms, ridge=options.ridge, rng=rng)
        damage = fit_damage(
            y_obs,
            w_obs,
            obs_h,
            crystal,
            profiles,
            sign,
            factors.scores,
            factors.loadings,
            factors.mu,
            y_work,
            result.w_model,
            result.strong,
            withheld,
            result.nuisances,
            epsilon,
            rho,
            ridge=options.ridge,
        )
    result.y = y_work
    result.factors = factors

    pred_with = np.sum(profiles * damage.loadings.T[np.clip(obs_h, 0, n_h - 1)], axis=1)
    pred_without = np.zeros_like(y_obs)
    allow_perm = any(getattr(r, "n_passes", 1) >= 2 for r in reports) or (pass_id is not None and np.unique(pass_id).size >= 2)
    diags = run_diagnostics(
        withheld=withheld,
        loadings=damage.loadings,
        hkl=hkl,
        s_sq=s2[0],
        y_obs=y_obs,
        w_obs=w_obs,
        obs_h=obs_h,
        bins=bins,
        phi=phi,
        pass_id=pass_id,
        pred_with=pred_with,
        pred_without=pred_without,
        smoothness_threshold=options.smoothness_threshold,
        n_perms=max(8, options.rank_perms),
        rng=rng,
        sites_frac=sites_frac[0] if sites_frac.ndim == 3 else sites_frac,
        allow_perm=allow_perm,
    )
    # A mode is physical only if diagnostics pass; otherwise systematic, not attributed.
    if diags.physical.size and not bool(np.any(diags.physical)):
        damage.stats["attribution"] = "systematic, not attributed"
    else:
        damage.stats["attribution"] = "physical"

    baselines = run_baselines(
        i_obs,
        dose,
        w_obs,
        obs_h,
        n_h,
        batch,
        y_obs - pred_with,
        crystal=crystal,
        n_crystal=n_data,
        frame_reject_k=options.frame_reject_k,
    )

    group = crystal.astype(np.int64) * (int(options.dose_bins) + 1) + bins
    dose_inf = analyze_grouped_influence(y_obs, w_obs, group)

    e_c = np.sqrt(np.maximum(_stack_nuisance(result.nuisances, "f_c_abs2") / np.maximum(_stack_nuisance(result.nuisances, "sigma_p"), 1e-12), 0.0))
    sa = _stack_nuisance(result.nuisances, "sigma_a")
    dmaps = damage_map_coefficients(result.s, damage.loadings, e_c, sa, centric, result.strong)

    anom_d = None
    anom_corr = None
    if options.anomalous:
        d_i, var_i, both = bijvoet_delta(i_plus, i_minus, sig_plus, sig_minus, mask_plus, mask_minus, centric)
        mean_dose = np.zeros_like(result.i_bar)
        for d in range(n_data):
            for h in range(n_h):
                sel = (crystal == d) & (obs_h == h)
                if np.any(sel):
                    mean_dose[d, h] = float(np.mean(dose[sel]))
        # Remove D3 dose-difference from the Bijvoet residual before site-q.
        g_mean_h = crystal_mean_profiles(profiles, crystal, obs_h, n_data, n_h=n_h)
        leak = np.zeros_like(d_i)
        for d in range(n_data):
            leak[d] = g_mean_h[d] @ damage.loadings
        d_i_corr = d_i.copy()
        if g_site is not None:
            s_mean = np.nanmean(np.where(np.abs(result.s) > 0, result.s, np.nan), axis=0)
            s_mean = np.where(np.isfinite(s_mean) & (np.abs(s_mean) > 1e-12), s_mean, 1.0)
            signature = -4.0 * np.imag(np.asarray(g_site, dtype=np.complex128) / s_mean)
            i_bar_safe = np.maximum(result.i_bar, 1e-8)
            var_ratio = var_i / (i_bar_safe**2)
            anom_d = fit_anom_site_decay(
                d_i_corr,
                result.i_bar,
                both,
                signature,
                mean_dose,
                var_ratio,
                sa,
                result.strong,
                n_bins=min(options.dose_bins, 6),
                n_crystal=n_data,
            )
        if result.maps is not None:
            from phridge.contrib.multixtal.maps_out import plugin_fom

            phase = result.maps.phase_s
            fom = plugin_fom(e_c, sa, centric)
            fom = np.where(result.strong, fom, 0.0)
            anom_corr = fom * damage.c_h * (1.0j * phase)

    result.damage = damage
    result.systematics = sys_fits
    result.dose_fit = dose_fit
    result.anom_damage = anom_d
    result.diagnostics = diags
    result.baselines = baselines
    result.dose_influence = dose_inf
    result.damage_maps = dmaps
    result.anom_corrected = anom_corr
    result.stats["damage_skipped"] = False
    result.stats["damage"] = {
        "rank": int(damage.loadings.shape[0]),
        "attribution": damage.stats.get("attribution"),
        "n_agree": damage.stats.get("n_agree"),
        "withheld": withheld.tolist(),
        "physical": diags.physical.tolist(),
        "smoothness": diags.smoothness.tolist(),
        "rho": rho.tolist(),
        "reports": [
            {
                "crystal": r.crystal,
                "dose_min": r.dose_min,
                "dose_max": r.dose_max,
                "n_passes": r.n_passes,
                "dose_angle_corr": r.dose_angle_corr,
                "confound": r.confound,
                "n_obs": r.n_obs,
            }
            for r in reports
        ],
    }
    if anom_d is not None:
        result.stats["anom_damage"] = {
            "d_s": anom_d.d_s,
            "half_dose": anom_d.half_dose,
            "q0": anom_d.q0.tolist(),
            "used_exp": anom_d.used_exp,
        }
