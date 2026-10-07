"""Paired-refinement ladder and the private steps of ``compare``."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
from scipy import stats

from benchmarks.intensity_benchmark import (
    ComparisonReport,
    ReflectionSplit,
    _check_intensity_observation_space,
    _eval_per_reflection,
    fit_nuisance,
)

def _prepare_models(models, model_nu, fit_nu, i_obs):
    model_keys = list(models.keys())
    if len(model_keys) < 2:
        raise ValueError("compare() requires at least 2 models, e.g. {'A': fc_A, 'B': fc_B}")

    key_A = model_keys[0]
    key_B = model_keys[1]
    fc_A = models[key_A]
    fc_B = models[key_B]

    # Pre-extract observation geometry
    d_spacings = np.asarray(i_obs.d_spacings().data(), dtype=np.float64)
    s_sq = 1.0 / np.maximum(d_spacings**2, 1e-12)
    hkl_list = [tuple(h) for h in i_obs.indices()]
    io_data = np.asarray(i_obs.data(), dtype=np.float64)
    sig_data = np.asarray(i_obs.sigmas(), dtype=np.float64) if i_obs.sigmas() is not None else np.ones_like(io_data)

    n_refl = len(d_spacings)

    # Check for identical models
    fc_A_data = np.asarray(fc_A.data()) if hasattr(fc_A, "data") else np.asarray(fc_A)
    fc_B_data = np.asarray(fc_B.data()) if hasattr(fc_B, "data") else np.asarray(fc_B)
    are_identical = np.array_equal(fc_A_data, fc_B_data)

    # Nu specifications
    nu_A = model_nu.get(key_A, None) if model_nu else None
    nu_B = model_nu.get(key_B, None) if model_nu else None
    fit_nu_A = fit_nu if nu_A is None else False
    fit_nu_B = fit_nu if nu_B is None else False

    # Evaluation storage
    ell_A_test = np.zeros(n_refl, dtype=np.float64)
    ell_B_test = np.zeros(n_refl, dtype=np.float64)
    ell_A_under_B_test = np.zeros(n_refl, dtype=np.float64)
    ell_B_under_A_test = np.zeros(n_refl, dtype=np.float64)

    test_scored_mask = np.zeros(n_refl, dtype=bool)
    return {
        'key_A': key_A,
        'key_B': key_B,
        'fc_A': fc_A,
        'fc_B': fc_B,
        'd_spacings': d_spacings,
        's_sq': s_sq,
        'hkl_list': hkl_list,
        'io_data': io_data,
        'sig_data': sig_data,
        'n_refl': n_refl,
        'fc_A_data': fc_A_data,
        'fc_B_data': fc_B_data,
        'are_identical': are_identical,
        'nu_A': nu_A,
        'nu_B': nu_B,
        'fit_nu_A': fit_nu_A,
        'fit_nu_B': fit_nu_B,
        'ell_A_test': ell_A_test,
        'ell_B_test': ell_B_test,
        'ell_A_under_B_test': ell_A_under_B_test,
        'ell_B_under_A_test': ell_B_under_A_test,
        'test_scored_mask': test_scored_mask,
    }


def _score_folds(split, bridge, i_obs, fc_A, fc_B, key_A, key_B, nu_bounds, fit_scale, fit_nu_A, fit_nu_B, nu_A, nu_B, s_sq, n_refl, ell_A_test, ell_B_test, ell_A_under_B_test, ell_B_under_A_test, test_scored_mask):
    # Cross-fit or three-way folds
    if split.mode == "cross_fit":
        folds = [
            (split.fold1_tune, split.fold1_test),
            (split.fold2_tune, split.fold2_test),
        ]
    else:
        folds = [(split.tune, split.test)]

    thetas_A: list[NuisanceParams] = []
    thetas_B: list[NuisanceParams] = []

    for fold_idx, (tune_mask, test_mask) in enumerate(folds):
        theta_A_f = fit_nuisance(
            bridge,
            i_obs,
            fc_A,
            tune_mask,
            nu_bounds=nu_bounds,
            fit_scale=fit_scale,
            fit_nu=fit_nu_A,
            nu=nu_A,
            s_sq=s_sq,
            model_name=key_A,
        )
        theta_B_f = fit_nuisance(
            bridge,
            i_obs,
            fc_B,
            tune_mask,
            nu_bounds=nu_bounds,
            fit_scale=fit_scale,
            fit_nu=fit_nu_B,
            nu=nu_B,
            s_sq=s_sq,
            model_name=key_B,
        )

        thetas_A.append(theta_A_f)
        thetas_B.append(theta_B_f)

        # Evaluate test reflections
        ell_A_f = _eval_per_reflection(bridge, fc_A, i_obs, theta_A_f, s_sq=s_sq)
        ell_B_f = _eval_per_reflection(bridge, fc_B, i_obs, theta_B_f, s_sq=s_sq)
        ell_AuB_f = _eval_per_reflection(bridge, fc_A, i_obs, theta_B_f, s_sq=s_sq)
        ell_BuA_f = _eval_per_reflection(bridge, fc_B, i_obs, theta_A_f, s_sq=s_sq)

        t_idx = np.where(test_mask)[0]
        ell_A_test[t_idx] = ell_A_f[t_idx]
        ell_B_test[t_idx] = ell_B_f[t_idx]
        ell_A_under_B_test[t_idx] = ell_AuB_f[t_idx]
        ell_B_under_A_test[t_idx] = ell_BuA_f[t_idx]
        test_scored_mask[t_idx] = True
    return {
        'thetas_A': thetas_A,
        'thetas_B': thetas_B,
    }


def _aggregate_scores(are_identical, nu_A, nu_B, ell_A_test, ell_B_test, ell_A_under_B_test, ell_B_under_A_test, test_scored_mask):
    # Scored test reflections
    t_idx = np.where(test_scored_mask)[0]
    n_test = len(t_idx)

    # Identical models special case: exact zeroes
    if are_identical and (nu_A == nu_B):
        ell_B_test[t_idx] = ell_A_test[t_idx]
        ell_A_under_B_test[t_idx] = ell_A_test[t_idx]
        ell_B_under_A_test[t_idx] = ell_A_test[t_idx]

    l_A = ell_A_test[t_idx]
    l_B = ell_B_test[t_idx]
    l_AuB = ell_A_under_B_test[t_idx]
    l_BuA = ell_B_under_A_test[t_idx]

    # Differences per reflection d_h = ell_h(A) - ell_h(B)
    d_h = l_A - l_B
    if are_identical and (nu_A == nu_B):
        d_h = np.zeros_like(d_h)

    nll_A = float(np.mean(l_A))
    nll_B = float(np.mean(l_B))
    nll_AuB = float(np.mean(l_AuB))
    nll_BuA = float(np.mean(l_BuA))

    delta_gain = float(np.mean(d_h))
    delta_nll = float(nll_B - nll_A)
    total_delta_gain = float(np.sum(d_h))

    # Two-way decomposition (Equations 1 & 2 from Section 5)
    # Way 1 (anchored at theta_A):
    struct_term_A = float(nll_BuA - nll_A)
    error_term_B = float(nll_B - nll_BuA)
    # Way 2 (anchored at theta_B):
    struct_term_B = float(nll_B - nll_AuB)
    error_term_A = float(nll_AuB - nll_A)

    if are_identical and (nu_A == nu_B):
        struct_term_A = 0.0
        error_term_B = 0.0
        struct_term_B = 0.0
        error_term_A = 0.0

    # Structural robustness conclusion
    if abs(struct_term_A) < 1e-6 and abs(struct_term_B) < 1e-6:
        is_robust_structure = True
        structure_conclusion = "Zero coordinate difference (pure error-model contrast)"
    elif (struct_term_A * struct_term_B) > 0:
        is_robust_structure = True
        structure_conclusion = "Robust: coordinate improvement carries the same sign under both error models"
    else:
        is_robust_structure = False
        structure_conclusion = "Fragile: structural difference disagrees between error models (difference lives in sigma_A/nu)"
    return {
        't_idx': t_idx,
        'n_test': n_test,
        'l_A': l_A,
        'l_B': l_B,
        'l_AuB': l_AuB,
        'l_BuA': l_BuA,
        'd_h': d_h,
        'nll_A': nll_A,
        'nll_B': nll_B,
        'nll_AuB': nll_AuB,
        'nll_BuA': nll_BuA,
        'delta_gain': delta_gain,
        'delta_nll': delta_nll,
        'total_delta_gain': total_delta_gain,
        'struct_term_A': struct_term_A,
        'error_term_B': error_term_B,
        'struct_term_B': struct_term_B,
        'error_term_A': error_term_A,
        'is_robust_structure': is_robust_structure,
        'structure_conclusion': structure_conclusion,
    }


def _bootstrap_gain(are_identical, nu_A, nu_B, split, t_idx, d_h, n_boot, seed, n_test):
    # Uncertainty: Block bootstrap over resolution shells
    shell_assignments_test = split.shell_indices[t_idx]
    unique_shells = np.unique(shell_assignments_test)
    n_unique_shells = len(unique_shells)

    rng = np.random.default_rng(seed)
    if are_identical and (nu_A == nu_B):
        se_boot = 0.0
        ci_boot = (0.0, 0.0)
        se_naive = 0.0
    else:
        # Pre-group d_h by shell
        shell_d_h = [d_h[shell_assignments_test == sh] for sh in unique_shells]
        boot_deltas = np.zeros(n_boot, dtype=np.float64)
        for b in range(n_boot):
            sampled_shell_indices = rng.choice(n_unique_shells, size=n_unique_shells, replace=True)
            sampled_dh = np.concatenate([shell_d_h[si] for si in sampled_shell_indices])
            boot_deltas[b] = np.mean(sampled_dh)

        se_boot = float(np.std(boot_deltas))
        ci_boot = (float(np.percentile(boot_deltas, 2.5)), float(np.percentile(boot_deltas, 97.5)))
        se_naive = float(np.std(d_h) / math.sqrt(n_test)) if n_test > 1 else 0.0

    se_ratio = float(se_boot / max(se_naive, 1e-12)) if se_naive > 0 else 1.0
    return {
        'se_boot': se_boot,
        'ci_boot': ci_boot,
        'se_naive': se_naive,
        'se_ratio': se_ratio,
        'unique_shells': unique_shells,
        'shell_assignments_test': shell_assignments_test,
        'rng': rng,
    }


def _breadth_tests(are_identical, nu_A, nu_B, d_h, n_test):
    # Breadth: win fraction, McNemar sign test, Wilcoxon
    if are_identical and (nu_A == nu_B):
        win_fraction = 0.5
        wins_B = 0
        wins_A = 0
        ties = n_test
        mcnemar_p = 1.0
        wilcoxon_stat = 0.0
        wilcoxon_p = 1.0
    else:
        wins_B = int(np.sum(d_h > 1e-12))
        wins_A = int(np.sum(d_h < -1e-12))
        ties = int(n_test - wins_B - wins_A)
        win_fraction = float(wins_B / n_test) if n_test > 0 else 0.5

        # McNemar / exact binomial sign test on discordant reflections
        n_discordant = wins_B + wins_A
        if n_discordant > 0:
            binom_res = stats.binomtest(wins_B, n_discordant, p=0.5, alternative="two-sided")
            mcnemar_p = float(binom_res.pvalue)
        else:
            mcnemar_p = 1.0

        # Wilcoxon signed-rank test
        if n_discordant >= 10:
            try:
                w_res = stats.wilcoxon(d_h[d_h != 0], alternative="two-sided")
                wilcoxon_stat = float(w_res.statistic)
                wilcoxon_p = float(w_res.pvalue)
            except Exception:
                wilcoxon_stat, wilcoxon_p = 0.0, 1.0
        else:
            wilcoxon_stat, wilcoxon_p = 0.0, 1.0
    return {
        'win_fraction': win_fraction,
        'wins_B': wins_B,
        'wins_A': wins_A,
        'ties': ties,
        'mcnemar_p': mcnemar_p,
        'wilcoxon_stat': wilcoxon_stat,
        'wilcoxon_p': wilcoxon_p,
    }


def _discordant_rows(n_test, are_identical, nu_A, nu_B, d_h, fc_A_data, fc_B_data, t_idx, hkl_list, d_spacings, io_data, sig_data, l_A, l_B):
    top_discordant = []
    if n_test > 0 and not (are_identical and (nu_A == nu_B)):
        fc_A_abs = np.abs(fc_A_data)
        fc_B_abs = np.abs(fc_B_data)
        abs_dh = np.abs(d_h)
        top_indices = np.argsort(abs_dh)[::-1][:20]
        for tidx in top_indices:
            orig_idx = t_idx[tidx]
            h, k, l = hkl_list[orig_idx]
            top_discordant.append({
                "h": int(h),
                "k": int(k),
                "l": int(l),
                "d": float(d_spacings[orig_idx]),
                "i_obs": float(io_data[orig_idx]),
                "sig_i": float(sig_data[orig_idx]),
                "i_over_sig": float(io_data[orig_idx] / max(sig_data[orig_idx], 1e-12)),
                "fc_A": float(fc_A_abs[orig_idx]),
                "fc_B": float(fc_B_abs[orig_idx]),
                "nll_A": float(l_A[tidx]),
                "nll_B": float(l_B[tidx]),
                "d_h": float(d_h[tidx]),
            })
    return {
        'top_discordant': top_discordant,
    }


def _shell_rows(unique_shells, shell_assignments_test, t_idx, d_spacings, d_h, l_A, l_B, rng):
    shells = []
    for sh_idx in sorted(unique_shells):
        sh_mask = shell_assignments_test == sh_idx
        sh_t_idx = t_idx[sh_mask]
        n_sh = int(np.sum(sh_mask))
        if n_sh == 0:
            continue
        d_sh = d_spacings[sh_t_idx]
        dh_sh = d_h[sh_mask]
        w_B = int(np.sum(dh_sh > 1e-12))
        w_A = int(np.sum(dh_sh < -1e-12))
        sh_boot_se = float(np.std([np.mean(rng.choice(dh_sh, size=n_sh, replace=True)) for _ in range(500)])) if n_sh > 2 else 0.0
        shells.append({
            "shell_idx": int(sh_idx),
            "d_max": float(np.max(d_sh)),
            "d_min": float(np.min(d_sh)),
            "n_refl": n_sh,
            "nll_A": float(np.mean(l_A[sh_mask])),
            "nll_B": float(np.mean(l_B[sh_mask])),
            "delta_gain": float(np.mean(dh_sh)),
            "se_boot": sh_boot_se,
            "win_fraction": float(w_B / n_sh),
            "wins": w_B,
            "losses": w_A,
        })
    return {
        'shells': shells,
    }


def _sigma_grids(d_spacings, thetas_A, thetas_B):
    s_min = float(np.min(1.0 / np.maximum(d_spacings, 1e-6)))
    s_max = float(np.max(1.0 / np.maximum(d_spacings, 1e-6)))
    s_grid = np.linspace(s_min, s_max, 50)
    d_grid = 1.0 / s_grid
    s_sq_grid = s_grid**2

    # Average curve parameters across folds
    k_A_vals = [th.sigma_a_params["k"] for th in thetas_A]
    b_A_vals = [th.sigma_a_params["b_delta"] for th in thetas_A]
    k_B_vals = [th.sigma_a_params["k"] for th in thetas_B]
    b_B_vals = [th.sigma_a_params["b_delta"] for th in thetas_B]

    sigma_a_A_grid = np.clip(np.sqrt(np.mean(k_A_vals)) * np.exp(-0.25 * np.mean(b_A_vals) * s_sq_grid), 1e-4, 0.9999)
    sigma_a_B_grid = np.clip(np.sqrt(np.mean(k_B_vals)) * np.exp(-0.25 * np.mean(b_B_vals) * s_sq_grid), 1e-4, 0.9999)

    # Uncertainty bands
    sigma_a_A_lower = np.clip(np.sqrt(np.min(k_A_vals)) * np.exp(-0.25 * np.max(b_A_vals) * s_sq_grid), 1e-4, 0.9999)
    sigma_a_A_upper = np.clip(np.sqrt(np.max(k_A_vals)) * np.exp(-0.25 * np.min(b_A_vals) * s_sq_grid), 1e-4, 0.9999)
    sigma_a_B_lower = np.clip(np.sqrt(np.min(k_B_vals)) * np.exp(-0.25 * np.max(b_B_vals) * s_sq_grid), 1e-4, 0.9999)
    sigma_a_B_upper = np.clip(np.sqrt(np.max(k_B_vals)) * np.exp(-0.25 * np.min(b_B_vals) * s_sq_grid), 1e-4, 0.9999)
    return {
        's_grid': s_grid,
        'd_grid': d_grid,
        'sigma_a_A_grid': sigma_a_A_grid,
        'sigma_a_B_grid': sigma_a_B_grid,
        'sigma_a_A_lower': sigma_a_A_lower,
        'sigma_a_A_upper': sigma_a_A_upper,
        'sigma_a_B_lower': sigma_a_B_lower,
        'sigma_a_B_upper': sigma_a_B_upper,
    }


def _optimism_work(split, thetas_A, thetas_B, nll_A, nll_B, bridge, fc_A, fc_B, i_obs, s_sq):
    n_tune = int(split.tune.sum())
    p_theta_A = thetas_A[0].p_theta
    p_theta_B = thetas_B[0].p_theta

    optimism_expected_A = float(p_theta_A / max(n_tune, 1))
    optimism_expected_B = float(p_theta_B / max(n_tune, 1))

    mean_tune_nll_A = float(np.mean([th.tune_nll for th in thetas_A]))
    mean_tune_nll_B = float(np.mean([th.tune_nll for th in thetas_B]))

    gap_tune_test_A = float(nll_A - mean_tune_nll_A)
    gap_tune_test_B = float(nll_B - mean_tune_nll_B)

    # Working set evaluation
    work_mask = split.work
    n_work = int(work_mask.sum())
    if n_work > 0:
        ell_A_work = _eval_per_reflection(bridge, fc_A, i_obs, thetas_A[0], s_sq=s_sq)
        ell_B_work = _eval_per_reflection(bridge, fc_B, i_obs, thetas_B[0], s_sq=s_sq)
        nll_work_A = float(np.mean(ell_A_work[work_mask]))
        nll_work_B = float(np.mean(ell_B_work[work_mask]))
    else:
        nll_work_A = nll_A
        nll_work_B = nll_B

    gap_work_test_A = float(nll_A - nll_work_A)
    gap_work_test_B = float(nll_B - nll_work_B)
    return {
        'n_tune': n_tune,
        'p_theta_A': p_theta_A,
        'p_theta_B': p_theta_B,
        'optimism_expected_A': optimism_expected_A,
        'optimism_expected_B': optimism_expected_B,
        'gap_tune_test_A': gap_tune_test_A,
        'gap_tune_test_B': gap_tune_test_B,
        'n_work': n_work,
        'nll_work_A': nll_work_A,
        'nll_work_B': nll_work_B,
        'gap_work_test_A': gap_work_test_A,
        'gap_work_test_B': gap_work_test_B,
    }


def _secondary_and_theta(n_test, fc_A_data, fc_B_data, t_idx, io_data, p_theta_A, p_theta_B, thetas_A, thetas_B):
    sec_metrics = {}
    if n_test > 0:
        fc_A_amp = np.abs(fc_A_data)[t_idx]
        fc_B_amp = np.abs(fc_B_data)[t_idx]
        io_test = io_data[t_idx]

        # direct intensity R (free): sum |I_obs - k*Fc^2| / sum I_obs
        k_A = float(np.sum(io_test * fc_A_amp**2) / max(np.sum(fc_A_amp**4), 1e-12))
        k_B = float(np.sum(io_test * fc_B_amp**2) / max(np.sum(fc_B_amp**4), 1e-12))
        r_A = float(np.sum(np.abs(io_test - k_A * fc_A_amp**2)) / max(np.sum(io_test), 1e-12)) * 100.0
        r_B = float(np.sum(np.abs(io_test - k_B * fc_B_amp**2)) / max(np.sum(io_test), 1e-12)) * 100.0

        # CC_int free
        cc_A = float(np.corrcoef(io_test, fc_A_amp**2)[0, 1]) if np.std(fc_A_amp) > 1e-6 else 0.0
        cc_B = float(np.corrcoef(io_test, fc_B_amp**2)[0, 1]) if np.std(fc_B_amp) > 1e-6 else 0.0

        sec_metrics = {
            "r_free_A": f"{r_A:.2f}",
            "r_free_B": f"{r_B:.2f}",
            "cc_free_A": f"{cc_A:.4f}",
            "cc_free_B": f"{cc_B:.4f}",
        }

    theta_A_summary = {
        "p_theta": p_theta_A,
        "nu": thetas_A[0].nu,
        "nu_se": thetas_A[0].nu_se,
        "scale_k": thetas_A[0].scale_k,
        "k": thetas_A[0].sigma_a_params["k"],
        "b_delta": thetas_A[0].sigma_a_params["b_delta"],
        "sigma_0": thetas_A[0].sigma_wilson_params["sigma_0"],
        "b_wilson": thetas_A[0].sigma_wilson_params["b_wilson"],
    }
    theta_B_summary = {
        "p_theta": p_theta_B,
        "nu": thetas_B[0].nu,
        "nu_se": thetas_B[0].nu_se,
        "scale_k": thetas_B[0].scale_k,
        "k": thetas_B[0].sigma_a_params["k"],
        "b_delta": thetas_B[0].sigma_a_params["b_delta"],
        "sigma_0": thetas_B[0].sigma_wilson_params["sigma_0"],
        "b_wilson": thetas_B[0].sigma_wilson_params["b_wilson"],
    }
    return {
        'sec_metrics': sec_metrics,
        'theta_A_summary': theta_A_summary,
        'theta_B_summary': theta_B_summary,
    }


# ---------------------------------------------------------------------------
# 7. Paired-Refinement Ladder: ladder()
# ---------------------------------------------------------------------------
@dataclass
class LadderStep:
    """One step of resolution extension in the paired-refinement ladder."""

    step_idx: int
    d_prev: float
    d_curr: float
    comparison: ComparisonReport
    predictive_shell_nll: float
    fisher_info_gain_bits: float
    realized_delta_bits: float
    stop_recommendation: bool


@dataclass
class LadderReport:
    """Summary of paired resolution-extension ladder benchmark."""

    base_d: float
    steps: list[LadderStep]
    stopped_at_step: Optional[int] = None

    def to_markdown(self) -> str:
        lines = [
            f"# Paired-Refinement Ladder Report (Base Cutoff: `{self.base_d:.2f}` Å)",
            "",
            "| Step | Cutoffs (Å) | Base $\\Delta$ Gain | Boot SE | Pred Shell NLL | Info Gain (bits) | Realized (bits) | Stop? |",
            "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for s in self.steps:
            stop_str = "STOP" if s.stop_recommendation else "continue"
            lines.append(
                f"| {s.step_idx} | {s.d_prev:.2f} $\\to$ {s.d_curr:.2f} | {s.comparison.delta_gain:+.4f} | "
                f"{s.comparison.se_boot:.4f} | {s.predictive_shell_nll:.4f} | {s.fisher_info_gain_bits:.2f} | "
                f"{s.realized_delta_bits:+.2f} | {stop_str} |"
            )
        return "\n".join(lines)


def ladder(
    bridge: Any,
    i_obs: Any,
    models: Union[dict[float, Any], list[tuple[float, Any]]],
    split: Optional[ReflectionSplit] = None,
    *,
    base_d: Optional[float] = None,
    steps: Optional[list[float]] = None,
    n_boot: int = 1000,
    seed: int = 0,
) -> LadderReport:
    """Paired-refinement ladder for resolution-extension benchmarks.

    Refine against cutoffs d_0 > d_1 > ...
    Consecutive cutoffs (d_i, d_{i+1}) are compared strictly on the base-cutoff
    test reflections (d >= d_0) so all models are evaluated on identical data.
    """
    _check_intensity_observation_space(i_obs)

    if bridge is None:
        from phridge.api import Bridge

        bridge = Bridge(memory=True)

    # Sort models by cutoff descending: d_0 > d_1 > ...
    if isinstance(models, dict):
        sorted_pairs = sorted(models.items(), key=lambda kv: float(kv[0]), reverse=True)
    else:
        sorted_pairs = sorted(models, key=lambda x: float(x[0]), reverse=True)

    if len(sorted_pairs) < 2:
        raise ValueError("ladder() requires at least 2 cutoff models")

    cutoffs = [float(p[0]) for p in sorted_pairs]
    f_calcs = [p[1] for p in sorted_pairs]

    effective_base_d = float(base_d) if base_d is not None else cutoffs[0]

    d_spacings = np.asarray(i_obs.d_spacings().data(), dtype=np.float64)

    # Filter base-cutoff reflections (d >= base_d)
    base_mask = d_spacings >= (effective_base_d - 1e-6)
    if not np.any(base_mask):
        raise ValueError(f"No reflections found with d >= {effective_base_d} Å")

    if split is None:
        # Default split from r_free_flags or synthetic 10%
        flags = np.zeros(len(d_spacings), dtype=bool)
        flags[::10] = True
        split = ReflectionSplit.from_arrays(d_spacings, flags, n_shells=15, seed=seed)

    ladder_steps: list[LadderStep] = []
    consecutive_insignificant = 0
    stopped_at: Optional[int] = None

    for i in range(len(cutoffs) - 1):
        d_prev = cutoffs[i]
        d_curr = cutoffs[i + 1]
        fc_prev = f_calcs[i]
        fc_curr = f_calcs[i + 1]

        # 1. Compare on base-cutoff reflections
        from benchmarks.intensity_benchmark import compare
        comp = compare(
            bridge,
            i_obs,
            {f"{d_prev:.2f}A": fc_prev, f"{d_curr:.2f}A": fc_curr},
            split,
            n_boot=n_boot,
            seed=seed + i,
        )

        # 2. Predictive NLL on the newly added shell (d_curr <= d < d_prev)
        shell_mask = (d_spacings >= d_curr - 1e-6) & (d_spacings < d_prev - 1e-6) & split.test
        if np.any(shell_mask):
            th_prev = fit_nuisance(bridge, i_obs, fc_prev, split.tune, model_name=f"{d_prev:.2f}A")
            ell_shell = _eval_per_reflection(bridge, fc_prev, i_obs, th_prev)
            pred_shell_nll = float(np.mean(ell_shell[shell_mask]))
        else:
            pred_shell_nll = float(comp.nll_A)

        # 3. Fisher information gain approximation in bits
        # Tr(H_base^{-1} H_shell) ~ N_shell / N_base * (1 / ln 2)
        n_shell_refl = int(np.sum(shell_mask)) if np.any(shell_mask) else 1
        n_base_refl = int(np.sum(base_mask))
        fisher_info_bits = float(0.5 * (n_shell_refl / max(n_base_refl, 1)) / math.log(2.0) * 100.0)
        realized_delta_bits = float(comp.delta_gain / math.log(2.0))

        # 4. Stop criterion: delta within its bootstrap SE for two consecutive steps
        is_insignificant = abs(comp.delta_gain) <= comp.se_boot
        if is_insignificant:
            consecutive_insignificant += 1
        else:
            consecutive_insignificant = 0

        stop_rec = consecutive_insignificant >= 2
        if stop_rec and stopped_at is None:
            stopped_at = i + 1

        ladder_steps.append(
            LadderStep(
                step_idx=i + 1,
                d_prev=d_prev,
                d_curr=d_curr,
                comparison=comp,
                predictive_shell_nll=pred_shell_nll,
                fisher_info_gain_bits=fisher_info_bits,
                realized_delta_bits=realized_delta_bits,
                stop_recommendation=stop_rec,
            )
        )

    return LadderReport(base_d=effective_base_d, steps=ladder_steps, stopped_at_step=stopped_at)
