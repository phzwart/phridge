"""Cross-validated held-out likelihood comparison of refined models (phridge / ``ml_i``).

Compares two or more refined crystallographic models of the same crystal by
**held-out negative log-likelihood (NLL)** under the intensity likelihood
(``ml_i``). Kept strictly valid by:
  1. Three-way resolution-shell split or cross-fitting within the free set
     so nuisance parameters (sigma_A(s), Wilson Sigma_N(s), scale, nu) are
     never fitted on scored reflections.
  2. Scoring all models under one likelihood in one observation space:
     intensities (ml_i).
  3. Decomposing differences into structure and error-model terms.
  4. Block-bootstrap uncertainty estimation over resolution shells.
  5. Breadth, optimism, and stereochemical checks.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import stats

from phridge.shims.cctbx import miller_from_cctbx
from phridge.legacy.intensity_ll_ops import NUISANCE_FIT_OP_NAME


# ---------------------------------------------------------------------------
# 1. Observation-space guard
# ---------------------------------------------------------------------------
def _check_intensity_observation_space(i_obs: Any) -> None:
    """Ensure that observations are experimental intensities, not amplitudes."""
    if hasattr(i_obs, "is_xray_intensity_array") and not i_obs.is_xray_intensity_array():
        raise ValueError(
            "Observation-space guard: held-out log-likelihood comparison requires "
            "experimental intensities (I_obs), not amplitudes (F_obs). Comparing "
            "models under amplitude likelihood or mixing observation spaces invalidates cross-validation."
        )
    if hasattr(i_obs, "observation_type"):
        obs_type = str(i_obs.observation_type()).lower()
        if "amplitude" in obs_type or obs_type == "f":
            raise ValueError(
                "Observation-space guard: held-out log-likelihood comparison requires "
                "experimental intensities (I_obs), not amplitudes (F_obs). Comparing "
                "models under amplitude likelihood or mixing observation spaces invalidates cross-validation."
            )


# ---------------------------------------------------------------------------
# 2. Data Splits: ReflectionSplit
# ---------------------------------------------------------------------------
@dataclass
class ReflectionSplit:
    """Disjoint partition of reflections into work, tune, and audit sets.

    Parameters
    ----------
    work : np.ndarray (bool)
        Reflections used for model refinement (the working set).
    tune : np.ndarray (bool)
        Reflections used strictly to estimate nuisance parameters (theta).
    test : np.ndarray (bool)
        Reflections in the audit set (held-out) scored to compare models.
    mode : str
        ``"cross_fit"`` (default) or ``"three_way"``.
    """

    work: np.ndarray
    tune: np.ndarray
    test: np.ndarray
    fold1_tune: np.ndarray
    fold1_test: np.ndarray
    fold2_tune: np.ndarray
    fold2_test: np.ndarray
    shell_indices: np.ndarray
    n_shells: int = 20
    seed: int = 0
    mode: str = "cross_fit"

    def __post_init__(self) -> None:
        self.work = np.asarray(self.work, dtype=bool)
        self.tune = np.asarray(self.tune, dtype=bool)
        self.test = np.asarray(self.test, dtype=bool)
        self.fold1_tune = np.asarray(self.fold1_tune, dtype=bool)
        self.fold1_test = np.asarray(self.fold1_test, dtype=bool)
        self.fold2_tune = np.asarray(self.fold2_tune, dtype=bool)
        self.fold2_test = np.asarray(self.fold2_test, dtype=bool)
        self.shell_indices = np.asarray(self.shell_indices, dtype=np.int32)

    @classmethod
    def from_arrays(
        cls,
        d_spacings: Any,
        r_free_in: Any,
        *,
        n_shells: int = 20,
        seed: int = 0,
        mode: str = "cross_fit",
    ) -> ReflectionSplit:
        """Construct ReflectionSplit from d-spacings and r_free flags."""
        d = np.asarray(d_spacings, dtype=np.float64)
        free = np.asarray(r_free_in, dtype=bool)
        n = len(free)
        work = ~free

        free_idx = np.where(free)[0]
        n_free = len(free_idx)
        if n_free < 4:
            raise ValueError(f"Free set has only {n_free} reflections; need >= 4 reflections for split")

        # Determine effective number of thin shells
        eff_shells = max(2, min(n_shells, n_free // 4))

        # Sort free reflections by resolution s = 1/d
        s_free = 1.0 / np.maximum(d[free_idx], 1e-6)
        sort_order = np.argsort(s_free)
        sorted_free_idx = free_idx[sort_order]

        # Assign shells
        shell_assignment = np.zeros(n, dtype=np.int32) - 1
        shell_splits = np.array_split(sorted_free_idx, eff_shells)

        half1 = np.zeros(n, dtype=bool)
        half2 = np.zeros(n, dtype=bool)

        rng = np.random.default_rng(seed)
        if mode == "cross_fit":
            # Alternating assignment of thin shells guarantees balanced resolution coverage
            # without leaking reciprocal-space correlations across shells.
            for i_sh, sh_indices in enumerate(shell_splits):
                shell_assignment[sh_indices] = i_sh
                if i_sh % 2 == 0:
                    half1[sh_indices] = True
                else:
                    half2[sh_indices] = True
        else:  # three_way (fixed tune / test)
            for i_sh, sh_indices in enumerate(shell_splits):
                shell_assignment[sh_indices] = i_sh
                if i_sh % 2 == 0:
                    half1[sh_indices] = True
                else:
                    half2[sh_indices] = True

        # Assign non-free reflections a shell index based on nearest d
        all_s = 1.0 / np.maximum(d, 1e-6)
        all_shell_splits = np.array_split(np.argsort(all_s), eff_shells)
        full_shell_indices = np.zeros(n, dtype=np.int32)
        for i_sh, sh_idx in enumerate(all_shell_splits):
            full_shell_indices[sh_idx] = i_sh

        fold1_tune = half1
        fold1_test = half2
        fold2_tune = half2
        fold2_test = half1

        if mode == "cross_fit":
            # In cross-fit, both halves are scored when in test set; test set covers the whole free set
            tune = half1
            test = free
        else:
            tune = half1
            test = half2

        return cls(
            work=work,
            tune=tune,
            test=test,
            fold1_tune=fold1_tune,
            fold1_test=fold1_test,
            fold2_tune=fold2_tune,
            fold2_test=fold2_test,
            shell_indices=full_shell_indices,
            n_shells=eff_shells,
            seed=seed,
            mode=mode,
        )

    @classmethod
    def from_miller(
        cls,
        i_obs: Any,
        r_free_flags: Any,
        *,
        n_shells: int = 20,
        seed: int = 0,
        mode: str = "cross_fit",
    ) -> ReflectionSplit:
        """Construct ReflectionSplit from cctbx miller arrays."""
        d = np.asarray(i_obs.d_spacings().data(), dtype=np.float64)
        if hasattr(r_free_flags, "data"):
            d_attr = r_free_flags.data
            d_val = d_attr() if callable(d_attr) else d_attr
            flags = np.asarray(d_val, dtype=bool)
        else:
            flags = np.asarray(r_free_flags, dtype=bool)
        return cls.from_arrays(d, flags, n_shells=n_shells, seed=seed, mode=mode)


# ---------------------------------------------------------------------------
# 3. Nuisance Parameters: NuisanceParams
# ---------------------------------------------------------------------------
@dataclass
class NuisanceParams:
    """Nuisance parameter vector theta fitted on tune reflections."""

    sigma_a_params: dict[str, float]
    sigma_a_curve: np.ndarray
    sigma_wilson_params: dict[str, float]
    sigma_wilson_curve: np.ndarray
    nu: Optional[float]
    nu_se: Optional[float]
    scale_k: float
    p_theta: int
    n_tune: int
    tune_nll: float
    model_name: Optional[str] = None


def fit_nuisance(
    bridge: Any,
    i_obs: Any,
    f_calc_model: Any,
    split_tune: Any,
    *,
    nu_bounds: tuple[float, float] = (2.5, 200.0),
    fit_scale: bool = False,
    fit_nu: bool = True,
    nu: Optional[float] = None,
    s_sq: Optional[np.ndarray] = None,
    model_name: Optional[str] = None,
) -> NuisanceParams:
    """Fit nuisance parameters theta = {sigma_A(s), Sigma_N(s), scale, nu} on tune reflections.

    Fitted strictly on tune reflections; never seen by scored test reflections.
    """
    _check_intensity_observation_space(i_obs)

    if bridge is None:
        from phridge.api import Bridge

        bridge = Bridge(memory=True)

    tune_mask = np.asarray(split_tune, dtype=bool)
    n_tune = int(tune_mask.sum())
    if n_tune == 0:
        raise ValueError("tune_mask contains 0 reflections")

    # Wire packaging
    packed_fc = miller_from_cctbx(f_calc_model) if hasattr(f_calc_model, "indices") else f_calc_model
    packed_io = miller_from_cctbx(i_obs) if hasattr(i_obs, "indices") else i_obs

    n = len(tune_mask)
    if hasattr(i_obs, "epsilons"):
        eps_np = np.asarray(i_obs.epsilons().data().as_double(), dtype=np.float64)
    else:
        eps_np = np.ones(n, dtype=np.float64)

    if hasattr(i_obs, "centric_flags"):
        cen_np = np.asarray(i_obs.centric_flags().data(), dtype=bool)
    else:
        cen_np = np.zeros(n, dtype=bool)

    if s_sq is None and hasattr(i_obs, "d_spacings"):
        d_val = np.asarray(i_obs.d_spacings().data(), dtype=np.float64)
        s_sq = 1.0 / np.maximum(d_val**2, 1e-12)

    res = bridge.call(
        NUISANCE_FIT_OP_NAME,
        f_calc=packed_fc,
        f_obs=packed_io,
        tune_mask=tune_mask,
        s_sq=s_sq,
        epsilon=eps_np,
        centric=cen_np,
        nu_bounds=list(nu_bounds),
        nu=nu,
        fit_nu=fit_nu,
        fit_scale=fit_scale,
    )

    return NuisanceParams(
        sigma_a_params=dict(res["sigma_a_params"]),
        sigma_a_curve=np.asarray(res["sigma_a"], dtype=np.float64),
        sigma_wilson_params=dict(res["sigma_wilson_params"]),
        sigma_wilson_curve=np.asarray(res["sigma_wilson"], dtype=np.float64),
        nu=res["nu"],
        nu_se=res["nu_se"],
        scale_k=float(res["scale_k"]),
        p_theta=int(res["p_theta"]),
        n_tune=n_tune,
        tune_nll=float(res["tune_nll"]),
        model_name=model_name,
    )


# ---------------------------------------------------------------------------
# 4. Helper for per-reflection target evaluation
# ---------------------------------------------------------------------------
def _eval_per_reflection(
    bridge: Any,
    f_calc: Any,
    i_obs: Any,
    theta: NuisanceParams,
    *,
    s_sq: Optional[np.ndarray] = None,
    scale_fc: bool = True,
) -> np.ndarray:
    """Evaluate per-reflection negative log-likelihood ell_h under given model and theta."""
    packed_fc = miller_from_cctbx(f_calc) if hasattr(f_calc, "indices") else f_calc
    packed_io = miller_from_cctbx(i_obs) if hasattr(i_obs, "indices") else i_obs

    n = len(theta.sigma_a_curve)
    if hasattr(i_obs, "epsilons"):
        eps_np = np.asarray(i_obs.epsilons().data().as_double(), dtype=np.float64)
    else:
        eps_np = np.ones(n, dtype=np.float64)

    if hasattr(i_obs, "centric_flags"):
        cen_np = np.asarray(i_obs.centric_flags().data(), dtype=bool)
    else:
        cen_np = np.zeros(n, dtype=bool)

    # Apply fitted overall scale on F_c if scale_fc is True
    if scale_fc and theta.scale_k != 1.0:
        if hasattr(packed_fc, "data"):
            fc_scaled_data = np.asarray(packed_fc.data) * theta.scale_k
            packed_fc = packed_fc.customized_copy(data=fc_scaled_data) if hasattr(packed_fc, "customized_copy") else packed_fc

    target_spec = {"name": "ml_i"}
    if theta.nu is not None and theta.nu < 199.0:
        target_spec["nu"] = float(theta.nu)

    res = bridge.call(
        "target_eval",
        f_calc=packed_fc,
        f_obs=packed_io,
        target=target_spec,
        alpha=theta.sigma_a_curve,
        beta=theta.sigma_wilson_curve,
        epsilon=eps_np,
        centric=cen_np,
        compute_curvature=False,
    )
    return np.asarray(res.per_reflection, dtype=np.float64)


# ---------------------------------------------------------------------------
# 5. Comparison Report
# ---------------------------------------------------------------------------
@dataclass
class ComparisonReport:
    """Statistical held-out likelihood comparison report between models."""

    name_A: str
    name_B: str
    n_test: int
    n_tune: int
    n_work: int

    # Test NLLs
    nll_A: float
    nll_B: float
    nll_A_under_B: float
    nll_B_under_A: float

    # Differences (in nats per reflection)
    # delta_gain = mean(d_h) = NLL(A) - NLL(B); positive means B is superior
    # delta_nll  = NLL(B) - NLL(A); negative means B is superior
    delta_gain: float
    delta_nll: float
    total_delta_gain: float  # log Bayes factor log(BF_B/A)

    # Decomposition (Equations 1 and 2 from Section 5)
    # Way 1 (under theta_A):
    struct_term_A: float  # NLL_T(B, theta_A) - NLL_T(A, theta_A)
    error_term_B: float   # NLL_T(B, theta_B) - NLL_T(B, theta_A)
    # Way 2 (under theta_B):
    struct_term_B: float  # NLL_T(B, theta_B) - NLL_T(A, theta_B)
    error_term_A: float   # NLL_T(A, theta_B) - NLL_T(A, theta_A)

    is_robust_structure: bool
    structure_conclusion: str

    # Uncertainty
    se_boot: float
    ci_boot: tuple[float, float]
    se_naive: float
    se_ratio: float

    # Breadth
    win_fraction: float
    wins_B: int
    wins_A: int
    ties: int
    mcnemar_p_value: float
    wilcoxon_stat: float
    wilcoxon_p_value: float
    top_discordant: list[dict[str, Any]]

    # Resolution shell breakdown
    shells: list[dict[str, Any]]

    # Held-out sigma_A curves
    s_grid: np.ndarray
    d_grid: np.ndarray
    sigma_a_A: np.ndarray
    sigma_a_A_lower: np.ndarray
    sigma_a_A_upper: np.ndarray
    sigma_a_B: np.ndarray
    sigma_a_B_lower: np.ndarray
    sigma_a_B_upper: np.ndarray

    # Optimism check
    optimism_expected_A: float
    optimism_expected_B: float
    gap_tune_test_A: float
    gap_tune_test_B: float
    nll_work_A: float
    nll_work_B: float
    gap_work_test_A: float
    gap_work_test_B: float

    # Nuisance parameter summaries
    theta_A: dict[str, Any]
    theta_B: dict[str, Any]

    # Secondary metrics (R-free, CC-free)
    secondary_metrics: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Concise terminal summary."""
        sign_str = "superior" if self.delta_gain > 0 else "inferior"
        return (
            f"Comparison: {self.name_B} vs {self.name_A}\n"
            f"  Delta NLL:      {self.delta_nll:+.4f} nats/refl (gain: {self.delta_gain:+.4f} nats/refl, total: {self.total_delta_gain:+.1f} nats)\n"
            f"  Bootstrap SE:   {self.se_boot:.4f} (95% CI: [{self.ci_boot[0]:+.4f}, {self.ci_boot[1]:+.4f}]) | Naive SE: {self.se_naive:.4f}\n"
            f"  Structure Term: under theta_A: {self.struct_term_A:+.4f}, under theta_B: {self.struct_term_B:+.4f} ({self.structure_conclusion})\n"
            f"  Win fraction:   {self.win_fraction:.1%} ({self.wins_B} wins / {self.wins_A} losses) | McNemar p: {self.mcnemar_p_value:.2e}\n"
            f"  Conclusion:     {self.name_B} is {sign_str} to {self.name_A}"
        )

    def to_markdown(self) -> str:
        """Generate comprehensive markdown report."""
        lines = [
            f"# Held-Out Log-Likelihood Comparison: `{self.name_B}` vs `{self.name_A}`",
            "",
            "## 1. Executive Summary",
            "",
            f"- **Scored Audit Reflections (|A|)**: {self.n_test} (audit set)",
            f"- **Nuisance Tune Reflections (|Tune|)**: {self.n_tune}",
            f"- **Working Reflections (|Work|)**: {self.n_work}",
            f"- **Model A NLL**: `{self.nll_A:.4f}` nats/refl",
            f"- **Model B NLL**: `{self.nll_B:.4f}` nats/refl",
            f"- **Difference (Gain $\\Delta$)**: `{self.delta_gain:+.4f}` nats/refl (`{self.delta_nll:+.4f}` nats NLL reduction)",
            f"- **Estimated Log Bayes Factor**: `{self.total_delta_gain:+.2f}` nats",
            f"- **Uncertainty**: Bootstrap SE = `{self.se_boot:.4f}` (95% CI: `[{self.ci_boot[0]:+.4f}, {self.ci_boot[1]:+.4f}]`) | Naive SE = `{self.se_naive:.4f}` (Ratio: `{self.se_ratio:.2f}`x)",
            f"- **Win Fraction $P(d_h > 0)$**: `{self.win_fraction:.1%}` ({self.wins_B} wins, {self.wins_A} losses, {self.ties} ties)",
            f"- **McNemar Sign Test $p$-value**: `{self.mcnemar_p_value:.2e}`",
            f"- **Wilcoxon Signed-Rank $p$-value**: `{self.wilcoxon_p_value:.2e}`",
            "",
            "## 2. Two-Way Likelihood Decomposition",
            "",
            "The comparison is decomposed into structural (atomic coordinates) and error-model ($\\sigma_A$, $\\nu$, scale) terms:",
            "",
            "| Decomposition Line | Structure Term | Error-Model Term | Total $\\Delta$ NLL | Interpretation |",
            "| :--- | :---: | :---: | :---: | :--- |",
            f"| **Anchored at $\\theta_A$** | `{self.struct_term_A:+.4f}` | `{self.error_term_B:+.4f}` | `{self.delta_nll:+.4f}` | $B$ vs $A$ under $\\theta_A$ + effect of changing error model to $\\theta_B$ |",
            f"| **Anchored at $\\theta_B$** | `{self.struct_term_B:+.4f}` | `{self.error_term_A:+.4f}` | `{self.delta_nll:+.4f}` | $B$ vs $A$ under $\\theta_B$ + effect of changing error model from $\\theta_A$ |",
            "",
            f"**Structural Robustness**: {self.structure_conclusion}",
            "",
            "## 3. Optimism and Overfitting Diagnosis",
            "",
            "| Diagnostic Metric | Model A (`" + self.name_A + "`) | Model B (`" + self.name_B + "`) | Description |",
            "| :--- | :---: | :---: | :--- |",
            f"| **Free Nuisance Parameters ($p_\\theta$)** | `{self.theta_A.get('p_theta', '-')}` | `{self.theta_B.get('p_theta', '-')}` | Total fitted nuisance parameters |",
            f"| **Expected In-Sample Optimism ($p_\\theta / |\\text{{tune}}|$)** | `{self.optimism_expected_A:.4f}` | `{self.optimism_expected_B:.4f}` | Theoretical shrinkage / optimism |",
            f"| **Tune $\\to$ Audit Gap (Audit − Tune NLL)** | `{self.gap_tune_test_A:+.4f}` | `{self.gap_tune_test_B:+.4f}` | Should be $\\approx p_\\theta / |\\text{{tune}}|$ if not overfit |",
            f"| **Work NLL** | `{self.nll_work_A:.4f}` | `{self.nll_work_B:.4f}` | NLL on refinement work set |",
            f"| **Work $\\to$ Audit Gap (Audit − Work NLL)** | `{self.gap_work_test_A:+.4f}` | `{self.gap_work_test_B:+.4f}` | Generalization gap across refinement |",
            "",
            "> **Note**: An improving work NLL together with a worsening audit NLL is the hallmark of overfitting, regardless of R-factor or CC values.",
            "",
            "## 4. Resolution Shell Breakdown",
            "",
            "| Shell | $d_{\\text{max}}$ (Å) | $d_{\\text{min}}$ (Å) | $N_{\\text{test}}$ | NLL(A) | NLL(B) | $\\Delta$ Gain | Boot SE | Win % | Wins / Losses |",
            "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for sh in self.shells:
            lines.append(
                f"| {sh['shell_idx']} | {sh['d_max']:.2f} | {sh['d_min']:.2f} | {sh['n_refl']} | "
                f"{sh['nll_A']:.4f} | {sh['nll_B']:.4f} | {sh['delta_gain']:+.4f} | {sh['se_boot']:.4f} | "
                f"{sh['win_fraction']:.1%} | {sh['wins']}/{sh['losses']} |"
            )

        if self.top_discordant:
            lines.extend([
                "",
                "## 5. Top Discordant Reflections (|d_h|)",
                "",
                "| $hkl$ | $d$ (Å) | $I_{\\text{obs}}$ | $\\sigma(I)$ | $I/\\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\\ell_h(A)$ | $\\ell_h(B)$ | $d_h$ (Gain) |",
                "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ])
            for r in self.top_discordant:
                hkl_str = f"({r['h']}, {r['k']}, {r['l']})"
                lines.append(
                    f"| {hkl_str} | {r['d']:.2f} | {r['i_obs']:.1f} | {r['sig_i']:.1f} | {r['i_over_sig']:.1f} | "
                    f"{r['fc_A']:.1f} | {r['fc_B']:.1f} | {r['nll_A']:.3f} | {r['nll_B']:.3f} | {r['d_h']:+.3f} |"
                )

        if self.secondary_metrics:
            lines.extend([
                "",
                "## 6. Secondary Reference Metrics",
                "",
                "*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*",
                "",
                "| Model | $R_{\\text{free}}$ (%) | $CC_{\\text{free}}$ | $\\nu$ | $\\text{SE}(\\nu)$ | Scale $k_F$ |",
                "| :--- | :---: | :---: | :---: | :---: | :---: |",
                f"| `{self.name_A}` | {self.secondary_metrics.get('r_free_A', '-')} | {self.secondary_metrics.get('cc_free_A', '-')} | {self.theta_A.get('nu', 'inf')} | {self.theta_A.get('nu_se', '-')} | {self.theta_A.get('scale_k', 1.0):.3f} |",
                f"| `{self.name_B}` | {self.secondary_metrics.get('r_free_B', '-')} | {self.secondary_metrics.get('cc_free_B', '-')} | {self.theta_B.get('nu', 'inf')} | {self.theta_B.get('nu_se', '-')} | {self.theta_B.get('scale_k', 1.0):.3f} |",
            ])

        return "\n".join(lines)

    def to_json(self) -> str:
        """Export report data as JSON string."""
        d = {
            "name_A": self.name_A,
            "name_B": self.name_B,
            "n_test": self.n_test,
            "n_tune": self.n_tune,
            "n_work": self.n_work,
            "nll_A": self.nll_A,
            "nll_B": self.nll_B,
            "nll_A_under_B": self.nll_A_under_B,
            "nll_B_under_A": self.nll_B_under_A,
            "delta_gain": self.delta_gain,
            "delta_nll": self.delta_nll,
            "total_delta_gain": self.total_delta_gain,
            "struct_term_A": self.struct_term_A,
            "error_term_B": self.error_term_B,
            "struct_term_B": self.struct_term_B,
            "error_term_A": self.error_term_A,
            "is_robust_structure": self.is_robust_structure,
            "structure_conclusion": self.structure_conclusion,
            "se_boot": self.se_boot,
            "ci_boot": list(self.ci_boot),
            "se_naive": self.se_naive,
            "se_ratio": self.se_ratio,
            "win_fraction": self.win_fraction,
            "wins_B": self.wins_B,
            "wins_A": self.wins_A,
            "ties": self.ties,
            "mcnemar_p_value": self.mcnemar_p_value,
            "wilcoxon_stat": self.wilcoxon_stat,
            "wilcoxon_p_value": self.wilcoxon_p_value,
            "top_discordant": self.top_discordant,
            "shells": self.shells,
            "s_grid": self.s_grid.tolist(),
            "d_grid": self.d_grid.tolist(),
            "sigma_a_A": self.sigma_a_A.tolist(),
            "sigma_a_A_lower": self.sigma_a_A_lower.tolist(),
            "sigma_a_A_upper": self.sigma_a_A_upper.tolist(),
            "sigma_a_B": self.sigma_a_B.tolist(),
            "sigma_a_B_lower": self.sigma_a_B_lower.tolist(),
            "sigma_a_B_upper": self.sigma_a_B_upper.tolist(),
            "optimism_expected_A": self.optimism_expected_A,
            "optimism_expected_B": self.optimism_expected_B,
            "gap_tune_test_A": self.gap_tune_test_A,
            "gap_tune_test_B": self.gap_tune_test_B,
            "nll_work_A": self.nll_work_A,
            "nll_work_B": self.nll_work_B,
            "gap_work_test_A": self.gap_work_test_A,
            "gap_work_test_B": self.gap_work_test_B,
            "theta_A": self.theta_A,
            "theta_B": self.theta_B,
            "secondary_metrics": self.secondary_metrics,
        }
        return json.dumps(d, indent=2)

def _pick(st: dict, keys: tuple[str, ...]) -> dict:
    return {key: st[key] for key in keys}


def compare(
    bridge: Any,
    i_obs: Any,
    models: dict[str, Any],
    split: ReflectionSplit,
    *,
    n_boot: int = 1000,
    fit_scale: bool = False,
    nu_bounds: tuple[float, float] = (2.5, 200.0),
    fit_nu: bool = True,
    model_nu: Optional[dict[str, Optional[float]]] = None,
    seed: int = 0,
) -> ComparisonReport:
    """Compare two refined models by held-out log-likelihood under ``ml_i``."""
    _check_intensity_observation_space(i_obs)
    if bridge is None:
        from phridge.api import Bridge

        bridge = Bridge(memory=True)
    from benchmarks.intensity_ladder import (
        _aggregate_scores,
        _bootstrap_gain,
        _breadth_tests,
        _discordant_rows,
        _optimism_work,
        _prepare_models,
        _score_folds,
        _secondary_and_theta,
        _shell_rows,
        _sigma_grids,
    )

    st = _prepare_models(models, model_nu, fit_nu, i_obs)
    st.update(_score_folds(
        split=split, bridge=bridge, i_obs=i_obs, nu_bounds=nu_bounds, fit_scale=fit_scale,
        **_pick(st, (
            "fc_A", "fc_B", "key_A", "key_B", "fit_nu_A", "fit_nu_B", "nu_A", "nu_B",
            "s_sq", "n_refl", "ell_A_test", "ell_B_test", "ell_A_under_B_test",
            "ell_B_under_A_test", "test_scored_mask",
        )),
    ))
    st.update(_aggregate_scores(**_pick(st, (
        "are_identical", "nu_A", "nu_B", "ell_A_test", "ell_B_test",
        "ell_A_under_B_test", "ell_B_under_A_test", "test_scored_mask",
    ))))
    st.update(_bootstrap_gain(
        n_boot=n_boot, seed=seed,
        **_pick(st, ("are_identical", "nu_A", "nu_B", "t_idx", "d_h", "n_test")),
        split=split,
    ))
    st.update(_breadth_tests(**_pick(st, ("are_identical", "nu_A", "nu_B", "d_h", "n_test"))))
    st.update(_discordant_rows(**_pick(st, (
        "n_test", "are_identical", "nu_A", "nu_B", "d_h", "fc_A_data", "fc_B_data",
        "t_idx", "hkl_list", "d_spacings", "io_data", "sig_data", "l_A", "l_B",
    ))))
    st.update(_shell_rows(**_pick(st, (
        "unique_shells", "shell_assignments_test", "t_idx", "d_spacings", "d_h", "l_A", "l_B", "rng",
    ))))
    st.update(_sigma_grids(**_pick(st, ("d_spacings", "thetas_A", "thetas_B"))))
    st.update(_optimism_work(
        split=split, bridge=bridge, i_obs=i_obs,
        **_pick(st, ("thetas_A", "thetas_B", "nll_A", "nll_B", "fc_A", "fc_B", "s_sq")),
    ))
    st.update(_secondary_and_theta(**_pick(st, (
        "n_test", "fc_A_data", "fc_B_data", "t_idx", "io_data", "p_theta_A", "p_theta_B",
        "thetas_A", "thetas_B",
    ))))
    return ComparisonReport(
        name_A=st["key_A"],
        name_B=st["key_B"],
        n_test=st["n_test"],
        n_tune=st["n_tune"],
        n_work=st["n_work"],
        nll_A=st["nll_A"],
        nll_B=st["nll_B"],
        nll_A_under_B=st["nll_AuB"],
        nll_B_under_A=st["nll_BuA"],
        delta_gain=st["delta_gain"],
        delta_nll=st["delta_nll"],
        total_delta_gain=st["total_delta_gain"],
        struct_term_A=st["struct_term_A"],
        error_term_B=st["error_term_B"],
        struct_term_B=st["struct_term_B"],
        error_term_A=st["error_term_A"],
        is_robust_structure=st["is_robust_structure"],
        structure_conclusion=st["structure_conclusion"],
        se_boot=st["se_boot"],
        ci_boot=st["ci_boot"],
        se_naive=st["se_naive"],
        se_ratio=st["se_ratio"],
        win_fraction=st["win_fraction"],
        wins_B=st["wins_B"],
        wins_A=st["wins_A"],
        ties=st["ties"],
        mcnemar_p_value=st["mcnemar_p"],
        wilcoxon_stat=st["wilcoxon_stat"],
        wilcoxon_p_value=st["wilcoxon_p"],
        top_discordant=st["top_discordant"],
        shells=st["shells"],
        s_grid=st["s_grid"],
        d_grid=st["d_grid"],
        sigma_a_A=st["sigma_a_A_grid"],
        sigma_a_A_lower=st["sigma_a_A_lower"],
        sigma_a_A_upper=st["sigma_a_A_upper"],
        sigma_a_B=st["sigma_a_B_grid"],
        sigma_a_B_lower=st["sigma_a_B_lower"],
        sigma_a_B_upper=st["sigma_a_B_upper"],
        optimism_expected_A=st["optimism_expected_A"],
        optimism_expected_B=st["optimism_expected_B"],
        gap_tune_test_A=st["gap_tune_test_A"],
        gap_tune_test_B=st["gap_tune_test_B"],
        nll_work_A=st["nll_work_A"],
        nll_work_B=st["nll_work_B"],
        gap_work_test_A=st["gap_work_test_A"],
        gap_work_test_B=st["gap_work_test_B"],
        theta_A=st["theta_A_summary"],
        theta_B=st["theta_B_summary"],
        secondary_metrics=st["sec_metrics"],
    )


def __getattr__(name: str):
    if name in {"ladder", "LadderReport", "LadderStep"}:
        import benchmarks.intensity_ladder as ladder_mod

        return getattr(ladder_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
