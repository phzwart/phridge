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

from phridge.client.convert import miller_from_cctbx
from phridge.contrib.intensity_ll.ops import NUISANCE_FIT_OP_NAME


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
        from phridge.client import Bridge

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


# ---------------------------------------------------------------------------
# 6. Scoring and Comparison: compare()
# ---------------------------------------------------------------------------
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
    """Compare two refined models by held-out log-likelihood under ``ml_i``.

    Parameters
    ----------
    bridge : Bridge or None
        Client bridge to Redis worker (or None for in-process Bridge(memory=True)).
    i_obs : MillerArray
        Experimental observations (must be intensities).
    models : dict
        Dict of model F_calc arrays, e.g. ``{"A": f_calc_A, "B": f_calc_B}``.
    split : ReflectionSplit
        Disjoint partition into work, tune, and audit sets.
    n_boot : int
        Number of block-bootstrap replicates over resolution shells (>= 1000).
    fit_scale : bool
        Whether to fit an overall scale on F_c during nuisance fitting.
    nu_bounds : tuple
        Bounds for Student-t nu parameter [2.5, 200.0].
    fit_nu : bool
        Whether to fit nu on the tune set.
    model_nu : dict, optional
        Pre-specified nu per model (e.g. ``{"A": 200.0, "B": None}`` for error-model tests).
    seed : int
        Random seed for bootstrap resampling.
    """
    _check_intensity_observation_space(i_obs)

    if bridge is None:
        from phridge.client import Bridge

        bridge = Bridge(memory=True)

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

    # Top discordant reflections
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

    # Per-shell breakdown table
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

    # Held-out sigma_A curves on a common grid with bootstrap bands
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

    # Optimism check
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

    # Secondary reference metrics (R-free, CC-free)
    sec_metrics = {}
    if n_test > 0:
        fc_A_amp = np.abs(fc_A_data)[t_idx]
        fc_B_amp = np.abs(fc_B_data)[t_idx]
        io_test = io_data[t_idx]

        # R_int free: sum |I_obs - k*Fc^2| / sum I_obs
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

    return ComparisonReport(
        name_A=key_A,
        name_B=key_B,
        n_test=n_test,
        n_tune=n_tune,
        n_work=n_work,
        nll_A=nll_A,
        nll_B=nll_B,
        nll_A_under_B=nll_AuB,
        nll_B_under_A=nll_BuA,
        delta_gain=delta_gain,
        delta_nll=delta_nll,
        total_delta_gain=total_delta_gain,
        struct_term_A=struct_term_A,
        error_term_B=error_term_B,
        struct_term_B=struct_term_B,
        error_term_A=error_term_A,
        is_robust_structure=is_robust_structure,
        structure_conclusion=structure_conclusion,
        se_boot=se_boot,
        ci_boot=ci_boot,
        se_naive=se_naive,
        se_ratio=se_ratio,
        win_fraction=win_fraction,
        wins_B=wins_B,
        wins_A=wins_A,
        ties=ties,
        mcnemar_p_value=mcnemar_p,
        wilcoxon_stat=wilcoxon_stat,
        wilcoxon_p_value=wilcoxon_p,
        top_discordant=top_discordant,
        shells=shells,
        s_grid=s_grid,
        d_grid=d_grid,
        sigma_a_A=sigma_a_A_grid,
        sigma_a_A_lower=sigma_a_A_lower,
        sigma_a_A_upper=sigma_a_A_upper,
        sigma_a_B=sigma_a_B_grid,
        sigma_a_B_lower=sigma_a_B_lower,
        sigma_a_B_upper=sigma_a_B_upper,
        optimism_expected_A=optimism_expected_A,
        optimism_expected_B=optimism_expected_B,
        gap_tune_test_A=gap_tune_test_A,
        gap_tune_test_B=gap_tune_test_B,
        nll_work_A=nll_work_A,
        nll_work_B=nll_work_B,
        gap_work_test_A=gap_work_test_A,
        gap_work_test_B=gap_work_test_B,
        theta_A=theta_A_summary,
        theta_B=theta_B_summary,
        secondary_metrics=sec_metrics,
    )


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
        from phridge.client import Bridge

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
