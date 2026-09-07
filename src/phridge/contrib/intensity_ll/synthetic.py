"""Synthetic crystallographic intensity generation with Student-t noise models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from cctbx.array_family import flex
except ImportError:
    flex = None


def sample_student_t(
    nu: float,
    size: Union[int, Tuple[int, ...]],
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample standard Student-t distributed variates with nu degrees of freedom.

    Returns:
        (t_samples, weights):
            t_samples: sampled Student-t variates
            weights: gamma scale variates such that var(t | w) = 1/w
    """
    if rng is None:
        rng = np.random.default_rng()

    # Student-t variate as scale-mixture of normals:
    # w ~ Gamma(nu/2, 2/nu) (mean 1)
    # z ~ Normal(0, 1)
    # t = z / sqrt(w)
    weights = rng.gamma(shape=nu / 2.0, scale=2.0 / nu, size=size)
    z = rng.normal(loc=0.0, scale=1.0, size=size)
    t = z / np.sqrt(np.maximum(weights, 1e-12))
    return t, weights


def compute_base_sigmas(
    i_true: np.ndarray,
    sig_exp: Optional[np.ndarray] = None,
    mode: str = "direct",
) -> np.ndarray:
    """Compute baseline standard deviations (sigmas) for synthetic noise injection.

    Modes:
      - 'direct': Use experimental sigmas if provided, else Poisson-like sqrt(max(I, 0)).
      - 'poisson': Pure counting statistics sqrt(max(I, 1.0)).
      - 'snr': Scaled to uniform target SNR.
    """
    if mode == "direct" and sig_exp is not None:
        return np.maximum(np.asarray(sig_exp, dtype=np.float64), 1e-4)
    elif mode == "poisson":
        return np.maximum(np.sqrt(np.maximum(np.asarray(i_true, dtype=np.float64), 1.0)), 0.1)
    elif mode == "snr":
        # Target SNR ~ 10 on average
        mean_i = max(float(np.mean(np.maximum(i_true, 0.0))), 1.0)
        return np.full_like(i_true, fill_value=mean_i / 10.0, dtype=np.float64)
    else:
        # Default fallback
        if sig_exp is not None:
            return np.maximum(np.asarray(sig_exp, dtype=np.float64), 1e-4)
        return np.maximum(np.sqrt(np.maximum(np.asarray(i_true, dtype=np.float64), 0.0)) * 0.05, 0.05)


@dataclass
class SyntheticIntensityResult:
    """Container for synthetic noisy reflection generation results."""

    multiplier: float
    mtz_path: Optional[Path] = None
    i_synth: np.ndarray = field(default_factory=lambda: np.empty(0))
    sig_synth: np.ndarray = field(default_factory=lambda: np.empty(0))
    stats: Dict[str, Any] = field(default_factory=dict)
    shell_stats: List[Dict[str, Any]] = field(default_factory=list)


def generate_synthetic_series(
    model: Any,
    multipliers: List[float],
    nu: float = 7.0,
    mode: str = "direct",
    output_dir: Optional[Union[str, Path]] = None,
    base_prefix: str = "synthetic_t",
    seed: int = 42,
    write_mtz: bool = True,
) -> Tuple[Dict[float, SyntheticIntensityResult], List[Dict[str, Any]]]:
    """Generate synthetic noisy intensity datasets across multiple noise multipliers.

    Returns:
        (results_dict, summary_rows)
    """
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = None

    # Ground truth intensities from model
    f_calc = model.f_calc
    if hasattr(f_calc, "data"):
        f_data = np.asarray(f_calc.data())
        i_true = np.abs(f_data) ** 2
    else:
        i_true = np.asarray(model.i_obs.data(), dtype=np.float64)

    sig_exp = np.asarray(model.i_obs.sigmas(), dtype=np.float64) if model.i_obs.sigmas() is not None else None
    sig_base = compute_base_sigmas(i_true, sig_exp, mode=mode)

    d_spacings = np.asarray(model.i_obs.d_spacings().data(), dtype=np.float64)
    n_refl = len(i_true)

    # Resolution shells setup
    n_bins = getattr(model, "n_bins", 10)
    d_min = float(np.min(d_spacings))
    d_max = float(np.max(d_spacings))
    bin_edges = np.linspace(1.0 / (d_max**2), 1.0 / (d_min**2), n_bins + 1)
    inv_d2 = 1.0 / (d_spacings**2)
    bin_indices = np.clip(np.digitize(inv_d2, bin_edges) - 1, 0, n_bins - 1)

    results: Dict[float, SyntheticIntensityResult] = {}
    summary_rows: List[Dict[str, Any]] = []

    for mult in multipliers:
        rng = np.random.default_rng(seed + int(mult * 100))
        m_val = float(mult)

        if m_val <= 0.0:
            i_synth = i_true.copy()
            sig_synth = np.maximum(np.sqrt(np.maximum(i_true, 0.0)) * 0.01, 0.01)
        else:
            sig_synth = m_val * sig_base
            t_noise, _ = sample_student_t(nu=nu, size=n_refl, rng=rng)
            i_synth = i_true + sig_synth * t_noise

        # Metrics
        i_over_sig = i_synth / np.maximum(sig_synth, 1e-12)
        mean_i_over_sig = float(np.mean(i_over_sig))
        median_i_over_sig = float(np.median(i_over_sig))

        # Highest resolution shell
        outer_mask = (bin_indices == (n_bins - 1))
        high_res_i_over_sig = float(np.mean(i_over_sig[outer_mask])) if np.any(outer_mask) else 0.0

        mean_sig = float(np.mean(sig_synth))
        n_neg = int(np.sum(i_synth < 0.0))
        frac_neg_pct = float(n_neg / max(n_refl, 1) * 100.0)

        # R_noise = sum(|I_synth - I_true|) / sum(I_true)
        sum_true = float(np.sum(np.maximum(i_true, 0.0)))
        r_noise_pct = float(np.sum(np.abs(i_synth - i_true)) / max(sum_true, 1e-12) * 100.0)

        corr_true_synth = float(np.corrcoef(i_true, i_synth)[0, 1]) if np.std(i_synth) > 0 and np.std(i_true) > 0 else 1.0

        # Shell breakdown
        shell_stats = []
        for b in range(n_bins):
            mask_b = (bin_indices == b)
            n_b = int(np.sum(mask_b))
            if n_b == 0:
                continue
            d_b = d_spacings[mask_b]
            d_max_b = float(np.max(d_b))
            d_min_b = float(np.min(d_b))
            ios_b = i_over_sig[mask_b]
            is_b = i_synth[mask_b]
            it_b = i_true[mask_b]
            sig_b = sig_synth[mask_b]

            n_neg_b = int(np.sum(is_b < 0.0))
            sum_t_b = float(np.sum(np.maximum(it_b, 0.0)))
            r_n_b = float(np.sum(np.abs(is_b - it_b)) / max(sum_t_b, 1e-12) * 100.0)

            shell_stats.append({
                "shell_index": b + 1,
                "d_max": d_max_b,
                "d_min": d_min_b,
                "n_reflections": n_b,
                "mean_i_over_sig": float(np.mean(ios_b)),
                "median_i_over_sig": float(np.median(ios_b)),
                "mean_expected_snr": float(np.mean(it_b / np.maximum(sig_b, 1e-12))),
                "mean_sig": float(np.mean(sig_b)),
                "n_negative": n_neg_b,
                "fraction_negative": float(n_neg_b / n_b),
                "r_noise_percent": r_n_b,
            })

        # MTZ output if requested
        mtz_file_name = f"{base_prefix}_mult_{m_val:g}.mtz".replace(".", "_", 1) if "." in f"{m_val:g}" else f"{base_prefix}_mult_{m_val:g}.mtz"
        mtz_path = None
        if write_mtz and out_dir is not None and flex is not None:
            mtz_path = out_dir / mtz_file_name
            try:
                iobs_synth = model.i_obs.customized_copy(
                    data=flex.double(i_synth.tolist()),
                    sigmas=flex.double(sig_synth.tolist()),
                )
                iobs_synth.set_observation_type_xray_intensity()
                mtz_dataset = iobs_synth.as_mtz_dataset(column_root_label="IOBS")
                if getattr(model, "r_free_flags", None) is not None:
                    mtz_dataset.add_miller_array(model.r_free_flags, column_root_label="FreeR_flag")
                # Add noise-free true intensities
                itrue_arr = model.i_obs.customized_copy(data=flex.double(i_true.tolist()))
                mtz_dataset.add_miller_array(itrue_arr, column_root_label="ITRUE")
                mtz_dataset.mtz_object().write(str(mtz_path))
            except Exception:
                pass

        stats_dict = {
            "mean_i_over_sig": mean_i_over_sig,
            "median_i_over_sig": median_i_over_sig,
            "high_res_i_over_sig": high_res_i_over_sig,
            "mean_sig": mean_sig,
            "n_negative": n_neg,
            "fraction_negative_pct": frac_neg_pct,
            "r_noise_percent": r_noise_pct,
            "corr_true_synth": corr_true_synth,
        }

        results[m_val] = SyntheticIntensityResult(
            multiplier=m_val,
            mtz_path=mtz_path,
            i_synth=i_synth,
            sig_synth=sig_synth,
            stats=stats_dict,
            shell_stats=shell_stats,
        )

        summary_rows.append({
            "multiplier": m_val,
            "mtz_file": mtz_file_name if mtz_path else "None",
            "mean_i_over_sig": mean_i_over_sig,
            "median_i_over_sig": median_i_over_sig,
            "high_res_i_over_sig": high_res_i_over_sig,
            "mean_sig": mean_sig,
            "n_negative": n_neg,
            "fraction_negative_pct": frac_neg_pct,
            "r_noise_pct": r_noise_pct,
            "corr_true_synth": corr_true_synth,
        })

    return results, summary_rows
