"""Omit map analysis, spatial indexing, and peak detection utilities."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from cctbx import maptbx
except ImportError:
    maptbx = None


def min_sym_dist_to_coords(
    p_frac: Tuple[float, float, float],
    coords_frac: List[Tuple[float, float, float]],
    uc: Any,
    sg: Any,
) -> Tuple[float, int]:
    """Compute the minimum distance between a fractional point and coordinate list under space group symmetry."""
    min_d2 = 1e9
    best_idx = -1
    for i, c_frac in enumerate(coords_frac):
        for op in sg:
            c_sym = op * c_frac
            d2 = uc.distance_mod_1(p_frac, c_sym).dist_sq
            if d2 < min_d2:
                min_d2 = d2
                best_idx = i
    return float(np.sqrt(min_d2)), best_idx


def build_periodic_expanded_coords(coords_cart: np.ndarray, uc: Any, sg: Any) -> np.ndarray:
    """Expand Cartesian coordinates under all space-group operations and 27 neighbor unit cells."""
    fracs = [uc.fractionalize(tuple(c)) for c in coords_cart]
    exp_cart = []
    for f in fracs:
        for op in sg:
            f_sym = op * f
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        f_shift = (f_sym[0] + dx, f_sym[1] + dy, f_sym[2] + dz)
                        exp_cart.append(uc.orthogonalize(f_shift))
    return np.array(exp_cart, dtype=np.float64)


def analyze_difference_map(
    coeffs: Any,
    target_coords: List[Tuple[float, float, float]],
    target_b: Optional[List[float]] = None,
    target_labels: Optional[List[str]] = None,
    match_radius: float = 0.8,
) -> Dict[str, Any]:
    """Perform peak search on difference map coefficients and evaluate target vs noise peaks."""
    if maptbx is None:
        raise ImportError("cctbx is required for difference map FFT and peak search")

    uc = coeffs.unit_cell()
    sg = coeffs.space_group()

    fft_map = coeffs.fft_map(symmetry_flags=maptbx.use_space_group_symmetry, resolution_factor=0.25)
    fft_map.apply_sigma_scaling()
    peaks = fft_map.peak_search()

    peak_sites_frac = peaks.sites()
    peak_heights = np.asarray(peaks.heights(), dtype=np.float64)

    target_frac = [uc.fractionalize(tuple(c)) for c in target_coords]
    n_targets = len(target_coords)

    best_peak_for_target = [-1] * n_targets
    min_dist_for_target = [1e9] * n_targets

    is_target_peak = np.zeros(len(peak_heights), dtype=bool)
    matched_target_idx = np.full(len(peak_heights), -1, dtype=int)

    for p_idx, p_frac in enumerate(peak_sites_frac):
        dist, t_idx = min_sym_dist_to_coords(p_frac, target_frac, uc, sg)
        if dist <= match_radius:
            is_target_peak[p_idx] = True
            matched_target_idx[p_idx] = t_idx
            if dist < min_dist_for_target[t_idx]:
                min_dist_for_target[t_idx] = dist
                best_peak_for_target[t_idx] = p_idx

    target_peak_heights = []
    target_peak_dists = []
    for t_idx in range(n_targets):
        p_idx = best_peak_for_target[t_idx]
        if p_idx >= 0:
            target_peak_heights.append(float(peak_heights[p_idx]))
            target_peak_dists.append(float(min_dist_for_target[t_idx]))
        else:
            target_peak_heights.append(0.0)
            target_peak_dists.append(float(match_radius))

    noise_peak_heights = peak_heights[~is_target_peak]
    target_peak_heights_arr = np.array(target_peak_heights)

    n_detected_3sigma = int(np.sum(target_peak_heights_arr >= 3.0))
    n_detected_4sigma = int(np.sum(target_peak_heights_arr >= 4.0))

    noise_max = float(np.max(noise_peak_heights)) if len(noise_peak_heights) > 0 else 0.0
    noise_p99 = float(np.percentile(noise_peak_heights, 99)) if len(noise_peak_heights) > 0 else 0.0
    noise_p95 = float(np.percentile(noise_peak_heights, 95)) if len(noise_peak_heights) > 0 else 0.0
    noise_median = float(np.median(noise_peak_heights)) if len(noise_peak_heights) > 0 else 0.0

    target_min = float(np.min(target_peak_heights_arr)) if len(target_peak_heights_arr) > 0 else 0.0
    target_median = float(np.median(target_peak_heights_arr)) if len(target_peak_heights_arr) > 0 else 0.0
    target_mean = float(np.mean(target_peak_heights_arr)) if len(target_peak_heights_arr) > 0 else 0.0

    b_corr = None
    if target_b is not None and len(target_b) == n_targets:
        tb = np.array(target_b)
        if np.std(tb) > 0 and np.std(target_peak_heights_arr) > 0:
            b_corr = float(np.corrcoef(tb, target_peak_heights_arr)[0, 1])

    return {
        "n_targets": n_targets,
        "n_detected_3sigma": n_detected_3sigma,
        "pct_detected_3sigma": float(n_detected_3sigma / max(n_targets, 1) * 100.0),
        "n_detected_4sigma": n_detected_4sigma,
        "pct_detected_4sigma": float(n_detected_4sigma / max(n_targets, 1) * 100.0),
        "target_min": target_min,
        "target_median": target_median,
        "target_mean": target_mean,
        "noise_max": noise_max,
        "noise_p99": noise_p99,
        "noise_p95": noise_p95,
        "noise_median": noise_median,
        "clean_gap": float(target_min - noise_max),
        "mean_dist": float(np.mean(target_peak_dists)),
        "b_corr": b_corr,
        "n_total_peaks": len(peak_heights),
        "n_noise_peaks": len(noise_peak_heights),
    }
