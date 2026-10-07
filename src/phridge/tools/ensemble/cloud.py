"""Latent-cloud summaries. No clustering and no manifold learning."""

from __future__ import annotations

from typing import Optional

import numpy as np


def deconvolved_covariance(scores: np.ndarray, score_cov: np.ndarray) -> np.ndarray:
    """Empirical score covariance minus the mean estimation covariance."""
    z = np.asarray(scores, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] == 0 or z.shape[0] < 2:
        k = 0 if z.ndim != 2 else z.shape[1]
        return np.zeros((k, k), dtype=np.float64)
    empirical = np.cov(z, rowvar=False)
    if empirical.ndim == 0:
        empirical = np.array([[float(empirical)]])
    estimation = np.mean(np.asarray(score_cov, dtype=np.float64), axis=0)
    return empirical - estimation


def quadratic_r2(scores: np.ndarray) -> list[float]:
    """R² of each score on quadratic terms of the preceding scores.

    The leading score has no preceding terms and is reported as 0.
    """
    z = np.asarray(scores, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] == 0:
        return []
    out = [0.0]
    for k in range(1, z.shape[1]):
        base = z[:, :k]
        cols = [base, base**2]
        if k > 1:
            products = []
            for i in range(k):
                for j in range(i + 1, k):
                    products.append((base[:, i] * base[:, j])[:, None])
            if products:
                cols.append(np.concatenate(products, axis=1))
        design = np.concatenate(cols, axis=1)
        design = np.concatenate([np.ones((z.shape[0], 1)), design], axis=1)
        target = z[:, k]
        coef, *_ = np.linalg.lstsq(design, target, rcond=None)
        pred = design @ coef
        ss_tot = float(np.sum((target - target.mean()) ** 2))
        ss_res = float(np.sum((target - pred) ** 2))
        out.append(0.0 if ss_tot <= 1e-18 else 1.0 - ss_res / ss_tot)
    return out


def positivity_margin(density: np.ndarray, mode_maps: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Per dataset, the minimum of ``ρ + Σ z_k u_k`` on the grid."""
    rho = np.asarray(density, dtype=np.float64)
    modes = np.asarray(mode_maps, dtype=np.float64)
    z = np.asarray(scores, dtype=np.float64)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    margins = np.zeros(z.shape[0], dtype=np.float64)
    for n in range(z.shape[0]):
        field = rho.copy()
        for k in range(min(modes.shape[0], z.shape[1])):
            field = field + float(z[n, k]) * modes[k]
        margins[n] = float(np.min(field))
    return margins


def latent_cloud(
    scores: np.ndarray,
    score_cov: np.ndarray,
    *,
    density: Optional[np.ndarray] = None,
    mode_maps: Optional[np.ndarray] = None,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "scores": np.asarray(scores, dtype=np.float64),
        "score_cov": np.asarray(score_cov, dtype=np.float64),
        "deconvolved_covariance": deconvolved_covariance(scores, score_cov),
        "quadratic_r2": quadratic_r2(scores),
    }
    if density is not None and mode_maps is not None:
        summary["positivity_margin"] = positivity_margin(density, mode_maps, scores)
    return summary
