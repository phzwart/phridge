"""Per-dataset explanation: chi²/dof under measurement weights, including LOO."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from phridge.contrib.multixtal.factors import leave_one_out_factors
from phridge.contrib.multixtal.loadings import FreeInPhaseLoadings


@dataclass
class DatasetExplanation:
    chi2_model: np.ndarray
    chi2_mean: np.ndarray
    chi2_in_sample: np.ndarray
    chi2_loo: np.ndarray
    dof: np.ndarray
    flagged: np.ndarray
    f_explained: np.ndarray


def chi2_dof(residual: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-dataset chi² and dof under ``W_meas``."""
    r = np.asarray(residual, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    chi2 = np.sum(w * r * r, axis=1)
    dof = np.sum(w > 0, axis=1).astype(np.float64)
    return chi2, dof


def robust_flag(values: np.ndarray, z: float = 3.0) -> np.ndarray:
    """Flag values more than ``z`` robust standard deviations (MAD) above the median."""
    v = np.asarray(values, dtype=np.float64)
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    sigma = 1.4826 * mad if mad > 0 else float(np.std(v))
    if sigma <= 0:
        return np.zeros(v.shape, dtype=bool)
    return v > (med + float(z) * sigma)


def explain_datasets(
    y: np.ndarray,
    w_meas: np.ndarray,
    mu: np.ndarray,
    scores: np.ndarray,
    loadings: np.ndarray,
    w_model: np.ndarray,
    ridge: float = 1e-6,
    flag_z: float = 3.0,
) -> DatasetExplanation:
    n_data, n_h = y.shape
    rank = loadings.shape[0]
    pred0 = np.zeros_like(y)
    pred_mu = np.broadcast_to(mu[None, :], y.shape)
    pred_full = pred_mu + FreeInPhaseLoadings().apply(loadings, scores)
    c0, dof = chi2_dof(y - pred0, w_meas)
    c_mu, _ = chi2_dof(y - pred_mu, w_meas)
    c_in, _ = chi2_dof(y - pred_full, w_meas)
    c_loo = np.zeros(n_data, dtype=np.float64)
    for d in range(n_data):
        if n_data < 2 or rank < 0:
            c_loo[d] = c_mu[d]
            continue
        mu_d, l_d, z_d = leave_one_out_factors(y, w_model, d, rank, ridge=ridge)
        pred = mu_d + (z_d @ l_d if rank else 0.0)
        chi2, _ = chi2_dof((y[d] - pred)[None, :], w_meas[d][None, :])
        c_loo[d] = chi2[0]
    ratio_loo = c_loo / np.maximum(dof, 1.0)
    flagged = robust_flag(ratio_loo, z=flag_z) if n_data >= 3 else np.zeros(n_data, dtype=bool)
    # f_d = (χ²_mean - χ²_LOO) / (χ²_mean - 1)  — here χ² means χ²/dof
    r_mean = c_mu / np.maximum(dof, 1.0)
    f = (r_mean - ratio_loo) / np.maximum(r_mean - 1.0, 1e-6)
    return DatasetExplanation(
        chi2_model=c0 / np.maximum(dof, 1.0),
        chi2_mean=r_mean,
        chi2_in_sample=c_in / np.maximum(dof, 1.0),
        chi2_loo=ratio_loo,
        dof=dof,
        flagged=flagged,
        f_explained=f,
    )
