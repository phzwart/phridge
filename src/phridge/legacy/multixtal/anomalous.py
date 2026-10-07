"""Within-dataset Bijvoet-difference regression and per-dataset anomalous occupancy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class AnomalousFit:
    intercept: np.ndarray
    slopes: np.ndarray
    q: np.ndarray
    q_se: np.ndarray


class BijvoetRegression:
    """Phase-1 ``AnomalousTerm``: weighted regression of ``dE²`` on ``[1, z_d]``."""

    def regress(
        self,
        d_e2: np.ndarray,
        scores: np.ndarray,
        weights: np.ndarray,
        observed: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        d = np.asarray(d_e2, dtype=np.float64)
        z = np.asarray(scores, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)
        obs = np.asarray(observed, dtype=bool)
        n_data, n_h = d.shape
        rank = z.shape[1] if z.ndim == 2 else 0
        intercept = np.zeros(n_h, dtype=np.float64)
        slopes = np.zeros((rank, n_h), dtype=np.float64)
        for h in range(n_h):
            sel = obs[:, h] & (w[:, h] > 0) & np.isfinite(d[:, h])
            if int(sel.sum()) < 2:
                continue
            if rank == 0:
                ww = w[sel, h]
                intercept[h] = float(np.sum(ww * d[sel, h]) / np.sum(ww))
                continue
            x = np.concatenate([np.ones((int(sel.sum()), 1)), z[sel]], axis=1)
            ww = w[sel, h]
            xw = x * ww[:, None]
            gram = xw.T @ x
            rhs = xw.T @ d[sel, h]
            try:
                coef = np.linalg.solve(gram + 1e-10 * np.eye(x.shape[1]), rhs)
            except np.linalg.LinAlgError:
                coef = np.linalg.lstsq(gram, rhs, rcond=None)[0]
            intercept[h] = float(coef[0])
            slopes[:, h] = coef[1:]
        return intercept, slopes


def bijvoet_delta(
    i_plus: np.ndarray,
    i_minus: np.ndarray,
    sig_plus: np.ndarray,
    sig_minus: np.ndarray,
    mask_plus: np.ndarray,
    mask_minus: np.ndarray,
    centric: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Within-dataset ``dI = I+ - I-`` where both mates are measured (acentrics)."""
    both = (
        np.asarray(mask_plus, dtype=bool)
        & np.asarray(mask_minus, dtype=bool)
        & ~np.asarray(centric, dtype=bool)[None, :]
    )
    d_i = np.asarray(i_plus, dtype=np.float64) - np.asarray(i_minus, dtype=np.float64)
    var = np.asarray(sig_plus, dtype=np.float64) ** 2 + np.asarray(sig_minus, dtype=np.float64) ** 2
    d_i = np.where(both, d_i, 0.0)
    var = np.where(both, np.maximum(var, 1e-12), np.inf)
    return d_i, var, both


def normalize_delta(
    d_i: np.ndarray,
    var_i: np.ndarray,
    scale_a: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    sigma_p: np.ndarray,
    epsilon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize Bijvoet differences by the same per-dataset scale as Step 4."""
    den = (
        np.asarray(scale_a, dtype=np.float64)
        * np.asarray(epsilon, dtype=np.float64)
        * (
            np.asarray(alpha, dtype=np.float64) ** 2 * np.asarray(sigma_p, dtype=np.float64)
            + np.asarray(beta, dtype=np.float64)
        )
    )
    den = np.maximum(den, 1e-12)
    d_e2 = d_i / den
    var_e2 = var_i / (den**2)
    return d_e2, var_e2


def correct_slopes_for_scale_leakage(
    intercept: np.ndarray,
    slopes: np.ndarray,
    d_log_scale_dz: np.ndarray,
) -> np.ndarray:
    """``slopes += intercept * d(log scale)/dz``."""
    if slopes.size == 0:
        return slopes
    leak = np.asarray(d_log_scale_dz, dtype=np.float64).reshape(-1)
    if leak.size != slopes.shape[0]:
        leak = np.resize(leak, slopes.shape[0])
    return slopes + np.asarray(intercept, dtype=np.float64)[None, :] * leak[:, None]


def fit_site_q(
    d_i: np.ndarray,
    i_bar: np.ndarray,
    both: np.ndarray,
    signature: np.ndarray,
    var_ratio: np.ndarray,
    sigma_a: np.ndarray,
    strong: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale-free Bijvoet-ratio projection: ``ΔI / Ī ≈ signature * q``.

    ``signature`` is ``-4 Im(G_site / S_c)``. Weights include the model-error
    variance of the predicted signature so a few strong low-resolution
    reflections do not dominate.
    """
    ratio = np.asarray(d_i, dtype=np.float64) / np.maximum(np.asarray(i_bar, dtype=np.float64), 1e-8)
    x = np.asarray(signature, dtype=np.float64)
    n_data = ratio.shape[0]
    q = np.zeros(n_data, dtype=np.float64)
    q_se = np.full(n_data, np.nan, dtype=np.float64)
    sa = np.asarray(sigma_a, dtype=np.float64)
    vr = np.asarray(var_ratio, dtype=np.float64)
    for d in range(n_data):
        sel = both[d] & strong & np.isfinite(ratio[d]) & np.isfinite(x) & (np.abs(x) > 0)
        if int(sel.sum()) < 4:
            continue
        sa_d = sa[d, sel] if sa.ndim == 2 else sa[sel]
        w = 1.0 / np.maximum(vr[d, sel] + (1.0 - sa_d**2) * (x[sel] ** 2), 1e-12)
        xtw = x[sel] * w
        gram = float(xtw @ x[sel])
        if gram <= 0:
            continue
        q[d] = float(xtw @ ratio[d, sel] / gram)
        q_se[d] = float(np.sqrt(1.0 / gram))
    return q, q_se
