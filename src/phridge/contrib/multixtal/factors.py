"""Population mean, rank selection, weighted ALS, scale-leakage check."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from phridge.contrib.multixtal.latent import PointMassGLS


@dataclass
class FactorFit:
    mu: np.ndarray
    scores: np.ndarray
    loadings: np.ndarray
    singular_values: np.ndarray
    null_top: np.ndarray
    rank: int
    scale_leakage_slopes: np.ndarray
    scale_leakage_significant: np.ndarray


def weighted_mean(y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    v = np.asarray(y, dtype=np.float64)
    num = np.sum(w * v, axis=0)
    den = np.sum(w, axis=0)
    out = np.zeros(v.shape[1], dtype=np.float64)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def _weighted_residual_matrix(y: np.ndarray, mu: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    r = (np.asarray(y, dtype=np.float64) - mu[None, :]) * np.sqrt(w)
    return r


def permute_columns(y: np.ndarray, weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle each reflection's observed values across datasets; keep missingness."""
    out = np.array(y, copy=True, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    n_data, n_h = out.shape
    for h in range(n_h):
        obs = w[:, h] > 0
        idx = np.flatnonzero(obs)
        if idx.size < 2:
            continue
        shuffled = rng.permutation(idx)
        out[idx, h] = y[shuffled, h]
    return out


def select_rank(
    y: np.ndarray,
    weights: np.ndarray,
    mu: np.ndarray,
    n_perms: int,
    rng: Optional[np.random.Generator] = None,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Rank = # singular values above the 95% quantile of the null top SV."""
    rng = rng or np.random.default_rng(0)
    mat = _weighted_residual_matrix(y, mu, weights)
    _, s_obs, _ = np.linalg.svd(mat, full_matrices=False)
    tops: list[float] = []
    for _ in range(int(n_perms)):
        y_p = permute_columns(y, weights, rng)
        mu_p = weighted_mean(y_p, weights)
        mat_p = _weighted_residual_matrix(y_p, mu_p, weights)
        sv = np.linalg.svd(mat_p, compute_uv=False)
        tops.append(float(sv[0]) if sv.size else 0.0)
    null = np.asarray(tops, dtype=np.float64)
    thresh = float(np.quantile(null, 0.95)) if null.size else np.inf
    rank = int(np.sum(s_obs > thresh))
    return rank, s_obs, null


def weighted_als(
    y: np.ndarray,
    weights: np.ndarray,
    mu: np.ndarray,
    rank: int,
    ridge: float = 1e-6,
    n_iter: int = 25,
) -> tuple[np.ndarray, np.ndarray]:
    """Missing-data weighted ALS for scores ``Z`` (N×r) and loadings ``L`` (r×H)."""
    r = int(rank)
    n_data, n_h = y.shape
    if r <= 0:
        return np.zeros((n_data, 0), dtype=np.float64), np.zeros((0, n_h), dtype=np.float64)
    residual = np.asarray(y, dtype=np.float64) - mu[None, :]
    w = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    mat = residual * np.sqrt(w)
    # SVD init (missing treated as 0 after weighting).
    u, s, vt = np.linalg.svd(mat, full_matrices=False)
    z = u[:, :r] * np.sqrt(s[:r])[None, :]
    l = (s[:r, None] ** 0.5) * vt[:r, :]
    gls = PointMassGLS()
    for _ in range(int(n_iter)):
        # Loadings per reflection: L_h = (Z^T W_h Z + λI)^{-1} Z^T W_h r_h
        new_l = np.zeros((r, n_h), dtype=np.float64)
        for h in range(n_h):
            wh = w[:, h]
            if float(wh.sum()) <= 0:
                continue
            zw = z.T * wh[None, :]
            gram = zw @ z + ridge * np.eye(r)
            rhs = zw @ residual[:, h]
            try:
                new_l[:, h] = np.linalg.solve(gram, rhs)
            except np.linalg.LinAlgError:
                new_l[:, h] = np.linalg.lstsq(gram, rhs, rcond=None)[0]
        l = new_l
        new_z = np.zeros((n_data, r), dtype=np.float64)
        for d in range(n_data):
            new_z[d] = gls.estimate(residual[d], w[d], l, ridge=ridge)
        z = new_z
        z, l = _orthogonalize(z, l)
    return z, l


def _orthogonalize(z: np.ndarray, l: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormalize scores; order factors by explained variance."""
    if z.size == 0:
        return z, l
    # QR on scores, absorb R into loadings, then SVD of the product.
    q, r = np.linalg.qr(z, mode="reduced")
    prod = r @ l
    u, s, vt = np.linalg.svd(prod, full_matrices=False)
    z_o = q @ u
    l_o = (s[:, None] * vt) if s.size else vt
    # Order already descending from SVD.
    # Scale scores to unit variance when possible.
    std = np.sqrt(np.mean(z_o**2, axis=0))
    std = np.where(std > 1e-12, std, 1.0)
    z_o = z_o / std[None, :]
    l_o = l_o * std[:, None]
    return z_o, l_o


def scale_leakage(
    log_alpha: np.ndarray,
    scores: np.ndarray,
    observed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Regress per-dataset mean ``log α`` on scores. Significant slope → leaked scale."""
    z = np.asarray(scores, dtype=np.float64)
    n_data = z.shape[0]
    rank = z.shape[1] if z.ndim == 2 else 0
    la = np.asarray(log_alpha, dtype=np.float64)
    obs = np.asarray(observed, dtype=bool)
    y = np.zeros(n_data, dtype=np.float64)
    for d in range(n_data):
        sel = obs[d] & np.isfinite(la[d])
        y[d] = float(np.mean(la[d, sel])) if sel.any() else 0.0
    if rank == 0 or n_data <= rank + 1:
        return np.zeros(rank, dtype=np.float64), np.zeros(rank, dtype=bool)
    x = np.concatenate([np.ones((n_data, 1)), z], axis=1)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    slopes = coef[1:]
    pred = x @ coef
    resid = y - pred
    dof = max(n_data - x.shape[1], 1)
    sigma2 = float(np.sum(resid**2) / dof)
    try:
        cov = sigma2 * np.linalg.inv(x.T @ x)
        se = np.sqrt(np.maximum(np.diag(cov)[1:], 1e-18))
    except np.linalg.LinAlgError:
        se = np.full(rank, np.inf)
    tstat = slopes / se
    significant = np.abs(tstat) > 2.0
    return slopes, significant


def apply_scale_leakage_to_y(
    y: np.ndarray,
    e2_o: np.ndarray,
    sigma_a: np.ndarray,
    scores: np.ndarray,
    slopes: np.ndarray,
) -> np.ndarray:
    """Add leaked ``d log α / dz`` back into ``y`` (see plan: undo α absorbing a mode)."""
    if scores.size == 0 or slopes.size == 0:
        return y
    leaked = scores @ np.asarray(slopes, dtype=np.float64)
    # ∂y / ∂log α ≈ -2 E²_o σ_A  → add it back
    corr = 2.0 * np.asarray(e2_o, dtype=np.float64) * np.asarray(sigma_a, dtype=np.float64) * leaked[:, None]
    return y + corr


def fit_factors(
    y: np.ndarray,
    w_model: np.ndarray,
    rank: Union[int, str],
    rank_perms: int,
    ridge: float = 1e-6,
    rng: Optional[np.random.Generator] = None,
) -> FactorFit:
    mu = weighted_mean(y, w_model)
    auto_rank, svals, null = select_rank(y, w_model, mu, rank_perms, rng=rng)
    if rank == "auto":
        use = auto_rank
    else:
        use = int(rank)
    use = max(0, min(use, min(y.shape) - 1 if min(y.shape) else 0))
    z, l = weighted_als(y, w_model, mu, use, ridge=ridge)
    return FactorFit(
        mu=mu,
        scores=z,
        loadings=l,
        singular_values=svals,
        null_top=null,
        rank=use,
        scale_leakage_slopes=np.zeros(use, dtype=np.float64),
        scale_leakage_significant=np.zeros(use, dtype=bool),
    )


def leave_one_out_factors(
    y: np.ndarray,
    w_model: np.ndarray,
    drop: int,
    rank: int,
    ridge: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Refit ``μ`` and ``L`` without dataset ``drop``; GLS ``z_d`` from that dataset."""
    keep = np.ones(y.shape[0], dtype=bool)
    keep[drop] = False
    y_k = y[keep]
    w_k = w_model[keep]
    mu = weighted_mean(y_k, w_k)
    z, l = weighted_als(y_k, w_k, mu, rank, ridge=ridge)
    gls = PointMassGLS()
    z_d = gls.estimate(y[drop] - mu, w_model[drop], l, ridge=ridge)
    return mu, l, z_d
