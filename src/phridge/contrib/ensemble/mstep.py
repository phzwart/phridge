"""Closed-form M-step: batched mode solve, per-dataset scores, shell scales.

The design is real, ``W = [1 | z]``. Weights are ``mask * D² / σ²_Δ``. Missing
reflections have weight 0 and are never imputed. The score covariance is the
inverse of the real Gram matrix.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from phridge.contrib.ensemble.gauge import absorb_score_mean, fix_gauge


def initialize_from_residuals(
    target: np.ndarray,
    weights: np.ndarray,
    rank: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted mean Δ0 plus the leading singular vectors of the residual embedding.

    ``target`` is ``(N, H)`` complex. ``weights`` is ``(N, H)`` real.
    """
    y = np.asarray(target, dtype=np.complex128)
    w = np.asarray(weights, dtype=np.float64)
    n_data, n_h = y.shape
    rank = int(max(0, rank))
    weight_sum = w.sum(axis=0)
    delta0 = np.divide(np.sum(w * y, axis=0), weight_sum, out=np.zeros(n_h, dtype=np.complex128), where=weight_sum > 0)
    if rank == 0 or n_data < 2:
        return delta0, np.zeros((0, n_h), dtype=np.complex128), np.zeros((n_data, 0), dtype=np.float64)
    resid = (y - delta0) * (w > 0)
    embedded = np.concatenate([resid.real, resid.imag], axis=1)
    u_mat, singular, vh = np.linalg.svd(embedded, full_matrices=False)
    rank = min(rank, int(singular.shape[0]), n_data - 1, n_h)
    positive = singular[:rank] > max(float(singular[0]) * 1e-10, 0.0) if rank else np.zeros(0, dtype=bool)
    rank = int(np.count_nonzero(positive)) if rank else 0
    if rank == 0:
        return delta0, np.zeros((0, n_h), dtype=np.complex128), np.zeros((n_data, 0), dtype=np.float64)
    scores = u_mat[:, :rank] * singular[:rank]
    modes = vh[:rank, :n_h] + 1j * vh[:rank, n_h:]
    return absorb_score_mean(delta0, modes, scores)


def solve_modes_torch(target: np.ndarray, scores: np.ndarray, weights: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    """Batched ``(K+1)`` solve over reflections. Right-hand side is complex."""
    import torch

    y = torch.as_tensor(np.asarray(target, dtype=np.complex128))
    z = torch.as_tensor(np.asarray(scores, dtype=np.float64))
    w = torch.as_tensor(np.asarray(weights, dtype=np.float64))
    n_data, n_h = y.shape
    if z.ndim == 1:
        z = z.reshape(n_data, -1)
    rank = int(z.shape[1])
    ones = torch.ones((n_data, 1), dtype=torch.float64)
    design = torch.cat([ones, z], dim=1)
    gram = torch.einsum("ni,nj,nh->hij", design, design, w)
    eye = torch.eye(rank + 1, dtype=torch.float64)
    gram = gram + float(ridge) * eye
    rhs = torch.einsum("ni,nh,nh->hi", design.to(y.dtype), w.to(y.dtype), y)
    beta = torch.linalg.solve(gram.to(y.dtype), rhs)
    delta0 = beta[:, 0].detach().cpu().numpy().astype(np.complex128)
    if rank == 0:
        modes = np.zeros((0, n_h), dtype=np.complex128)
    else:
        modes = beta[:, 1:].transpose(0, 1).detach().cpu().numpy().astype(np.complex128)
    dead = w.sum(dim=0).detach().cpu().numpy() <= 0
    delta0 = delta0.copy()
    delta0[dead] = 0
    if modes.size:
        modes = modes.copy()
        modes[:, dead] = 0
    return delta0, modes


def solve_scores_torch(
    target: np.ndarray,
    delta0: np.ndarray,
    modes: np.ndarray,
    weights: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-dataset real solve. Covariance is ``(Re Δ̄ Ω Δᵀ + ridge I)⁻¹``."""
    import torch

    y = torch.as_tensor(np.asarray(target, dtype=np.complex128))
    delta = torch.as_tensor(np.asarray(delta0, dtype=np.complex128))
    loads = torch.as_tensor(np.asarray(modes, dtype=np.complex128))
    w = torch.as_tensor(np.asarray(weights, dtype=np.float64))
    n_data = int(y.shape[0])
    if loads.ndim == 1:
        loads = loads.reshape(1, -1)
    rank = int(loads.shape[0])
    if rank == 0:
        return np.zeros((n_data, 0), dtype=np.float64), np.zeros((n_data, 0, 0), dtype=np.float64)
    resid = y - delta
    weighted = loads.unsqueeze(0) * w.to(loads.dtype).unsqueeze(1)
    gram = torch.einsum("kh,njh->nkj", loads.conj(), weighted).real
    eye = torch.eye(rank, dtype=torch.float64)
    gram = gram + float(ridge) * eye
    rhs = torch.einsum("kh,nh,nh->nk", loads.conj(), w.to(loads.dtype), resid).real
    scores = torch.linalg.solve(gram, rhs)
    cov = torch.linalg.inv(gram)
    return (
        scores.detach().cpu().numpy().astype(np.float64),
        cov.detach().cpu().numpy().astype(np.float64),
    )


def update_shell_scales(
    mean_f: np.ndarray,
    mean_intensity: np.ndarray,
    ensemble: np.ndarray,
    f_var: np.ndarray,
    shell_index: np.ndarray,
    mask: np.ndarray,
    n_data: int,
    rank: int,
    *,
    dof_correct: bool = True,
    sigma2_floor: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Shell-wise real D and σ²_Δ from latent first and second moments.

    σ² is the mean complete-data residual, times ``N/(N-K-1)`` when
    ``dof_correct`` is set. The factor is applied once to that sweep's
    residual; it is not compounded inside the residual itself.
    """
    m1 = np.asarray(mean_f, dtype=np.complex128)
    m2 = np.asarray(mean_intensity, dtype=np.float64)
    ens = np.asarray(ensemble, dtype=np.complex128)
    var = np.asarray(f_var, dtype=np.complex128)
    shells = np.asarray(shell_index, dtype=np.int32).reshape(-1)
    observed = np.asarray(mask, dtype=bool)
    n_shell = int(shells.max()) + 1 if shells.size and shells.max() >= 0 else 1
    shells = np.where(shells < 0, 0, shells)
    d_out = np.ones((m1.shape[0], n_shell), dtype=np.float64)
    s_out = np.ones((m1.shape[0], n_shell), dtype=np.float64)
    factor = 1.0
    if dof_correct:
        factor = float(n_data) / float(max(n_data - int(rank) - 1, 1))
    for dataset in range(m1.shape[0]):
        for shell in range(n_shell):
            sel = observed[dataset] & (shells == shell)
            if not np.any(sel):
                continue
            ens_h = ens[dataset, sel]
            target = m1[dataset, sel] - var[dataset, sel]
            denom = float(np.sum(np.abs(ens_h) ** 2))
            if denom <= 1e-18:
                d_hat = 1.0
            else:
                d_hat = float(np.real(np.vdot(ens_h, target)) / denom)
            d_hat = float(np.clip(d_hat, 1e-3, 5.0))
            pred = var[dataset, sel] + d_hat * ens_h
            resid = m2[dataset, sel] - 2.0 * np.real(np.conj(pred) * m1[dataset, sel]) + np.abs(pred) ** 2
            sigma2 = float(np.mean(np.maximum(resid, 0.0))) * factor
            d_out[dataset, shell] = d_hat
            s_out[dataset, shell] = max(sigma2, sigma2_floor)
    return d_out, s_out


def expand_shells(values: np.ndarray, shell_index: np.ndarray) -> np.ndarray:
    """Broadcast ``(N, n_shell)`` onto reflections."""
    shells = np.asarray(shell_index, dtype=np.int32).reshape(-1)
    shells = np.where(shells < 0, 0, shells)
    table = np.asarray(values, dtype=np.float64)
    if table.ndim == 1:
        return table[shells]
    return table[:, shells]


def als_known_f(
    target: np.ndarray,
    weights: np.ndarray,
    rank: int,
    n_iter: int,
    ridge: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Weighted alternating least squares. Returns delta0, modes, scores, covariance."""
    y = np.asarray(target, dtype=np.complex128)
    w = np.asarray(weights, dtype=np.float64)
    delta0, modes, scores = initialize_from_residuals(y, w, rank)
    cov = np.zeros((y.shape[0], modes.shape[0], modes.shape[0]), dtype=np.float64)
    for _sweep in range(int(n_iter)):
        delta0, modes = solve_modes_torch(y, scores, w, ridge)
        scores, cov = solve_scores_torch(y, delta0, modes, w, ridge)
        delta0, modes, scores = fix_gauge(delta0, modes, scores)
    return delta0, modes, scores, cov


def work_weights(
    mask: np.ndarray,
    d_refl: np.ndarray,
    sigma2_refl: np.ndarray,
    roles: Optional[np.ndarray],
    *,
    include_calibration: bool,
) -> np.ndarray:
    """Zero weight on absent, audit, and (unless requested) calibration reflections."""
    w = np.asarray(mask, dtype=np.float64) * (np.asarray(d_refl, dtype=np.float64) ** 2)
    w = w / np.maximum(np.asarray(sigma2_refl, dtype=np.float64), 1e-12)
    if roles is None:
        return w
    role = np.asarray(roles)
    if include_calibration:
        keep = (role == 0) | (role == 1)
    else:
        keep = role == 0
    return w * keep
