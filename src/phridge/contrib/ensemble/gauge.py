"""Gauge fixing, half-set inner product, Procrustes, and principal angles.

Orthonormality is a gauge: it does not change Δ0 + Σ z_k Δ_k. Locality,
smoothness, and positivity are constraints and live in ``projection.py``.
"""

from __future__ import annotations

import numpy as np


def half_set_inner(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Real inner product ``Re Σ_h conj(left) right`` on a Friedel half-set.

    ``left`` and ``right`` are complex arrays whose last axis is the reflection
    index. The result has that axis removed.
    """
    a = np.asarray(left, dtype=np.complex128)
    b = np.asarray(right, dtype=np.complex128)
    return np.real(np.sum(np.conj(a) * b, axis=-1))


def real_embedding(modes: np.ndarray) -> np.ndarray:
    """Stack ``[Re Δ | Im Δ]`` along the reflection axis. Shape ``(K, 2H)``."""
    modes = np.asarray(modes, dtype=np.complex128)
    if modes.ndim == 1:
        modes = modes.reshape(1, -1)
    return np.concatenate([modes.real, modes.imag], axis=1)


def _as_modes(array: np.ndarray) -> np.ndarray:
    modes = np.asarray(array, dtype=np.complex128)
    if modes.ndim == 1:
        modes = modes.reshape(1, -1)
    return modes


def absorb_score_mean(
    delta0: np.ndarray,
    modes: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Move the mean of each score column into Δ0. Σ_n z_nk = 0 afterwards."""
    delta0 = np.asarray(delta0, dtype=np.complex128).reshape(-1)
    modes = _as_modes(modes)
    scores = np.asarray(scores, dtype=np.float64)
    if modes.shape[0] == 0 or scores.size == 0:
        return delta0, modes.reshape(0, delta0.shape[0]), np.zeros((scores.shape[0], 0), dtype=np.float64)
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)
    mean = scores.mean(axis=0)
    delta0 = delta0 + mean.astype(np.complex128) @ modes
    scores = scores - mean
    return delta0, modes, scores


def fix_gauge(
    delta0: np.ndarray,
    modes: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Thin SVD of the real embedding of Σ z_k Δ_k.

    Modes become orthonormal under :func:`half_set_inner`. Score columns are
    uncorrelated with descending variances. Δ0 absorbs the score mean. The
    field Δ0 + Σ z_k Δ_k is unchanged.
    """
    delta0 = np.asarray(delta0, dtype=np.complex128).reshape(-1)
    modes = _as_modes(modes)
    scores = np.asarray(scores, dtype=np.float64)
    n_mode, n_h = modes.shape
    n_data = int(scores.shape[0]) if scores.ndim == 2 else int(scores.shape[0] if scores.ndim else 0)
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)
    if n_mode == 0 or n_data == 0:
        empty = np.zeros((n_data, 0), dtype=np.float64)
        return delta0, np.zeros((0, n_h), dtype=np.complex128), empty
    delta0, modes, scores = absorb_score_mean(delta0, modes, scores)
    embedded = real_embedding(modes)
    product = scores @ embedded
    _u, singular, vh = np.linalg.svd(product, full_matrices=False)
    rank = int(min(n_mode, singular.shape[0], vh.shape[0]))
    used = singular[:rank] > max(float(singular[0]) * 1e-12, 0.0) if rank else []
    rank = int(np.count_nonzero(used)) if rank else 0
    if rank == 0:
        return delta0, np.zeros((0, n_h), dtype=np.complex128), np.zeros((n_data, 0), dtype=np.float64)
    scores_new = _u[:, :rank] * singular[:rank]
    embedded_new = vh[:rank, :]
    modes_new = embedded_new[:, :n_h] + 1j * embedded_new[:, n_h:]
    # Row-centred product: nonzero left singular vectors stay mean-free.
    mean = scores_new.mean(axis=0)
    if float(np.max(np.abs(mean))) > 0.0:
        delta0 = delta0 + mean.astype(np.complex128) @ modes_new
        scores_new = scores_new - mean
    return delta0, modes_new.astype(np.complex128), scores_new.astype(np.float64)


def reconstruction(delta0: np.ndarray, modes: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Δ0 + Σ z_k Δ_k, shape ``(N, H)``."""
    delta0 = np.asarray(delta0, dtype=np.complex128).reshape(-1)
    modes = _as_modes(modes)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 1:
        scores = scores.reshape(1, -1)
    if modes.shape[0] == 0:
        return np.broadcast_to(delta0, (scores.shape[0], delta0.shape[0])).copy()
    return delta0 + scores.astype(np.complex128) @ modes


def procrustes_align(reference: np.ndarray, modes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orthogonal map taking ``modes`` onto ``reference``.

    Returns ``(aligned_modes, rotation)`` with ``aligned = rotation @ modes``.
    """
    ref = real_embedding(reference)
    tgt = real_embedding(modes)
    rank = min(ref.shape[0], tgt.shape[0])
    cross = ref[:rank] @ tgt[:rank].T
    u_mat, _s, vh = np.linalg.svd(cross, full_matrices=False)
    rotation = u_mat @ vh
    if np.linalg.det(rotation) < 0 and rotation.shape[0] > 0:
        u_mat = u_mat.copy()
        u_mat[:, -1] *= -1
        rotation = u_mat @ vh
    aligned = rotation @ _as_modes(modes)[:rank]
    return aligned.astype(np.complex128), rotation.astype(np.float64)


def principal_angles(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Principal angles in degrees between the real embeddings, ascending."""
    a = real_embedding(left)
    b = real_embedding(right)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    qa, _ = np.linalg.qr(a.T)
    qb, _ = np.linalg.qr(b.T)
    rank = min(qa.shape[1], qb.shape[1])
    singular = np.linalg.svd(qa[:, :rank].T @ qb[:, :rank], compute_uv=False)
    singular = np.clip(singular, 0.0, 1.0)
    return np.degrees(np.arccos(singular)).astype(np.float64)


def optimistic_correlation(correlation: np.ndarray) -> np.ndarray:
    """``sqrt(2C/(1+C))``. The note §6 calls this conversion optimistic."""
    c = np.clip(np.asarray(correlation, dtype=np.float64), -0.999, 0.999)
    return np.sqrt(2.0 * c / (1.0 + c))
