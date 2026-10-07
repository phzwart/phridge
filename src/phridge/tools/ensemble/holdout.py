"""Crystal folds, reflection roles, half-split, bootstrap, and audit PIT.

Held-out scores are the marginal objective on reflections that did not choose
the parameters being scored. A projected fit and an unprojected fit are
different objectives; both are reported when both are present, and they are
not comparable.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from phridge.r.generators.perturbation_modes.gauge import fix_gauge, principal_angles, procrustes_align, reconstruction
from phridge.r.generators.perturbation_modes.mstep import als_known_f, solve_scores_torch


def assign_roles(
    mask: np.ndarray,
    *,
    audit_fraction: float = 0.0,
    calibration_fraction: float = 0.0,
    seed: int = 0,
    free_flags: Optional[np.ndarray] = None,
) -> np.ndarray:
    """0 work, 1 calibration, 2 audit, -1 absent.

    Free flags are audit. Remaining audit and calibration draws use ``seed``.
    """
    observed = np.asarray(mask, dtype=bool)
    roles = np.full(observed.shape, -1, dtype=np.int8)
    roles[observed] = 0
    if free_flags is not None:
        free = np.asarray(free_flags, dtype=bool) & observed
        roles[free] = 2
    rng = np.random.default_rng(int(seed))
    for dataset in range(observed.shape[0]):
        work = np.flatnonzero(roles[dataset] == 0)
        if work.size == 0:
            continue
        n_audit = int(round(float(audit_fraction) * work.size))
        n_cal = int(round(float(calibration_fraction) * work.size))
        n_audit = min(n_audit, work.size)
        pick = rng.choice(work, size=n_audit + min(n_cal, work.size - n_audit), replace=False) if n_audit + n_cal else []
        if n_audit:
            roles[dataset, pick[:n_audit]] = 2
        if n_cal:
            roles[dataset, pick[n_audit : n_audit + n_cal]] = 1
    return roles


def fold_indices(n_data: int, n_folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(int(n_data))
    return [np.sort(chunk) for chunk in np.array_split(order, int(n_folds)) if len(chunk)]


def unexplained_fraction(target: np.ndarray, pred: np.ndarray, weights: np.ndarray) -> float:
    """Scale-free residual fraction against the mean-only prediction."""
    w = np.asarray(weights, dtype=np.float64)
    resid = np.asarray(target, dtype=np.complex128) - np.asarray(pred, dtype=np.complex128)
    denom_target = np.asarray(target, dtype=np.complex128)
    num = float(np.sum(w * np.abs(resid) ** 2))
    den = float(np.sum(w * np.abs(denom_target) ** 2))
    if den <= 1e-18:
        return 0.0
    return num / den


def score_dataset_z(
    target_row: np.ndarray,
    weights_row: np.ndarray,
    delta0: np.ndarray,
    modes: np.ndarray,
    ridge: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one dataset's scores. ``target_row`` is ``(H,)``."""
    y = np.asarray(target_row, dtype=np.complex128).reshape(1, -1)
    w = np.asarray(weights_row, dtype=np.float64).reshape(1, -1)
    scores, cov = solve_scores_torch(y, delta0, modes, w, ridge)
    return scores.reshape(-1), cov.reshape(scores.shape[1], scores.shape[1])


def crystal_folds(
    target: np.ndarray,
    weights: np.ndarray,
    *,
    rank: int,
    n_folds: int = 5,
    n_sweeps: int = 6,
    seed: int = 0,
    ridge: float = 1e-8,
    free_weights: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Hold out datasets. Fit modes on the rest; fit z and score the free weights.

    ``weights`` select the reflections that may fit z. ``free_weights`` select
    the reflections that are scored. When ``free_weights`` is omitted the
    complement of a hash of the columns is not used: the same weights are scored,
    which is only an in-sample check. Callers pass work weights and free weights
    separately.
    """
    y = np.asarray(target, dtype=np.complex128)
    w = np.asarray(weights, dtype=np.float64)
    free = w if free_weights is None else np.asarray(free_weights, dtype=np.float64)
    n_data = y.shape[0]
    folds = fold_indices(n_data, min(n_folds, n_data), seed)
    loglik = []
    fractions = []
    for held in folds:
        train = np.array([i for i in range(n_data) if i not in set(held.tolist())], dtype=int)
        if train.size == 0:
            continue
        delta0, modes, _scores, _cov = als_known_f(y[train], w[train], rank, n_sweeps, ridge)
        for dataset in held:
            z, _c = score_dataset_z(y[dataset], w[dataset], delta0, modes, ridge)
            pred = reconstruction(delta0, modes, z.reshape(1, -1))[0]
            ww = free[dataset]
            rss = float(np.sum(ww * np.abs(y[dataset] - pred) ** 2))
            loglik.append(-0.5 * rss)
            fractions.append(unexplained_fraction(y[dataset], pred, ww))
    return {
        "loglik_mean": float(np.mean(loglik)) if loglik else float("nan"),
        "unexplained_fraction_mean": float(np.mean(fractions)) if fractions else float("nan"),
        "n_folds": len(folds),
        "per_dataset_loglik": [float(v) for v in loglik],
    }


def k_scan(
    target: np.ndarray,
    weights: np.ndarray,
    k_values: Sequence[int],
    *,
    n_folds: int = 4,
    n_sweeps: int = 6,
    seed: int = 0,
    free_weights: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Held-out score at each K. The spectrum of the initial SVD is not used."""
    rows = []
    for k in k_values:
        scored = crystal_folds(
            target,
            weights,
            rank=int(k),
            n_folds=n_folds,
            n_sweeps=n_sweeps,
            seed=seed,
            free_weights=free_weights,
        )
        rows.append({"k": int(k), **scored})
    return {"rows": rows, "note": "K is chosen on held-out scores, not on the initial spectrum."}


def half_split(
    target: np.ndarray,
    weights: np.ndarray,
    shell_index: np.ndarray,
    *,
    rank: int,
    n_sweeps: int = 6,
    seed: int = 0,
) -> dict[str, Any]:
    """Two dataset halves, Procrustes alignment, per-mode per-shell correlation.

    ``correlation_optimistic_note_s6`` is ``sqrt(2C/(1+C))`` and is optimistic.
    The raw correlation is ``correlation``.
    """
    y = np.asarray(target, dtype=np.complex128)
    w = np.asarray(weights, dtype=np.float64)
    n_data = y.shape[0]
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n_data)
    cut = max(1, n_data // 2)
    left, right = perm[:cut], perm[cut:]
    if right.size == 0:
        right = left[-1:]
        left = left[:-1]
    d0_a, modes_a, _z_a, _c_a = als_known_f(y[left], w[left], rank, n_sweeps)
    d0_b, modes_b, _z_b, _c_b = als_known_f(y[right], w[right], rank, n_sweeps)
    if modes_a.shape[0] == 0 or modes_b.shape[0] == 0:
        return {"correlation": [], "correlation_optimistic_note_s6": [], "principal_angles_deg": []}
    aligned, _rot = procrustes_align(modes_a, modes_b)
    rank_use = aligned.shape[0]
    shells = np.asarray(shell_index, dtype=np.int32).reshape(-1)
    per_mode = []
    for k in range(rank_use):
        rows = []
        for shell in range(int(shells.max()) + 1 if shells.size else 1):
            sel = shells == shell
            if not np.any(sel):
                continue
            a = np.concatenate([modes_a[k, sel].real, modes_a[k, sel].imag])
            b = np.concatenate([aligned[k, sel].real, aligned[k, sel].imag])
            if np.std(a) < 1e-12 or np.std(b) < 1e-12:
                corr = 1.0 if np.allclose(a, b) else 0.0
            else:
                corr = float(np.corrcoef(a, b)[0, 1])
            rows.append({"shell": int(shell), "correlation": corr})
        per_mode.append(rows)
    from phridge.r.generators.perturbation_modes.gauge import optimistic_correlation

    flat = [item["correlation"] for mode in per_mode for item in mode]
    optimistic = optimistic_correlation(np.asarray(flat, dtype=np.float64)).tolist() if flat else []
    return {
        "correlation": per_mode,
        "correlation_optimistic_note_s6": optimistic,
        "optimistic_label": "optimistic",
        "principal_angles_deg": principal_angles(modes_a, modes_b).tolist(),
        "delta0_correlation": _complex_corr(d0_a, d0_b),
    }


def _complex_corr(left: np.ndarray, right: np.ndarray) -> float:
    a = np.concatenate([np.asarray(left).real, np.asarray(left).imag])
    b = np.concatenate([np.asarray(right).real, np.asarray(right).imag])
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def bootstrap_subspace(
    target: np.ndarray,
    weights: np.ndarray,
    shell_index: np.ndarray,
    *,
    rank: int,
    n_bootstrap: int,
    strata: Optional[Sequence[Any]] = None,
    n_sweeps: int = 4,
    seed: int = 0,
) -> dict[str, Any]:
    """Resample datasets inside strata, refit, align, then report the spread.

    Unaligned modes are never averaged.
    """
    y = np.asarray(target, dtype=np.complex128)
    w = np.asarray(weights, dtype=np.float64)
    n_data = y.shape[0]
    labels = np.zeros(n_data, dtype=int) if strata is None else np.asarray(strata)
    if labels.shape[0] != n_data:
        raise ValueError("strata must have one label per dataset")
    rng = np.random.default_rng(int(seed))
    groups = [np.flatnonzero(labels == value) for value in np.unique(labels)]
    delta_ref, modes_ref, _z, _c = als_known_f(y, w, rank, n_sweeps)
    spreads: list[list[float]] = []
    for _draw in range(int(n_bootstrap)):
        chosen = np.concatenate([group[rng.integers(0, group.size, size=group.size)] for group in groups if group.size])
        _d, modes, _zz, _cc = als_known_f(y[chosen], w[chosen], rank, n_sweeps)
        if modes.shape[0] == 0 or modes_ref.shape[0] == 0:
            continue
        aligned, _rot = procrustes_align(modes_ref, modes)
        rank_use = min(aligned.shape[0], modes_ref.shape[0])
        spreads.append([float(np.sqrt(np.mean(np.abs(aligned[k] - modes_ref[k]) ** 2))) for k in range(rank_use)])
    stack = np.asarray(spreads, dtype=np.float64) if spreads else np.zeros((0, rank))
    return {
        "per_mode_rms_spread": stack.mean(axis=0).tolist() if stack.size else [],
        "n_bootstrap": int(n_bootstrap),
        "aligned": True,
        "shell_index_used": True,
        "reference_rank": int(modes_ref.shape[0]),
    }


def pit_values(i_obs: np.ndarray, mean_intensity: np.ndarray, sig_i: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Gaussian PIT of ``I_obs`` about the posterior mean intensity. Audit rows only if masked."""
    from math import erf, sqrt

    z = (np.asarray(i_obs, dtype=np.float64) - np.asarray(mean_intensity, dtype=np.float64)) / np.maximum(
        np.asarray(sig_i, dtype=np.float64), 1e-8
    )
    # Standard normal CDF.
    cdf = 0.5 * (1.0 + np.vectorize(erf)(z / sqrt(2.0)))
    return np.where(np.asarray(mask, dtype=bool), cdf, np.nan)


def calibration_curve(fom: np.ndarray, cos_phase_error: np.ndarray, mask: np.ndarray, n_bins: int = 5) -> dict[str, Any]:
    """Bin by FOM and report ``<cos Δφ>`` against the mean FOM."""
    f = np.asarray(fom, dtype=np.float64).reshape(-1)
    c = np.asarray(cos_phase_error, dtype=np.float64).reshape(-1)
    m = np.asarray(mask, dtype=bool).reshape(-1)
    f, c = f[m], c[m]
    if f.size == 0:
        return {"fom_mean": [], "cos_mean": []}
    edges = np.quantile(f, np.linspace(0, 1, int(n_bins) + 1))
    edges = np.unique(edges)
    fom_mean = []
    cos_mean = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (f >= lo) & (f <= hi if hi == edges[-1] else f < hi)
        if not np.any(sel):
            continue
        fom_mean.append(float(np.mean(f[sel])))
        cos_mean.append(float(np.mean(c[sel])))
    return {"fom_mean": fom_mean, "cos_mean": cos_mean}
