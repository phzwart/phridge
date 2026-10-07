"""Zero-dose baselines the damage model is scored against.

(i) Independent per-``h`` linear extrapolation of intensity to ``D = 0``.
(ii) Reject batches whose scale/B-corrected residual exceeds ``k`` times the
minimum, then run phase 1 on the surviving observations' crystal means.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class BaselineResult:
    linear_i0: np.ndarray
    linear_se: np.ndarray
    kept_batch: np.ndarray
    rejected_batch: np.ndarray
    reject_residual: np.ndarray
    stats: dict = field(default_factory=dict)


def linear_zero_dose(
    i: np.ndarray,
    dose: np.ndarray,
    weights: np.ndarray,
    obs_h: np.ndarray,
    n_h: int,
    crystal: Optional[np.ndarray] = None,
    n_crystal: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-reflection (and optional per-crystal) ``I = a + b D``; return ``a``."""
    i = np.asarray(i, dtype=np.float64).reshape(-1)
    d = np.asarray(dose, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    h = np.asarray(obs_h, dtype=np.int64).reshape(-1)
    if crystal is None:
        crystal = np.zeros(i.size, dtype=np.int64)
    c = np.asarray(crystal, dtype=np.int64).reshape(-1)
    i0 = np.full((n_crystal, n_h), np.nan, dtype=np.float64)
    se = np.full((n_crystal, n_h), np.nan, dtype=np.float64)
    for di in range(n_crystal):
        for hi in range(n_h):
            sel = (c == di) & (h == hi) & (w > 0) & np.isfinite(i) & np.isfinite(d)
            n = int(sel.sum())
            if n < 2:
                if n == 1:
                    i0[di, hi] = float(i[sel][0])
                continue
            x = np.stack([np.ones(n), d[sel]], axis=1)
            ww = w[sel]
            xw = x * ww[:, None]
            gram = xw.T @ x
            try:
                coef = np.linalg.solve(gram + 1e-10 * np.eye(2), xw.T @ i[sel])
                cov = np.linalg.inv(gram + 1e-10 * np.eye(2))
            except np.linalg.LinAlgError:
                coef, *_ = np.linalg.lstsq(x * np.sqrt(ww)[:, None], i[sel] * np.sqrt(ww), rcond=None)
                cov = np.eye(2)
            i0[di, hi] = float(coef[0])
            se[di, hi] = float(np.sqrt(max(cov[0, 0], 0.0)))
    return i0, se


def reject_bad_batches(
    residual: np.ndarray,
    batch: np.ndarray,
    k: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batches whose mean |residual| exceeds ``k ×`` the minimum are rejected.

    Does **not** auto-reject frames in the main fit; this is a baseline only.
    Returns ``(kept_mask, rejected_ids, per_batch_residual)``.
    """
    r = np.abs(np.asarray(residual, dtype=np.float64).reshape(-1))
    b = np.asarray(batch).reshape(-1)
    ids = np.unique(b)
    means = np.zeros(ids.size, dtype=np.float64)
    for i, bid in enumerate(ids):
        sel = b == bid
        means[i] = float(np.nanmean(r[sel])) if np.any(sel) else np.inf
    finite = means[np.isfinite(means)]
    floor = float(np.min(finite)) if finite.size else 0.0
    cut = float(k) * max(floor, 1e-12)
    rejected = ids[means > cut]
    kept = ~np.isin(b, rejected)
    return kept, np.asarray(rejected), means


def run_baselines(
    i: np.ndarray,
    dose: np.ndarray,
    weights: np.ndarray,
    obs_h: np.ndarray,
    n_h: int,
    batch: np.ndarray,
    residual: np.ndarray,
    crystal: Optional[np.ndarray] = None,
    n_crystal: int = 1,
    frame_reject_k: float = 3.0,
) -> BaselineResult:
    i0, se = linear_zero_dose(i, dose, weights, obs_h, n_h, crystal=crystal, n_crystal=n_crystal)
    kept, rejected, means = reject_bad_batches(residual, batch, k=frame_reject_k)
    return BaselineResult(
        linear_i0=i0,
        linear_se=se,
        kept_batch=kept,
        rejected_batch=rejected,
        reject_residual=means,
        stats={
            "n_rejected_batches": int(rejected.size),
            "n_kept_obs": int(kept.sum()),
        },
    )
