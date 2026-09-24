"""Dose attachment, collection order, and dose–angle confound. Torch-free.

Collection order is the BATCH/serial order **across sweeps**, not the image
number within a sweep. Relative dose is ``rho_d * dose`` with one crystal fixed
at ``rho = 1``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


@dataclass
class CrystalDoseReport:
    crystal: int
    dose_min: float
    dose_max: float
    n_passes: int
    dose_angle_corr: float
    n_obs: int
    confound: bool
    dose_diversity: np.ndarray


def load_dose_table(path: str | Path) -> dict[tuple[str, int], float]:
    """CSV columns: dataset, batch|image, dose_MGy."""
    out: dict[tuple[str, int], float] = {}
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = str(row.get("dataset") or row.get("name") or row.get("file") or "").strip()
            batch_raw = row.get("batch") or row.get("image") or row.get("frame")
            dose_raw = row.get("dose_MGy") or row.get("dose") or row.get("MGy")
            if batch_raw is None or dose_raw is None:
                continue
            out[(name, int(float(batch_raw)))] = float(dose_raw)
    return out


def attach_dose(
    batch: np.ndarray,
    *,
    dataset_name: str = "",
    table: Optional[dict[tuple[str, int], float]] = None,
    dose_rate: float = 0.05,
    collection_order: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Dose per observation. Table wins; else ``dose_rate × collection_order``.

    ``collection_order`` defaults to the rank of unique BATCH values (across
    sweeps), not the raw image number.
    """
    b = np.asarray(batch, dtype=np.int64).reshape(-1)
    dose = np.full(b.shape, np.nan, dtype=np.float64)
    if table:
        for i, bid in enumerate(b):
            key = (dataset_name, int(bid))
            if key in table:
                dose[i] = table[key]
            elif ("", int(bid)) in table:
                dose[i] = table[("", int(bid))]
        if np.isfinite(dose).all():
            return dose
    if collection_order is None:
        collection_order = collection_index(b)
    order = np.asarray(collection_order, dtype=np.float64).reshape(-1)
    filled = float(dose_rate) * order
    missing = ~np.isfinite(dose)
    dose[missing] = filled[missing]
    return dose


def collection_index(batch: np.ndarray) -> np.ndarray:
    """0-based collection order of unique BATCH ids, sorted as they first appear."""
    b = np.asarray(batch, dtype=np.int64).reshape(-1)
    seen: dict[int, int] = {}
    next_i = 0
    out = np.zeros(b.size, dtype=np.int64)
    for i, bid in enumerate(b):
        key = int(bid)
        if key not in seen:
            seen[key] = next_i
            next_i += 1
        out[i] = seen[key]
    return out


def circular_linear_corr(angle: np.ndarray, linear: np.ndarray) -> float:
    """Mardia circular-linear correlation of ``angle`` (rad) with ``linear``."""
    phi = np.asarray(angle, dtype=np.float64).reshape(-1)
    x = np.asarray(linear, dtype=np.float64).reshape(-1)
    ok = np.isfinite(phi) & np.isfinite(x)
    if int(ok.sum()) < 8:
        return float("nan")
    c = np.cos(phi[ok])
    s = np.sin(phi[ok])
    y = x[ok]
    rxc = float(np.corrcoef(y, c)[0, 1])
    rxs = float(np.corrcoef(y, s)[0, 1])
    rcs = float(np.corrcoef(c, s)[0, 1])
    den = 1.0 - rcs**2
    if den <= 1e-12 or not np.isfinite(rxc + rxs):
        return float("nan")
    r2 = (rxc**2 + rxs**2 - 2.0 * rxc * rxs * rcs) / den
    return float(np.sqrt(np.clip(r2, 0.0, 1.0)))


def dose_bins(dose: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-count dose-bin ids, shape matching ``dose``."""
    d = np.asarray(dose, dtype=np.float64).reshape(-1)
    n_bins = max(1, min(int(n_bins), max(int(np.isfinite(d).sum()), 1)))
    order = np.argsort(d, kind="stable")
    ids = np.empty(d.size, dtype=np.int64)
    finite = np.isfinite(d)
    ids[:] = 0
    n = int(finite.sum())
    if n == 0:
        return ids
    ranked = np.empty(n, dtype=np.int64)
    ranked[np.argsort(d[finite], kind="stable")] = (np.arange(n) * n_bins) // n
    ids[finite] = ranked
    return ids


def dose_diversity(obs_h: np.ndarray, bins: np.ndarray, n_h: int) -> np.ndarray:
    """Number of distinct dose bins per reflection."""
    h = np.asarray(obs_h, dtype=np.int64).reshape(-1)
    b = np.asarray(bins, dtype=np.int64).reshape(-1)
    out = np.zeros(int(n_h), dtype=np.int64)
    for hi in np.unique(h):
        if 0 <= hi < n_h:
            out[int(hi)] = int(np.unique(b[h == hi]).size)
    return out


def n_passes(phi: np.ndarray, batch: np.ndarray) -> int:
    """Heuristic pass count: wraps of rotation, else unique BATCH runs."""
    ang = np.asarray(phi, dtype=np.float64).reshape(-1)
    if ang.size and np.isfinite(ang).any():
        unwrapped = np.unwrap(ang[np.isfinite(ang)])
        span = float(unwrapped.max() - unwrapped.min()) if unwrapped.size else 0.0
        return max(1, int(np.floor(span / (2.0 * np.pi) + 0.5)))
    return 1


def report_crystal(
    crystal: int,
    dose: np.ndarray,
    phi: np.ndarray,
    batch: np.ndarray,
    obs_h: np.ndarray,
    n_h: int,
    n_bins: int = 10,
    confound_corr: float = 0.85,
) -> CrystalDoseReport:
    bins = dose_bins(dose, n_bins)
    corr = circular_linear_corr(phi, dose)
    passes = n_passes(phi, batch)
    d = np.asarray(dose, dtype=np.float64)
    has_dose = bool(d.size and (np.nanmax(d) - np.nanmin(d) > 1e-8))
    # A single pass cannot separate dose from rotation; that is the confound.
    confound = bool(has_dose and (passes <= 1 or (np.isfinite(corr) and corr >= float(confound_corr))))
    return CrystalDoseReport(
        crystal=int(crystal),
        dose_min=float(np.nanmin(d)) if d.size else float("nan"),
        dose_max=float(np.nanmax(d)) if d.size else float("nan"),
        n_passes=passes,
        dose_angle_corr=corr,
        n_obs=int(d.size),
        confound=confound,
        dose_diversity=dose_diversity(obs_h, bins, n_h),
    )


def beam_frame_vectors(
    hkl: np.ndarray,
    unit_cell: Any,
    phi: np.ndarray,
    spindle: tuple[float, float, float] = (0.0, 1.0, 0.0),
    wavelength: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fallback incident/diffracted unit vectors: rotate s about the spindle by ``phi``.

    ``s_inc`` is the (constant) beam direction; ``s_dif = s_inc + λ s_rot``.
    """
    from phridge.contrib.intensity_ll.wilson import reciprocal_cartesian

    s = reciprocal_cartesian(unit_cell, np.asarray(hkl, dtype=np.float64))
    axis = np.asarray(spindle, dtype=np.float64)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    ang = np.asarray(phi, dtype=np.float64).reshape(-1)
    # Rodrigues rotation of s.
    k = axis
    c = np.cos(ang)[:, None]
    si = np.sin(ang)[:, None]
    s_rot = s * c + np.cross(np.broadcast_to(k, s.shape), s) * si + k[None, :] * (s @ k)[:, None] * (1.0 - c)
    s_inc = np.zeros_like(s_rot)
    s_inc[:, 2] = 1.0  # beam along +z in the fallback frame
    wave = float(wavelength) if wavelength and wavelength > 0 else 1.0
    s_dif = s_inc + wave * s_rot
    nrm = np.linalg.norm(s_dif, axis=1, keepdims=True)
    nrm = np.where(nrm > 0, nrm, 1.0)
    s_inc_n = s_inc / np.maximum(np.linalg.norm(s_inc, axis=1, keepdims=True), 1e-12)
    return s_inc_n, s_dif / nrm


def sensitivities(n_crystal: int, reference: int = 0) -> np.ndarray:
    """``rho_d``; reference crystal fixed at 1 (relative dose)."""
    rho = np.ones(int(n_crystal), dtype=np.float64)
    if 0 <= int(reference) < rho.size:
        rho[int(reference)] = 1.0
    return rho
