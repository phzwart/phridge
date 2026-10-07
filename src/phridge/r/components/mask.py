"""F_mask from the solvent mask plus a distance-transform modulation.

The binary solvent mask M is the bulk constant. A nonlinear function of the
periodic distance into solvent from the protein–solvent interface supplies a
near-surface correction:

    ρ(x) = M(x) · (1 + α φ(d(x)))     (baked, default)
    ρ    = k_sol [ e^{−B s²/4} FFT(M) + α FFT(M φ) ]   (two-component)

    φ(∞) → 0 so the mask level stays the bulk density. Torch-free.

Phenix refine does not install this. The bake changed the F_mask enough to
hurt k_mask / NLL and did not beat the flat Jiang–Brünger mask. The builders
stay here for ``IntensityModel`` and tests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator

PhiKind = Literal["exp", "shell", "logistic"]
MaskMode = Literal["baked", "two_component"]

DEFAULT_ALPHA = 0.35
DEFAULT_LENGTH_SCALE = 2.0
DEFAULT_SHELL_CENTER = 1.4
DEFAULT_SHELL_WIDTH = 1.0

_ENV_ENABLED = "PHRIDGE_DT_MASK"
_ENV_ALPHA = "PHRIDGE_DT_MASK_ALPHA"
_ENV_LENGTH = "PHRIDGE_DT_MASK_LENGTH"
_ENV_PHI = "PHRIDGE_DT_MASK_PHI"
_ENV_MODE = "PHRIDGE_DT_MASK_MODE"
_ENV_CENTER = "PHRIDGE_DT_MASK_CENTER"
_ENV_WIDTH = "PHRIDGE_DT_MASK_WIDTH"


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return float(default)
    try:
        value = float(raw)
    except ValueError:
        return float(default)
    if value != value:
        return float(default)
    return value


class DistanceMaskOptions(BaseModel):
    """JSON / env options for a distance-modulated solvent mask."""

    model_config = {"extra": "forbid"}

    enabled: bool = False
    alpha: float = Field(default=DEFAULT_ALPHA, ge=0.0, description="Modulation amplitude on top of M.")
    length_scale: float = Field(
        default=DEFAULT_LENGTH_SCALE,
        gt=0.0,
        description="Decay length λ in Å for φ(d)=exp(−d/λ).",
    )
    phi: PhiKind = Field(default="exp", description="Nonlinear function of the distance transform.")
    mode: MaskMode = Field(
        default="baked",
        description="baked: one F_mask = FFT(M(1+αφ)). two_component: FFT(M) and FFT(M φ).",
    )
    center: float = Field(
        default=DEFAULT_SHELL_CENTER,
        ge=0.0,
        description="Shell / logistic midpoint d0 in Å.",
    )
    width: float = Field(
        default=DEFAULT_SHELL_WIDTH,
        gt=0.0,
        description="Shell σ or logistic width w in Å.",
    )

    @field_validator("phi", mode="before")
    @classmethod
    def _phi_name(cls, value: Any) -> str:
        raw = str(value).strip().lower()
        aliases = {"exponential": "exp", "gauss": "shell", "gaussian": "shell", "logit": "logistic"}
        return aliases.get(raw, raw)


def options_from_env() -> DistanceMaskOptions:
    """Read ``PHRIDGE_DT_MASK*``. Default is off."""
    raw_mode = os.environ.get(_ENV_MODE, "baked").strip().lower()
    mode: MaskMode = "two_component" if raw_mode in {"two", "two_component", "split"} else "baked"
    raw_phi = os.environ.get(_ENV_PHI, "exp")
    return DistanceMaskOptions(
        enabled=_flag(_ENV_ENABLED, "0"),
        alpha=max(0.0, _float(_ENV_ALPHA, DEFAULT_ALPHA)),
        length_scale=max(1e-3, _float(_ENV_LENGTH, DEFAULT_LENGTH_SCALE)),
        phi=raw_phi,  # type: ignore[arg-type]
        mode=mode,
        center=max(0.0, _float(_ENV_CENTER, DEFAULT_SHELL_CENTER)),
        width=max(1e-3, _float(_ENV_WIDTH, DEFAULT_SHELL_WIDTH)),
    )


def phi_of_distance(distance: np.ndarray, options: DistanceMaskOptions) -> np.ndarray:
    """φ(d) with φ(∞) → 0. ``distance`` is Å into solvent; protein voxels should be 0."""
    d = np.asarray(distance, dtype=np.float64)
    if options.phi == "exp":
        return np.exp(-d / float(options.length_scale))
    if options.phi == "shell":
        t = (d - float(options.center)) / float(options.width)
        return np.exp(-(t * t))
    t = (d - float(options.center)) / float(options.width)
    return 1.0 / (1.0 + np.exp(t))


def modulate_mask(mask: np.ndarray, distance: np.ndarray, options: DistanceMaskOptions) -> np.ndarray:
    """ρ = M · (1 + α φ(d)). Protein voxels stay 0."""
    m = np.asarray(mask, dtype=np.float64)
    return m * (1.0 + float(options.alpha) * phi_of_distance(distance, options))


def modulation_component(mask: np.ndarray, distance: np.ndarray, options: DistanceMaskOptions) -> np.ndarray:
    """ρ_mod = M · φ(d), the zero-at-infinity piece without the constant."""
    m = np.asarray(mask, dtype=np.float64)
    return m * phi_of_distance(distance, options)


def voxel_sampling_angstrom(unit_cell: Any, n_real: tuple[int, int, int]) -> tuple[float, float, float]:
    """Å per voxel along the three grid axes (orthogonal approximation)."""
    if hasattr(unit_cell, "parameters"):
        a, b, c = (float(x) for x in unit_cell.parameters()[:3])
    else:
        a, b, c = (float(x) for x in tuple(unit_cell)[:3])
    nx, ny, nz = (int(n_real[0]), int(n_real[1]), int(n_real[2]))
    if min(nx, ny, nz) < 1:
        raise ValueError("n_real must be positive")
    return (a / nx, b / ny, c / nz)


def _edt_1d_squared(f: np.ndarray, sampling: float) -> np.ndarray:
    """Felzenszwalb 1-D squared Euclidean DT. ``f`` is already a squared-distance row."""
    n = int(f.shape[0])
    if n == 0:
        return f.copy()
    s2 = float(sampling) * float(sampling)
    if s2 <= 0.0:
        raise ValueError("sampling must be positive")
    v = np.zeros(n, dtype=np.int64)
    z = np.empty(n + 1, dtype=np.float64)
    z[0] = -np.inf
    z[1] = np.inf
    k = 0
    for q in range(1, n):
        fq = float(f[q])
        while True:
            vk = int(v[k])
            denom = 2.0 * s2 * (q - vk)
            s = ((fq + s2 * q * q) - (float(f[vk]) + s2 * vk * vk)) / denom
            if s > z[k]:
                break
            k -= 1
            if k < 0:
                k = 0
                vk = int(v[0])
                denom = 2.0 * s2 * (q - vk)
                s = ((fq + s2 * q * q) - (float(f[vk]) + s2 * vk * vk)) / denom
                break
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = np.inf
    out = np.empty(n, dtype=np.float64)
    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        vk = int(v[k])
        out[q] = s2 * (q - vk) * (q - vk) + float(f[vk])
    return out


def _periodic_edt_1d_squared(f: np.ndarray, sampling: float) -> np.ndarray:
    n = int(f.shape[0])
    if n == 0:
        return f.copy()
    ext = np.concatenate([f, f, f])
    return _edt_1d_squared(ext, sampling)[n : 2 * n]


def _squared_edt_3d(feature: np.ndarray, sampling: tuple[float, float, float]) -> np.ndarray:
    """Squared Euclidean DT of a 0/1 feature map (1 = feature / protein). Periodic."""
    inf = 1.0e30
    f = np.where(feature, 0.0, inf).astype(np.float64, copy=False)
    nx, ny, nz = f.shape
    tmp = np.empty_like(f)
    for j in range(ny):
        for k in range(nz):
            tmp[:, j, k] = _periodic_edt_1d_squared(f[:, j, k], sampling[0])
    f2 = np.empty_like(f)
    for i in range(nx):
        for k in range(nz):
            f2[i, :, k] = _periodic_edt_1d_squared(tmp[i, :, k], sampling[1])
    out = np.empty_like(f)
    for i in range(nx):
        for j in range(ny):
            out[i, j, :] = _periodic_edt_1d_squared(f2[i, j, :], sampling[2])
    return out


def periodic_distance_transform(
    solvent: np.ndarray,
    sampling: tuple[float, float, float],
) -> np.ndarray:
    """Å from each solvent voxel to the nearest protein voxel, periodic.

    ``solvent`` is True / 1 inside solvent. Protein voxels return 0.
    """
    m = np.asarray(solvent)
    if m.ndim != 3:
        raise ValueError(f"solvent mask must be 3-D, got shape {m.shape}")
    solvent_b = m.astype(bool, copy=False)
    if not np.any(solvent_b):
        return np.zeros(solvent_b.shape, dtype=np.float64)
    if np.all(solvent_b):
        # No protein: distance is undefined; treat as far so φ → 0.
        return np.full(solvent_b.shape, 1.0e6, dtype=np.float64)
    protein = ~solvent_b
    try:
        from phridge.numeric.distance import distance_transform_edt

        tiled = np.tile(solvent_b, (3, 3, 3))
        d_big = distance_transform_edt(tiled, sampling=sampling)
        nx, ny, nz = solvent_b.shape
        return np.asarray(d_big[nx : 2 * nx, ny : 2 * ny, nz : 2 * nz], dtype=np.float64)
    except ImportError:
        d2 = _squared_edt_3d(protein, sampling)
        d2 = np.maximum(d2, 0.0)
        d2[~solvent_b] = 0.0
        return np.sqrt(d2)


def as_solvent_ones(mask: np.ndarray) -> np.ndarray:
    """Return a 0/1 map. Input True / >0.5 is treated as solvent."""
    raw = np.asarray(mask)
    if raw.dtype == bool:
        return raw.astype(np.float64)
    return (np.asarray(raw, dtype=np.float64) > 0.5).astype(np.float64)


def _corr_abs(a: Any, b: Any) -> float:
    x = np.abs(np.asarray(a.data(), dtype=np.complex128))
    y = np.abs(np.asarray(b.data(), dtype=np.complex128))
    x = x - float(x.mean())
    y = y - float(y.mean())
    den = float(np.linalg.norm(x) * np.linalg.norm(y))
    if den <= 0.0:
        return 0.0
    return float(np.dot(x, y) / den)


def resolve_solvent_ones(
    raw: np.ndarray,
    miller_array: Optional[Any] = None,
    f_ref: Optional[Any] = None,
) -> np.ndarray:
    """0/1 solvent map. Prefers the polarity whose FFT matches ``f_ref``."""
    a = as_solvent_ones(raw)
    b = 1.0 - a
    if miller_array is not None and f_ref is not None:
        try:
            fa = structure_factors_from_density(miller_array, a)
            fb = structure_factors_from_density(miller_array, b)
            return a if _corr_abs(fa, f_ref) >= _corr_abs(fb, f_ref) else b
        except Exception:
            pass
    # cctbx atom_mask is typically 1 = protein; invert a mid-range 1-fraction.
    frac = float(a.mean()) if a.size else 0.0
    if 0.15 <= frac <= 0.85:
        return b
    if frac < 0.15:
        return b
    return a


@dataclass
class BuiltDistanceMask:
    """Real-space pieces and the F_mask miller array(s)."""

    mask: np.ndarray
    distance: np.ndarray
    density: np.ndarray
    density_mod: np.ndarray
    f_mask: Any
    f_mask_mod: Any
    sampling: tuple[float, float, float]


def densities_from_mask(
    mask: np.ndarray,
    sampling: tuple[float, float, float],
    options: DistanceMaskOptions,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(distance, ρ_baked, ρ_mod)`` from a 0/1 solvent mask."""
    solvent = as_solvent_ones(mask)
    distance = periodic_distance_transform(solvent, sampling)
    baked = modulate_mask(solvent, distance, options)
    mod = modulation_component(solvent, distance, options)
    return distance, baked, mod




def __getattr__(name: str):
    from phridge.shims.cctbx.mask import __dict__ as _shim

    if name in _shim and not name.startswith("__"):
        return _shim[name]
    raise AttributeError(name)
