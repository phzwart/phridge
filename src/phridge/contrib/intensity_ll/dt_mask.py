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
        from scipy.ndimage import distance_transform_edt

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


def structure_factors_from_density(miller_array: Any, density: np.ndarray) -> Any:
    """FFT a whole-cell real map onto ``miller_array``'s hkl list."""
    from cctbx import miller
    from cctbx.array_family import flex

    rho = np.ascontiguousarray(np.asarray(density, dtype=np.float64))
    if rho.ndim != 3:
        raise ValueError(f"density must be 3-D, got shape {rho.shape}")
    n_real = tuple(int(x) for x in rho.shape)
    flex_map = flex.double(rho.ravel())
    flex_map.reshape(flex.grid(n_real))
    ms = miller.set(
        crystal_symmetry=miller_array.crystal_symmetry(),
        indices=miller_array.indices(),
        anomalous_flag=bool(miller_array.anomalous_flag()),
    )
    f = ms.structure_factors_from_map(map=flex_map, use_sg=False, in_place_fft=False)
    return f.common_set(ms)


def _mask_params_radii(mask_params: Any) -> tuple[float, float]:
    solvent = 1.11
    shrink = 0.9
    if mask_params is None:
        return solvent, shrink
    solvent = float(getattr(mask_params, "solvent_radius", solvent) or solvent)
    shrink = float(getattr(mask_params, "shrink_truncation_radius", shrink) or shrink)
    return solvent, shrink


def _atom_radii(xray_structure: Any) -> Any:
    from cctbx.array_family import flex

    try:
        from mmtbx.masks import vdw_radii_from_xray_structure

        return vdw_radii_from_xray_structure(xray_structure)
    except Exception:
        pass
    try:
        from cctbx.eltbx import van_der_waals_radii

        table = van_der_waals_radii.vdw.table
        vals = []
        for sc in xray_structure.scatterers():
            el = "".join(ch for ch in sc.element_symbol() if ch.isalpha()).capitalize()
            try:
                vals.append(float(table(el, False)))
            except Exception:
                vals.append(1.5)
        return flex.double(vals)
    except Exception:
        return flex.double(xray_structure.scatterers().size(), 1.5)


def resolution_factor_from_mask_params(mask_params: Any) -> float:
    """cctbx ``crystal_gridding`` resolution_factor in (0, 0.5].

    mmtbx ``grid_step_factor`` is ``d_min / spacing`` (default 4.0). cctbx
    ``resolution_factor`` is ``spacing / d_min`` and must be ≤ 0.5. Passing
    the mmtbx value through unchanged trips
    ``CCTBX_ASSERT(resolution_factor <= 0.5)``.
    """
    factor = 1.0 / 3.0
    if mask_params is None:
        return factor
    raw = getattr(mask_params, "grid_step_factor", None)
    if raw is None:
        return factor
    try:
        step = float(raw)
    except (TypeError, ValueError):
        return factor
    if step <= 0.0 or step != step:
        return factor
    rf = (1.0 / step) if step > 0.5 else step
    return min(max(rf, 1e-3), 0.5)


def _n_real(miller_array: Any, mask_params: Any) -> tuple[int, int, int]:
    if mask_params is not None:
        explicit = getattr(mask_params, "n_real", None)
        if explicit is not None:
            try:
                n = tuple(int(x) for x in explicit)
                if len(n) == 3 and min(n) > 0:
                    return n
            except (TypeError, ValueError):
                pass
    factor = resolution_factor_from_mask_params(mask_params)
    d_min = float(miller_array.d_min())
    try:
        grid = miller_array.crystal_gridding(resolution_factor=factor, d_min=d_min)
    except RuntimeError:
        grid = miller_array.crystal_gridding(resolution_factor=1.0 / 3.0, d_min=d_min)
    n = grid.n_real()
    return (int(n[0]), int(n[1]), int(n[2]))


def _flex_map_to_numpy(data: Any) -> np.ndarray:
    if hasattr(data, "as_numpy_array"):
        arr = np.asarray(data.as_numpy_array(), dtype=np.float64)
    else:
        arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 3:
        return arr
    if hasattr(data, "all"):
        return arr.reshape(tuple(int(x) for x in data.all()))
    raise ValueError("could not reshape mask data to 3-D")


def _try_atom_mask(xray_structure: Any, n_real: tuple[int, int, int], solvent_radius: float, shrink: float) -> Optional[np.ndarray]:
    try:
        from cctbx.masks import atom_mask
    except ImportError:
        return None
    radii = _atom_radii(xray_structure)
    try:
        am = atom_mask(
            unit_cell=xray_structure.unit_cell(),
            space_group=xray_structure.space_group(),
            gridding_n_real=n_real,
            solvent_radius=float(solvent_radius),
            shrink_truncation_radius=float(shrink),
        )
        am.compute(xray_structure.sites_frac(), radii)
        return _flex_map_to_numpy(am.mask_data_whole_uc())
    except Exception:
        pass
    try:
        am = atom_mask(
            xray_structure.unit_cell(),
            xray_structure.space_group(),
            xray_structure.sites_frac(),
            radii,
            n_real,
            float(solvent_radius),
            float(shrink),
        )
        data = am.mask_data_whole_uc() if hasattr(am, "mask_data_whole_uc") else am
        return _flex_map_to_numpy(data)
    except Exception:
        return None


def _try_mask_from_xray_structure(
    miller_array: Any,
    xray_structure: Any,
    n_real: tuple[int, int, int],
    solvent_radius: float,
    shrink: float,
) -> Optional[np.ndarray]:
    try:
        from mmtbx.masks import mask_from_xray_structure
    except ImportError:
        return None
    try:
        obj = mask_from_xray_structure(
            xray_structure=xray_structure,
            p1=True,
            for_structure_factors=True,
            n_real=n_real,
            solvent_radius=float(solvent_radius),
            shrink_truncation_radius=float(shrink),
        )
        data = getattr(obj, "mask_data", None)
        if data is None:
            return None
        return _flex_map_to_numpy(data)
    except Exception:
        return None


def _try_mmtbx_manager_mask(miller_array: Any, xray_structure: Any) -> Optional[np.ndarray]:
    try:
        from mmtbx.masks import manager as mask_manager
    except ImportError:
        return None
    try:
        mm = mask_manager(miller_array=miller_array, xray_structure=xray_structure)
    except Exception:
        return None
    for name in (
        "mask_data_whole_uc",
        "bulk_solvent_mask",
        "solvent_mask",
        "get_solvent_mask",
    ):
        attr = getattr(mm, name, None)
        if attr is None:
            continue
        try:
            data = attr() if callable(attr) else attr
            if hasattr(data, "data") and not hasattr(data, "as_numpy_array"):
                data = data.data
                data = data() if callable(data) else data
            return _flex_map_to_numpy(data)
        except Exception:
            continue
    for name in ("asu_mask", "_asu_mask"):
        asu = getattr(mm, name, None)
        if asu is None:
            continue
        getter = getattr(asu, "mask_data_whole_uc", None)
        if getter is None:
            getter = getattr(asu, "mask_data", None)
        if getter is None:
            continue
        try:
            data = getter() if callable(getter) else getter
            return _flex_map_to_numpy(data)
        except Exception:
            continue
    return None


def solvent_mask_real_space(
    miller_array: Any,
    xray_structure: Any,
    mask_params: Any = None,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Whole-cell 0/1 solvent map and Å/voxel sampling."""
    n_real = _n_real(miller_array, mask_params)
    solvent_radius, shrink = _mask_params_radii(mask_params)
    raw = _try_mask_from_xray_structure(miller_array, xray_structure, n_real, solvent_radius, shrink)
    if raw is None:
        raw = _try_mmtbx_manager_mask(miller_array, xray_structure)
    if raw is None:
        raw = _try_atom_mask(xray_structure, n_real, solvent_radius, shrink)
    if raw is None:
        raise RuntimeError("could not build a real-space solvent mask (mmtbx/cctbx masks unavailable)")
    sampling = voxel_sampling_angstrom(xray_structure.unit_cell(), raw.shape)
    f_ref = None
    try:
        f_ref = _flat_f_mask(miller_array, xray_structure)
    except Exception:
        f_ref = None
    ones = resolve_solvent_ones(raw, miller_array=miller_array, f_ref=f_ref)
    return ones, sampling


def _flat_f_mask(miller_array: Any, xray_structure: Any) -> Any:
    from mmtbx.masks import manager as mask_manager

    mm = mask_manager(miller_array=miller_array, xray_structure=xray_structure)
    return mm.shell_f_masks()[0]


def build_f_masks(
    miller_array: Any,
    xray_structure: Any,
    options: Optional[DistanceMaskOptions] = None,
    mask_params: Any = None,
) -> BuiltDistanceMask:
    """Build F_mask (and optional F_mod) from M and its distance transform.

    When ``options.enabled`` is false this still returns the flat mmtbx F_mask
    and a dummy real-space field so callers have one code path.
    """
    opts = options if options is not None else options_from_env()
    if not opts.enabled:
        f_flat = _flat_f_mask(miller_array, xray_structure)
        empty = np.zeros((1, 1, 1), dtype=np.float64)
        return BuiltDistanceMask(
            mask=empty,
            distance=empty,
            density=empty,
            density_mod=empty,
            f_mask=f_flat,
            f_mask_mod=None,
            sampling=(1.0, 1.0, 1.0),
        )
    mask, sampling = solvent_mask_real_space(miller_array, xray_structure, mask_params)
    distance, baked, mod = densities_from_mask(mask, sampling, opts)
    if opts.mode == "two_component":
        f_mask = structure_factors_from_density(miller_array, mask)
        f_mod = structure_factors_from_density(miller_array, mod)
        density = mask
    else:
        f_mask = structure_factors_from_density(miller_array, baked)
        f_mod = None
        density = baked
    return BuiltDistanceMask(
        mask=mask,
        distance=distance,
        density=density,
        density_mod=mod,
        f_mask=f_mask,
        f_mask_mod=f_mod,
        sampling=sampling,
    )


def _as_flex_complex(miller: Any) -> Any:
    from cctbx.array_family import flex

    data = np.ascontiguousarray(np.asarray(miller.data(), dtype=np.complex128))
    return miller.customized_copy(data=flex.complex_double(data))


def _f_mask_compare(old: Any, new: Any) -> str:
    """Short |corr| and norm ratio of two miller F_mask arrays."""
    try:
        a = np.asarray(old.data(), dtype=np.complex128).ravel()
        b = np.asarray(new.data(), dtype=np.complex128).ravel()
    except Exception:
        return "no prior F_mask"
    if a.shape != b.shape or a.size == 0:
        return "shape mismatch"
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0:
        return f"‖old‖={na:.3g} ‖new‖={nb:.3g}"
    corr = float(np.abs(np.vdot(a, b)) / (na * nb))
    return f"|corr|={corr:.3f} ‖new‖/‖old‖={nb / na:.3f}"


def real_space_stats(built: BuiltDistanceMask) -> str:
    """Solvent fraction, mean distance, and how far ρ is from the binary mask."""
    mask = np.asarray(built.mask, dtype=np.float64)
    if mask.size <= 1:
        return "no real-space map"
    sol = mask > 0.5
    frac = float(sol.mean())
    dist = np.asarray(built.distance, dtype=np.float64)
    mean_d = float(dist[sol].mean()) if np.any(sol) else 0.0
    rho = np.asarray(built.density, dtype=np.float64)
    den = float(np.linalg.norm(mask))
    rms = float(np.linalg.norm(rho - mask) / den) if den > 0.0 else 0.0
    return f"sol={frac:.2f} ⟨d⟩={mean_d:.2f}Å rms(ρ−M)/M={rms:.3f}"


def _match_norm(source: Any, reference: Any) -> Any:
    """Rescale ``source`` so ‖source‖ = ‖reference‖. Phase / shape unchanged."""
    from cctbx.array_family import flex

    src = np.asarray(source.data(), dtype=np.complex128).ravel()
    ref = np.asarray(reference.data(), dtype=np.complex128).ravel()
    if src.shape != ref.shape or src.size == 0:
        return source
    ns = float(np.linalg.norm(src))
    nr = float(np.linalg.norm(ref))
    if ns <= 0.0 or nr <= 0.0:
        return source
    scale = nr / ns
    if abs(scale - 1.0) < 1e-6:
        return source
    return source.customized_copy(data=flex.complex_double(np.ascontiguousarray(src * scale)))


def _core_f_mask(fmodel: Any) -> Any:
    getter = getattr(fmodel, "f_mask", None)
    if not callable(getter):
        return None
    try:
        return getter()
    except Exception:
        return None


def _fresh_ls_f_mask(fmodel: Any, miller_array: Any) -> Any:
    xs = getattr(fmodel, "_xray_structure", None) or getattr(fmodel, "xray_structure", None)
    if xs is None:
        return None
    try:
        return _flat_f_mask(miller_array, xs)
    except Exception:
        return None


def install_built(fmodel: Any, built: BuiltDistanceMask, options: DistanceMaskOptions, log: Any = None) -> bool:
    """Write ``built`` into ``fmodel``'s core. Safe no-op when the core is missing.

    Replaces F_mask only. Existing k_mask is kept so a least-squares curve fitted
    to the flat mask remains the start for the NLL refit — ``update_core(f_mask=)``
    alone can zero k_mask on some mmtbx builds.

    Our FFT of M is a different convention from mmtbx ``shell_f_masks`` (~4× on
    a typical grid). The installed vector is rescaled to the LS F_mask norm so a
    k_mask fitted to the binary mask stays on the same scale.
    """
    if not options.enabled:
        return False
    if not (hasattr(fmodel, "update_core") and getattr(fmodel, "arrays", None) is not None):
        return False
    miller_array = fmodel.i_obs() if callable(getattr(fmodel, "i_obs", None)) else getattr(fmodel, "_i_obs", None)
    if miller_array is None:
        return False
    f_new = _as_flex_complex(built.f_mask)
    prior = _core_f_mask(fmodel)
    ls = _fresh_ls_f_mask(fmodel, miller_array)
    vs_core = _f_mask_compare(prior, f_new) if prior is not None else "no core F_mask"
    vs_ls = _f_mask_compare(ls, f_new) if ls is not None else "no LS F_mask"
    scale_ref = ls if ls is not None else prior
    if scale_ref is not None:
        f_new = _match_norm(f_new, scale_ref)
    k_masks = fmodel.k_masks() if callable(getattr(fmodel, "k_masks", None)) else []
    if options.mode == "two_component" and built.f_mask_mod is not None:
        from cctbx.array_family import flex

        f_mod = _as_flex_complex(built.f_mask_mod)
        if scale_ref is not None:
            f_mod = _match_norm(f_mod, scale_ref)
        n = int(miller_array.size())
        if k_masks:
            k0 = k_masks[0]
            k0_np = np.asarray(k0, dtype=np.float64).reshape(-1)
            k1 = flex.double(np.ascontiguousarray(np.full(n, float(np.mean(k0_np)) * options.alpha)))
            fmodel.update_core(f_mask=[f_new, f_mod], k_mask=[k0, k1])
        else:
            ones = flex.double(n, 1.0)
            fmodel.update_core(
                f_mask=[f_new, f_mod],
                k_mask=[ones, flex.double(n, float(options.alpha))],
            )
    elif k_masks:
        fmodel.update_core(f_mask=[f_new], k_mask=[k_masks[0]])
    else:
        fmodel.update_core(f_mask=[f_new])
    for name in ("_f_model", "_last_maps", "_r_values", "_f_post", "_f_mode", "_last_gradients"):
        if hasattr(fmodel, name):
            setattr(fmodel, name, None)
    if log is not None:
        k_note = ""
        if k_masks:
            k0 = np.asarray(k_masks[0], dtype=np.float64).ravel()
            if k0.size:
                k_note = f"; k_mask max={float(np.nanmax(np.abs(k0))):.3f}"
        print(
            f"[mli_quad] DT-modulated F_mask ({options.mode}, φ={options.phi}, "
            f"α={options.alpha:g}, λ={options.length_scale:g} Å; vs core {vs_core}; "
            f"vs LS {vs_ls}; {real_space_stats(built)}{k_note})",
            file=log,
        )
    return True


def install_on_fmodel(fmodel: Any, options: Optional[DistanceMaskOptions] = None, log: Any = None) -> bool:
    """Replace ``fmodel``'s F_mask list with the DT-modulated construction.

    Returns True when a replacement was installed. Safe no-op when disabled
    or when the core cannot be updated.
    """
    opts = options if options is not None else options_from_env()
    if not opts.enabled:
        return False
    miller_array = fmodel.i_obs() if callable(getattr(fmodel, "i_obs", None)) else getattr(fmodel, "_i_obs", None)
    xs = getattr(fmodel, "_xray_structure", None) or getattr(fmodel, "xray_structure", None)
    if miller_array is None or xs is None:
        return False
    built = build_f_masks(miller_array, xs, opts, mask_params=getattr(fmodel, "mask_params", None))
    return install_built(fmodel, built, opts, log=log)


def options_signature(options: DistanceMaskOptions) -> str:
    return options.model_dump_json()


def sites_signature(xray_structure: Any) -> tuple[int, float]:
    sites = np.asarray(xray_structure.sites_frac(), dtype=np.float64)
    return (int(sites.shape[0]), float(np.sum(sites)))
