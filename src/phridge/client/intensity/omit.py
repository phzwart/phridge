"""Client-side driver for ``ml_i_omit_windows`` (no torch import).

Per-window omit coefficients from the worker are composited in real space
(stitched), then written as a **single** MTZ of Fourier coefficients
(``FWT`` / ``DELFWT``).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from phridge.contrib.intensity_ll.omit_windows import OmitWindowOptions, OmitWindowStore


def omit_windows_enabled(default: str = "0") -> bool:
    return os.environ.get("PHRIDGE_OMIT_WINDOWS", default).strip().lower() in ("1", "true", "yes", "on")


def omit_save_npz(default: str = "0") -> bool:
    return os.environ.get("PHRIDGE_OMIT_SAVE_NPZ", default).strip().lower() in ("1", "true", "yes", "on")


def omit_options_from_env() -> OmitWindowOptions:
    chunk = os.environ.get("PHRIDGE_OMIT_CHUNK_SIZE", "").strip()
    return OmitWindowOptions(
        enabled=omit_windows_enabled(),
        box_size=float(os.environ.get("PHRIDGE_OMIT_BOX_SIZE", "10.0")),
        mode=os.environ.get("PHRIDGE_OMIT_MODE", "boxes").strip().lower() or "boxes",  # type: ignore[arg-type]
        block_size=int(os.environ.get("PHRIDGE_OMIT_BLOCK_SIZE", "5")),
        coefficient_types=os.environ.get("PHRIDGE_OMIT_COEFFICIENT_TYPES", "both").strip().lower() or "both",  # type: ignore[arg-type]
        when=os.environ.get("PHRIDGE_OMIT_WHEN", "end_of_refinement").strip().lower() or "end_of_refinement",  # type: ignore[arg-type]
        output_prefix=os.environ.get("PHRIDGE_OMIT_PREFIX") or None,
        chunk_size=int(chunk) if chunk.isdigit() else None,
    )


def _miller_set_from_hkl(template: Any, hkl: np.ndarray) -> Any:
    """Build a miller.set on ``hkl`` matching ``template`` crystal symmetry."""
    from cctbx.array_family import flex

    indices = flex.miller_index()
    for row in np.asarray(hkl, dtype=np.int32).reshape(-1, 3):
        indices.append((int(row[0]), int(row[1]), int(row[2])))
    return template.customized_copy(indices=indices)


def _complex_miller(ms: Any, data: np.ndarray) -> Any:
    from cctbx import miller
    from cctbx.array_family import flex

    cd = flex.complex_double(np.ascontiguousarray(data, dtype=np.complex128))
    return miller.array(ms, data=cd).set_observation_type_xray_amplitude()


def _partition_shift(store: OmitWindowStore) -> np.ndarray:
    part = store.window_meta.get("partition") or {}
    sh = part.get("shift")
    if sh is None:
        return np.zeros(3, dtype=np.float64)
    return np.asarray(sh, dtype=np.float64).reshape(3)


def fractional_box_mask(
    n_real: Sequence[int],
    bounds_frac: Sequence[Sequence[float]],
    shift: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Boolean mask on an ``(nx,ny,nz)`` grid for a half-open fractional box.

    Grid index ``(i,j,k)`` maps to fractional ``(i/nx, j/ny, k/nz)``, then
    shifted by ``shift`` (same convention as ``assign_box_windows``) before
    comparison to ``bounds_frac``.
    """
    nx, ny, nz = (int(n_real[0]), int(n_real[1]), int(n_real[2]))
    sh = np.zeros(3, dtype=np.float64) if shift is None else np.asarray(shift, dtype=np.float64).reshape(3)
    (x0, x1), (y0, y1), (z0, z1) = bounds_frac
    i = np.arange(nx, dtype=np.float64)[:, None, None]
    j = np.arange(ny, dtype=np.float64)[None, :, None]
    k = np.arange(nz, dtype=np.float64)[None, None, :]
    xf = np.mod(i / nx + sh[0], 1.0)
    yf = np.mod(j / ny + sh[1], 1.0)
    zf = np.mod(k / nz + sh[2], 1.0)
    return (xf >= float(x0)) & (xf < float(x1)) & (yf >= float(y0)) & (yf < float(y1)) & (zf >= float(z0)) & (
        zf < float(z1)
    )


def atom_voronoi_assignment(
    n_real: Sequence[int],
    sites_frac: np.ndarray,
    *,
    unit_cell: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """For each voxel, index of the nearest atom (min-image, orthogonal Å metric)."""
    nx, ny, nz = (int(n_real[0]), int(n_real[1]), int(n_real[2]))
    sites = np.asarray(sites_frac, dtype=np.float64).reshape(-1, 3)
    if sites.shape[0] == 0:
        raise ValueError("sites_frac must be non-empty")
    if unit_cell is not None and len(unit_cell) >= 3:
        scale = np.asarray(unit_cell[:3], dtype=np.float64)
    else:
        scale = np.ones(3, dtype=np.float64)

    i = np.arange(nx, dtype=np.float64)[:, None, None]
    j = np.arange(ny, dtype=np.float64)[None, :, None]
    k = np.arange(nz, dtype=np.float64)[None, None, :]
    g = np.stack([i / nx, j / ny, k / nz], axis=-1).reshape(-1, 3)

    best_d2 = np.full(g.shape[0], np.inf, dtype=np.float64)
    best_idx = np.zeros(g.shape[0], dtype=np.int64)
    # Process atoms in blocks to bound peak memory
    block = 256
    for start in range(0, sites.shape[0], block):
        refs = sites[start : start + block]
        # (P, B, 3)
        d = g[:, None, :] - refs[None, :, :]
        d -= np.round(d)
        dA = d * scale[None, None, :]
        d2 = np.sum(dA * dA, axis=2)
        local = np.argmin(d2, axis=1)
        local_d2 = d2[np.arange(d2.shape[0]), local]
        better = local_d2 < best_d2
        best_d2[better] = local_d2[better]
        best_idx[better] = start + local[better]
    return best_idx.reshape(nx, ny, nz)


def atom_voronoi_window_mask(
    n_real: Sequence[int],
    sites_frac: np.ndarray,
    atom_to_window: np.ndarray,
    window_id: int,
    *,
    unit_cell: Optional[Sequence[float]] = None,
    assignment: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Mask voxels whose nearest atom belongs to ``window_id``."""
    a2w = np.asarray(atom_to_window, dtype=np.int64).reshape(-1)
    if assignment is None:
        assignment = atom_voronoi_assignment(n_real, sites_frac, unit_cell=unit_cell)
    return a2w[assignment] == int(window_id)


def _windows_need_voronoi(store: OmitWindowStore) -> bool:
    for w in store.window_meta.get("windows", []):
        if int(w.get("window_id", -1)) in set(int(x) for x in store.window_ids) and w.get("bounds_frac") is None:
            return True
    return False


def _window_mask_for(
    store: OmitWindowStore,
    window_id: int,
    n_real: Sequence[int],
    *,
    sites_frac: Optional[np.ndarray] = None,
    unit_cell: Optional[Sequence[float]] = None,
    voronoi_assignment: Optional[np.ndarray] = None,
) -> np.ndarray:
    wid = int(window_id)
    meta_by_id = {int(w["window_id"]): w for w in store.window_meta.get("windows", [])}
    wmeta = meta_by_id.get(wid, {})
    bounds = wmeta.get("bounds_frac")
    shift = _partition_shift(store)
    if bounds is not None:
        return fractional_box_mask(n_real, bounds, shift=shift)
    if sites_frac is None:
        raise ValueError(
            f"window {wid} has no bounds_frac; pass sites_frac for Voronoi stitching "
            f"(mode={store.window_meta.get('mode')!r})"
        )
    return atom_voronoi_window_mask(
        n_real,
        sites_frac,
        store.atom_to_window,
        wid,
        unit_cell=unit_cell,
        assignment=voronoi_assignment,
    )


def stitch_omit_map_coefficients(
    store: OmitWindowStore,
    miller_template: Any,
    kind: str = "difference",
    *,
    sites_frac: Optional[np.ndarray] = None,
    unit_cell: Optional[Sequence[float]] = None,
    resolution_factor: float = 0.25,
) -> Any:
    """Real-space composite of per-window omit maps → one miller array of coeffs.

    For each computed window: FFT → keep voxels inside the window mask → accumulate.
    Inverse FFT yields Fourier coefficients of the stitched map on ``store.hkl``.
    """
    from cctbx import miller
    from cctbx.array_family import flex

    ms = _miller_set_from_hkl(miller_template, store.hkl)
    wids = [int(w) for w in store.window_ids]
    if not wids:
        raise ValueError("store has no computed windows to stitch")

    first = _complex_miller(ms, store.coefficients(wids[0], kind))
    gridding = first.fft_map(resolution_factor=float(resolution_factor))
    n_real = tuple(int(x) for x in gridding.n_real())
    composite = np.zeros(n_real, dtype=np.float64)

    voronoi_assignment = None
    if _windows_need_voronoi(store):
        if sites_frac is None:
            raise ValueError("sites_frac required to stitch residue_blocks (no bounds_frac)")
        voronoi_assignment = atom_voronoi_assignment(n_real, sites_frac, unit_cell=unit_cell)

    for wid in wids:
        coeffs = _complex_miller(ms, store.coefficients(wid, kind))
        fft = miller.fft_map(crystal_gridding=gridding, fourier_coefficients=coeffs)
        rho = np.asarray(fft.real_map_unpadded().as_numpy_array(), dtype=np.float64)
        if rho.shape != composite.shape:
            raise RuntimeError(f"map shape mismatch for window {wid}: {rho.shape} vs {composite.shape}")
        mask = _window_mask_for(
            store,
            wid,
            n_real,
            sites_frac=sites_frac,
            unit_cell=unit_cell,
            voronoi_assignment=voronoi_assignment,
        )
        composite[mask] = rho[mask]

    flex_map = flex.double(np.ascontiguousarray(composite).ravel())
    flex_map.reshape(flex.grid(n_real))
    # Treat the composite as a general map so hard window masks are not
    # re-symmetrized; then restrict back to the original miller set.
    stitched = first.structure_factors_from_map(
        map=flex_map,
        use_sg=False,
        in_place_fft=False,
    )
    return stitched.common_set(first)


def write_stitched_omit_mtz(
    store: OmitWindowStore,
    miller_template: Any,
    path: str,
    *,
    sites_frac: Optional[np.ndarray] = None,
    unit_cell: Optional[Sequence[float]] = None,
    resolution_factor: Optional[float] = None,
    model_label: str = "FWT",
    difference_label: str = "DELFWT",
) -> str:
    """Write one MTZ with Fourier coefficients of the stitched omit map."""
    rf = resolution_factor
    if rf is None:
        env = os.environ.get("PHRIDGE_OMIT_RESOLUTION_FACTOR", "").strip()
        rf = float(env) if env else 0.25

    mtz = None
    if store.coef_model is not None:
        model = stitch_omit_map_coefficients(
            store,
            miller_template,
            "model",
            sites_frac=sites_frac,
            unit_cell=unit_cell,
            resolution_factor=rf,
        )
        mtz = model.as_mtz_dataset(column_root_label=model_label)
    if store.coef_difference is not None:
        diff = stitch_omit_map_coefficients(
            store,
            miller_template,
            "difference",
            sites_frac=sites_frac,
            unit_cell=unit_cell,
            resolution_factor=rf,
        )
        if mtz is None:
            mtz = diff.as_mtz_dataset(column_root_label=difference_label)
        else:
            mtz.add_miller_array(diff, column_root_label=difference_label)
    if mtz is None:
        raise ValueError("store has neither coef_model nor coef_difference")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mtz.mtz_object().write(str(path))
    meta_path = Path(str(path)).with_suffix(".window_meta.json")
    meta_path.write_text(json.dumps(store.window_meta, indent=2, default=str))
    return str(path)


# ---------------------------------------------------------------------------
# Legacy / diagnostic per-window writers (opt-in)
# ---------------------------------------------------------------------------


def write_omit_window_mtz(
    store: OmitWindowStore,
    miller_template: Any,
    window_id: int,
    path: str,
    *,
    model_label: str = "FWT",
    difference_label: str = "DELFWT",
) -> str:
    """Write one window's omit coefficients to MTZ (diagnostic)."""
    ms = _miller_set_from_hkl(miller_template, store.hkl)
    mtz = None
    if store.coef_model is not None:
        model = store.coefficients(window_id, "model")
        mtz = _complex_miller(ms, model).as_mtz_dataset(column_root_label=model_label)
    if store.coef_difference is not None:
        diff = store.coefficients(window_id, "difference")
        arr = _complex_miller(ms, diff)
        if mtz is None:
            mtz = arr.as_mtz_dataset(column_root_label=difference_label)
        else:
            mtz.add_miller_array(arr, column_root_label=difference_label)
    if mtz is None:
        raise ValueError("store has neither coef_model nor coef_difference")
    mtz.mtz_object().write(str(path))
    return str(path)


export_omit_window_mtz = write_omit_window_mtz


def run_omit_windows_from_fmodel(
    fmodel: Any,
    *,
    output_prefix: Optional[str] = None,
    options: Optional[OmitWindowOptions] = None,
    log: Any = None,
) -> Optional[OmitWindowStore]:
    """Dispatch ``ml_i_omit_windows`` for an ``IntensityFModel``; never raises to the caller.

    Writes a single stitched-omit MTZ ``{prefix}_omit_windows.mtz`` (``FWT`` /
    ``DELFWT``). Set ``PHRIDGE_OMIT_SAVE_NPZ=1`` for a diagnostic npz.
    """
    out = log if log is not None else sys.stdout
    try:
        opts = options or omit_options_from_env()
        if not opts.enabled:
            return None
        from phridge.client.convert_xtal import scattering_table_from_cctbx, xray_from_cctbx
        from phridge.contrib.intensity_ll.client import RemoteOmitWindowCoefficients
        from phridge.contrib.intensity_ll.ops import register_ops

        register_ops()
        bridge = fmodel.bridge
        i_obs = fmodel._i_obs
        f_model = fmodel.f_model_scaled_with_k1() if hasattr(fmodel, "f_model_scaled_with_k1") else fmodel.f_model()
        xs = fmodel._xray_structure
        xray = xray_from_cctbx(xs)
        table = scattering_table_from_cctbx(xs)
        params = fmodel.params
        n = i_obs.size()
        try:
            k_iso = np.asarray(fmodel.k_isotropic(), dtype=np.float64)
            k_aniso = np.asarray(fmodel.k_anisotropic(), dtype=np.float64)
            k_scale = k_iso * k_aniso
        except Exception:
            k_scale = np.ones(n, dtype=np.float64)

        target_spec = dict(getattr(fmodel, "target_spec", None) or {"name": "ml_i"})
        target_spec["name"] = "ml_i"
        remote = RemoteOmitWindowCoefficients(
            bridge,
            i_obs,
            target_spec,
            omit=opts.model_dump(),
            alpha=fmodel.sigma_a,
            beta=fmodel.sigma_wilson,
            epsilon=i_obs.epsilons().data().as_double(),
            centric=i_obs.centric_flags().data(),
            r_free=fmodel._r_free_flags.data() if fmodel._r_free_flags is not None else None,
            nu=getattr(fmodel, "nu_per_refl", None),
        )
        store = remote(
            f_model,
            xray,
            table,
            params,
            k_scale=k_scale,
        )
        prefix = output_prefix or opts.output_prefix or "mli_omit"
        miller_template = i_obs
        sites_frac = np.asarray(xs.sites_frac(), dtype=np.float64)
        try:
            unit_cell = tuple(float(x) for x in xs.unit_cell().parameters())
        except Exception:
            unit_cell = None

        mtz_path = f"{prefix}_omit_windows.mtz"
        write_stitched_omit_mtz(
            store,
            miller_template,
            mtz_path,
            sites_frac=sites_frac,
            unit_cell=unit_cell,
        )
        print(
            f"[mli_omit_windows] wrote stitched omit MTZ {mtz_path} "
            f"(FWT / DELFWT; W_computed={len(store.window_ids)})",
            file=out,
            flush=True,
        )
        if omit_save_npz():
            npz_path = f"{prefix}_omit_windows.npz"
            store.save(npz_path)
            print(f"[mli_omit_windows] also wrote {npz_path}", file=out, flush=True)
        return store
    except Exception as exc:
        print(f"[mli_omit_windows] skipped (non-fatal): {exc}", file=out, flush=True)
        return None
