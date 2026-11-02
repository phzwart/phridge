#!/usr/bin/env python3
"""Anisotropic least-squares scale of each median-set dataset onto 8UVZ.

8UVZ is the reference. Each other dataset is scaled in amplitude by an
overall factor and an anisotropic displacement tensor
(``mmtbx.scaling.relative_scaling.ls_rel_scale_driver``). The scaled
amplitudes are written to ``structures/{id}/{id}_scaled_to_8uvz.mtz``.

Usage::

    python examples/case_study/scale_to_8uvz.py
"""

from __future__ import annotations

import argparse
import math
import traceback
from pathlib import Path

import cctbx  # noqa: F401
import maps
import numpy as np
from cctbx.array_family import flex

import download

_ROOT = Path(__file__).resolve().parent
_STRUCTURES = _ROOT / "structures"
_REFERENCE = "8uvz"


def scale_to_reference(reference, target):
    """Scale ``target`` so its anisotropic Wilson surface matches ``reference``.

    The two crystals are not the same space group, so there is no common index.
    Least squares fits ``ln I = a - s·B·s`` on each set inside the shared
    resolution range, and the amplitude scale is the square root of the ratio
    of those two surfaces.
    """
    low, high = _shared_resolution(reference, target)
    reference_coef = _wilson_coefficients(reference.resolution_filter(d_min=high, d_max=low))
    target_coef = _wilson_coefficients(target.resolution_filter(d_min=high, d_max=low))
    scale = _amplitude_scale(target, reference_coef, target_coef)
    scaled = _apply_scale(target, scale)
    return scaled, reference_coef, target_coef


def _shared_resolution(reference, target) -> tuple[float, float]:
    reference_low, reference_high = reference.d_max_min()
    target_low, target_high = target.d_max_min()
    return min(reference_low, target_low), max(reference_high, target_high)


def _wilson_coefficients(array) -> np.ndarray:
    """Weighted least squares for ``ln I = a0 - b · v(s)``."""
    amplitude = np.asarray(array.data(), dtype=np.float64)
    sigma = np.asarray(array.sigmas(), dtype=np.float64) if array.sigmas() is not None else np.ones_like(amplitude)
    use = np.isfinite(amplitude) & (amplitude > 0) & np.isfinite(sigma) & (sigma > 0)
    intensity = amplitude[use] ** 2
    sigma_i = np.maximum(2.0 * amplitude[use] * sigma[use], 1e-8)
    weight = np.clip((intensity / sigma_i) ** 2, 1e-4, 1e6)
    design = np.column_stack([np.ones(int(use.sum())), -_quadratic_rows(array)[use]])
    weighted = np.sqrt(weight)[:, None]
    coefficients, *_ = np.linalg.lstsq(design * weighted, np.log(intensity) * weighted[:, 0], rcond=None)
    return coefficients


def _quadratic_rows(array) -> np.ndarray:
    cell = array.unit_cell()
    indices = np.array([(int(h), int(k), int(l)) for h, k, l in array.indices()], dtype=np.float64)
    orthogonal = np.linalg.inv(np.array(cell.orthogonalization_matrix()).reshape(3, 3)).T
    scattering = indices @ orthogonal.T
    x, y, z = scattering.T
    return np.column_stack([x * x, y * y, z * z, 2 * x * y, 2 * x * z, 2 * y * z])


def _amplitude_scale(array, reference_coef: np.ndarray, target_coef: np.ndarray) -> np.ndarray:
    rows = _quadratic_rows(array)
    ln_intensity_ratio = (reference_coef[0] - rows @ reference_coef[1:]) - (target_coef[0] - rows @ target_coef[1:])
    return np.exp(0.5 * ln_intensity_ratio)


def _apply_scale(array, scale: np.ndarray):
    data = np.asarray(array.data(), dtype=np.float64) * scale
    sigmas = None if array.sigmas() is None else np.asarray(array.sigmas(), dtype=np.float64) * scale
    scaled = array.customized_copy(
        data=flex.double(np.ascontiguousarray(data)),
        sigmas=None if sigmas is None else flex.double(np.ascontiguousarray(sigmas)),
    )
    scaled.set_observation_type_xray_amplitude()
    return scaled


def write_scaled(scaled, flags, dest: Path) -> None:
    """Write scaled amplitudes and, when present, the free-set flags."""
    dataset = scaled.as_mtz_dataset(column_root_label="FOBS")
    if flags is not None:
        paired = _pair(scaled, flags)
        if paired is not None:
            dataset.add_miller_array(paired, column_root_label="FreeR_flag")
    dataset.mtz_object().write(str(dest))


def _pair(scaled, flags):
    if flags.anomalous_flag():
        flags = flags.as_non_anomalous_array().merge_equivalents().array()
    flags = flags.map_to_asu()
    pairs = scaled.match_indices(flags).pairs()
    if pairs.size() == 0:
        return None
    values = flags.select(pairs.column(1)).data()
    full = flex.bool(scaled.size(), False)
    full.set_selected(pairs.column(0), values)
    return scaled.customized_copy(data=full, sigmas=None)


def run_one(folder: Path, reference, *, force: bool) -> str:
    """Scale one dataset. Return a one-line summary."""
    pdb_id = folder.name
    dest = folder / f"{pdb_id}_scaled_to_8uvz.mtz"
    if dest.is_file() and dest.stat().st_size > 0 and not force:
        return f"{pdb_id}: using existing scaled amplitudes"
    (target, flags), kind = maps.load_amplitudes(folder / f"{pdb_id}-sf.cif")
    scaled, reference_coef, target_coef = scale_to_reference(reference, target)
    write_scaled(scaled, flags, dest)
    delta = target_coef[1:] - reference_coef[1:]
    return (
        f"{pdb_id}: {kind}; n={scaled.size()}; "
        f"k0={math.exp(0.5 * (reference_coef[0] - target_coef[0])):.4f}; "
        f"ΔB=({delta[0]:.2f}, {delta[1]:.2f}, {delta[2]:.2f})"
    )


def reference_amplitudes(structures: Path):
    """Download 8UVZ if needed and return its amplitudes."""
    download.fetch_entry(_REFERENCE, structures, force=False)
    folder = structures / _REFERENCE
    (amplitudes, _flags), kind = maps.load_amplitudes(folder / f"{_REFERENCE}-sf.cif")
    print(f"reference {_REFERENCE}: {kind}; n={amplitudes.size()}")
    return amplitudes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Anisotropic LSQ scale of the median set onto 8UVZ.")
    parser.add_argument("--ids", type=Path, default=_ROOT / "median_ids.txt")
    parser.add_argument("--structures", type=Path, default=_STRUCTURES)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    reference = reference_amplitudes(args.structures)
    failures: list[str] = []
    for pdb_id in download.read_ids(args.ids):
        folder = args.structures / pdb_id
        if not (folder / f"{pdb_id}-sf.cif").is_file():
            failures.append(f"{pdb_id}: no structure factors")
            continue
        try:
            print(run_one(folder, reference, force=args.force))
        except Exception as exc:
            failures.append(f"{pdb_id}: {exc}")
            traceback.print_exc()
    print(f"done: {len(download.read_ids(args.ids)) - len(failures)} ok, {len(failures)} failed")
    for item in failures:
        print(f"failed: {item}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
