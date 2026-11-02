#!/usr/bin/env python3
"""SVD of each with-water cluster after scaling onto its centroid.

Observed amplitudes are anisotropically scaled onto the centroid dataset.
Maps are sigma-A weighted 2mFo−DFc in the SVD frame, then rephased with the
centroid map. Each map is zero-mean and unit-variance before the SVD.
Components are written as ``FWT`` / ``PHWT``.

Usage::

    python examples/case_study/cluster_svd.py
"""

from __future__ import annotations

from pathlib import Path

import cluster_phases
import gradient_svd
import iotbx.pdb
import maps
import numpy as np
import redo_svd
import svd_frame
from cctbx import crystal, miller

_ROOT = Path(__file__).resolve().parent
_CLUSTERS = _ROOT / "clusters" / "with_water_1-4"
_STRUCTURES = _ROOT / "structures"


def member_ids(directory: Path) -> list[str]:
    """Centroid first, then the other members, as written in ``members.txt``."""
    ids = []
    for raw in (directory / "members.txt").read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip().lower()
        if line:
            ids.append(line)
    if not ids:
        raise RuntimeError(f"no members in {directory}")
    return ids


def _scale(reference, mobile):
    """Anisotropic scale. A cell that cctbx rejects is placed in the reference symmetry and retried."""
    try:
        return redo_svd.scale_to_reference(reference.deep_copy(), mobile.deep_copy())
    except AssertionError:
        transferred = mobile.customized_copy(crystal_symmetry=reference.crystal_symmetry())
        return redo_svd.scale_to_reference(reference.deep_copy(), transferred.deep_copy())


def _flags(scaled, flags):
    """Free-R flags on the scaled array. The cell is copied across when cctbx rejects it."""
    if flags is None:
        return None
    if not scaled.is_similar_symmetry(flags):
        flags = flags.customized_copy(crystal_symmetry=scaled.crystal_symmetry())
    return redo_svd._flags_for(scaled, flags)


def warped_map(folder: Path, reference, pose: dict[int, np.ndarray]) -> tuple[np.ndarray, str, float, float]:
    """FWT map scaled onto ``reference`` amplitudes, still carrying this model's phases."""
    (amplitudes, flags), kind = maps.load_amplitudes(folder / f"{folder.name}-sf.cif")
    scaled, driver = _scale(reference, amplitudes)
    structure = redo_svd.prepared_structure(svd_frame.model_path(folder), True)
    real_map, native_cell = redo_svd.density_map(structure, scaled, _flags(scaled, flags))
    loaded = iotbx.pdb.input(file_name=str(svd_frame.model_path(folder)))
    anchor = svd_frame.anchor_chain(loaded.construct_hierarchy())
    rotation, translation, residual = svd_frame.place(
        svd_frame.trace(anchor, loaded.crystal_symmetry().unit_cell()),
        pose,
    )
    warped = redo_svd.warp_map(real_map, native_cell, rotation, translation, redo_svd._GRID)
    warped -= np.float32(warped.mean())
    deviation = float(warped.std())
    if deviation > 0.0:
        warped /= np.float32(deviation)
    print(
        f"{folder.name}: {kind}; scale R {driver.r_val_before:.3f} -> {driver.r_val_after:.3f}; fit {residual:.3f} A"
    )
    return warped.astype(np.float32, copy=False), kind, float(driver.r_val_before), float(driver.r_val_after)


def coefficients_of(values: np.ndarray, cell: tuple[float, ...]):
    """Structure factors of one real-space map on the common grid."""
    data = gradient_svd._sigma_scaled(values, redo_svd._GRID)
    symmetry = crystal.symmetry(unit_cell=cell, space_group_symbol="P 21 21 21")
    nyquist = max(cell[index] * 2.0 / redo_svd._GRID[index] for index in range(3))
    return miller.build_set(
        crystal_symmetry=symmetry,
        anomalous_flag=False,
        d_min=nyquist,
    ).structure_factors_from_map(map=data, use_scale=True, anomalous_flag=False, use_sg=True)


def _normalize(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float32).ravel()
    flat -= np.float32(flat.mean())
    deviation = float(flat.std())
    if deviation > 0.0:
        flat /= np.float32(deviation)
    return flat


def collect(directory: Path, pose: dict[int, np.ndarray], cell: tuple[float, ...]) -> tuple[list[str], np.ndarray]:
    """Scaled, centroid-phased maps. Also refreshes ``maps/`` and writes ``scale.txt``."""
    ids = member_ids(directory)
    centroid = ids[0]
    cache = directory / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    cached: dict[str, np.ndarray] = {}
    scale_lines = ["id kind r_before r_after reference"]
    for pdb_id in ids:
        stored = cache / f"{pdb_id}.npy"
        note = cache / f"{pdb_id}.scale"
        if stored.is_file() and note.is_file():
            kind, before_text, after_text = note.read_text(encoding="utf-8").split()
            warped = np.load(stored)
            before, after = float(before_text), float(after_text)
            print(f"{pdb_id}: cached; scale R {before:.3f} -> {after:.3f}")
        else:
            reference = maps.load_amplitudes(_STRUCTURES / centroid / f"{centroid}-sf.cif")[0][0]
            warped, kind, before, after = warped_map(_STRUCTURES / pdb_id, reference, pose)
            np.save(stored, warped)
            note.write_text(f"{kind} {before:.6f} {after:.6f}\n", encoding="utf-8")
        cached[pdb_id] = warped
        scale_lines.append(f"{pdb_id} {kind} {before:.4f} {after:.4f} {centroid}")
    (directory / "scale.txt").write_text("\n".join(scale_lines) + "\n", encoding="utf-8")
    transformed = {pdb_id: coefficients_of(cached[pdb_id], cell) for pdb_id in ids}
    phases = transformed[centroid]
    maps_dir = directory / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    columns = []
    for pdb_id in ids:
        combined = cluster_phases.rephased(transformed[pdb_id], phases)
        label = f"{pdb_id} FWT scaled to {centroid}, phased by {centroid}"
        cluster_phases.write_map(combined, maps_dir / f"{pdb_id}.ccp4", label)
        columns.append(_normalize(cluster_phases._real_map(combined)))
        print(f"  rephased {pdb_id}")
    return ids, np.column_stack(columns)


def main() -> int:
    pose = redo_svd._pose()
    cell = svd_frame.map_cell()
    for directory in sorted(path for path in _CLUSTERS.iterdir() if (path / "members.txt").is_file()):
        loadings = directory / "svd" / "loadings.txt"
        if loadings.is_file() and len(loadings.read_text(encoding="utf-8").splitlines()) - 1 == len(member_ids(directory)):
            print(f"--- {directory.name} already written ---")
            continue
        print(f"--- {directory.name} ---")
        ids, columns = collect(directory, pose, cell)
        out = directory / "svd"
        redo_svd._write_set(out, ids, columns)
        print(f"wrote {len(ids)} components to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
