#!/usr/bin/env python3
"""Centroid-model 2mFo−DFc maps for every cluster member, then the residual SVD.

Each member's observed amplitudes are scaled onto the centroid dataset. The
centroid model, with waters, is moved into the member's crystal frame. Fc,
m, and σA come from that model against the member's amplitudes, so the phase
is the centroid's while Fo is the member's. Features the centroid model lacks
appear through 2mFo−DFc. The maps are warped into the SVD frame,
normalized, averaged, and the mean-subtracted residuals are factored.

Usage::

    python examples/case_study/cluster_model_phases.py
    python examples/case_study/cluster_model_phases.py --only 6n0d
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cluster_residual_svd
import cluster_svd
import iotbx.pdb
import maps
import numpy as np
import redo_svd
import svd_frame
from scitbx.array_family import flex

_ROOT = Path(__file__).resolve().parent
_CLUSTERS = _ROOT / "clusters" / "with_water_1-4"
_STRUCTURES = _ROOT / "structures"


def frame_transform(path: Path, pose: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Rotation and translation from this crystal's Cartesian frame into the SVD frame."""
    loaded = iotbx.pdb.input(file_name=str(path))
    anchor = svd_frame.anchor_chain(loaded.construct_hierarchy())
    rotation, translation, _residual = svd_frame.place(
        svd_frame.trace(anchor, loaded.crystal_symmetry().unit_cell()),
        pose,
    )
    return rotation, translation


def centroid_in_member(centroid: str, member: str, pose: dict[int, np.ndarray]):
    """Centroid dimer, waters kept, placed on the member's chain A in the member's cell."""
    centroid_path = svd_frame.model_path(_STRUCTURES / centroid)
    member_path = svd_frame.model_path(_STRUCTURES / member)
    rotation_c, translation_c = frame_transform(centroid_path, pose)
    rotation_m, translation_m = frame_transform(member_path, pose)
    hierarchy = iotbx.pdb.input(file_name=str(centroid_path)).construct_hierarchy()
    xyz = np.array(hierarchy.atoms().extract_xyz(), dtype=np.float64)
    common = xyz @ rotation_c.T + translation_c
    native = (common - translation_m) @ rotation_m
    hierarchy.atoms().set_xyz(flex.vec3_double(np.ascontiguousarray(native)))
    symmetry = iotbx.pdb.input(file_name=str(member_path)).crystal_symmetry()
    return hierarchy.extract_xray_structure(crystal_symmetry=symmetry), rotation_m, translation_m


def member_map(centroid: str, member: str, pose: dict[int, np.ndarray]) -> np.ndarray:
    """Member Fo, centroid model phases, warped into the SVD frame and normalized."""
    reference = maps.load_amplitudes(_STRUCTURES / centroid / f"{centroid}-sf.cif")[0][0]
    (amplitudes, flags), kind = maps.load_amplitudes(_STRUCTURES / member / f"{member}-sf.cif")
    scaled, driver = cluster_svd._scale(reference, amplitudes)
    structure, rotation, translation = centroid_in_member(centroid, member, pose)
    real_map, native_cell = redo_svd.density_map(structure, scaled, cluster_svd._flags(scaled, flags))
    warped = redo_svd.warp_map(real_map, native_cell, rotation, translation, redo_svd._GRID)
    print(f"{member}: {kind}; Fo scaled to {centroid} R {driver.r_val_before:.3f} -> {driver.r_val_after:.3f}")
    return cluster_svd._normalize(warped)


def one_cluster(directory: Path, pose: dict[int, np.ndarray], cell: tuple[float, ...], only: str) -> None:
    """Maps, average, residuals, and SVD under ``centroid_model/``."""
    ids = cluster_svd.member_ids(directory)
    centroid = ids[0]
    if only:
        ids = [pdb_id for pdb_id in ids if pdb_id == only]
        if not ids:
            return
    out = directory / "centroid_model"
    maps_dir = out / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    columns = []
    for pdb_id in ids:
        values = member_map(centroid, pdb_id, pose)
        label = f"{pdb_id} Fo, {centroid} model 2mFo-DFc"
        redo_svd.write_fwt(values, redo_svd._GRID, cell, maps_dir / f"{pdb_id}.ccp4", label)
        columns.append(values)
    if only:
        return
    stack = np.column_stack(columns)
    mean = stack.mean(axis=1).astype(np.float32)
    residuals = stack - mean[:, None]
    svd_dir = out / "residual_svd"
    svd_dir.mkdir(parents=True, exist_ok=True)
    redo_svd.write_fwt(mean, redo_svd._GRID, cell, svd_dir / "average.ccp4", f"{directory.name} average")
    cluster_residual_svd._write_columns(svd_dir / "residuals", ids, residuals, cell, "minus cluster average")
    cluster_residual_svd._write_modes(svd_dir, ids, residuals, cell)
    print(f"wrote centroid-model residual SVD for {directory.name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Centroid-model 2mFo-DFc maps and residual SVD per cluster.")
    parser.add_argument("--only", default="")
    args = parser.parse_args(argv)
    pose = redo_svd._pose()
    cell = svd_frame.map_cell()
    for directory in sorted(path for path in _CLUSTERS.iterdir() if (path / "members.txt").is_file()):
        print(f"--- {directory.name} ---")
        one_cluster(directory, pose, cell, args.only.lower())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
