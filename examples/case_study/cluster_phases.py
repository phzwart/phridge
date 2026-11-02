#!/usr/bin/env python3
"""Split the with-water FWT weight UMAP into four clusters and rephase maps.

The embedding is components 1–4 of ``fwt_with_water``, the same coordinates as
``fwt_weight_umap.png``. Each cluster's centroid is the member nearest the mean
of that cluster in the UMAP plane. Every map in the cluster keeps its own FWT
amplitudes and takes PHWT from the centroid structure.

Usage::

    python examples/case_study/cluster_phases.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import gradient_svd
import iotbx.mrcfile
import iotbx.mtz
import numpy as np
from cctbx import crystal, maptbx
from cctbx.array_family import flex
from iotbx.mtz import label_decorator
from sklearn.cluster import KMeans

_ROOT = Path(__file__).resolve().parent
_SOURCE = _ROOT / "fwt_with_water" / "residuals"
_FRAMES = _ROOT / "svd_frame"
_OUT = _ROOT / "clusters" / "with_water_1-4"
_GRID = (120, 250, 480)
_UMAP = Path("/tmp/fwt_umap.json")


class _FwtLabels(label_decorator):
    def phases(self, root_label, anomalous_sign=None):
        return "PHWT"


def embedding() -> tuple[list[str], np.ndarray]:
    """PDB ids and the with-water components 1–4 UMAP coordinates."""
    block = json.loads(_UMAP.read_text(encoding="utf-8"))["fwt_with_water"]
    ids = [pdb_id.lower() for pdb_id in block["ids"]]
    return ids, np.asarray(block["panels"]["4"], dtype=np.float64)


def four_clusters(ids: list[str], xy: np.ndarray) -> list[dict]:
    """Four groups. The centroid is the member nearest the cluster mean."""
    labels = KMeans(n_clusters=4, n_init=50, random_state=0).fit_predict(xy)
    clusters = []
    for label in range(4):
        chosen = labels == label
        points = xy[chosen]
        members = [ids[index] for index in np.flatnonzero(chosen)]
        center = points.mean(axis=0)
        distance = np.linalg.norm(points - center, axis=1)
        nearest = int(np.argmin(distance))
        clusters.append(
            {
                "members": members,
                "center": center,
                "centroid": members[nearest],
                "distance": float(distance[nearest]),
            }
        )
    clusters.sort(key=lambda item: (-len(item["members"]), item["centroid"]))
    for index, cluster in enumerate(clusters, start=1):
        cluster["index"] = index
    return clusters


def read_fwt(path: Path):
    """Complex FWT/PHWT array. One miller set per residual MTZ."""
    arrays = iotbx.mtz.object(file_name=str(path)).as_miller_arrays()
    if len(arrays) != 1:
        raise RuntimeError(f"{path} has {len(arrays)} miller arrays")
    return arrays[0]


def rephased(member, centroid):
    """Member amplitudes with the centroid's phases. Indices must already match."""
    if not member.indices().all_eq(centroid.indices()):
        raise RuntimeError("FWT miller sets differ")
    member_values = np.asarray(member.data(), dtype=np.complex128)
    centroid_values = np.asarray(centroid.data(), dtype=np.complex128)
    combined = np.abs(member_values) * np.exp(1j * np.angle(centroid_values))
    data = flex.complex_double(np.ascontiguousarray(combined))
    return member.customized_copy(data=data, sigmas=None)


def _real_map(coefficients) -> np.ndarray:
    gridding = maptbx.crystal_gridding(
        unit_cell=coefficients.unit_cell(),
        space_group_info=coefficients.space_group_info(),
        pre_determined_n_real=_GRID,
    )
    values = np.array(coefficients.fft_map(crystal_gridding=gridding).real_map_unpadded(), dtype=np.float64)
    return values.reshape(_GRID, order="C")


def write_map(coefficients, path: Path, label: str) -> None:
    """Sigma-scaled CCP4 of the rephased coefficients, plus the FWT/PHWT MTZ."""
    symmetry = crystal.symmetry(unit_cell=coefficients.unit_cell().parameters(), space_group_symbol="P 21 21 21")
    iotbx.mrcfile.write_ccp4_map(
        file_name=str(path),
        crystal_symmetry=symmetry,
        map_data=gradient_svd._sigma_scaled(_real_map(coefficients).ravel(), _GRID),
        labels=flex.std_string([label]),
        unit_cell_grid=_GRID,
    )
    dataset = coefficients.as_mtz_dataset(
        column_root_label="FWT",
        label_decorator=_FwtLabels(phases_prefix="PH"),
        title=label[:70],
    )
    dataset.mtz_object().write(file_name=str(path.with_suffix(".mtz")))


def write_cluster(cluster: dict, phases) -> None:
    """Structures, member list, and rephased maps for one cluster."""
    name = f"{cluster['index']:02d}_{cluster['centroid']}"
    directory = _OUT / name
    structures = directory / "structures"
    maps = directory / "maps"
    structures.mkdir(parents=True, exist_ok=True)
    maps.mkdir(parents=True, exist_ok=True)
    ordered = [cluster["centroid"]] + sorted(pdb_id for pdb_id in cluster["members"] if pdb_id != cluster["centroid"])
    (directory / "members.txt").write_text("\n".join(ordered) + "\n", encoding="utf-8")
    center = cluster["center"]
    (directory / "centroid.txt").write_text(
        "\n".join(
            [
                f"centroid {cluster['centroid']}",
                f"n {len(cluster['members'])}",
                f"distance_to_center {cluster['distance']:.4f}",
                f"umap_center {center[0]:.4f} {center[1]:.4f}",
                "phases PHWT of the centroid FWT map",
                "amplitudes FWT of each member",
                "",
            ]
        ),
        encoding="utf-8",
    )
    for pdb_id in ordered:
        shutil.copy2(_FRAMES / f"{pdb_id}.pdb", structures / f"{pdb_id}.pdb")
        coefficients = rephased(read_fwt(_SOURCE / f"{pdb_id}.mtz"), phases)
        write_map(coefficients, maps / f"{pdb_id}.ccp4", f"{pdb_id} FWT phased by {cluster['centroid']}")
        print(f"  {name} {pdb_id}")


def write_assignments(ids: list[str], xy: np.ndarray, clusters: list[dict]) -> None:
    """Coordinates and the four assignments, centroid first in each member list."""
    _OUT.mkdir(parents=True, exist_ok=True)
    lines = ["id umap1 umap2 cluster centroid"]
    cluster_of = {}
    for cluster in clusters:
        for pdb_id in cluster["members"]:
            cluster_of[pdb_id] = cluster
    for pdb_id, point in zip(ids, xy):
        cluster = cluster_of[pdb_id]
        lines.append(
            f"{pdb_id} {point[0]:.4f} {point[1]:.4f} {cluster['index']:02d} {cluster['centroid']}"
        )
    (_OUT / "assignments.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ids, xy = embedding()
    clusters = four_clusters(ids, xy)
    write_assignments(ids, xy, clusters)
    for cluster in clusters:
        print(
            f"cluster {cluster['index']:02d} centroid {cluster['centroid']} "
            f"n={len(cluster['members'])} distance {cluster['distance']:.3f}"
        )
        phases = read_fwt(_SOURCE / f"{cluster['centroid']}.mtz")
        write_cluster(cluster, phases)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
