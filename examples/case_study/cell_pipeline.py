#!/usr/bin/env python3
"""The FWT analysis on the median P2₁2₁2₁ cell, one Fourier grid for every map.

1. Each crystal's 2mFo−DFc with its own model on the placement onto 8V0C, Fo
   scaled onto 8V0C, with and without waters. Coefficients are indexed on the
   median cell and every map is an FFT on the shared grid.
2. Per set: the average map, each map minus the average, the SVD of those
   residuals, and power maps.
3. UMAP of the singular-value-weighted residual loadings on components 1–4,
   1–5, and 1–10.
4. Four k-means clusters on with-water components 1–4. The centroid is the
   member nearest the cluster mean. Per cluster, step 2 again with Fo scaled
   onto the centroid.

Maps are normalized to zero mean and unit σ inside a 5 Å mask around the 8V0C
dimer. Average, residual, and component maps are written as MTZ coefficients
built from the member coefficients, so they carry no mask edge; each is scaled
to unit σ inside the mask. Power maps are CCP4.

Usage::

    source .venv/bin/activate
    python examples/case_study/cell_pipeline.py
"""

from __future__ import annotations

import json
import shutil
import traceback
from pathlib import Path

import box_pipeline
import cell_frame
import cluster_svd
import gradient_svd
import iotbx.pdb
import maps
import numpy as np
import redo_svd

_ROOT = Path(__file__).resolve().parent
_STRUCTURES = _ROOT / "structures"
_OUT = _ROOT / "cell"
_SETS = (("fwt_with_water", True), ("fwt_no_water", False))
_PANELS = (4, 5, 10)


def member(crystal_id: str, reference_id: str, keep_water: bool) -> tuple[np.ndarray, np.ndarray]:
    """Masked map and common-set coefficients, both normalized inside the mask. Cached."""
    water = "water" if keep_water else "nowater"
    cache = _STRUCTURES / crystal_id / f"cell_{reference_id}_{water}.npz"
    if cache.is_file():
        stored = np.load(cache)
        return stored["vector"], stored["coefficients"]
    reference = maps.load_amplitudes(_STRUCTURES / reference_id / f"{reference_id}-sf.cif")[0][0]
    (amplitudes, flags), kind = maps.load_amplitudes(_STRUCTURES / crystal_id / f"{crystal_id}-sf.cif")
    scaled, driver = cluster_svd._scale(reference, amplitudes)
    structure = cell_frame.placed_structure(crystal_id, keep_water)
    coefficients = cell_frame.on_common(redo_svd.model_coefficients(structure, scaled, cluster_svd._flags(scaled, flags)))
    values = cell_frame.real(coefficients)
    mean, deviation = cell_frame.masked_stats(values)
    vector = ((values[cell_frame.mask()] - mean) / deviation).astype(np.float32)
    coefficients = (coefficients / deviation).astype(np.complex64)
    np.savez(cache, vector=vector, coefficients=coefficients)
    place = cell_frame.placement(crystal_id)
    print(
        f"{crystal_id}: {kind}; Fo to {reference_id} R {driver.r_val_before:.3f} -> {driver.r_val_after:.3f}; "
        f"placement rmsd {place.rmsd:.2f} A, chains swapped {place.swap}"
    )
    return vector, coefficients


def gather(ids: list[str], reference_id: str, keep_water: bool, directory: Path) -> tuple[list, np.ndarray, np.ndarray]:
    used, vectors, coefficients = [], [], []
    for pdb_id in ids:
        try:
            vector, values = member(pdb_id, reference_id, keep_water)
        except Exception as exc:
            print(f"failed {pdb_id}: {exc}")
            traceback.print_exc()
            continue
        cell_frame.write_mtz(values, directory / "maps" / f"{pdb_id}.mtz", f"{pdb_id} FWT, Fo scaled to {reference_id}")
        used.append(pdb_id)
        vectors.append(vector)
        coefficients.append(values)
    return used, np.column_stack(vectors), np.vstack(coefficients)


def residue_power(power: np.ndarray, model: Path) -> list[tuple[float, str]]:
    """Mean power at each residue's non-water atoms, highest first."""
    sums: dict[str, list[float]] = {}
    hierarchy = iotbx.pdb.input(file_name=str(model)).construct_hierarchy()
    for atom in hierarchy.atoms():
        group = atom.parent()
        if group.resname.strip() in cell_frame.box_frame._WATER:
            continue
        key = f"{group.resname.strip()} {group.parent().parent().id.strip()}{group.parent().resseq.strip()}"
        sums.setdefault(key, []).append(float(power[cell_frame.grid_index(atom.xyz)]))
    return sorted(((float(np.mean(values)), key) for key, values in sums.items()), reverse=True)


def _unit(coefficients: np.ndarray, masked: np.ndarray) -> np.ndarray:
    deviation = float(masked.std())
    return coefficients / deviation if deviation > 0.0 else coefficients


def residual_svd(directory: Path, ids: list[str], vectors: np.ndarray, coefficients: np.ndarray, title: str,
                 model: Path) -> tuple[np.ndarray, np.ndarray]:
    """Average, residuals, residual SVD, and power maps in ``directory / residual_svd``."""
    out = directory / "residual_svd"
    if out.is_dir():
        shutil.rmtree(out)
    mean_vector = vectors.mean(axis=1)
    mean_coefficients = coefficients.mean(axis=0)
    residuals = vectors - mean_vector[:, None]
    residual_coefficients = coefficients - mean_coefficients[None, :]
    cell_frame.write_mtz(_unit(mean_coefficients, mean_vector), out / "average.mtz", f"{title} average")
    for index, pdb_id in enumerate(ids):
        cell_frame.write_mtz(_unit(residual_coefficients[index], residuals[:, index]),
                             out / "residuals" / f"{pdb_id}.mtz", f"{pdb_id} minus average")
    singular, spatial, structures = gradient_svd.singular_vectors(residuals)
    total = float(np.sum(singular**2))
    for index, value in enumerate(singular):
        if total == 0.0 or value**2 / total < 1e-5:
            continue
        component = (structures[:, index] @ residual_coefficients) / value
        cell_frame.write_mtz(_unit(component, spatial[:, index]), out / f"component_{index + 1:02d}.mtz",
                             f"{title} residual component {index + 1} fraction {value**2 / total:.4f}")
    gradient_svd.write_loadings(out / "loadings.txt", ids, singular, structures)
    for label, count in (("all", singular.size), ("1-4", 4)):
        power = np.sum((spatial[:, :count] ** 2) * (singular[None, :count] ** 2), axis=1)
        power = cell_frame.expand((power / power.mean()).astype(np.float32))
        cell_frame.write_ccp4(power, out / f"power_{label}.ccp4", f"{title} power {label}")
        ranked = residue_power(power, model)
        lines = ["power residue"] + [f"{value:.3f} {key}" for value, key in ranked]
        (out / f"power_{label}_residues.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        if label == "all":
            print(f"{title} top power: " + ", ".join(f"{key} {value:.1f}" for value, key in ranked[:8]))
    return singular, structures


def embeddings(results: dict) -> dict:
    import umap

    output = {}
    for name, (ids, singular, structures) in results.items():
        panels = {}
        for count in _PANELS:
            features = structures[:, :count] * singular[None, :count]
            reducer = umap.UMAP(n_neighbors=min(10, len(ids) - 1), min_dist=0.15, random_state=0)
            panels[str(count)] = [[round(float(x), 4), round(float(y), 4)] for x, y in reducer.fit_transform(features)]
        output[name] = {"ids": ids, "panels": panels}
    (_OUT / "fwt_umap.json").write_text(json.dumps(output), encoding="utf-8")
    box_pipeline.plot_umap(output, _OUT / "fwt_weight_umap.png", "Residual SVD weight manifold, median cell")
    return output


def clusters(data: dict) -> list[dict]:
    from sklearn.cluster import KMeans

    block = data["fwt_with_water"]
    ids = block["ids"]
    points = np.array(block["panels"]["4"], dtype=np.float64)
    labels = KMeans(n_clusters=4, n_init=50, random_state=0).fit_predict(points)
    groups = []
    for label in range(4):
        chosen = np.flatnonzero(labels == label)
        center = points[chosen].mean(axis=0)
        distance = np.linalg.norm(points[chosen] - center, axis=1)
        centroid = ids[chosen[int(np.argmin(distance))]]
        others = sorted(ids[index] for index in chosen if ids[index] != centroid)
        groups.append({"centroid": centroid, "members": [centroid] + others, "distance": float(distance.min()),
                       "center": center})
    groups.sort(key=lambda group: (-len(group["members"]), group["centroid"]))
    root = _OUT / "clusters"
    if root.is_dir():
        shutil.rmtree(root)
    lines = ["id umap1 umap2 cluster centroid"]
    for index, group in enumerate(groups, start=1):
        group["directory"] = root / f"{index:02d}_{group['centroid']}"
        group["directory"].mkdir(parents=True)
        (group["directory"] / "members.txt").write_text("\n".join(group["members"]) + "\n", encoding="utf-8")
        (group["directory"] / "centroid.txt").write_text(
            f"centroid {group['centroid']}\nn {len(group['members'])}\ndistance_to_center {group['distance']:.4f}\n"
            f"umap_center {group['center'][0]:.4f} {group['center'][1]:.4f}\n", encoding="utf-8")
        (group["directory"] / "structures").mkdir()
        for pdb_id in group["members"]:
            shutil.copy(next((_OUT / "models").glob(f"{pdb_id}.*")), group["directory"] / "structures")
    for pdb_id, point in zip(ids, points):
        index, group = next((i, g) for i, g in enumerate(groups, start=1) if pdb_id in g["members"])
        lines.append(f"{pdb_id} {point[0]:.4f} {point[1]:.4f} {index:02d} {group['centroid']}")
    (root / "assignments.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return groups


def main() -> int:
    ids = box_pipeline.structure_ids()
    _OUT.mkdir(exist_ok=True)
    print(f"median cell {cell_frame.symmetry().unit_cell().parameters()}, grid {cell_frame.grid()}, "
          f"d_min {cell_frame.d_min():.2f}, mask {int(cell_frame.mask().sum())} points")
    for pdb_id in ids:
        cell_frame.write_model(pdb_id, _OUT / "models")
    pose_model = next((_OUT / "models").glob(f"{cell_frame.POSE}.*"))
    results = {}
    for name, keep_water in _SETS:
        print(f"--- {name} ---")
        directory = _OUT / name
        used, vectors, coefficients = gather(ids, cell_frame.POSE, keep_water, directory)
        singular, structures = residual_svd(directory, used, vectors, coefficients, name, pose_model)
        results[name] = (used, singular, structures)
    groups = clusters(embeddings(results))
    for group in groups:
        print(f"--- cluster {group['directory'].name}: {' '.join(group['members'])} ---")
        directory = group["directory"]
        used, vectors, coefficients = gather(group["members"], group["centroid"], True, directory)
        model = next((directory / "structures").glob(f"{group['centroid']}.*"))
        residual_svd(directory, used, vectors, coefficients, directory.name, model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
