#!/usr/bin/env python3
"""The FWT analysis on the molecule-centred box.

1. FWT maps, each crystal with its own model, amplitudes scaled onto 8V0C,
   with and without waters. SVD of each set, then the average map, residuals
   from it, residual SVD, and power maps in ``residual_svd/``.
2. UMAP of the singular-value-weighted loadings on components 1–4, 1–5, 1–10.
3. Four clusters from with-water components 1–4. The centroid is the member
   nearest the cluster mean.
4. Per cluster: each member's 2mFo−DFc with its own model, Fo scaled onto the
   centroid. Average, residuals, residual SVD, power maps.

All maps are sampled through each crystal's own superposition on the 6DHU dimer.
Map vectors are cached as ``structures/{id}/box_*.npy`` so a rerun skips them.

Usage::

    source .venv/bin/activate
    python examples/case_study/box_pipeline.py
    python examples/case_study/box_pipeline.py --residuals
"""

from __future__ import annotations

import json
import shutil
import traceback
from pathlib import Path

import box_frame
import cluster_svd
import gradient_svd
import maps
import numpy as np
import redo_svd

_ROOT = Path(__file__).resolve().parent
_STRUCTURES = _ROOT / "structures"
_OUT = _ROOT / "box"
_REFERENCE = "8v0c"
_SETS = (("fwt_with_water", True), ("fwt_no_water", False))
_PANELS = (4, 5, 10)


def structure_ids() -> list[str]:
    ids = []
    for folder in sorted(path for path in _STRUCTURES.iterdir() if path.is_dir()):
        if folder.name == "8uvz" or not (folder / f"{folder.name}-sf.cif").is_file():
            continue
        if redo_svd.svd_frame.model_path(folder) is not None:
            ids.append(folder.name)
    return ids


def fwt_vector(crystal_id: str, model_id: str, reference_id: str, keep_water: bool) -> np.ndarray:
    """Normalized box map of ``crystal_id`` Fo against ``model_id``, Fo scaled onto ``reference_id``."""
    reference = maps.load_amplitudes(_STRUCTURES / reference_id / f"{reference_id}-sf.cif")[0][0]
    (amplitudes, flags), kind = maps.load_amplitudes(_STRUCTURES / crystal_id / f"{crystal_id}-sf.cif")
    scaled, driver = cluster_svd._scale(reference, amplitudes)
    if model_id == crystal_id:
        structure = box_frame.native_structure(crystal_id, keep_water)
    else:
        structure = box_frame.placed_structure(model_id, crystal_id, keep_water)
    real_map, native_cell = redo_svd.density_map(structure, scaled, cluster_svd._flags(scaled, flags))
    rotation, translation, rmsd = box_frame.fit(crystal_id)
    values = box_frame.normalized(box_frame.sample(real_map, native_cell, rotation, translation))
    print(
        f"{crystal_id} model {model_id}: {kind}; Fo to {reference_id} R {driver.r_val_before:.3f} -> "
        f"{driver.r_val_after:.3f}; fit {rmsd:.2f} A"
    )
    return values


def cached_vector(crystal_id: str, model_id: str, reference_id: str, keep_water: bool, path: Path) -> np.ndarray:
    """Masked map vector, cached in the structure folder. The full map is written to ``path``."""
    water = "water" if keep_water else "nowater"
    cache = _STRUCTURES / crystal_id / f"box_{model_id}_{reference_id}_{water}.npy"
    if cache.is_file() and path.is_file():
        return np.load(cache)
    values = fwt_vector(crystal_id, model_id, reference_id, keep_water)
    label = f"{crystal_id} Fo, {model_id} model, scaled to {reference_id}"
    box_frame.write_map(values, path, label)
    masked = values[box_frame.mask()]
    np.save(cache, masked)
    return masked


def write_svd(directory: Path, ids: list[str], columns: np.ndarray, title: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Components of ``columns`` as box maps, plus ``loadings.txt``. Null modes are not written."""
    singular, spatial, structures = gradient_svd.singular_vectors(columns)
    total = float(np.sum(singular**2))
    for stale in directory.glob("component_*"):
        stale.unlink()
    for index, value in enumerate(singular):
        fraction = float(value**2 / total) if total else 0.0
        if fraction < 1e-5:
            continue
        box_frame.write_map(
            box_frame.expand(spatial[:, index]),
            directory / f"component_{index + 1:02d}.ccp4",
            f"{title} component {index + 1} fraction {fraction:.4f}",
        )
    gradient_svd.write_loadings(directory / "loadings.txt", ids, singular, structures)
    return singular, spatial, structures


def own_sets(ids: list[str]) -> dict[str, tuple[list[str], np.ndarray, np.ndarray]]:
    """Step 1. Returns, per set, the ids, singular values, and structure vectors."""
    results = {}
    for name, keep_water in _SETS:
        print(f"--- {name} ---")
        directory = _OUT / name
        used, columns = [], []
        for pdb_id in ids:
            try:
                columns.append(cached_vector(pdb_id, pdb_id, _REFERENCE, keep_water, directory / "maps" / f"{pdb_id}.ccp4"))
            except Exception as exc:
                print(f"failed {pdb_id}: {exc}")
                traceback.print_exc()
                continue
            used.append(pdb_id)
        stack = np.column_stack(columns)
        stack -= stack.mean(axis=0, keepdims=True)
        singular, _spatial, structures = write_svd(directory, used, stack, f"box FWT {name}")
        results[name] = (used, singular, structures)
    return results


def embeddings(results: dict) -> dict:
    """Step 2. UMAP coordinates per set and panel."""
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
    plot_umap(output, _OUT / "fwt_weight_umap.png")
    return output


def plot_umap(data: dict, path: Path, title: str = "FWT SVD weight manifold, molecule box") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = {"1": "#c2410c", "6": "#1d4ed8"}
    figure, axes = plt.subplots(2, 3, figsize=(14.2, 9.4))
    for row, (name, title) in enumerate((("fwt_with_water", "With water"), ("fwt_no_water", "No water"))):
        block = data[name]
        for column, count in enumerate(_PANELS):
            axis = axes[row, column]
            points = np.array(block["panels"][str(count)])
            ids = [pdb_id.upper() for pdb_id in block["ids"]]
            axis.scatter(points[:, 0], points[:, 1], c=[colors.get(i[0], "#047857") for i in ids], s=36,
                         linewidths=0.4, edgecolors="white", zorder=3)
            for (x, y), pdb_id in zip(points, ids):
                axis.annotate(pdb_id, (x, y), textcoords="offset points", xytext=(4, 3), fontsize=5.5, color="#334155")
            axis.set_title(f"{title}, components 1–{count}", fontsize=11)
            axis.set_xlabel("UMAP 1")
            axis.set_ylabel("UMAP 2")
            axis.tick_params(labelsize=8)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor="white", markersize=8, label=label)
        for color, label in (("#c2410c", "1 series"), ("#1d4ed8", "6 series"), ("#047857", "7–9 series"))
    ]
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    figure.suptitle(title, fontsize=14)
    figure.tight_layout(rect=(0, 0.03, 1, 0.96))
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def clusters(data: dict) -> list[dict]:
    """Step 3. Four k-means groups on with-water components 1–4; Ward linkage is reported for comparison."""
    from sklearn.cluster import AgglomerativeClustering, KMeans

    block = data["fwt_with_water"]
    ids = block["ids"]
    points = np.array(block["panels"]["4"], dtype=np.float64)
    labels = KMeans(n_clusters=4, n_init=50, random_state=0).fit_predict(points)
    ward = AgglomerativeClustering(n_clusters=4, linkage="ward").fit_predict(points)
    same = len({(a, b) for a, b in zip(labels, ward)}) == 4
    print(f"k-means and Ward agree: {same}")
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
        for pdb_id in group["members"]:
            box_frame.write_model(pdb_id, group["directory"] / "structures")
    for pdb_id, point in zip(ids, points):
        index, group = next((i, g) for i, g in enumerate(groups, start=1) if pdb_id in g["members"])
        lines.append(f"{pdb_id} {point[0]:.4f} {point[1]:.4f} {index:02d} {group['centroid']}")
    (root / "assignments.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return groups


def residue_power(power: np.ndarray, model: Path) -> list[tuple[float, str]]:
    """Mean power at each residue's non-water atoms in the box model, highest first."""
    import iotbx.pdb

    _low, grid, _cell = box_frame.geometry()
    volume = power.reshape(grid)
    sums: dict[str, list[float]] = {}
    hierarchy = iotbx.pdb.input(file_name=str(model)).construct_hierarchy()
    for atom in hierarchy.atoms():
        group = atom.parent()
        if group.resname.strip() in box_frame._WATER:
            continue
        index = np.clip(np.rint(np.array(atom.xyz) / box_frame._SPACING).astype(int), 0, np.array(grid) - 1)
        key = f"{group.resname.strip()} {group.parent().parent().id.strip()}{group.parent().resseq.strip()}"
        sums.setdefault(key, []).append(float(volume[index[0], index[1], index[2]]))
    return sorted(((float(np.mean(values)), key) for key, values in sums.items()), reverse=True)


def cluster_maps(group: dict) -> None:
    """Step 4 for one cluster: each member with its own model, Fo scaled onto the centroid."""
    directory = group["directory"]
    centroid = group["centroid"]
    ids, columns = [], []
    for pdb_id in group["members"]:
        try:
            columns.append(cached_vector(pdb_id, pdb_id, centroid, True, directory / "maps" / f"{pdb_id}.ccp4"))
        except Exception as exc:
            print(f"failed {pdb_id}: {exc}")
            traceback.print_exc()
            continue
        ids.append(pdb_id)
    model = next((directory / "structures").glob(f"{centroid}.*"))
    residual_svd(directory / "residual_svd", ids, np.column_stack(columns), directory.name, model)


def residual_svd(svd_dir: Path, ids: list[str], stack: np.ndarray, title: str, model: Path) -> None:
    """Average map, residuals from it, their SVD, and power maps."""
    mean = stack.mean(axis=1)
    residuals = stack - mean[:, None]
    box_frame.write_map(box_frame.expand(mean), svd_dir / "average.ccp4", f"{title} average")
    for pdb_id, values in zip(ids, residuals.T):
        box_frame.write_map(box_frame.expand(values), svd_dir / "residuals" / f"{pdb_id}.ccp4", f"{pdb_id} minus average")
    singular, spatial, _structures = write_svd(svd_dir, ids, residuals, f"{title} residual")
    for label, count in (("all", singular.size), ("1-4", 4)):
        power = np.sum((spatial[:, :count] ** 2) * (singular[None, :count] ** 2), axis=1)
        power = box_frame.expand((power / power.mean()).astype(np.float32))
        box_frame.write_map(power, svd_dir / f"power_{label}.ccp4", f"{title} power {label}", coefficients=False)
        ranked = residue_power(power, model)
        lines = ["power residue"] + [f"{value:.3f} {key}" for value, key in ranked]
        (svd_dir / f"power_{label}_residues.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        if label == "all":
            print(f"{title} top power: " + ", ".join(f"{key} {value:.1f}" for value, key in ranked[:8]))


def existing_clusters() -> list[dict]:
    groups = []
    for directory in sorted(path for path in (_OUT / "clusters").iterdir() if (path / "members.txt").is_file()):
        members = (directory / "members.txt").read_text(encoding="utf-8").split()
        groups.append({"directory": directory, "centroid": members[0], "members": members})
    return groups


def own_residuals() -> None:
    """Average and residual SVD of the own-model sets, on the cached maps."""
    for name, keep_water in _SETS:
        directory = _OUT / name
        ids = (directory / "loadings.txt").read_text(encoding="utf-8").splitlines()[0].split()[3:]
        columns = [cached_vector(pdb_id, pdb_id, _REFERENCE, keep_water, directory / "maps" / f"{pdb_id}.ccp4") for pdb_id in ids]
        print(f"--- {name} residual ---")
        residual_svd(directory / "residual_svd", ids, np.column_stack(columns), name, _OUT / "models" / f"{box_frame._POSE}.pdb")


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--residuals", action="store_true", help="steps 1 and 4 residual SVDs only, on the existing clusters")
    args = parser.parse_args(argv)
    if args.residuals:
        own_residuals()
        groups = existing_clusters()
    else:
        ids = structure_ids()
        _OUT.mkdir(exist_ok=True)
        for pdb_id in ids:
            box_frame.write_model(pdb_id, _OUT / "models")
        results = own_sets(ids)
        own_residuals()
        groups = clusters(embeddings(results))
    for group in groups:
        print(f"--- cluster {group['directory'].name}: {' '.join(group['members'])} ---")
        cluster_maps(group)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
