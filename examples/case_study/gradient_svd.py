#!/usr/bin/env python3
"""SVD of the median-set gradient maps, written back as CCP4 components.

The 16 maps are resampled onto the median unit cell and the median grid.
Each map has its voxel mean removed, then the maps are factored as a
voxel-by-structure matrix. Each spatial singular vector is sigma-scaled and
written as ``gradient_svd/component_XX.ccp4`` and as
``gradient_svd/component_XX.mtz`` with difference-map columns ``FOFCWT`` and
``PHFOFCWT``. ``gradient_svd/loadings.txt`` lists the singular values and the weight of
each structure.

Usage::

    python examples/case_study/gradient_svd.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cctbx  # noqa: F401
import iotbx.ccp4_map
import iotbx.mrcfile
import numpy as np
from cctbx import crystal, miller, maptbx, uctbx
from scitbx.array_family import flex

_ROOT = Path(__file__).resolve().parent
_STRUCTURES = _ROOT / "structures"
_OUT = _ROOT / "gradient_svd"


def read_ids(path: Path) -> list[str]:
    """One PDB id per line."""
    ids = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip().lower()
        if line:
            ids.append(line)
    if not ids:
        raise SystemExit(f"no PDB IDs in {path}")
    return ids


def map_path(structures: Path, pdb_id: str) -> Path:
    path = structures / pdb_id / f"{pdb_id}_gradient.ccp4"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def common_frame(structures: Path, ids: list[str]) -> tuple[tuple[float, ...], tuple[int, int, int]]:
    """Median cell and median grid of the gradient maps."""
    cells = []
    grids = []
    for pdb_id in ids:
        reader = iotbx.ccp4_map.map_reader(file_name=str(map_path(structures, pdb_id)))
        cells.append(reader.unit_cell_parameters)
        grids.append(reader.unit_cell_grid)
    cell = tuple(float(value) for value in np.median(np.asarray(cells, dtype=np.float64), axis=0))
    grid = tuple(int(round(value)) for value in np.median(np.asarray(grids, dtype=np.float64), axis=0))
    return cell, grid


def resampled(path: Path, grid: tuple[int, int, int]) -> np.ndarray:
    """Fractional resample of one CCP4 map onto ``grid``. The map's own cell is the metric."""
    reader = iotbx.ccp4_map.map_reader(file_name=str(path))
    destination = flex.double(flex.grid(*grid))
    maptbx.resample(reader.data.as_double(), destination, uctbx.unit_cell(reader.unit_cell_parameters))
    return np.array(destination, dtype=np.float32).ravel()


def stack_maps(structures: Path, ids: list[str], grid: tuple[int, int, int]) -> np.ndarray:
    """Columns are zero-mean maps. Shape is ``(n_voxels, n_structures)``."""
    columns = [resampled(map_path(structures, pdb_id), grid) for pdb_id in ids]
    maps = np.column_stack(columns)
    maps -= maps.mean(axis=0, dtype=np.float64).astype(np.float32)
    return maps


def singular_vectors(maps: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Economy SVD. Returns singular values, spatial vectors ``(n, k)``, and structure vectors ``(m, k)``."""
    gram = np.asarray(maps.T @ maps, dtype=np.float64)
    eigenvalues, vectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    vectors = vectors[:, order]
    singular = np.sqrt(eigenvalues)
    spatial = np.empty((maps.shape[0], singular.size), dtype=np.float32)
    for index in range(singular.size):
        if singular[index] <= 0.0:
            spatial[:, index] = 0.0
            continue
        mode = maps @ vectors[:, index].astype(np.float32)
        mode /= np.float32(singular[index])
        if vectors[int(np.argmax(np.abs(vectors[:, index]))), index] < 0.0:
            vectors[:, index] *= -1.0
            mode *= -1.0
        spatial[:, index] = mode
    return singular, spatial, vectors


def write_component(values: np.ndarray, grid: tuple[int, int, int], cell: tuple[float, ...], path: Path, label: str) -> None:
    """Sigma-scale one component and write CCP4 plus a difference-map MTZ."""
    data = _sigma_scaled(values, grid)
    symmetry = crystal.symmetry(unit_cell=cell, space_group_symbol="P 21 21 21")
    iotbx.mrcfile.write_ccp4_map(
        file_name=str(path),
        crystal_symmetry=symmetry,
        map_data=data,
        labels=flex.std_string([label]),
        unit_cell_grid=grid,
    )
    _write_difference_mtz(data, grid, symmetry, path.with_suffix(".mtz"), label)


def _sigma_scaled(values: np.ndarray, grid: tuple[int, int, int]):
    flat = np.asarray(values, dtype=np.float64)
    scale = float(np.std(flat))
    if scale > 0.0:
        flat = flat / scale
    data = flex.double(np.ascontiguousarray(flat))
    data.reshape(flex.grid(*grid))
    return data


def _write_difference_mtz(data, grid: tuple[int, int, int], symmetry, path: Path, label: str) -> None:
    """Fourier coefficients labeled ``FOFCWT`` / ``PHFOFCWT``."""
    from iotbx.mtz import label_decorator

    cell = symmetry.unit_cell().parameters()
    nyquist = max(cell[index] * 2.0 / grid[index] for index in range(3))
    coefficients = miller.build_set(
        crystal_symmetry=symmetry,
        anomalous_flag=False,
        d_min=nyquist,
    ).structure_factors_from_map(map=data, use_scale=True, anomalous_flag=False, use_sg=True)
    dataset = coefficients.as_mtz_dataset(
        column_root_label="FOFCWT",
        label_decorator=label_decorator(phases_prefix="PH"),
        title=label[:70],
    )
    dataset.mtz_object().write(file_name=str(path))


def write_loadings(path: Path, ids: list[str], singular: np.ndarray, structures: np.ndarray) -> None:
    """Singular values and the structure weights of each component."""
    total = float(np.sum(singular**2))
    header = ["component", "singular_value", "fraction"] + ids
    lines = [" ".join(header)]
    running = 0.0
    for index, value in enumerate(singular):
        fraction = float(value**2 / total) if total > 0.0 else 0.0
        running += fraction
        weights = " ".join(f"{weight:.6f}" for weight in structures[:, index])
        lines.append(f"{index + 1:02d} {value:.6e} {fraction:.6f} {weights}")
        print(f"component {index + 1:02d}: fraction {fraction:.3f} cumulative {running:.3f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SVD of the median-set gradient maps.")
    parser.add_argument("--ids", type=Path, default=_ROOT / "median_ids.txt")
    parser.add_argument("--structures", type=Path, default=_STRUCTURES)
    parser.add_argument("--out", type=Path, default=_OUT)
    args = parser.parse_args(argv)
    ids = read_ids(args.ids)
    cell, grid = common_frame(args.structures, ids)
    print(f"common cell {cell[0]:.3f} {cell[1]:.3f} {cell[2]:.3f} grid {grid}")
    maps = stack_maps(args.structures, ids, grid)
    singular, spatial, structures = singular_vectors(maps)
    args.out.mkdir(parents=True, exist_ok=True)
    total = float(np.sum(singular**2))
    for index, value in enumerate(singular):
        fraction = float(value**2 / total) if total else 0.0
        destination = args.out / f"component_{index + 1:02d}.ccp4"
        write_component(
            spatial[:, index],
            grid,
            cell,
            destination,
            f"gradient SVD component {index + 1} fraction {fraction:.4f}",
        )
    write_loadings(args.out / "loadings.txt", ids, singular, structures)
    print(f"wrote {singular.size} components in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
