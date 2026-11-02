#!/usr/bin/env python3
"""SVD of each cluster after subtracting its average FWT map.

The input maps are the centroid-scaled, centroid-phased maps in ``maps/``.
Each map is normalized, the cluster mean is removed, and the residuals are
factored. The mean map and the residuals are written beside the components.

Usage::

    python examples/case_study/cluster_residual_svd.py
"""

from __future__ import annotations

from pathlib import Path

import cluster_svd
import gradient_svd
import iotbx.ccp4_map
import numpy as np
import redo_svd

_CLUSTERS = Path(__file__).resolve().parent / "clusters" / "with_water_1-4"
_NULL = 1e-5


def read_map(path: Path) -> tuple[np.ndarray, tuple[float, ...]]:
    """Flat C-order map and its unit cell."""
    reader = iotbx.ccp4_map.map_reader(file_name=str(path))
    shape = tuple(int(value) for value in reader.data.focus())
    values = np.array(reader.data, dtype=np.float32).reshape(shape, order="C").ravel()
    cell = tuple(float(value) for value in reader.unit_cell_parameters)
    return cluster_svd._normalize(values), cell


def residuals_of(directory: Path) -> tuple[list[str], np.ndarray, np.ndarray, tuple[float, ...]]:
    """Normalized maps, their mean, and the mean-subtracted columns."""
    ids = cluster_svd.member_ids(directory)
    columns = []
    cell = None
    for pdb_id in ids:
        values, cell = read_map(directory / "maps" / f"{pdb_id}.ccp4")
        columns.append(values)
    stack = np.column_stack(columns)
    mean = stack.mean(axis=1).astype(np.float32)
    return ids, mean, stack - mean[:, None], cell


def _write_modes(directory: Path, ids: list[str], columns: np.ndarray, cell: tuple[float, ...]) -> None:
    """Components of the residual matrix. A null mode from the subtracted mean is omitted."""
    singular, spatial, structures = gradient_svd.singular_vectors(columns)
    total = float(np.sum(singular**2))
    kept = [index for index, value in enumerate(singular) if total and float(value**2 / total) >= _NULL]
    for index in kept:
        fraction = float(singular[index] ** 2 / total)
        redo_svd.write_fwt(
            spatial[:, index],
            redo_svd._GRID,
            cell,
            directory / f"component_{index + 1:02d}.ccp4",
            f"residual SVD component {index + 1} fraction {fraction:.4f}",
        )
    gradient_svd.write_loadings(directory / "loadings.txt", ids, singular, structures)


def _write_columns(directory: Path, ids: list[str], columns: np.ndarray, cell: tuple[float, ...], label: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for pdb_id, values in zip(ids, columns.T):
        redo_svd.write_fwt(values, redo_svd._GRID, cell, directory / f"{pdb_id}.ccp4", f"{pdb_id} {label}")


def one_cluster(directory: Path) -> None:
    """Mean map, residual maps, and their SVD for one cluster."""
    ids, mean, residuals, cell = residuals_of(directory)
    out = directory / "residual_svd"
    out.mkdir(parents=True, exist_ok=True)
    redo_svd.write_fwt(mean, redo_svd._GRID, cell, out / "average.ccp4", f"{directory.name} average FWT")
    _write_columns(out / "residuals", ids, residuals, cell, "FWT minus cluster average")
    _write_modes(out, ids, residuals, cell)
    print(f"wrote residual SVD for {directory.name}")


def main() -> int:
    for directory in sorted(path for path in _CLUSTERS.iterdir() if (path / "members.txt").is_file()):
        print(f"--- {directory.name} ---")
        one_cluster(directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
