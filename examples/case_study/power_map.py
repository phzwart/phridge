#!/usr/bin/env python3
"""Power maps from each cluster's centroid-model residual SVD.

The power at a voxel is ``P(x) = sum_k s_k^2 u_k(x)^2``, where ``u_k`` is the
unit-norm spatial vector and ``s_k`` its singular value. Over all components
this is the summed squared residual across the cluster's structures. Maps are
divided by their mean, so 1 is the average power in the cell. Residues are
ranked by the mean power at their atoms in the centroid model.

Usage::

    python examples/case_study/power_map.py
    python examples/case_study/power_map.py --components 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import iotbx.ccp4_map
import iotbx.mrcfile
import numpy as np
from cctbx import crystal, uctbx
from scitbx.array_family import flex

_CLUSTERS = Path(__file__).resolve().parent / "clusters" / "with_water_1-4"
_WATER = {"HOH", "WAT", "DOD"}


def singular_values(path: Path) -> np.ndarray:
    lines = path.read_text(encoding="utf-8").splitlines()[1:]
    return np.array([float(line.split()[1]) for line in lines], dtype=np.float64)


def read_volume(path: Path) -> tuple[np.ndarray, tuple[float, ...], tuple[int, int, int]]:
    reader = iotbx.ccp4_map.map_reader(file_name=str(path))
    shape = tuple(int(value) for value in reader.data.focus())
    values = np.array(reader.data, dtype=np.float64).reshape(shape, order="C")
    return values, tuple(float(value) for value in reader.unit_cell_parameters), shape


def power(directory: Path, count: int) -> tuple[np.ndarray, tuple[float, ...], tuple[int, int, int], int]:
    """Weighted sum of squared unit-norm components. ``count`` of 0 means every written component."""
    singular = singular_values(directory / "loadings.txt")
    paths = sorted(directory.glob("component_*.ccp4"))
    if count:
        paths = paths[:count]
    total = None
    cell, shape = None, None
    for path in paths:
        index = int(path.stem.split("_")[1]) - 1
        values, cell, shape = read_volume(path)
        values /= np.linalg.norm(values)
        term = singular[index] ** 2 * values**2
        total = term if total is None else total + term
    return total / total.mean(), cell, shape, len(paths)


def write_power(values: np.ndarray, cell: tuple[float, ...], shape: tuple[int, int, int], path: Path, label: str) -> None:
    data = flex.double(np.ascontiguousarray(values.ravel()))
    data.reshape(flex.grid(*shape))
    iotbx.mrcfile.write_ccp4_map(
        file_name=str(path),
        crystal_symmetry=crystal.symmetry(unit_cell=cell, space_group_symbol="P 21 21 21"),
        map_data=data,
        labels=flex.std_string([label]),
        unit_cell_grid=shape,
    )


def residue_power(values: np.ndarray, cell: tuple[float, ...], model: Path) -> list[tuple[float, str]]:
    """Mean power at each residue's atoms, highest first."""
    fractional = np.linalg.inv(np.array(uctbx.unit_cell(cell).orthogonalization_matrix()).reshape(3, 3))
    shape = np.array(values.shape)
    sums: dict[str, list[float]] = {}
    for line in model.read_text(encoding="utf-8").splitlines():
        if not line.startswith(("ATOM", "HETATM")) or line[17:20].strip() in _WATER:
            continue
        xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        grid = np.rint(np.mod(xyz @ fractional.T, 1.0) * shape).astype(int) % shape
        key = f"{line[17:20].strip()} {line[21]}{line[22:26].strip()}"
        sums.setdefault(key, []).append(float(values[grid[0], grid[1], grid[2]]))
    return sorted(((float(np.mean(items)), key) for key, items in sums.items()), reverse=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Singular-value-weighted power maps per cluster.")
    parser.add_argument("--components", type=int, default=0, help="first K components; 0 uses all")
    args = parser.parse_args(argv)
    suffix = f"1-{args.components}" if args.components else "all"
    for cluster in sorted(path for path in _CLUSTERS.iterdir() if (path / "members.txt").is_file()):
        directory = cluster / "centroid_model" / "residual_svd"
        values, cell, shape, used = power(directory, args.components)
        destination = directory / f"power_{suffix}.ccp4"
        write_power(values, cell, shape, destination, f"{cluster.name} power, {used} components, mean 1")
        centroid = cluster.name.split("_", 1)[1]
        ranked = residue_power(values, cell, cluster / "structures" / f"{centroid}.pdb")
        lines = ["power residue"] + [f"{value:.3f} {key}" for value, key in ranked]
        (directory / f"power_{suffix}_residues.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"{cluster.name}: {used} components, max {values.max():.1f}; top " + ", ".join(
            f"{key} {value:.1f}" for value, key in ranked[:8]
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
