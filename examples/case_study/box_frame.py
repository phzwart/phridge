"""A molecule-centred box in the frame of the 6DHU dimer.

Each crystal is superposed on 6DHU by a least-squares fit of the Cα atoms of
both chains, matched by chain ID and residue number. Its native map is sampled
at the box voxels through that fit. The box is written as a P1 cell whose
origin is the box corner, so maps and models load together in Coot.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import iotbx.mrcfile
import iotbx.pdb
import numpy as np
import redo_svd
import svd_frame
from cctbx import crystal, miller
from iotbx.mtz import label_decorator
from scipy.spatial import cKDTree
from scitbx.array_family import flex

_ROOT = Path(__file__).resolve().parent
_STRUCTURES = _ROOT / "structures"
_POSE = "6dhu"
_SPACING = 0.4
_MARGIN = 6.0
_MASK_RADIUS = 5.0
_MTZ_D_MIN = 1.5
_WATER = {"HOH", "WAT", "DOD"}
_NICE = (2, 3, 5)


class _FwtLabels(label_decorator):
    def phases(self, root_label, anomalous_sign=None):
        return "PHWT"


def _nice(length: int) -> int:
    value = length
    while True:
        rest = value
        for factor in _NICE:
            while rest % factor == 0:
                rest //= factor
        if rest == 1:
            return value
        value += 1


def _hierarchy(pdb_id: str):
    loaded = iotbx.pdb.input(file_name=str(svd_frame.model_path(_STRUCTURES / pdb_id)))
    return loaded.construct_hierarchy(), loaded.crystal_symmetry()


def dimer_trace(hierarchy, unit_cell) -> dict[tuple[str, int], np.ndarray]:
    """Cα by (chain, residue number) for every protein chain."""
    points: dict[tuple[str, int], np.ndarray] = {}
    for chain in hierarchy.chains():
        if svd_frame._protein_count(chain) < 80:
            continue
        for key, xyz in svd_frame.trace(chain, unit_cell).items():
            points.setdefault((chain.id.strip(), key), xyz)
    return points


@lru_cache(maxsize=1)
def geometry() -> tuple[np.ndarray, tuple[int, int, int], tuple[float, ...]]:
    """Box origin in the 6DHU Cartesian frame, grid, and P1 cell."""
    hierarchy, _symmetry = _hierarchy(_POSE)
    xyz = np.array(hierarchy.atoms().extract_xyz(), dtype=np.float64)
    low = xyz.min(axis=0) - _MARGIN
    span = xyz.max(axis=0) + _MARGIN - low
    grid = tuple(_nice(int(np.ceil(length / _SPACING))) for length in span)
    cell = tuple(float(count * _SPACING) for count in grid) + (90.0, 90.0, 90.0)
    return low, grid, cell


@lru_cache(maxsize=1)
def _reference() -> dict[tuple[str, int], np.ndarray]:
    hierarchy, symmetry = _hierarchy(_POSE)
    return dimer_trace(hierarchy, symmetry.unit_cell())


def fit(pdb_id: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Rotation and translation from this crystal's Cartesian frame into the box, and the Cα rmsd."""
    hierarchy, symmetry = _hierarchy(pdb_id)
    mobile = dimer_trace(hierarchy, symmetry.unit_cell())
    reference = _reference()
    common = sorted(set(mobile) & set(reference))
    if len(common) < 160:
        raise ValueError(f"{pdb_id}: only {len(common)} Cα in common with 6DHU")
    source = np.vstack([mobile[key] for key in common])
    target = np.vstack([reference[key] for key in common])
    rotation, translation = svd_frame.kabsch(source, target)
    moved = source @ rotation.T + translation
    rmsd = float(np.sqrt(np.mean(np.sum((moved - target) ** 2, axis=1))))
    low, _grid, _cell = geometry()
    return rotation, translation - low, rmsd


@lru_cache(maxsize=1)
def points() -> np.ndarray:
    """Voxel positions in box Cartesian coordinates, x slowest."""
    _low, grid, _cell = geometry()
    axes = [np.arange(count, dtype=np.float64) * _SPACING for count in grid]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    return np.stack([x, y, z], axis=-1).reshape(-1, 3)


@lru_cache(maxsize=1)
def mask() -> np.ndarray:
    """Voxels within ``_MASK_RADIUS`` of a non-water 6DHU atom."""
    hierarchy, _symmetry = _hierarchy(_POSE)
    low, _grid, _cell = geometry()
    xyz = [
        np.array(atom.xyz) - low
        for atom in hierarchy.atoms()
        if atom.parent().resname.strip() not in _WATER
    ]
    distance, _index = cKDTree(np.array(xyz)).query(points(), k=1, distance_upper_bound=_MASK_RADIUS)
    return np.isfinite(distance)


def sample(real_map, native_cell, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Native map at every box voxel. ``rotation`` and ``translation`` map native Cartesian into the box."""
    native = (points() - translation) @ rotation
    fractional = native @ np.linalg.inv(redo_svd._matrix(native_cell)).T
    return redo_svd._trilinear(redo_svd._volume(real_map), fractional).astype(np.float32)


def normalized(values: np.ndarray) -> np.ndarray:
    """Zero mean and unit variance inside the mask. The whole box is returned."""
    inside = mask()
    result = np.asarray(values, dtype=np.float32).copy()
    result -= np.float32(result[inside].mean())
    deviation = float(result[inside].std())
    if deviation > 0.0:
        result /= np.float32(deviation)
    return result


def expand(masked: np.ndarray) -> np.ndarray:
    """Masked voxel values back on the full box, zero outside."""
    full = np.zeros(mask().size, dtype=np.float32)
    full[mask()] = masked
    return full


def write_map(values: np.ndarray, path: Path, label: str, coefficients: bool = True) -> None:
    """CCP4 of the box in P1. With ``coefficients``, also an MTZ with ``FWT`` / ``PHWT`` to 1.5 Å."""
    _low, grid, cell = geometry()
    data = flex.double(np.ascontiguousarray(np.asarray(values, dtype=np.float64)))
    data.reshape(flex.grid(*grid))
    symmetry = crystal.symmetry(unit_cell=cell, space_group_symbol="P 1")
    path.parent.mkdir(parents=True, exist_ok=True)
    iotbx.mrcfile.write_ccp4_map(
        file_name=str(path),
        crystal_symmetry=symmetry,
        map_data=data,
        labels=flex.std_string([label[:80]]),
        unit_cell_grid=grid,
    )
    if not coefficients:
        return
    structure_factors = miller.build_set(
        crystal_symmetry=symmetry,
        anomalous_flag=False,
        d_min=_MTZ_D_MIN,
    ).structure_factors_from_map(map=data, use_scale=True, anomalous_flag=False, use_sg=False)
    dataset = structure_factors.as_mtz_dataset(
        column_root_label="FWT",
        label_decorator=_FwtLabels(phases_prefix="PH"),
        title=label[:70],
    )
    dataset.mtz_object().write(file_name=str(path.with_suffix(".mtz")))


def moved_hierarchy(pdb_id: str, rotation: np.ndarray, translation: np.ndarray, keep_water: bool = True):
    """This model's atoms under a rigid transform, waters kept unless ``keep_water`` is false."""
    hierarchy, _symmetry = _hierarchy(pdb_id)
    if not keep_water:
        water = hierarchy.atom_selection_cache().selection("resname HOH or resname WAT or resname DOD")
        hierarchy = hierarchy.select(~water)
    xyz = np.array(hierarchy.atoms().extract_xyz(), dtype=np.float64)
    hierarchy.atoms().set_xyz(flex.vec3_double(np.ascontiguousarray(xyz @ rotation.T + translation)))
    return hierarchy


def write_model(pdb_id: str, directory: Path) -> Path:
    """Deposited model, waters kept, in box coordinates. mmCIF when PDB columns overflow."""
    rotation, translation, _rmsd = fit(pdb_id)
    hierarchy = moved_hierarchy(pdb_id, rotation, translation)
    _low, _grid, cell = geometry()
    symmetry = crystal.symmetry(unit_cell=cell, space_group_symbol="P 1")
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{pdb_id}.pdb"
    try:
        hierarchy.write_pdb_file(file_name=str(destination), crystal_symmetry=symmetry)
    except Exception:
        destination = directory / f"{pdb_id}.cif"
        hierarchy.write_mmcif_file(file_name=str(destination), crystal_symmetry=symmetry)
    return destination


def native_structure(pdb_id: str, keep_water: bool):
    """X-ray structure of the deposited model in its own cell."""
    return redo_svd.prepared_structure(svd_frame.model_path(_STRUCTURES / pdb_id), keep_water)


def placed_structure(model_id: str, crystal_id: str, keep_water: bool = True):
    """``model_id`` moved into the crystal frame of ``crystal_id``, with that crystal's symmetry."""
    rotation_model, translation_model, _rmsd = fit(model_id)
    rotation_crystal, translation_crystal, _rmsd = fit(crystal_id)
    rotation = rotation_crystal.T @ rotation_model
    translation = rotation_crystal.T @ (translation_model - translation_crystal)
    hierarchy = moved_hierarchy(model_id, rotation, translation, keep_water)
    _hierarchy_crystal, symmetry = _hierarchy(crystal_id)
    return hierarchy.extract_xray_structure(crystal_symmetry=symmetry)
