"""Maps of every crystal on the median P2₁2₁2₁ cell, on one Fourier grid.

All 44 crystals are P2₁2₁2₁ with cell edges within 1% of the median. Each model
is placed on 8V0C by the space-group operator and allowed origin shift that
best superpose its Cα atoms, with chains A and B swapped when that fits better.
Its 2mFo−DFc coefficients are then indexed on the median cell, so every map is
an FFT on the same grid and no map is interpolated.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import box_frame
import iotbx.mrcfile
import iotbx.pdb
import maps
import numpy as np
import svd_frame
from cctbx import crystal, maptbx, miller
from scipy.spatial import cKDTree
from scitbx.array_family import flex

_ROOT = Path(__file__).resolve().parent
_STRUCTURES = _ROOT / "structures"
POSE = "8v0c"
_MASK_RADIUS = 5.0
_RESOLUTION_FACTOR = 0.25
_SWAP = {"A": "B", "B": "A"}


@dataclass(frozen=True)
class Placement:
    rotation: np.ndarray
    translation: np.ndarray
    chain_shift: dict
    swap: bool
    rmsd: float


def _loaded(pdb_id: str):
    loaded = iotbx.pdb.input(file_name=str(svd_frame.model_path(_STRUCTURES / pdb_id)))
    return loaded.construct_hierarchy(), loaded.crystal_symmetry()


@lru_cache(maxsize=1)
def _datasets() -> tuple[tuple[float, float, float], float]:
    import box_pipeline

    edges, d_min = [], np.inf
    for pdb_id in box_pipeline.structure_ids():
        amplitudes = maps.load_amplitudes(_STRUCTURES / pdb_id / f"{pdb_id}-sf.cif")[0][0]
        edges.append(amplitudes.unit_cell().parameters()[:3])
        d_min = min(d_min, amplitudes.d_min())
    return tuple(float(value) for value in np.median(np.array(edges), axis=0)), float(d_min)


@lru_cache(maxsize=1)
def symmetry():
    """Median cell edges of the amplitude data, in P2₁2₁2₁."""
    edges, _d_min = _datasets()
    return crystal.symmetry(unit_cell=edges + (90.0, 90.0, 90.0), space_group_symbol="P 21 21 21")


def d_min() -> float:
    return _datasets()[1]


@lru_cache(maxsize=1)
def gridding():
    return maptbx.crystal_gridding(
        unit_cell=symmetry().unit_cell(),
        space_group_info=symmetry().space_group_info(),
        d_min=d_min(),
        resolution_factor=_RESOLUTION_FACTOR,
    )


def grid() -> tuple[int, int, int]:
    return tuple(int(value) for value in gridding().n_real())


@lru_cache(maxsize=1)
def common_set():
    """Every reflection of the median cell to the best resolution in the set."""
    return miller.build_set(crystal_symmetry=symmetry(), anomalous_flag=False, d_min=d_min()).map_to_asu()


def _edges() -> np.ndarray:
    return np.array(symmetry().unit_cell().parameters()[:3], dtype=np.float64)


def _ca_fractional(pdb_id: str) -> dict[tuple[str, int], np.ndarray]:
    hierarchy, crystal_symmetry = _loaded(pdb_id)
    cell = crystal_symmetry.unit_cell()
    points = {}
    for chain in hierarchy.models()[0].chains():
        for group in chain.residue_groups():
            for atom in group.atoms():
                if atom.name.strip() == "CA" and atom.parent().resname.strip() not in box_frame._WATER:
                    points.setdefault((chain.id.strip(), group.resseq_as_int()), np.array(cell.fractionalize(atom.xyz)))
    return points


@lru_cache(maxsize=None)
def placement(pdb_id: str) -> Placement:
    """Symmetry operator, origin shift, and per-chain lattice translation onto ``POSE``."""
    reference = _ca_fractional(POSE)
    mobile = _ca_fractional(pdb_id)
    space_group = _loaded(pdb_id)[1].space_group()
    shifts = [np.array(shift, dtype=np.float64) / 2.0 for shift in itertools.product((0, 1), repeat=3)]
    best = None
    for operator in space_group.all_ops():
        rotation = np.array(operator.r().as_double(), dtype=np.float64).reshape(3, 3)
        translation = np.array(operator.t().as_double(), dtype=np.float64)
        for swap in (False, True):
            rename = _SWAP if swap else {}
            keys = [key for key in mobile if (rename.get(key[0], key[0]), key[1]) in reference]
            if len(keys) < 100:
                continue
            moved = np.array([rotation @ mobile[key] + translation for key in keys])
            target = np.array([reference[(rename.get(key[0], key[0]), key[1])] for key in keys])
            for shift in shifts:
                delta = target - moved - shift
                delta -= np.round(delta)
                rmsd = float(np.sqrt(np.mean(np.sum((delta * _edges()) ** 2, axis=1))))
                if best is None or rmsd < best[0]:
                    best = (rmsd, rotation, translation + shift, swap, keys, moved + shift, target)
    rmsd, rotation, translation, swap, keys, moved, target = best
    chain_shift = {}
    for chain in {key[0] for key in keys}:
        rows = [index for index, key in enumerate(keys) if key[0] == chain]
        chain_shift[chain] = np.round(np.mean(target[rows] - moved[rows], axis=0))
    return Placement(rotation, translation, chain_shift, swap, rmsd)


@lru_cache(maxsize=1)
def _pose_centre() -> np.ndarray:
    return np.mean(np.array(list(_ca_fractional(POSE).values())), axis=0)


def _hierarchy(pdb_id: str, keep_water: bool):
    hierarchy, crystal_symmetry = _loaded(pdb_id)
    if not keep_water:
        water = hierarchy.atom_selection_cache().selection("resname HOH or resname WAT or resname DOD")
        hierarchy = hierarchy.select(~water)
    return hierarchy, crystal_symmetry


def placed_fractional(pdb_id: str, hierarchy, cell) -> np.ndarray:
    """Fractional coordinates of every atom after the placement, in the crystal's own cell."""
    place = placement(pdb_id)
    native = np.array([cell.fractionalize(xyz) for xyz in hierarchy.atoms().extract_xyz()], dtype=np.float64)
    moved = native @ place.rotation.T + place.translation
    chains = [chain.id.strip() for chain in hierarchy.chains() for _atom in chain.atoms()]
    for index, chain in enumerate(chains):
        shift = place.chain_shift.get(chain)
        if shift is None:
            shift = np.round(_pose_centre() - moved[index])
        moved[index] += shift
    return moved


def placed_structure(pdb_id: str, keep_water: bool):
    """X-ray structure on the placement, in the crystal's own cell, for Fc and phases."""
    hierarchy, crystal_symmetry = _hierarchy(pdb_id, keep_water)
    structure = hierarchy.extract_xray_structure(crystal_symmetry=crystal_symmetry)
    moved = placed_fractional(pdb_id, hierarchy, crystal_symmetry.unit_cell())
    structure.set_sites_frac(flex.vec3_double(np.ascontiguousarray(moved)))
    return structure


def on_common(coefficients) -> np.ndarray:
    """Coefficients indexed on the median cell, zero where the crystal has no reflection."""
    moved = coefficients.customized_copy(crystal_symmetry=symmetry()).map_to_asu()
    pairs = np.array(miller.match_indices(common_set().indices(), moved.indices()).pairs(), dtype=np.int64)
    values = np.zeros(common_set().size(), dtype=np.complex64)
    data = np.asarray(moved.data(), dtype=np.complex128)
    if pairs.size:
        values[pairs[:, 0]] = data[pairs[:, 1]]
    return values


def _array(values: np.ndarray):
    return common_set().array(data=flex.complex_double(np.ascontiguousarray(values, dtype=np.complex128)))


def real(values: np.ndarray) -> np.ndarray:
    """FFT of common-set coefficients on the shared grid, flattened with x slowest."""
    transform = _array(values).fft_map(crystal_gridding=gridding())
    return np.asarray(transform.real_map_unpadded(), dtype=np.float32).reshape(-1)


@lru_cache(maxsize=1)
def mask() -> np.ndarray:
    """Grid points within ``_MASK_RADIUS`` of a non-water ``POSE`` atom, across cell edges."""
    hierarchy, crystal_symmetry = _hierarchy(POSE, keep_water=False)
    edges = _edges()
    sites = placed_fractional(POSE, hierarchy, crystal_symmetry.unit_cell())
    tree = cKDTree(np.mod(sites, 1.0) * edges, boxsize=edges)
    nx, ny, nz = grid()
    y, z = np.meshgrid(np.arange(ny) / ny, np.arange(nz) / nz, indexing="ij")
    plane = np.stack([np.zeros(y.size), y.reshape(-1), z.reshape(-1)], axis=1) * edges
    inside = np.zeros((nx, ny * nz), dtype=bool)
    for index in range(nx):
        plane[:, 0] = index / nx * edges[0]
        distance, _ = tree.query(plane, k=1, distance_upper_bound=_MASK_RADIUS)
        inside[index] = np.isfinite(distance)
    return inside.reshape(-1)


def masked_stats(values: np.ndarray) -> tuple[float, float]:
    inside = values[mask()]
    return float(inside.mean()), float(inside.std())


def expand(masked: np.ndarray) -> np.ndarray:
    full = np.zeros(mask().size, dtype=np.float32)
    full[mask()] = masked
    return full


def write_mtz(values: np.ndarray, path: Path, label: str) -> None:
    """Coefficients as ``FWT`` / ``PHWT`` on the median cell.

    ``real`` has no 1/V factor; viewers apply one, so the written amplitudes are
    multiplied by the cell volume to give the same map values.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = _array(values * symmetry().unit_cell().volume()).as_mtz_dataset(
        column_root_label="FWT",
        label_decorator=box_frame._FwtLabels(phases_prefix="PH"),
        title=label[:70],
    )
    dataset.mtz_object().write(file_name=str(path))


def write_ccp4(values: np.ndarray, path: Path, label: str) -> None:
    """A full-cell map on the shared grid."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = flex.double(np.ascontiguousarray(np.asarray(values, dtype=np.float64)))
    data.reshape(flex.grid(*grid()))
    iotbx.mrcfile.write_ccp4_map(
        file_name=str(path),
        crystal_symmetry=symmetry(),
        map_data=data,
        labels=flex.std_string([label[:80]]),
        unit_cell_grid=grid(),
    )


def write_model(pdb_id: str, directory: Path) -> Path:
    """Deposited model with waters on the placement, in median-cell coordinates."""
    hierarchy, crystal_symmetry = _hierarchy(pdb_id, keep_water=True)
    moved = placed_fractional(pdb_id, hierarchy, crystal_symmetry.unit_cell())
    cell = symmetry().unit_cell()
    hierarchy.atoms().set_xyz(flex.vec3_double([cell.orthogonalize(tuple(site)) for site in moved]))
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{pdb_id}.pdb"
    try:
        hierarchy.write_pdb_file(file_name=str(destination), crystal_symmetry=symmetry())
    except Exception:
        destination = directory / f"{pdb_id}.cif"
        hierarchy.write_mmcif_file(file_name=str(destination), crystal_symmetry=symmetry())
    return destination


def grid_index(xyz: tuple[float, float, float]) -> int:
    """Flat index of the grid point nearest a median-cell Cartesian position."""
    shape = np.array(grid())
    point = np.mod(np.rint(np.array(symmetry().unit_cell().fractionalize(xyz)) * shape).astype(int), shape)
    return int((point[0] * shape[1] + point[1]) * shape[2] + point[2])
