#!/usr/bin/env python3
"""FWT maps for every aligned TDP1 dimer, then their SVD.

Observed amplitudes are anisotropically scaled onto 8V0C, the median model.
Each map is sigma-A weighted 2mFo−DFc, warped into the SVD map frame.
Waters are kept in ``fwt_with_water`` and removed in ``fwt_no_water``.
CCP4 maps are written beside MTZ files whose columns are ``FWT`` and ``PHWT``.

Usage::

    python examples/case_study/redo_svd.py
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import gradient_svd
import iotbx.pdb
import maps
import numpy as np
import svd_frame
from cctbx import maptbx
from cctbx.array_family import flex
from mmtbx.scaling.absolute_scaling import anisotropic_correction
from mmtbx.scaling.relative_scaling import ls_rel_scale_driver

_ROOT = Path(__file__).resolve().parent
_STRUCTURES = _ROOT / "structures"
_REFERENCE = "8v0c"
_GRID = (120, 250, 480)
_SETS = (
    ("fwt_with_water", True),
    ("fwt_no_water", False),
)
_MAP_ORDER: dict[str, str] = {}


def scale_to_reference(reference, mobile):
    """Anisotropic least-squares scale of ``mobile`` onto ``reference`` amplitudes."""
    driver = ls_rel_scale_driver(
        reference,
        mobile.map_to_asu(),
        use_intensities=False,
        scale_weight=False,
        use_weights=True,
    )
    scaled = anisotropic_correction(mobile.map_to_asu(), driver.p_scale, driver.u_star)
    scaled.set_observation_type_xray_amplitude()
    return scaled, driver


def fwt_coefficients(model):
    """Sigma-A weighted 2mFo−DFc from amplitude data and model phases."""
    from phridge.tools.intensity.math_utils import _fom_rice_woolfson

    f_model = np.asarray(model.f_model.data(), dtype=np.complex128)
    observed = np.asarray(model.get_f_obs().data(), dtype=np.float64)
    epsilon = np.asarray(model.i_obs.epsilons().data().as_double(), dtype=np.float64)
    wilson = np.asarray(model.mean_i_per_refl, dtype=np.float64)
    sigma_a = np.asarray(model.sigma_a_per_refl, dtype=np.float64)
    centric = np.asarray(model.i_obs.centric_flags().data(), dtype=bool)
    denominator = np.maximum(epsilon * wilson, 1e-12)
    normalized_calc = np.abs(f_model) / np.sqrt(denominator)
    normalized_obs = observed**2 / denominator
    figure_of_merit = _fom_rice_woolfson(normalized_calc, sigma_a, normalized_obs, centric)
    phase = np.exp(1j * np.angle(f_model))
    coefficients = (2.0 * figure_of_merit * observed - sigma_a * np.abs(f_model)) * phase
    return model.i_obs.customized_copy(
        data=flex.complex_double(np.ascontiguousarray(coefficients, dtype=np.complex128)),
        sigmas=None,
    )


def density_map(structure, amplitudes, flags):
    """Sigma-scaled FWT map in the native cell."""
    transform = model_coefficients(structure, amplitudes, flags).fft_map(resolution_factor=0.25)
    transform.apply_sigma_scaling()
    return transform.real_map_unpadded(), structure.crystal_symmetry().unit_cell()


def model_coefficients(structure, amplitudes, flags):
    """FWT coefficients in the native cell, with bulk solvent, scale, and sigma-A fitted to ``amplitudes``."""
    from phridge.tools.intensity.model import IntensityModel

    model = IntensityModel(
        xray_structure=structure,
        i_obs=amplitudes,
        r_free_flags=flags,
        use_bulk_solvent=True,
        target_type="amplitude",
        device="cuda",
    )
    maps.wilson_from_amplitudes(model)
    model.refine_scale_and_solvent()
    maps.fit_sigma_a(model)
    model.compute_target_and_gradients()
    return fwt_coefficients(model)


def prepared_structure(path: Path, keep_water: bool):
    """Deposited dimer in its own cell. Waters are dropped when ``keep_water`` is false."""
    loaded = iotbx.pdb.input(file_name=str(path))
    hierarchy = loaded.construct_hierarchy()
    if not keep_water:
        water = hierarchy.atom_selection_cache().selection("resname HOH or resname WAT or resname DOD")
        hierarchy = hierarchy.select(~water)
    symmetry = loaded.crystal_symmetry()
    return hierarchy.extract_xray_structure(crystal_symmetry=symmetry)


def warp_map(real_map, native_cell, rotation: np.ndarray, translation: np.ndarray, grid: tuple[int, int, int]) -> np.ndarray:
    """Sample the native map on the SVD grid. ``rotation`` maps native Cartesian onto the frame."""
    values = _volume(real_map)
    fractional = _output_fractional(grid)
    cartesian = fractional @ _matrix(svd_frame.map_cell()).T
    native = (rotation.T @ (cartesian - translation).T).T
    native_fractional = native @ np.linalg.inv(_matrix(native_cell)).T
    return _trilinear(values, native_fractional)


def _volume(real_map) -> np.ndarray:
    """NumPy view of a flex map, with the axis order that matches fractional interpolation."""
    shape = tuple(int(value) for value in real_map.all())
    flat = np.array(real_map, dtype=np.float32)
    order = _MAP_ORDER.get("order")
    if order is None:
        order = _axis_order(real_map, flat, shape)
        _MAP_ORDER["order"] = order
    return flat.reshape(shape, order=order)


def _axis_order(real_map, flat: np.ndarray, shape: tuple[int, int, int]) -> str:
    probes = np.array([[0.2, 0.3, 0.4], [0.5, 0.5, 0.5], [0.8, 0.1, 0.6], [0.15, 0.7, 0.25]], dtype=np.float64)
    observed = [maptbx.eight_point_interpolation(real_map, tuple(point)) for point in probes]
    best = "C"
    best_error = None
    for order in ("C", "F"):
        predicted = _trilinear(flat.reshape(shape, order=order), probes.astype(np.float32))
        error = float(np.max(np.abs(predicted - np.array(observed, dtype=np.float32))))
        if best_error is None or error < best_error:
            best, best_error = order, error
    print(f"map axis order {best}, interpolation error {best_error:.4f}")
    return best


def _output_fractional(grid: tuple[int, int, int]) -> np.ndarray:
    axes = [np.arange(length, dtype=np.float32) / length for length in grid]
    x, y, z = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    return np.stack([x, y, z], axis=-1).reshape(-1, 3)


def _matrix(unit_cell) -> np.ndarray:
    if not isinstance(unit_cell, tuple):
        parameters = unit_cell.parameters()
    else:
        parameters = unit_cell
    from cctbx import uctbx

    cell = uctbx.unit_cell(tuple(parameters))
    return np.array(cell.orthogonalization_matrix(), dtype=np.float64).reshape(3, 3)


def _trilinear(grid: np.ndarray, fractional: np.ndarray) -> np.ndarray:
    nx, ny, nz = grid.shape
    x = np.mod(fractional[:, 0], 1.0) * nx
    y = np.mod(fractional[:, 1], 1.0) * ny
    z = np.mod(fractional[:, 2], 1.0) * nz
    x0 = np.floor(x).astype(np.int32) % nx
    y0 = np.floor(y).astype(np.int32) % ny
    z0 = np.floor(z).astype(np.int32) % nz
    x1 = (x0 + 1) % nx
    y1 = (y0 + 1) % ny
    z1 = (z0 + 1) % nz
    xd = (x - np.floor(x)).astype(np.float32)
    yd = (y - np.floor(y)).astype(np.float32)
    zd = (z - np.floor(z)).astype(np.float32)
    c00 = grid[x0, y0, z0] * (1.0 - xd) + grid[x1, y0, z0] * xd
    c01 = grid[x0, y0, z1] * (1.0 - xd) + grid[x1, y0, z1] * xd
    c10 = grid[x0, y1, z0] * (1.0 - xd) + grid[x1, y1, z0] * xd
    c11 = grid[x0, y1, z1] * (1.0 - xd) + grid[x1, y1, z1] * xd
    c0 = c00 * (1.0 - yd) + c10 * yd
    c1 = c01 * (1.0 - yd) + c11 * yd
    return c0 * (1.0 - zd) + c1 * zd


def _flags_for(scaled, flags):
    if flags is None:
        return None
    pairs = scaled.match_indices(flags.map_to_asu()).pairs()
    if pairs.size() == 0:
        return None
    values = flags.map_to_asu().select(pairs.column(1)).data()
    full = flex.bool(scaled.size(), False)
    full.set_selected(pairs.column(0), values)
    return scaled.customized_copy(data=full, sigmas=None)


def one_structure(folder: Path, reference, pose: dict[int, np.ndarray], keep_water: bool) -> np.ndarray:
    """Scaled residual of one dimer on the common grid, zero mean and unit variance."""
    (amplitudes, flags), kind = maps.load_amplitudes(folder / f"{folder.name}-sf.cif")
    scaled, driver = scale_to_reference(reference, amplitudes)
    structure = prepared_structure(svd_frame.model_path(folder), keep_water)
    real_map, native_cell = density_map(structure, scaled, _flags_for(scaled, flags))
    loaded = iotbx.pdb.input(file_name=str(svd_frame.model_path(folder)))
    anchor = svd_frame.anchor_chain(loaded.construct_hierarchy())
    rotation, translation, residual = svd_frame.place(
        svd_frame.trace(anchor, loaded.crystal_symmetry().unit_cell()), pose
    )
    warped = warp_map(real_map, native_cell, rotation, translation, _GRID)
    warped -= np.float32(warped.mean())
    deviation = float(warped.std())
    if deviation > 0.0:
        warped /= np.float32(deviation)
    waters = "with water" if keep_water else "no water"
    print(
        f"{folder.name} {waters}: {kind}; scale R {driver.r_val_before:.3f} -> {driver.r_val_after:.3f}; "
        f"fit {residual:.3f} A"
    )
    return warped.astype(np.float32, copy=False)


def _pose() -> dict[int, np.ndarray]:
    from cctbx import crystal as crystal_module

    path = svd_frame.model_path(_STRUCTURES / "6dhu")
    loaded = iotbx.pdb.input(file_name=str(path))
    chain = next(item for item in loaded.construct_hierarchy().chains() if item.id.strip() == "A")
    output = crystal_module.symmetry(unit_cell=svd_frame.map_cell(), space_group_symbol="P 21 21 21").unit_cell()
    return svd_frame.map_trace(chain, loaded.crystal_symmetry().unit_cell(), output)


def _folders(only: str) -> list[Path]:
    folders = []
    for folder in sorted(path for path in _STRUCTURES.iterdir() if path.is_dir()):
        if folder.name == "8uvz" or svd_frame.model_path(folder) is None:
            continue
        if not (folder / f"{folder.name}-sf.cif").is_file():
            continue
        if only and folder.name != only:
            continue
        folders.append(folder)
    return folders


def write_fwt(values: np.ndarray, grid: tuple[int, int, int], cell: tuple[float, ...], path: Path, label: str) -> None:
    """Sigma-scaled CCP4 map plus an MTZ with columns ``FWT`` and ``PHWT``."""
    from cctbx import crystal, miller
    import iotbx.mrcfile
    from iotbx.mtz import label_decorator

    data = gradient_svd._sigma_scaled(values, grid)
    symmetry = crystal.symmetry(unit_cell=cell, space_group_symbol="P 21 21 21")
    iotbx.mrcfile.write_ccp4_map(
        file_name=str(path),
        crystal_symmetry=symmetry,
        map_data=data,
        labels=flex.std_string([label]),
        unit_cell_grid=grid,
    )
    nyquist = max(cell[index] * 2.0 / grid[index] for index in range(3))
    coefficients = miller.build_set(
        crystal_symmetry=symmetry,
        anomalous_flag=False,
        d_min=nyquist,
    ).structure_factors_from_map(map=data, use_scale=True, anomalous_flag=False, use_sg=True)

    class _FwtLabels(label_decorator):
        def phases(self, root_label, anomalous_sign=None):
            return "PHWT"

    dataset = coefficients.as_mtz_dataset(
        column_root_label="FWT",
        label_decorator=_FwtLabels(phases_prefix="PH"),
        title=label[:70],
    )
    dataset.mtz_object().write(file_name=str(path.with_suffix(".mtz")))


def _write_set(directory: Path, ids: list[str], columns: np.ndarray) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for stale in directory.glob("component_*.ccp4"):
        stale.unlink()
    for stale in directory.glob("component_*.mtz"):
        stale.unlink()
    singular, spatial, structures = gradient_svd.singular_vectors(columns)
    cell = svd_frame.map_cell()
    total = float(np.sum(singular**2))
    for index, value in enumerate(singular):
        fraction = float(value**2 / total) if total else 0.0
        write_fwt(
            spatial[:, index],
            _GRID,
            cell,
            directory / f"component_{index + 1:02d}.ccp4",
            f"FWT SVD component {index + 1} fraction {fraction:.4f}",
        )
    gradient_svd.write_loadings(directory / "loadings.txt", ids, singular, structures)


def _collect(
    folders: list[Path],
    reference,
    pose: dict[int, np.ndarray],
    keep_water: bool,
    directory: Path,
) -> tuple[list[str], np.ndarray]:
    residuals = directory / "residuals"
    residuals.mkdir(parents=True, exist_ok=True)
    cell = svd_frame.map_cell()
    columns = []
    ids = []
    label = "with water" if keep_water else "no water"
    for folder in folders:
        try:
            warped = one_structure(folder, reference, pose, keep_water)
        except Exception as exc:
            print(f"failed {folder.name}: {exc}")
            traceback.print_exc()
            continue
        write_fwt(
            warped,
            _GRID,
            cell,
            residuals / f"{folder.name}.ccp4",
            f"{folder.name} FWT {label}",
        )
        columns.append(warped)
        ids.append(folder.name)
    if not columns:
        raise RuntimeError("no residual maps")
    return ids, np.column_stack(columns)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scale to 8V0C, residual maps with and without water, then SVD.")
    parser.add_argument("--only", default="")
    args = parser.parse_args(argv)
    reference = maps.load_amplitudes(_STRUCTURES / _REFERENCE / f"{_REFERENCE}-sf.cif")[0][0]
    pose = _pose()
    folders = _folders(args.only.lower())
    for name, keep_water in _SETS:
        if args.only and name != "fwt_no_water":
            continue
        print(f"--- {name} ---")
        directory = _ROOT / name
        ids, columns = _collect(folders, reference, pose, keep_water, directory)
        if args.only:
            print(f"{name}: computed {len(ids)} map(s)")
            continue
        _write_set(directory, ids, columns)
        print(f"wrote {len(ids)} components to {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
