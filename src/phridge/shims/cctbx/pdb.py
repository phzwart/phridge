"""PDB and symmetry helpers. The only ensemble door onto cctbx."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from phridge.shims.cctbx._flex import complex_double, miller_index, vec3_double


def origin_preserves_group(symbol: str, shift: tuple[float, float, float]) -> bool:
    from cctbx import sgtbx

    group = sgtbx.space_group_info(symbol=symbol).group()
    origin = sgtbx.tr_vec([int(round(12 * x)) for x in shift], tr_den=12)
    try:
        changed = group.change_basis(sgtbx.change_of_basis_op(sgtbx.rt_mx(sgtbx.rot_mx(), origin)))
    except Exception:
        return False
    return changed == group


def ideal_helix_tables(n_res: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import iotbx.pdb
    from mmtbx.secondary_structure.build import ss_idealization as ssb

    built = ssb.secondary_structure_from_sequence(ssb.alpha_helix_str, "A" * int(n_res))
    raw = iotbx.pdb.input(source_info=None, lines=built.as_pdb_string().splitlines())
    hierarchy = raw.construct_hierarchy()
    atoms = list(hierarchy.atoms())
    xyz = np.array(list(hierarchy.atoms().extract_xyz()), dtype=np.float64)
    name = np.array([atom.name for atom in atoms])
    resseq = np.array([atom.parent().parent().resseq_as_int() for atom in atoms], dtype=np.int32)
    element = np.array([atom.element.strip() or atom.name.strip()[:1] for atom in atoms])
    return xyz, name, resseq, element


def direct_f_calc_pdb(path: str, hkl: np.ndarray, cell: Sequence[float], space_group: str | None) -> np.ndarray:
    import iotbx.pdb
    from cctbx import crystal, miller

    pdb = iotbx.pdb.input(file_name=str(path))
    xs = pdb.xray_structure_simple()
    symbol = space_group or "P 1"
    cs = crystal.symmetry(unit_cell=tuple(float(x) for x in cell), space_group_symbol=symbol)
    indices = miller_index([tuple(int(v) for v in row) for row in np.asarray(hkl)])
    miller_set = miller.set(crystal_symmetry=cs, indices=indices, anomalous_flag=False)
    xs = xs.customized_copy(crystal_symmetry=cs)
    fc = xs.structure_factors(miller_set=miller_set, algorithm="direct").f_calc()
    return np.asarray(list(fc.data()), dtype=np.complex128)


def sigma_scaled_ccp4(pdb_path: str, map_path: str, cell: Sequence[float], d_min: float) -> None:
    import iotbx.pdb
    from cctbx import crystal, miller

    inp = iotbx.pdb.input(file_name=str(pdb_path))
    cs = crystal.symmetry(unit_cell=tuple(float(x) for x in cell), space_group_symbol="P 1")
    xrs = inp.xray_structure_simple(crystal_symmetry=cs)
    xrs.set_b_iso(value=15.0)
    miller_set = miller.build_set(crystal_symmetry=cs, anomalous_flag=False, d_min=float(d_min))
    fc = miller_set.structure_factors_from_scatterers(xray_structure=xrs, algorithm="direct").f_calc()
    fft = fc.fft_map(resolution_factor=0.33)
    fft.apply_sigma_scaling()
    fft.as_ccp4_map(file_name=str(map_path), labels=["poly-Ala alpha helix Fc; sigma = 1"])


def pdb_model_coordinates(path: str) -> list[np.ndarray]:
    import iotbx.pdb

    hierarchy = iotbx.pdb.input(file_name=str(path)).construct_hierarchy()
    return [np.array(list(model.atoms().extract_xyz()), dtype=np.float64) for model in hierarchy.models()]


def direct_f_calc_frames(
    reference_pdb: str,
    frames: np.ndarray,
    cell: Sequence[float],
    d_min: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import iotbx.pdb
    from cctbx import crystal, miller

    cs = crystal.symmetry(unit_cell=tuple(float(x) for x in cell), space_group_symbol="P 1")
    inp = iotbx.pdb.input(file_name=str(reference_pdb))
    xrs = inp.xray_structure_simple(crystal_symmetry=cs)
    xrs.scattering_type_registry(table="wk1995")
    miller_set = miller.build_set(crystal_symmetry=cs, anomalous_flag=False, d_min=float(d_min))
    if xrs.scatterers().size() != frames.shape[1]:
        raise RuntimeError("reference PDB atom count does not match the trajectory")
    columns = []
    fc: Any = None
    for xyz in frames:
        moved = xrs.deep_copy_scatterers()
        moved.set_sites_cart(vec3_double(np.asarray(xyz, dtype=np.float64)))
        fc = miller_set.structure_factors_from_scatterers(xray_structure=moved, algorithm="direct").f_calc()
        columns.append(np.array(fc.data(), dtype=np.complex128))
    if fc is None:
        raise RuntimeError("no frames")
    hkl = np.array([tuple(h) for h in fc.indices()], dtype=np.int32)
    centric = np.array(fc.centric_flags().data(), dtype=bool)
    epsilon = np.array(fc.epsilons().data().as_double(), dtype=np.float64)
    return hkl, np.stack(columns), centric, epsilon


def write_mode_ccp4(
    directory: Path,
    hkl: np.ndarray,
    modes: np.ndarray,
    f_mu: np.ndarray,
    cell: Sequence[float],
) -> None:
    from cctbx import crystal, miller

    cs = crystal.symmetry(unit_cell=tuple(float(x) for x in cell), space_group_symbol="P 1")
    indices = miller_index([tuple(int(x) for x in row) for row in hkl])
    miller_set = miller.set(crystal_symmetry=cs, indices=indices, anomalous_flag=False)
    base = miller.array(miller_set=miller_set, data=complex_double(np.asarray(f_mu, dtype=np.complex128)))
    fft = base.fft_map(resolution_factor=0.5)
    fft.apply_sigma_scaling()
    fft.as_ccp4_map(file_name=str(directory / "mean_fc.ccp4"), labels=["mean model Fc; sigma = 1"])
    grids = []
    for coeff in modes:
        arr = base.customized_copy(data=complex_double(np.asarray(coeff, dtype=np.complex128)))
        mode_fft = arr.fft_map(resolution_factor=0.5)
        mode_fft.apply_volume_scaling()
        rm = mode_fft.real_map_unpadded()
        grids.append(np.array(rm, dtype=np.float64).reshape(tuple(int(v) for v in rm.focus())))
    scale = float(np.sqrt(np.mean(np.stack(grids) ** 2)))
    for index, coeff in enumerate(modes, start=1):
        arr = base.customized_copy(data=complex_double(np.asarray(coeff / scale, dtype=np.complex128)))
        mode_fft = arr.fft_map(resolution_factor=0.5)
        mode_fft.apply_volume_scaling()
        mode_fft.as_ccp4_map(
            file_name=str(directory / f"em_mode_{index:02d}.ccp4"),
            labels=[f"EM mode {index}; pooled mode RMS = 1"],
        )
