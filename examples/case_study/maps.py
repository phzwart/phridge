#!/usr/bin/env python3
"""Strip waters and write amplitude-likelihood gradient maps.

Deposited amplitudes are used as amplitudes. An entry that deposited only
intensities is converted with French–Wilson first. The map is the Fourier
transform of ``-d(ml_f)/dF_model``.

Each entry under ``structures/{id}/`` gets:

- ``{id}_nowater.pdb``
- ``{id}_gradient.ccp4``
- ``{id}_gradient.mtz`` (column ``FGRAD``)

Usage::

    python examples/case_study/maps.py
    python examples/case_study/maps.py --only 1mu7
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import cctbx  # noqa: F401  — import before torch
import iotbx.pdb
import numpy as np
from cctbx.array_family import flex
from iotbx.reflection_file_reader import any_reflection_file

_ROOT = Path(__file__).resolve().parent
_STRUCTURES = _ROOT / "structures"
_WATER = "resname HOH or resname WAT or resname DOD"


def strip_waters(source: Path, dest: Path) -> tuple[Path, int]:
    """Write ``source`` without water residues. Return the written path and atoms removed.

    Ligand names that do not fit the PDB columns are written as mmCIF.
    """
    pdb_in = iotbx.pdb.input(file_name=str(source))
    hierarchy = pdb_in.construct_hierarchy()
    water = hierarchy.atom_selection_cache().selection(_WATER)
    removed = int(water.count(True))
    kept = hierarchy.select(~water)
    written = kept.write_pdb_or_mmcif_file(
        target_filename=str(dest),
        crystal_symmetry=pdb_in.crystal_symmetry(),
    )
    return Path(written), removed


def model_path(folder: Path) -> Path:
    """Deposited model: PDB, or mmCIF when RCSB has no PDB-format file."""
    pdb = folder / f"{folder.name}.pdb"
    if pdb.is_file():
        return pdb
    cif = folder / f"{folder.name}.cif"
    if cif.is_file():
        return cif
    raise FileNotFoundError(f"no model in {folder}")


def load_amplitudes(sf_path: Path):
    """Return an amplitude array, free-set flags on that set, and the source kind."""
    arrays = any_reflection_file(str(sf_path)).as_miller_arrays()
    amplitudes, kind = _amplitudes(arrays)
    return _attach_flags(arrays, amplitudes), kind


def _amplitudes(arrays: list) -> tuple:
    measured = [array for array in arrays if array.is_xray_amplitude_array() and _measured(array)]
    if measured:
        return _merge(measured[0]), "amplitude"
    amplitudes = [array for array in arrays if array.is_xray_amplitude_array()]
    if amplitudes:
        return _merge(amplitudes[0]), "amplitude"
    intensities = [array for array in arrays if array.is_xray_intensity_array()]
    if not intensities:
        raise ValueError("no amplitudes or intensities")
    from libtbx.utils import null_out

    merged = _merge(intensities[0])
    amplitudes = merged.french_wilson(log=null_out())
    amplitudes.set_observation_type_xray_amplitude()
    return amplitudes, "french_wilson"


def _measured(array) -> bool:
    return any("meas" in label or "obs" in label for label in _labels(array))


def _merge(array):
    if array.anomalous_flag():
        array = array.as_non_anomalous_array().merge_equivalents().array()
    return array.map_to_asu()


def _attach_flags(arrays: list, amplitudes):
    flags = _best_free_flags(arrays)
    if flags is None:
        return amplitudes, None
    if flags.anomalous_flag():
        flags = flags.as_non_anomalous_array().merge_equivalents().array()
    flags = flags.map_to_asu()
    pairs = amplitudes.match_indices(flags).pairs()
    amplitudes = amplitudes.select(pairs.column(0))
    flags = flags.select(pairs.column(1))
    return amplitudes, amplitudes.customized_copy(data=flags.data(), sigmas=None)


def _best_free_flags(arrays: list):
    ranked = []
    for array in arrays:
        labels = _labels(array)
        if any("free" in label for label in labels):
            rank = 0
        elif any("status" in label or label.endswith("test") for label in labels):
            rank = 1
        else:
            continue
        flags = _as_free_bool(array)
        if flags is not None:
            ranked.append((rank, flags))
    if not ranked:
        return None
    return sorted(ranked, key=lambda item: item[0])[0][1]


def _as_free_bool(array):
    if array.is_bool_array():
        return array
    if array.is_string_array():
        data = array.data()
        free = (data == "f") | (data == "F")
        if free.count(True) == 0:
            return None
        return array.customized_copy(data=free)
    if array.is_integer_array() or array.is_real_array():
        data = array.data()
        zeros = (data == 0).count(True)
        ones = (data == 1).count(True)
        if zeros == 0 and ones == 0:
            return None
        free = (data == 0) if zeros < ones else (data == 1)
        return array.customized_copy(data=free)
    return None


def _labels(array) -> list[str]:
    info = array.info()
    if info is None:
        return []
    return [label.lower() for label in info.labels]


def wilson_from_amplitudes(model) -> None:
    """Set the per-reflection Wilson intensity from ``<F^2/epsilon>`` in each bin."""
    amplitude = np.asarray(model.i_obs.data(), dtype=np.float64)
    epsilon = np.asarray(model.i_obs.epsilons().data().as_double(), dtype=np.float64)
    intensity = amplitude * amplitude
    mean_i = np.ones(amplitude.shape[0], dtype=np.float64)
    for i_bin in model.binner.range_used():
        selected = np.asarray(model.binner.selection(i_bin), dtype=bool)
        if not np.any(selected):
            continue
        mean = float(np.mean(intensity[selected] / np.maximum(epsilon[selected], 1e-8)))
        mean_i[selected] = max(mean, 1e-4)
    model.mean_i_per_refl = flex.double(mean_i)


def fit_sigma_a(model) -> None:
    """One sigma_A per resolution bin, minimizing the amplitude likelihood."""
    import torch

    from phridge.d.rules.amplitude import MaximumLikelihoodAmplitude, Observations

    observed = model.get_f_obs()
    amplitude = np.asarray(observed.data(), dtype=np.float64)
    sigma = np.asarray(observed.sigmas(), dtype=np.float64)
    epsilon = np.asarray(model.i_obs.epsilons().data().as_double(), dtype=np.float64)
    centric = np.asarray(model.i_obs.centric_flags().data(), dtype=bool)
    free = np.asarray(model.r_free_flags.data(), dtype=bool)
    wilson = np.asarray(model.mean_i_per_refl, dtype=np.float64)
    f_model = np.asarray(model.f_model.data(), dtype=np.complex128)
    bins = list(model.binner.range_used())
    selections = [np.asarray(model.binner.selection(i_bin), dtype=bool) for i_bin in bins]
    device = model.torch_device
    dtype = model.float_dtype
    raw = torch.zeros(len(bins), dtype=dtype, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([raw], lr=1.0, max_iter=30, line_search_fn="strong_wolfe")
    amplitude_t = torch.as_tensor(amplitude, dtype=dtype, device=device)
    sigma_t = torch.as_tensor(sigma, dtype=dtype, device=device)
    epsilon_t = torch.as_tensor(epsilon, dtype=dtype, device=device)
    centric_t = torch.as_tensor(centric, device=device)
    free_t = torch.as_tensor(free, device=device)
    wilson_t = torch.as_tensor(wilson, dtype=dtype, device=device)
    f_model_t = torch.as_tensor(f_model, dtype=model.complex_dtype, device=device)
    target = MaximumLikelihoodAmplitude()

    def closure():
        optimizer.zero_grad()
        sigma_a_bins = 0.05 + 0.94 * torch.sigmoid(raw)
        sigma_a = torch.zeros_like(amplitude_t)
        for index, selected in enumerate(selections):
            sigma_a[selected] = sigma_a_bins[index]
        beta = wilson_t * (1.0 - sigma_a**2)
        observations = Observations(
            data=amplitude_t,
            sigmas=sigma_t,
            epsilon=epsilon_t,
            centric=centric_t,
            alpha=sigma_a,
            beta=beta,
            r_free=free_t,
        )
        loss = target.value(f_model_t, observations)
        loss.backward()
        return loss

    optimizer.step(closure)
    fitted = (0.05 + 0.94 * torch.sigmoid(raw)).detach().cpu().numpy()
    per_reflection = np.empty(amplitude.shape[0], dtype=np.float64)
    model.sigma_a_binned = {}
    for index, i_bin in enumerate(bins):
        model.sigma_a_binned[i_bin] = float(fitted[index])
        per_reflection[selections[index]] = fitted[index]
    model.sigma_a_per_refl = flex.double(per_reflection)


def write_gradient_map(model, prefix: Path) -> None:
    """FFT of ``-d(ml_f)/dF_model``, sigma-scaled, plus the complex coefficients."""
    gradient = -np.asarray(model.d_target_d_f_model, dtype=np.complex128)
    coefficients = model.i_obs.customized_copy(
        data=flex.complex_double(np.ascontiguousarray(gradient)),
        sigmas=None,
    )
    real_map = coefficients.fft_map(resolution_factor=0.25)
    real_map.apply_sigma_scaling()
    real_map.as_ccp4_map(file_name=str(prefix.with_name(prefix.name + "_gradient.ccp4")), labels=["ml_f gradient"])
    dataset = coefficients.as_mtz_dataset(column_root_label="FGRAD")
    dataset.mtz_object().write(str(prefix.with_name(prefix.name + "_gradient.mtz")))


def _ready(folder: Path, pdb_id: str) -> bool:
    return (folder / f"{pdb_id}_gradient.ccp4").is_file()


def run_entry(folder: Path, *, device: str, force: bool, data_path: Path | None = None) -> None:
    """Strip waters and write the amplitude gradient map for one entry."""
    from phridge.tools.intensity.model import IntensityModel

    pdb_id = folder.name
    if _ready(folder, pdb_id) and not force:
        print(f"{pdb_id}: gradient map already written")
        return
    stripped, removed = strip_waters(model_path(folder), folder / f"{pdb_id}_nowater.pdb")
    source = data_path if data_path is not None else folder / f"{pdb_id}-sf.cif"
    (amplitudes, flags), kind = load_amplitudes(source)
    pdb_in = iotbx.pdb.input(file_name=str(stripped))
    symmetry = pdb_in.crystal_symmetry() or amplitudes.crystal_symmetry()
    structure = pdb_in.xray_structure_simple(crystal_symmetry=symmetry)
    model = IntensityModel(
        xray_structure=structure,
        i_obs=amplitudes,
        hierarchy=pdb_in.construct_hierarchy(),
        r_free_flags=flags,
        use_bulk_solvent=True,
        target_type="amplitude",
        device=device,
    )
    wilson_from_amplitudes(model)
    scales = model.refine_scale_and_solvent()
    fit_sigma_a(model)
    model.compute_target_and_gradients()
    write_gradient_map(model, folder / pdb_id)
    mean_sigma_a = float(np.mean(list(model.sigma_a_binned.values())))
    print(
        f"{pdb_id}: removed {removed} waters; {kind}; "
        f"k={scales['k_total']:.3e}; mean sigma_A={mean_sigma_a:.3f}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Amplitude-likelihood gradient maps for the case study.")
    parser.add_argument("--structures", type=Path, default=_STRUCTURES)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--only", default="", help="one PDB id")
    parser.add_argument("--ids", type=Path, help="PDB id list; only these entries are used")
    parser.add_argument("--scaled", action="store_true", help="use amplitudes scaled onto 8UVZ")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    folders = _folders(args.structures, args.only, args.ids)
    failures: list[str] = []
    for folder in folders:
        try:
            data_path = None
            if args.scaled:
                data_path = folder / f"{folder.name}_scaled_to_8uvz.mtz"
                if not data_path.is_file():
                    raise FileNotFoundError(data_path)
            run_entry(folder, device=args.device, force=args.force, data_path=data_path)
        except Exception as exc:
            failures.append(f"{folder.name}: {exc}")
            traceback.print_exc()
    print(f"done: {len(folders) - len(failures)} ok, {len(failures)} failed")
    for item in failures:
        print(f"failed: {item}")
    return 1 if failures else 0


def _folders(root: Path, only: str, ids_path: Path | None) -> list[Path]:
    wanted_one = only.lower().strip()
    folders = sorted(path for path in root.iterdir() if path.is_dir() and (path / f"{path.name}-sf.cif").is_file())
    if ids_path is not None:
        import download

        wanted = set(download.read_ids(ids_path))
        folders = [path for path in folders if path.name in wanted]
        missing = wanted.difference(path.name for path in folders)
        if missing:
            raise SystemExit("missing structures: " + ", ".join(sorted(missing)))
    if wanted_one:
        folders = [path for path in folders if path.name == wanted_one]
        if not folders:
            raise SystemExit(f"no structure {wanted_one}")
    return folders


if __name__ == "__main__":
    raise SystemExit(main())
