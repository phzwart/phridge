"""cctbx I/O, placement, common Miller set, F_mask. No torch."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from phridge.contrib.multixtal.indexing import IndexingChoice, choose_indexing
from phridge.contrib.multixtal.options import DatasetManifest, DatasetSpec, MultixtalOptions, Placement


@dataclass
class PreparedDataset:
    name: str
    path: str
    labels: str
    wavelength: Optional[float]
    notes: Optional[str]
    cell: tuple[float, ...]
    indexing: IndexingChoice
    i_plus: np.ndarray
    i_minus: np.ndarray
    sig_plus: np.ndarray
    sig_minus: np.ndarray
    mask_plus: np.ndarray
    mask_minus: np.ndarray
    f_mask: np.ndarray
    sites_frac: np.ndarray


@dataclass
class MultixtalJob:
    xray: Any
    table: Any
    params: Any
    hkl: np.ndarray
    centric: np.ndarray
    epsilon: np.ndarray
    cells: np.ndarray
    sites_frac: np.ndarray
    wavelengths: np.ndarray
    i_plus: np.ndarray
    i_minus: np.ndarray
    sig_plus: np.ndarray
    sig_minus: np.ndarray
    mask_plus: np.ndarray
    mask_minus: np.ndarray
    f_mask: np.ndarray
    options: MultixtalOptions
    datasets: list[PreparedDataset]
    placement: Placement
    g_site: Optional[np.ndarray] = None
    obs: Optional[Any] = None
    logs: list[str] = field(default_factory=list)


def _parse_labels(text: str) -> list[str]:
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError('labels must be four comma-separated names, e.g. "I(+),SIGI(+),I(-),SIGI(-)"')
    return parts


def load_manifest(path: str | Path) -> DatasetManifest:
    text = Path(path).read_text()
    stripped = text.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        data = json.loads(text)
        return DatasetManifest.from_mapping(data)
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        return DatasetManifest.from_mapping(data)
    except ImportError:
        return DatasetManifest.from_mapping(_simple_yaml_datasets(text))


def _simple_yaml_datasets(text: str) -> dict[str, Any]:
    """Minimal fallback for the documented manifest shape."""
    datasets: list[dict[str, Any]] = []
    current: Optional[dict[str, Any]] = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("- "):
            if current:
                datasets.append(current)
            current = {}
            line = line[2:].strip()
            if line and ":" in line:
                key, val = line.split(":", 1)
                current[key.strip()] = _scalar(val)
            continue
        if ":" in line and current is not None:
            key, val = line.split(":", 1)
            if key.strip() != "datasets":
                current[key.strip()] = _scalar(val)
    if current:
        datasets.append(current)
    return {"datasets": datasets}


def _scalar(value: str) -> Any:
    text = value.strip().strip('"').strip("'")
    if not text:
        return None
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def _read_pdb(path: str | Path, space_group: Optional[str], cell: Optional[Sequence[float]]) -> Any:
    import iotbx.pdb
    from cctbx import crystal, uctbx

    pdb = iotbx.pdb.input(file_name=str(path))
    hierarchy = pdb.construct_hierarchy()
    xray = pdb.xray_structure_simple()
    if space_group is not None or cell is not None:
        params = tuple(cell) if cell is not None else xray.unit_cell().parameters()
        if space_group is not None:
            from cctbx import sgtbx

            group = sgtbx.space_group_info(symbol=str(space_group)).group()
        else:
            group = xray.space_group()
        cs = crystal.symmetry(unit_cell=uctbx.unit_cell(parameters=tuple(float(x) for x in params)), space_group=group)
        xray = xray.customized_copy(crystal_symmetry=cs)
    return hierarchy, xray


def _intensity_from_mtz(path: str | Path, labels: str) -> Any:
    from iotbx.reflection_file_reader import any_reflection_file

    names = _parse_labels(labels)
    reader = any_reflection_file(str(path))
    arrays = reader.as_miller_arrays()
    wanted = {n.lower(): n for n in names}
    found: dict[str, Any] = {}
    for arr in arrays:
        info = arr.info()
        labs = [str(x) for x in (info.labels if info is not None and info.labels else [])]
        joined = ",".join(labs).lower()
        for key in list(wanted):
            if key in joined or any(key == lab.lower() for lab in labs):
                found[key] = arr
    # Prefer a single anomalous intensity array.
    for arr in arrays:
        if arr.is_xray_intensity_array() and arr.anomalous_flag() and arr.sigmas() is not None:
            return arr
    if len(found) >= 4:
        # Build from four columns if needed — fall through to first intensity.
        pass
    for arr in arrays:
        if arr.is_xray_intensity_array() and arr.sigmas() is not None:
            if not arr.anomalous_flag():
                arr = arr.generate_bijvoet_mates()
            return arr
    raise ValueError(f"no intensity array with sigmas in {path}")


def _fc_abs2(xray: Any, miller_set: Any) -> Any:
    fc = xray.structure_factors(d_min=float(miller_set.d_min()), algorithm="direct").f_calc()
    fc = fc.set_observation_type_xray_amplitude()
    return (fc.as_intensity_array()).map_to_asu()


def _unique_asu_indices(arrays: list[Any], d_min: float, model_symmetry: Any) -> Any:
    from cctbx import miller
    from cctbx.array_family import flex

    seen: set[tuple[int, int, int]] = set()
    for arr in arrays:
        cut = arr.resolution_filter(d_min=d_min)
        non_anom = cut.as_non_anomalous_array() if cut.anomalous_flag() else cut
        for h in non_anom.map_to_asu().indices():
            seen.add((int(h[0]), int(h[1]), int(h[2])))
    if not seen:
        raise ValueError("common Miller set is empty")
    indices = flex.miller_index(sorted(seen))
    return miller.set(model_symmetry, indices, anomalous_flag=False)


def _lookup(arr: Any) -> dict[tuple[int, int, int], tuple[float, float]]:
    out: dict[tuple[int, int, int], tuple[float, float]] = {}
    sig = arr.sigmas()
    for i, h in enumerate(arr.indices()):
        key = (int(h[0]), int(h[1]), int(h[2]))
        s = float(sig[i]) if sig is not None else 1.0
        out[key] = (float(arr.data()[i]), s)
    return out


def _place_sites(sites_frac: np.ndarray, model_cell: Any, dataset_cell: Any, placement: Placement) -> np.ndarray:
    if placement == "fractional":
        return np.asarray(sites_frac, dtype=np.float64)
    cart = np.asarray(model_cell.orthogonalize(sites_frac.tolist()), dtype=np.float64)
    frac = np.asarray(dataset_cell.fractionalize(cart.tolist()), dtype=np.float64)
    return frac


def _f_mask_on_set(xray: Any, miller_set: Any) -> np.ndarray:
    from phridge.contrib.intensity_ll.dt_mask import build_f_masks

    dummy = miller_set.array(data=miller_set.indices().as_vec3_double().parts()[0])
    try:
        built = build_f_masks(dummy, xray)
        arr = built.f_mask
        lookup = { (int(h[0]), int(h[1]), int(h[2])): complex(v) for h, v in zip(arr.indices(), arr.data()) }
    except Exception:
        lookup = {}
    out = np.zeros(miller_set.size(), dtype=np.complex128)
    for i, h in enumerate(miller_set.indices()):
        out[i] = lookup.get((int(h[0]), int(h[1]), int(h[2])), 0.0)
    return out


def prepare_job(
    model_path: str | Path,
    data_paths: Sequence[str | Path],
    options: MultixtalOptions,
    manifest: Optional[DatasetManifest] = None,
    anom_sel: Optional[str] = None,
) -> MultixtalJob:
    """Read PDB + MTZs, reindex, build the common set, compute F_mask, pack arrays."""
    from phridge.client.convert_xtal import scattering_table_from_cctbx, xray_from_cctbx
    from phridge.models import SfEngineParams

    specs: list[DatasetSpec]
    if manifest is not None:
        specs = list(manifest.datasets)
    else:
        specs = [
            DatasetSpec(file=str(p), labels=options.labels, name=Path(p).stem)
            for p in data_paths
        ]
    if not specs:
        raise ValueError("at least one dataset is required")

    _hier, xray_cctbx = _read_pdb(model_path, options.space_group, options.cell)
    model_sym = xray_cctbx.crystal_symmetry()
    sites = np.asarray(list(xray_cctbx.sites_frac()), dtype=np.float64)
    model_cell = xray_cctbx.unit_cell()

    intensities: list[Any] = []
    choices: list[IndexingChoice] = []
    names: list[DatasetSpec] = []
    logs: list[str] = [f"placement={options.placement}"]
    fc_ref = None
    for spec in specs:
        labels = spec.labels or options.labels
        intens = _intensity_from_mtz(spec.file, labels)
        if fc_ref is None:
            fc_ref = _fc_abs2(xray_cctbx, intens.resolution_filter(d_min=options.d_min))
        chosen, choice = choose_indexing(intens, fc_ref, intens.crystal_symmetry(), model_sym)
        logs.append(
            f"{Path(spec.file).name}: indexing {choice.operator}  corr={choice.correlation:.4f}  margin={choice.margin}"
        )
        intensities.append(chosen)
        choices.append(choice)
        names.append(spec)

    common = _unique_asu_indices(intensities, options.d_min, model_sym)
    n_h = common.size()
    hkl = np.asarray([(int(h[0]), int(h[1]), int(h[2])) for h in common.indices()], dtype=np.int32)
    centric = np.asarray(list(common.centric_flags().data()), dtype=bool)
    epsilon = np.asarray(list(common.epsilons().data()), dtype=np.float64)

    n_data = len(intensities)
    i_plus = np.zeros((n_data, n_h), dtype=np.float64)
    i_minus = np.zeros((n_data, n_h), dtype=np.float64)
    sig_plus = np.ones((n_data, n_h), dtype=np.float64)
    sig_minus = np.ones((n_data, n_h), dtype=np.float64)
    mask_plus = np.zeros((n_data, n_h), dtype=bool)
    mask_minus = np.zeros((n_data, n_h), dtype=bool)
    f_mask = np.zeros((n_data, n_h), dtype=np.complex128)
    cells = np.zeros((n_data, 6), dtype=np.float64)
    sites_all = np.zeros((n_data, sites.shape[0], 3), dtype=np.float64)
    waves = np.full(n_data, np.nan, dtype=np.float64)
    prepared: list[PreparedDataset] = []

    for d, (spec, intens, choice) in enumerate(zip(names, intensities, choices)):
        cell = intens.unit_cell()
        cells[d] = cell.parameters()
        if spec.wavelength is not None:
            waves[d] = spec.wavelength
        sites_all[d] = _place_sites(sites, model_cell, cell, options.placement)
        from cctbx import crystal

        ds_xray = xray_cctbx.customized_copy(
            crystal_symmetry=crystal.symmetry(unit_cell=cell, space_group=model_sym.space_group())
        )
        try:
            from scitbx.array_family import flex

            ds_xray.set_sites_frac(flex.vec3_double(sites_all[d].tolist()))
        except Exception:
            ds_xray.set_sites_frac(sites_all[d].tolist())
        f_mask[d] = _f_mask_on_set(ds_xray, common.customized_copy(crystal_symmetry=ds_xray.crystal_symmetry()))
        plus = _lookup(intens)
        # After map_to_asu with anomalous_flag, +h is ASU and -h is the mate.
        for i, h in enumerate(common.indices()):
            hp = (int(h[0]), int(h[1]), int(h[2]))
            hm = (-hp[0], -hp[1], -hp[2])
            if hp in plus:
                i_plus[d, i], sig_plus[d, i] = plus[hp]
                mask_plus[d, i] = True
            if hm in plus:
                i_minus[d, i], sig_minus[d, i] = plus[hm]
                mask_minus[d, i] = True
            # Some files store the minus mate already as the ASU of -h with a flag;
            # hemispheres helper covers the usual anomalous array.
        if intens.anomalous_flag():
            try:
                p_arr, m_arr = intens.hemispheres_acentrics()
                p_map, m_map = _lookup(p_arr), _lookup(m_arr)
                for i, h in enumerate(common.indices()):
                    hp = (int(h[0]), int(h[1]), int(h[2]))
                    if hp in p_map:
                        i_plus[d, i], sig_plus[d, i] = p_map[hp]
                        mask_plus[d, i] = True
                    if hp in m_map:
                        i_minus[d, i], sig_minus[d, i] = m_map[hp]
                        mask_minus[d, i] = True
            except Exception:
                pass
        prepared.append(
            PreparedDataset(
                name=spec.name or Path(spec.file).stem,
                path=str(spec.file),
                labels=spec.labels or options.labels,
                wavelength=spec.wavelength,
                notes=spec.notes,
                cell=tuple(float(x) for x in cells[d]),
                indexing=choice,
                i_plus=i_plus[d],
                i_minus=i_minus[d],
                sig_plus=sig_plus[d],
                sig_minus=sig_minus[d],
                mask_plus=mask_plus[d],
                mask_minus=mask_minus[d],
                f_mask=f_mask[d],
                sites_frac=sites_all[d],
            )
        )

    packed_xray = xray_from_cctbx(xray_cctbx)
    table = scattering_table_from_cctbx(xray_cctbx)
    params = SfEngineParams(d_min=float(options.d_min), dtype="float64")

    g_site = None
    if options.anom_sites:
        g_site = _site_g(xray_cctbx, common, options.anom_sites)

    obs = _load_unmerged(specs, options, hkl, cells, waves, names, logs)
    if obs is not None and not any(np.any(mask_plus[d]) or np.any(mask_minus[d]) for d in range(n_data)):
        from phridge.contrib.multixtal.unmerged import crystal_means_from_obs

        i_plus, i_minus, sig_plus, sig_minus, mask_plus, mask_minus = crystal_means_from_obs(
            obs, n_data, n_h, centric
        )

    return MultixtalJob(
        xray=packed_xray,
        table=table,
        params=params,
        hkl=hkl,
        centric=centric,
        epsilon=epsilon,
        cells=cells,
        sites_frac=sites_all,
        wavelengths=waves,
        i_plus=i_plus,
        i_minus=i_minus,
        sig_plus=sig_plus,
        sig_minus=sig_minus,
        mask_plus=mask_plus,
        mask_minus=mask_minus,
        f_mask=f_mask,
        options=options,
        datasets=prepared,
        placement=options.placement,
        g_site=g_site,
        obs=obs,
        logs=logs,
    )


def _load_unmerged(
    specs: list[DatasetSpec],
    options: MultixtalOptions,
    hkl: np.ndarray,
    cells: np.ndarray,
    waves: np.ndarray,
    names: list[DatasetSpec],
    logs: list[str],
) -> Optional[Any]:
    """Read optional unmerged MTZ / DIALS files and attach dose + geometry."""
    paths = [s.unmerged for s in specs]
    if not any(paths):
        return None
    from phridge.contrib.multixtal.dose_io import load_dose_table
    from phridge.contrib.multixtal.unmerged import observations_from_arrays, read_dials_refl, read_unmerged_mtz

    crystals: list[dict[str, np.ndarray]] = []
    dose_tables: list[Optional[dict]] = []
    present = []
    for spec, path in zip(specs, paths):
        if not path:
            continue
        p = Path(path)
        if p.suffix.lower() == ".refl":
            expt = p.with_suffix(".expt")
            raw = read_dials_refl(p, expt if expt.is_file() else p)
        else:
            raw = read_unmerged_mtz(p)
        crystals.append(raw)
        present.append(spec)
        table = None
        if spec.dose_table:
            table = load_dose_table(spec.dose_table)
        dose_tables.append(table)
    if not crystals:
        return None
    # Align to the full spec list: missing crystals get empty arrays skipped by observations_from_arrays.
    aligned: list[dict[str, np.ndarray]] = []
    aligned_tables: list[Optional[dict]] = []
    aligned_names: list[str] = []
    idx = 0
    for spec, path in zip(specs, paths):
        if path:
            aligned.append(crystals[idx])
            aligned_tables.append(dose_tables[idx])
            idx += 1
        else:
            aligned.append(
                {
                    "hkl": np.zeros((0, 3), dtype=np.int64),
                    "i": np.zeros(0),
                    "sig": np.zeros(0),
                    "batch": np.zeros(0, dtype=np.int64),
                    "phi": np.zeros(0),
                    "sign": np.zeros(0, dtype=np.int64),
                }
            )
            aligned_tables.append(None)
        aligned_names.append(spec.name or Path(spec.file).stem)
    obs = observations_from_arrays(
        aligned,
        hkl,
        cells,
        dose_tables=aligned_tables,
        dose_rate=float(options.dose_rate),
        names=aligned_names,
        n_bins=options.dose_bins,
        confound_corr=options.confound_corr,
        wavelengths=waves,
    )
    for rep in obs.reports:
        logs.append(
            f"crystal {rep.crystal}: dose [{rep.dose_min:.3g}, {rep.dose_max:.3g}] "
            f"passes={rep.n_passes} dose-angle={rep.dose_angle_corr:.3f} confound={rep.confound} n={rep.n_obs}"
        )
    return obs


def _site_g(xray: Any, miller_set: Any, site_indices: Sequence[int]) -> np.ndarray:
    """Approximate G_site from a single-site fdp structure-factor calculation."""
    clone = xray.deep_copy_scatterers()
    scs = clone.scatterers()
    for i, sc in enumerate(scs):
        if i not in set(int(x) for x in site_indices):
            sc.occupancy = 0.0
            sc.fdp = 0.0
        else:
            if sc.fdp in (None, 0):
                sc.fdp = 1.0
    fc = clone.structure_factors(d_min=float(miller_set.d_min()), algorithm="direct").f_calc()
    fc = fc.map_to_asu()
    lookup = {(int(h[0]), int(h[1]), int(h[2])): complex(v) for h, v in zip(fc.indices(), fc.data())}
    out = np.zeros(miller_set.size(), dtype=np.complex128)
    for i, h in enumerate(miller_set.indices()):
        out[i] = lookup.get((int(h[0]), int(h[1]), int(h[2])), 0.0)
    # F ≈ i G when only fdp is present (real form factor zeroed by occ=0 on others).
    return out / 1.0j


def job_to_op_kwargs(job: MultixtalJob) -> dict[str, Any]:
    payload = {
        "xray": job.xray,
        "table": job.table,
        "params": job.params,
        "sites_frac": job.sites_frac,
        "cells": job.cells,
        "wavelengths": job.wavelengths,
        "hkl": job.hkl,
        "centric": job.centric,
        "epsilon": job.epsilon,
        "i_plus": job.i_plus,
        "sig_plus": job.sig_plus,
        "i_minus": job.i_minus,
        "sig_minus": job.sig_minus,
        "mask_plus": job.mask_plus,
        "mask_minus": job.mask_minus,
        "f_mask": job.f_mask,
        "options": job.options.model_dump(),
    }
    if job.g_site is not None:
        payload["g_site"] = job.g_site
    if job.obs is not None:
        payload.update(
            {
                "obs_i": job.obs.i,
                "obs_sig": job.obs.sig,
                "obs_h": job.obs.h,
                "obs_crystal": job.obs.crystal,
                "obs_batch": job.obs.batch,
                "obs_dose": job.obs.dose,
                "obs_phi": job.obs.phi,
                "obs_sign": job.obs.sign,
                "obs_s_inc": job.obs.s_inc,
                "obs_s_dif": job.obs.s_dif,
                "obs_wedge": job.obs.wedge,
                "obs_pass": job.obs.pass_id,
            }
        )
    return payload
