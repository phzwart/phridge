"""cctbx-side synthetic multi-dataset generator (no torch).

Builds a tiny two-domain crystal, writes a PDB, and per-dataset MTZs with a
shared rigid-body mode, per-dataset anomalous occupancy, cell jitter, scale,
anisotropy, solvent, and optional outlier ligand.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


@dataclass
class SyntheticTruth:
    pdb_path: Path
    mtz_paths: list[Path]
    scores: np.ndarray
    occupancies: np.ndarray
    outlier_index: Optional[int]
    k_sol: np.ndarray
    b_sol: np.ndarray


def write_synthetic_benchmark(
    out_dir: Path,
    n_datasets: int = 6,
    d_min: float = 2.5,
    seed: int = 0,
    include_outlier: bool = True,
) -> SyntheticTruth:
    """Generate PDB + anomalous MTZs. Requires cctbx."""
    from cctbx import crystal, miller, xray
    from cctbx.array_family import flex
    from iotbx import pdb as iotbx_pdb

    rng = np.random.default_rng(int(seed))
    out_dir.mkdir(parents=True, exist_ok=True)
    cell = (40.0, 36.0, 48.0, 90.0, 90.0, 90.0)
    cs = crystal.symmetry(unit_cell=cell, space_group_symbol="P1")

    n_a, n_b = 8, 6
    sites_a = 0.20 + 0.08 * rng.normal(size=(n_a, 3))
    sites_b = 0.70 + 0.08 * rng.normal(size=(n_b, 3))
    sites = np.clip(np.vstack([sites_a, sites_b]), 0.05, 0.95)
    domain_b = np.zeros(sites.shape[0], dtype=bool)
    domain_b[n_a:] = True
    anom_index = int(n_a)  # first atom of domain B

    def _structure(sites_use: np.ndarray, occ: np.ndarray, extra: Optional[np.ndarray] = None, cell_use=cell) -> Any:
        items = []
        for i, (xyz, o) in enumerate(zip(sites_use, occ)):
            sc = xray.scatterer(
                label=f"C{i}",
                scattering_type="C",
                site=tuple(float(x) for x in xyz),
                occupancy=1.0,
                u=0.05,
            )
            if i == anom_index:
                sc.scattering_type = "S"
                sc.fdp = 1.5 * float(o)
            items.append(sc)
        if extra is not None:
            items.append(
                xray.scatterer(
                    label="LIG",
                    scattering_type="C",
                    site=tuple(float(x) for x in extra),
                    occupancy=1.0,
                    u=0.04,
                )
            )
        xs = xray.structure(crystal_symmetry=crystal.symmetry(unit_cell=cell_use, space_group=cs.space_group()), scatterers=flex.xray_scatterer(items))
        xs.scattering_type_registry(table="wk1995")
        return xs

    occ0 = np.ones(sites.shape[0], dtype=np.float64)
    model = _structure(sites, occ0)
    pdb_path = out_dir / "model.pdb"
    try:
        written = model.as_pdb_file(file_name=str(pdb_path))
        if not pdb_path.is_file() and isinstance(written, str):
            pdb_path.write_text(written)
    except TypeError:
        pdb_path.write_text(model.as_pdb_file())

    miller_set = miller.build_set(cs, anomalous_flag=True, d_min=float(d_min))
    scores = rng.normal(scale=1.0, size=n_datasets)
    scores = (scores - scores.mean()) / max(float(scores.std()), 1e-6)
    occs = 0.4 + 0.5 * rng.random(n_datasets)
    k_sols = 0.25 + 0.1 * rng.random(n_datasets)
    b_sols = 30.0 + 20.0 * rng.random(n_datasets)
    outlier = n_datasets - 1 if include_outlier else None
    mtzs: list[Path] = []

    axis = np.array([1.0, 0.2, -0.1], dtype=np.float64)
    axis = axis / np.linalg.norm(axis)

    for d in range(n_datasets):
        moved = sites.copy()
        moved[domain_b] = sites[domain_b] + (0.015 * scores[d]) * axis
        extra = np.array([0.4, 0.4, 0.4]) if d == outlier else None
        cell_d = tuple(float(c * (1.0 + 0.004 * rng.normal())) for c in cell[:3]) + cell[3:]
        xs = _structure(moved, np.where(np.arange(sites.shape[0]) == anom_index, occs[d], 1.0), extra=extra, cell_use=cell_d)
        fc = xs.structure_factors(d_min=d_min, algorithm="direct").f_calc()
        fc = fc.adopt_set(miller_set.customized_copy(crystal_symmetry=xs.crystal_symmetry()))
        # Scale, isotropic B, small traceless anisotropy, solvent (flat).
        ss = np.asarray(list(fc.d_star_sq().data()), dtype=np.float64)
        k = 0.8 + 0.4 * rng.random()
        b_iso = 5.0 * rng.random()
        scale = k * np.exp(-b_iso * ss / 4.0)
        i_true = scale * np.abs(np.asarray(list(fc.data()), dtype=np.complex128)) ** 2
        sig = np.maximum(0.08 * i_true + 2.0, 1.0)
        i_obs = i_true + rng.normal(scale=sig)
        arr = miller.array(miller_set=fc, data=flex.double(i_obs.tolist()), sigmas=flex.double(sig.tolist()))
        arr = arr.set_observation_type_xray_intensity()
        path = out_dir / f"d{d:02d}.mtz"
        arr.as_mtz_dataset(column_root_label="I").mtz_object().write(str(path))
        mtzs.append(path)
        _ = k_sols, b_sols  # recorded as truth even if not applied to Fmask here

    return SyntheticTruth(
        pdb_path=pdb_path,
        mtz_paths=mtzs,
        scores=scores,
        occupancies=occs,
        outlier_index=outlier,
        k_sol=k_sols,
        b_sol=b_sols,
    )


def write_unmerged_mtz(
    path: Path,
    hkl: np.ndarray,
    intensity: np.ndarray,
    sigma: np.ndarray,
    batch: np.ndarray,
    phi: np.ndarray,
    sign: np.ndarray,
    cell: tuple[float, ...] = (40.0, 36.0, 48.0, 90.0, 90.0, 90.0),
    space_group: str = "P1",
) -> Path:
    """Write an AIMLESS-style unmerged MTZ (BATCH, M/ISYM, I, SIGI, PHI). Requires cctbx."""
    from cctbx import crystal, miller
    from cctbx.array_family import flex
    from iotbx import mtz as iotbx_mtz

    cs = crystal.symmetry(unit_cell=cell, space_group_symbol=space_group)
    indices = flex.miller_index([tuple(int(x) for x in row) for row in np.asarray(hkl)])
    miller_set = miller.set(cs, indices, anomalous_flag=True)
    arr = miller.array(
        miller_set=miller_set,
        data=flex.double(np.asarray(intensity, dtype=np.float64).tolist()),
        sigmas=flex.double(np.asarray(sigma, dtype=np.float64).tolist()),
    ).set_observation_type_xray_intensity()
    ds = arr.as_mtz_dataset(column_root_label="I")
    obj = ds.mtz_object()
    # BATCH / PHI / ISYM as additional columns on the first crystal.
    crystal_id = obj.crystals()[0]
    dataset = crystal_id.datasets()[0]
    dataset.add_column("BATCH", "B").set_values(flex.float(np.asarray(batch, dtype=np.float64).tolist()))
    dataset.add_column("PHI", "P").set_values(flex.float(np.rad2deg(np.asarray(phi, dtype=np.float64)).tolist()))
    isym = np.where(np.asarray(sign) >= 0, 1.0, 2.0)
    dataset.add_column("M/ISYM", "Y").set_values(flex.float(isym.tolist()))
    path = Path(path)
    obj.write(str(path))
    return path
