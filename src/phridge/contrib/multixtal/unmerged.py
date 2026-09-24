"""Read unmerged AIMLESS/DIALS reflections. Torch-free; cctbx/dials optional."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from phridge.contrib.multixtal.dose_io import (
    CrystalDoseReport,
    attach_dose,
    beam_frame_vectors,
    load_dose_table,
    report_crystal,
)
from phridge.contrib.multixtal.indexing import choose_indexing


@dataclass
class ObservationTable:
    """Flat observation arrays aligned to a common ASU Miller set."""

    i: np.ndarray
    sig: np.ndarray
    h: np.ndarray
    crystal: np.ndarray
    batch: np.ndarray
    dose: np.ndarray
    phi: np.ndarray
    sign: np.ndarray
    s_inc: np.ndarray
    s_dif: np.ndarray
    wedge: np.ndarray
    pass_id: np.ndarray
    reports: list[CrystalDoseReport] = field(default_factory=list)


def bijvoet_sign_from_isym(isym: int) -> int:
    """cctbx/CCP4 ISYM: odd → plus hemisphere, even → minus (centric → 0 handled by caller)."""
    v = int(isym)
    if v <= 0:
        return 1
    return 1 if (v % 2 == 1) else -1


def _column(obj: Any, *names: str) -> Optional[np.ndarray]:
    labels = [str(x).lower() for x in obj.column_labels()]
    for name in names:
        key = name.lower()
        if key in labels:
            col = obj.get_column(obj.column_labels()[labels.index(key)])
            return np.asarray(list(col.extract_values()), dtype=np.float64)
    return None


def read_unmerged_mtz(path: str | Path) -> dict[str, np.ndarray]:
    """Pull I, SIGI, BATCH, M/ISYM, PHI, HKL from an unmerged MTZ."""
    import iotbx.mtz

    mtz = iotbx.mtz.object(file_name=str(path))
    hkl = np.asarray(list(mtz.extract_miller_indices()), dtype=np.int32)
    i = _column(mtz, "I", "IMEAN", "I(+)")
    sig = _column(mtz, "SIGI", "SIGIMEAN", "SIGI(+)")
    if i is None:
        raise ValueError(f"no intensity column in {path}")
    if sig is None:
        sig = np.ones_like(i)
    batch = _column(mtz, "BATCH", "IMAGE", "FRAME")
    if batch is None:
        batch = np.zeros(i.size)
    isym = _column(mtz, "M/ISYM", "ISYM", "M_ISYM")
    phi = _column(mtz, "PHI", "ROT", "OMEGA")
    if phi is None:
        phi = np.zeros(i.size)
    else:
        # AIMLESS stores degrees.
        if np.nanmax(np.abs(phi)) > 2.0 * np.pi + 0.5:
            phi = np.deg2rad(phi)
    sign = np.ones(i.size, dtype=np.int64)
    if isym is not None:
        sign = np.array([bijvoet_sign_from_isym(int(v)) for v in isym], dtype=np.int64)
    return {
        "hkl": hkl,
        "i": i,
        "sig": np.maximum(sig, 1e-8),
        "batch": batch.astype(np.int64),
        "phi": phi,
        "sign": sign,
    }


def read_dials_refl(refl_path: str | Path, expt_path: str | Path) -> dict[str, np.ndarray]:
    """Optional DIALS ``scaled.refl`` + ``scaled.expt``. Raises if dials is missing."""
    from dxtbx.model.experiment_list import ExperimentListFactory  # type: ignore
    from dials.array_family import flex  # type: ignore

    experiments = ExperimentListFactory.from_json_file(str(expt_path), check_format=False)
    table = flex.reflection_table.from_file(str(refl_path))
    hkl = np.asarray(list(table["miller_index"]), dtype=np.int32)
    i = np.asarray(list(table["intensity.scale.value" if "intensity.scale.value" in table else "intensity.sum.value"]))
    sig2 = np.asarray(
        list(table["intensity.scale.variance" if "intensity.scale.variance" in table else "intensity.sum.variance"])
    )
    batch = np.asarray(list(table["xyzobs.px.value"]))[:, 2] if "xyzobs.px.value" in table else np.zeros(len(i))
    phi = np.zeros(len(i))
    if "xyzobs.mm.value" in table:
        phi = np.asarray(list(table["xyzobs.mm.value"]))[:, 2]
    sign = np.ones(len(i), dtype=np.int64)
    _ = experiments
    return {
        "hkl": hkl,
        "i": np.asarray(i, dtype=np.float64),
        "sig": np.sqrt(np.maximum(np.asarray(sig2, dtype=np.float64), 1e-12)),
        "batch": np.asarray(batch, dtype=np.int64),
        "phi": np.asarray(phi, dtype=np.float64),
        "sign": sign,
    }


def observations_from_arrays(
    crystals: list[dict[str, np.ndarray]],
    common_hkl: np.ndarray,
    cells: np.ndarray,
    *,
    dose_tables: Optional[list[Optional[dict[tuple[str, int], float]]]] = None,
    dose_rate: float = 0.05,
    names: Optional[list[str]] = None,
    n_bins: int = 10,
    confound_corr: float = 0.85,
    wavelengths: Optional[np.ndarray] = None,
) -> ObservationTable:
    """Stack per-crystal unmerged arrays onto the common ASU index set."""
    lookup = {tuple(int(x) for x in row): i for i, row in enumerate(np.asarray(common_hkl, dtype=np.int64))}
    n_h = int(common_hkl.shape[0])
    parts: list[dict[str, np.ndarray]] = []
    reports: list[CrystalDoseReport] = []
    names = names or [str(i) for i in range(len(crystals))]
    dose_tables = dose_tables or [None] * len(crystals)
    waves = np.ones(len(crystals)) if wavelengths is None else np.asarray(wavelengths, dtype=np.float64)
    for d, raw in enumerate(crystals):
        hkl = np.asarray(raw["hkl"], dtype=np.int64)
        mapped = np.array([lookup.get((int(h[0]), int(h[1]), int(h[2])), -1) for h in hkl], dtype=np.int64)
        # Also try Friedel mate as the ASU of -h with opposite sign.
        for i, h in enumerate(hkl):
            if mapped[i] >= 0:
                continue
            key = (-int(h[0]), -int(h[1]), -int(h[2]))
            if key in lookup:
                mapped[i] = lookup[key]
                raw["sign"] = np.array(raw["sign"], dtype=np.int64)
                raw["sign"][i] = -int(raw["sign"][i])
        keep = mapped >= 0
        if not np.any(keep):
            continue
        batch = np.asarray(raw["batch"], dtype=np.int64)[keep]
        phi = np.asarray(raw["phi"], dtype=np.float64)[keep]
        table = dose_tables[d]
        dose = attach_dose(batch, dataset_name=names[d], table=table, dose_rate=dose_rate)
        h_idx = mapped[keep]
        s_inc, s_dif = beam_frame_vectors(
            hkl[keep],
            tuple(float(x) for x in np.asarray(cells[d]).reshape(6)),
            phi,
            wavelength=float(waves[d]) if d < waves.size and np.isfinite(waves[d]) else 1.0,
        )
        wedge = collection_wedge(batch, phi)
        reports.append(
            report_crystal(d, dose, phi, batch, h_idx, n_h, n_bins=n_bins, confound_corr=confound_corr)
        )
        parts.append(
            {
                "i": np.asarray(raw["i"], dtype=np.float64)[keep],
                "sig": np.asarray(raw["sig"], dtype=np.float64)[keep],
                "h": h_idx,
                "crystal": np.full(int(keep.sum()), d, dtype=np.int64),
                "batch": batch,
                "dose": dose,
                "phi": phi,
                "sign": np.asarray(raw["sign"], dtype=np.int64)[keep],
                "s_inc": s_inc,
                "s_dif": s_dif,
                "wedge": wedge,
                "pass_id": np.full(int(keep.sum()), reports[-1].n_passes > 1, dtype=np.int64) * 0
                + _pass_id(phi),
            }
        )
    if not parts:
        raise ValueError("no unmerged observations mapped onto the common Miller set")
    return ObservationTable(
        i=np.concatenate([p["i"] for p in parts]),
        sig=np.concatenate([p["sig"] for p in parts]),
        h=np.concatenate([p["h"] for p in parts]),
        crystal=np.concatenate([p["crystal"] for p in parts]),
        batch=np.concatenate([p["batch"] for p in parts]),
        dose=np.concatenate([p["dose"] for p in parts]),
        phi=np.concatenate([p["phi"] for p in parts]),
        sign=np.concatenate([p["sign"] for p in parts]),
        s_inc=np.vstack([p["s_inc"] for p in parts]),
        s_dif=np.vstack([p["s_dif"] for p in parts]),
        wedge=np.concatenate([p["wedge"] for p in parts]),
        pass_id=np.concatenate([p["pass_id"] for p in parts]),
        reports=reports,
    )


def collection_wedge(batch: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Within-wedge frame position in [0, 1] from first-seen BATCH order."""
    from phridge.contrib.multixtal.dose_io import collection_index

    idx = collection_index(batch).astype(np.float64)
    mx = float(idx.max()) if idx.size else 0.0
    return idx / mx if mx > 0 else np.zeros_like(idx)


def _pass_id(phi: np.ndarray) -> np.ndarray:
    ang = np.asarray(phi, dtype=np.float64)
    if ang.size == 0:
        return np.zeros(0, dtype=np.int64)
    unwrapped = np.unwrap(np.where(np.isfinite(ang), ang, 0.0))
    return np.maximum(np.floor(unwrapped / (2.0 * np.pi)).astype(np.int64), 0)


def crystal_means_from_obs(
    obs: ObservationTable,
    n_crystal: int,
    n_h: int,
    centric: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-crystal I+/I- means from observations. Never merge across crystals."""
    i_plus = np.zeros((n_crystal, n_h), dtype=np.float64)
    i_minus = np.zeros((n_crystal, n_h), dtype=np.float64)
    sig_plus = np.ones((n_crystal, n_h), dtype=np.float64)
    sig_minus = np.ones((n_crystal, n_h), dtype=np.float64)
    mask_plus = np.zeros((n_crystal, n_h), dtype=bool)
    mask_minus = np.zeros((n_crystal, n_h), dtype=bool)
    cen = np.asarray(centric, dtype=bool)
    for d in range(n_crystal):
        for h in range(n_h):
            sel_p = (obs.crystal == d) & (obs.h == h) & ((obs.sign >= 0) | cen[h])
            sel_m = (obs.crystal == d) & (obs.h == h) & (obs.sign < 0) & (~cen[h])
            if np.any(sel_p):
                w = 1.0 / np.maximum(obs.sig[sel_p] ** 2, 1e-12)
                i_plus[d, h] = float(np.sum(w * obs.i[sel_p]) / np.sum(w))
                sig_plus[d, h] = float(np.sqrt(1.0 / np.sum(w)))
                mask_plus[d, h] = True
            if np.any(sel_m):
                w = 1.0 / np.maximum(obs.sig[sel_m] ** 2, 1e-12)
                i_minus[d, h] = float(np.sum(w * obs.i[sel_m]) / np.sum(w))
                sig_minus[d, h] = float(np.sqrt(1.0 / np.sum(w)))
                mask_minus[d, h] = True
    return i_plus, i_minus, sig_plus, sig_minus, mask_plus, mask_minus
