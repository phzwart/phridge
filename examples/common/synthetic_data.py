"""Shared synthetic data utilities for MTZ reading and ground truth extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    from cctbx.array_family import flex
    from iotbx.reflection_file_reader import any_reflection_file
except ImportError:
    flex = None
    any_reflection_file = None


def extract_itrue_from_mtz(mtz_path: Path, ref_miller_array: Optional[Any] = None) -> Optional[np.ndarray]:
    """Extract noise-free ground truth intensities (ITRUE) from MTZ, matching ref_miller_array if provided."""
    if any_reflection_file is None:
        raise ImportError("cctbx/iotbx is required to read reflection files")

    reader = any_reflection_file(str(mtz_path))
    arrays = reader.as_miller_arrays()
    itrue_arr = None
    for arr in arrays:
        lbls = [l.lower() for l in (arr.info().labels if arr.info() else [])]
        if "itrue" in lbls:
            itrue_arr = arr
            break

    if itrue_arr is None:
        return None

    if ref_miller_array is not None:
        matches = ref_miller_array.match_indices(itrue_arr)
        itrue_arr = itrue_arr.select(matches.permutation())

    return np.asarray(itrue_arr.data(), dtype=np.float64)


def build_noise_free_mtz(synth_mtz_path: Path, out_path: Path) -> Path:
    """Extract ITRUE from a synthetic MTZ and write an ideal noise-free dataset."""
    if any_reflection_file is None or flex is None:
        raise ImportError("cctbx/iotbx is required to read and write MTZ datasets")

    reader = any_reflection_file(str(synth_mtz_path))
    arrays = reader.as_miller_arrays()

    itrue_arr = None
    r_free_arr = None
    for a in arrays:
        lbls = [l.lower() for l in (a.info().labels if a.info() else [])]
        if "itrue" in lbls:
            itrue_arr = a
        elif "freer_flag" in lbls:
            r_free_arr = a

    if itrue_arr is None:
        raise ValueError(f"No ITRUE array found in {synth_mtz_path}")

    io_true = itrue_arr.data()
    sig_true = np.maximum(np.sqrt(np.maximum(np.asarray(io_true), 0.0)) * 0.01, 0.01)
    iobs_arr = itrue_arr.customized_copy(
        data=io_true,
        sigmas=flex.double(sig_true.tolist()),
    )
    iobs_arr.set_observation_type_xray_intensity()

    mtz_dataset = iobs_arr.as_mtz_dataset(column_root_label="IOBS")
    if r_free_arr is not None:
        mtz_dataset.add_miller_array(r_free_arr, column_root_label="FreeR_flag")
    mtz_dataset.add_miller_array(itrue_arr, column_root_label="ITRUE")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mtz_dataset.mtz_object().write(str(out_path))
    return out_path
