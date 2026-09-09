#!/usr/bin/env python3
"""Bin MTZ/mmCIF intensities and report I/σ (+ optional σ_A / Wilson) diagnostics.

Uses the same table as the in-refine ``PHRIDGE_STATS_REPORT`` banner.

Example:
  python scripts/mtz_intensity_bins.py data.mtz
  python scripts/mtz_intensity_bins.py 9RRL-sf.cif --bin-size 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

from phridge.client.intensity.stats_report import (
    compute_intensity_stats_report,
    print_intensity_stats_report,
)


def _load_intensity_array(mtz_path: Path, label: Optional[str] = None) -> Any:
    from iotbx.reflection_file_reader import any_reflection_file

    reader = any_reflection_file(str(mtz_path))
    arrays = reader.as_miller_arrays()
    if not arrays:
        raise SystemExit(f"No Miller arrays found in {mtz_path}")

    chosen = None
    if label is not None:
        key = label.lower()
        for arr in arrays:
            labels = [str(x).lower() for x in (arr.info().labels if arr.info() else [])]
            if any(key in lab for lab in labels):
                chosen = arr
                break
        if chosen is None:
            avail = []
            for arr in arrays:
                labs = ",".join(arr.info().labels) if arr.info() and arr.info().labels else type(arr).__name__
                avail.append(labs)
            raise SystemExit(
                f"No array matching --label={label!r} in {mtz_path}.\n"
                f"Available: {', '.join(avail)}"
            )
    else:
        for arr in arrays:
            if arr.is_xray_intensity_array():
                chosen = arr
                break
        if chosen is None:
            for arr in arrays:
                labels = [str(x).lower() for x in (arr.info().labels if arr.info() else [])]
                if any(lab in ("iobs", "i", "imean", "i-obs", "i_obs") for lab in labels):
                    chosen = arr
                    break
        if chosen is None:
            raise SystemExit(
                f"No X-ray intensity array found in {mtz_path}. "
                "Pass --label to select a column."
            )

    if chosen.sigmas() is None:
        raise SystemExit(
            f"Selected array has no σ(I) column "
            f"({','.join(chosen.info().labels) if chosen.info() else 'unknown'})."
        )

    labels_before = None
    if chosen.info() is not None and chosen.info().labels:
        labels_before = list(chosen.info().labels)

    if chosen.anomalous_flag():
        chosen = chosen.average_bijvoet_mates()
    chosen = chosen.map_to_asu()
    if not chosen.is_xray_intensity_array():
        chosen.set_observation_type_xray_intensity()
    if labels_before is not None and (chosen.info() is None or not chosen.info().labels):
        from cctbx.miller import array_info

        chosen = chosen.set_info(array_info(labels=labels_before))
    return chosen


def _wilson_sigma_from_intensities(i_obs: Any) -> tuple[np.ndarray, dict[str, float]]:
    """Crude Wilson Σ(s) = Σ0 exp(-0.5 B s²) for standalone (no model) reports."""
    io = np.asarray(i_obs.data(), dtype=np.float64)
    d = np.asarray(i_obs.d_spacings().data(), dtype=np.float64)
    eps = np.asarray(i_obs.epsilons().data().as_double(), dtype=np.float64)
    s2 = 1.0 / np.maximum(d * d, 1e-12)
    y = io / np.maximum(eps, 1e-12)
    pos = y > 0
    if not np.any(pos):
        return np.full(io.shape, max(float(np.median(np.abs(y))), 1.0))
    # Shell fit
    n_shells = min(10, max(2, int(pos.sum()) // 50))
    bins = np.linspace(s2[pos].min(), s2[pos].max(), n_shells + 1)
    idx = np.digitize(s2[pos], bins[:-1]) - 1
    bx, by = [], []
    for bi in range(n_shells):
        m = idx == bi
        if np.any(m):
            bx.append(float(np.mean(s2[pos][m])))
            by.append(max(float(np.mean(y[pos][m])), 1e-6))
    if len(bx) >= 2:
        slope, intercept = np.polyfit(bx, np.log(by), 1)
        sigma_0 = float(np.exp(intercept))
        b_w = float(max(-2.0 * slope, 0.0))
    else:
        sigma_0 = float(np.median(y[pos]))
        b_w = 0.0
    return sigma_0 * np.exp(-0.5 * b_w * s2), {"sigma_0": sigma_0, "b_wilson": b_w}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Resolution-ordered intensity bins with I/σ (+ Wilson data-fraction) diagnostics."
    )
    parser.add_argument("mtz", type=Path, help="Input MTZ or reflection mmCIF")
    parser.add_argument(
        "--bin-size",
        type=int,
        default=500,
        help="Reflections per bin (default: 500). Last bin may be smaller.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Substring to select intensity column (e.g. IOBS, IMEAN).",
    )
    parser.add_argument(
        "--no-wilson",
        action="store_true",
        help="Skip Wilson Σ fit (no data-fraction column).",
    )
    args = parser.parse_args(argv)

    if args.bin_size < 1:
        raise SystemExit("--bin-size must be >= 1")
    if not args.mtz.is_file():
        raise SystemExit(f"Reflection file not found: {args.mtz}")

    i_obs = _load_intensity_array(args.mtz, label=args.label)
    labels = ",".join(i_obs.info().labels) if i_obs.info() and i_obs.info().labels else "?"
    sw = None
    sw_params = None
    if not args.no_wilson:
        sw, sw_params = _wilson_sigma_from_intensities(i_obs)

    report = compute_intensity_stats_report(
        intensities=i_obs.data(),
        sigmas=i_obs.sigmas(),
        d_spacings=i_obs.d_spacings().data(),
        epsilon=i_obs.epsilons().data().as_double(),
        sigma_wilson=sw,
        sigma_wilson_params=sw_params,
        bin_size=args.bin_size,
        label=f"{args.mtz.name} | {labels}",
    )
    print_intensity_stats_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
