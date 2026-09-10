#!/usr/bin/env python
"""Export stitched omit-window map coefficients to MTZ (FWT / DELFWT).

From a diagnostic npz written with ``PHRIDGE_OMIT_SAVE_NPZ=1``::

    phenix.python scripts/export_omit_window_mtz.py mli_omit_omit_windows.npz \\
        --miller data.mtz --out omit_stitched.mtz

With ``--window N``, write that window alone (no stitch; diagnostic only).
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("npz", help="Path to *_omit_windows.npz (diagnostic dump)")
    p.add_argument("--miller", required=True, help="MTZ with crystal / miller set for symmetry")
    p.add_argument("--out", required=True, help="Output MTZ path")
    p.add_argument(
        "--window",
        type=int,
        default=None,
        help="If set, export only this window (no stitch). Default: stitch all windows.",
    )
    args = p.parse_args(argv)

    from iotbx import reflection_file_reader
    from phridge.client.intensity.omit import write_omit_window_mtz, write_stitched_omit_mtz
    from phridge.contrib.intensity_ll.omit_windows import load_omit_windows

    store = load_omit_windows(args.npz)
    miller_arrays = reflection_file_reader.any_reflection_file(args.miller).as_miller_arrays()
    if not miller_arrays:
        print("No miller arrays in --miller", file=sys.stderr)
        return 1
    tmpl = miller_arrays[0]
    if args.window is not None:
        path = write_omit_window_mtz(store, tmpl, args.window, args.out)
        print(f"Wrote {path} (single window {args.window}; columns FWT / DELFWT)")
    else:
        path = write_stitched_omit_mtz(store, tmpl, args.out)
        print(f"Wrote stitched {path} (columns FWT / DELFWT)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
