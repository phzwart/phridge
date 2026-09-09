#!/usr/bin/env phenix.python
"""Standalone driver to run phenix.refine with Phridge intensity-based likelihood (mli_quad).

Usage:
    phenix.python scripts/phenix_refine_mli.py model.pdb data.mtz [options]

Example:
    phenix.python scripts/phenix_refine_mli.py 6czg.pdb 6czg.mtz \
        refinement.main.number_of_macro_cycles=3

Notes:
    - Automatically activates the mli_quad intensity likelihood hook in phenix/mmtbx.
    - Sets refinement.main.target=mli_quad by default if not specified.
    - Twinning is strictly NOT supported; specifying twin_law or twin=True will raise a hard error.
"""

from __future__ import annotations

import os
import sys

# Ensure repository root is on sys.path if running in phenix.python
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(repo_root, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Default: in-process torch (no Redis). Past jobs hung on Redis socket reads.
os.environ.setdefault("PHRIDGE_MEMORY", "1")
os.environ.setdefault("PHRIDGE_HEARTBEAT", "1")
os.environ.setdefault("PHRIDGE_HEARTBEAT_INTERVAL", "30")

from phridge.client.intensity import (
    IntensityTwinningError,
    enable_intensity_in_phenix,
    is_intensity_enabled,
)


def run(args: list[str]) -> int:
    print("=" * 76)
    print(" Phridge Direct Intensity-Based Likelihood Refinement (`mli_quad`)")
    print("=" * 76)

    # Pre-check command line arguments for twinning
    for arg in args:
        lower_arg = arg.strip().lower()
        if "twin_law" in lower_arg or lower_arg in ("twin=true", "twinning=true", "--twin"):
            if "=" in lower_arg:
                val = lower_arg.split("=", 1)[1].strip().strip('"').strip("'")
                if val not in ("none", "false", ""):
                    raise IntensityTwinningError(
                        f"Twinning is not supported for intensity-based likelihood ('mli_quad') refinement: "
                        f"found '{arg}' on command line."
                    )
            elif lower_arg in ("twin=true", "twinning=true", "--twin"):
                raise IntensityTwinningError(
                    f"Twinning is not supported for intensity-based likelihood ('mli_quad') refinement: "
                    f"found '{arg}' on command line."
                )

    # 1. Activate the mli_quad hooks in phenix/mmtbx
    enable_intensity_in_phenix()
    print("✓ Registered `mli_quad` target in mmtbx and phenix.refine master params.")
    print("✓ Patched fmodel_manager2 and target_functor for direct intensity calculation.")
    redis_url = os.environ.get("PHRIDGE_REDIS_URL", "redis://localhost:6379/0")
    print(f"✓ PyTorch worker communication bridge: {redis_url}")

    # Process custom phridge flags (nu, fit_nu, precondition)
    filtered_args = []
    for arg in args:
        lower = arg.strip().lower()
        if lower.startswith("--nu=") or (lower.startswith("nu=") and not lower.startswith("number_")):
            val = arg.split("=", 1)[1].strip()
            os.environ["PHRIDGE_NU"] = val
            print(f"✓ Student-t degrees of freedom set: nu={val}")
        elif lower in ("--fit-nu", "fit_nu=true", "fit_nu=1"):
            os.environ["PHRIDGE_FIT_NU"] = "1"
            print("✓ Student-t nu refinement enabled (fit_nu=True)")
        elif lower in ("--no-fit-nu", "fit_nu=false", "fit_nu=0"):
            os.environ["PHRIDGE_FIT_NU"] = "0"
            print("✓ Student-t nu refinement disabled (fit_nu=False)")
        elif lower in ("--precondition", "precondition=true", "precondition=1"):
            os.environ["PHRIDGE_PRECONDITION"] = "1"
            print("✓ Gauss-Newton diagonal preconditioning enabled for XYZ, occupancy, and ADP gradients")
        elif lower in ("--no-precondition", "precondition=false", "precondition=0"):
            os.environ["PHRIDGE_PRECONDITION"] = "0"
            print("✓ Gauss-Newton XYZ/occ/ADP preconditioning disabled (precondition=False)")
        elif lower in ("--stats-report", "stats_report=true", "stats_report=1"):
            os.environ["PHRIDGE_STATS_REPORT"] = "1"
            print("✓ Resolution I/σ + σ_A stats report enabled (after each update_all_scales)")
        elif lower in ("--no-stats-report", "stats_report=false", "stats_report=0"):
            os.environ["PHRIDGE_STATS_REPORT"] = "0"
            print("✓ Resolution I/σ + σ_A stats report disabled")
        elif lower.startswith("--stats-bin-size=") or lower.startswith("stats_bin_size="):
            val = arg.split("=", 1)[1].strip()
            os.environ["PHRIDGE_STATS_BIN_SIZE"] = val
            print(f"✓ Stats report bin size: {val} reflections")
        elif lower.startswith("--sigma-a-mode=") or lower.startswith("sigma_a_mode="):
            val = arg.split("=", 1)[1].strip().lower()
            os.environ["PHRIDGE_SIGMA_A_MODE"] = val
            print(f"✓ σ_A(s) fit mode: {val} (bins|read)")
        elif lower.startswith("--sigma-a-bins=") or lower.startswith("sigma_a_bins="):
            val = arg.split("=", 1)[1].strip()
            os.environ["PHRIDGE_SIGMA_A_BINS"] = val
            print(f"✓ σ_A resolution shells: {val}")
        elif lower in ("--memory", "memory=true", "memory=1"):
            os.environ["PHRIDGE_MEMORY"] = "1"
            os.environ.pop("PHRIDGE_REDIS_URL", None)
            print("✓ In-process Bridge (PHRIDGE_MEMORY=1) — no Redis")
        elif lower in ("--redis", "memory=false", "memory=0"):
            os.environ["PHRIDGE_MEMORY"] = "0"
            print("✓ Redis Bridge mode (PHRIDGE_MEMORY=0)")
        elif lower in ("--verbose-target", "verbose_target=true", "verbose_target=1"):
            os.environ["PHRIDGE_VERBOSE_TARGET"] = "1"
            print("✓ Verbose mli_quad target banners enabled (PHRIDGE_VERBOSE_TARGET=1)")
        elif lower in ("--quiet-target", "verbose_target=false", "verbose_target=0"):
            os.environ["PHRIDGE_VERBOSE_TARGET"] = "0"
            print("✓ Verbose mli_quad target banners disabled")
        elif lower.startswith("--weight-metric=") or lower.startswith("weight_metric="):
            val = arg.split("=", 1)[1].strip().lower()
            os.environ["PHRIDGE_WEIGHT_METRIC"] = val
            print(f"✓ Weight trial metric: {val} (PHRIDGE_WEIGHT_METRIC)")
        else:
            filtered_args.append(arg)

    if os.environ.get("PHRIDGE_MEMORY", "1").strip().lower() in ("1", "true", "yes", "on", ""):
        os.environ.pop("PHRIDGE_REDIS_URL", None)
        print("✓ Bridge: in-process memory (no Redis sockets)")
        try:
            import torch  # noqa: F401
            print(f"✓ torch available for in-process worker ({torch.__version__})")
        except ImportError:
            print(
                "\nERROR: PHRIDGE_MEMORY=1 but torch is not importable in phenix.python.\n"
                f"  interpreter: {sys.executable}\n\n"
                "Install torch into Phenix, or use Redis:\n"
                "  phenix.python -m pip install torch\n"
                "  ./run_phenix_intensity.sh --redis model.pdb data.mtz ...\n",
                file=sys.stderr,
            )
            return 2
    else:
        print(f"✓ Bridge: Redis ({os.environ.get('PHRIDGE_REDIS_URL', 'redis://127.0.0.1:6379/0')})")
    print(f"✓ Heartbeat every {os.environ.get('PHRIDGE_HEARTBEAT_INTERVAL', '30')}s")
    metric = os.environ.get("PHRIDGE_WEIGHT_METRIC", "nll")
    print(f"✓ XYZ/ADP weight selection metric: {metric} (nll|rfree)")
    if os.environ.get("PHRIDGE_VERBOSE_TARGET", "0").strip() in ("1", "true", "yes", "on"):
        print("✓ Per-eval mli_quad target banners: ON")
    else:
        print("· Per-eval mli_quad target banners: off (set PHRIDGE_VERBOSE_TARGET=1 to enable)")
    stats_on = os.environ.get("PHRIDGE_STATS_REPORT", "1").strip().lower() in ("1", "true", "yes", "on", "")
    if stats_on:
        bin_sz = os.environ.get("PHRIDGE_STATS_BIN_SIZE", "500")
        print(f"✓ Resolution I/σ + σ_A stats report: ON (bin_size={bin_sz})")
    else:
        print("· Resolution I/σ + σ_A stats report: off")

    # 2. Check if refinement.main.target is specified; if not, default to mli_quad
    has_target = any("refinement.main.target" in a.lower() for a in filtered_args)
    effective_args = list(filtered_args)
    if not has_target:
        print("✓ Setting default target: refinement.main.target=mli_quad")
        effective_args.append("refinement.main.target=mli_quad")

    # Disable French–Wilson / F² so negative intensities are retained for mli_quad.
    has_fw = any("french_wilson_scale" in a.lower() for a in effective_args)
    if not has_fw:
        effective_args.append("xray_data.french_wilson_scale=False")
        print("✓ Disabled French–Wilson (xray_data.french_wilson_scale=False)")
    else:
        # Still force False even if user tried to enable it
        effective_args = [
            a for a in effective_args
            if "french_wilson_scale" not in a.lower()
        ]
        effective_args.append("xray_data.french_wilson_scale=False")
        print("✓ Forced French–Wilson off (mli_quad requires genuine I_obs)")

    # 3. Dispatch to phenix.refine
    try:
        from phenix.command_line import phenix_refine
        return phenix_refine.run_phenix_refine(args=effective_args)
    except ImportError:
        try:
            from phenix.command_line import refine
            return refine.run(effective_args)
        except ImportError:
            print(
                "\nError: `phenix.refine` is not importable. Please execute this script using `phenix.python`:\n"
                "    phenix.python scripts/phenix_refine_mli.py <model.pdb> <data.mtz> ...\n",
                file=sys.stderr,
            )
            return 1


def main() -> None:
    sys.exit(run(sys.argv[1:]))


if __name__ == "__main__":
    main()
