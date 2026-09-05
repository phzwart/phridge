#!/usr/bin/env python3
"""Download HEWL lysozyme (1IEE) from the PDB with deposited intensities and run the intensity refinement tool.

PDB entry 1IEE:
  - Hen egg white lysozyme (tetragonal, space group P 43 21 2)
  - Deposited experimental intensities: _refln.intensity_meas / _refln.intensity_sigma
  - Deposited cross-validation test set: _refln.status == 'f' (Free R flags)
  - Resolution: 0.94 Å

This script:
  1. Downloads 1iee.pdb and 1iee-sf.cif from RCSB (if not already downloaded).
  2. Converts reflection data into a standard MTZ file (1iee.mtz) with IOBS and FreeR_flag.
  3. Executes the phridge intensity pipeline (scale & bulk solvent refinement, direct sigma_A
     estimation, target evaluation, gradient calculation, optional coordinate refinement,
     and map export in MTZ and CCP4 formats).

Usage::

    python examples/lysozyme/run_lysozyme.py [--d-min 1.5] [--refine 1] [--bulk-solvent]
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

# Import cctbx before torch for MKL / OpenMP safety
import cctbx  # noqa: F401
from cctbx.array_family import flex
import iotbx.pdb
from iotbx.reflection_file_reader import any_reflection_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from phridge.client.intensity_tool import run_intensity_pipeline

LYSOZYME_DIR = Path(__file__).resolve().parent
PDB_FILE = LYSOZYME_DIR / "1iee.pdb"
CIF_FILE = LYSOZYME_DIR / "1iee-sf.cif"
MTZ_FILE = LYSOZYME_DIR / "1iee.mtz"


def download_file(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  Using existing {dest.name} ({dest.stat().st_size / 1024:.1f} KB)")
        return dest
    print(f"  Downloading {url} -> {dest.name}...")
    req = urllib.request.Request(url, headers={"User-Agent": "phridge/0.1.0"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        f.write(resp.read())
    print(f"  Downloaded {dest.name} ({dest.stat().st_size / 1024:.1f} KB)")
    return dest


def prepare_data(force_download: bool = False) -> tuple[Path, Path]:
    print("\n--- 1. Fetching Lysozyme Data from RCSB PDB ---")
    if force_download:
        if PDB_FILE.exists():
            PDB_FILE.unlink()
        if CIF_FILE.exists():
            CIF_FILE.unlink()
        if MTZ_FILE.exists():
            MTZ_FILE.unlink()

    download_file("https://files.rcsb.org/download/1iee.pdb", PDB_FILE)
    download_file("https://files.rcsb.org/download/1iee-sf.cif", CIF_FILE)

    if not MTZ_FILE.exists() or MTZ_FILE.stat().st_size == 0:
        print(f"\n--- 2. Converting {CIF_FILE.name} to {MTZ_FILE.name} ---")
        reader = any_reflection_file(str(CIF_FILE))
        arrays = reader.as_miller_arrays()

        iobs = None
        status = None
        for a in arrays:
            labels = a.info().labels if a.info() else []
            if a.is_xray_intensity_array():
                iobs = a
            elif any("status" in l.lower() for l in labels) and a.is_string_array():
                status = a

        if iobs is None:
            raise ValueError(f"No intensity array found in {CIF_FILE}")

        # Map to ASU
        iobs = iobs.map_to_asu()
        iobs.set_observation_type_xray_intensity()

        # Build Free R flag
        if status is not None:
            status = status.map_to_asu()
            r_free_data = (status.data() == "f") | (status.data() == "F")
            r_free = iobs.customized_copy(data=r_free_data, sigmas=None)
        else:
            r_free = iobs.generate_r_free_flags(fraction=0.05)

        # Write standard MTZ
        mtz_ds = iobs.as_mtz_dataset(column_root_label="IOBS")
        mtz_ds.add_miller_array(r_free, column_root_label="FreeR_flag")
        mtz_ds.mtz_object().write(str(MTZ_FILE))
        print(f"  Wrote {MTZ_FILE.name} with {iobs.size()} reflections (Free R count: {r_free.data().count(True)})")
    else:
        print(f"  Using existing {MTZ_FILE.name}")

    return PDB_FILE, MTZ_FILE


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run phridge intensity refinement tool on Lysozyme 1IEE from the PDB",
    )
    parser.add_argument("--d-min", type=float, default=1.5, help="High-resolution cutoff in Angstroms (default: 1.5)")
    parser.add_argument("--n-bins", type=int, default=10, help="Number of resolution bins (default: 10)")
    parser.add_argument("--overlap-bins", type=int, default=1, help="Adjacent resolution bins for overlapping sigmaA estimation (default: 1)")
    parser.add_argument("--tv-norm", type=float, default=0.04, help="Total variation regularization weight lambda_TV (default: 0.04)")
    parser.add_argument("--enforce-monotonic", action="store_true", default=False, help="Enforce monotonic non-increasing sigmaA across resolution")
    parser.add_argument("--nu", type=float, default=None, help="Fixed degrees of freedom for Student-t noise (e.g. 4.0; None uses Normal noise unless --estimate-nu is set)")
    parser.add_argument("--estimate-nu", action="store_true", default=False, help="Estimate / refine Student-t degrees of freedom nu directly from intensities")
    parser.add_argument("--nu-mode", default="binned", choices=["binned", "global"], help="Nu estimation mode: 'binned' (per resolution shell) or 'global' (single scalar, default: binned)")
    parser.add_argument("--bulk-solvent", action="store_true", default=True, help="Enable cctbx map-gridded bulk solvent")
    parser.add_argument("--no-bulk-solvent", action="store_false", dest="bulk_solvent", help="Disable bulk solvent")
    parser.add_argument("--refine", type=int, default=1, dest="max_iterations", help="Number of coordinate refinement steps (default: 1)")
    parser.add_argument("--prefix", default=None, help="Output prefix for maps and logs (default: 1iee_omit_<mode> or 1iee_out)")
    parser.add_argument("--omit", default="45:52", help="Residues or atom selection to omit (default: '45:52', pass 'none' to disable)")
    parser.add_argument("--omit-mode", default="delete", choices=["delete", "zero_occ"], help="Omit treatment: 'delete' (removes atoms) or 'zero_occ' (sets occupancy to 0.0, default: delete)")
    parser.add_argument("--force-download", action="store_true", help="Force redownload of PDB and CIF files")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Compute device (default: cpu)")
    parser.add_argument("--view", action="store_true", help="Launch browser-based 3D viewer (Mol*) to inspect maps and model")

    opts = parser.parse_args(args)

    omit_arg = None if opts.omit and opts.omit.lower() == "none" else opts.omit
    if opts.prefix is not None:
        prefix = opts.prefix
    elif omit_arg:
        prefix = str(LYSOZYME_DIR / f"1iee_omit_{opts.omit_mode}")
    else:
        prefix = str(LYSOZYME_DIR / "1iee_out")

    # 1. Download & prep
    pdb_path, mtz_path = prepare_data(force_download=opts.force_download)

    # 2. Run pipeline
    omit_desc = f"omit={omit_arg} ({opts.omit_mode})" if omit_arg else "no omit"
    print(f"\n--- 3. Running Intensity Pipeline (d_min={opts.d_min} Å, bulk_solvent={opts.bulk_solvent}, {omit_desc}) ---")
    result = run_intensity_pipeline(
        pdb_path=pdb_path,
        mtz_path=mtz_path,
        prefix=prefix,
        intensity_label="IOBS",
        use_bulk_solvent=opts.bulk_solvent,
        n_bins=opts.n_bins,
        d_min=opts.d_min,
        overlap_bins=opts.overlap_bins,
        tv_norm=opts.tv_norm,
        enforce_monotonic=opts.enforce_monotonic,
        nu=opts.nu,
        estimate_nu=opts.estimate_nu,
        nu_mode=opts.nu_mode,
        max_iterations=opts.max_iterations,
        device=opts.device,
        view=opts.view,
        omit=omit_arg,
        omit_mode=opts.omit_mode,
    )

    print("\n--- 4. Run Completed Successfully ---")
    print("Generated output files:")
    for file_type, file_path in result["files"].items():
        print(f"  - {file_type.upper()}: {file_path}")
    if opts.max_iterations > 0:
        print(f"  - REFINED_PDB: {prefix}_refined.pdb")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
