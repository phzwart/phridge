#!/usr/bin/env python3
"""End-to-end demonstration of the phridge intensity refinement and map generation tool.

Demonstrates:
  1. Reading macromolecular PDB and MTZ reflection files (intensities).
  2. Setup of resolution bins and normalization factors Sigma_wilson = <I/eps>_bin.
  3. Real-space solvent mask construction via cctbx map gridding (mmtbx.masks.manager).
  4. Scale factor (k_total) and bulk solvent (k_sol, B_sol) refinement directly
     minimizing the phridge ml_i negative log-likelihood (NLL).
  5. Direct sigma_A estimation per resolution bin by minimizing ml_i NLL on the
     test set (free reflections), without conversion to amplitudes.
  6. Parameter gradients (dQ/dF_model -> dQ/dF_calc -> atomic site cartesian gradients).
  7. Optional coordinate refinement steps showing monotonic target decrease.
  8. Synthesis and export of reciprocal-space gradient map (-0.5 * dQ/dF_model) and
     sigma_A-weighted 2mFo-DFc and mFo-DFc maps to MTZ and CCP4 formats.

Run::

    python examples/intensity_tool_demo.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Import cctbx before torch for MKL / OpenMP safety
import cctbx  # noqa: F401
from cctbx import sgtbx
from cctbx.array_family import flex
from cctbx.development import random_structure
import iotbx.ccp4_map
import iotbx.pdb
from iotbx.reflection_file_reader import any_reflection_file
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phridge.client.intensity_tool import IntensityModel, from_files, run_intensity_pipeline

DEMO_DIR = ROOT / "examples" / "intensity_demo_output"
DEMO_DIR.mkdir(parents=True, exist_ok=True)
DEMO_MD = ROOT / "examples" / "intensity_tool_demo.md"


def generate_sample_data(
    pdb_path: Path,
    mtz_path: Path,
    space_group: str = "P21",
    n_residues: int = 25,
    d_min: float = 2.0,
    seed: int = 12345,
):
    """Generate realistic synthetic model and intensity reflection file."""
    flex.set_random_seed(seed)
    np.random.seed(seed)

    elements = ["C", "N", "O", "C", "C", "O"] * n_residues
    xs = random_structure.xray_structure(
        space_group_info=sgtbx.space_group_info(space_group),
        elements=elements,
        volume_per_atom=45.0,
        random_u_iso=True,
    )

    # Calculate base structure factors
    fc = xs.structure_factors(d_min=d_min).f_calc()

    # Add realistic bulk solvent effect and scaling to observed intensities
    s2 = np.asarray(fc.d_star_sq().data(), dtype=np.float64)
    # Simulate true scale = 2.0, k_sol = 0.35, B_sol = 45.0
    true_k = 2.0
    fc_complex = np.asarray(fc.data(), dtype=np.complex128)

    # Compute solvent mask on ground truth
    from mmtbx.masks import manager as mask_manager
    mm = mask_manager(miller_array=fc, xray_structure=xs)
    fm = np.asarray(mm.shell_f_masks()[0].data(), dtype=np.complex128)

    f_true = true_k * (fc_complex + 0.35 * np.exp(-45.0 * s2 / 4.0) * fm)
    i_true = np.abs(f_true) ** 2

    # Add 5% observational counting noise
    sigmas_np = np.maximum(0.05 * i_true, 1.0)
    noise = np.random.normal(0.0, sigmas_np)
    i_obs_np = np.maximum(i_true + noise, 0.5)

    i_obs = fc.customized_copy(
        data=flex.double(i_obs_np.tolist()),
        sigmas=flex.double(sigmas_np.tolist()),
    )
    i_obs.set_observation_type_xray_intensity()

    # Generate 10% test set for R-free / cross-validation
    r_free = i_obs.generate_r_free_flags(fraction=0.10)

    # Write input PDB
    pdb_path.write_text(xs.as_pdb_file())

    # Write input MTZ
    mtz_ds = i_obs.as_mtz_dataset(column_root_label="IOBS")
    mtz_ds.add_miller_array(r_free, column_root_label="FreeR_flag")
    mtz_ds.mtz_object().write(str(mtz_path))

    return xs, i_obs, r_free


def run_demo() -> int:
    print("=" * 70)
    print(" phridge Intensity Refinement & Map Generation Demo")
    print("=" * 70)

    pdb_file = DEMO_DIR / "demo_model.pdb"
    mtz_file = DEMO_DIR / "demo_data.mtz"
    out_prefix = str(DEMO_DIR / "phridge_demo")

    print(f"\n1. Generating synthetic macromolecular test dataset ({pdb_file.name}, {mtz_file.name})...")
    xs, i_obs, r_free = generate_sample_data(pdb_file, mtz_file, n_residues=30, d_min=2.0)
    print(f"   Space Group: {xs.space_group_info().type().lookup_symbol()}")
    print(f"   Unit Cell:   {xs.unit_cell()}")
    print(f"   Atoms:       {xs.scatterers().size()}")
    print(f"   Reflections: {i_obs.size()} (resolution: {i_obs.d_max_min()[0]:.2f} - {i_obs.d_max_min()[1]:.2f} Å)")

    print("\n2. Initializing IntensityModel with in-memory phridge bridge...")
    model = from_files(
        pdb_path=pdb_file,
        mtz_path=mtz_file,
        use_bulk_solvent=True,
        n_bins=8,
    )

    print("\n3. Refining scale and bulk solvent parameters directly against ml_i intensity NLL...")
    scales = model.refine_scale_and_solvent(max_iter=40)
    print(f"   Fitted overall scale k_total: {scales['k_total']:.4f}")
    print(f"   Fitted solvent k_sol:         {scales['k_sol']:.4f}")
    print(f"   Fitted solvent B_sol:         {scales['b_sol']:.2f} Å²")

    print("\n4. Estimating sigma_A directly from intensities per resolution bin...")
    sa_bins = model.estimate_sigma_a()
    bin_table_lines = []
    for b_idx in sorted(sa_bins.keys()):
        d_hi, d_lo = model.binner.bin_d_range(b_idx)
        sa_val = sa_bins[b_idx]
        mean_sw = float(flex.mean(model.mean_i_per_refl.select(model.binner.selection(b_idx))))
        print(f"   Bin {b_idx:2d} ({d_hi:5.2f} - {d_lo:5.2f} Å):  sigma_A = {sa_val:.4f}  <Sigma> = {mean_sw:8.1f}")
        bin_table_lines.append(f"| {b_idx} | {d_hi:.2f} - {d_lo:.2f} | {sa_val:.4f} | {mean_sw:.1f} |")

    print("\n5. Evaluating ml_i target and calculating scatterer gradients...")
    target, grads = model.compute_target_and_gradients()
    print(f"   Target NLL: {target:.4f}")
    g_cart = np.asarray(grads.d_target_d_site_cart(), dtype=np.float64)
    rms_cart = np.sqrt(np.mean(np.sum(g_cart**2, axis=1)))
    print(f"   Cartesian Site Gradient RMS: {rms_cart:.4e} (max: {np.max(np.abs(g_cart)):.4e})")

    print("\n6. Running 3 coordinate refinement steps...")
    history = model.refine_coordinates(max_iterations=3, step_scale=0.0005)
    for step_i, val in enumerate(history):
        print(f"   Iteration {step_i}: Target NLL = {val:.4f}")
    refined_pdb = f"{out_prefix}_refined.pdb"
    model.write_pdb(refined_pdb)
    print(f"   Wrote refined coordinates to {refined_pdb}")

    print("\n7. Synthesizing reciprocal-space gradient map and sigma_A-weighted maps...")
    files = model.write_maps(prefix=out_prefix, resolution_factor=0.33)
    for k, v in files.items():
        size_kb = os.path.getsize(v) / 1024.0
        print(f"   Wrote {k:10s} map: {v} ({size_kb:.1f} KB)")

    print("\n8. Validating exported files...")
    # Read MTZ
    reader = any_reflection_file(files["mtz"])
    cols = []
    for a in reader.as_miller_arrays():
        if a.info() and a.info().labels:
            cols.extend(a.info().labels)
    assert "FGRAD" in cols and "2FOFCWT" in cols and "FOFCWT" in cols
    print("   ✓ MTZ validated: contains FGRAD, 2FOFCWT, FOFCWT")

    # Read CCP4 map
    grad_map_reader = iotbx.ccp4_map.map_reader(file_name=files["gradient"])
    assert grad_map_reader.unit_cell() is not None
    print(f"   ✓ Gradient CCP4 map validated: grid dimensions {grad_map_reader.data.all()}")

    summary = model.summary()
    model.print_summary()

    # Generate Markdown documentation
    md_content = f"""# Intensity Refinement & Map Generation Demo

Generated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}

## Executive Summary
This demo exercises the end-to-end integration between `cctbx` and `phridge` for intensity-based crystallographic refinement.

- **Data Alignment**: Ingested PDB coordinates and MTZ observed intensities.
- **Bulk Solvent via Map Gridding**: Real-space mask computed via `mmtbx.masks.manager` and FFT transformed to $F_{{mask}}$.
- **Direct Parameter Estimation**: Scaled model ($k_{{total}} = {scales['k_total']:.4f}, k_{{sol}} = {scales['k_sol']:.3f}, B_{{sol}} = {scales['b_sol']:.1f}\\text{{ Å}}^2$) directly minimizing the phridge `ml_i` intensity negative log-likelihood.
- **Direct $\\sigma_A$ Estimation**: Estimated $\\sigma_A$ across {len(sa_bins)} resolution shells by 1D bounded minimization of intensity NLL on the test set.
- **Gradient Maps**: Computed $F_{{grad}} = -\\frac{{1}}{{2}} \\frac{{dQ}}{{dF_{{model}}}}$ and Fourier transformed into real-space CCP4 format.
- **Weighted Difference Maps**: Synthesized $2mF_o - DF_c$ and $mF_o - DF_c$ maps with figure of merit $m$ from the Rice/Woolfson conditional distribution.

## Resolution Binning & $\\sigma_A$ Distribution

| Bin | Resolution Range (Å) | $\\sigma_A$ | $\\langle \\Sigma \\rangle$ |
| --- | ------------------- | ---------- | -------------------------- |
""" + "\n".join(bin_table_lines) + f"""

## Refinement Progression

| Iteration | Target NLL |
| --------- | ---------- |
""" + "\n".join(f"| {it} | {val:.4f} |" for it, val in enumerate(history)) + f"""

## Generated Artifacts

| Type | Path |
| ---- | ---- |
| MTZ Fourier Coefficients | `{files['mtz']}` |
| Gradient CCP4 Map | `{files['gradient']}` |
| $2mF_o - DF_c$ CCP4 Map | `{files['2fofc']}` |
| $mF_o - DF_c$ CCP4 Map | `{files['fofc']}` |
| Refined PDB Model | `{refined_pdb}` |

All checks passed successfully!
"""
    DEMO_MD.write_text(md_content)
    print(f"\nWrote demonstration report to {DEMO_MD}")
    print("\n✓ Demo completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_demo())
