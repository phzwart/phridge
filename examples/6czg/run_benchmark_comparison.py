"""Run held-out cross-validated likelihood benchmark comparisons on 6CZG models.

Compares:
  1. Shaken (0.50 Å) vs Recovered (0.50 Å refined)
  2. Shaken (0.10 Å) vs Recovered (0.10 Å refined)
  3. Deposited (PDB 6CZG) vs Re-refined from deposited coordinates
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from cctbx.array_family import flex

from phridge.client import Bridge
from phridge.client.intensity_tool import from_files
from phridge.contrib.intensity_ll.benchmark import ReflectionSplit, compare


def main():
    parser = argparse.ArgumentParser(description="Cross-validated held-out likelihood benchmark on 6CZG")
    parser.add_argument("--mtz", default="examples/6czg/6czg.mtz", help="Experimental intensities MTZ")
    parser.add_argument("--pdb-ref", default="examples/6czg/6czg.pdb", help="Deposited reference PDB")
    parser.add_argument("--n-boot", type=int, default=1000, help="Number of bootstrap replicates")
    parser.add_argument("--out-dir", default="examples/6czg/benchmark_results", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bridge = Bridge(memory=True)

    print("=" * 80)
    print(" HELD-OUT CROSS-VALIDATED LIKELIHOOD BENCHMARK (6CZG)")
    print("=" * 80)

    # Base reference model and split
    ref_model = from_files(args.pdb_ref, args.mtz, d_min=2.2, device="cpu", use_bulk_solvent=True)
    ref_model.refine_scale_and_solvent()
    fc_dep = ref_model.i_obs.customized_copy(
        data=flex.complex_double(ref_model._assemble_f_model(ref_model.k_total, ref_model.k_sol, ref_model.b_sol).tolist())
    )

    split = ReflectionSplit.from_miller(ref_model.i_obs, ref_model.r_free_flags, n_shells=20, mode="cross_fit")
    print(f"Total reflections: {len(split.work)} | Free reflections: {int(split.test.sum())} (scored once via cross-fit)")

    comparisons = [
        (
            "0.50A_recovery",
            "shaken_0.50A",
            "examples/6czg/perturbed_models/6czg_rmsd_0.50_rep_00.pdb",
            "recovered_0.50A",
            "examples/6czg/shake_recover_results/6czg_intensity_lbfgs_rmsd_0.50_scale_0.1_rep_00_refined.pdb",
        ),
        (
            "0.10A_recovery",
            "shaken_0.10A",
            "examples/6czg/perturbed_models/6czg_rmsd_0.10_rep_00.pdb",
            "recovered_0.10A",
            "examples/6czg/shake_recover_results/6czg_intensity_lbfgs_rmsd_0.10_scale_0.1_rep_00_refined.pdb",
        ),
        (
            "deposited_vs_rerefined",
            "deposited_6czg",
            args.pdb_ref,
            "rerefined_deposited_i",
            "examples/6czg/shake_recover_results/6czg_intensity_lbfgs_deposited_scale_0.1_refined.pdb",
        ),
        (
            "deposited_vs_f_rerefined",
            "deposited_6czg",
            args.pdb_ref,
            "rerefined_deposited_f",
            "examples/6czg/shake_recover_results/6czg_amplitude_lbfgs_deposited_scale_0.1_refined.pdb",
        ),
        (
            "f_vs_i_rerefined",
            "rerefined_deposited_f",
            "examples/6czg/shake_recover_results/6czg_amplitude_lbfgs_deposited_scale_0.1_refined.pdb",
            "rerefined_deposited_i",
            "examples/6czg/shake_recover_results/6czg_intensity_lbfgs_deposited_scale_0.1_refined.pdb",
        ),
    ]

    reports = {}

    for comp_id, name_A, pdb_A, name_B, pdb_B in comparisons:
        print("\n" + "-" * 80)
        print(f" Running Comparison: {name_B} vs {name_A}")
        print(f"   Model A: {pdb_A}")
        print(f"   Model B: {pdb_B}")
        print("-" * 80)

        mA = from_files(pdb_A, args.mtz, d_min=2.2, device="cpu", use_bulk_solvent=True)
        mA.refine_scale_and_solvent()
        fc_A = mA.i_obs.customized_copy(
            data=flex.complex_double(mA._assemble_f_model(mA.k_total, mA.k_sol, mA.b_sol).tolist())
        )

        mB = from_files(pdb_B, args.mtz, d_min=2.2, device="cpu", use_bulk_solvent=True)
        mB.refine_scale_and_solvent()
        fc_B = mB.i_obs.customized_copy(
            data=flex.complex_double(mB._assemble_f_model(mB.k_total, mB.k_sol, mB.b_sol).tolist())
        )

        rep = compare(bridge, ref_model.i_obs, {name_A: fc_A, name_B: fc_B}, split, n_boot=args.n_boot)
        reports[comp_id] = rep

        print(rep.summary())

        # Save markdown and JSON reports
        md_file = out_dir / f"{comp_id}_report.md"
        json_file = out_dir / f"{comp_id}_report.json"
        md_file.write_text(rep.to_markdown())
        json_file.write_text(rep.to_json())
        print(f"   Saved report: {md_file}")

    print("\n" + "=" * 80)
    print(" ALL 6CZG BENCHMARK COMPARISONS COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
