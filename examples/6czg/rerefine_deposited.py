#!/usr/bin/env python3
"""Direct Re-Refinement of Deposited 6CZG Structure (No Shaking / Perturbation).

Performs clean, unperturbed re-refinement of the deposited 6CZG structure using:
  1. Amplitude maximum likelihood target (ml_f, F)
  2. Intensity log-likelihood target (ml_i, I) with Student-t noise model (nu ~ 7)

Key Features:
  - Intact stereochemical geometry scale (w_geom = 1.0, preserving physical kcal/mol units).
  - Exact Hessian curvature ratio reweighting: w_xray = xray_scale * (Tr(H_geom) / Tr(H_xray)).
  - Hierarchical, scale-invariant Student-t ADP prior (pairwise 1-2, 1-3, sphere pairs, Wilson B anchor).
  - Empirical Bayes hyperparameter determination (tau_12, tau_sphere, lambda).
  - Preconditioned log-B Newton steps with exact Gauss-Newton curvature.
  - L-BFGS joint refinement of coordinates, scales, bulk solvent, and noise parameters on Apple Silicon MPS.
  - Rigorous held-out cross-validated likelihood evaluation on the test reflections.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
from cctbx.array_family import flex

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from phridge.client.intensity_tool import from_files
from phridge.contrib.intensity_ll.benchmark import ReflectionSplit, compare
from examples.common import print_detailed_comparison, run_rerefinement

DIR_6CZG = Path(__file__).resolve().parent
PDB_REF = DIR_6CZG / "6czg.pdb"
MTZ_FILE = DIR_6CZG / "6czg.mtz"


def main():
    parser = argparse.ArgumentParser(description="Clean re-refinement of deposited 6CZG (no shaking)")
    parser.add_argument("--pdb", default=str(PDB_REF), help="Deposited 6CZG PDB")
    parser.add_argument("--mtz", default=str(MTZ_FILE), help="Experimental 6CZG MTZ")
    parser.add_argument("--macrocycles", type=int, default=3, help="Macrocycles (default: 3)")
    parser.add_argument("--lbfgs-iter", type=int, default=15, help="L-BFGS iter/cycle (default: 15)")
    parser.add_argument("--b-iter", type=int, default=5, help="B-factor Newton iter/cycle (default: 5)")
    parser.add_argument("--weight-mode", choices=["unit", "hessian"], default="unit", help="Weighting mode: 'unit' (physical absolute scale w_xray=N_work, w_geom=1.0) or 'hessian'")
    parser.add_argument("--xray-scale", type=float, default=1.0, help="X-ray scale factor (default: 1.0 for unit, 0.1 for hessian)")
    parser.add_argument("--w-geom", type=float, default=1.0, help="Geometry weight (default: 1.0)")
    parser.add_argument("--device", default="mps", help="Device (default: mps)")
    parser.add_argument("--out-dir", default=str(DIR_6CZG / "rerefined_no_shake"), help="Output directory")
    parser.add_argument("--force-refine", action="store_true", help="Force re-running refinement even if output PDB exists")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdb_path = Path(args.pdb)
    mtz_path = Path(args.mtz)

    print("=" * 85)
    print(" 6CZG UNPERTURBED DEPOSITED STRUCTURE RE-REFINEMENT")
    print(f" PDB: {pdb_path.name} | MTZ: {mtz_path.name}")
    print(f" Macrocycles: {args.macrocycles} | L-BFGS Iter/Cycle: {args.lbfgs_iter} | B-factor Iter: {args.b_iter}")
    print(f" Weight mode: {args.weight_mode} (scale: {args.xray_scale}, w_geom: {args.w_geom}) | Device: {args.device}")
    print(f" Output Directory: {out_dir}")
    print("=" * 85)

    out_f = out_dir / "6czg_amplitude_rerefined.pdb"
    out_i = out_dir / "6czg_intensity_rerefined.pdb"

    # 1. Re-refine with Amplitude Target (F, ml_f)
    if not out_f.exists() or args.force_refine:
        res_f = run_rerefinement(
            target="amplitude",
            pdb_path=pdb_path,
            mtz_path=mtz_path,
            out_pdb=out_f,
            macrocycles=args.macrocycles,
            lbfgs_max_iter=args.lbfgs_iter,
            b_iterations=args.b_iter,
            weight_mode=args.weight_mode,
            xray_scale=args.xray_scale,
            w_geom=args.w_geom,
            device=args.device,
        )
    else:
        print(f"\n[Reusing existing refined model: {out_f.name}]")
        m_f = from_files(str(out_f), str(mtz_path), d_min=2.2, device="cpu", use_bulk_solvent=True)
        m_f.refine_scale_and_solvent()
        m_f.compute_target_and_gradients()
        s_f = m_f.summary()
        gst_f = m_f.compute_geometry_statistics()
        b_f = [sc.u_iso * 8.0 * np.pi**2 for sc in m_f.xray_structure.scatterers()]
        b_dep_arr = [sc.u_iso * 8.0 * np.pi**2 for sc in from_files(str(pdb_path), str(mtz_path), d_min=2.2, device="cpu").xray_structure.scatterers()]
        b_corr_f = float(np.corrcoef(b_f, b_dep_arr)[0, 1]) if np.std(b_f) > 0 and np.std(b_dep_arr) > 0 else 1.0
        res_f = {
            "r_work_final": s_f["r_work"],
            "r_free_final": s_f["r_free"],
            "cc_free_final": s_f["cc_free_i"],
            "nll_work_final": s_f["target_work"],
            "nll_free_final": s_f["target_test"],
            "b_mean_final": float(np.mean(b_f)),
            "b_corr_final": b_corr_f,
            "adp_energy_final": s_f.get("adp_energy", float("nan")),
            "bond_rms": gst_f["bond_rms"],
            "angle_rms": gst_f["angle_rms"],
            "planarity_rms": gst_f["planarity_rms"],
            "is_acceptable": gst_f["is_acceptable"],
        }

    # 2. Re-refine with Intensity Target (I, ml_i)
    if not out_i.exists() or args.force_refine:
        res_i = run_rerefinement(
            target="intensity",
            pdb_path=pdb_path,
            mtz_path=mtz_path,
            out_pdb=out_i,
            macrocycles=args.macrocycles,
            lbfgs_max_iter=args.lbfgs_iter,
            b_iterations=args.b_iter,
            weight_mode=args.weight_mode,
            xray_scale=args.xray_scale,
            w_geom=args.w_geom,
            device=args.device,
        )
    else:
        print(f"\n[Reusing existing refined model: {out_i.name}]")
        m_i = from_files(str(out_i), str(mtz_path), d_min=2.2, device="cpu", use_bulk_solvent=True)
        m_i.refine_scale_and_solvent()
        m_i.compute_target_and_gradients()
        s_i = m_i.summary()
        gst_i = m_i.compute_geometry_statistics()
        b_i = [sc.u_iso * 8.0 * np.pi**2 for sc in m_i.xray_structure.scatterers()]
        b_dep_arr = [sc.u_iso * 8.0 * np.pi**2 for sc in from_files(str(pdb_path), str(mtz_path), d_min=2.2, device="cpu").xray_structure.scatterers()]
        b_corr_i = float(np.corrcoef(b_i, b_dep_arr)[0, 1]) if np.std(b_i) > 0 and np.std(b_dep_arr) > 0 else 1.0
        res_i = {
            "r_work_final": s_i["r_work"],
            "r_free_final": s_i["r_free"],
            "cc_free_final": s_i["cc_free_i"],
            "nll_work_final": s_i["target_work"],
            "nll_free_final": s_i["target_test"],
            "b_mean_final": float(np.mean(b_i)),
            "b_corr_final": b_corr_i,
            "adp_energy_final": s_i.get("adp_energy", float("nan")),
            "bond_rms": gst_i["bond_rms"],
            "angle_rms": gst_i["angle_rms"],
            "planarity_rms": gst_i["planarity_rms"],
            "is_acceptable": gst_i["is_acceptable"],
        }

    # 3. Held-out Cross-Validated Benchmark Comparisons
    print("\n" + "=" * 85)
    print(" HELD-OUT CROSS-VALIDATED LIKELIHOOD BENCHMARKS")
    print("=" * 85)

    from phridge.client import Bridge
    bridge = Bridge(memory=True)

    ref_model = from_files(str(pdb_path), str(mtz_path), d_min=2.2, device="cpu", use_bulk_solvent=True)
    ref_model.refine_scale_and_solvent()
    ref_model.compute_target_and_gradients()
    split = ReflectionSplit.from_miller(ref_model.i_obs, ref_model.r_free_flags, n_shells=20, mode="cross_fit")

    comparisons = [
        ("rerefined_f_vs_deposited", "deposited_6czg", str(pdb_path), "rerefined_f", str(out_f)),
        ("rerefined_i_vs_deposited", "deposited_6czg", str(pdb_path), "rerefined_i", str(out_i)),
        ("rerefined_i_vs_rerefined_f", "rerefined_f", str(out_f), "rerefined_i", str(out_i)),
    ]

    bench_reports = {}
    for comp_id, name_a, path_a, name_b, path_b in comparisons:
        print(f"\n--- Benchmark: {name_b} vs {name_a} ---")
        mA = from_files(path_a, str(mtz_path), d_min=2.2, device="cpu", use_bulk_solvent=True)
        mA.refine_scale_and_solvent()
        fc_A = mA.i_obs.customized_copy(
            data=flex.complex_double(mA._assemble_f_model(mA.k_total, mA.k_sol, mA.b_sol).tolist())
        )

        mB = from_files(path_b, str(mtz_path), d_min=2.2, device="cpu", use_bulk_solvent=True)
        mB.refine_scale_and_solvent()
        fc_B = mB.i_obs.customized_copy(
            data=flex.complex_double(mB._assemble_f_model(mB.k_total, mB.k_sol, mB.b_sol).tolist())
        )

        rep = compare(bridge, ref_model.i_obs, {name_a: fc_A, name_b: fc_B}, split, n_boot=1000, seed=42)
        bench_reports[comp_id] = rep
        md_file = out_dir / f"{comp_id}_report.md"
        json_file = out_dir / f"{comp_id}_report.json"
        md_file.write_text(rep.to_markdown())
        json_file.write_text(rep.to_json())
        print(rep.summary())
        print_detailed_comparison(rep)

    # Summary table
    print("\n" + "=" * 125)
    print(" SUMMARY OF UNPERTURBED 6CZG RE-REFINEMENT")
    print("=" * 125)
    print(f"{'Model':<22} | {'R_work':<7} | {'R_free':<7} | {'CC_free(I)':<10} | {'B_mean':<7} | {'r(B,B0)':<7} | {'NLL(work)':<9} | {'NLL(test)':<9} | {'ADP E':<7} | {'Bonds':<7} | {'Angles':<7} | {'Planar':<7} | {'Status'}")
    print("-" * 125)

    # Evaluate all models under consistent ml_i target for work and test NLL
    m_dep_eval = from_files(str(pdb_path), str(mtz_path), d_min=2.2, device="cpu", use_bulk_solvent=True, target="intensity")
    m_dep_eval.refine_scale_and_solvent()
    m_dep_eval.compute_target_and_gradients()
    m_dep_eval.build_adp_restraints()
    s_dep = m_dep_eval.summary()
    gst_dep = m_dep_eval.compute_geometry_statistics()
    b_dep = [sc.u_iso * 8.0 * np.pi**2 for sc in m_dep_eval.xray_structure.scatterers()]
    adp_e_dep = float(m_dep_eval.adp_prior.energy_vec(
        np.log(np.maximum(np.array(b_dep, dtype=np.float64), 1e-4)),
        sites=np.asarray(m_dep_eval.xray_structure.sites_cart(), dtype=np.float64)
    ).item())

    m_f_eval = from_files(str(out_f), str(mtz_path), d_min=2.2, device="cpu", use_bulk_solvent=True, target="intensity")
    m_f_eval.refine_scale_and_solvent()
    m_f_eval.compute_target_and_gradients()
    m_f_eval.build_adp_restraints()
    s_f_eval = m_f_eval.summary()
    gst_f = m_f_eval.compute_geometry_statistics()
    b_f = [sc.u_iso * 8.0 * np.pi**2 for sc in m_f_eval.xray_structure.scatterers()]
    b_corr_f = float(np.corrcoef(b_f, b_dep)[0, 1]) if np.std(b_f) > 0 and np.std(b_dep) > 0 else 1.0
    adp_e_f = float(m_f_eval.adp_prior.energy_vec(
        np.log(np.maximum(np.array(b_f, dtype=np.float64), 1e-4)),
        sites=np.asarray(m_f_eval.xray_structure.sites_cart(), dtype=np.float64)
    ).item())

    m_i_eval = from_files(str(out_i), str(mtz_path), d_min=2.2, device="cpu", use_bulk_solvent=True, target="intensity")
    m_i_eval.refine_scale_and_solvent()
    m_i_eval.compute_target_and_gradients()
    m_i_eval.build_adp_restraints()
    s_i_eval = m_i_eval.summary()
    gst_i = m_i_eval.compute_geometry_statistics()
    b_i = [sc.u_iso * 8.0 * np.pi**2 for sc in m_i_eval.xray_structure.scatterers()]
    b_corr_i = float(np.corrcoef(b_i, b_dep)[0, 1]) if np.std(b_i) > 0 and np.std(b_dep) > 0 else 1.0
    adp_e_i = float(m_i_eval.adp_prior.energy_vec(
        np.log(np.maximum(np.array(b_i, dtype=np.float64), 1e-4)),
        sites=np.asarray(m_i_eval.xray_structure.sites_cart(), dtype=np.float64)
    ).item())

    nll_test_dep = bench_reports["rerefined_f_vs_deposited"].nll_A if "rerefined_f_vs_deposited" in bench_reports else s_dep['target_test']
    nll_test_f = bench_reports["rerefined_f_vs_deposited"].nll_B if "rerefined_f_vs_deposited" in bench_reports else s_f_eval['target_test']
    nll_test_i = bench_reports["rerefined_i_vs_deposited"].nll_B if "rerefined_i_vs_deposited" in bench_reports else s_i_eval['target_test']

    print(f"{'Deposited (6czg.pdb)':<22} | {s_dep['r_work']*100:5.2f}% | {s_dep['r_free']*100:5.2f}% | {s_dep['cc_free_i']:8.4f}   | {np.mean(b_dep):5.2f}Å² | {'1.0000':<7} | {s_dep['target_work']:9.4f} | {nll_test_dep:9.4f} | {adp_e_dep:<7.1f} | {gst_dep['bond_rms']:7.4f} | {gst_dep['angle_rms']:6.2f}° | {gst_dep['planarity_rms']:7.4f} | {'[ACCEPTABLE]' if gst_dep['is_acceptable'] else '[OUT]'}")
    print(f"{'Re-refined F (ml_f)':<22} | {res_f['r_work_final']*100:5.2f}% | {res_f['r_free_final']*100:5.2f}% | {res_f['cc_free_final']:8.4f}   | {res_f['b_mean_final']:5.2f}Å² | {b_corr_f:7.4f} | {s_f_eval['target_work']:9.4f} | {nll_test_f:9.4f} | {adp_e_f:<7.1f} | {gst_f['bond_rms']:7.4f} | {gst_f.get('angle_rms', float('nan')):6.2f}° | {gst_f['planarity_rms']:7.4f} | {'[ACCEPTABLE]' if gst_f['is_acceptable'] else '[OUT]'}")
    print(f"{'Re-refined I (ml_i)':<22} | {res_i['r_work_final']*100:5.2f}% | {res_i['r_free_final']*100:5.2f}% | {res_i['cc_free_final']:8.4f}   | {res_i['b_mean_final']:5.2f}Å² | {b_corr_i:7.4f} | {s_i_eval['target_work']:9.4f} | {nll_test_i:9.4f} | {adp_e_i:<7.1f} | {gst_i['bond_rms']:7.4f} | {gst_i.get('angle_rms', float('nan')):6.2f}° | {gst_i['planarity_rms']:7.4f} | {'[ACCEPTABLE]' if gst_i['is_acceptable'] else '[OUT]'}")
    print("=" * 125)


if __name__ == "__main__":
    main()
