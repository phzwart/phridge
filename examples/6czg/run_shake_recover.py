#!/usr/bin/env python3
"""Shake-and-Recover Refinement Benchmark on 0.50 Å Perturbed 6CZG Models.

Refines perturbed models using the Intensity (I) or Amplitude (F) target while:
  1. Leaving the stereochemical geometry scale intact (w_geom = 1.0, preserving physical kcal/mol units).
  2. Reweighting the X-ray likelihood term based on the ratio of geometry to X-ray Hessians:
         w_xray = xray_scale * (Tr(H_geom) / Tr(H_xray))
  3. Preconditioning site coordinates with the balanced total Hessian:
         H_total = w_xray * H_xray + H_geom
  4. Evaluating coordinate recovery against the ground truth reference structure on MPS.
  5. Building posterior and gradient electron density maps (MTZ and CCP4).

Usage:
    python examples/6czg/run_shake_recover.py --target intensity --device mps --n-reps 2 --steps 40 --lr-sites 0.05
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from phridge.client.intensity_tool import from_files

DIR_6CZG = Path(__file__).resolve().parent
PDB_REF = DIR_6CZG / "6czg.pdb"
MTZ_FILE = DIR_6CZG / "6czg.mtz"
MODELS_DIR = DIR_6CZG / "perturbed_models"


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Shake-and-recover refinement using Hessian-based X-ray reweighting and map building."
    )
    parser.add_argument("--pdb-ref", default=str(PDB_REF), help="Reference ground-truth PDB")
    parser.add_argument("--mtz", default=str(MTZ_FILE), help="Experimental MTZ file")
    parser.add_argument("--models-dir", default=str(MODELS_DIR), help="Directory of perturbed models")
    parser.add_argument("--rmsd", type=float, default=0.50, help="Perturbation RMSD to refine (default: 0.50 Å)")
    parser.add_argument("--n-reps", type=int, default=2, help="Number of replicate models to test (default: 2)")
    parser.add_argument(
        "--target",
        choices=["intensity", "amplitude"],
        default="intensity",
        help="Likelihood target function (intensity or amplitude; default: intensity)",
    )
    parser.add_argument(
        "--optimizer",
        choices=["lbfgs", "adam"],
        default="lbfgs",
        help="Optimizer algorithm: 'lbfgs' (quasi-Newton with Wolfe line search, preserves bonds) or 'adam' (default: lbfgs)",
    )
    parser.add_argument(
        "--macrocycles",
        type=int,
        default=2,
        help="Number of macrocycles for L-BFGS refinement (default: 2)",
    )
    parser.add_argument(
        "--lbfgs-max-iter",
        type=int,
        default=15,
        help="Max iterations per L-BFGS coordinate cycle (default: 15)",
    )
    parser.add_argument(
        "--no-regularize",
        action="store_true",
        default=False,
        help="Disable pure geometry pre-regularization in L-BFGS",
    )
    parser.add_argument(
        "--no-polish",
        action="store_true",
        default=False,
        help="Disable pure geometry post-polish in L-BFGS",
    )
    parser.add_argument(
        "--xray-scales",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.2, 0.5],
        help="Scale factors for Hessian-based X-ray weight (default: 0.05 0.1 0.2 0.5)",
    )
    parser.add_argument("--steps", type=int, default=40, help="Refinement iterations per model (default: 40)")
    parser.add_argument("--lr-sites", type=float, default=0.05, help="Learning rate for sites (default: 0.05 Å)")
    parser.add_argument("--max-shift", type=float, default=0.10, help="Max coordinate shift clamp in Å (default: 0.10)")
    parser.add_argument("--nu", type=float, default=7.0, help="Degrees of freedom for Student-t noise (default: 7.0)")
    parser.add_argument("--nu-mode", default="global", choices=["global", "binned"], help="Nu mode: 'global' (single scalar for full dataset, default: global) or 'binned'")
    parser.add_argument("--device", default="mps", help="Compute device (auto, mps, cuda, cpu; default: mps)")
    parser.add_argument(
        "--b-shake-fraction",
        type=float,
        default=0.25,
        help="Fractional B-factor perturbation (+/- fraction, default: 0.25 for +/- 25%%)",
    )
    parser.add_argument(
        "--reset-b",
        action="store_true",
        default=True,
        help="Reset all B-factors to mean B before applying fractional shake (default: True)",
    )
    parser.add_argument(
        "--no-reset-b",
        action="store_false",
        dest="reset_b",
        help="Shake existing B-factors without resetting to mean B",
    )
    parser.add_argument(
        "--use-deposited",
        action="store_true",
        default=False,
        help="Refine directly from deposited PDB (opts.pdb_ref) instead of perturbed models",
    )
    parser.add_argument(
        "--shake-b",
        action="store_true",
        default=False,
        help="Reset and shake B-factors upon loading perturbed models (default: False)",
    )
    parser.add_argument(
        "--use-adp-restraints",
        dest="use_adp_restraints",
        action="store_true",
        default=True,
        help="Enable hierarchical scale-invariant Student-t ADP prior / B-factor restraints (default: True)",
    )
    parser.add_argument(
        "--no-adp-restraints",
        dest="use_adp_restraints",
        action="store_false",
        help="Disable ADP prior / B-factor restraints",
    )
    parser.add_argument(
        "--optimize-adp-weights",
        action="store_true",
        default=True,
        help="Optimize ADP prior hyperparameters via empirical Bayes on working set (default: True)",
    )
    parser.add_argument(
        "--w-adp",
        type=float,
        default=1.0,
        help="Relative weight on ADP prior (default: 1.0)",
    )
    parser.add_argument(
        "--b-iterations",
        type=int,
        default=5,
        help="Number of B-factor Newton iterations per macrocycle (default: 5)",
    )
    parser.add_argument("--verbose", action="store_true", default=True, help="Print detailed progress (default: True)")
    parser.add_argument("--quiet", dest="verbose", action="store_false", help="Quiet output")
    parser.add_argument("--no-maps", action="store_true", help="Skip building MTZ and CCP4 maps")
    parser.add_argument("--out-dir", default=str(DIR_6CZG / "shake_recover_results"), help="Output directory")

    opts = parser.parse_args(args)
    out_dir = Path(opts.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    build_maps = not opts.no_maps

    print("=" * 85)
    print(" SHAKE-AND-RECOVER REFINEMENT & MAP SYNTHESIS BENCHMARK (6CZG)")
    print(f" Perturbation RMSD: {opts.rmsd:.2f} Å | Replicates: {opts.n_reps}")
    print(f" Target: {opts.target.upper()} | Optimizer: {opts.optimizer.upper()} | Device: {opts.device}")
    print(f" Geometry Scale: Intact (w_geom = 1.0)")
    print(f" X-ray Weighting: Hessian Ratio [w_xray = xray_scale * (H_geom / H_xray)]")
    print(f" X-ray Scales tested: {opts.xray_scales}")
    if opts.optimizer == "lbfgs":
        print(f" Macrocycles: {opts.macrocycles} | L-BFGS iter/cycle: {opts.lbfgs_max_iter}")
    else:
        print(f" Refinement Steps: {opts.steps} | Learning Rate: {opts.lr_sites} Å")
    print(f" Output Directory: {out_dir}")
    print("=" * 85)

    # 1. Load ground-truth reference coordinates
    ref_model = from_files(opts.pdb_ref, opts.mtz, d_min=2.2, device="cpu", use_bulk_solvent=False)
    ref_sites = np.asarray(ref_model.xray_structure.sites_cart(), dtype=np.float64)
    ref_b = np.array([float(sc.u_iso * 8.0 * np.pi**2) for sc in ref_model.xray_structure.scatterers()])

    # Atom selections for split RMSD analysis (excluding water)
    asc = ref_model.hierarchy.atom_selection_cache()
    sel_mc = asc.selection("protein and (name N or name CA or name C or name O or name OXT)").as_numpy_array()
    sel_sc = asc.selection("protein and not (name N or name CA or name C or name O or name OXT)").as_numpy_array()
    sel_prot = asc.selection("protein").as_numpy_array()
    sel_wat = asc.selection("water").as_numpy_array()

    print(f" Structure breakdown: {len(ref_sites)} total atoms")
    print(f"   - Protein: {sel_prot.sum()} atoms (Main chain: {sel_mc.sum()}, Side chain: {sel_sc.sum()})")
    print(f"   - Solvent: {sel_wat.sum()} water molecules (excluded from split protein RMSDs)")

    def calc_split_rmsds(sites: np.ndarray) -> Dict[str, float]:
        d = sites - ref_sites
        d2 = np.sum(d**2, axis=-1)
        return {
            "rmsd_mc_3d": float(np.sqrt(np.mean(d2[sel_mc]))),
            "rmsd_sc_3d": float(np.sqrt(np.mean(d2[sel_sc]))),
            "rmsd_prot_3d": float(np.sqrt(np.mean(d2[sel_prot]))),
            "rmsd_wat_3d": float(np.sqrt(np.mean(d2[sel_wat]))),
            "rmsd_all_3d": float(np.sqrt(np.mean(d2))),
            "rmsd_mc_1d": float(np.sqrt(np.mean(d[sel_mc]**2))),
            "rmsd_sc_1d": float(np.sqrt(np.mean(d[sel_sc]**2))),
            "rmsd_prot_1d": float(np.sqrt(np.mean(d[sel_prot]**2))),
            "rmsd_wat_1d": float(np.sqrt(np.mean(d[sel_wat]**2))),
            "rmsd_all_1d": float(np.sqrt(np.mean(d**2))),
        }

    # 2. Collect model file(s)
    if opts.use_deposited:
        pert_files = [Path(opts.pdb_ref)]
    else:
        models_dir = Path(opts.models_dir)
        pattern = f"6czg_rmsd_{opts.rmsd:0.2f}_rep_*.pdb"
        pert_files = sorted(models_dir.glob(pattern))[: opts.n_reps]
        if not pert_files:
            print(f"No perturbed models matching '{pattern}' in {models_dir}. Please run run_perturbation_study.py first.")
            return 1

    records: List[Dict[str, Any]] = []

    for xs in opts.xray_scales:
        print(f"\n--- Testing X-ray Scale: {xs:.2f} ({len(pert_files)} replicates) on {opts.target.upper()} Target ---")
        for i_rep, pdb_path in enumerate(pert_files):
            t0 = time.time()
            model = from_files(
                pdb_path=str(pdb_path),
                mtz_path=opts.mtz,
                d_min=2.2,
                device=opts.device,
                use_bulk_solvent=True,
                convert_to_isotropic=True,
                target=opts.target,
                nu=opts.nu if opts.target == "intensity" else None,
                estimate_nu=(opts.target == "intensity"),
                nu_mode=opts.nu_mode,
            )
            if opts.shake_b:
                model.shake_b_iso(
                    fraction=opts.b_shake_fraction,
                    reset=opts.reset_b,
                    rng=np.random.default_rng(42 + i_rep),
                    update_model=True,
                )

            # Measure initial unrefined metrics
            sites_init = np.asarray(model.xray_structure.sites_cart(), dtype=np.float64)
            m_start = calc_split_rmsds(sites_init)
            rmsd_start = m_start["rmsd_all_1d"]

            b_init = np.array([float(sc.u_iso * 8.0 * np.pi**2) for sc in model.xray_structure.scatterers()])
            b_rmsd_start = float(np.sqrt(np.mean((b_init - ref_b)**2)))
            b_corr_start = float(np.corrcoef(b_init, ref_b)[0, 1])

            model.refine_scale_and_solvent()
            if opts.target == "intensity":
                model.refine_sigma_a_and_nu(max_cycles=2, nu_mode=opts.nu_mode)
            else:
                model.estimate_sigma_a()

            val0, _ = model.compute_target_and_gradients()
            nll_free_start = float(model.target_value_test)
            nll_work_start = float(model.target_value)
            summ0 = model.summary()
            e_geom0, _ = model.compute_geometry_energy_and_gradients(sites_init)
            gst_start = model.compute_geometry_statistics(sites_init)
            nu_start = float(model.nu) if model.nu is not None else None

            # Run refinement (L-BFGS or Adam)
            if opts.optimizer == "lbfgs":
                hist = model.refine_lbfgs(
                    macrocycles=opts.macrocycles,
                    max_iterations_per_cycle=opts.lbfgs_max_iter,
                    regularize_geometry=not opts.no_regularize,
                    regularize_steps=15,
                    xray_weight_mode="hessian",
                    xray_scale=xs,
                    refine_scales=True,
                    refine_b=True,
                    use_adp_restraints=opts.use_adp_restraints,
                    optimize_adp_weights=opts.optimize_adp_weights,
                    w_adp=opts.w_adp,
                    b_iterations=opts.b_iterations,
                    refine_sites=True,
                    refine_sigma_a=True,
                    refine_nu=(opts.target == "intensity"),
                    polish_geometry=not opts.no_polish,
                    polish_steps=10,
                    verbose=opts.verbose,
                )
            else:
                hist = model.refine_adam(
                    max_iterations=opts.steps,
                    lr_sites=opts.lr_sites,
                    refine_scales=True,
                    refine_b=True,
                    refine_sites=True,
                    refine_sigma_a=True,
                    refine_nu=(opts.target == "intensity"),
                    sigma_a_interval=4,
                    use_preconditioner=True,
                    xray_weight_mode="hessian",
                    xray_scale=xs,
                    max_shift_angstrom=opts.max_shift,
                    verbose=opts.verbose,
                )

            # Write refined PDB model
            if opts.use_deposited:
                if opts.target == "intensity":
                    prefix_base = f"6czg_{opts.target}_{opts.optimizer}_deposited_{opts.nu_mode}_nu_scale_{xs:0.1f}"
                else:
                    prefix_base = f"6czg_{opts.target}_{opts.optimizer}_deposited_scale_{xs:0.1f}"
            else:
                prefix_base = f"6czg_{opts.target}_{opts.optimizer}_rmsd_{opts.rmsd:0.2f}_scale_{xs:0.1f}_rep_{i_rep:02d}"
            out_pdb_name = f"{prefix_base}_refined.pdb"
            model.write_pdb(str(out_dir / out_pdb_name))

            # Measure final refined metrics
            sites_final = np.asarray(model.xray_structure.sites_cart(), dtype=np.float64)
            m_final = calc_split_rmsds(sites_final)
            rmsd_final = m_final["rmsd_all_1d"]
            delta_rmsd = rmsd_final - rmsd_start

            b_final = np.array([float(sc.u_iso * 8.0 * np.pi**2) for sc in model.xray_structure.scatterers()])
            b_rmsd_final = float(np.sqrt(np.mean((b_final - ref_b)**2)))
            b_corr_final = float(np.corrcoef(b_final, ref_b)[0, 1])

            model.compute_target_and_gradients()
            nll_free_final = float(model.target_value_test)
            nll_work_final = float(model.target_value)
            summ_final = model.summary()
            e_geom_final, _ = model.compute_geometry_energy_and_gradients(sites_final)
            gst_final = model.compute_geometry_statistics(sites_final)
            w_xray_final = hist["w_xray"][-1] if hist["w_xray"] else 1.0
            nu_final = float(model.nu) if model.nu is not None else None

            # Build maps
            map_files = {}
            if build_maps:
                t_map = time.time()
                map_prefix = str(out_dir / f"{prefix_base}")
                map_files = model.write_maps(prefix=map_prefix)
                print(f"    Built maps in {time.time() - t_map:.2f}s: MTZ + {len(map_files)-1} CCP4 maps")

            if opts.use_deposited:
                model.print_summary()

            rec: Dict[str, Any] = {
                "target": opts.target,
                "optimizer": opts.optimizer,
                "nu_mode": opts.nu_mode,
                "device": opts.device,
                "xray_scale": xs,
                "replicate": i_rep,
                "pdb_file": pdb_path.name,
                "refined_pdb": out_pdb_name,
                "w_xray": float(w_xray_final),
                "rmsd_start": rmsd_start,
                "rmsd_final": rmsd_final,
                "delta_rmsd": delta_rmsd,
                "rmsd_mc_3d_start": m_start["rmsd_mc_3d"],
                "rmsd_mc_3d_final": m_final["rmsd_mc_3d"],
                "delta_rmsd_mc_3d": m_final["rmsd_mc_3d"] - m_start["rmsd_mc_3d"],
                "rmsd_sc_3d_start": m_start["rmsd_sc_3d"],
                "rmsd_sc_3d_final": m_final["rmsd_sc_3d"],
                "delta_rmsd_sc_3d": m_final["rmsd_sc_3d"] - m_start["rmsd_sc_3d"],
                "rmsd_prot_3d_start": m_start["rmsd_prot_3d"],
                "rmsd_prot_3d_final": m_final["rmsd_prot_3d"],
                "delta_rmsd_prot_3d": m_final["rmsd_prot_3d"] - m_start["rmsd_prot_3d"],
                "rmsd_mc_1d_start": m_start["rmsd_mc_1d"],
                "rmsd_mc_1d_final": m_final["rmsd_mc_1d"],
                "delta_rmsd_mc_1d": m_final["rmsd_mc_1d"] - m_start["rmsd_mc_1d"],
                "rmsd_sc_1d_start": m_start["rmsd_sc_1d"],
                "rmsd_sc_1d_final": m_final["rmsd_sc_1d"],
                "delta_rmsd_sc_1d": m_final["rmsd_sc_1d"] - m_start["rmsd_sc_1d"],
                "rmsd_prot_1d_start": m_start["rmsd_prot_1d"],
                "rmsd_prot_1d_final": m_final["rmsd_prot_1d"],
                "delta_rmsd_prot_1d": m_final["rmsd_prot_1d"] - m_start["rmsd_prot_1d"],
                "nll_work_start": nll_work_start,
                "nll_work_final": nll_work_final,
                "nll_free_start": nll_free_start,
                "nll_free_final": nll_free_final,
                "delta_nll_free": nll_free_final - nll_free_start,
                "r_work_start": float(summ0["r_work"]),
                "r_work_final": float(summ_final["r_work"]),
                "r_free_start": float(summ0["r_free"]),
                "r_free_final": float(summ_final["r_free"]),
                "cc_free_start": float(summ0["cc_free_i"]),
                "cc_free_final": float(summ_final["cc_free_i"]),
                "r_intensity_free_start": float(summ0["r_intensity_free"]),
                "r_intensity_free_final": float(summ_final["r_intensity_free"]),
                "geom_e_start": float(e_geom0),
                "geom_e_final": float(e_geom_final),
                "bonds_rms_start": gst_start["bond_rms"],
                "bonds_rms_final": gst_final["bond_rms"],
                "angles_rms_start": gst_start["angle_rms"],
                "angles_rms_final": gst_final["angle_rms"],
                "planarity_rms_start": gst_start["planarity_rms"],
                "planarity_rms_final": gst_final["planarity_rms"],
                "chirality_rms_start": gst_start["chirality_rms"],
                "chirality_rms_final": gst_final["chirality_rms"],
                "geom_is_acceptable": gst_final["is_acceptable"],
                "b_rmsd_start": b_rmsd_start,
                "b_rmsd_final": b_rmsd_final,
                "delta_b_rmsd": b_rmsd_final - b_rmsd_start,
                "b_corr_start": b_corr_start,
                "b_corr_final": b_corr_final,
                "nu_start": nu_start,
                "nu_final": nu_final,
                "map_files": json.dumps(map_files),
                "elapsed_s": float(time.time() - t0),
            }
            records.append(rec)

            acc_str = "ACCEPT" if gst_final["is_acceptable"] else "STRAINED"
            nu_str = f", nu: {nu_final:.2f}" if nu_final is not None else ""
            b_str = f" | B-RMS: {b_rmsd_start:.2f}->{b_rmsd_final:.2f} Å² (corr: {b_corr_start:.2f}->{b_corr_final:.2f})" if opts.shake_b else ""
            print(
                f"  [rep {i_rep+1:2d}/{len(pert_files)}] "
                f"w_xray={w_xray_final:.2e} | "
                f"MC(3D): {m_start['rmsd_mc_3d']:.4f}->{m_final['rmsd_mc_3d']:.4f} Å | "
                f"Bonds: {gst_final['bond_rms']:.4f} Å | "
                f"Angles: {gst_final['angle_rms']:.2f}° | "
                f"Planar: {gst_final['planarity_rms']:.4f} Å [{acc_str}] | "
                f"R_free: {summ_final['r_free']*100:.2f}% | "
                f"CC_free: {summ_final['cc_free_i']:.4f}{nu_str}{b_str} | "
                f"Time: {rec['elapsed_s']:.1f}s"
            )

    # 3. Export CSV and JSON
    csv_file = out_dir / f"shake_recover_{opts.target}_summary.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"\nWrote results to {csv_file}")

    # 4. Aggregate by xray_scale
    agg: Dict[float, Dict[str, float]] = {}
    for xs in opts.xray_scales:
        group = [r for r in records if np.isclose(r["xray_scale"], xs)]
        if not group:
            continue
        agg[xs] = {
            "n": len(group),
            "w_xray_mean": float(np.mean([g["w_xray"] for g in group])),
            "nll_free_final_mean": float(np.mean([g["nll_free_final"] for g in group])),
            "delta_nll_free_mean": float(np.mean([g["delta_nll_free"] for g in group])),
            "r_free_final_mean": float(np.mean([g["r_free_final"] for g in group])),
            "cc_free_final_mean": float(np.mean([g["cc_free_final"] for g in group])),
            "r_intensity_free_final_mean": float(np.mean([g["r_intensity_free_final"] for g in group])),
            "rmsd_final_mean": float(np.mean([g["rmsd_final"] for g in group])),
            "delta_rmsd_mean": float(np.mean([g["delta_rmsd"] for g in group])),
            "rmsd_mc_3d_mean": float(np.mean([g["rmsd_mc_3d_final"] for g in group])),
            "delta_rmsd_mc_3d_mean": float(np.mean([g["delta_rmsd_mc_3d"] for g in group])),
            "rmsd_sc_3d_mean": float(np.mean([g["rmsd_sc_3d_final"] for g in group])),
            "delta_rmsd_sc_3d_mean": float(np.mean([g["delta_rmsd_sc_3d"] for g in group])),
            "rmsd_prot_3d_mean": float(np.mean([g["rmsd_prot_3d_final"] for g in group])),
            "delta_rmsd_prot_3d_mean": float(np.mean([g["delta_rmsd_prot_3d"] for g in group])),
            "rmsd_mc_1d_mean": float(np.mean([g["rmsd_mc_1d_final"] for g in group])),
            "delta_rmsd_mc_1d_mean": float(np.mean([g["delta_rmsd_mc_1d"] for g in group])),
            "bonds_rms_mean": float(np.mean([g["bonds_rms_final"] for g in group])),
            "angles_rms_mean": float(np.mean([g["angles_rms_final"] for g in group])),
            "planarity_rms_mean": float(np.mean([g["planarity_rms_final"] for g in group])),
            "chirality_rms_mean": float(np.mean([g["chirality_rms_final"] for g in group])),
            "geom_e_final_mean": float(np.mean([g["geom_e_final"] for g in group])),
            "all_acceptable": bool(all(g["geom_is_acceptable"] for g in group)),
        }

    # 5. Print Summary Table
    print("\n" + "=" * 155)
    print(f" SHAKE-AND-RECOVER REFINEMENT RESULTS ({opts.target.upper()} TARGET, {opts.optimizer.upper()} OPTIMIZER ON {opts.device.upper()})")
    print("=" * 155)
    hdr = f"{'Scale':<7} | {'w_xray':<10} | {'NLL_free':<10} | {'R_free (%)':<10} | {'CC_free(I)':<10} | {'MC RMSD(3D)':<12} | {'Δ MC(3D)':<10} | {'Bonds (Å)':<10} | {'Angles (°)':<10} | {'Planar (Å)':<10} | {'Stereo Status':<13}"
    print(hdr)
    print("-" * len(hdr))
    for xs, d in agg.items():
        val_status = "[ACCEPTABLE]" if d["all_acceptable"] else "[OUT-OF-TARGET]"
        print(
            f"{xs:<7.2f} | "
            f"{d['w_xray_mean']:<10.2e} | "
            f"{d['nll_free_final_mean']:<10.4f} | "
            f"{d['r_free_final_mean']*100:<10.2f} | "
            f"{d['cc_free_final_mean']:<10.4f} | "
            f"{d['rmsd_mc_3d_mean']:<12.4f} | "
            f"{d['delta_rmsd_mc_3d_mean']:<+10.4f} | "
            f"{d['bonds_rms_mean']:<10.4f} | "
            f"{d['angles_rms_mean']:<10.2f} | "
            f"{d['planarity_rms_mean']:<10.4f} | "
            f"{val_status:<13}"
        )
    print("=" * 155 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
