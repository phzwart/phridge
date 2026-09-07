#!/usr/bin/env python
"""Shake-and-recover refinement benchmark on synthetic Student-t intensities for 6CZG.

Applies a controlled random perturbation (0.20 Å Cartesian RMSD and B-factor shift)
to the 6CZG reference model and refines it back against synthetic noisy intensity
datasets across a series of noise multipliers ([1.0, 1.5, 2.0, 3.0, 5.0, 10.0])
under both the Amplitude (ml_f) and Intensity (ml_i) protocols.

Evaluates all recoveries using the 3-state cross-validated holdout scheme
(work / tune / test) with block-bootstrap standard errors, sign tests,
likelihood decompositions, and coordinate/B-factor recovery statistics.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from cctbx.array_family import flex
    from iotbx import pdb
    from iotbx.reflection_file_reader import any_reflection_file
    HAS_CCTBX = True
except ImportError:
    HAS_CCTBX = False

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from phridge.client import Bridge
from phridge.client.intensity_tool import from_files, IntensityModel
from phridge.contrib.intensity_ll.benchmark import ReflectionSplit, compare, ComparisonReport

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from examples.common import extract_itrue_from_mtz, generate_shaken_model as _generate_shaken_model

DIR_6CZG = ROOT / "examples" / "6czg"
SYNTH_DIR = DIR_6CZG / "synthetic_t"
DEFAULT_PDB = DIR_6CZG / "6czg.pdb"
DEFAULT_OUT_DIR = SYNTH_DIR / "shake_recover_results"


def generate_shaken_model(
    ref_pdb_path: Path,
    out_pdb_path: Path,
    pos_rmsd: float = 0.20,
    b_shake_fraction: float = 0.20,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate a single reproducible shaken model with controlled positional and B shifts."""
    return _generate_shaken_model(
        ref_pdb_path=ref_pdb_path,
        out_pdb_path=out_pdb_path,
        mtz_ref_path=DIR_6CZG / "6czg.mtz",
        pos_rmsd=pos_rmsd,
        b_shake_fraction=b_shake_fraction,
        seed=seed,
    )


def run_single_refinement(
    target: str,
    shaken_pdb: Path,
    mtz_path: Path,
    out_pdb: Path,
    multiplier: float,
    macrocycles: int = 2,
    lbfgs_iter: int = 10,
    b_iter: int = 3,
    nu: float = 7.0,
    device: str = "mps",
) -> Dict[str, Any]:
    """Run L-BFGS refinement of shaken model against synthetic noisy MTZ."""
    mult_str = f"{multiplier:g}".replace(".", "_")
    print(f"\n--- Refinement: Target = {target.upper()} | Multiplier = {multiplier}x ---", flush=True)

    t0 = time.time()
    model = from_files(
        pdb_path=str(shaken_pdb),
        mtz_path=str(mtz_path),
        d_min=2.2,
        device=device,
        use_bulk_solvent=True,
        convert_to_isotropic=True,
        target=target,
        nu=nu if target == "intensity" else None,
        estimate_nu=False,
        nu_mode="global",
    )

    # Initial scale, solvent, and sigma_A
    model.refine_scale_and_solvent()
    model.estimate_sigma_a()

    val0, _ = model.compute_target_and_gradients()
    summ0 = model.summary()

    # L-BFGS refinement with physical unit weighting and empirical Bayes ADP prior
    hist = model.refine_lbfgs(
        macrocycles=macrocycles,
        max_iterations_per_cycle=lbfgs_iter,
        b_iterations=b_iter,
        regularize_geometry=False,
        xray_weight_mode="unit",
        xray_scale=1.0,
        w_geom=1.0,
        refine_scales=True,
        refine_b=True,
        use_adp_restraints=True,
        optimize_adp_weights=True,
        w_adp=1.0,
        refine_sites=True,
        refine_sigma_a=True,
        refine_nu=False,
        polish_geometry=False,
        verbose=False,
    )
    elapsed = time.time() - t0

    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    model.write_pdb(str(out_pdb))

    summ_fin = model.summary()
    val_fin, _ = model.compute_target_and_gradients()
    sites_fin = np.asarray(model.xray_structure.sites_cart(), dtype=np.float64)
    gst_fin = model.compute_geometry_statistics(sites_fin)

    res = {
        "target": target,
        "multiplier": multiplier,
        "out_pdb": str(out_pdb),
        "elapsed_s": elapsed,
        "nll_work_init": float(val0),
        "nll_test_init": float(model.target_value_test),
        "r_work_init": float(summ0["r_work"]),
        "r_free_init": float(summ0["r_free"]),
        "cc_free_init": float(summ0["cc_free_i"]),
        "nll_work_final": float(val_fin),
        "nll_test_final": float(model.target_value_test),
        "r_work_final": float(summ_fin["r_work"]),
        "r_free_final": float(summ_fin["r_free"]),
        "cc_free_final": float(summ_fin["cc_free_i"]),
        "b_mean": float(summ_fin["b_mean"]),
        "bond_rms": float(gst_fin["bond_rms"]),
        "angle_rms": float(gst_fin["angle_rms"]),
        "planarity_rms": float(gst_fin["planarity_rms"]),
        "is_acceptable": bool(gst_fin["is_acceptable"]),
    }

    print(f"  Final: R_work = {res['r_work_final']*100:.2f}% | R_free = {res['r_free_final']*100:.2f}% | CC_free = {res['cc_free_final']:.4f}", flush=True)
    print(f"  Stereo: Bonds = {res['bond_rms']:.4f} Å | Angles = {res['angle_rms']:.2f}° | Planar = {res['planarity_rms']:.4f} Å | Time = {elapsed:.1f}s", flush=True)
    return res


def evaluate_noise_case(
    multiplier: float,
    mtz_path: Path,
    shaken_pdb: Path,
    refined_f_pdb: Path,
    refined_i_pdb: Path,
    ref_pdb: Path,
    bridge: Bridge,
    out_dir: Path,
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate recovered models under the 3-state holdout scheme and coordinate metrics."""
    mult_str = f"{multiplier:g}".replace(".", "_")
    print(f"\n" + "=" * 80)
    print(f" 3-STATE HOLDOUT EVALUATION: MULTIPLIER = {multiplier}x")
    print("=" * 80)

    # 1. Base model to extract observations and construct 3-state ReflectionSplit
    base_m = from_files(str(ref_pdb), str(mtz_path), d_min=2.2, device="cpu", use_bulk_solvent=True)
    base_m.refine_scale_and_solvent()
    split = ReflectionSplit.from_miller(base_m.i_obs, base_m.r_free_flags, n_shells=20, mode="cross_fit")

    # 2. Build F_model for each candidate structure
    model_paths = {
        "shaken": shaken_pdb,
        "recovered_f": refined_f_pdb,
        "recovered_i": refined_i_pdb,
        "deposited": ref_pdb,
    }
    f_models = {}
    atom_coords = {}
    b_factors = {}
    r_stats = {}
    truth_fit = {}

    # Extract true noise-free intensities ITRUE if available
    i_true_data = extract_itrue_from_mtz(mtz_path, base_m.i_obs)
    sqrt_i_true = np.sqrt(np.maximum(i_true_data, 0.0)) if i_true_data is not None else None

    for name, p in model_paths.items():
        m = from_files(str(p), str(mtz_path), d_min=2.2, device="cpu", use_bulk_solvent=True)
        m.refine_scale_and_solvent()
        fmod_complex = m._assemble_f_model(m.k_total, m.k_sol, m.b_sol)
        fmod = m.i_obs.customized_copy(
            data=flex.complex_double(fmod_complex.tolist())
        )
        f_models[name] = fmod
        atom_coords[name] = np.asarray(m.xray_structure.sites_cart(), dtype=np.float64)
        b_factors[name] = np.array([float(sc.u_iso * 8.0 * np.pi**2) for sc in m.xray_structure.scatterers()])
        s = m.summary()
        gst = m.compute_geometry_statistics()
        r_stats[name] = {
            "r_work": s["r_work"],
            "r_free": s["r_free"],
            "cc_free": s["cc_free_i"],
            "mean_b": s["b_mean"],
            "bond_rms": gst["bond_rms"],
            "angle_rms": gst["angle_rms"],
            "planarity_rms": gst["planarity_rms"],
        }

        if i_true_data is not None:
            i_calc = np.abs(fmod_complex)**2
            sqrt_i_calc = np.sqrt(i_calc)
            r_true_amp = float(np.sum(np.abs(sqrt_i_calc - sqrt_i_true)) / max(np.sum(sqrt_i_true), 1e-12))
            r_true_int = float(np.sum(np.abs(i_calc - i_true_data)) / max(np.sum(i_true_data), 1e-12))
            cc_true_i = float(np.corrcoef(i_calc, i_true_data)[0, 1])
            truth_fit[name] = {
                "r_true_amp": r_true_amp,
                "r_true_int": r_true_int,
                "cc_true_i": cc_true_i,
            }

    # Reference coordinates for ground truth comparison
    ref_coords = atom_coords["deposited"]
    ref_b = b_factors["deposited"]

    atom_names = [a.name.strip() for a in base_m.hierarchy.atoms()]
    is_mc = np.array([name in ["N", "CA", "C", "O"] for name in atom_names])
    is_wat = np.array(["HOH" in a.fetch_labels().resname for a in base_m.hierarchy.atoms()])
    is_sc = ~is_mc & ~is_wat

    coord_metrics = {}
    for name in ["shaken", "recovered_f", "recovered_i"]:
        coords = atom_coords[name]
        b_arr = b_factors[name]
        diff_all_1d = float(np.sqrt(np.mean((coords - ref_coords)**2)))
        diff_3d = np.sqrt(np.sum((coords - ref_coords)**2, axis=1))
        diff_all_3d = float(np.sqrt(np.mean(diff_3d**2)))
        diff_mc_3d = float(np.sqrt(np.mean(diff_3d[is_mc]**2)))
        diff_sc_3d = float(np.sqrt(np.mean(diff_3d[is_sc]**2)))
        max_drift = float(np.max(diff_3d))

        b_diff = float(np.sqrt(np.mean((b_arr - ref_b)**2)))
        b_corr = float(np.corrcoef(ref_b, b_arr)[0, 1]) if not np.all(ref_b == b_arr) else 1.0
        delta_b_mean = float(np.mean(b_arr) - np.mean(ref_b))

        coord_metrics[name] = {
            "rmsd_1d": diff_all_1d,
            "rmsd_3d": diff_all_3d,
            "rmsd_all": diff_all_1d,  # legacy compatibility
            "rmsd_mc": diff_mc_3d,
            "rmsd_sc": diff_sc_3d,
            "max_drift": max_drift,
            "b_rmsd": b_diff,
            "b_corr": b_corr,
            "delta_b_mean": delta_b_mean,
        }

    # Recovery percentage relative to shaken starting model
    rmsd_shk_1d = coord_metrics["shaken"]["rmsd_1d"]
    rmsd_shk_3d = coord_metrics["shaken"]["rmsd_3d"]
    rmsd_shk_mc = coord_metrics["shaken"]["rmsd_mc"]
    b_rmsd_shk = coord_metrics["shaken"]["b_rmsd"]

    for name in ["recovered_f", "recovered_i"]:
        recov_1d_pct = (rmsd_shk_1d - coord_metrics[name]["rmsd_1d"]) / max(rmsd_shk_1d, 1e-12) * 100.0
        recov_3d_pct = (rmsd_shk_3d - coord_metrics[name]["rmsd_3d"]) / max(rmsd_shk_3d, 1e-12) * 100.0
        recov_mc_pct = (rmsd_shk_mc - coord_metrics[name]["rmsd_mc"]) / max(rmsd_shk_mc, 1e-12) * 100.0
        recov_b_pct = (b_rmsd_shk - coord_metrics[name]["b_rmsd"]) / max(b_rmsd_shk, 1e-12) * 100.0

        coord_metrics[name]["recovery_pct"] = recov_1d_pct
        coord_metrics[name]["recovery_3d_pct"] = recov_3d_pct
        coord_metrics[name]["recovery_mc_pct"] = recov_mc_pct
        coord_metrics[name]["recovery_b_pct"] = recov_b_pct

    # 3. Held-out pairwise comparisons under 3-state holdout
    comparisons = [
        ("recovered_i_vs_recovered_f", "recovered_f", "recovered_i"),
        ("recovered_i_vs_shaken", "shaken", "recovered_i"),
        ("recovered_f_vs_shaken", "shaken", "recovered_f"),
        ("recovered_f_vs_true", "deposited", "recovered_f"),
        ("recovered_i_vs_true", "deposited", "recovered_i"),
    ]

    bench_reports = {}
    for comp_id, name_a, name_b in comparisons:
        rep = compare(
            bridge=bridge,
            i_obs=base_m.i_obs,
            models={name_a: f_models[name_a], name_b: f_models[name_b]},
            split=split,
            n_boot=n_boot,
            seed=seed,
        )
        bench_reports[comp_id] = rep
        md_file = out_dir / f"{comp_id}_mult_{mult_str}.md"
        json_file = out_dir / f"{comp_id}_mult_{mult_str}.json"
        md_file.write_text(rep.to_markdown())
        json_file.write_text(rep.to_json())

    # Print summary of I vs F comparison
    rep_ivf = bench_reports["recovered_i_vs_recovered_f"]
    print(f" Hold-Out Likelihood Comparison (recovered_i vs recovered_f):")
    print(f"   Delta NLL (nats/refl):   {rep_ivf.delta_nll:+.4f} (positive = recovered_i wins)")
    print(f"   Bootstrap SE:            {rep_ivf.se_boot:.4f} (95% CI: [{rep_ivf.ci_boot[0]:+.4f}, {rep_ivf.ci_boot[1]:+.4f}])")
    print(f"   Win Fraction P(d_h > 0): {rep_ivf.win_fraction*100:.1f}% ({rep_ivf.wins_B} wins / {rep_ivf.wins_A} losses / {rep_ivf.ties} ties)")
    print(f"   Sign Tests:              McNemar p = {rep_ivf.mcnemar_p_value:.2e} | Wilcoxon p = {rep_ivf.wilcoxon_p_value:.2e}")
    print(f"   Likelihood Decomposition:")
    print(f"     Structure term (d_S):  {rep_ivf.struct_term_B:+.4f} | Error-model term: {rep_ivf.error_term_A:+.4f}")
    print(f"   Coordinate Recovery:")
    print(f"     Shaken:                1D: {rmsd_shk_1d:.4f} Å | 3D: {rmsd_shk_3d:.4f} Å (MC: {rmsd_shk_mc:.4f} Å)")
    print(f"     recovered_f:           1D: {coord_metrics['recovered_f']['rmsd_1d']:.4f} Å (rec: {coord_metrics['recovered_f']['recovery_pct']:+.1f}%) | MC: {coord_metrics['recovered_f']['rmsd_mc']:.4f} Å (rec: {coord_metrics['recovered_f']['recovery_mc_pct']:+.1f}%)")
    print(f"     recovered_i:           1D: {coord_metrics['recovered_i']['rmsd_1d']:.4f} Å (rec: {coord_metrics['recovered_i']['recovery_pct']:+.1f}%) | MC: {coord_metrics['recovered_i']['rmsd_mc']:.4f} Å (rec: {coord_metrics['recovered_i']['recovery_mc_pct']:+.1f}%)")
    print(f"     B-factor correlation:  f: {coord_metrics['recovered_f']['b_corr']:.4f} | i: {coord_metrics['recovered_i']['b_corr']:.4f} (shaken: {coord_metrics['shaken']['b_corr']:.4f})")
    if truth_fit:
        print(f"   Fit to Ground Truth (ITRUE):")
        print(f"     shaken:      R_true(amp) = {truth_fit['shaken']['r_true_amp']*100:.2f}% | CC_true = {truth_fit['shaken']['cc_true_i']:.4f}")
        print(f"     recovered_f: R_true(amp) = {truth_fit['recovered_f']['r_true_amp']*100:.2f}% | CC_true = {truth_fit['recovered_f']['cc_true_i']:.4f}")
        print(f"     recovered_i: R_true(amp) = {truth_fit['recovered_i']['r_true_amp']*100:.2f}% | CC_true = {truth_fit['recovered_i']['cc_true_i']:.4f}")

    case_summary = {
        "multiplier": multiplier,
        "coord_metrics": coord_metrics,
        "r_stats": r_stats,
        "truth_fit": truth_fit,
        "i_vs_f": {
            "delta_nll": rep_ivf.delta_nll,
            "se_boot": rep_ivf.se_boot,
            "ci_boot": rep_ivf.ci_boot,
            "win_fraction": rep_ivf.win_fraction,
            "wins_i": rep_ivf.wins_B,
            "wins_f": rep_ivf.wins_A,
            "mcnemar_p": rep_ivf.mcnemar_p_value,
            "wilcoxon_p": rep_ivf.wilcoxon_p_value,
            "struct_term": rep_ivf.struct_term_B,
            "error_term": rep_ivf.error_term_A,
            "conclusion": rep_ivf.structure_conclusion,
        },
        "i_vs_shaken": {
            "delta_nll": bench_reports["recovered_i_vs_shaken"].delta_nll,
            "se_boot": bench_reports["recovered_i_vs_shaken"].se_boot,
            "win_fraction": bench_reports["recovered_i_vs_shaken"].win_fraction,
        },
        "f_vs_shaken": {
            "delta_nll": bench_reports["recovered_f_vs_shaken"].delta_nll,
            "se_boot": bench_reports["recovered_f_vs_shaken"].se_boot,
            "win_fraction": bench_reports["recovered_f_vs_shaken"].win_fraction,
        },
        "f_vs_true": {
            "delta_nll": bench_reports["recovered_f_vs_true"].delta_nll,
            "se_boot": bench_reports["recovered_f_vs_true"].se_boot,
            "win_fraction": bench_reports["recovered_f_vs_true"].win_fraction,
        },
        "i_vs_true": {
            "delta_nll": bench_reports["recovered_i_vs_true"].delta_nll,
            "se_boot": bench_reports["recovered_i_vs_true"].se_boot,
            "win_fraction": bench_reports["recovered_i_vs_true"].win_fraction,
        },
    }

    return case_summary


def print_master_summary(all_summaries: List[Dict[str, Any]]) -> str:
    """Format and print master summary table across all noise multipliers."""
    lines = []
    lines.append("=" * 165)
    lines.append(" MASTER BENCHMARK SUMMARY: SHAKE-AND-RECOVER UNDER STUDENT-T NOISE (nu=7.0)")
    lines.append("=" * 165)
    lines.append(
        f"{'Mult':<5} | {'Model':<11} | {'1D RMSD':<8} | {'3D RMSD':<8} | {'MC RMSD':<8} | {'Recov MC':<8} | "
        f"{'B RMSD':<7} | {'r(B,B0)':<7} | {'R_noisy':<7} | {'R_true':<7} | {'CC_true':<7} | {'Bonds':<6} | {'Angles':<6} | {'d_NLL(vs Shk)':<13} | {'Delta(I-F)':<10} | {'Win%(I>F)':<9}"
    )
    lines.append("-" * 165)

    for s in all_summaries:
        mult = s["multiplier"]
        cm = s["coord_metrics"]
        rs = s["r_stats"]
        tf = s.get("truth_fit", {})
        ivf = s["i_vs_f"]

        # Shaken row
        shk_c = cm["shaken"]
        shk_r = rs["shaken"]
        shk_t = tf.get("shaken", {})
        r_tr_s = f"{shk_t.get('r_true_amp', 0.0)*100:.2f}%" if shk_t else "---"
        cc_tr_s = f"{shk_t.get('cc_true_i', 0.0):.4f}" if shk_t else "---"
        lines.append(
            f"{mult:<5.1f} | {'shaken':<11} | {shk_c['rmsd_1d']:<8.4f} | {shk_c['rmsd_3d']:<8.4f} | {shk_c['rmsd_mc']:<8.4f} | {'---':<8} | "
            f"{shk_c['b_rmsd']:<7.2f} | {shk_c['b_corr']:<7.4f} | {shk_r['r_free']*100:<6.2f}% | {r_tr_s:<7} | {cc_tr_s:<7} | "
            f"{shk_r['bond_rms']:<6.4f} | {shk_r['angle_rms']:<6.2f} | {'---':<13} | {'---':<10} | {'---':<9}"
        )

        # F row
        fc_c = cm["recovered_f"]
        fc_r = rs["recovered_f"]
        fc_t = tf.get("recovered_f", {})
        r_tr_f = f"{fc_t.get('r_true_amp', 0.0)*100:.2f}%" if fc_t else "---"
        cc_tr_f = f"{fc_t.get('cc_true_i', 0.0):.4f}" if fc_t else "---"
        d_f_shk = f"{s['f_vs_shaken']['delta_nll']:+.4f}"
        lines.append(
            f"{'':<5} | {'ml_f (Amp)':<11} | {fc_c['rmsd_1d']:<8.4f} | {fc_c['rmsd_3d']:<8.4f} | {fc_c['rmsd_mc']:<8.4f} | {fc_c.get('recovery_mc_pct', fc_c['recovery_pct']):<+7.1f}% | "
            f"{fc_c['b_rmsd']:<7.2f} | {fc_c['b_corr']:<7.4f} | {fc_r['r_free']*100:<6.2f}% | {r_tr_f:<7} | {cc_tr_f:<7} | "
            f"{fc_r['bond_rms']:<6.4f} | {fc_r['angle_rms']:<6.2f} | {d_f_shk:<13} | {'reference':<10} | {'---':<9}"
        )

        # I row
        ic_c = cm["recovered_i"]
        ic_r = rs["recovered_i"]
        ic_t = tf.get("recovered_i", {})
        r_tr_i = f"{ic_t.get('r_true_amp', 0.0)*100:.2f}%" if ic_t else "---"
        cc_tr_i = f"{ic_t.get('cc_true_i', 0.0):.4f}" if ic_t else "---"
        d_i_shk = f"{s['i_vs_shaken']['delta_nll']:+.4f}"
        d_nll_str = f"{ivf['delta_nll']:+.4f}"
        win_str = f"{ivf['win_fraction']*100:.1f}%"
        lines.append(
            f"{'':<5} | {'ml_i (Int)':<11} | {ic_c['rmsd_1d']:<8.4f} | {ic_c['rmsd_3d']:<8.4f} | {ic_c['rmsd_mc']:<8.4f} | {ic_c.get('recovery_mc_pct', ic_c['recovery_pct']):<+7.1f}% | "
            f"{ic_c['b_rmsd']:<7.2f} | {ic_c['b_corr']:<7.4f} | {ic_r['r_free']*100:<6.2f}% | {r_tr_i:<7} | {cc_tr_i:<7} | "
            f"{ic_r['bond_rms']:<6.4f} | {ic_r['angle_rms']:<6.2f} | {d_i_shk:<13} | {d_nll_str:<10} | {win_str:<9}"
        )
        lines.append("-" * 165)

    lines.append("=" * 165)
    summary_text = "\n".join(lines)
    print(summary_text, flush=True)
    return summary_text


def save_master_markdown_report(all_summaries: List[Dict[str, Any]], out_dir: Path, pos_rmsd: float = 0.10, b_shake_fraction: float = 0.10) -> Path:
    """Save comprehensive master report in markdown."""
    report_path = out_dir / "shake_recover_synthetic_t_report.md"
    md = []
    md.append(f"# Shake-and-Recover Refinement Benchmark on Synthetic Student-t Intensities ({pos_rmsd:.2f} Å Perturbation)")
    md.append("")
    md.append("## Overview")
    md.append("This study evaluates the comparative recovery performance of **Intensity-based likelihood (`ml_i`)** versus **Amplitude-based likelihood (`ml_f`)** starting from a perturbed model under controlled heavy-tailed noise.")
    md.append("")
    md.append("### Protocol Details")
    md.append("- **System**: 6CZG (high-resolution model)")
    md.append(f"- **Perturbation**: {pos_rmsd:.2f} Å Cartesian RMSD on positions; +/- {b_shake_fraction*100:.0f}% fractional Gaussian shift on B-factors.")
    md.append("- **Noise Distribution**: Heavy-tailed Student-t error model ($\\nu = 7.0$) preserving per-reflection experimental $I/\\sigma$.")
    md.append("- **Noise Multipliers**: `[1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 50.0, 100.0]` applied to $\\sigma_I$.")
    md.append("- **Cross-Validation**: 3-state holdout scheme (`ReflectionSplit` with 2-fold cross-fitting over the 10% test set).")
    md.append("- **Hyperparameters & Regularization**: Physical unit weighting ($w_{\\text{xray}}=N_{\\text{work}}, w_{\\text{geom}}=1.0$) and hierarchical Student-t ADP prior ($\nu_{\\text{ADP}}=4, \\tau_{1-2} < \\tau_{1-3} < \\tau_{\\text{sphere}}$) optimized via empirical Bayes.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Master Performance Table")
    md.append("")
    md.append("| Multiplier | Model | 1D RMSD (Å) | 3D RMSD (Å) | MC RMSD (Å) | Recov MC % | B RMSD (Å²) | $r(B, B_0)$ | $R_{\\text{free}}$ (Noisy) | $R_{\\text{true}}$ (Amp) | $CC_{\\text{true}}$ | Bonds (Å) | Angles (°) | $\\Delta\\text{NLL}_{I - F}$ | Boot SE | Win % ($I > F$) |")
    md.append("|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    for s in all_summaries:
        mult = s["multiplier"]
        cm = s["coord_metrics"]
        rs = s["r_stats"]
        tf = s.get("truth_fit", {})
        ivf = s["i_vs_f"]

        shk_c, shk_r = cm["shaken"], rs["shaken"]
        shk_t = tf.get("shaken", {})
        r_tr_s = f"{shk_t.get('r_true_amp', 0.0)*100:.2f}%" if shk_t else "---"
        cc_tr_s = f"{shk_t.get('cc_true_i', 0.0):.4f}" if shk_t else "---"
        md.append(f"| **{mult:g}x** | Shaken | `{shk_c['rmsd_1d']:.4f}` | `{shk_c['rmsd_3d']:.4f}` | `{shk_c['rmsd_mc']:.4f}` | --- | `{shk_c['b_rmsd']:.2f}` | `{shk_c['b_corr']:.4f}` | `{shk_r['r_free']*100:.2f}%` | `{r_tr_s}` | `{cc_tr_s}` | `{shk_r['bond_rms']:.4f}` | `{shk_r['angle_rms']:.2f}` | --- | --- | --- |")

        fc_c, fc_r = cm["recovered_f"], rs["recovered_f"]
        fc_t = tf.get("recovered_f", {})
        r_tr_f = f"{fc_t.get('r_true_amp', 0.0)*100:.2f}%" if fc_t else "---"
        cc_tr_f = f"{fc_t.get('cc_true_i', 0.0):.4f}" if fc_t else "---"
        rec_mc_f = fc_c.get("recovery_mc_pct", fc_c["recovery_pct"])
        md.append(f"| | `ml_f` (Amp) | `{fc_c['rmsd_1d']:.4f}` | `{fc_c['rmsd_3d']:.4f}` | `{fc_c['rmsd_mc']:.4f}` | `{rec_mc_f:+.1f}%` | `{fc_c['b_rmsd']:.2f}` | `{fc_c['b_corr']:.4f}` | `{fc_r['r_free']*100:.2f}%` | `{r_tr_f}` | `{cc_tr_f}` | `{fc_r['bond_rms']:.4f}` | `{fc_r['angle_rms']:.2f}` | *reference* | --- | --- |")

        ic_c, ic_r = cm["recovered_i"], rs["recovered_i"]
        ic_t = tf.get("recovered_i", {})
        r_tr_i = f"{ic_t.get('r_true_amp', 0.0)*100:.2f}%" if ic_t else "---"
        cc_tr_i = f"{ic_t.get('cc_true_i', 0.0):.4f}" if ic_t else "---"
        rec_mc_i = ic_c.get("recovery_mc_pct", ic_c["recovery_pct"])
        md.append(f"| | **`ml_i` (Int)** | **`{ic_c['rmsd_1d']:.4f}`** | **`{ic_c['rmsd_3d']:.4f}`** | **`{ic_c['rmsd_mc']:.4f}`** | **`{rec_mc_i:+.1f}%`** | **`{ic_c['b_rmsd']:.2f}`** | **`{ic_c['b_corr']:.4f}`** | `{ic_r['r_free']*100:.2f}%` | **`{r_tr_i}`** | **`{cc_tr_i}`** | `{ic_r['bond_rms']:.4f}` | `{ic_r['angle_rms']:.2f}` | **`{ivf['delta_nll']:+.4f}`** | `{ivf['se_boot']:.4f}` | **`{ivf['win_fraction']*100:.1f}%`** |")

    md.append("")
    md.append("---")
    md.append("")
    md.append("## Statistical Insights")
    md.append("")
    for s in all_summaries:
        mult = s["multiplier"]
        ivf = s["i_vs_f"]
        cm = s["coord_metrics"]
        md.append(f"### Noise Multiplier {mult:g}x")
        md.append(f"- **Held-Out NLL Gain ($\\Delta$ Gain)**: `{ivf['delta_nll']:+.4f}` nats/reflection (Bootstrap SE: `{ivf['se_boot']:.4f}`, 95% CI: `[{ivf['ci_boot'][0]:+.4f}, {ivf['ci_boot'][1]:+.4f}]`)")
        md.append(f"- **Pairwise Win Fraction**: `{ivf['win_fraction']*100:.1f}%` ({ivf['wins_i']} wins vs {ivf['wins_f']} losses)")
        md.append(f"- **Sign Tests**: McNemar $p = {ivf['mcnemar_p']:.2e}$ | Wilcoxon signed-rank $p = {ivf['wilcoxon_p']:.2e}$")
        md.append(f"- **Two-Way Decomposition**: Structure Term $\\Delta_S = {ivf['struct_term']:+.4f}$, Error-Model Term $\\Delta_E = {ivf['error_term']:+.4f}$ ({ivf['conclusion']})")
        md.append(f"- **Coordinate Recovery (MC)**: `ml_i` recovered `{cm['recovered_i'].get('recovery_mc_pct', 0.0):+.1f}%` vs `{cm['recovered_f'].get('recovery_mc_pct', 0.0):+.1f}%` for `ml_f`.")
        md.append("")

    report_path.write_text("\n".join(md))
    print(f"\nWrote master markdown report to: {report_path}")
    return report_path


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Shake-and-recover refinement benchmark on synthetic Student-t intensities."
    )
    parser.add_argument("--pdb", default=str(DEFAULT_PDB), help="Reference PDB file")
    parser.add_argument("--synthetic-dir", default=str(SYNTH_DIR), help="Directory of synthetic MTZ files")
    parser.add_argument(
        "--multipliers",
        type=float,
        nargs="+",
        default=[1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 50.0, 100.0],
        help="List of noise multipliers to process (default: 1.0 1.5 2.0 3.0 5.0 10.0 50.0 100.0)",
    )
    parser.add_argument("--pos-rmsd", type=float, default=0.10, help="Target coordinate shift RMSD in Å (default: 0.10)")
    parser.add_argument("--b-shake-fraction", type=float, default=0.10, help="Fractional B shift (default: 0.10)")
    parser.add_argument("--nu", type=float, default=7.0, help="Student-t degrees of freedom (default: 7.0)")
    parser.add_argument("--macrocycles", type=int, default=2, help="Macrocycles of L-BFGS refinement (default: 2)")
    parser.add_argument("--lbfgs-iter", type=int, default=10, help="Max L-BFGS iterations per cycle (default: 10)")
    parser.add_argument("--b-iter", type=int, default=3, help="B-factor Newton iterations per cycle (default: 3)")
    parser.add_argument("--device", default="mps", help="Compute device (mps, cuda, cpu; default: mps)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shaking and bootstrap (default: 42)")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: synthetic_t/shake_recover_<pos_rmsd>A)")
    parser.add_argument("--force-refine", action="store_true", help="Force re-running refinement even if output PDB exists")
    opts = parser.parse_args(args)

    synth_dir = Path(opts.synthetic_dir)
    ref_pdb = Path(opts.pdb)
    if opts.out_dir is None:
        out_dir = synth_dir / f"shake_recover_{opts.pos_rmsd:0.2f}A"
    else:
        out_dir = Path(opts.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print(" 6CZG SYNTHETIC STUDENT-T SHAKE-AND-RECOVER BENCHMARK")
    print(f" Reference PDB:     {ref_pdb}")
    print(f" Synthetic MTZ Dir: {synth_dir}")
    print(f" Multipliers:       {opts.multipliers}")
    print(f" Pos RMSD Shift:    {opts.pos_rmsd:.2f} Å | B Shift: +/- {opts.b_shake_fraction*100:.0f}%")
    print(f" Student-t nu:      {opts.nu:.1f}")
    print(f" Refinement:        {opts.macrocycles} macrocycles x ({opts.lbfgs_iter} L-BFGS + {opts.b_iter} B iter)")
    print(f" Device:            {opts.device}")
    print(f" Output Directory:  {out_dir}")
    print("=" * 85)

    # 1. Generate or load single reproducible shaken starting model
    shaken_pdb = out_dir / f"6czg_shaken_{opts.pos_rmsd:0.2f}A.pdb"
    if not shaken_pdb.exists() or opts.force_refine:
        shaken_stats = generate_shaken_model(
            ref_pdb_path=ref_pdb,
            out_pdb_path=shaken_pdb,
            pos_rmsd=opts.pos_rmsd,
            b_shake_fraction=opts.b_shake_fraction,
            seed=opts.seed,
        )
    else:
        print(f"\n[Reusing existing shaken model: {shaken_pdb.name}]")

    bridge = Bridge(memory=True)
    all_summaries: List[Dict[str, Any]] = []

    # Check for existing summary report to reuse already evaluated cases
    existing_summaries: Dict[float, Dict[str, Any]] = {}
    json_path = out_dir / "shake_recover_synthetic_t_report.json"
    if json_path.exists() and not opts.force_refine:
        try:
            prev_data = json.loads(json_path.read_text())
            for item in prev_data:
                existing_summaries[float(item["multiplier"])] = item
        except Exception:
            pass

    # 2. Iterate over all noise multipliers
    for mult in opts.multipliers:
        mult_str = f"{mult:g}".replace(".", "_")
        mtz_name = f"6czg_synthetic_t_mult_{mult_str}.mtz"
        mtz_path = synth_dir / mtz_name
        if not mtz_path.exists():
            print(f"Warning: MTZ file {mtz_path} not found. Skipping multiplier {mult}.")
            continue

        out_f = out_dir / f"6czg_recovered_f_mult_{mult_str}.pdb"
        out_i = out_dir / f"6czg_recovered_i_mult_{mult_str}.pdb"
        f_recomputed = False
        i_recomputed = False

        # Refine Amplitude (ml_f)
        if not out_f.exists() or opts.force_refine:
            run_single_refinement(
                target="amplitude",
                shaken_pdb=shaken_pdb,
                mtz_path=mtz_path,
                out_pdb=out_f,
                multiplier=mult,
                macrocycles=opts.macrocycles,
                lbfgs_iter=opts.lbfgs_iter,
                b_iter=opts.b_iter,
                nu=opts.nu,
                device=opts.device,
            )
            f_recomputed = True
        else:
            print(f"\n[Reusing existing recovered_f model: {out_f.name}]")

        # Refine Intensity (ml_i)
        if not out_i.exists() or opts.force_refine:
            run_single_refinement(
                target="intensity",
                shaken_pdb=shaken_pdb,
                mtz_path=mtz_path,
                out_pdb=out_i,
                multiplier=mult,
                macrocycles=opts.macrocycles,
                lbfgs_iter=opts.lbfgs_iter,
                b_iter=opts.b_iter,
                nu=opts.nu,
                device=opts.device,
            )
            i_recomputed = True
        else:
            print(f"\n[Reusing existing recovered_i model: {out_i.name}]")

        # Evaluate noise case under 3-state holdout
        if not f_recomputed and not i_recomputed and mult in existing_summaries:
            print(f"\n[Reusing existing evaluation for multiplier = {mult:g}x]")
            case_summary = existing_summaries[mult]
        else:
            case_summary = evaluate_noise_case(
                multiplier=mult,
                mtz_path=mtz_path,
                shaken_pdb=shaken_pdb,
                refined_f_pdb=out_f,
                refined_i_pdb=out_i,
                ref_pdb=ref_pdb,
                bridge=bridge,
                out_dir=out_dir,
                n_boot=1000,
                seed=opts.seed,
            )
        all_summaries.append(case_summary)

    # 3. Master Summary
    print_master_summary(all_summaries)
    save_master_markdown_report(all_summaries, out_dir, pos_rmsd=opts.pos_rmsd, b_shake_fraction=opts.b_shake_fraction)

    json_path = out_dir / "shake_recover_synthetic_t_report.json"
    json_path.write_text(json.dumps(all_summaries, indent=2))
    print(f"Wrote master JSON report to: {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
