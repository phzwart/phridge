#!/usr/bin/env python
"""Synthetic Student-t Noise Drift Study for 6CZG.

Takes the true unperturbed 6CZG model and refines it directly against synthetic
noisy intensity datasets across a series of noise multipliers ([1.0, 1.5, 2.0, 3.0, 5.0, 10.0])
under both Amplitude (ml_f) and Intensity (ml_i) likelihood targets.

Measures how much the model drifts away from ground-truth atomic positions,
B-factors, and noise-free structure factors (ITRUE), and evaluates generalization
under the 3-state cross-validated holdout scheme.
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

DIR_6CZG = ROOT / "examples" / "6czg"
SYNTH_DIR = DIR_6CZG / "synthetic_t"
DEFAULT_PDB = DIR_6CZG / "6czg.pdb"
DEFAULT_OUT_DIR = SYNTH_DIR / "drift_results"


def extract_itrue_from_mtz(mtz_path: Path, ref_miller_array: Any) -> np.ndarray:
    """Extract noise-free ground truth intensities (ITRUE) matched to ref_miller_array."""
    reader = any_reflection_file(str(mtz_path))
    arrays = reader.as_miller_arrays()
    itrue_arr = None
    for arr in arrays:
        lbls = [l.lower() for l in (arr.info().labels if arr.info() else [])]
        if "itrue" in lbls:
            itrue_arr = arr
            break

    if itrue_arr is None:
        raise ValueError(f"Could not find ITRUE column in {mtz_path}")

    # Match indices to ref_miller_array
    matches = ref_miller_array.match_indices(itrue_arr)
    matched_itrue = itrue_arr.select(matches.permutation())
    return np.asarray(matched_itrue.data(), dtype=np.float64)


def run_drift_refinement(
    target: str,
    true_pdb: Path,
    mtz_path: Path,
    out_pdb: Path,
    multiplier: float,
    macrocycles: int = 2,
    lbfgs_iter: int = 10,
    b_iter: int = 3,
    nu: float = 7.0,
    device: str = "mps",
) -> Dict[str, Any]:
    """Refine true model from zero perturbation against noisy MTZ."""
    mult_str = f"{multiplier:g}".replace(".", "_")
    print(f"\n--- Refining True Model: Target = {target.upper()} | Multiplier = {multiplier}x ---", flush=True)

    t0 = time.time()
    model = from_files(
        pdb_path=str(true_pdb),
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


def evaluate_drift_case(
    multiplier: float,
    mtz_path: Path,
    true_pdb: Path,
    drifted_f_pdb: Path,
    drifted_i_pdb: Path,
    bridge: Bridge,
    out_dir: Path,
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate drift metrics relative to the true unperturbed ground-truth model."""
    mult_str = f"{multiplier:g}".replace(".", "_")
    print(f"\n" + "=" * 80)
    print(f" EVALUATING DRIFT: NOISE MULTIPLIER = {multiplier}x")
    print("=" * 80)

    # 1. Base model to extract observations, ITRUE, and construct 3-state ReflectionSplit
    base_m = from_files(str(true_pdb), str(mtz_path), d_min=2.2, device="cpu", use_bulk_solvent=True)
    base_m.refine_scale_and_solvent()
    split = ReflectionSplit.from_miller(base_m.i_obs, base_m.r_free_flags, n_shells=20, mode="cross_fit")

    # Extract true noise-free intensities ITRUE
    i_true_data = extract_itrue_from_mtz(mtz_path, base_m.i_obs)
    sqrt_i_true = np.sqrt(np.maximum(i_true_data, 0.0))

    # 2. Build F_model for each candidate structure
    model_paths = {
        "true_model": true_pdb,
        "drifted_f": drifted_f_pdb,
        "drifted_i": drifted_i_pdb,
    }
    f_models = {}
    atom_coords = {}
    b_factors = {}
    r_stats = {}
    truth_fit = {}

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

        # Quality against ground truth ITRUE
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
    ref_coords = atom_coords["true_model"]
    ref_b = b_factors["true_model"]

    atom_names = [a.name.strip() for a in base_m.hierarchy.atoms()]
    is_mc = np.array([name in ["N", "CA", "C", "O"] for name in atom_names])
    is_wat = np.array(["HOH" in a.fetch_labels().resname for a in base_m.hierarchy.atoms()])
    is_sc = ~is_mc & ~is_wat

    coord_metrics = {}
    for name in ["true_model", "drifted_f", "drifted_i"]:
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
            "rmsd_mc": diff_mc_3d,
            "rmsd_sc": diff_sc_3d,
            "max_drift": max_drift,
            "b_rmsd": b_diff,
            "b_corr": b_corr,
            "delta_b_mean": delta_b_mean,
        }

    # 3. Held-out pairwise comparisons under 3-state holdout
    comparisons = [
        ("drifted_i_vs_drifted_f", "drifted_f", "drifted_i"),
        ("drifted_f_vs_true", "true_model", "drifted_f"),
        ("drifted_i_vs_true", "true_model", "drifted_i"),
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

    # Summary prints
    rep_ivf = bench_reports["drifted_i_vs_drifted_f"]
    rep_fvt = bench_reports["drifted_f_vs_true"]
    rep_ivt = bench_reports["drifted_i_vs_true"]

    print(f" Coordinate Drift from True Model:")
    print(f"   drifted_f: 1D RMSD = {coord_metrics['drifted_f']['rmsd_1d']:.4f} Å | 3D RMSD = {coord_metrics['drifted_f']['rmsd_3d']:.4f} Å (MC: {coord_metrics['drifted_f']['rmsd_mc']:.4f} Å, Max: {coord_metrics['drifted_f']['max_drift']:.4f} Å)")
    print(f"   drifted_i: 1D RMSD = {coord_metrics['drifted_i']['rmsd_1d']:.4f} Å | 3D RMSD = {coord_metrics['drifted_i']['rmsd_3d']:.4f} Å (MC: {coord_metrics['drifted_i']['rmsd_mc']:.4f} Å, Max: {coord_metrics['drifted_i']['max_drift']:.4f} Å)")
    print(f" B-Factor Drift:")
    print(f"   drifted_f: B-RMSD = {coord_metrics['drifted_f']['b_rmsd']:.2f} Å² | Delta B_mean = {coord_metrics['drifted_f']['delta_b_mean']:+.2f} Å² | r(B,B_true) = {coord_metrics['drifted_f']['b_corr']:.4f}")
    print(f"   drifted_i: B-RMSD = {coord_metrics['drifted_i']['b_rmsd']:.2f} Å² | Delta B_mean = {coord_metrics['drifted_i']['delta_b_mean']:+.2f} Å² | r(B,B_true) = {coord_metrics['drifted_i']['b_corr']:.4f}")
    print(f" Fit to Ground Truth Structure Factors (ITRUE):")
    print(f"   True Model: R_true(amp) = {truth_fit['true_model']['r_true_amp']*100:.2f}% | CC_true = {truth_fit['true_model']['cc_true_i']:.4f}")
    print(f"   drifted_f:  R_true(amp) = {truth_fit['drifted_f']['r_true_amp']*100:.2f}% | CC_true = {truth_fit['drifted_f']['cc_true_i']:.4f}")
    print(f"   drifted_i:  R_true(amp) = {truth_fit['drifted_i']['r_true_amp']*100:.2f}% | CC_true = {truth_fit['drifted_i']['cc_true_i']:.4f}")
    print(f" 3-State Holdout Comparison (drifted_i vs drifted_f):")
    print(f"   Delta NLL (nats/refl):   {rep_ivf.delta_nll:+.4f} (Boot SE: {rep_ivf.se_boot:.4f}) | Win %: {rep_ivf.win_fraction*100:.1f}%")
    print(f"   Held-out vs True Model:  f vs True = {rep_fvt.delta_nll:+.4f} | i vs True = {rep_ivt.delta_nll:+.4f}")

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
        "f_vs_true": {
            "delta_nll": rep_fvt.delta_nll,
            "se_boot": rep_fvt.se_boot,
            "win_fraction": rep_fvt.win_fraction,
        },
        "i_vs_true": {
            "delta_nll": rep_ivt.delta_nll,
            "se_boot": rep_ivt.se_boot,
            "win_fraction": rep_ivt.win_fraction,
        },
    }

    return case_summary


def print_master_drift_summary(all_summaries: List[Dict[str, Any]]) -> str:
    """Format and print master drift summary table."""
    lines = []
    lines.append("=" * 155)
    lines.append(" MASTER DRIFT SUMMARY: REFINING TRUE 6CZG MODEL AGAINST STUDENT-T NOISE (nu=7.0)")
    lines.append("=" * 155)
    lines.append(
        f"{'Mult':<5} | {'Model':<11} | {'Pos Drift':<9} | {'MC Drift':<9} | {'Max Drift':<9} | "
        f"{'B RMSD':<7} | {'Delta B':<7} | {'r(B,B0)':<7} | {'R_noisy':<7} | {'R_true':<7} | {'CC_true':<7} | "
        f"{'Bonds':<6} | {'Angles':<6} | {'d_NLL(vs True)':<14} | {'Delta(I-F)':<10} | {'Win%(I>F)':<9}"
    )
    lines.append("-" * 155)

    for s in all_summaries:
        mult = s["multiplier"]
        cm = s["coord_metrics"]
        rs = s["r_stats"]
        tf = s["truth_fit"]
        ivf = s["i_vs_f"]

        # True row
        tr_c = cm["true_model"]
        tr_r = rs["true_model"]
        tr_t = tf["true_model"]
        lines.append(
            f"{mult:<5.1f} | {'true_model':<11} | {'0.0000':<9} | {'0.0000':<9} | {'0.0000':<9} | "
            f"{'0.00':<7} | {'+0.00':<7} | {'1.0000':<7} | {tr_r['r_free']*100:<6.2f}% | {tr_t['r_true_amp']*100:<6.2f}% | {tr_t['cc_true_i']:<7.4f} | "
            f"{tr_r['bond_rms']:<6.4f} | {tr_r['angle_rms']:<6.2f} | {'reference':<14} | {'---':<10} | {'---':<9}"
        )

        # F row
        fc_c = cm["drifted_f"]
        fc_r = rs["drifted_f"]
        fc_t = tf["drifted_f"]
        d_f_true = f"{s['f_vs_true']['delta_nll']:+.4f}"
        lines.append(
            f"{'':<5} | {'ml_f (Amp)':<11} | {fc_c['rmsd_1d']:<9.4f} | {fc_c['rmsd_mc']:<9.4f} | {fc_c['max_drift']:<9.4f} | "
            f"{fc_c['b_rmsd']:<7.2f} | {fc_c['delta_b_mean']:<+7.2f} | {fc_c['b_corr']:<7.4f} | {fc_r['r_free']*100:<6.2f}% | {fc_t['r_true_amp']*100:<6.2f}% | {fc_t['cc_true_i']:<7.4f} | "
            f"{fc_r['bond_rms']:<6.4f} | {fc_r['angle_rms']:<6.2f} | {d_f_true:<14} | {'reference':<10} | {'---':<9}"
        )

        # I row
        ic_c = cm["drifted_i"]
        ic_r = rs["drifted_i"]
        ic_t = tf["drifted_i"]
        d_i_true = f"{s['i_vs_true']['delta_nll']:+.4f}"
        d_nll_str = f"{ivf['delta_nll']:+.4f}"
        win_str = f"{ivf['win_fraction']*100:.1f}%"
        lines.append(
            f"{'':<5} | {'ml_i (Int)':<11} | {ic_c['rmsd_1d']:<9.4f} | {ic_c['rmsd_mc']:<9.4f} | {ic_c['max_drift']:<9.4f} | "
            f"{ic_c['b_rmsd']:<7.2f} | {ic_c['delta_b_mean']:<+7.2f} | {ic_c['b_corr']:<7.4f} | {ic_r['r_free']*100:<6.2f}% | {ic_t['r_true_amp']*100:<6.2f}% | {ic_t['cc_true_i']:<7.4f} | "
            f"{ic_r['bond_rms']:<6.4f} | {ic_r['angle_rms']:<6.2f} | {d_i_true:<14} | {d_nll_str:<10} | {win_str:<9}"
        )
        lines.append("-" * 155)

    lines.append("=" * 155)
    summary_text = "\n".join(lines)
    print(summary_text, flush=True)
    return summary_text


def save_master_drift_report(all_summaries: List[Dict[str, Any]], out_dir: Path) -> Path:
    """Save comprehensive drift report in markdown."""
    report_path = out_dir / "synthetic_t_drift_report.md"
    md = []
    md.append("# Synthetic Student-t Noise Drift Study: Refining from the Ground Truth Model")
    md.append("")
    md.append("## Overview")
    md.append("This study evaluates **parameter drift and noise overfitting** when refining the unperturbed ground-truth 6CZG model directly against synthetic Student-t noisy data across increasing noise multipliers (1.0x to 10.0x).")
    md.append("")
    md.append("### Key Questions")
    md.append("1. **Coordinate Drift**: How far do atomic positions wander from the true structure as data quality degrades?")
    md.append("2. **B-Factor Drift**: How do atomic displacement parameters adapt to noise, and does the prior preserve true B-factor rank-order?")
    md.append("3. **Ground Truth Fit**: When refining against noisy data, how does the model's agreement with the true noise-free structure factors (`ITRUE`) evolve?")
    md.append("4. **Protocol Comparison**: Does the Intensity target (`ml_i`) resist noise drift better than Amplitude (`ml_f`), especially in weak and negative reflection regimes?")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Master Drift Summary Table")
    md.append("")
    md.append("| Multiplier | Model | Pos Drift 1D (Å) | MC Drift 3D (Å) | Max Drift (Å) | B RMSD (Å²) | $\\Delta B_{\\text{mean}}$ (Å²) | $r(B, B_{\\text{true}})$ | $R_{\\text{free}}$ (Noisy) | $R_{\\text{true}}$ (Amp) | $CC_{\\text{true}}$ | Bonds (Å) | Angles (°) | $\\Delta\\text{NLL}_{I - F}$ | Win % ($I > F$) |")
    md.append("|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    for s in all_summaries:
        mult = s["multiplier"]
        cm = s["coord_metrics"]
        rs = s["r_stats"]
        tf = s["truth_fit"]
        ivf = s["i_vs_f"]

        tr_c, tr_r, tr_t = cm["true_model"], rs["true_model"], tf["true_model"]
        md.append(f"| **{mult:g}x** | True Model | `0.0000` | `0.0000` | `0.0000` | `0.00` | `+0.00` | `1.0000` | `{tr_r['r_free']*100:.2f}%` | `{tr_t['r_true_amp']*100:.2f}%` | `{tr_t['cc_true_i']:.4f}` | `{tr_r['bond_rms']:.4f}` | `{tr_r['angle_rms']:.2f}` | --- | --- |")

        fc_c, fc_r, fc_t = cm["drifted_f"], rs["drifted_f"], tf["drifted_f"]
        md.append(f"| | `ml_f` (Amp) | `{fc_c['rmsd_1d']:.4f}` | `{fc_c['rmsd_mc']:.4f}` | `{fc_c['max_drift']:.4f}` | `{fc_c['b_rmsd']:.2f}` | `{fc_c['delta_b_mean']:+.2f}` | `{fc_c['b_corr']:.4f}` | `{fc_r['r_free']*100:.2f}%` | `{fc_t['r_true_amp']*100:.2f}%` | `{fc_t['cc_true_i']:.4f}` | `{fc_r['bond_rms']:.4f}` | `{fc_r['angle_rms']:.2f}` | *reference* | --- |")

        ic_c, ic_r, ic_t = cm["drifted_i"], rs["drifted_i"], tf["drifted_i"]
        md.append(f"| | **`ml_i` (Int)** | **`{ic_c['rmsd_1d']:.4f}`** | **`{ic_c['rmsd_mc']:.4f}`** | **`{ic_c['max_drift']:.4f}`** | **`{ic_c['b_rmsd']:.2f}`** | **`{ic_c['delta_b_mean']:+.2f}`** | **`{ic_c['b_corr']:.4f}`** | `{ic_r['r_free']*100:.2f}%` | **`{ic_t['r_true_amp']*100:.2f}%`** | **`{ic_t['cc_true_i']:.4f}`** | `{ic_r['bond_rms']:.4f}` | `{ic_r['angle_rms']:.2f}` | **`{ivf['delta_nll']:+.4f}`** | **`{ivf['win_fraction']*100:.1f}%`** |")

    md.append("")
    md.append("---")
    md.append("")
    md.append("## Detailed Noise Case Analysis")
    md.append("")
    for s in all_summaries:
        mult = s["multiplier"]
        ivf = s["i_vs_f"]
        cm = s["coord_metrics"]
        tf = s["truth_fit"]
        md.append(f"### Noise Multiplier {mult:g}x")
        md.append(f"- **Coordinate Drift**: `ml_i` drifted `{cm['drifted_i']['rmsd_1d']:.4f} Å` (MC: `{cm['drifted_i']['rmsd_mc']:.4f} Å`, Max: `{cm['drifted_i']['max_drift']:.4f} Å`) vs `ml_f` `{cm['drifted_f']['rmsd_1d']:.4f} Å`.")
        md.append(f"- **B-Factor Drift**: B-factor RMSD = `{cm['drifted_i']['b_rmsd']:.2f} Å²`, correlation with true B = `{cm['drifted_i']['b_corr']:.4f}`.")
        cc_val = tf['drifted_i']['cc_true_i']
        r_i_val = tf['drifted_i']['r_true_amp'] * 100.0
        r_tr_val = tf['true_model']['r_true_amp'] * 100.0
        md.append(f"- **Fit to Noise-Free Ground Truth (ITRUE)**: Amplitude R-factor vs truth = `{r_i_val:.2f}%` (true model = `{r_tr_val:.2f}%`). Correlation $CC(I_\\text{{calc}}, I_\\text{{true}}) = {cc_val:.4f}$.")
        md.append(f"- **Held-Out NLL Comparison ($I$ vs $F$)**: `Delta NLL = {ivf['delta_nll']:+.4f} nats/refl` (Bootstrap SE: `{ivf['se_boot']:.4f}`, Win Fraction: `{ivf['win_fraction']*100:.1f}%`).")
        md.append("")

    report_path.write_text("\n".join(md))
    print(f"\nWrote master markdown drift report to: {report_path}")
    return report_path


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Synthetic Student-t noise drift study: refining true 6CZG model against noisy MTZs."
    )
    parser.add_argument("--pdb", default=str(DEFAULT_PDB), help="Reference true PDB file")
    parser.add_argument("--synthetic-dir", default=str(SYNTH_DIR), help="Directory of synthetic MTZ files")
    parser.add_argument(
        "--multipliers",
        type=float,
        nargs="+",
        default=[1.0, 1.5, 2.0, 3.0, 5.0, 10.0],
        help="List of noise multipliers to process (default: 1.0 1.5 2.0 3.0 5.0 10.0)",
    )
    parser.add_argument("--nu", type=float, default=7.0, help="Student-t degrees of freedom (default: 7.0)")
    parser.add_argument("--macrocycles", type=int, default=2, help="Macrocycles of L-BFGS refinement (default: 2)")
    parser.add_argument("--lbfgs-iter", type=int, default=10, help="Max L-BFGS iterations per cycle (default: 10)")
    parser.add_argument("--b-iter", type=int, default=3, help="B-factor Newton iterations per cycle (default: 3)")
    parser.add_argument("--device", default="mps", help="Compute device (mps, cuda, cpu; default: mps)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for bootstrap (default: 42)")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory")
    parser.add_argument("--force-refine", action="store_true", help="Force re-running refinement even if output PDB exists")
    opts = parser.parse_args(args)

    out_dir = Path(opts.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    synth_dir = Path(opts.synthetic_dir)
    true_pdb = Path(opts.pdb)

    print("=" * 85)
    print(" 6CZG SYNTHETIC STUDENT-T DRIFT STUDY (STARTING AT TRUE MODEL)")
    print(f" True Reference PDB: {true_pdb}")
    print(f" Synthetic MTZ Dir:  {synth_dir}")
    print(f" Multipliers:        {opts.multipliers}")
    print(f" Student-t nu:       {opts.nu:.1f}")
    print(f" Refinement:         {opts.macrocycles} macrocycles x ({opts.lbfgs_iter} L-BFGS + {opts.b_iter} B iter)")
    print(f" Device:             {opts.device}")
    print(f" Output Directory:   {out_dir}")
    print("=" * 85)

    bridge = Bridge(memory=True)
    all_summaries: List[Dict[str, Any]] = []

    # Check for existing summary report to reuse already evaluated cases
    existing_summaries: Dict[float, Dict[str, Any]] = {}
    json_path = out_dir / "synthetic_t_drift_report.json"
    if json_path.exists() and not opts.force_refine:
        try:
            prev_data = json.loads(json_path.read_text())
            for item in prev_data:
                existing_summaries[float(item["multiplier"])] = item
        except Exception:
            pass

    for mult in opts.multipliers:
        mult_str = f"{mult:g}".replace(".", "_")
        mtz_name = f"6czg_synthetic_t_mult_{mult_str}.mtz"
        mtz_path = synth_dir / mtz_name
        if not mtz_path.exists():
            print(f"Warning: MTZ file {mtz_path} not found. Skipping multiplier {mult}.")
            continue

        out_f = out_dir / f"6czg_drifted_f_mult_{mult_str}.pdb"
        out_i = out_dir / f"6czg_drifted_i_mult_{mult_str}.pdb"

        f_recomputed = False
        i_recomputed = False

        # Refine Amplitude (ml_f)
        if not out_f.exists() or opts.force_refine:
            run_drift_refinement(
                target="amplitude",
                true_pdb=true_pdb,
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
            print(f"\n[Reusing existing drifted_f model: {out_f.name}]")

        # Refine Intensity (ml_i)
        if not out_i.exists() or opts.force_refine:
            run_drift_refinement(
                target="intensity",
                true_pdb=true_pdb,
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
            print(f"\n[Reusing existing drifted_i model: {out_i.name}]")

        # Evaluate drift metrics (or reuse existing evaluation if neither model was recomputed)
        if not f_recomputed and not i_recomputed and mult in existing_summaries:
            print(f"\n[Reusing existing drift evaluation for multiplier = {mult:g}x]")
            case_summary = existing_summaries[mult]
        else:
            case_summary = evaluate_drift_case(
                multiplier=mult,
                mtz_path=mtz_path,
                true_pdb=true_pdb,
                drifted_f_pdb=out_f,
                drifted_i_pdb=out_i,
                bridge=bridge,
                out_dir=out_dir,
                n_boot=1000,
                seed=opts.seed,
            )
        all_summaries.append(case_summary)

    # Master summary
    print_master_drift_summary(all_summaries)
    save_master_drift_report(all_summaries, out_dir)

    json_path = out_dir / "synthetic_t_drift_report.json"
    json_path.write_text(json.dumps(all_summaries, indent=2))
    print(f"Wrote master JSON drift report to: {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
