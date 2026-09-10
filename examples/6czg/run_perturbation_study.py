#!/usr/bin/env python3
"""Systematic Coordinate Shift Perturbation Study on 6CZG.

Evaluates how Amplitude-based (ml_f) vs Intensity-based (ml_i) crystallographic likelihood
functions and their gradients respond to synthetic coordinate error.

Generates 50 perturbed models for each RMSD noise level (0.05, 0.10, 0.25, and 0.50 Å):
  - Normal noise scaled to exact Cartesian RMSD
  - Perturbed models written as individual PDB files in a separate directory
  - Evaluates both F and I likelihood functions (working NLL, free NLL, R-factors, CC)
  - Evaluates Cartesian gradient norm and restoring cosine similarity (fidelity to ground truth)
  - Evaluates stereochemical geometry restraint energy (CCTBX)
  - Exports full results to CSV, JSON, summary tables, and publication-quality diagnostic plots

Usage:
    # Run quick test with 2 samples per level
    python examples/6czg/run_perturbation_study.py --n-samples 2

    # Run full study with 50 samples per level (200 models total)
    python examples/6czg/run_perturbation_study.py --n-samples 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cctbx  # noqa: F401
from cctbx.array_family import flex
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from phridge.client.intensity_tool import from_files

DIR_6CZG = Path(__file__).resolve().parent
PDB_FILE = DIR_6CZG / "6czg.pdb"
MTZ_FILE = DIR_6CZG / "6czg.mtz"


def generate_perturbed_coords(
    sites_orig: np.ndarray,
    target_rmsd: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Generate perturbed Cartesian coordinates with exact target RMSD."""
    noise_raw = rng.normal(loc=0.0, scale=1.0, size=sites_orig.shape)
    current_rmsd = float(np.sqrt(np.mean(noise_raw**2)))
    noise = noise_raw * (target_rmsd / max(current_rmsd, 1e-12))
    actual_rmsd = float(np.sqrt(np.mean(noise**2)))
    sites_perturbed = sites_orig + noise
    return sites_perturbed, noise, actual_rmsd


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Systematic coordinate perturbation study for F- and I-based crystallographic likelihood.",
    )
    parser.add_argument("--pdb", default=str(PDB_FILE), help="Reference PDB file")
    parser.add_argument("--mtz", default=str(MTZ_FILE), help="Experimental MTZ reflection file")
    parser.add_argument("--d-min", type=float, default=2.2, help="High resolution limit in Å (default: 2.2)")
    parser.add_argument("--device", default="auto", help="Compute device (auto, mps, cuda, cpu)")
    parser.add_argument(
        "--rmsd-levels",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.25, 0.50],
        help="Target coordinate shift RMSDs in Å (default: 0.05 0.10 0.25 0.50)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="Number of perturbed models to generate per RMSD level (default: 50)",
    )
    parser.add_argument(
        "--models-dir",
        default=str(DIR_6CZG / "perturbed_models"),
        help="Directory to save perturbed PDB models",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DIR_6CZG),
        help="Directory to save analysis CSV, JSON, and PNG files",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
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
        help="Perturb existing B-factors without resetting to mean B",
    )
    parser.add_argument(
        "--no-shake-b",
        action="store_true",
        default=False,
        help="Disable B-factor perturbation (only shake coordinates)",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip generating comparison plot")

    opts = parser.parse_args(args)

    models_dir = Path(opts.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(opts.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" SYSTEMATIC COORDINATE PERTURBATION STUDY (6CZG)")
    print(f" PDB: {opts.pdb}")
    print(f" MTZ: {opts.mtz}")
    print(f" RMSD Levels: {opts.rmsd_levels} Å")
    print(f" Samples per level: {opts.n_samples} (Total models: {len(opts.rmsd_levels) * opts.n_samples})")
    print(f" Models output directory: {models_dir}")
    print("=" * 80)

    # 1. Initialize reference model
    print("\n--- 1. Initializing and Scaling Reference Model ---")
    t0 = time.time()
    model = from_files(
        pdb_path=opts.pdb,
        mtz_path=opts.mtz,
        d_min=opts.d_min,
        device=opts.device,
        use_bulk_solvent=True,
        convert_to_isotropic=True,
        target="intensity",
        estimate_nu=True,
    )
    print(f"  Initialized model in {time.time() - t0:.2f}s on device: {model.device}")

    # Scale and bulk solvent on reference model
    scales = model.refine_scale_and_solvent()
    sa = model.estimate_sigma_a()
    model.estimate_nu()
    summ_ref = model.summary()

    # Pre-compute French-Wilson amplitudes once so it's cached for all amplitude target evals
    model.get_f_obs()

    # Baseline reference evaluations
    model.target_type = "intensity"
    ref_nll_i_work, ref_grads_i = model.compute_target_and_gradients()
    ref_nll_i_free = model.target_value_test

    model.target_type = "amplitude"
    ref_nll_f_work, ref_grads_f = model.compute_target_and_gradients()
    ref_nll_f_free = model.target_value_test

    sites_orig = np.asarray(model.xray_structure.sites_cart(), dtype=np.float64).copy()
    ref_geom_e, _ = model.compute_geometry_energy_and_gradients(sites_orig)

    uc = model.xray_structure.unit_cell()
    fract = np.asarray(uc.fractionalization_matrix(), dtype=np.float64).reshape(3, 3)

    print(f"  Reference Model Metrics:")
    print(f"    R_work = {summ_ref['r_work']*100:.2f}%, R_free = {summ_ref['r_free']*100:.2f}%")
    print(f"    CC_work(I) = {summ_ref['cc_work_i']:.4f}, CC_free(I) = {summ_ref['cc_free_i']:.4f}")
    print(f"    NLL_I (work/free) = {ref_nll_i_work:.4f} / {ref_nll_i_free:.4f}")
    print(f"    NLL_F (work/free) = {ref_nll_f_work:.4f} / {ref_nll_f_free:.4f}")
    print(f"    Geom Energy = {ref_geom_e:.2f}")

    rng = np.random.default_rng(opts.seed)

    records: List[Dict[str, Any]] = []
    total_models = len(opts.rmsd_levels) * opts.n_samples
    model_counter = 0

    print(f"\n--- 2. Generating & Evaluating {total_models} Perturbed Models ---")
    t_start_loop = time.time()

    for rmsd_target in opts.rmsd_levels:
        print(f"\n>> Target RMSD: {rmsd_target:.2f} Å ({opts.n_samples} replicates)")
        for rep in range(opts.n_samples):
            model_counter += 1
            t_model = time.time()

            # Generate perturbed coordinates
            sites_pert, noise, actual_rmsd = generate_perturbed_coords(sites_orig, rmsd_target, rng)

            # Update coordinates in model
            sites_frac = sites_pert @ fract.T
            model.xray_structure.set_sites_frac(flex.vec3_double([tuple(r) for r in sites_frac]))

            # Reset and shake B-factors if enabled (+/- 25% fractional)
            if not opts.no_shake_b and opts.b_shake_fraction > 0:
                b_shaken = model.shake_b_iso(
                    fraction=opts.b_shake_fraction,
                    reset=opts.reset_b,
                    rng=rng,
                    update_model=False,
                )
            else:
                b_shaken = np.array([float(sc.u_iso * 8.0 * np.pi**2) for sc in model.xray_structure.scatterers()])

            if model.hierarchy is not None:
                model.hierarchy.adopt_xray_structure(model.xray_structure)

            # Save perturbed model PDB
            pdb_filename = f"6czg_rmsd_{rmsd_target:0.2f}_rep_{rep:02d}.pdb"
            pdb_path = models_dir / pdb_filename
            model.write_pdb(str(pdb_path))

            # Recalculate F_calc and F_model
            model.compute_f_calc()
            fmod_complex = model._assemble_f_model(model.k_total, model.k_sol, model.b_sol)
            model.f_model = model.i_obs.customized_copy(data=flex.complex_double(fmod_complex.tolist()))

            # 1. Evaluate Intensity target (ml_i)
            model.target_type = "intensity"
            model.compute_target_and_gradients()
            nll_i_work = float(model.target_value)
            nll_i_free = float(model.target_value_test)
            g_cart_i = np.asarray(model.gradients.d_target_d_site_cart(), dtype=np.float64)
            grad_norm_i = float(np.linalg.norm(g_cart_i))

            # Restoring cosine similarity: (desc_dir . v_restore) = (g_cart . noise)
            norm_noise = float(np.linalg.norm(noise))
            cos_sim_i = float(np.sum(g_cart_i * noise) / max(grad_norm_i * norm_noise, 1e-12))

            # 2. Evaluate Amplitude target (ml_f)
            model.target_type = "amplitude"
            model.compute_target_and_gradients()
            nll_f_work = float(model.target_value)
            nll_f_free = float(model.target_value_test)
            g_cart_f = np.asarray(model.gradients.d_target_d_site_cart(), dtype=np.float64)
            grad_norm_f = float(np.linalg.norm(g_cart_f))
            cos_sim_f = float(np.sum(g_cart_f * noise) / max(grad_norm_f * norm_noise, 1e-12))

            # 3. Crystallographic summary metrics
            summ = model.summary()

            # 4. CCTBX stereochemical geometry energy
            e_geom, _ = model.compute_geometry_energy_and_gradients(sites_pert)

            rec: Dict[str, Any] = {
                "rmsd_target": float(rmsd_target),
                "rmsd_actual": float(actual_rmsd),
                "rep": int(rep),
                "pdb_file": pdb_filename,
                "nll_i_work": nll_i_work,
                "nll_i_free": nll_i_free,
                "delta_nll_i_work": float(nll_i_work - ref_nll_i_work),
                "delta_nll_i_free": float(nll_i_free - ref_nll_i_free),
                "nll_f_work": nll_f_work,
                "nll_f_free": nll_f_free,
                "delta_nll_f_work": float(nll_f_work - ref_nll_f_work),
                "delta_nll_f_free": float(nll_f_free - ref_nll_f_free),
                "r_work": float(summ["r_work"]),
                "r_free": float(summ["r_free"]),
                "r_intensity_work": float(summ["r_intensity"]),
                "r_intensity_free": float(summ["r_intensity_free"]),
                "cc_work_i": float(summ["cc_work_i"]),
                "cc_free_i": float(summ["cc_free_i"]),
                "geom_energy": float(e_geom),
                "delta_geom_energy": float(e_geom - ref_geom_e),
                "grad_norm_i": grad_norm_i,
                "grad_norm_f": grad_norm_f,
                "cos_sim_i": cos_sim_i,
                "cos_sim_f": cos_sim_f,
                "b_mean": float(np.mean(b_shaken)),
                "b_std": float(np.std(b_shaken)),
                "b_min": float(np.min(b_shaken)),
                "b_max": float(np.max(b_shaken)),
            }
            records.append(rec)

            if (rep + 1) % 10 == 0 or rep == 0 or (rep + 1) == opts.n_samples:
                elapsed_total = time.time() - t_start_loop
                rate = elapsed_total / model_counter
                eta = rate * (total_models - model_counter)
                print(
                    f"  [{model_counter:3d}/{total_models:3d}] RMSD {rmsd_target:.2f} Å (rep {rep+1:2d}/{opts.n_samples}): "
                    f"R_free={rec['r_free']*100:.2f}%, CC_free={rec['cc_free_i']:.4f}, "
                    f"cos_sim_I={cos_sim_i:.3f}, cos_sim_F={cos_sim_f:.3f} | "
                    f"ETA: {eta:.0f}s"
                )

    # Reset model to original sites
    sites_orig_frac = sites_orig @ fract.T
    model.xray_structure.set_sites_frac(flex.vec3_double([tuple(r) for r in sites_orig_frac]))

    print(f"\nCompleted {total_models} models in {time.time() - t_start_loop:.1f}s!")

    # 3. Export CSV Results
    print("\n--- 3. Saving Output Results for Downstream Analysis ---")
    csv_file = out_dir / "perturbation_results.csv"
    import csv
    fieldnames = list(records[0].keys())
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"  Wrote full per-model dataset: {csv_file} ({len(records)} rows)")

    # 4. Compute Statistical Summary Table
    summary_by_rmsd = {}
    for rmsd in opts.rmsd_levels:
        group = [r for r in records if np.isclose(r["rmsd_target"], rmsd)]
        if not group:
            continue
        stats: Dict[str, Any] = {"rmsd": rmsd, "n": len(group)}
        metrics = [
            "r_work", "r_free", "r_intensity_free", "cc_free_i",
            "delta_nll_i_free", "delta_nll_f_free",
            "cos_sim_i", "cos_sim_f", "geom_energy",
            "grad_norm_i", "grad_norm_f",
        ]
        for m in metrics:
            vals = [g[m] for g in group if np.isfinite(g[m])]
            stats[f"{m}_mean"] = float(np.mean(vals))
            stats[f"{m}_std"] = float(np.std(vals))
            stats[f"{m}_median"] = float(np.median(vals))
        summary_by_rmsd[rmsd] = stats

    summary_csv = out_dir / "perturbation_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        sum_fields = list(next(iter(summary_by_rmsd.values())).keys())
        writer = csv.DictWriter(f, fieldnames=sum_fields)
        writer.writeheader()
        writer.writerows(summary_by_rmsd.values())
    print(f"  Wrote aggregated summary: {summary_csv}")

    # 5. Export Structured JSON
    json_file = out_dir / "perturbation_results.json"
    full_export = {
        "metadata": {
            "pdb": opts.pdb,
            "mtz": opts.mtz,
            "d_min": opts.d_min,
            "device": model.device,
            "rmsd_levels": opts.rmsd_levels,
            "n_samples_per_level": opts.n_samples,
            "total_models": total_models,
            "models_dir": str(models_dir),
            "reference": {
                "r_work": summ_ref["r_work"],
                "r_free": summ_ref["r_free"],
                "cc_work_i": summ_ref["cc_work_i"],
                "cc_free_i": summ_ref["cc_free_i"],
                "nll_i_work": ref_nll_i_work,
                "nll_i_free": ref_nll_i_free,
                "nll_f_work": ref_nll_f_work,
                "nll_f_free": ref_nll_f_free,
                "geom_energy": ref_geom_e,
                "k_total": model.k_total,
                "k_sol": model.k_sol,
                "b_sol": model.b_sol,
            },
        },
        "summary": summary_by_rmsd,
        "records": records,
    }
    with open(json_file, "w") as f:
        json.dump(full_export, f, indent=2)
    print(f"  Wrote JSON export: {json_file}")

    # 6. Print Console Summary Table
    print_console_summary(summary_by_rmsd, summ_ref, ref_geom_e)

    # 7. Generate Diagnostic Plot
    if not opts.no_plot:
        plot_file = out_dir / "perturbation_analysis.png"
        generate_plots(records, summary_by_rmsd, plot_file)

    return 0


def print_console_summary(summary_by_rmsd: Dict[float, Dict[str, Any]], summ_ref: Dict[str, Any], ref_geom_e: float) -> None:
    print("\n" + "=" * 110)
    print(" SYSTEMATIC COORDINATE PERTURBATION SUMMARY TABLE (6CZG)")
    print("=" * 110)
    hdr = f"{'Metric':<30} | {'Reference':<14} | " + " | ".join(f"RMSD {rmsd:.2f} Å (n={d['n']})" for rmsd, d in summary_by_rmsd.items())
    print(hdr)
    print("-" * len(hdr))

    rows = [
        ("R_work (%)", f"{summ_ref['r_work']*100:.2f}%", [f"{d['r_work_mean']*100:.2f} ± {d['r_work_std']*100:.2f}%" for d in summary_by_rmsd.values()]),
        ("R_free (%)", f"{summ_ref['r_free']*100:.2f}%", [f"{d['r_free_mean']*100:.2f} ± {d['r_free_std']*100:.2f}%" for d in summary_by_rmsd.values()]),
        ("CC_free (I)", f"{summ_ref['cc_free_i']:.4f}", [f"{d['cc_free_i_mean']:.4f} ± {d['cc_free_i_std']:.4f}" for d in summary_by_rmsd.values()]),
        ("R_intensity (free) (%)", f"{summ_ref['r_intensity_free']*100:.2f}%", [f"{d['r_intensity_free_mean']*100:.2f} ± {d['r_intensity_free_std']*100:.2f}%" for d in summary_by_rmsd.values()]),
        ("Δ NLL_free (Intensity)", "0.0000", [f"{d['delta_nll_i_free_mean']:+.4f} ± {d['delta_nll_i_free_std']:.4f}" for d in summary_by_rmsd.values()]),
        ("Δ NLL_free (Amplitude)", "0.0000", [f"{d['delta_nll_f_free_mean']:+.4f} ± {d['delta_nll_f_free_std']:.4f}" for d in summary_by_rmsd.values()]),
        ("Cos Sim: Intensity Grad", "N/A", [f"{d['cos_sim_i_mean']:.4f} ± {d['cos_sim_i_std']:.4f}" for d in summary_by_rmsd.values()]),
        ("Cos Sim: Amplitude Grad", "N/A", [f"{d['cos_sim_f_mean']:.4f} ± {d['cos_sim_f_std']:.4f}" for d in summary_by_rmsd.values()]),
        ("Geom Restraint Energy", f"{ref_geom_e:.1f}", [f"{d['geom_energy_mean']:.1f} ± {d['geom_energy_std']:.1f}" for d in summary_by_rmsd.values()]),
        ("Norm: Intensity Grad", "N/A", [f"{d['grad_norm_i_mean']:.3f} ± {d['grad_norm_i_std']:.3f}" for d in summary_by_rmsd.values()]),
        ("Norm: Amplitude Grad", "N/A", [f"{d['grad_norm_f_mean']:.3f} ± {d['grad_norm_f_std']:.3f}" for d in summary_by_rmsd.values()]),
    ]

    for label, ref_val, vals in rows:
        val_str = " | ".join(f"{v:<18}" for v in vals)
        print(f"{label:<30} | {ref_val:<14} | {val_str}")
    print("=" * 110 + "\n")


def generate_plots(records: List[Dict[str, Any]], summary_by_rmsd: Dict[float, Dict[str, Any]], out_png: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("6CZG Systematic Coordinate Shift Perturbation Analysis", fontsize=15, fontweight="bold")

    rmsds = sorted(summary_by_rmsd.keys())

    # Panel 1: Target NLL response (Delta NLL_free)
    ax = axes[0, 0]
    nll_i_means = [summary_by_rmsd[r]["delta_nll_i_free_mean"] for r in rmsds]
    nll_i_stds = [summary_by_rmsd[r]["delta_nll_i_free_std"] for r in rmsds]
    nll_f_means = [summary_by_rmsd[r]["delta_nll_f_free_mean"] for r in rmsds]
    nll_f_stds = [summary_by_rmsd[r]["delta_nll_f_free_std"] for r in rmsds]

    ax.errorbar(rmsds, nll_i_means, yerr=nll_i_stds, marker="o", color="#2ca02c", lw=2, capsize=4, label="Intensity Target (ml_i)")
    ax.errorbar(rmsds, nll_f_means, yerr=nll_f_stds, marker="s", color="#1f77b4", lw=2, capsize=4, label="Amplitude Target (ml_f)")
    ax.set_title("Cross-Validation Likelihood Penalty (ΔNLL_free)", fontweight="bold")
    ax.set_xlabel("Coordinate Shift RMSD (Å)")
    ax.set_ylabel("Δ NLL_free relative to reference")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)

    # Panel 2: Free-set cross-validation metrics (R_free and CC_free)
    ax = axes[0, 1]
    cc_means = [summary_by_rmsd[r]["cc_free_i_mean"] for r in rmsds]
    cc_stds = [summary_by_rmsd[r]["cc_free_i_std"] for r in rmsds]
    rfree_means = [summary_by_rmsd[r]["r_free_mean"] * 100.0 for r in rmsds]
    rfree_stds = [summary_by_rmsd[r]["r_free_std"] * 100.0 for r in rmsds]

    color_cc = "#2ecc71"
    ax.errorbar(rmsds, cc_means, yerr=cc_stds, marker="^", color=color_cc, lw=2, capsize=4, label="CC_free(I) [higher=better]")
    ax.set_xlabel("Coordinate Shift RMSD (Å)")
    ax.set_ylabel("CC_free(I)", color=color_cc)
    ax.tick_params(axis="y", labelcolor=color_cc)
    ax.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax.twinx()
    color_rf = "#e74c3c"
    ax2.errorbar(rmsds, rfree_means, yerr=rfree_stds, marker="v", color=color_rf, lw=2, capsize=4, linestyle="--", label="R_free (%) [lower=better]")
    ax2.set_ylabel("R_free (%)", color=color_rf)
    ax2.tick_params(axis="y", labelcolor=color_rf)
    ax.set_title("Cross-Validation Metrics vs Perturbation RMSD", fontweight="bold")

    # Panel 3: Restoring Gradient Fidelity (Cosine Similarity)
    ax = axes[1, 0]
    cos_i_means = [summary_by_rmsd[r]["cos_sim_i_mean"] for r in rmsds]
    cos_i_stds = [summary_by_rmsd[r]["cos_sim_i_std"] for r in rmsds]
    cos_f_means = [summary_by_rmsd[r]["cos_sim_f_mean"] for r in rmsds]
    cos_f_stds = [summary_by_rmsd[r]["cos_sim_f_std"] for r in rmsds]

    ax.errorbar(rmsds, cos_i_means, yerr=cos_i_stds, marker="o", color="#2ca02c", lw=2, capsize=4, label="Intensity Likelihood (ml_i)")
    ax.errorbar(rmsds, cos_f_means, yerr=cos_f_stds, marker="s", color="#1f77b4", lw=2, capsize=4, label="Amplitude Likelihood (ml_f)")
    ax.axhline(0.0, color="gray", linestyle=":", alpha=0.7)
    ax.set_title("Gradient Restoring Direction Fidelity (cos θ)", fontweight="bold")
    ax.set_xlabel("Coordinate Shift RMSD (Å)")
    ax.set_ylabel("Cosine similarity to true displacement (-Δx)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)

    # Panel 4: Stereochemical Geometry Restraint Energy
    ax = axes[1, 1]
    geom_means = [summary_by_rmsd[r]["geom_energy_mean"] for r in rmsds]
    geom_stds = [summary_by_rmsd[r]["geom_energy_std"] for r in rmsds]

    ax.errorbar(rmsds, geom_means, yerr=geom_stds, marker="D", color="#9b59b6", lw=2, capsize=4, label="CCTBX Restraint Target")
    ax.set_title("Stereochemical Restraint Energy vs RMSD", fontweight="bold")
    ax.set_xlabel("Coordinate Shift RMSD (Å)")
    ax.set_ylabel("CCTBX Geometry Restraint Target Energy")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"  Wrote diagnostic figure: {out_png}")


if __name__ == "__main__":
    sys.exit(main())
