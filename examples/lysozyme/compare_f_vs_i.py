#!/usr/bin/env python3
"""Comprehensive comparison of crystallographic refinement against intensities (ml_i)
vs refinement against amplitudes (ml_f) on 1IEE lysozyme.

Compares:
  1. Scale & bulk solvent parameters (k_total, k_sol, B_sol)
  2. Estimated sigma_A across resolution bins
  3. Coordinate refinement progression (NLL reduction & coordinate RMSD)
  4. R-factors (R_work, R_free, R_int)
  5. Gradient-derived 2mFo-DFc and mFo-DFc electron density maps (real-space CC)
"""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torch

from phridge.client.intensity_tool import from_files

HERE = Path(__file__).resolve().parent
PDB_PATH = HERE / "1iee.pdb"
MTZ_PATH = HERE / "1iee.mtz"


def main():
    print("=" * 70)
    print(" Crystallographic Refinement Comparison: Intensities (ml_i) vs Amplitudes (ml_f)")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. Setup Models
    # -------------------------------------------------------------------------
    print("\n--- 1. Initializing Models ---")
    print("Initializing Model I (Intensity Target ml_i)...")
    model_i = from_files(
        pdb_path=PDB_PATH,
        mtz_path=MTZ_PATH,
        d_min=1.5,
        use_bulk_solvent=True,
        overlap_bins=1,
        tv_norm=0.04,
        enforce_monotonic=True,
        target="intensity",
    )

    print("Initializing Model F (Amplitude Target ml_f with French-Wilson amplitudes)...")
    model_f = from_files(
        pdb_path=PDB_PATH,
        mtz_path=MTZ_PATH,
        d_min=1.5,
        use_bulk_solvent=True,
        overlap_bins=1,
        tv_norm=0.04,
        enforce_monotonic=True,
        target="amplitude",
    )

    # -------------------------------------------------------------------------
    # 2. Refine Scale & Bulk Solvent
    # -------------------------------------------------------------------------
    print("\n--- 2. Scale & Bulk Solvent Parameter Refinement ---")
    scales_i = model_i.refine_scale_and_solvent()
    scales_f = model_f.refine_scale_and_solvent()

    print(f"Model I: k_total = {scales_i['k_total']:.4e}, k_sol = {scales_i['k_sol']:.3f}, B_sol = {scales_i['b_sol']:.1f} Å²")
    print(f"Model F: k_total = {scales_f['k_total']:.4e}, k_sol = {scales_f['k_sol']:.3f}, B_sol = {scales_f['b_sol']:.1f} Å²")

    # -------------------------------------------------------------------------
    # 3. Estimate Sigma_A
    # -------------------------------------------------------------------------
    print("\n--- 3. Sigma_A Estimation Across Resolution Bins ---")
    sa_i = model_i.estimate_sigma_a()
    sa_f = model_f.estimate_sigma_a()

    bins = list(model_i.binner.range_used())
    d_spacings = [0.5 * (model_i.binner.bin_d_range(b)[0] + model_i.binner.bin_d_range(b)[1]) for b in bins]

    print(" Bin | Resolution (Å) | Sigma_A (ml_i) | Sigma_A (ml_f) | Difference")
    print(" " + "-" * 62)
    for b in bins:
        d_range = model_i.binner.bin_d_range(b)
        val_i = sa_i[b]
        val_f = sa_f[b]
        diff = val_i - val_f
        print(f" {b:3d} | {d_range[0]:5.2f}-{d_range[1]:5.2f} Å | {val_i:14.4f} | {val_f:14.4f} | {diff:+10.4f}")

    # -------------------------------------------------------------------------
    # 4. Initial Target & Gradients
    # -------------------------------------------------------------------------
    print("\n--- 4. Target Evaluation & Scatterer Gradients ---")
    val_i_init, grads_i = model_i.compute_target_and_gradients()
    val_f_init, grads_f = model_f.compute_target_and_gradients()

    g_cart_i = np.asarray(grads_i.d_target_d_site_cart(), dtype=np.float64)
    g_cart_f = np.asarray(grads_f.d_target_d_site_cart(), dtype=np.float64)

    rms_g_i = np.sqrt(np.mean(np.sum(g_cart_i**2, axis=1)))
    rms_g_f = np.sqrt(np.mean(np.sum(g_cart_f**2, axis=1)))

    print(f"Model I Initial Target NLL: {val_i_init:.4f}, Scatterer Gradient RMS: {rms_g_i:.4e} Å⁻¹")
    print(f"Model F Initial Target NLL: {val_f_init:.4f}, Scatterer Gradient RMS: {rms_g_f:.4e} Å⁻¹")

    # Gradient directional correlation per atom
    dot_prods = np.sum(g_cart_i * g_cart_f, axis=1) / (np.linalg.norm(g_cart_i, axis=1) * np.linalg.norm(g_cart_f, axis=1) + 1e-12)
    print(f"Mean cosine angle between atom gradients (ml_i vs ml_f): {np.mean(dot_prods):.4f}")

    # -------------------------------------------------------------------------
    # 5. Coordinate Refinement (3 steps)
    # -------------------------------------------------------------------------
    print("\n--- 5. Coordinate Refinement (3 steps, step_scale=0.001) ---")
    xyz_init = np.asarray(list(model_i.xray_structure.sites_cart()), dtype=np.float64)

    hist_i = model_i.refine_coordinates(max_iterations=3, step_scale=0.001)
    hist_f = model_f.refine_coordinates(max_iterations=3, step_scale=0.001)

    xyz_i = np.asarray(list(model_i.xray_structure.sites_cart()), dtype=np.float64)
    xyz_f = np.asarray(list(model_f.xray_structure.sites_cart()), dtype=np.float64)

    shift_i = np.sqrt(np.sum((xyz_i - xyz_init)**2, axis=1))
    shift_f = np.sqrt(np.sum((xyz_f - xyz_init)**2, axis=1))
    rmsd_if = np.sqrt(np.mean(np.sum((xyz_i - xyz_f)**2, axis=1)))

    print(f"Model I Refinement NLL History: {[round(x, 4) for x in hist_i]}")
    print(f"Model F Refinement NLL History: {[round(x, 4) for x in hist_f]}")
    print(f"Mean Coordinate Shift: Model I = {np.mean(shift_i):.5f} Å, Model F = {np.mean(shift_f):.5f} Å")
    print(f"RMSD between Model I and Model F coordinates: {rmsd_if:.5f} Å")

    # -------------------------------------------------------------------------
    # 6. R-factors and Model Diagnostics
    # -------------------------------------------------------------------------
    print("\n--- 6. Post-Refinement Statistics & Diagnostics ---")
    s_i = model_i.summary()
    s_f = model_f.summary()

    print(f"Model I: R_work = {s_i['r_work']*100:.2f}%, R_free = {s_i['r_free']*100:.2f}%, R_int = {s_i['r_int']*100:.2f}%")
    print(f"Model F: R_work = {s_f['r_work']*100:.2f}%, R_free = {s_f['r_free']*100:.2f}%, R_int = {s_f['r_int']*100:.2f}%")

    # -------------------------------------------------------------------------
    # 7. Map Synthesis & Real-Space Comparison
    # -------------------------------------------------------------------------
    print("\n--- 7. Map Synthesis & Real-Space Correlation ---")
    grad_i, map2f_i, fofc_i = model_i.compute_map_coefficients(weighted=True, from_gradient=True)
    grad_f, map2f_f, fofc_f = model_f.compute_map_coefficients(weighted=True, from_gradient=True)

    rho_2f_i = map2f_i.fft_map(resolution_factor=0.25).apply_sigma_scaling().real_map_unpadded().as_numpy_array()
    rho_2f_f = map2f_f.fft_map(resolution_factor=0.25).apply_sigma_scaling().real_map_unpadded().as_numpy_array()

    rho_diff_i = fofc_i.fft_map(resolution_factor=0.25).apply_sigma_scaling().real_map_unpadded().as_numpy_array()
    rho_diff_f = fofc_f.fft_map(resolution_factor=0.25).apply_sigma_scaling().real_map_unpadded().as_numpy_array()

    cc_2f = np.corrcoef(rho_2f_i.ravel(), rho_2f_f.ravel())[0, 1]
    cc_diff = np.corrcoef(rho_diff_i.ravel(), rho_diff_f.ravel())[0, 1]

    print(f"Real-space Map Correlation (2mFo - DFc, ml_i vs ml_f): {cc_2f:.4f}")
    print(f"Real-space Map Correlation (mFo - DFc,  ml_i vs ml_f): {cc_diff:.4f}")

    # Write output maps for F refinement
    model_f.write_maps(prefix=str(HERE / "1iee_f_refined"))
    model_f.write_pdb(str(HERE / "1iee_f_refined_model.pdb"))
    print(f"Wrote Model F maps and coordinates to: {HERE / '1iee_f_refined_*'}")

    # -------------------------------------------------------------------------
    # 8. Generate Visual Comparison Figure
    # -------------------------------------------------------------------------
    print("\n--- 8. Generating Visual Comparison Plot ---")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Panel A: Sigma_A comparison across resolution
    ax = axes[0, 0]
    inv_d2 = [1.0 / (d**2) for d in d_spacings]
    ax.plot(inv_d2, [sa_i[b] for b in bins], "o-", color="#2563eb", linewidth=2, label="ml_i (Intensities)")
    ax.plot(inv_d2, [sa_f[b] for b in bins], "s--", color="#dc2626", linewidth=2, label="ml_f (French-Wilson Amplitudes)")
    ax.set_title("A. Sigma_A vs Resolution Shells", fontsize=12, fontweight="bold")
    ax.set_xlabel("1 / d² (Å⁻²)", fontsize=10)
    ax.set_ylabel("sigma_A", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=10)

    # Panel B: Per-atom shift comparison
    ax = axes[0, 1]
    atom_indices = np.arange(len(shift_i))
    ax.plot(atom_indices, shift_i, color="#2563eb", alpha=0.7, linewidth=1.2, label=f"ml_i shift (mean={np.mean(shift_i):.4f} Å)")
    ax.plot(atom_indices, shift_f, color="#dc2626", alpha=0.7, linewidth=1.2, label=f"ml_f shift (mean={np.mean(shift_f):.4f} Å)")
    ax.set_title("B. Atomic Coordinate Shifts per Atom", fontsize=12, fontweight="bold")
    ax.set_xlabel("Atom Index", fontsize=10)
    ax.set_ylabel("Shift Magnitude (Å)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=10)

    # Panel C: Map density correlation scatter (sample 10000 voxels)
    ax = axes[1, 0]
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(rho_2f_i.size, size=8000, replace=False)
    ax.scatter(rho_2f_i.ravel()[sample_idx], rho_2f_f.ravel()[sample_idx], alpha=0.25, s=6, color="#0d9488")
    ax.plot([-3, 14], [-3, 14], "k--", linewidth=1.2, label="y = x")
    ax.set_title(f"C. 2mFo - DFc Density Correlation (CC = {cc_2f:.4f})", fontsize=12, fontweight="bold")
    ax.set_xlabel("ml_i 2mFo - DFc density (sigma)", fontsize=10)
    ax.set_ylabel("ml_f 2mFo - DFc density (sigma)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=10)

    # Panel D: Difference Map density correlation scatter
    ax = axes[1, 1]
    ax.scatter(rho_diff_i.ravel()[sample_idx], rho_diff_f.ravel()[sample_idx], alpha=0.25, s=6, color="#ea580c")
    ax.plot([-6, 8], [-6, 8], "k--", linewidth=1.2, label="y = x")
    ax.set_title(f"D. mFo - DFc Difference Map Correlation (CC = {cc_diff:.4f})", fontsize=12, fontweight="bold")
    ax.set_xlabel("ml_i difference density (sigma)", fontsize=10)
    ax.set_ylabel("ml_f difference density (sigma)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=10)

    plt.tight_layout()
    out_plot = HERE / "compare_f_vs_i.png"
    fig.savefig(out_plot, dpi=200)
    plt.close(fig)
    print(f"Saved comparison figure: {out_plot}")
    print("=" * 70)


if __name__ == "__main__":
    main()
