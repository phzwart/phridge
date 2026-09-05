#!/usr/bin/env python3
"""Comprehensive coordinate perturbation study comparing sigma_A estimation methods:
1. Raw disjoint bins (independent per-bin minimization)
2. Overlapping resolution bins (overlap=1)
3. Total Variation (TV) regularized disjoint bins
4. Overlapping bins + TV regularization
5. Overlapping bins + monotonic (isotonic) enforcement

Saves a high-resolution comparison plot to examples/lysozyme/sigma_a_perturbation_study.png.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cctbx.array_family import flex
from phridge.client.intensity_tool import from_files

LYSOZYME_DIR = Path(__file__).resolve().parent
PDB_PATH = LYSOZYME_DIR / "1iee.pdb"
MTZ_PATH = LYSOZYME_DIR / "1iee.mtz"
PLOT_PATH = LYSOZYME_DIR / "sigma_a_perturbation_study.png"


def run_study():
    print(f"Loading {PDB_PATH} and {MTZ_PATH}...")
    model = from_files(PDB_PATH, MTZ_PATH, d_min=1.5, use_bulk_solvent=True)
    orig_sites = model.xray_structure.sites_cart().deep_copy()

    bins = list(model.binner.range_used())
    n_bins = len(bins)
    d_star_sq_centers = []
    d_ranges = []
    for b in bins:
        d_range = model.binner.bin_d_range(b)
        d_ranges.append(d_range)
        # Midpoint in 1/d^2
        s2_mid = 0.5 * ((1.0 / d_range[0]) ** 2 + (1.0 / d_range[1]) ** 2)
        d_star_sq_centers.append(s2_mid)

    shifts = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2]
    methods = [
        ("disjoint", {"overlap": 0, "tv_lambda": 0.0, "enforce_monotonic": False}),
        ("overlap", {"overlap": 1, "tv_lambda": 0.0, "enforce_monotonic": False}),
        ("tv_only", {"overlap": 0, "tv_lambda": 0.08, "enforce_monotonic": False}),
        ("overlap_tv", {"overlap": 1, "tv_lambda": 0.04, "enforce_monotonic": False}),
        ("overlap_mono", {"overlap": 1, "tv_lambda": 0.0, "enforce_monotonic": True}),
    ]

    results = {m_name: {s: [] for s in shifts} for m_name, _ in methods}

    for shift in shifts:
        print(f"\n--- Testing Coordinate Perturbation dr = {shift:.1f} Å ---")
        if shift > 0.0:
            np.random.seed(42)
            noise = np.random.normal(0.0, shift / np.sqrt(3.0), size=(len(orig_sites), 3))
            model.xray_structure.set_sites_cart(orig_sites + flex.vec3_double(noise.tolist()))
        else:
            model.xray_structure.set_sites_cart(orig_sites)

        # Invalidate cached calculations and fit scale/solvent
        model.f_calc = None
        model.f_mask = None
        model.f_model = None
        model.refine_scale_and_solvent()

        for m_name, m_kwargs in methods:
            sa_dict = model.estimate_sigma_a(
                overlap=m_kwargs["overlap"],
                tv_lambda=m_kwargs["tv_lambda"],
                enforce_monotonic=m_kwargs["enforce_monotonic"],
            )
            vals = [sa_dict[b] for b in bins]
            results[m_name][shift] = vals
            print(f"  {m_name:12s}: " + " ".join(f"{v:6.3f}" for v in vals))

    # Print summary table
    print("\n" + "=" * 90)
    print(" SUMMARY TABLE: Mean Sigma_A vs Perturbation Shift (dr in Å)")
    print("=" * 90)
    header = f"{'Shift (Å)':10s} | " + " | ".join(f"{m_name:>13s}" for m_name, _ in methods)
    print(header)
    print("-" * len(header))
    for s in shifts:
        line = f"{s:10.1f} | " + " | ".join(f"{np.mean(results[m_name][s]):13.4f}" for m_name, _ in methods)
        print(line)
    print("=" * 90)

    # Plot comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=150)

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(shifts)))

    # Panel 1: Disjoint bins (independent)
    ax1 = axes[0, 0]
    for s, c in zip(shifts, colors):
        ax1.plot(
            d_star_sq_centers,
            results["disjoint"][s],
            marker="o",
            label=f"dr = {s:.1f} Å",
            color=c,
            linewidth=1.8,
            markersize=5,
        )
    ax1.set_title("1. Disjoint Resolution Bins (Independent)", fontsize=13, fontweight="bold")
    ax1.set_xlabel(r"Resolution $s^2 = 1/d^2$ (Å$^{-2}$)", fontsize=11)
    ax1.set_ylabel(r"$\sigma_A$", fontsize=12)
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower left", fontsize=8.5, framealpha=0.9)

    # Panel 2: Overlapping bins (overlap=1)
    ax2 = axes[0, 1]
    for s, c in zip(shifts, colors):
        ax2.plot(
            d_star_sq_centers,
            results["overlap"][s],
            marker="s",
            label=f"dr = {s:.1f} Å",
            color=c,
            linewidth=1.8,
            markersize=5,
        )
    ax2.set_title("2. Overlapping Resolution Bins (overlap=1)", fontsize=13, fontweight="bold")
    ax2.set_xlabel(r"Resolution $s^2 = 1/d^2$ (Å$^{-2}$)", fontsize=11)
    ax2.set_ylabel(r"$\sigma_A$", fontsize=12)
    ax2.set_ylim(0.0, 1.05)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower left", fontsize=8.5, framealpha=0.9)

    # Panel 3: Overlapping bins + TV Regularization
    ax3 = axes[1, 0]
    for s, c in zip(shifts, colors):
        ax3.plot(
            d_star_sq_centers,
            results["overlap_tv"][s],
            marker="^",
            label=f"dr = {s:.1f} Å",
            color=c,
            linewidth=1.8,
            markersize=5,
        )
    ax3.set_title(r"3. Overlapping Bins + TV Regularization ($\lambda=0.04$)", fontsize=13, fontweight="bold")
    ax3.set_xlabel(r"Resolution $s^2 = 1/d^2$ (Å$^{-2}$)", fontsize=11)
    ax3.set_ylabel(r"$\sigma_A$", fontsize=12)
    ax3.set_ylim(0.0, 1.05)
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="lower left", fontsize=8.5, framealpha=0.9)

    # Panel 4: Method comparison at dr = 0.5 Å and 0.8 Å
    ax4 = axes[1, 1]
    method_styles = [
        ("disjoint", "Disjoint", "o--", "#e74c3c", 1.8),
        ("tv_only", "TV Disjoint", "x-.", "#9b59b6", 1.8),
        ("overlap", "Overlapping", "s-", "#2ecc71", 2.0),
        ("overlap_tv", "Overlap + TV", "^-", "#3498db", 2.2),
        ("overlap_mono", "Overlap + Monotonic", "D:", "#f39c12", 2.0),
    ]

    for m_key, m_label, m_fmt, m_col, m_lw in method_styles:
        ax4.plot(
            d_star_sq_centers,
            results[m_key][0.5],
            m_fmt,
            label=f"{m_label} (dr=0.5Å)",
            color=m_col,
            linewidth=m_lw,
            markersize=5,
        )

    for m_key, m_label, m_fmt, m_col, m_lw in [
        ("disjoint", "Disjoint", "o--", "#c0392b", 1.5),
        ("overlap_tv", "Overlap + TV", "^-", "#2980b9", 2.0),
    ]:
        ax4.plot(
            d_star_sq_centers,
            results[m_key][0.8],
            m_fmt,
            label=f"{m_label} (dr=0.8Å)",
            color=m_col,
            alpha=0.6,
            linewidth=m_lw,
            markersize=4,
        )

    ax4.set_title(r"4. Method Comparison at $\Delta r = 0.5$ Å & $0.8$ Å", fontsize=13, fontweight="bold")
    ax4.set_xlabel(r"Resolution $s^2 = 1/d^2$ (Å$^{-2}$)", fontsize=11)
    ax4.set_ylabel(r"$\sigma_A$", fontsize=12)
    ax4.set_ylim(0.0, 1.05)
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.legend(loc="upper right", fontsize=8, framealpha=0.9)

    plt.suptitle(
        r"$\sigma_A$ Estimation Under Coordinate Perturbations: Disjoint vs. Overlapping Bins vs. Total Variation",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    print(f"\nSaved comparison plot to: {PLOT_PATH}")


if __name__ == "__main__":
    run_study()
