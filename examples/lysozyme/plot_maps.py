#!/usr/bin/env python3
"""Generate high-resolution visual comparisons of the intensity gradient map,
sigma_A weighted difference map (mFo-DFc), and 2mFo-DFc map for lysozyme (1IEE).
"""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

import iotbx.ccp4_map
import iotbx.pdb

LYSOZYME_DIR = Path(__file__).resolve().parent
GRAD_PATH = LYSOZYME_DIR / "1iee_out_gradient.ccp4"
FOFC_PATH = LYSOZYME_DIR / "1iee_out_fofc.ccp4"
TWOFOFC_PATH = LYSOZYME_DIR / "1iee_out_2fofc.ccp4"
PDB_PATH = LYSOZYME_DIR / "1iee.pdb"


def load_maps():
    m_grad = iotbx.ccp4_map.map_reader(str(GRAD_PATH))
    m_fofc = iotbx.ccp4_map.map_reader(str(FOFC_PATH))
    m_2fofc = iotbx.ccp4_map.map_reader(str(TWOFOFC_PATH))

    d_grad = m_grad.data.as_numpy_array()
    d_fofc = m_fofc.data.as_numpy_array()
    d_2fofc = m_2fofc.data.as_numpy_array()
    uc = m_grad.unit_cell()
    return d_grad, d_fofc, d_2fofc, uc


def plot_map_comparisons(d_grad, d_fofc, d_2fofc, uc, z_idx=63):
    """Plot side-by-side comparison of the three maps along XY plane at z_idx."""
    a, b, c = uc.parameters()[:3]
    grid_u, grid_v, grid_w = d_grad.shape
    z_angstrom = (z_idx / grid_w) * c

    # Extract 2D slices (X vs Y)
    # in cctbx, indices are (u, v, w) -> (x, y, z)
    slice_grad = d_grad[:, :, z_idx].T
    slice_fofc = d_fofc[:, :, z_idx].T
    slice_2fofc = d_2fofc[:, :, z_idx].T

    extent = [0, a, 0, b]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.15)

    # 1. Gradient Map (-0.5 * dQ/dF_model)
    vmax_diff = max(abs(slice_grad.min()), abs(slice_grad.max()), 5.0)
    norm_grad = TwoSlopeNorm(vmin=-vmax_diff, vcenter=0.0, vmax=vmax_diff)
    im0 = axes[0].imshow(
        slice_grad,
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        norm=norm_grad,
    )
    axes[0].set_title(r"$\mathbf{Intensity\ Target\ Gradient\ Map}$" + f"\n$F_{{grad}} = -\\frac{{1}}{{2}} \\frac{{\\partial Q}}{{\\partial F_{{model}}}}$ (Z = {z_angstrom:.1f} Å)", fontsize=13)
    axes[0].set_xlabel("X (Å)", fontsize=11)
    axes[0].set_ylabel("Y (Å)", fontsize=11)
    cb0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cb0.set_label("Gradient Density (σ)", fontsize=10)
    # Add contour lines: +3σ (green) and -3σ (red)
    axes[0].contour(slice_grad, levels=[3.0], extent=extent, colors=["green"], linewidths=1.2)
    axes[0].contour(slice_grad, levels=[-3.0], extent=extent, colors=["red"], linestyles="dashed", linewidths=1.2)

    # 2. mFo - DFc Difference Map
    vmax_fofc = max(abs(slice_fofc.min()), abs(slice_fofc.max()), 5.0)
    norm_fofc = TwoSlopeNorm(vmin=-vmax_fofc, vcenter=0.0, vmax=vmax_fofc)
    im1 = axes[1].imshow(
        slice_fofc,
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        norm=norm_fofc,
    )
    axes[1].set_title(r"$\mathbf{\sigma_A-Weighted\ Difference\ Map}$" + f"\n$mF_o - DF_c$ (Z = {z_angstrom:.1f} Å)", fontsize=13)
    axes[1].set_xlabel("X (Å)", fontsize=11)
    cb1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cb1.set_label("Difference Density (σ)", fontsize=10)
    axes[1].contour(slice_fofc, levels=[3.0], extent=extent, colors=["green"], linewidths=1.2)
    axes[1].contour(slice_fofc, levels=[-3.0], extent=extent, colors=["red"], linestyles="dashed", linewidths=1.2)

    # 3. 2mFo - DFc Map
    im2 = axes[2].imshow(
        slice_2fofc,
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=-1.0,
        vmax=6.0,
    )
    axes[2].set_title(r"$\mathbf{2mF_o - DF_c\ Electron\ Density}$" + f"\n(Z = {z_angstrom:.1f} Å)", fontsize=13)
    axes[2].set_xlabel("X (Å)", fontsize=11)
    cb2 = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    cb2.set_label("Electron Density (σ)", fontsize=10)
    axes[2].contour(slice_2fofc, levels=[1.0, 2.5, 4.0], extent=extent, colors=["white", "cyan", "yellow"], linewidths=0.9, alpha=0.8)

    out_file = LYSOZYME_DIR / "gradient_vs_difference_maps.png"
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")
    return out_file


def plot_active_site_zoom(d_grad, d_fofc, d_2fofc, uc):
    """Detailed zoom around catalytic residues Glu 35 & Asp 52."""
    pdb_inp = iotbx.pdb.input(str(PDB_PATH))
    hier = pdb_inp.construct_hierarchy()
    grid_u, grid_v, grid_w = d_grad.shape
    a, b, c = uc.parameters()[:3]

    # Asp 52 OD1 is at ~ (7.6, 19.5, 23.6) Å; Glu 35 is nearby at ~ (4.9, 23.0, 18.3) Å
    # Let's slice at z = 23.5 Å (w_idx = 63)
    w_idx = 63
    z_val = (w_idx / grid_w) * c

    # Region of interest: X in [0, 25] Å, Y in [10, 35] Å
    x_min, x_max = 0.0, 25.0
    y_min, y_max = 10.0, 35.0

    u_min, u_max = int((x_min / a) * grid_u), int((x_max / a) * grid_u)
    v_min, v_max = int((y_min / b) * grid_v), int((y_max / b) * grid_v)

    zoom_grad = d_grad[u_min:u_max, v_min:v_max, w_idx].T
    zoom_fofc = d_fofc[u_min:u_max, v_min:v_max, w_idx].T
    zoom_2fofc = d_2fofc[u_min:u_max, v_min:v_max, w_idx].T
    extent = [x_min, x_max, y_min, y_max]

    # Collect nearby atoms within 2.0 Å of the slice
    nearby_atoms = []
    for rg in hier.models()[0].chains()[0].residue_groups():
        resseq = rg.resseq.strip()
        for ag in rg.atom_groups():
            resname = ag.resname.strip()
            for atom in ag.atoms():
                x, y, z = atom.xyz
                if x_min <= x <= x_max and y_min <= y <= y_max and abs(z - z_val) <= 2.5:
                    nearby_atoms.append((resname, resseq, atom.name.strip(), x, y, z))

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Panel 1: Gradient Map with Contours & Atoms
    vmax = max(abs(zoom_grad.min()), abs(zoom_grad.max()), 4.0)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im0 = axes[0].imshow(zoom_grad, origin="lower", extent=extent, cmap="PuOr", norm=norm)
    axes[0].contour(zoom_grad, levels=[2.0, 3.5], extent=extent, colors=["#2ca02c", "#1b7837"], linewidths=1.5)
    axes[0].contour(zoom_grad, levels=[-3.5, -2.0], extent=extent, colors=["#b2182b", "#d6604d"], linestyles="dashed", linewidths=1.5)

    axes[0].set_title(r"$\mathbf{Intensity\ Target\ Gradient\ Map\ (Active\ Site\ Zoom)}$" + f"\nSlice at Z = {z_val:.1f} Å (Green: +2σ, +3.5σ; Red: -2σ, -3.5σ)", fontsize=11)
    axes[0].set_xlabel("X (Å)", fontsize=11)
    axes[0].set_ylabel("Y (Å)", fontsize=11)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Gradient (σ)")

    # Overlay atoms
    labeled_residues = set()
    for resname, resseq, aname, x, y, z in nearby_atoms:
        is_key = resname in ["GLU", "ASP", "TRP"] and resseq in ["35", "52", "62", "108"]
        color = "black" if not is_key else "blue"
        marker = "o" if not is_key else "s"
        size = 20 if not is_key else 50
        axes[0].scatter(x, y, color=color, s=size, marker=marker, edgecolors="white", linewidth=0.5, zorder=5)

        key_id = f"{resname}{resseq}"
        if is_key and key_id not in labeled_residues and aname in ["CA", "OE1", "OD1"]:
            axes[0].annotate(
                f"{resname} {resseq}",
                (x, y),
                xytext=(x + 0.6, y + 0.6),
                fontsize=10,
                fontweight="bold",
                color="navy",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"),
                zorder=6,
            )
            labeled_residues.add(key_id)

    # Panel 2: 2mFo-DFc Density with Gradient Contours Overlaid
    im1 = axes[1].imshow(zoom_2fofc, origin="lower", extent=extent, cmap="Greys_r", vmin=-0.5, vmax=5.0)
    # Overlay positive & negative gradient contours
    cs_pos = axes[1].contour(zoom_grad, levels=[2.0, 3.5], extent=extent, colors=["#00ff00", "#00aa00"], linewidths=1.5)
    cs_neg = axes[1].contour(zoom_grad, levels=[-3.5, -2.0], extent=extent, colors=["#ff3333", "#cc0000"], linestyles="dashed", linewidths=1.5)
    axes[1].contour(zoom_2fofc, levels=[1.0, 2.5], extent=extent, colors=["yellow", "orange"], linewidths=1.0, alpha=0.7)

    axes[1].set_title(r"$\mathbf{2mF_o - DF_c\ Density\ with\ Gradient\ Contours}$" + "\n(Yellow/Orange: 2mFo-DFc; Green: +Grad; Red: -Grad)", fontsize=11)
    axes[1].set_xlabel("X (Å)", fontsize=11)
    axes[1].set_ylabel("Y (Å)", fontsize=11)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="2mFo-DFc (σ)")

    # Overlay atoms
    for resname, resseq, aname, x, y, z in nearby_atoms:
        is_key = resname in ["GLU", "ASP", "TRP"] and resseq in ["35", "52", "62", "108"]
        color = "white" if not is_key else "cyan"
        axes[1].scatter(x, y, color=color, s=25 if not is_key else 55, marker="o", edgecolors="black", linewidth=0.5, zorder=5)

    out_file = LYSOZYME_DIR / "active_site_gradient_detail.png"
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")
    return out_file


def plot_map_correlation_and_histograms(d_grad, d_fofc):
    """Plot global correlation and density distributions between Gradient and mFo-DFc maps."""
    g_flat = d_grad.flatten()
    f_flat = d_fofc.flatten()

    # Subsample for responsive hexbin plotting
    sub_indices = np.random.choice(len(g_flat), size=min(150000, len(g_flat)), replace=False)
    g_sub = g_flat[sub_indices]
    f_sub = f_flat[sub_indices]

    corr = np.corrcoef(g_flat, f_flat)[0, 1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: Hexbin correlation
    hb = axes[0].hexbin(f_sub, g_sub, gridsize=60, cmap="plasma", mincnt=1, bins="log")
    axes[0].set_xlabel(r"$\sigma_A$-Weighted Difference Map Density: $mF_o - DF_c$ ($\sigma$)", fontsize=11)
    axes[0].set_ylabel(r"Intensity Target Gradient Density: $-\frac{1}{2}\frac{\partial Q}{\partial F_{model}}$ ($\sigma$)", fontsize=11)
    axes[0].set_title(f"Global Map Density Correlation: r = {corr:.3f}", fontsize=13)
    axes[0].axvline(0, color="gray", linestyle="--", alpha=0.6)
    axes[0].axhline(0, color="gray", linestyle="--", alpha=0.6)
    cb = fig.colorbar(hb, ax=axes[0], label="log10(Grid Points)")

    # Regression line
    m, b = np.polyfit(f_sub, g_sub, 1)
    x_vals = np.linspace(-6, 6, 100)
    axes[0].plot(x_vals, m * x_vals + b, color="cyan", linewidth=2.0, label=f"Fit: y = {m:.2f}x + {b:.2f}")
    axes[0].legend(loc="upper left")

    # Panel 2: Density Value Histograms
    axes[1].hist(g_flat, bins=100, range=(-8, 8), density=True, alpha=0.6, label="Intensity Gradient Map", color="crimson")
    axes[1].hist(f_flat, bins=100, range=(-8, 8), density=True, alpha=0.6, label="mFo - DFc Difference Map", color="royalblue")
    axes[1].set_xlabel("Voxel Density Value (σ)", fontsize=11)
    axes[1].set_ylabel("Probability Density", fontsize=11)
    axes[1].set_title("Voxel Distribution Across Unit Cell", fontsize=13)
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    out_file = LYSOZYME_DIR / "map_correlation_and_histograms.png"
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")
    return out_file


def plot_omit_loop_zoom(d_omit_g, uc, w_idx, z_val):
    """Detailed zoom showing the omitted loop (Arg45 - Asp52) inside positive gradient contours."""
    pdb_inp = iotbx.pdb.input(str(PDB_PATH))
    hier = pdb_inp.construct_hierarchy()
    grid_u, grid_v, grid_w = d_omit_g.shape
    a, b, c = uc.parameters()[:3]

    x_min, x_max = 5.0, 22.0
    y_min, y_max = 11.0, 26.0

    u_min, u_max = int((x_min / a) * grid_u), int((x_max / a) * grid_u)
    v_min, v_max = int((y_min / b) * grid_v), int((y_max / b) * grid_v)

    zoom_grad = d_omit_g[u_min:u_max, v_min:v_max, w_idx].T
    extent = [x_min, x_max, y_min, y_max]

    # Collect atoms within 2.5 Å of the slice
    omitted_atoms = []
    surrounding_atoms = []
    for rg in hier.models()[0].chains()[0].residue_groups():
        resseq = rg.resseq.strip()
        is_omit = resseq.isdigit() and 45 <= int(resseq) <= 52
        for ag in rg.atom_groups():
            resname = ag.resname.strip()
            for atom in ag.atoms():
                x, y, z = atom.xyz
                if x_min <= x <= x_max and y_min <= y <= y_max and abs(z - z_val) <= 2.5:
                    if is_omit:
                        omitted_atoms.append((resname, resseq, atom.name.strip(), x, y, z))
                    else:
                        surrounding_atoms.append((resname, resseq, atom.name.strip(), x, y, z))

    fig, ax = plt.subplots(figsize=(10, 8))
    vmax = max(18.0, zoom_grad.max())
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(zoom_grad, origin="lower", extent=extent, cmap="coolwarm", norm=norm)

    # Positive difference contours
    ax.contour(zoom_grad, levels=[3.0, 6.0, 10.0, 14.0], extent=extent, colors=["#15803d", "#16a34a", "#22c55e", "#4ade80"], linewidths=[1.2, 1.5, 1.8, 2.0])
    ax.contour(zoom_grad, levels=[-3.0], extent=extent, colors=["#dc2626"], linestyles="dashed", linewidths=1.2)

    # Plot surrounding model atoms in gray
    for resname, resseq, aname, x, y, z in surrounding_atoms:
        ax.scatter(x, y, color="#64748b", s=25, marker="o", alpha=0.6, zorder=4)

    # Plot omitted atoms in bright magenta/orange to show how perfectly the gradient fills the missing atoms
    labeled = set()
    for resname, resseq, aname, x, y, z in omitted_atoms:
        ax.scatter(x, y, color="#e11d48", s=60, marker="o", edgecolors="white", linewidth=0.8, zorder=6)
        rid = f"{resname}{resseq}"
        if rid not in labeled and aname in ["CA", "OD1", "CG"]:
            labeled.add(rid)
            ax.annotate(
                f"{resname} {resseq} (Omitted)",
                (x, y),
                xytext=(x + 0.4, y + 0.4),
                fontsize=10,
                fontweight="bold",
                color="#881337",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#ffe4e6", alpha=0.9, edgecolor="#f43f5e"),
                zorder=7,
            )

    ax.set_title(
        r"$\mathbf{Omitted\ Loop\ (Arg45\ -\ Asp52)\ Peak\ Gradient\ Density}$"
        + f"\nSlice at Z = {z_val:.1f} Å (Green Contours: +3σ, +6σ, +10σ, +14σ; Red Dots: Omitted Atoms)",
        fontsize=12,
    )
    ax.set_xlabel("X (Å)", fontsize=11)
    ax.set_ylabel("Y (Å)", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Intensity Target Gradient (σ)")

    out_file = LYSOZYME_DIR / "lysozyme_omit_loop_zoom.png"
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")
    return out_file


def plot_omit_comparison(prefix="1iee_omit_delete"):
    """Generate visual comparison of the complete model vs omit model (residues 45-52 omitted)."""
    omit_grad_path = LYSOZYME_DIR / f"{prefix}_gradient.ccp4"
    omit_fofc_path = LYSOZYME_DIR / f"{prefix}_fofc.ccp4"

    if not omit_grad_path.exists():
        print(f"Skipping omit plots: {omit_grad_path} does not exist.")
        return None

    m_orig_grad = iotbx.ccp4_map.map_reader(str(GRAD_PATH))
    m_omit_grad = iotbx.ccp4_map.map_reader(str(omit_grad_path))
    m_omit_fofc = iotbx.ccp4_map.map_reader(str(omit_fofc_path))

    d_orig_g = m_orig_grad.data.as_numpy_array()
    d_omit_g = m_omit_grad.data.as_numpy_array()
    d_omit_f = m_omit_fofc.data.as_numpy_array()

    d_orig_g = (d_orig_g - d_orig_g.mean()) / d_orig_g.std()
    d_omit_g = (d_omit_g - d_omit_g.mean()) / d_omit_g.std()
    d_omit_f = (d_omit_f - d_omit_f.mean()) / d_omit_f.std()

    uc = m_omit_grad.unit_cell()
    a, b, c = uc.parameters()[:3]
    grid_u, grid_v, grid_w = d_omit_g.shape

    # w_idx = 66 (Z = 24.6 Å), global peak
    w_idx = 66
    z_ang = (w_idx / grid_w) * c
    extent = [0, a, 0, b]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.15)

    vmax = max(18.0, float(d_omit_g.max()))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    # 1. Complete Model Gradient Map
    im0 = axes[0].imshow(d_orig_g[:, :, w_idx].T, origin="lower", extent=extent, cmap="coolwarm", norm=norm)
    axes[0].set_title(r"$\mathbf{Complete\ Model\ (1IEE)}$" + f"\nIntensity Gradient Map (Z = {z_ang:.1f} Å)", fontsize=12)
    axes[0].set_xlabel("X (Å)", fontsize=11)
    axes[0].set_ylabel("Y (Å)", fontsize=11)
    axes[0].contour(d_orig_g[:, :, w_idx].T, levels=[3.0], extent=extent, colors=["green"], linewidths=1.0)
    axes[0].contour(d_orig_g[:, :, w_idx].T, levels=[-3.0], extent=extent, colors=["red"], linestyles="dashed", linewidths=1.0)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Gradient (σ)")

    # 2. Omit Model Gradient Map
    im1 = axes[1].imshow(d_omit_g[:, :, w_idx].T, origin="lower", extent=extent, cmap="coolwarm", norm=norm)
    axes[1].set_title(r"$\mathbf{Omit\ Model\ (Residues\ 45-52\ Omitted)}$" + f"\nIntensity Gradient Map (Peak: +{d_omit_g[:, :, w_idx].max():.1f}σ)", fontsize=12)
    axes[1].set_xlabel("X (Å)", fontsize=11)
    axes[1].contour(d_omit_g[:, :, w_idx].T, levels=[3.0, 8.0, 14.0], extent=extent, colors=["#16a34a", "#22c55e", "#86efac"], linewidths=1.2)
    axes[1].contour(d_omit_g[:, :, w_idx].T, levels=[-3.0], extent=extent, colors=["red"], linestyles="dashed", linewidths=1.0)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Gradient (σ)")

    # 3. Omit Model Fo-Fc Map
    im2 = axes[2].imshow(d_omit_f[:, :, w_idx].T, origin="lower", extent=extent, cmap="coolwarm", norm=norm)
    axes[2].set_title(r"$\mathbf{Omit\ Model\ (Residues\ 45-52\ Omitted)}$" + f"\nDifference Map $mF_o - DF_c$ (Peak: +{d_omit_f[:, :, w_idx].max():.1f}σ)", fontsize=12)
    axes[2].set_xlabel("X (Å)", fontsize=11)
    axes[2].contour(d_omit_f[:, :, w_idx].T, levels=[3.0, 8.0, 14.0], extent=extent, colors=["#16a34a", "#22c55e", "#86efac"], linewidths=1.2)
    axes[2].contour(d_omit_f[:, :, w_idx].T, levels=[-3.0], extent=extent, colors=["red"], linestyles="dashed", linewidths=1.0)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="Difference (σ)")

    out_file = LYSOZYME_DIR / "lysozyme_omit_difference_map.png"
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")

    plot_omit_loop_zoom(d_omit_g, uc, w_idx, z_ang)
    return out_file


def main():
    print("Loading CCP4 maps...")
    d_grad, d_fofc, d_2fofc, uc = load_maps()
    print("Plotting map comparisons...")
    f1 = plot_map_comparisons(d_grad, d_fofc, d_2fofc, uc, z_idx=63)
    print("Plotting active site zoom...")
    f2 = plot_active_site_zoom(d_grad, d_fofc, d_2fofc, uc)
    print("Plotting correlation and histograms...")
    f3 = plot_map_correlation_and_histograms(d_grad, d_fofc)
    print("Plotting omit comparison...")
    plot_omit_comparison()
    print("Done generating plots.")


if __name__ == "__main__":
    main()
