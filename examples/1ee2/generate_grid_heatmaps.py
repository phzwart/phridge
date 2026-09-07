#!/usr/bin/env python3
"""Generate publication-quality PNG heatmaps for 1EE2 CHD Occupancy vs Noise Grid Scan.

Outputs:
  1. chd_occupancy_noise_4panel_heatmap.png (2x2 comprehensive overview)
  2. chd_target_advantage_heatmap.png (focused difference heatmap ml_i - ml_f)
  3. chd_ml_i_detection_heatmap.png (focused ml_i detection heatmap)
"""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

DIR = Path(__file__).resolve().parent
JSON_PATH = DIR / "synthetic_chd_grid_scan" / "chd_grid_scan_report.json"
OUT_DIR = DIR / "synthetic_chd_grid_scan"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(JSON_PATH, "r") as f:
    data = json.load(f)

occupancies = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.0]
multipliers = [0.0, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0]

eff_res_labels = [
    "0.0x\n(1.54Å)",
    "1.0x\n(1.54Å)",
    "2.0x\n(1.70Å)",
    "3.0x\n(1.85Å)",
    "5.0x\n(2.10Å)",
    "7.5x\n(2.60Å)",
    "10.0x\n(3.20Å)",
    "15.0x\n(4.00Å)",
]

occ_labels = [f"q = {q:.2f}" for q in occupancies]

n_occ = len(occupancies)
n_mult = len(multipliers)

mat_det_i = np.zeros((n_occ, n_mult))
mat_det_f = np.zeros((n_occ, n_mult))
mat_med_i = np.zeros((n_occ, n_mult))
mat_med_f = np.zeros((n_occ, n_mult))
mat_gap_i = np.zeros((n_occ, n_mult))

for i, q in enumerate(occupancies):
    q_str = f"{q:0.2f}".replace(".", "_")
    for j, m in enumerate(multipliers):
        m_str = f"{m:g}".replace(".", "_")
        key = f"m_{m_str}_q_{q_str}"
        cell = data["cells"][key]
        mat_det_i[i, j] = cell["intensity"]["det_3sig_pct"]
        mat_det_f[i, j] = cell["amplitude"]["det_3sig_pct"]
        mat_med_i[i, j] = cell["intensity"]["chd_overall"]["median"]
        mat_med_f[i, j] = cell["amplitude"]["chd_overall"]["median"]
        mat_gap_i[i, j] = cell["intensity"]["gap_clean"]

mat_diff = mat_det_i - mat_det_f

# Set general style
plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.family": "sans-serif",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.0,
})

# ==============================================================================
# 1. FOUR-PANEL MASTER HEATMAP FIGURE
# ==============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 13), dpi=300)

# Helper function to annotate heatmap
def annotate_matrix(ax, matrix, fmt="{:.1f}%", threshold=50, text_color_dark="black", text_color_light="white", fontsize=8.5):
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            val = matrix[r, c]
            txt = fmt.format(val)
            color = text_color_light if val > threshold else text_color_dark
            ax.text(c, r, txt, ha="center", va="center", color=color, fontsize=fontsize, fontweight="semibold")

# --- Panel (a): ml_i Detection Rate ---
ax_a = axes[0, 0]
im_a = ax_a.imshow(mat_det_i, cmap="viridis", vmin=0, vmax=100, aspect="auto")
ax_a.set_title(r"(a) Intensity Likelihood ($ml\_i$) Detection Rate ($\geq 3.0\sigma$)", fontsize=13, fontweight="bold", pad=10)
ax_a.set_xticks(range(n_mult))
ax_a.set_xticklabels(eff_res_labels, fontsize=9.5)
ax_a.set_yticks(range(n_occ))
ax_a.set_yticklabels(occ_labels, fontsize=10, fontweight="bold")
ax_a.set_ylabel(r"Ligand Occupancy ($q$)", fontsize=11, fontweight="bold")
annotate_matrix(ax_a, mat_det_i, fmt="{:.1f}%", threshold=50)
cbar_a = fig.colorbar(im_a, ax=ax_a, fraction=0.046, pad=0.04)
cbar_a.set_label("Atoms Detected (%)", fontsize=10, fontweight="bold")

# --- Panel (b): ml_f Detection Rate ---
ax_b = axes[0, 1]
im_b = ax_b.imshow(mat_det_f, cmap="viridis", vmin=0, vmax=100, aspect="auto")
ax_b.set_title(r"(b) Amplitude Likelihood ($ml\_f$, French-Wilson) Detection Rate ($\geq 3.0\sigma$)", fontsize=13, fontweight="bold", pad=10)
ax_b.set_xticks(range(n_mult))
ax_b.set_xticklabels(eff_res_labels, fontsize=9.5)
ax_b.set_yticks(range(n_occ))
ax_b.set_yticklabels(occ_labels, fontsize=10, fontweight="bold")
annotate_matrix(ax_b, mat_det_f, fmt="{:.1f}%", threshold=50)
cbar_b = fig.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04)
cbar_b.set_label("Atoms Detected (%)", fontsize=10, fontweight="bold")

# --- Panel (c): Target Advantage: ml_i - ml_f ---
ax_c = axes[1, 0]
norm_c = plt.Normalize(vmin=-10, vmax=70)
im_c = ax_c.imshow(mat_diff, cmap="YlOrRd", vmin=-5, vmax=70, aspect="auto")
ax_c.set_title(r"(c) Intensity Advantage: $\Delta(ml\_i - ml\_f)$ Detection Rate (%)", fontsize=13, fontweight="bold", pad=10)
ax_c.set_xticks(range(n_mult))
ax_c.set_xticklabels(eff_res_labels, fontsize=9.5)
ax_c.set_yticks(range(n_occ))
ax_c.set_yticklabels(occ_labels, fontsize=10, fontweight="bold")
ax_c.set_xlabel(r"Noise Multiplier ($m$) & Effective Resolution", fontsize=11, fontweight="bold")
ax_c.set_ylabel(r"Ligand Occupancy ($q$)", fontsize=11, fontweight="bold")

for r in range(mat_diff.shape[0]):
    for c in range(mat_diff.shape[1]):
        val = mat_diff[r, c]
        txt = f"{val:+.1f}%" if abs(val) >= 0.05 else "0.0%"
        color = "white" if val > 35 else "black"
        fw = "bold" if val >= 20 else "normal"
        ax_c.text(c, r, txt, ha="center", va="center", color=color, fontsize=8.5, fontweight=fw)

cbar_c = fig.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.04)
cbar_c.set_label("Gain in Atoms Detected (%)", fontsize=10, fontweight="bold")

# --- Panel (d): ml_i Median Difference Peak Height ---
ax_d = axes[1, 1]
im_d = ax_d.imshow(mat_med_i, cmap="magma", vmin=-2, vmax=14, aspect="auto")
ax_d.set_title(r"(d) $ml\_i$ Median Ligand Difference Density ($\sigma$)", fontsize=13, fontweight="bold", pad=10)
ax_d.set_xticks(range(n_mult))
ax_d.set_xticklabels(eff_res_labels, fontsize=9.5)
ax_d.set_yticks(range(n_occ))
ax_d.set_yticklabels(occ_labels, fontsize=10, fontweight="bold")
ax_d.set_xlabel(r"Noise Multiplier ($m$) & Effective Resolution", fontsize=11, fontweight="bold")

for r in range(mat_med_i.shape[0]):
    for c in range(mat_med_i.shape[1]):
        val = mat_med_i[r, c]
        txt = f"{val:.2f}σ"
        color = "white" if val > 6.0 else ("gold" if val >= 3.0 else "#cccccc")
        fw = "bold" if val >= 3.0 else "normal"
        ax_d.text(c, r, txt, ha="center", va="center", color=color, fontsize=8.5, fontweight=fw)

cbar_d = fig.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.04)
cbar_d.set_label(r"Peak Height ($\sigma$)", fontsize=10, fontweight="bold")

plt.suptitle(r"1EE2 Cholate (CHD) Ligand Omission: Occupancy ($q$) vs Noise ($m$) Landscape", fontsize=15, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])

path_4panel = OUT_DIR / "chd_occupancy_noise_4panel_heatmap.png"
fig.savefig(path_4panel, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path_4panel}")


# ==============================================================================
# 2. FOCUSED STANDALONE HEATMAP: TARGET ADVANTAGE (ml_i - ml_f)
# ==============================================================================
fig, ax = plt.subplots(figsize=(11, 7.5), dpi=300)
im = ax.imshow(mat_diff, cmap="YlOrRd", vmin=0, vmax=70, aspect="auto")

ax.set_title(r"1EE2 Difference Density Sensitivity Gain: $\Delta(ml\_i - ml\_f)$ (% Atoms $\geq 3\sigma$)", fontsize=13, fontweight="bold", pad=14)
ax.set_xticks(range(n_mult))
ax.set_xticklabels(eff_res_labels, fontsize=10)
ax.set_yticks(range(n_occ))
ax.set_yticklabels(occ_labels, fontsize=11, fontweight="bold")
ax.set_xlabel(r"Noise Multiplier ($m$) & Effective Resolution Limit (Student-t, $\nu = 7.0$)", fontsize=11, fontweight="bold", labelpad=8)
ax.set_ylabel(r"Ligand Occupancy ($q$)", fontsize=11, fontweight="bold", labelpad=8)

for r in range(mat_diff.shape[0]):
    for c in range(mat_diff.shape[1]):
        val = mat_diff[r, c]
        txt = f"{val:+.1f}%" if abs(val) >= 0.05 else "0.0%"
        color = "white" if val > 35 else "black"
        fw = "bold" if val >= 20 else "normal"
        fs = 11 if val >= 30 else 9.5
        ax.text(c, r, txt, ha="center", va="center", color=color, fontsize=fs, fontweight=fw)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cbar.set_label("Intensity Target Advantage (% More Atoms Detected)", fontsize=10.5, fontweight="bold")

# Add highlight box around peak advantage cell (q=0.40, m=5.0x)
rect = plt.Rectangle((4 - 0.48, 3 - 0.48), 0.96, 0.96, fill=False, edgecolor="#00ffff", linewidth=2.5, linestyle="--")
ax.add_patch(rect)
ax.annotate(r"Peak Advantage (+67.2%)" + "\n" + r"$ml\_i$: 74.1% vs $ml\_f$: 6.9%",
            xy=(4, 3), xytext=(4.3, 2.1),
            arrowprops=dict(facecolor="#00ffff", shrink=0.08, width=1.5, headwidth=6),
            fontsize=9.5, fontweight="bold", color="#00ffff",
            bbox=dict(boxstyle="round,pad=0.3", fc="#222222", ec="#00ffff", lw=1.2))

plt.tight_layout()
path_adv = OUT_DIR / "chd_target_advantage_heatmap.png"
fig.savefig(path_adv, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path_adv}")


# ==============================================================================
# 3. FOCUSED STANDALONE HEATMAP: ml_i DETECTION RATE
# ==============================================================================
fig, ax = plt.subplots(figsize=(11, 7.5), dpi=300)
im = ax.imshow(mat_det_i, cmap="viridis", vmin=0, vmax=100, aspect="auto")

ax.set_title(r"1EE2 Cholate (CHD) Detection Landscape under Intensity Likelihood ($ml\_i$)", fontsize=13, fontweight="bold", pad=14)
ax.set_xticks(range(n_mult))
ax.set_xticklabels(eff_res_labels, fontsize=10)
ax.set_yticks(range(n_occ))
ax.set_yticklabels(occ_labels, fontsize=11, fontweight="bold")
ax.set_xlabel(r"Noise Multiplier ($m$) & Effective Resolution Limit (Student-t, $\nu = 7.0$)", fontsize=11, fontweight="bold", labelpad=8)
ax.set_ylabel(r"Ligand Occupancy ($q$)", fontsize=11, fontweight="bold", labelpad=8)

for r in range(mat_det_i.shape[0]):
    for c in range(mat_det_i.shape[1]):
        val = mat_det_i[r, c]
        txt = f"{val:.1f}%"
        color = "white" if val > 50 else ("yellow" if val >= 20 else "#888888")
        fw = "bold" if val >= 50 else "normal"
        ax.text(c, r, txt, ha="center", va="center", color=color, fontsize=10, fontweight=fw)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cbar.set_label(r"CHD Atoms Detected ($\geq 3.0\sigma$, %)", fontsize=10.5, fontweight="bold")

# Add contour line dividing >= 50% detection from < 50%
plt.tight_layout()
path_det = OUT_DIR / "chd_ml_i_detection_heatmap.png"
fig.savefig(path_det, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path_det}")
