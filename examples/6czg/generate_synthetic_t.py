#!/usr/bin/env python3
"""Generate a series of synthetic noisy intensities under a Student-t noise model (nu=7).

Uses the 6CZG model and bulk solvent to compute noise-free ground truth intensities F_model,
obtains the per-reflection I/sigma observational error profile from the experimental MTZ,
and applies controlled noise multipliers under a heavy-tailed Student-t distribution (nu=7)
to generate a series of MTZ files representing progressively degraded data quality.

Usage:
    .venv/bin/python examples/6czg/generate_synthetic_t.py
    .venv/bin/python examples/6czg/generate_synthetic_t.py --multipliers 1.0 1.5 2.0 3.0 5.0 10.0 --nu 7.0
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from phridge.client.intensity_tool import from_files
from phridge.contrib.intensity_ll.synthetic import (
    generate_synthetic_series,
    SyntheticIntensityResult,
)

DEFAULT_DIR = Path(__file__).resolve().parent
DEFAULT_PDB = DEFAULT_DIR / "6czg.pdb"
DEFAULT_MTZ = DEFAULT_DIR / "6czg.mtz"
DEFAULT_OUT_DIR = DEFAULT_DIR / "synthetic_t"


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic noisy intensities under Student-t model (nu=7) with per-reflection I/sigma scaling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pdb", type=str, default=str(DEFAULT_PDB), help="Path to input 6CZG PDB file.")
    parser.add_argument("--mtz", type=str, default=str(DEFAULT_MTZ), help="Path to reference experimental 6CZG MTZ file.")
    parser.add_argument("--nu", type=float, default=7.0, help="Degrees of freedom for Student-t noise model.")
    parser.add_argument(
        "--multipliers",
        type=float,
        nargs="+",
        default=[1.0, 1.5, 2.0, 3.0, 5.0, 10.0],
        help="List of noise multipliers (>= 1.0 makes data progressively worse).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["direct", "poisson", "snr"],
        default="direct",
        help="Per-reflection sigma behavior mode ('direct', 'poisson', or 'snr').",
    )
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output directory for MTZ files and reports.")
    parser.add_argument("--prefix", type=str, default="6czg_synthetic_t", help="Prefix for generated MTZ files.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for reproducibility.")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device for solvent refinement ('cpu' or 'mps').")
    return parser.parse_args(args)


def write_markdown_report(
    summary_rows: list[dict],
    results: dict[float, SyntheticIntensityResult],
    solvent_params: dict[str, float],
    args: argparse.Namespace,
    output_path: Path,
) -> None:
    """Write an executive summary markdown report."""
    lines: list[str] = [
        "# 6CZG Synthetic Intensity Series: Student-t Noise Model",
        "",
        "## 1. Experimental Setup & Model Ground Truth",
        "",
        f"- **Reference Model PDB**: `{args.pdb}`",
        f"- **Reference MTZ**: `{args.mtz}`",
        f"- **Student-t Degrees of Freedom ($\\nu$)**: `{args.nu:.1f}`",
        f"- **Error Scaling Mode**: `{args.mode}`",
        f"- **Refined Bulk Solvent Scale ($k_\\text{{total}}$)**: `{solvent_params.get('k_total', 0.0):.4f}`",
        f"- **Refined Solvent Parameter ($k_\\text{{sol}}$)**: `{solvent_params.get('k_sol', 0.0):.4f}`",
        f"- **Refined Solvent $B$-factor ($B_\\text{{sol}}$)**: `{solvent_params.get('b_sol', 0.0):.2f}` Å²",
        f"- **Random Seed**: `{args.seed}`",
        "",
        "## 2. Multiplier Series Summary",
        "",
        "| Multiplier | MTZ File | $\\langle I/\\sigma \\rangle$ | Median $I/\\sigma$ | Outer Shell $\\langle I/\\sigma \\rangle$ | $\\langle \\sigma \\rangle$ | Neg Count | Neg % | $R_\\text{noise}$ (%) | True/Synth Corr |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for row in summary_rows:
        m_str = f"{row['multiplier']:g}"
        lines.append(
            f"| `{m_str}x` | `{row['mtz_file']}` | {row['mean_i_over_sig']:.2f} | {row['median_i_over_sig']:.2f} | "
            f"{row['high_res_i_over_sig']:.2f} | {row['mean_sig']:.1f} | {row['n_negative']} | "
            f"{row['fraction_negative_pct']:.2f}% | {row['r_noise_pct']:.2f}% | {row['corr_true_synth']:.4f} |"
        )

    lines.extend([
        "",
        "## 3. Resolution Shell Breakdown Across Multipliers",
        "",
    ])

    for mult, res in results.items():
        m_str = f"{mult:g}"
        lines.extend([
            f"### Multiplier {m_str}x (`{res.stats['mean_i_over_sig']:.2f}` mean $I/\\sigma$, $R_\\text{{noise}} = {res.stats['r_noise_percent']:.2f}\\%$)",
            "",
            "| Shell | $d_\\text{max}$ (Å) | $d_\\text{min}$ (Å) | $N_\\text{refl}$ | $\\langle I/\\sigma \\rangle$ | Median $I/\\sigma$ | Expected SNR | $\\langle \\sigma \\rangle$ | Neg Count | Neg % | $R_\\text{noise}$ (%) |",
            "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        for sh in res.shell_stats:
            lines.append(
                f"| {sh['shell_index']} | {sh['d_max']:.2f} | {sh['d_min']:.2f} | {sh['n_reflections']} | "
                f"{sh['mean_i_over_sig']:.2f} | {sh['median_i_over_sig']:.2f} | {sh['mean_expected_snr']:.2f} | "
                f"{sh['mean_sig']:.1f} | {sh['n_negative']} | {sh['fraction_negative']*100:.2f}% | {sh['r_noise_percent']:.2f}% |"
            )
        lines.append("")

    output_path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.time()

    print("=" * 100)
    print(" 6CZG SYNTHETIC NOISY INTENSITIES GENERATOR (Student-t, nu=7.0)")
    print(f" Model PDB: {args.pdb}")
    print(f" Reference MTZ: {args.mtz}")
    print(f" Degrees of Freedom (nu): {args.nu:.1f}")
    print(f" Multipliers: {args.multipliers}")
    print(f" Sigma Scaling Mode: {args.mode}")
    print(f" Output Directory: {args.output_dir}")
    print("=" * 100)

    # 1. Load 6CZG model and refine scale and solvent
    print("\n1. Loading 6CZG model and refining flat bulk solvent...")
    model = from_files(
        args.pdb,
        args.mtz,
        d_min=None,
        device=args.device,
        use_bulk_solvent=True,
    )
    solvent_params = model.refine_scale_and_solvent()
    print(f"   Refined bulk solvent: k_total = {solvent_params['k_total']:.4f}, "
          f"k_sol = {solvent_params['k_sol']:.4f}, B_sol = {solvent_params['b_sol']:.2f} Å²")
    print(f"   Ground truth F_model reflections: {model.f_model.size():,d}")

    # 2. Experimental baseline diagnostics
    io_exp = np.asarray(model.i_obs.data(), dtype=np.float64)
    sig_exp = np.asarray(model.i_obs.sigmas(), dtype=np.float64)
    snr_exp = io_exp / np.maximum(sig_exp, 1e-6)
    print(f"   Experimental 6CZG baseline:")
    print(f"     Mean <I/sig>: {np.mean(snr_exp):.2f} | Median: {np.median(snr_exp):.2f}")
    print(f"     Mean sigma: {np.mean(sig_exp):.2f} | Negative reflections: {np.sum(io_exp < 0)} ({100.0 * np.sum(io_exp < 0) / len(io_exp):.2f}%)")

    # 3. Generate series
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n2. Generating synthetic series across {len(args.multipliers)} noise multipliers...")
    results, summary_rows = generate_synthetic_series(
        model=model,
        multipliers=args.multipliers,
        nu=args.nu,
        mode=args.mode,
        output_dir=out_dir,
        base_prefix=args.prefix,
        seed=args.seed,
        write_mtz=True,
    )

    # 4. Print summary table
    print("\n" + "=" * 115)
    print(" SYNTHETIC INTENSITY SERIES SUMMARY TABLE")
    print("=" * 115)
    print(f"{'Multiplier':<11} | {'MTZ File':<30} | {'<I/sig>':<8} | {'Med(I/s)':<8} | {'High-Res':<8} | {'<sig>':<7} | {'Neg Cnt':<7} | {'Neg %':<6} | {'R_noise':<8} | {'Corr':<6}")
    print("-" * 115)
    for row in summary_rows:
        m_str = f"{row['multiplier']:g}x"
        print(f"{m_str:<11} | {row['mtz_file']:<30} | {row['mean_i_over_sig']:8.2f} | {row['median_i_over_sig']:8.2f} | "
              f"{row['high_res_i_over_sig']:8.2f} | {row['mean_sig']:7.1f} | {row['n_negative']:7d} | "
              f"{row['fraction_negative_pct']:5.2f}% | {row['r_noise_pct']:7.2f}% | {row['corr_true_synth']:6.4f}")
    print("=" * 115)

    # 5. Print high-res vs low-res shell comparison for representative multipliers
    rep_mults = [m for m in [1.0, 2.0, 5.0, 10.0] if m in results]
    if rep_mults:
        print("\n" + "=" * 115)
        print(" RESOLUTION SHELL BREAKDOWN FOR REPRESENTATIVE MULTIPLIERS")
        print("=" * 115)
        header = f"{'Shell':<5} | {'d_range (Å)':<15} | {'N_refl':<6} | " + " | ".join(f"{m:g}x <I/s> ({m:g}x R%)" for m in rep_mults)
        print(header)
        print("-" * 115)
        n_shells = len(results[rep_mults[0]].shell_stats)
        for sh_idx in range(n_shells):
            sh0 = results[rep_mults[0]].shell_stats[sh_idx]
            d_str = f"{sh0['d_max']:5.2f} - {sh0['d_min']:5.2f}"
            vals = []
            for m in rep_mults:
                sh_m = results[m].shell_stats[sh_idx]
                vals.append(f"{sh_m['mean_i_over_sig']:8.2f} ({sh_m['r_noise_percent']:5.2f}%)")
            print(f"{sh_idx:<5} | {d_str:<15} | {sh0['n_reflections']:<6} | " + " | ".join(vals))
        print("=" * 115)

    # 6. Save JSON and Markdown reports
    json_path = out_dir / "synthetic_t_series_summary.json"
    md_path = out_dir / "synthetic_t_series_summary.md"

    json_data = {
        "pdb": str(args.pdb),
        "mtz": str(args.mtz),
        "nu": args.nu,
        "mode": args.mode,
        "solvent_params": solvent_params,
        "summary": summary_rows,
        "shell_stats": {f"{m:g}": res.shell_stats for m, res in results.items()},
    }
    json_path.write_text(json.dumps(json_data, indent=2))
    write_markdown_report(summary_rows, results, solvent_params, args, md_path)

    print(f"\nWrote full reports to:")
    print(f"  - Markdown: {md_path}")
    print(f"  - JSON:     {json_path}")
    print(f"  - MTZ files saved in: {out_dir}/")
    print(f"Total time: {time.time() - t0:.2f}s\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
