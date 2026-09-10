#!/usr/bin/env python3
"""Four-Way Crystallographic Refinement Comparison on Horse Liver Alcohol Dehydrogenase (1EE2).

Disentangles the target likelihood formulation from Hessian curvature preconditioning
via a full 2x2 factorial comparison:

  1. Classic F-based likelihood (ml_f, unpreconditioned):
     - French-Wilson amplitudes F_obs and sigma(F_obs)
     - Rice / Luzzati amplitude likelihood target (ml_f)
     - CCTBX map-gridded bulk solvent (k_sol, B_sol) and overall scale (k_total)
     - CCTBX geometry restraints (bonds, angles, dihedrals, chiralities, planarities)
     - Standard Adam optimizer without preconditioner (raw gradient updates)
     - Periodic re-estimation of sigma_A across resolution bins

  2. Preconditioned F-based likelihood (ml_f, preconditioned):
     - French-Wilson amplitudes F_obs and sigma(F_obs)
     - Rice / Luzzati amplitude likelihood target (ml_f)
     - Gauss-Newton Hessian block preconditioner on atomic coordinates (3x3 blocks)
       and isotropic B-factors (diagonal blocks) from amplitude target curvatures
     - CCTBX bulk solvent, CCTBX geometry restraints, and preconditioned Adam
     - Periodic re-estimation of sigma_A across resolution bins

  3. Intensity-based likelihood (ml_i, unpreconditioned):
     - Directly against measured intensities I_obs and sigma(I_obs)
     - Student-t noise model with refined degrees of freedom nu per resolution shell
     - Bayesian posterior likelihood (ml_i quadrature)
     - CCTBX map-gridded bulk solvent and CCTBX geometry restraints
     - Standard Adam optimizer without preconditioner
     - Dynamic co-refinement of sigma_A and nu

  4. Preconditioned intensity likelihood (ml_i, preconditioned):
     - Directly against measured intensities I_obs and sigma(I_obs)
     - Student-t noise model with refined nu
     - Gauss-Newton Hessian block preconditioner on atomic coordinates (3x3 blocks)
       and isotropic B-factors (diagonal blocks) from intensity target curvatures
     - CCTBX bulk solvent, CCTBX geometry restraints, and preconditioned Adam
     - Dynamic co-refinement of sigma_A and nu

Usage::

    # Run full 4-way comparison (default 3 steps per mode, 1.8 Å resolution limit)
    python examples/1ee2/run_1ee2.py --steps 3 --d-min 1.8

    # Run a specific mode only
    python examples/1ee2/run_1ee2.py --mode precond_f --steps 5
    python examples/1ee2/run_1ee2.py --mode precond_intensity --steps 5
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import cctbx before torch for MKL / OpenMP safety
import cctbx  # noqa: F401
from cctbx.array_family import flex
import iotbx.pdb
from iotbx.reflection_file_reader import any_reflection_file
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from phridge.client.intensity_tool import from_files

DIR_1EE2 = Path(__file__).resolve().parent
PDB_FILE = DIR_1EE2 / "1ee2.pdb"
CIF_FILE = DIR_1EE2 / "1ee2-sf.cif"
MTZ_FILE = DIR_1EE2 / "1ee2.mtz"


def download_file(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  Using existing {dest.name} ({dest.stat().st_size / 1024:.1f} KB)")
        return dest
    print(f"  Downloading {url} -> {dest.name}...")
    req = urllib.request.Request(url, headers={"User-Agent": "phridge/0.1.0"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        f.write(resp.read())
    print(f"  Downloaded {dest.name} ({dest.stat().st_size / 1024:.1f} KB)")
    return dest


def prepare_data(force_download: bool = False) -> tuple[Path, Path]:
    print("\n--- 1. Fetching 1EE2 Data from RCSB PDB ---")
    if force_download:
        if PDB_FILE.exists():
            PDB_FILE.unlink()
        if CIF_FILE.exists():
            CIF_FILE.unlink()
        if MTZ_FILE.exists():
            MTZ_FILE.unlink()

    download_file("https://files.rcsb.org/download/1ee2.pdb", PDB_FILE)
    download_file("https://files.rcsb.org/download/1ee2-sf.cif", CIF_FILE)

    if not MTZ_FILE.exists() or MTZ_FILE.stat().st_size == 0:
        print(f"\n--- 2. Converting {CIF_FILE.name} to {MTZ_FILE.name} ---")
        reader = any_reflection_file(str(CIF_FILE))
        arrays = reader.as_miller_arrays()

        iobs = None
        status = None
        for a in arrays:
            labels = a.info().labels if a.info() else []
            if a.is_xray_intensity_array():
                iobs = a
            elif any("status" in l.lower() for l in labels) and a.is_string_array():
                status = a

        if iobs is None:
            raise ValueError(f"No intensity array found in {CIF_FILE}")

        # Map to asymmetric unit
        iobs = iobs.map_to_asu()
        iobs.set_observation_type_xray_intensity()

        # Build Free R flag (use deposited if present, otherwise 5% random flags with fixed seed)
        if status is not None:
            status = status.map_to_asu()
            s_data = status.data()
            if "f" in s_data or "F" in s_data:
                r_free_data = (s_data == "f") | (s_data == "F")
                r_free = iobs.customized_copy(data=r_free_data, sigmas=None)
            else:
                flex.set_random_seed(42)
                r_free = iobs.generate_r_free_flags(fraction=0.05)
        else:
            flex.set_random_seed(42)
            r_free = iobs.generate_r_free_flags(fraction=0.05)

        mtz_ds = iobs.as_mtz_dataset(column_root_label="IOBS")
        mtz_ds.add_miller_array(r_free, column_root_label="FreeR_flag")
        mtz_ds.mtz_object().write(str(MTZ_FILE))
        print(f"  Wrote {MTZ_FILE.name} with {iobs.size()} reflections (Free R count: {r_free.data().count(True)})")
    else:
        print(f"  Using existing {MTZ_FILE.name}")

    return PDB_FILE, MTZ_FILE


def run_single_refinement(
    name: str,
    target: str,
    use_preconditioner: bool,
    steps: int,
    d_min: float,
    n_bins: int,
    bulk_solvent: bool,
    lr_sites: float,
    lr_b: float,
    lr_scale: float,
    lr_solvent: float,
    damping_factor: float,
    prefix: str,
    write_maps: bool = True,
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print(f" RUNNING MODE: {name}")
    print(f" Target: {target} | Preconditioner: {use_preconditioner} | Steps: {steps}")
    print("=" * 70)

    t0 = time.time()
    model = from_files(
        pdb_path=PDB_FILE,
        mtz_path=MTZ_FILE,
        d_min=d_min,
        n_bins=n_bins,
        use_bulk_solvent=bulk_solvent,
        convert_to_isotropic=True,
        target=target,
        estimate_nu=(target == "intensity"),
    )

    # Initial scale & bulk solvent
    scales = model.refine_scale_and_solvent()
    sa = model.estimate_sigma_a()

    init_sites = np.asarray(model.xray_structure.sites_cart(), dtype=np.float64)
    init_b = np.asarray([float(sc.u_iso * 8.0 * np.pi**2) for sc in model.xray_structure.scatterers()], dtype=np.float64)
    init_geom_e, _ = model.compute_geometry_energy_and_gradients(init_sites)
    summ_init = model.summary()

    print(f"Initial: R_work = {summ_init['r_work']*100:.2f}%, R_free = {summ_init['r_free']*100:.2f}%, CC_free(I) = {summ_init['cc_free_i']:.4f}")
    print(f"         Geom E = {init_geom_e:.2f}, Mean B = {np.mean(init_b):.2f} Å², k_total = {scales['k_total']:.4e}")

    # Run Adam refinement
    history = model.refine_adam(
        max_iterations=steps,
        lr_sites=lr_sites,
        lr_b=lr_b,
        lr_scale=lr_scale,
        lr_solvent=lr_solvent,
        refine_scales=True,
        refine_b=True,
        refine_sites=True,
        refine_sigma_a=True,
        refine_nu=(target == "intensity"),
        use_preconditioner=use_preconditioner,
        preconditioner_interval=steps + 1,  # compute once at step 0 for speed
        damping_factor=damping_factor,
        verbose=True,
    )

    final_sites = np.asarray(model.xray_structure.sites_cart(), dtype=np.float64)
    final_b = np.asarray([float(sc.u_iso * 8.0 * np.pi**2) for sc in model.xray_structure.scatterers()], dtype=np.float64)
    final_geom_e, _ = model.compute_geometry_energy_and_gradients(final_sites)
    summ_final = model.summary()

    # Calculate coordinate shift RMSD
    coord_shift_rmsd = float(np.sqrt(np.mean(np.sum((final_sites - init_sites)**2, axis=1))))
    b_shift_rmsd = float(np.sqrt(np.mean((final_b - init_b)**2)))

    # Write output model
    pdb_out = f"{prefix}_refined.pdb"
    model.write_pdb(pdb_out)
    print(f"  Wrote refined PDB: {pdb_out}")

    map_files = {}
    if write_maps:
        map_files = model.write_maps(prefix=prefix)
        print(f"  Wrote maps: {prefix}_maps.mtz and CCP4 maps")

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")

    return {
        "name": name,
        "target": target,
        "use_preconditioner": use_preconditioner,
        "elapsed_s": elapsed,
        "model": model,
        "history": history,
        "init_summary": summ_init,
        "final_summary": summ_final,
        "init_geom_e": init_geom_e,
        "final_geom_e": final_geom_e,
        "init_mean_b": float(np.mean(init_b)),
        "final_mean_b": float(np.mean(final_b)),
        "coord_shift_rmsd": coord_shift_rmsd,
        "b_shift_rmsd": b_shift_rmsd,
        "final_k_total": model.k_total,
        "final_k_sol": model.k_sol,
        "final_b_sol": model.b_sol,
        "sigma_a": model.sigma_a_binned,
        "pdb_file": pdb_out,
        "map_files": map_files,
    }


def print_comparison_table(results: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 115)
    print(" CRYSTALLOGRAPHIC REFINEMENT COMPARISON TABLE (1EE2)")
    print("=" * 115)
    hdr = f"{'Metric':<30} | " + " | ".join(f"{r['name']:<18}" for r in results)
    print(hdr)
    print("-" * len(hdr))

    rows = [
        ("Likelihood Target", [r["target"] for r in results]),
        ("Preconditioner", ["Gauss-Newton" if r["use_preconditioner"] else "None (Raw Adam)" for r in results]),
        ("Initial R_work", [f"{r['init_summary']['r_work']*100:.2f}%" for r in results]),
        ("Final R_work", [f"{r['final_summary']['r_work']*100:.2f}%" for r in results]),
        ("Initial R_free", [f"{r['init_summary']['r_free']*100:.2f}%" for r in results]),
        ("Final R_free", [f"{r['final_summary']['r_free']*100:.2f}%" for r in results]),
        ("Initial CC_free (I)", [f"{r['init_summary']['cc_free_i']:.4f}" for r in results]),
        ("Final CC_free (I)", [f"{r['final_summary']['cc_free_i']:.4f}" for r in results]),
        ("Initial CC_work (I)", [f"{r['init_summary']['cc_work_i']:.4f}" for r in results]),
        ("Final CC_work (I)", [f"{r['final_summary']['cc_work_i']:.4f}" for r in results]),
        ("Initial R_intensity (free)", [f"{r['init_summary']['r_intensity_free']*100:.2f}%" for r in results]),
        ("Final R_intensity (free)", [f"{r['final_summary']['r_intensity_free']*100:.2f}%" for r in results]),
        ("Initial Geom Energy", [f"{r['init_geom_e']:.1f}" for r in results]),
        ("Final Geom Energy", [f"{r['final_geom_e']:.1f}" for r in results]),
        ("Coord Shift RMSD (Å)", [f"{r['coord_shift_rmsd']:.4f}" for r in results]),
        ("B-factor Shift RMSD (Å²)", [f"{r['b_shift_rmsd']:.3f}" for r in results]),
        ("Mean B (Å²)", [f"{r['final_mean_b']:.2f}" for r in results]),
        ("Fitted k_total", [f"{r['final_k_total']:.4e}" for r in results]),
        ("Fitted k_sol", [f"{r['final_k_sol']:.3f}" for r in results]),
        ("Fitted B_sol (Å²)", [f"{r['final_b_sol']:.1f}" for r in results]),
        ("Elapsed Time", [f"{r['elapsed_s']:.1f} s" for r in results]),
    ]

    for label, vals in rows:
        val_str = " | ".join(f"{v:<18}" for v in vals)
        print(f"{label:<30} | {val_str}")
    print("=" * 115 + "\n")


def plot_comparison(results: List[Dict[str, Any]], out_png: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Horse Liver Alcohol Dehydrogenase (1EE2): Refinement Comparison", fontsize=15, fontweight="bold")

    colors = {
        "Classic F-based": "#1f77b4",
        "Precond F-based": "#9467bd",
        "Intensity (unprecond)": "#ff7f0e",
        "Intensity (precond)": "#2ca02c",
    }

    # Panel 1: Target NLL progression
    ax = axes[0, 0]
    for r in results:
        name = r["name"]
        nll = r["history"]["nll"]
        steps = list(range(1, len(nll) + 1))
        nll_rel = [val - nll[0] for val in nll]
        c = colors.get(name, "black")
        ax.plot(steps, nll_rel, marker="o", lw=2, label=f"{name} (ΔNLL)", color=c)
    ax.set_title("Relative Target NLL Reduction (ΔNLL)", fontweight="bold")
    ax.set_xlabel("Adam Step")
    ax.set_ylabel("NLL - NLL_step1")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)

    # Panel 2: Free-set metrics comparison (R_free and CC_free)
    ax = axes[0, 1]
    names = [r["name"] for r in results]
    r_free_final = [r["final_summary"]["r_free"] * 100.0 for r in results]
    cc_free_final = [r["final_summary"]["cc_free_i"] * 100.0 for r in results]

    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width/2, r_free_final, width, label="R_free (%) [lower=better]", color="#e74c3c")
    ax.bar(x + width/2, cc_free_final, width, label="CC_free(I) (%) [higher=better]", color="#2ecc71")
    ax.set_title("Cross-Validation Metrics on Free Set", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12, ha="right", fontsize=9)
    ax.set_ylabel("Percent (%)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend()

    # Panel 3: Geometry restraint energy progression
    ax = axes[1, 0]
    for r in results:
        name = r["name"]
        ge = [r["init_geom_e"]] + r["history"]["geom_energy"]
        steps = list(range(len(ge)))
        c = colors.get(name, "black")
        ax.plot(steps, ge, marker="s", lw=2, label=name, color=c)
    ax.set_title("CCTBX Geometry Restraint Energy", fontweight="bold")
    ax.set_xlabel("Refinement Iteration (0=Init)")
    ax.set_ylabel("Target Energy (arbitrary units)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)

    # Panel 4: Estimated Sigma_A across resolution bins
    ax = axes[1, 1]
    for r in results:
        name = r["name"]
        sa_dict = r["sigma_a"]
        bins = sorted(sa_dict.keys())
        sa_vals = [sa_dict[b] for b in bins]
        c = colors.get(name, "black")
        ax.plot(bins, sa_vals, marker="^", lw=2, label=name, color=c)
    ax.set_title("Estimated Sigma_A across Resolution Bins", fontweight="bold")
    ax.set_xlabel("Resolution Bin (Low -> High)")
    ax.set_ylabel("Sigma_A")
    ax.set_ylim(0.85, 1.02)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"Wrote comparison figure: {out_png}")


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Four-way refinement test on 1EE2: classic F vs precond F vs intensity vs precond intensity.",
    )
    parser.add_argument(
        "--mode",
        default="all",
        choices=["all", "f_based", "precond_f", "intensity", "precond_intensity"],
        help="Which test to run: 'all' (default), 'f_based', 'precond_f', 'intensity', or 'precond_intensity'",
    )
    parser.add_argument("--steps", type=int, default=3, help="Number of refinement steps per mode (default: 3)")
    parser.add_argument("--d-min", type=float, default=1.8, help="High resolution limit in Å (default: 1.8)")
    parser.add_argument("--n-bins", type=int, default=10, help="Number of resolution bins (default: 10)")
    parser.add_argument("--no-bulk-solvent", action="store_true", help="Disable bulk solvent modeling")
    parser.add_argument("--lr-sites", type=float, default=0.005, help="Adam learning rate for coordinates (default: 0.005)")
    parser.add_argument("--lr-b", type=float, default=0.2, help="Adam learning rate for B-factors (default: 0.2)")
    parser.add_argument("--lr-scale", type=float, default=0.001, help="Adam learning rate for scale (default: 0.001)")
    parser.add_argument("--lr-solvent", type=float, default=0.005, help="Adam learning rate for bulk solvent (default: 0.005)")
    parser.add_argument("--damping-factor", type=float, default=0.05, help="Hessian damping factor (default: 0.05)")
    parser.add_argument("--output-dir", default=str(DIR_1EE2), help="Output directory for results")
    parser.add_argument("--no-maps", action="store_true", help="Skip writing MTZ/CCP4 map files")
    parser.add_argument("--no-plot", action="store_true", help="Skip generating comparison plot")
    parser.add_argument("--force-download", action="store_true", help="Force re-download and re-conversion of 1EE2 files")

    opts = parser.parse_args(args)
    out_dir = Path(opts.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prepare_data(force_download=opts.force_download)

    modes_to_run = []
    if opts.mode in ("all", "f_based"):
        modes_to_run.append({
            "name": "Classic F-based",
            "target": "amplitude",
            "use_preconditioner": False,
            "prefix": str(out_dir / "1ee2_f_unprecond"),
        })
    if opts.mode in ("all", "precond_f"):
        modes_to_run.append({
            "name": "Precond F-based",
            "target": "amplitude",
            "use_preconditioner": True,
            "prefix": str(out_dir / "1ee2_f_precond"),
        })
    if opts.mode in ("all", "intensity"):
        modes_to_run.append({
            "name": "Intensity (unprecond)",
            "target": "intensity",
            "use_preconditioner": False,
            "prefix": str(out_dir / "1ee2_i_unprecond"),
        })
    if opts.mode in ("all", "precond_intensity"):
        modes_to_run.append({
            "name": "Intensity (precond)",
            "target": "intensity",
            "use_preconditioner": True,
            "prefix": str(out_dir / "1ee2_i_precond"),
        })

    results = []
    for m in modes_to_run:
        res = run_single_refinement(
            name=m["name"],
            target=m["target"],
            use_preconditioner=m["use_preconditioner"],
            steps=opts.steps,
            d_min=opts.d_min,
            n_bins=opts.n_bins,
            bulk_solvent=not opts.no_bulk_solvent,
            lr_sites=opts.lr_sites,
            lr_b=opts.lr_b,
            lr_scale=opts.lr_scale,
            lr_solvent=opts.lr_solvent,
            damping_factor=opts.damping_factor,
            prefix=m["prefix"],
            write_maps=not opts.no_maps,
        )
        results.append(res)

    print_comparison_table(results)

    if not opts.no_plot and len(results) > 1:
        plot_comparison(results, out_dir / "four_way_comparison.png")

    return 0


if __name__ == "__main__":
    sys.exit(main())
