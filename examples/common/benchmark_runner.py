"""Shared pipeline functions for model re-refinement and benchmark reporting."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:
    from cctbx.array_family import flex
except ImportError:
    flex = None

from phridge.client.intensity import from_files


def print_detailed_comparison(rep: Any) -> None:
    """Print a comprehensive audit set comparison report."""
    print("\n" + "-" * 85)
    print(f" AUDIT SET COMPARISON: {rep.name_B} vs {rep.name_A}")
    print("-" * 85)
    print(f"  Scored Audit Reflections: {rep.n_test} (Tune: {rep.n_tune}, Work: {rep.n_work})")
    print(f"  Model A NLL (audit):     {rep.nll_A:.4f} nats/refl")
    print(f"  Model B NLL (audit):     {rep.nll_B:.4f} nats/refl")
    print(f"  Delta NLL (B - A):       {rep.delta_nll:+.4f} nats/refl (Gain: {rep.delta_gain:+.4f} nats/refl)")
    print(f"  Total Log Bayes Factor:  {rep.total_delta_gain:+.1f} nats across {rep.n_test} audit reflections")
    print(f"  Bootstrap SE:            {rep.se_boot:.4f} (95% CI: [{rep.ci_boot[0]:+.4f}, {rep.ci_boot[1]:+.4f}])")
    print(f"  Naive SE:                {rep.se_naive:.4f} (Clustering Ratio: {rep.se_ratio:.2f}x)")
    print(f"  Win Fraction P(d_h > 0): {rep.win_fraction*100:.1f}% ({rep.wins_B} wins / {rep.wins_A} losses / {rep.ties} ties)")
    print(f"  Sign Tests:              McNemar p = {rep.mcnemar_p_value:.2e} | Wilcoxon p = {rep.wilcoxon_p_value:.2e}")
    print(f"  Likelihood Decomposition:")
    print(f"    Anchored at theta_A:   Structure = {rep.struct_term_A:+.4f} | Error-Model = {rep.error_term_B:+.4f} | Total = {rep.delta_nll:+.4f}")
    print(f"    Anchored at theta_B:   Structure = {rep.struct_term_B:+.4f} | Error-Model = {rep.error_term_A:+.4f} | Total = {rep.delta_nll:+.4f}")
    print(f"    Robustness:            {rep.structure_conclusion}")

    print(f"\n  Resolution Shell Breakdown:")
    print(f"  {'Shell':<5} | {'d_range (Å)':<14} | {'N_test':<6} | {'NLL(A)':<8} | {'NLL(B)':<8} | {'Delta Gain':<10} | {'Boot SE':<8} | {'Win %':<6}")
    print(f"  {'-'*5}-|-{'-'*14}-|-{'-'*6}-|-{'-'*8}-|-{'-'*8}-|-{'-'*10}-|-{'-'*8}-|-{'-'*6}")
    for idx, sh in enumerate(rep.shells):
        d_str = f"{sh['d_max']:.2f} - {sh['d_min']:.2f}"
        sh_idx = sh.get("shell_idx", idx + 1)
        n_sh = sh.get("n_refl", sh.get("n_test", 0))
        print(f"  {sh_idx:<5} | {d_str:<14} | {n_sh:<6} | {sh['nll_A']:8.4f} | {sh['nll_B']:8.4f} | {sh['delta_gain']:+10.4f} | {sh['se_boot']:8.4f} | {sh['win_fraction']*100:5.1f}%")


def generate_shaken_model(
    ref_pdb_path: Path,
    out_pdb_path: Path,
    mtz_ref_path: Path,
    pos_rmsd: float = 0.20,
    b_shake_fraction: float = 0.20,
    seed: int = 42,
    d_min: float = 2.2,
) -> Dict[str, Any]:
    """Generate a single reproducible shaken model with controlled positional and B shifts."""
    if flex is None:
        raise ImportError("cctbx is required to generate shaken crystallographic models")

    print("=" * 80)
    print(" GENERATING SHAKEN STARTING MODEL")
    print(f" Reference PDB:     {ref_pdb_path.name}")
    print(f" Positional shift:  {pos_rmsd:.3f} Å RMSD")
    print(f" B-factor shift:    +/- {b_shake_fraction*100:.1f}% fractional Gaussian")
    print(f" Random seed:       {seed}")
    print(f" Output PDB:        {out_pdb_path.name}")
    print("=" * 80)

    model = from_files(
        str(ref_pdb_path),
        str(mtz_ref_path),
        d_min=d_min,
        device="cpu",
        use_bulk_solvent=False,
        convert_to_isotropic=True,
    )

    rng = np.random.default_rng(seed)
    sites_orig = np.asarray(model.xray_structure.sites_cart(), dtype=np.float64)
    b_orig = np.array([float(sc.u_iso * 8.0 * np.pi**2) for sc in model.xray_structure.scatterers()])

    # 1. Positional perturbation with exact target RMSD
    noise_raw = rng.normal(loc=0.0, scale=1.0, size=sites_orig.shape)
    current_rmsd = float(np.sqrt(np.mean(noise_raw**2)))
    noise = noise_raw * (pos_rmsd / max(current_rmsd, 1e-12))
    sites_shaken = sites_orig + noise

    uc = model.xray_structure.unit_cell()
    sites_frac = uc.fractionalize(flex.vec3_double([tuple(r) for r in sites_shaken]))
    model.xray_structure.set_sites_frac(sites_frac)

    # 2. B-factor perturbation (fractional normal shake)
    b_noise = rng.normal(loc=0.0, scale=b_shake_fraction, size=b_orig.shape)
    b_shaken = np.clip(b_orig * (1.0 + b_noise), 1.0, 200.0)
    model.xray_structure.set_b_iso(values=flex.double(b_shaken.tolist()))

    if model.hierarchy is not None:
        model.hierarchy.adopt_xray_structure(model.xray_structure)

    out_pdb_path.parent.mkdir(parents=True, exist_ok=True)
    model.write_pdb(str(out_pdb_path))

    # Reload and verify
    m_reloaded = from_files(str(out_pdb_path), str(mtz_ref_path), d_min=d_min, device="cpu", use_bulk_solvent=False, convert_to_isotropic=True)
    sites_rel = np.asarray(m_reloaded.xray_structure.sites_cart(), dtype=np.float64)
    b_rel = np.array([float(sc.u_iso * 8.0 * np.pi**2) for sc in m_reloaded.xray_structure.scatterers()])

    # Classify atom selections
    atom_names = [a.name.strip() for a in model.hierarchy.atoms()]
    is_mc = np.array([name in ["N", "CA", "C", "O"] for name in atom_names])
    is_wat = np.array(["HOH" in a.fetch_labels().resname for a in model.hierarchy.atoms()])
    is_sc = ~is_mc & ~is_wat

    diff_all = float(np.sqrt(np.mean((sites_rel - sites_orig)**2)))
    diff_mc = float(np.sqrt(np.mean((sites_rel[is_mc] - sites_orig[is_mc])**2))) if np.any(is_mc) else diff_all
    diff_sc = float(np.sqrt(np.mean((sites_rel[is_sc] - sites_orig[is_sc])**2))) if np.any(is_sc) else diff_all
    b_diff = float(np.sqrt(np.mean((b_rel - b_orig)**2)))
    b_corr = float(np.corrcoef(b_orig, b_rel)[0, 1]) if np.std(b_orig) > 0 and np.std(b_rel) > 0 else 1.0

    gst = model.compute_geometry_statistics(sites_rel)

    stats = {
        "shaken_pdb": str(out_pdb_path),
        "target_pos_rmsd": float(pos_rmsd),
        "actual_pos_rmsd_1d": diff_all,
        "actual_pos_rmsd_mc": diff_mc,
        "actual_pos_rmsd_sc": diff_sc,
        "b_rmsd": b_diff,
        "b_corr": b_corr,
        "mean_b_orig": float(np.mean(b_orig)),
        "mean_b_shaken": float(np.mean(b_rel)),
        "bond_rms": gst["bond_rms"],
        "angle_rms": gst["angle_rms"],
        "planarity_rms": gst["planarity_rms"],
    }
    return stats


def run_rerefinement(
    target: str,
    pdb_path: Path,
    mtz_path: Path,
    out_pdb: Path,
    d_min: float = 2.2,
    macrocycles: int = 3,
    lbfgs_max_iter: int = 15,
    b_iterations: int = 5,
    weight_mode: str = "unit",
    xray_scale: float = 1.0,
    w_geom: float = 1.0,
    nu: float = 7.0,
    device: str = "auto",
    build_maps: bool = True,
) -> Dict[str, Any]:
    """Execute standard re-refinement macrocycle loop for a given target."""
    print("\n" + "=" * 75)
    print(f" RE-REFINING: TARGET = {target.upper()}")
    print(f" PDB: {pdb_path.name} | MTZ: {mtz_path.name}")
    print(f" Macrocycles: {macrocycles} | L-BFGS iter/cycle: {lbfgs_max_iter} | B iter/cycle: {b_iterations}")
    print(f" Weight mode: {weight_mode} (scale: {xray_scale}, w_geom: {w_geom}) | Device: {device}")
    print("=" * 75)

    t0 = time.time()
    model = from_files(
        pdb_path=str(pdb_path),
        mtz_path=str(mtz_path),
        d_min=d_min,
        device=device,
        use_bulk_solvent=True,
        convert_to_isotropic=True,
        target=target,
        nu=nu if target == "intensity" else None,
        estimate_nu=(target == "intensity"),
        nu_mode="global",
    )

    model.refine_scale_and_solvent()
    if target == "intensity":
        model.refine_sigma_a_and_nu(max_cycles=2, nu_mode="global")
    else:
        model.estimate_sigma_a()

    val0, _ = model.compute_target_and_gradients()
    summ0 = model.summary()
    sites_init = np.asarray(model.xray_structure.sites_cart(), dtype=np.float64)
    gst0 = model.compute_geometry_statistics(sites_init)

    print(f"\nInitial State:")
    print(f"  Target NLL (work): {val0:.6f} | Test NLL: {model.target_value_test:.6f}")
    print(f"  R_work: {summ0['r_work']*100:.2f}% | R_free: {summ0['r_free']*100:.2f}% | CC_free(I): {summ0['cc_free_i']:.4f}")
    print(f"  Bonds: {gst0['bond_rms']:.4f} Å | Angles: {gst0['angle_rms']:.2f}° | Planar: {gst0['planarity_rms']:.4f} Å [{ 'ACCEPTABLE' if gst0['is_acceptable'] else 'OUT-OF-TARGET' }]")

    model.refine_lbfgs(
        macrocycles=macrocycles,
        max_iterations_per_cycle=lbfgs_max_iter,
        regularize_geometry=False,
        xray_weight_mode=weight_mode,
        xray_scale=xray_scale,
        w_geom=w_geom,
        refine_scales=True,
        refine_b=True,
        use_adp_restraints=True,
        optimize_adp_weights=True,
        w_adp=1.0,
        b_iterations=b_iterations,
        refine_sites=True,
        refine_sigma_a=True,
        refine_nu=(target == "intensity"),
        polish_geometry=False,
        verbose=True,
    )

    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    model.write_pdb(str(out_pdb))
    print(f"\nWrote refined PDB to: {out_pdb.name}")

    if build_maps:
        map_prefix = str(out_pdb.with_suffix(""))
        built = model.write_maps(
            prefix=map_prefix,
            write_ccp4=True,
            compute_all=(target == "intensity"),
        )
        print(f"Built electron density maps ({len(built)} files written)")

    elapsed = time.time() - t0
    final_summ = model.summary()
    final_sites = np.asarray(model.xray_structure.sites_cart(), dtype=np.float64)
    final_gst = model.compute_geometry_statistics(final_sites)
    final_b = np.array([float(sc.u_iso * 8.0 * np.pi**2) for sc in model.xray_structure.scatterers()])
    b_init = np.array([float(sc.u_iso * 8.0 * np.pi**2) for sc in from_files(str(pdb_path), str(mtz_path), d_min=d_min, device="cpu").xray_structure.scatterers()])
    b_corr = float(np.corrcoef(final_b, b_init)[0, 1]) if np.std(final_b) > 0 and np.std(b_init) > 0 else 1.0

    mc_shift = float(np.sqrt(np.mean(np.sum((final_sites - sites_init)**2, axis=-1))))

    return {
        "target": target,
        "elapsed_s": elapsed,
        "nll_work_init": val0,
        "nll_free_init": float(model.target_value_test),
        "r_work_init": summ0["r_work"],
        "r_free_init": summ0["r_free"],
        "cc_free_init": summ0["cc_free_i"],
        "nll_work_final": final_summ["target_work"],
        "nll_free_final": final_summ["target_test"],
        "r_work_final": final_summ["r_work"],
        "r_free_final": final_summ["r_free"],
        "cc_free_final": final_summ["cc_free_i"],
        "b_mean_final": float(np.mean(final_b)),
        "b_std_final": float(np.std(final_b)),
        "b_corr_final": b_corr,
        "adp_energy_final": final_summ.get("adp_energy", float("nan")),
        "bond_rms": final_gst["bond_rms"],
        "angle_rms": final_gst["angle_rms"],
        "planarity_rms": final_gst["planarity_rms"],
        "chirality_rms": final_gst["chirality_rms"],
        "is_acceptable": final_gst["is_acceptable"],
        "mc_shift": mc_shift,
    }
