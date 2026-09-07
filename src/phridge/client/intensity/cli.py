"""Command-line interface and workflow pipelines for intensity refinement."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

try:
    import iotbx.pdb
    from iotbx.reflection_file_reader import any_reflection_file
except ImportError:
    iotbx = None
    any_reflection_file = None

from phridge.client.intensity.model import IntensityModel


def parse_omit_selection(sel_str: str) -> str:
    """Convert convenient user strings like '45-52', '45:52', '52' into cctbx selection syntax."""
    s = sel_str.strip()
    if "-" in s and not any(kw in s.lower() for kw in ["resseq", "chain", "name", "resid", "segid", "element"]):
        parts = s.split("-")
        if len(parts) == 2 and parts[0].strip().isdigit() and parts[1].strip().isdigit():
            return f"resseq {parts[0].strip()}:{parts[1].strip()}"
    if ":" in s and not any(kw in s.lower() for kw in ["resseq", "chain", "name", "resid", "segid", "element"]):
        return f"resseq {s}"
    if s.isdigit():
        return f"resseq {s}"
    return s


def apply_omit(
    hierarchy: Any,
    xray_structure: Any,
    omit_spec: str,
    omit_mode: str = "delete",
) -> tuple[Any, Any, int]:
    """Omit specified residues or atoms from hierarchy and xray_structure.

    Parameters
    ----------
    hierarchy : iotbx.pdb.hierarchy.root
    xray_structure : cctbx.xray.structure
    omit_spec : str
        Residue range (e.g. '45-52', '45:52') or cctbx selection string.
    omit_mode : str
        'delete' (remove atoms completely) or 'zero_occ' (set occupancy to 0.0).

    Returns
    -------
    (hierarchy, xray_structure, num_omitted_atoms)
    """
    parsed_sel = parse_omit_selection(omit_spec)
    cache = hierarchy.atom_selection_cache()
    sel = cache.selection(parsed_sel)
    n_omitted = sel.count(True)
    if n_omitted == 0:
        raise ValueError(
            f"Omit selection '{omit_spec}' (parsed as '{parsed_sel}') matched 0 atoms."
        )

    if omit_mode == "delete":
        keep_sel = ~sel
        xray_structure = xray_structure.select(keep_sel)
        hierarchy = hierarchy.select(keep_sel)
    elif omit_mode == "zero_occ":
        scs = xray_structure.scatterers()
        for i, flag in enumerate(sel):
            if flag:
                scs[i].occupancy = 0.0
                scs[i].flags.set_grad_site(False)
                scs[i].flags.set_grad_u_iso(False)
        for atom, flag in zip(hierarchy.atoms(), sel):
            if flag:
                atom.occ = 0.0
    else:
        raise ValueError(f"Unknown omit_mode: '{omit_mode}'. Choose 'delete' or 'zero_occ'.")

    return hierarchy, xray_structure, n_omitted


def from_files(
    pdb_path: str | Path,
    mtz_path: str | Path,
    intensity_label: Optional[str] = None,
    use_bulk_solvent: bool = True,
    n_bins: int = 10,
    d_min: Optional[float] = None,
    overlap_bins: int = 1,
    tv_norm: float = 0.0,
    enforce_monotonic: bool = False,
    interpolate_sigma_a: bool = True,
    nu: Optional[float] = None,
    estimate_nu: bool = False,
    nu_mode: str = "global",
    target: str = "intensity",
    device: str = "auto",
    omit: Optional[str] = None,
    omit_mode: str = "delete",
    convert_to_isotropic: bool = False,
) -> IntensityModel:
    """Instantiate an IntensityModel from input PDB and MTZ files."""
    if any_reflection_file is None or iotbx is None:
        raise ImportError("cctbx/iotbx is required to load models and reflections from files")
    reader = any_reflection_file(str(mtz_path))
    miller_arrays = reader.as_miller_arrays()

    i_obs = None
    r_free = None

    for arr in miller_arrays:
        labels = [l.lower() for l in (arr.info().labels if arr.info() else [])]
        if intensity_label is not None:
            if any(intensity_label.lower() in l for l in labels):
                i_obs = arr
        elif arr.is_xray_intensity_array():
            if i_obs is None:
                i_obs = arr
        elif any(l in ["iobs", "i", "imean", "i-obs", "i_obs", "iobs(+)", "iobs(-)"] for l in labels):
            if i_obs is None:
                i_obs = arr
        elif arr.is_real_array() and not arr.is_bool_array() and not any("free" in l or "test" in l or "flag" in l for l in labels):
            if i_obs is None and not arr.is_complex_array():
                i_obs = arr

        if any("free" in l for l in labels) or any("test" in l for l in labels) or any("status" in l for l in labels):
            if arr.is_bool_array():
                r_free = arr
            elif arr.is_integer_array():
                data = arr.data()
                val0 = (data == 0).count(True)
                val1 = (data == 1).count(True)
                r_free = arr.customized_copy(data=(data == 0) if val0 < val1 else (data == 1))
            elif arr.is_string_array():
                data = arr.data()
                is_free = (data == "f") | (data == "F") | (data == "free")
                if is_free.count(True) > 0:
                    r_free = arr.customized_copy(data=is_free)

    if i_obs is None:
        raise ValueError(
            f"No intensity array found in {mtz_path}. "
            "Please specify --intensity-label matching one of the MTZ column labels."
        )

    i_obs.set_observation_type_xray_intensity()

    if i_obs.anomalous_flag():
        i_obs = i_obs.as_non_anomalous_array().merge_equivalents().array()

    i_obs = i_obs.map_to_asu()

    if r_free is not None:
        if r_free.anomalous_flag():
            r_free = r_free.as_non_anomalous_array().merge_equivalents().array()
        r_free = r_free.map_to_asu()

    mtz_cs = i_obs.crystal_symmetry()

    pdb_inp = iotbx.pdb.input(file_name=str(pdb_path))
    pdb_cs = pdb_inp.crystal_symmetry()
    effective_cs = pdb_cs if (pdb_cs is not None and pdb_cs.unit_cell() is not None) else mtz_cs

    restraints_manager = None
    try:
        import mmtbx.model
        from libtbx.utils import null_out

        m = mmtbx.model.manager(model_input=pdb_inp, crystal_symmetry=effective_cs, log=null_out())
        m.process(make_restraints=True)
        xray_structure = m.get_xray_structure()
        hierarchy = m.get_hierarchy()
        rm = m.get_restraints_manager()
        restraints_manager = rm.geometry if rm is not None else None
    except Exception:
        xray_structure = pdb_inp.xray_structure_simple(crystal_symmetry=effective_cs)
        hierarchy = pdb_inp.construct_hierarchy()

    if effective_cs is not None and (xray_structure.unit_cell() is None or not xray_structure.crystal_symmetry().is_compatible_unit_cell()):
        xray_structure = xray_structure.customized_copy(crystal_symmetry=effective_cs)

    if convert_to_isotropic:
        xray_structure.convert_to_isotropic()
        if hierarchy is not None:
            hierarchy.adopt_xray_structure(xray_structure)

    if omit:
        hierarchy, xray_structure, n_omitted = apply_omit(
            hierarchy=hierarchy,
            xray_structure=xray_structure,
            omit_spec=omit,
            omit_mode=omit_mode,
        )
        print(f"Applied omit selection '{omit}' ({omit_mode} mode): {n_omitted} atoms omitted.")
        try:
            import mmtbx.model
            from libtbx.utils import null_out

            pdb_str = hierarchy.as_pdb_string(crystal_symmetry=xray_structure.crystal_symmetry())
            inp_omit = iotbx.pdb.input(source_info=None, lines=pdb_str.splitlines())
            m_omit = mmtbx.model.manager(model_input=inp_omit, crystal_symmetry=xray_structure.crystal_symmetry(), log=null_out())
            m_omit.process(make_restraints=True)
            rm = m_omit.get_restraints_manager()
            restraints_manager = rm.geometry if rm is not None else None
        except Exception:
            restraints_manager = None

    return IntensityModel(
        xray_structure=xray_structure,
        i_obs=i_obs,
        hierarchy=hierarchy,
        r_free_flags=r_free,
        use_bulk_solvent=use_bulk_solvent,
        n_bins=n_bins,
        d_min=d_min,
        overlap_bins=overlap_bins,
        tv_norm=tv_norm,
        enforce_monotonic=enforce_monotonic,
        interpolate_sigma_a=interpolate_sigma_a,
        nu=nu,
        estimate_nu=estimate_nu,
        nu_mode=nu_mode,
        target_type=target,
        restraints_manager=restraints_manager,
        device=device,
    )


def run_intensity_pipeline(
    pdb_path: str | Path,
    mtz_path: str | Path,
    prefix: str = "output",
    intensity_label: Optional[str] = None,
    use_bulk_solvent: bool = True,
    n_bins: int = 10,
    d_min: Optional[float] = None,
    overlap_bins: int = 1,
    tv_norm: float = 0.0,
    enforce_monotonic: bool = False,
    interpolate_sigma_a: bool = True,
    nu: Optional[float] = None,
    estimate_nu: bool = False,
    nu_mode: str = "global",
    target: str = "intensity",
    max_iterations: int = 0,
    refine_mode: str = "sites",
    macrocycles: int = 3,
    lbfgs_max_iter: int = 20,
    regularize_geometry: bool = True,
    polish_geometry: bool = True,
    xray_weight_mode: str = "hessian",
    xray_scale: float = 1.0,
    refine_b: int = 0,
    b_step_scale: float = 2000.0,
    max_shift_b: float = 5.0,
    min_b: float = 1.0,
    max_b: float = 200.0,
    convert_to_isotropic: bool = False,
    lr_sites: float = 0.005,
    lr_b: float = 0.2,
    lr_scale: float = 0.001,
    lr_solvent: float = 0.005,
    w_geom: Optional[float] = None,
    geom_scale: float = 0.5,
    refine_scales: bool = True,
    refine_sigma_a: bool = True,
    refine_nu: bool = True,
    sigma_a_interval: int = 2,
    use_preconditioner: bool = True,
    preconditioner_interval: int = 5,
    damping_factor: float = 0.05,
    device: str = "auto",
    view: bool = False,
    omit: Optional[str] = None,
    omit_mode: str = "delete",
    shake_sites: Optional[float] = None,
    shake_b: bool = False,
    b_shake_fraction: float = 0.25,
    reset_b: bool = True,
    reset_b_value: Optional[float] = None,
) -> Dict[str, Any]:
    """Execute the full intensity refinement and map generation workflow."""
    is_iso = convert_to_isotropic or (refine_mode in ("b_iso", "adam", "lbfgs"))
    model = from_files(
        pdb_path=pdb_path,
        mtz_path=mtz_path,
        intensity_label=intensity_label,
        use_bulk_solvent=use_bulk_solvent,
        n_bins=n_bins,
        d_min=d_min,
        overlap_bins=overlap_bins,
        tv_norm=tv_norm,
        enforce_monotonic=enforce_monotonic,
        interpolate_sigma_a=interpolate_sigma_a,
        nu=nu,
        estimate_nu=estimate_nu,
        nu_mode=nu_mode,
        target=target,
        device=device,
        omit=omit,
        omit_mode=omit_mode,
        convert_to_isotropic=is_iso,
    )

    if shake_sites is not None and shake_sites > 0:
        print(f"Shaking atomic coordinates (RMSD = {shake_sites:.2f} Å)...")
        model.shake_sites(rmsd=shake_sites)
    if shake_b:
        print(f"Resetting and shaking B-factors (+/- {b_shake_fraction*100:.1f}% fractional, reset={reset_b})...")
        model.shake_b_iso(fraction=b_shake_fraction, reset=reset_b, reset_to_value=reset_b_value)

    print(f"Loaded {pdb_path} and {mtz_path}")
    print(f"Refining scale factor (bulk_solvent={use_bulk_solvent})...")
    scales = model.refine_scale_and_solvent()
    print(f"  Fitted scale: k_total = {scales['k_total']:.4e}")
    if use_bulk_solvent:
        print(f"  Fitted solvent: k_sol = {scales['k_sol']:.3f}, B_sol = {scales['b_sol']:.1f} Å²")

    info_details = []
    if overlap_bins > 0:
        info_details.append(f"overlap={overlap_bins}")
    else:
        info_details.append("disjoint bins")
    if tv_norm > 0:
        info_details.append(f"tv_lambda={tv_norm}")
    if enforce_monotonic:
        info_details.append("monotonic")
    sa_mode_msg = f" ({', '.join(info_details)})" if info_details else ""

    if estimate_nu:
        print(f"Co-refining sigma_A and Student-t nu (mode={nu_mode}, overlap={overlap_bins})...")
        sa_bins, _ = model.refine_sigma_a_and_nu(
            max_cycles=2,
            nu_mode=nu_mode,
            overlap=overlap_bins,
            tv_lambda=tv_norm,
            enforce_monotonic=enforce_monotonic,
            interpolate=interpolate_sigma_a,
        )
        for b_idx, sa_val in sorted(sa_bins.items()):
            d_range = model.binner.bin_d_range(b_idx)
            nu_str = f", nu = {model.nu_binned[b_idx]:.2f}" if b_idx in model.nu_binned else ""
            print(f"  Bin {b_idx} (d={d_range[0]:.2f}-{d_range[1]:.2f} Å): sigma_A = {sa_val:.4f}{nu_str}")
        if nu_mode == "global" and model.nu is not None:
            print(f"  Global dataset-wide fitted nu = {model.nu:.2f}")
    elif model.target_type == "intensity" and model.nu is not None:
        print(f"Estimating sigma_A with Student-t nu = {model.nu:.2f} across {n_bins} resolution bins{sa_mode_msg}...")
        sa_bins = model.estimate_sigma_a(
            overlap=overlap_bins,
            tv_lambda=tv_norm,
            enforce_monotonic=enforce_monotonic,
            interpolate=interpolate_sigma_a,
        )
        for b_idx, sa_val in sorted(sa_bins.items()):
            d_range = model.binner.bin_d_range(b_idx)
            print(f"  Bin {b_idx} (d={d_range[0]:.2f}-{d_range[1]:.2f} Å): sigma_A = {sa_val:.4f}")
    else:
        print(f"Estimating sigma_A across {n_bins} resolution bins{sa_mode_msg}...")
        sa_bins = model.estimate_sigma_a(
            overlap=overlap_bins,
            tv_lambda=tv_norm,
            enforce_monotonic=enforce_monotonic,
            interpolate=interpolate_sigma_a,
        )
        for b_idx, sa_val in sorted(sa_bins.items()):
            d_range = model.binner.bin_d_range(b_idx)
            print(f"  Bin {b_idx} (d={d_range[0]:.2f}-{d_range[1]:.2f} Å): sigma_A = {sa_val:.4f}")

    if refine_mode == "lbfgs":
        print(f"Running L-BFGS refinement ({macrocycles} macrocycles, {lbfgs_max_iter} iter/cycle, weight_mode={xray_weight_mode})...")
        model.refine_lbfgs(
            macrocycles=macrocycles,
            max_iterations_per_cycle=lbfgs_max_iter,
            regularize_geometry=regularize_geometry,
            polish_geometry=polish_geometry,
            xray_weight_mode=xray_weight_mode,
            xray_scale=xray_scale,
            refine_scales=True,
            refine_b=True,
            use_adp_restraints=True,
            optimize_adp_weights=True,
            refine_sites=True,
            refine_sigma_a=refine_sigma_a,
            refine_nu=refine_nu,
            b_iterations=5,
            min_b=min_b,
            max_b=max_b,
            verbose=True,
        )
    elif refine_mode == "adam":
        steps = max_iterations if max_iterations > 0 else 10
        print(f"Running Adam refinement for {steps} steps (lr_sites={lr_sites}, lr_b={lr_b}, precond={use_preconditioner})...")
        model.refine_adam(
            max_iterations=steps,
            lr_sites=lr_sites,
            lr_b=lr_b,
            lr_scale=lr_scale,
            lr_solvent=lr_solvent,
            w_geom=w_geom,
            geom_scale=geom_scale,
            refine_scales=refine_scales,
            refine_b=True,
            refine_sites=True,
            refine_sigma_a=refine_sigma_a,
            refine_nu=refine_nu,
            sigma_a_interval=sigma_a_interval,
            use_preconditioner=use_preconditioner,
            preconditioner_interval=preconditioner_interval,
            damping_factor=damping_factor,
            min_b=min_b,
            max_b=max_b,
            verbose=True,
        )
    elif max_iterations > 0 or refine_b > 0:
        if refine_mode in ("sites", "both") and max_iterations > 0:
            print(f"Refining coordinates for {max_iterations} steps...")
            model.refine_coordinates(max_iterations=max_iterations)
        if refine_mode in ("b_iso", "both") or refine_b > 0:
            b_steps = refine_b if refine_b > 0 else max_iterations
            print(f"Refining isotropic B-factors for {b_steps} steps...")
            model.refine_b_iso(
                max_iterations=b_steps,
                step_scale=b_step_scale,
                max_shift_b=max_shift_b,
                min_b=min_b,
                max_b=max_b,
            )

    model.print_summary()

    print(f"Synthesizing maps (prefix='{prefix}')...")
    files = model.write_maps(prefix=prefix)
    refined_pdb = f"{prefix}_refined.pdb"
    model.write_pdb(refined_pdb)
    files["pdb"] = refined_pdb
    print(f"Wrote output files: {files}")

    if view:
        from phridge.client.viewer import view_refinement_result
        fofc_path = Path(files["fofc"]) if "fofc" in files else None
        view_refinement_result(
            model_pdb=Path(refined_pdb),
            two_fofc_map=Path(files["2fofc"]),
            fofc_map=fofc_path,
            output_html=Path(f"{prefix}_viewer.html"),
            open_browser=True,
        )

    return {"model": model, "files": files, "summary": model.summary()}


def main(args: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Intensity-Based Crystallographic Refinement & Map Generation Tool (phridge)",
    )
    parser.add_argument("pdb", help="Path to input PDB file")
    parser.add_argument("mtz", help="Path to input MTZ reflection file with intensities")
    parser.add_argument("--prefix", default="output", help="Output file prefix (default: output)")
    parser.add_argument("--intensity-label", default=None, help="Label of intensity column in MTZ")
    parser.add_argument("--bulk-solvent", action="store_true", default=True, help="Enable cctbx map-gridded bulk solvent modeling (default: True)")
    parser.add_argument("--no-bulk-solvent", action="store_false", dest="bulk_solvent", help="Disable cctbx map-gridded bulk solvent modeling")
    parser.add_argument("--n-bins", type=int, default=10, help="Number of resolution bins (default: 10)")
    parser.add_argument("--d-min", type=float, default=None, help="High resolution limit in Angstroms")
    parser.add_argument("--overlap-bins", type=int, default=1, help="Number of adjacent resolution bins to include on each side for overlapping sigma_A estimation (default: 1; set to 0 for disjoint bins)")
    parser.add_argument("--tv-norm", type=float, default=0.0, help="Total Variation regularization weight lambda_TV across resolution bins (default: 0.0; e.g. 0.05 or 0.1)")
    parser.add_argument("--enforce-monotonic", action="store_true", default=False, help="Enforce monotonic non-increasing sigma_A across resolution via isotonic regression")
    parser.add_argument("--no-interpolate-sigma-a", action="store_true", default=False, help="Disable continuous resolution interpolation of sigma_A across reflections")
    parser.add_argument("--nu", type=float, default=None, help="Degrees of freedom for Student-t noise (e.g. 7.0; default: 7.0 when target is intensity)")
    parser.add_argument("--estimate-nu", action="store_true", default=False, help="Estimate / refine Student-t degrees of freedom nu directly from intensities")
    parser.add_argument("--nu-mode", default="global", choices=["global", "binned"], help="Nu estimation mode: 'global' (single scalar for full dataset, default: global) or 'binned' (per resolution shell)")
    parser.add_argument("--target", default="intensity", choices=["intensity", "amplitude", "ml_i", "ml_f"], help="Refinement target: 'intensity' (ml_i) or 'amplitude' (ml_f) (default: intensity)")
    parser.add_argument("--refine", type=int, default=0, dest="max_iterations", help="Number of refinement steps (default: 0)")
    parser.add_argument("--refine-mode", default="sites", choices=["sites", "b_iso", "both", "adam", "lbfgs"], help="Parameters to refine: 'sites' (coordinates), 'b_iso' (isotropic B-factors only, sites fixed), 'both', 'adam' (Adam optimizer over sites, B-factors, scales with Hessian preconditioner), or 'lbfgs' (L-BFGS quasi-Newton optimizer with strong Wolfe line search respecting geometry restraints, default: sites)")
    parser.add_argument("--macrocycles", type=int, default=3, help="Number of macrocycles for L-BFGS refinement (default: 3)")
    parser.add_argument("--lbfgs-max-iter", type=int, default=20, help="Maximum L-BFGS iterations per coordinate cycle (default: 20)")
    parser.add_argument("--xray-scale", type=float, default=1.0, help="Scale factor multiplier for X-ray weight in L-BFGS/Adam (default: 1.0)")
    parser.add_argument("--xray-weight-mode", default="hessian", choices=["hessian", "gradient", "fixed"], help="Weighting mode for X-ray vs geometry in L-BFGS/Adam (default: hessian)")
    parser.add_argument("--no-regularize-geometry", action="store_true", default=False, help="Disable pre-refinement geometry regularization in L-BFGS")
    parser.add_argument("--no-polish-geometry", action="store_true", default=False, help="Disable post-refinement geometry polish in L-BFGS")
    parser.add_argument("--refine-b", type=int, default=0, help="Number of B-factor refinement steps")
    parser.add_argument("--b-step-scale", type=float, default=2000.0, help="Step scale for B-factor refinement (default: 2000.0)")
    parser.add_argument("--max-shift-b", type=float, default=5.0, help="Maximum B-factor shift per step in Å² (default: 5.0)")
    parser.add_argument("--min-b", type=float, default=1.0, help="Minimum allowed B-factor in Å² (default: 1.0)")
    parser.add_argument("--max-b", type=float, default=200.0, help="Maximum allowed B-factor in Å² (default: 200.0)")
    parser.add_argument("--convert-to-isotropic", action="store_true", default=False, help="Convert model to isotropic B-factors before refinement")
    parser.add_argument("--lr-sites", type=float, default=0.005, help="Adam learning rate for Cartesian coordinates (default: 0.005)")
    parser.add_argument("--lr-b", type=float, default=0.2, help="Adam learning rate for isotropic B-factors (default: 0.2)")
    parser.add_argument("--lr-scale", type=float, default=0.001, help="Adam learning rate for overall scale factor (default: 0.001)")
    parser.add_argument("--lr-solvent", type=float, default=0.005, help="Adam learning rate for bulk solvent parameters (default: 0.005)")
    parser.add_argument("--w-geom", type=float, default=None, help="Explicit geometry restraint weight (default: None, auto-scaled from gradient RMS ratio)")
    parser.add_argument("--geom-scale", type=float, default=0.5, help="Scale multiplier for auto-weighted geometry restraints (default: 0.5)")
    parser.add_argument("--sigma-a-interval", type=int, default=2, help="Interval in Adam steps between sigma_A and nu re-estimation (default: 2)")
    parser.add_argument("--damping-factor", type=float, default=0.05, help="Damping factor for Hessian preconditioner (default: 0.05)")
    parser.add_argument("--no-preconditioner", action="store_true", default=False, help="Disable Hessian preconditioning in Adam refinement")
    parser.add_argument("--preconditioner-interval", type=int, default=5, help="Interval in steps to recompute Hessian preconditioner (default: 5)")
    parser.add_argument("--no-refine-sigma-a", action="store_true", default=False, help="Disable periodic sigma_A refinement during Adam optimization")
    parser.add_argument("--no-refine-nu", action="store_true", default=False, help="Disable periodic nu refinement during Adam optimization")
    parser.add_argument("--omit", default=None, help="Residues or atom selection to omit (e.g. '45-52', '45:52', or 'resseq 45:52')")
    parser.add_argument("--omit-mode", default="delete", choices=["delete", "zero_occ"], help="Omit treatment: 'delete' (remove atoms) or 'zero_occ' (set occupancy to 0.0)")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"], help="Compute device for phridge (auto, mps, cuda, cpu)")
    parser.add_argument("--view", action="store_true", help="Launch browser-based 3D viewer (Mol*) to inspect maps and model")
    parser.add_argument("--shake-sites", type=float, default=None, help="Shake coordinates with target RMSD in Å before refinement (e.g. 0.10)")
    parser.add_argument("--shake-b", action="store_true", default=False, help="Shake isotropic B-factors before refinement")
    parser.add_argument("--b-shake-fraction", type=float, default=0.25, help="Fractional shake magnitude for B-factors (default: 0.25 for +/- 25%%)")
    parser.add_argument("--reset-b", action="store_true", default=True, help="Reset B-factors to mean B before applying fractional shake (default: True)")
    parser.add_argument("--no-reset-b", action="store_false", dest="reset_b", help="Shake existing B-factors without resetting to mean B")
    parser.add_argument("--reset-b-value", type=float, default=None, help="Explicit base B-factor in Å² to reset to before shaking (default: mean B)")

    opts = parser.parse_args(args)

    try:
        run_intensity_pipeline(
            pdb_path=opts.pdb,
            mtz_path=opts.mtz,
            prefix=opts.prefix,
            intensity_label=opts.intensity_label,
            use_bulk_solvent=opts.bulk_solvent,
            n_bins=opts.n_bins,
            d_min=opts.d_min,
            overlap_bins=opts.overlap_bins,
            tv_norm=opts.tv_norm,
            enforce_monotonic=opts.enforce_monotonic,
            interpolate_sigma_a=not opts.no_interpolate_sigma_a,
            nu=opts.nu,
            estimate_nu=opts.estimate_nu,
            nu_mode=opts.nu_mode,
            target=opts.target,
            max_iterations=opts.max_iterations,
            refine_mode=opts.refine_mode,
            macrocycles=opts.macrocycles,
            lbfgs_max_iter=opts.lbfgs_max_iter,
            regularize_geometry=not opts.no_regularize_geometry,
            polish_geometry=not opts.no_polish_geometry,
            xray_weight_mode=opts.xray_weight_mode,
            xray_scale=opts.xray_scale,
            refine_b=opts.refine_b,
            b_step_scale=opts.b_step_scale,
            max_shift_b=opts.max_shift_b,
            min_b=opts.min_b,
            max_b=opts.max_b,
            convert_to_isotropic=opts.convert_to_isotropic,
            lr_sites=opts.lr_sites,
            lr_b=opts.lr_b,
            lr_scale=opts.lr_scale,
            lr_solvent=opts.lr_solvent,
            w_geom=opts.w_geom,
            geom_scale=opts.geom_scale,
            refine_sigma_a=not opts.no_refine_sigma_a,
            refine_nu=not opts.no_refine_nu,
            sigma_a_interval=opts.sigma_a_interval,
            use_preconditioner=not opts.no_preconditioner,
            preconditioner_interval=opts.preconditioner_interval,
            damping_factor=opts.damping_factor,
            omit=opts.omit,
            omit_mode=opts.omit_mode,
            device=opts.device,
            view=opts.view,
            shake_sites=opts.shake_sites,
            shake_b=opts.shake_b,
            b_shake_fraction=opts.b_shake_fraction,
            reset_b=opts.reset_b,
            reset_b_value=opts.reset_b_value,
        )
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
