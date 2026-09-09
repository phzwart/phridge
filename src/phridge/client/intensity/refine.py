"""Optimization and refinement routines mixin for IntensityModel."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.optimize
import torch
try:
    from cctbx.array_family import flex
except ImportError:
    flex = None

from phridge.sfcalc.targets import Observations


class IntensityRefineMixin:
    """Provides coordinate, B-factor, scale, solvent, and joint optimization loops."""

    def update_coordinates_step(self: Any, step_scale: float = 0.001, max_shift_angstrom: float = 0.1) -> float:
        """Apply a line-search gradient step to fractional coordinates."""
        if self.gradients is None:
            self.compute_target_and_gradients()

        g_frac = np.asarray(self.gradients.raw.d_site_frac, dtype=np.float64)
        uc = self.xray_structure.unit_cell()
        ortho = np.asarray(uc.orthogonalization_matrix(), dtype=np.float64).reshape(3, 3)

        # Shift in Cartesian Angstroms
        shift_cart = -step_scale * (g_frac @ ortho.T)
        norm_cart = np.linalg.norm(shift_cart, axis=1)
        max_norm = float(np.max(norm_cart)) if len(norm_cart) > 0 else 0.0

        if max_norm > max_shift_angstrom:
            shift_cart *= max_shift_angstrom / max_norm

        fract = np.asarray(uc.fractionalization_matrix(), dtype=np.float64).reshape(3, 3)
        delta_frac = shift_cart @ fract.T

        current_frac = np.asarray(list(self.xray_structure.sites_frac()), dtype=np.float64)
        new_frac = current_frac + delta_frac
        self.xray_structure.set_sites_frac(flex.vec3_double([tuple(r) for r in new_frac]))

        if self.hierarchy is not None:
            self.hierarchy.adopt_xray_structure(self.xray_structure)

        # Recompute structure factors and target
        self.compute_f_calc()
        fmod_complex = self._assemble_f_model(self.k_total, self.k_sol, self.b_sol)
        c_arr = np.ascontiguousarray(fmod_complex, dtype=np.complex128)
        self.f_model = self.i_obs.customized_copy(data=flex.complex_double(c_arr))
        new_target, _ = self.compute_target_and_gradients()
        return new_target

    def refine_coordinates(self: Any, max_iterations: int = 5, step_scale: float = 0.001) -> List[float]:
        """Perform iterative coordinate refinement steps."""
        for sc in self.xray_structure.scatterers():
            sc.flags.set_grad_site(sc.occupancy > 0.0)
            sc.flags.set_grad_u_iso(False)

        history = [self.target_value]
        for it in range(max_iterations):
            val = self.update_coordinates_step(step_scale=step_scale)
            history.append(val)
        return history

    def convert_to_isotropic(self: Any) -> None:
        """Convert all scatterers to isotropic B-factors."""
        self.xray_structure.convert_to_isotropic()
        for sc in self.xray_structure.scatterers():
            if sc.occupancy > 0.0:
                sc.flags.set_grad_u_iso(True)
            else:
                sc.flags.set_grad_u_iso(False)
        if self.hierarchy is not None:
            self.hierarchy.adopt_xray_structure(self.xray_structure)

    def shake_b_iso(
        self: Any,
        fraction: float = 0.25,
        reset: bool = True,
        reset_to_value: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
        min_b: float = 1.0,
        max_b: float = 200.0,
        distribution: str = "uniform",
        update_model: bool = True,
    ) -> np.ndarray:
        """Reset and/or shake isotropic B-factors with fractional noise (e.g. +/- 25%)."""
        if any(sc.flags.use_u_aniso() for sc in self.xray_structure.scatterers()):
            self.convert_to_isotropic()

        if rng is None:
            rng = np.random.default_rng()

        b_curr = np.array([float(sc.u_iso * 8.0 * np.pi**2) for sc in self.xray_structure.scatterers()])

        if reset:
            b_base = float(reset_to_value) if reset_to_value is not None else float(np.mean(b_curr))
            b_ref = np.full_like(b_curr, b_base)
        else:
            b_ref = b_curr

        if distribution == "normal":
            delta = rng.normal(loc=0.0, scale=fraction, size=b_ref.shape)
        else:
            delta = rng.uniform(-fraction, fraction, size=b_ref.shape)

        b_shaken = np.clip(b_ref * (1.0 + delta), min_b, max_b).astype(np.float64)

        if update_model:
            b_arr = np.ascontiguousarray(b_shaken, dtype=np.float64)
            self.xray_structure.set_b_iso(values=flex.double(b_arr))
            if self.hierarchy is not None:
                self.hierarchy.adopt_xray_structure(self.xray_structure)
            self.compute_f_calc()
            fmod_c = self._assemble_f_model(self.k_total, self.k_sol, self.b_sol)
            c_arr = np.ascontiguousarray(fmod_c, dtype=np.complex128)
            self.f_model = self.i_obs.customized_copy(data=flex.complex_double(c_arr))
            self.compute_target_and_gradients()

        return b_shaken

    def shake_sites(
        self: Any,
        rmsd: float = 0.10,
        rng: Optional[np.random.Generator] = None,
        update_model: bool = True,
    ) -> np.ndarray:
        """Perturb atomic coordinates with random 3D Gaussian displacement of specified target RMSD."""
        if rng is None:
            rng = np.random.default_rng()

        sites_curr = np.asarray(self.xray_structure.sites_cart(), dtype=np.float64)
        n_atoms = len(sites_curr)
        if n_atoms == 0:
            return sites_curr

        sigma_1d = float(rmsd) / math.sqrt(3.0)
        disp = rng.normal(loc=0.0, scale=sigma_1d, size=sites_curr.shape)
        actual_rmsd = float(np.sqrt(np.mean(np.sum(disp**2, axis=1))))
        if actual_rmsd > 1e-12:
            disp *= float(rmsd) / actual_rmsd

        sites_shaken = sites_curr + disp

        if update_model:
            uc = self.xray_structure.unit_cell()
            fract = np.asarray(uc.fractionalization_matrix(), dtype=np.float64).reshape(3, 3)
            new_frac = (sites_shaken @ fract.T).astype(np.float64)
            self.xray_structure.set_sites_frac(flex.vec3_double([tuple(r) for r in new_frac]))
            if self.hierarchy is not None:
                self.hierarchy.adopt_xray_structure(self.xray_structure)
            self.compute_f_calc()
            fmod_c = self._assemble_f_model(self.k_total, self.k_sol, self.b_sol)
            c_arr = np.ascontiguousarray(fmod_c, dtype=np.complex128)
            self.f_model = self.i_obs.customized_copy(data=flex.complex_double(c_arr))
            self.compute_target_and_gradients()

        return sites_shaken

    def shake(
        self: Any,
        sites_rmsd: Optional[float] = None,
        b_fraction: Optional[float] = None,
        b_reset: bool = True,
        b_reset_to: Optional[float] = None,
        seed: Optional[int] = 42,
        min_b: float = 1.0,
        max_b: float = 200.0,
        *,
        rmsd: Optional[float] = None,
        reset_b: Optional[bool] = None,
        rng: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Convenience function to perturbation-test a model by shaking coordinates, B-factors, or both."""
        if rmsd is not None and sites_rmsd is None:
            sites_rmsd = rmsd
        if reset_b is not None:
            b_reset = reset_b
        if rng is None:
            rng = np.random.default_rng(seed)
        result: Dict[str, Any] = {"seed": seed}

        if sites_rmsd is not None and sites_rmsd > 0:
            sites_shaken = self.shake_sites(
                rmsd=sites_rmsd,
                rng=rng,
                update_model=True,
            )
            sites_ref = np.asarray(self.xray_structure.sites_cart(), dtype=np.float64)
            actual_disp = np.linalg.norm(sites_shaken - sites_ref, axis=1)
            result["sites_shaken"] = sites_shaken
            result["sites_rmsd_target"] = sites_rmsd
            result["sites_rmsd_actual"] = float(np.sqrt(np.mean(actual_disp**2)))

        if b_fraction is not None and b_fraction > 0:
            b_shaken = self.shake_b_iso(
                fraction=b_fraction,
                reset=b_reset,
                reset_to_value=b_reset_to,
                rng=rng,
                min_b=min_b,
                max_b=max_b,
                update_model=True,
            )
            result["b_shaken"] = b_shaken
            result["b_fraction"] = b_fraction
            result["b_reset"] = b_reset
            result["b_shaken"] = b_shaken
            result["b_mean"] = float(np.mean(b_shaken))

        return result

    def update_b_iso_step(
        self: Any,
        step_scale: Optional[float] = None,
        max_shift_b: float = 5.0,
        max_shift_log_b: float = 0.25,
        min_b: float = 1.0,
        max_b: float = 200.0,
        p_b: Optional[np.ndarray] = None,
        use_adp_prior: bool = True,
        w_adp: float = 1.0,
        damping: float = 1e-3,
    ) -> float:
        """Apply a gradient descent / preconditioned step with line search to isotropic B-factors."""
        from phridge.worker.geometry.curvature import GaussNewtonPreconditioner

        if any(sc.flags.use_u_aniso() for sc in self.xray_structure.scatterers()):
            self.convert_to_isotropic()
            self.compute_f_calc()
            fmod_c = self._assemble_f_model(self.k_total, self.k_sol, self.b_sol)
            c_arr = np.ascontiguousarray(fmod_c, dtype=np.complex128)
            self.f_model = self.i_obs.customized_copy(data=flex.complex_double(c_arr))
            self.compute_target_and_gradients()
        elif self.gradients is None:
            self.compute_target_and_gradients()

        if use_adp_prior:
            if self.adp_prior is None:
                self.build_adp_restraints()

            X = np.asarray(self.xray_structure.sites_cart(), dtype=np.float64)
            u_vals = np.array([float(sc.u_iso) for sc in self.xray_structure.scatterers()])
            b_curr = u_vals * (8.0 * np.pi**2)
            beta_curr = np.log(np.maximum(b_curr, 1e-4))

            g_u = np.asarray(self.gradients.raw.d_u_iso, dtype=np.float64)
            g_xray_beta = g_u * u_vals

            if p_b is None:
                try:
                    if self.target_eval is None or getattr(self.target_eval, "curv_radial", None) is None:
                        self.compute_target_and_gradients(compute_curvature=True)
                    _, p_b = self.compute_hessian_preconditioners(self.target_eval)
                except Exception:
                    p_b = None

            if p_b is not None:
                h_u = 1.0 / np.maximum(p_b, 1e-12)
            else:
                h_u = np.ones_like(u_vals)
            h_xray_beta = (u_vals**2) * h_u

            n_work = float(len(self.i_obs.data()) - (int(np.sum(self.r_free_flags.data())) if self.r_free_flags else 0))
            w_adp_eff = float(w_adp) / max(n_work, 1.0)

            g_adp_beta = self.adp_prior.gradient(beta_curr, sites=X).detach().cpu().numpy()
            g_tot = g_xray_beta + w_adp_eff * g_adp_beta

            prec = GaussNewtonPreconditioner(self.adp_prior, beta_curr, weight=w_adp_eff, extra_diag=h_xray_beta, damping=damping)
            raw_shift = - prec.solve(g_tot)
            if step_scale is not None:
                raw_shift *= step_scale

            max_raw = float(np.max(np.abs(raw_shift))) if len(raw_shift) > 0 else 0.0
            if max_raw > max_shift_log_b:
                raw_shift *= max_shift_log_b / max_raw

            t_curr = self.target_value + w_adp_eff * float(self.adp_prior.energy_vec(beta_curr, sites=X).item())
            alpha = 1.0
            best_t = t_curr
            best_beta = beta_curr
            best_grads = self.gradients

            for trial in range(6):
                beta_trial = beta_curr + alpha * raw_shift
                b_trial = np.clip(np.exp(beta_trial), min_b, max_b)
                b_arr = np.ascontiguousarray(b_trial, dtype=np.float64)
                self.xray_structure.set_b_iso(values=flex.double(b_arr))
                self.compute_f_calc()
                fmod_c = self._assemble_f_model(self.k_total, self.k_sol, self.b_sol)
                c_arr = np.ascontiguousarray(fmod_c, dtype=np.complex128)
                self.f_model = self.i_obs.customized_copy(data=flex.complex_double(c_arr))
                t_trial, g_trial = self.compute_target_and_gradients()
                tot_trial = t_trial + w_adp_eff * float(self.adp_prior.energy_vec(beta_trial, sites=X).item())
                if tot_trial < t_curr:
                    best_t = tot_trial
                    best_beta = beta_trial
                    best_grads = g_trial
                    break
                alpha *= 0.5

            if best_t < t_curr:
                b_final = np.clip(np.exp(best_beta), min_b, max_b)
                b_arr = np.ascontiguousarray(b_final, dtype=np.float64)
                self.xray_structure.set_b_iso(values=flex.double(b_arr))
                self.gradients = best_grads
            else:
                b_arr = np.ascontiguousarray(b_curr, dtype=np.float64)
                self.xray_structure.set_b_iso(values=flex.double(b_arr))
                self.compute_f_calc()
                fmod_c = self._assemble_f_model(self.k_total, self.k_sol, self.b_sol)
                c_arr = np.ascontiguousarray(fmod_c, dtype=np.complex128)
                self.f_model = self.i_obs.customized_copy(data=flex.complex_double(c_arr))
                self.compute_target_and_gradients()

            if self.hierarchy is not None:
                self.hierarchy.adopt_xray_structure(self.xray_structure)

            return self.target_value

        # Legacy unconstrained mode
        g_u = np.asarray(self.gradients.raw.d_u_iso, dtype=np.float64)
        g_b = g_u / (8.0 * np.pi**2)

        if p_b is not None:
            raw_shift = - (p_b * g_b)
            if step_scale is not None:
                raw_shift *= step_scale
        else:
            scale = 50000.0 if step_scale is None else step_scale
            raw_shift = -scale * g_b

        max_raw = float(np.max(np.abs(raw_shift))) if len(raw_shift) > 0 else 0.0
        if max_raw > max_shift_b:
            raw_shift *= max_shift_b / max_raw

        b_curr = np.array([float(sc.u_iso * 8.0 * np.pi**2) for sc in self.xray_structure.scatterers()])
        t_curr = self.target_value
        alpha = 1.0
        best_t = t_curr
        best_b = b_curr
        best_grads = self.gradients

        for trial in range(6):
            b_trial = np.clip(b_curr + alpha * raw_shift, min_b, max_b)
            b_arr = np.ascontiguousarray(b_trial, dtype=np.float64)
            self.xray_structure.set_b_iso(values=flex.double(b_arr))
            self.compute_f_calc()
            fmod_c = self._assemble_f_model(self.k_total, self.k_sol, self.b_sol)
            c_arr = np.ascontiguousarray(fmod_c, dtype=np.complex128)
            self.f_model = self.i_obs.customized_copy(data=flex.complex_double(c_arr))
            t_trial, g_trial = self.compute_target_and_gradients()
            if t_trial < t_curr:
                best_t = t_trial
                best_b = b_trial
                best_grads = g_trial
                break
            alpha *= 0.5

        if best_t < t_curr:
            b_arr = np.ascontiguousarray(best_b, dtype=np.float64)
            self.xray_structure.set_b_iso(values=flex.double(b_arr))
            self.gradients = best_grads
        else:
            b_arr = np.ascontiguousarray(b_curr, dtype=np.float64)
            self.xray_structure.set_b_iso(values=flex.double(b_arr))
            self.compute_f_calc()
            fmod_c = self._assemble_f_model(self.k_total, self.k_sol, self.b_sol)
            c_arr = np.ascontiguousarray(fmod_c, dtype=np.complex128)
            self.f_model = self.i_obs.customized_copy(data=flex.complex_double(c_arr))
            self.compute_target_and_gradients()

        if self.hierarchy is not None:
            self.hierarchy.adopt_xray_structure(self.xray_structure)

        return self.target_value

    def refine_b_iso(
        self: Any,
        max_iterations: int = 5,
        step_scale: Optional[float] = None,
        max_shift_b: float = 5.0,
        max_shift_log_b: float = 0.25,
        min_b: float = 1.0,
        max_b: float = 200.0,
        use_preconditioner: bool = True,
        use_adp_prior: bool = True,
        w_adp: float = 1.0,
        p_b: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> List[float]:
        """Perform iterative isotropic B-factor refinement steps (atomic sites fixed)."""
        if any(sc.flags.use_u_aniso() for sc in self.xray_structure.scatterers()):
            self.convert_to_isotropic()
        for sc in self.xray_structure.scatterers():
            sc.flags.set_grad_site(False)
            sc.flags.set_grad_u_iso(sc.occupancy > 0.0)

        if use_preconditioner and p_b is None:
            try:
                if (
                    self.target_eval is None
                    or getattr(self.target_eval, "curv_radial", None) is None
                    or getattr(self.target_eval, "curv_tangential", None) is None
                ):
                    self.compute_target_and_gradients(compute_curvature=True)
                _, p_b = self.compute_hessian_preconditioners(self.target_eval)
            except Exception:
                p_b = None

        if use_adp_prior and self.adp_prior is None:
            self.build_adp_restraints()

        history = [self.target_value]
        if verbose:
            b_vals = np.array([float(sc.u_iso * 8.0 * np.pi**2) for sc in self.xray_structure.scatterers() if sc.flags.use_u_iso()])
            adp_e_str = ""
            if use_adp_prior and self.adp_prior is not None:
                X = np.asarray(self.xray_structure.sites_cart(), dtype=np.float64)
                beta_c = np.log(np.maximum(b_vals, 1e-4))
                e_adp_val = float(self.adp_prior.energy_vec(beta_c, sites=X).item())
                adp_e_str = f", ADP E: {e_adp_val:.2f}"
            print(f"  Initial target NLL: {self.target_value:.6f}, B_iso mean: {b_vals.mean():.2f} Å²{adp_e_str} (min: {b_vals.min():.2f}, max: {b_vals.max():.2f})")

        for it in range(max_iterations):
            val = self.update_b_iso_step(
                step_scale=step_scale,
                max_shift_b=max_shift_b,
                max_shift_log_b=max_shift_log_b,
                min_b=min_b,
                max_b=max_b,
                p_b=p_b,
                use_adp_prior=use_adp_prior,
                w_adp=w_adp,
            )
            history.append(val)
            if verbose:
                b_vals = np.array([float(sc.u_iso * 8.0 * np.pi**2) for sc in self.xray_structure.scatterers() if sc.flags.use_u_iso()])
                delta = history[-1] - history[-2]
                adp_e_str = ""
                if use_adp_prior and self.adp_prior is not None:
                    X = np.asarray(self.xray_structure.sites_cart(), dtype=np.float64)
                    beta_c = np.log(np.maximum(b_vals, 1e-4))
                    e_adp_val = float(self.adp_prior.energy_vec(beta_c, sites=X).item())
                    adp_e_str = f", ADP E: {e_adp_val:.2f}"
                print(f"  Iter {it+1:2d}: target NLL: {val:.6f} (Δ: {delta:+.6f}), B_iso mean: {b_vals.mean():.2f} Å²{adp_e_str}")

        return history

    def build_geometry_restraints(self: Any) -> Any:
        """Build or retrieve cctbx geometry restraints manager for the current model."""
        if self.restraints_manager is not None:
            return self.restraints_manager
        try:
            import mmtbx.model
            from libtbx.utils import null_out
            import iotbx.pdb

            if self.hierarchy is not None:
                pdb_str = self.hierarchy.as_pdb_string(crystal_symmetry=self.xray_structure.crystal_symmetry())
            else:
                pdb_str = self.xray_structure.as_pdb_file()
            inp = iotbx.pdb.input(source_info=None, lines=pdb_str.splitlines())
            m = mmtbx.model.manager(model_input=inp, log=null_out())
            m.process(make_restraints=True)
            rm = m.get_restraints_manager()
            self.restraints_manager = rm.geometry if rm is not None else None
        except Exception as e:
            import warnings
            warnings.warn(f"Could not build cctbx geometry restraints: {e}")
            self.restraints_manager = None
        return self.restraints_manager

    def build_adp_restraints(
        self: Any,
        options: Optional[Any] = None,
        wilson_b: Optional[float] = None,
    ) -> Any:
        """Build hierarchical, scale-invariant ADP prior / B-factor restraints."""
        from phridge.client.convert_geometry import restraints_from_cctbx
        from phridge.client.adp_restraints import ADPPriorOptions, build_adp_restraints
        from phridge.worker.geometry.adp import ADPPrior, ADPPriorTables

        if self.restraints_manager is None:
            self.build_geometry_restraints()

        n = self.xray_structure.scatterers().size()
        if self.packed_restraints is None:
            self.packed_restraints = restraints_from_cctbx(self.restraints_manager, n_sites=n)

        X = np.asarray(self.xray_structure.sites_cart(), dtype=np.float64)
        if wilson_b is None:
            b_vals = [float(sc.u_iso * 8.0 * np.pi**2) for sc in self.xray_structure.scatterers()]
            wilson_b = float(np.mean(b_vals)) if b_vals else 25.0

        opts = options or ADPPriorOptions(wilson_b=wilson_b)
        self.adp_options = opts
        pr_adp = build_adp_restraints(X, self.packed_restraints, options=opts)
        self.adp_tables = ADPPriorTables.from_packed(pr_adp, sites=X)
        self.adp_prior = ADPPrior(self.adp_tables)
        return self.adp_prior

    def optimize_adp_hyperparameters(
        self: Any,
        maxiter: int = 25,
        verbose: bool = True,
    ) -> dict[str, float]:
        """Fit empirical Bayes hyperparameters for the ADP prior on working reflections."""
        from phridge.client.adp_restraints import fit_adp_hyperparameters

        if self.adp_prior is None:
            self.build_adp_restraints()

        X = np.asarray(self.xray_structure.sites_cart(), dtype=np.float64)
        b_vals = np.array([float(sc.u_iso * 8.0 * np.pi**2) for sc in self.xray_structure.scatterers()])
        beta = np.log(np.maximum(b_vals, 1e-4))

        if self.target_eval is None or getattr(self.target_eval, "curv_radial", None) is None:
            self.compute_target_and_gradients(compute_curvature=True)
        _, p_b = self.compute_hessian_preconditioners(self.target_eval)
        h_u = 1.0 / np.maximum(p_b, 1e-12)
        u_vals = b_vals / (8.0 * np.pi**2)
        h_beta = (u_vals**2) * h_u
        n_work = float(len(self.i_obs.data()) - (int(np.sum(self.r_free_flags.data())) if self.r_free_flags else 0))

        res = fit_adp_hyperparameters(self.adp_prior, beta, h_xray_diag=n_work * h_beta, sites=X, maxiter=maxiter)
        self.adp_prior = self.adp_prior.with_hyperparameters(
            tau_12=res["tau_12"],
            tau_sphere=res["tau_sphere"],
            level_weight=res["level_weight"],
        )
        if verbose:
            print("  [Empirical Bayes ADP Prior] Fitted hyperparameters:")
            print(f"    tau_12: {res['tau_12']:.4f} (SE: {res.get('tau_12_se', float('nan')):.4f})")
            print(f"    tau_sphere: {res['tau_sphere']:.4f} (SE: {res.get('tau_sphere_se', float('nan')):.4f})")
            print(f"    level_weight (lambda): {res['level_weight']:.4f} (SE: {res.get('level_weight_se', float('nan')):.4f})")
        return res

    def compute_geometry_energy_and_gradients(
        self: Any,
        sites_cart: Optional[np.ndarray] = None,
    ) -> Tuple[float, np.ndarray]:
        """Evaluate CCTBX geometry restraint target energy and Cartesian gradients."""
        if self.restraints_manager is None:
            self.build_geometry_restraints()
        if self.restraints_manager is None:
            n = self.xray_structure.scatterers().size()
            return 0.0, np.zeros((n, 3), dtype=np.float64)

        if sites_cart is None:
            sites_cart = np.asarray(self.xray_structure.sites_cart(), dtype=np.float64)
        else:
            sites_cart = np.asarray(sites_cart, dtype=np.float64)
        sites_cctbx = flex.vec3_double([tuple(r) for r in sites_cart])
        energies = self.restraints_manager.energies_sites(sites_cart=sites_cctbx, compute_gradients=True)
        e_geom = float(energies.target)
        g_geom = np.asarray(list(energies.gradients), dtype=np.float64).reshape(-1, 3)
        return e_geom, g_geom

    def compute_geometry_statistics(
        self: Any,
        sites_cart: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Compute standard macromolecular stereochemical validation metrics (bonds, angles, planarity, chirality)."""
        if self.restraints_manager is None:
            self.build_geometry_restraints()
        if self.restraints_manager is None:
            return {
                "geom_available": False,
                "target": 0.0,
                "bond_rms": float("nan"), "bond_max": float("nan"), "bond_count": 0,
                "angle_rms": float("nan"), "angle_max": float("nan"), "angle_count": 0,
                "planarity_rms": float("nan"), "planarity_max": float("nan"), "planarity_count": 0,
                "chirality_rms": float("nan"), "chirality_max": float("nan"), "chirality_count": 0,
                "dihedral_rms": float("nan"), "dihedral_max": float("nan"), "dihedral_count": 0,
                "is_acceptable": False,
            }

        if sites_cart is None:
            sites_cart = np.asarray(self.xray_structure.sites_cart(), dtype=np.float64)
        else:
            sites_cart = np.asarray(sites_cart, dtype=np.float64)
        sites_cctbx = flex.vec3_double([tuple(r) for r in sites_cart])
        energies = self.restraints_manager.energies_sites(sites_cart=sites_cctbx, compute_gradients=False)

        b = energies.bond_deviations() if hasattr(energies, "bond_deviations") else (0, 0, 0, 0)
        a = energies.angle_deviations() if hasattr(energies, "angle_deviations") else (0, 0, 0, 0)
        p = energies.planarity_deviations() if hasattr(energies, "planarity_deviations") else (0, 0, 0)
        c = energies.chirality_deviations() if hasattr(energies, "chirality_deviations") else (0, 0, 0)
        d = energies.dihedral_deviations() if hasattr(energies, "dihedral_deviations") else (0, 0, 0)

        bond_rms = float(b[2]) if len(b) > 2 else float("nan")
        bond_max = float(b[1]) if len(b) > 1 else float("nan")
        bond_count = int(b[3]) if len(b) > 3 else 0

        angle_rms = float(a[2]) if len(a) > 2 else float("nan")
        angle_max = float(a[1]) if len(a) > 1 else float("nan")
        angle_count = int(a[3]) if len(a) > 3 else 0

        planarity_rms = float(p[2]) if len(p) > 2 else float("nan")
        planarity_max = float(p[1]) if len(p) > 1 else float("nan")
        planarity_count = int(getattr(energies, "n_planarity_proxies", 0))

        chirality_rms = float(c[2]) if len(c) > 2 else float("nan")
        chirality_max = float(c[1]) if len(c) > 1 else float("nan")
        chirality_count = int(getattr(energies, "n_chirality_proxies", 0))

        dihedral_rms = float(d[2]) if len(d) > 2 else float("nan")
        dihedral_max = float(d[1]) if len(d) > 1 else float("nan")
        dihedral_count = int(getattr(energies, "n_dihedral_proxies", 0))

        # Acceptability: bond RMS <= 0.020 Å, angle RMS <= 2.20°, planarity RMS <= 0.010 Å
        is_acceptable = bool(
            np.isfinite(bond_rms) and bond_rms <= 0.020 and
            np.isfinite(angle_rms) and angle_rms <= 2.20 and
            (not np.isfinite(planarity_rms) or planarity_rms <= 0.010)
        )

        return {
            "geom_available": True,
            "target": float(energies.target),
            "bond_rms": bond_rms,
            "bond_max": bond_max,
            "bond_count": bond_count,
            "angle_rms": angle_rms,
            "angle_max": angle_max,
            "angle_count": angle_count,
            "planarity_rms": planarity_rms,
            "planarity_max": planarity_max,
            "planarity_count": planarity_count,
            "chirality_rms": chirality_rms,
            "chirality_max": chirality_max,
            "chirality_count": chirality_count,
            "dihedral_rms": dihedral_rms,
            "dihedral_max": dihedral_max,
            "dihedral_count": dihedral_count,
            "is_acceptable": is_acceptable,
        }

    def estimate_geometry_curvature(
        self: Any,
        n_shakes: int = 3,
        spread: float = 0.03,
        sites_cart: Optional[np.ndarray] = None,
    ) -> float:
        """Estimate the mean trace / eigenvalue of the CCTBX geometry Hessian via directional finite differences."""
        if self.restraints_manager is None:
            self.build_geometry_restraints()
        if self.restraints_manager is None:
            return 1000.0

        if sites_cart is None:
            sites0 = np.asarray(self.xray_structure.sites_cart(), dtype=np.float64)
        else:
            sites0 = np.asarray(sites_cart, dtype=np.float64)

        _, g0 = self.compute_geometry_energy_and_gradients(sites0)
        curvs = []
        rng = np.random.default_rng(42)
        for _ in range(n_shakes):
            dx = rng.normal(scale=spread, size=sites0.shape)
            _, gk = self.compute_geometry_energy_and_gradients(sites0 + dx)
            curv = float(np.sum(dx * (gk - g0)) / max(np.sum(dx**2), 1e-12))
            if curv > 0 and np.isfinite(curv):
                curvs.append(curv)
        return float(np.mean(curvs)) if curvs else 1000.0

    def compute_hessian_preconditioners(
        self: Any,
        ev: Any,
        damping_factor: float = 0.05,
        min_floor: float = 1e-6,
        w_xray: Optional[float] = None,
        h_geom: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute diagonal Hessian preconditioners for Cartesian sites and isotropic B-factors."""
        from phridge.sfcalc.client import _packed_xray
        from phridge.sfcalc.ops import _engine
        from phridge.models import SfEngineParams

        hkl = np.asarray(list(self.i_obs.indices()), dtype=np.int32)
        px, table = _packed_xray(self.xray_structure, None)
        params = SfEngineParams(d_min=self.d_min)
        eng = _engine(px, table, hkl, params)
        blocks = eng.gauss_newton_blocks(ev.curv_radial, ev.curv_tangential)

        # 1. B-factor curvature: H_B = H_u / (64 pi^4)
        H_u = np.asarray(blocks["u_iso"], dtype=np.float64)
        H_B = H_u / ((8.0 * np.pi**2)**2)
        med_B = float(np.median(H_B[H_B > 0])) if np.any(H_B > 0) else 1.0
        lam_B = max(min_floor, med_B * damping_factor)
        p_B = 1.0 / (np.maximum(H_B, 0.0) + lam_B)

        # 2. Site curvature in Cartesian frame: H_cart = F^T H_frac F
        uc = self.xray_structure.unit_cell()
        fract = np.asarray(uc.fractionalization_matrix(), dtype=np.float64).reshape(3, 3)
        H_frac = np.asarray(blocks["site_frac"], dtype=np.float64)  # (N, 3, 3)
        H_cart = np.einsum("ia,nab,bj->nij", fract.T, H_frac, fract)
        diag_H_cart = np.diagonal(H_cart, axis1=1, axis2=2)  # (N, 3)
        self.h_xray_mean = float(np.mean(diag_H_cart[diag_H_cart > 0])) if np.any(diag_H_cart > 0) else 1.0

        if w_xray is not None and h_geom is not None and h_geom > 0:
            H_total = w_xray * np.maximum(diag_H_cart, 0.0) + h_geom
            med_cart = float(np.median(H_total[H_total > 0])) if np.any(H_total > 0) else 1.0
            lam_cart = max(min_floor, med_cart * damping_factor)
            p_sites = 1.0 / (H_total + lam_cart)
        else:
            med_cart = float(np.median(diag_H_cart[diag_H_cart > 0])) if np.any(diag_H_cart > 0) else 1.0
            lam_cart = max(min_floor, med_cart * damping_factor)
            p_sites = 1.0 / (np.maximum(diag_H_cart, 0.0) + lam_cart)

        return p_sites, p_B

    def refine_adam(
        self: Any,
        max_iterations: int = 10,
        lr_sites: float = 0.005,
        lr_b: float = 0.2,
        lr_scale: float = 0.001,
        lr_solvent: float = 0.005,
        w_geom: Optional[float] = None,
        geom_scale: float = 0.5,
        xray_weight_mode: str = "hessian",
        xray_scale: float = 1.0,
        refine_scales: bool = True,
        refine_b: bool = True,
        refine_sites: bool = True,
        refine_sigma_a: bool = True,
        refine_nu: bool = True,
        use_preconditioner: bool = True,
        preconditioner_interval: int = 5,
        sigma_a_interval: int = 2,
        min_b: float = 1.0,
        max_b: float = 200.0,
        damping_factor: float = 0.05,
        max_shift_angstrom: float = 0.1,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Refine scale, solvent parameters, isotropic B-factors, and atomic coordinates with Adam."""
        from phridge.sfcalc.targets.maximum_likelihood import MaximumLikelihoodAmplitude
        from phridge.contrib.intensity_ll.target import IntensityLogLikelihood

        # Ensure isotropic B-factors
        if any(sc.flags.use_u_aniso() for sc in self.xray_structure.scatterers()):
            self.convert_to_isotropic()

        # Ensure bulk solvent mask and initial scales are present
        if self.use_bulk_solvent and self.f_mask is None:
            self.compute_mask()
        if self.f_calc is None or self.f_model is None:
            self.refine_scale_and_solvent()

        # Build geometry restraints if not already present
        if self.restraints_manager is None:
            self.build_geometry_restraints()

        # Initial sigma_A and nu estimation if needed
        if refine_sigma_a and refine_nu and self.target_type == "intensity":
            self.refine_sigma_a_and_nu(nu_mode=self.nu_mode)
        elif self.sigma_a_per_refl is None:
            self.estimate_sigma_a()
        if refine_nu and self.target_type == "intensity" and (self.nu is None or self.estimate_nu_flag):
            self.estimate_nu(mode=self.nu_mode)

        uc = self.xray_structure.unit_cell()
        fract = np.asarray(uc.fractionalization_matrix(), dtype=np.float64).reshape(3, 3)

        # Initialize Torch tensors
        init_sites = np.asarray(self.xray_structure.sites_cart(), dtype=np.float64)
        init_b = np.asarray([float(sc.u_iso * 8.0 * np.pi**2) for sc in self.xray_structure.scatterers()], dtype=np.float64)

        sites_cart_t = torch.tensor(init_sites, dtype=self.float_dtype, device=self.torch_device, requires_grad=refine_sites)
        b_iso_t = torch.tensor(init_b, dtype=self.float_dtype, device=self.torch_device, requires_grad=refine_b)
        k_tot_t = torch.tensor(float(self.k_total), dtype=self.float_dtype, device=self.torch_device, requires_grad=refine_scales)
        k_sol_t = torch.tensor(float(self.k_sol), dtype=self.float_dtype, device=self.torch_device, requires_grad=refine_scales and self.use_bulk_solvent)
        b_sol_t = torch.tensor(float(self.b_sol), dtype=self.float_dtype, device=self.torch_device, requires_grad=refine_scales and self.use_bulk_solvent)

        param_groups = []
        if refine_sites:
            param_groups.append({"params": [sites_cart_t], "lr": lr_sites})
        if refine_b:
            param_groups.append({"params": [b_iso_t], "lr": lr_b})
        if refine_scales:
            param_groups.append({"params": [k_tot_t], "lr": lr_scale})
            if self.use_bulk_solvent:
                param_groups.append({"params": [k_sol_t], "lr": lr_solvent})
                param_groups.append({"params": [b_sol_t], "lr": lr_solvent * 20.0})

        opt = torch.optim.Adam(param_groups)

        s2_t = torch.as_tensor(np.asarray(self.i_obs.d_star_sq().data(), dtype=np.float64), dtype=self.float_dtype, device=self.torch_device)
        fm_t = torch.as_tensor(np.asarray(self.f_mask.data(), dtype=np.complex128), dtype=self.complex_dtype, device=self.torch_device) if self.use_bulk_solvent and self.f_mask is not None else None

        io_np = np.asarray(self.i_obs.data(), dtype=np.float64)
        si_np = np.asarray(self.i_obs.sigmas(), dtype=np.float64)
        eps_np = np.asarray(self.i_obs.epsilons().data().as_double(), dtype=np.float64)
        cen_np = np.asarray(self.i_obs.centric_flags().data(), dtype=bool)
        r_free_np = np.asarray(self.r_free_flags.data(), dtype=bool)
        sw_np = np.asarray(self.mean_i_per_refl, dtype=np.float64)

        tgt_desc = "ml_f Amplitude" if self.target_type == "amplitude" else "ml_i Intensity"
        precond_desc = "Preconditioned" if use_preconditioner else "Un-preconditioned"

        history: Dict[str, list] = {
            "nll": [],
            "geom_energy": [],
            "r_work": [],
            "r_free": [],
            "b_mean": [],
            "k_total": [],
            "k_sol": [],
            "b_sol": [],
            "w_xray": [],
        }

        if verbose:
            print("\n" + "=" * 65)
            print(f" phridge {precond_desc} Adam Refinement ({tgt_desc} + CCTBX Restraints)")
            print("=" * 65)
            if self.target_value is None or np.isnan(self.target_value):
                self.compute_target_and_gradients()
            geom0, _ = self.compute_geometry_energy_and_gradients(init_sites)
            summ0 = self.summary()
            gst0 = self.compute_geometry_statistics(init_sites)
            print(f"  Initial: NLL = {self.target_value:.6f}, Geom E = {geom0:.2f}")
            print(f"           R_work = {summ0['r_work']*100:.2f}%, R_free = {summ0['r_free']*100:.2f}%, Mean B = {np.mean(init_b):.2f} Å²")
            print(f"           k_total = {self.k_total:.4e}" + (f", k_sol = {self.k_sol:.3f}, B_sol = {self.b_sol:.1f} Å²" if self.use_bulk_solvent else ""))
            if gst0["geom_available"]:
                acc_tag = "[ACCEPTABLE]" if gst0["is_acceptable"] else "[OUT-OF-TARGET]"
                print(f"           Stereochemistry {acc_tag}: Bonds = {gst0['bond_rms']:.4f} Å, Angles = {gst0['angle_rms']:.2f}°, Planar = {gst0['planarity_rms']:.4f} Å, Chiral = {gst0['chirality_rms']:.4f} Å³")
            print("-" * 65)

        p_sites = None
        p_B = None
        h_geom = None
        w_xray = 1.0
        if self.restraints_manager is not None and xray_weight_mode == "hessian":
            h_geom = self.estimate_geometry_curvature(n_shakes=3, spread=0.03, sites_cart=init_sites)

        for it in range(max_iterations):
            # Step 1: Periodic sigma_A and nu refinement
            if (it > 0 and it % sigma_a_interval == 0):
                if refine_sigma_a and refine_nu and self.target_type == "intensity":
                    self.refine_sigma_a_and_nu(nu_mode=self.nu_mode)
                elif refine_sigma_a:
                    self.estimate_sigma_a()
                elif refine_nu and self.target_type == "intensity" and (self.nu is not None or self.estimate_nu_flag):
                    self.estimate_nu(mode=self.nu_mode)

            sa_np = np.asarray(self.sigma_a_per_refl, dtype=np.float64)

            if self.target_type == "amplitude":
                f_obs = self.get_f_obs()
                fo_np = np.asarray(f_obs.data(), dtype=np.float64)
                si_f_np = np.asarray(f_obs.sigmas(), dtype=np.float64)
                beta_np = np.maximum(sw_np * (1.0 - np.clip(sa_np, 0.0, 0.9999)**2), 1e-6)
                obs = Observations.from_numpy(
                    device=self.device,
                    dtype=self.float_dtype,
                    data=fo_np,
                    sigmas=si_f_np,
                    epsilon=eps_np,
                    centric=cen_np,
                    alpha=sa_np,
                    beta=beta_np,
                    r_free=r_free_np,
                )
                tgt = MaximumLikelihoodAmplitude(scale_factor=1.0)
            else:
                nu_np = np.asarray(self.nu_per_refl, dtype=np.float64) if self.nu_per_refl is not None else (np.full(self.i_obs.size(), self.nu, dtype=np.float64) if self.nu is not None else None)
                obs = Observations.from_numpy(
                    device=self.device,
                    dtype=self.float_dtype,
                    data=io_np,
                    sigmas=si_np,
                    epsilon=eps_np,
                    centric=cen_np,
                    alpha=sa_np,
                    beta=sw_np,
                    r_free=r_free_np,
                    nu=nu_np,
                )
                tgt_kwargs: Dict[str, Any] = {}
                if self.nu is not None:
                    tgt_kwargs["nu"] = float(self.nu)
                tgt = IntensityLogLikelihood(**tgt_kwargs)

            # Step 2: Update structure factors from current tensors
            new_sites_frac = (sites_cart_t.detach().cpu().numpy() @ fract.T).astype(np.float64)
            self.xray_structure.set_sites_frac(flex.vec3_double([tuple(r) for r in new_sites_frac]))
            b_curr = np.clip(b_iso_t.detach().cpu().numpy(), min_b, max_b).astype(np.float64)
            b_arr = np.ascontiguousarray(b_curr, dtype=np.float64)
            self.xray_structure.set_b_iso(values=flex.double(b_arr))
            self.compute_f_calc()

            # Assemble F_model on torch side
            fc_t = torch.as_tensor(np.asarray(self.f_calc.data(), dtype=np.complex128), dtype=self.complex_dtype, device=self.torch_device)
            if self.use_bulk_solvent and fm_t is not None:
                f_sol = k_sol_t * torch.exp(-b_sol_t * s2_t / 4.0) * fm_t
                f_mod = k_tot_t * (fc_t + f_sol)
            else:
                f_mod = k_tot_t * fc_t

            # Step 3: Target evaluation and scale autograd
            need_curv = use_preconditioner and (p_sites is None or (it > 0 and it % preconditioner_interval == 0))
            ev = tgt.evaluate(f_mod, obs, compute_curvature=need_curv)
            nll_work = tgt.reduce(tgt.per_reflection(f_mod, obs), obs)
            self.target_value = float(nll_work.item())

            if refine_scales:
                scale_params = [k_tot_t] + ([k_sol_t, b_sol_t] if self.use_bulk_solvent else [])
                d_scales = torch.autograd.grad(nll_work, scale_params, retain_graph=True)

            # Step 4: Scatterer X-ray gradients & Gauss-Newton curvature blocks
            dtdf_mod = ev.d_target_d_f_calc
            dtdf_calc = float(k_tot_t.item()) * dtdf_mod
            c_arr = np.ascontiguousarray(dtdf_calc, dtype=np.complex128)
            grads = self.sf.gradients(self.xray_structure, self.i_obs, flex.complex_double(c_arr), d_min=self.d_min)
            self.gradients = grads

            g_xray_cart = np.asarray(grads.d_target_d_site_cart(), dtype=np.float64)
            g_u = np.asarray(grads.raw.d_u_iso, dtype=np.float64)
            g_B = g_u / (8.0 * np.pi**2)

            # Step 5: CCTBX geometry restraints evaluation (intact geometry scale w_geom = 1.0)
            e_geom, g_geom = self.compute_geometry_energy_and_gradients(sites_cart_t.detach().cpu().numpy())
            if self.restraints_manager is not None:
                if xray_weight_mode == "hessian":
                    if h_geom is None or (it > 0 and it % preconditioner_interval == 0):
                        h_geom = self.estimate_geometry_curvature(
                            n_shakes=3, spread=0.03, sites_cart=sites_cart_t.detach().cpu().numpy()
                        )
                    if need_curv:
                        self.compute_hessian_preconditioners(ev, damping_factor=damping_factor)
                    h_xray = getattr(self, "h_xray_mean", 1.0)
                    w_xray = xray_scale * (h_geom / max(h_xray, 1e-12))
                    if need_curv:
                        p_sites, p_B = self.compute_hessian_preconditioners(
                            ev, damping_factor=damping_factor, w_xray=w_xray, h_geom=h_geom
                        )
                elif xray_weight_mode == "gradient":
                    rms_xray = float(np.sqrt(np.mean(g_xray_cart**2)))
                    rms_geom = float(np.sqrt(np.mean(g_geom**2)))
                    w_xray = xray_scale * (rms_geom / max(rms_xray, 1e-12))
                    if need_curv:
                        p_sites, p_B = self.compute_hessian_preconditioners(
                            ev, damping_factor=damping_factor, w_xray=w_xray, h_geom=1.0
                        )
                else:
                    w_xray = xray_scale
                    if need_curv:
                        p_sites, p_B = self.compute_hessian_preconditioners(
                            ev, damping_factor=damping_factor, w_xray=w_xray, h_geom=1.0
                        )

                g_joint_cart = w_xray * g_xray_cart + g_geom
            else:
                w_xray = 1.0
                g_joint_cart = g_xray_cart
                if need_curv:
                    p_sites, p_B = self.compute_hessian_preconditioners(ev, damping_factor=damping_factor)

            # Step 6: Apply Preconditioners
            if use_preconditioner and p_sites is not None:
                eff_grad_sites = p_sites * g_joint_cart
            else:
                eff_grad_sites = g_joint_cart

            if use_preconditioner and p_B is not None:
                eff_grad_b = p_B * g_B
            else:
                eff_grad_b = g_B

            # Step 7: Adam optimizer step
            opt.zero_grad()
            if refine_sites:
                sites_cart_t.grad = torch.as_tensor(eff_grad_sites, dtype=self.float_dtype, device=self.torch_device)
            if refine_b:
                b_iso_t.grad = torch.as_tensor(eff_grad_b, dtype=self.float_dtype, device=self.torch_device)
            if refine_scales:
                k_tot_t.grad = d_scales[0]
                if self.use_bulk_solvent:
                    k_sol_t.grad = d_scales[1]
                    b_sol_t.grad = d_scales[2]

            opt.step()

            # Step 8: Clamping and boundary checks
            with torch.no_grad():
                b_iso_t.clamp_(min=min_b, max=max_b)
                k_tot_t.clamp_(min=1e-6)
                if self.use_bulk_solvent:
                    k_sol_t.clamp_(min=0.0, max=1.0)
                    b_sol_t.clamp_(min=10.0, max=300.0)

                disp = sites_cart_t.detach().cpu().numpy() - init_sites
                disp_norm = np.linalg.norm(disp, axis=1)
                for a_i, d_val in enumerate(disp_norm):
                    if d_val > max_shift_angstrom:
                        sites_cart_t[a_i] = torch.as_tensor(
                            init_sites[a_i] + disp[a_i] * (max_shift_angstrom / d_val),
                            dtype=self.float_dtype,
                            device=self.torch_device,
                        )

            # Step 9: Sync model parameters
            self.k_total = float(k_tot_t.item())
            if self.use_bulk_solvent:
                self.k_sol = float(k_sol_t.item())
                self.b_sol = float(b_sol_t.item())

            # Update F_model
            fmod_np = f_mod.detach().cpu().numpy()
            c_arr = np.ascontiguousarray(fmod_np, dtype=np.complex128)
            self.f_model = self.i_obs.customized_copy(data=flex.complex_double(c_arr))

            # Logging & Diagnostics
            summ = self.summary()
            history["nll"].append(self.target_value)
            history["geom_energy"].append(e_geom)
            history["r_work"].append(summ["r_work"])
            history["r_free"].append(summ["r_free"])
            history["b_mean"].append(float(np.mean(b_iso_t.detach().cpu().numpy())))
            history["k_total"].append(self.k_total)
            history["k_sol"].append(self.k_sol if self.use_bulk_solvent else 0.0)
            history["b_sol"].append(self.b_sol if self.use_bulk_solvent else 0.0)
            history["w_xray"].append(w_xray)

            if verbose and ((it + 1) % max(1, max_iterations // 5) == 0 or it == max_iterations - 1):
                cc_free_str = f", CC_free = {summ['cc_free_i']:.4f}" if "cc_free_i" in summ else ""
                print(
                    f"  Iter {it+1:3d}/{max_iterations}: NLL = {self.target_value:.6f}, Geom E = {e_geom:7.1f}, "
                    f"R_work = {summ['r_work']*100:5.2f}%, R_free = {summ['r_free']*100:5.2f}%{cc_free_str}, "
                    f"w_xray = {w_xray:7.3f}, k_tot = {self.k_total:.3e}"
                )

        # Final sync
        final_sites_frac = (sites_cart_t.detach().cpu().numpy() @ fract.T).astype(np.float64)
        self.xray_structure.set_sites_frac(flex.vec3_double([tuple(r) for r in final_sites_frac]))
        final_b = np.clip(b_iso_t.detach().cpu().numpy(), min_b, max_b).astype(np.float64)
        b_arr = np.ascontiguousarray(final_b, dtype=np.float64)
        self.xray_structure.set_b_iso(values=flex.double(b_arr))
        if self.hierarchy is not None:
            self.hierarchy.adopt_xray_structure(self.xray_structure)
        self.compute_f_calc()
        fmod_complex = self._assemble_f_model(self.k_total, self.k_sol, self.b_sol)
        c_arr = np.ascontiguousarray(fmod_complex, dtype=np.complex128)
        self.f_model = self.i_obs.customized_copy(data=flex.complex_double(c_arr))
        self.compute_target_and_gradients()

        summ_final = self.summary()
        gst_final = self.compute_geometry_statistics()
        if verbose:
            print("-" * 65)
            print(f"  Final:   NLL = {self.target_value:.6f}, Geom E = {history['geom_energy'][-1]:.2f}")
            print(f"           R_work = {summ_final['r_work']*100:.2f}%, R_free = {summ_final['r_free']*100:.2f}%, CC_free = {summ_final.get('cc_free_i', float('nan')):.4f}")
            if gst_final["geom_available"]:
                acc_tag = "[ACCEPTABLE]" if gst_final["is_acceptable"] else "[OUT-OF-TARGET]"
                print(f"           Stereo {acc_tag}: Bonds = {gst_final['bond_rms']:.4f} Å, Angles = {gst_final['angle_rms']:.2f}°, Planar = {gst_final['planarity_rms']:.4f} Å, Chiral = {gst_final['chirality_rms']:.4f} Å³")
            print("=" * 65 + "\n")

        return history

    def refine_lbfgs(
        self: Any,
        macrocycles: int = 3,
        max_iterations_per_cycle: int = 20,
        regularize_geometry: bool = True,
        regularize_steps: int = 15,
        w_geom: Optional[float] = None,
        xray_weight_mode: str = "hessian",
        xray_scale: float = 1.0,
        refine_scales: bool = True,
        refine_b: bool = True,
        use_adp_restraints: bool = True,
        optimize_adp_weights: bool = True,
        w_adp: float = 1.0,
        refine_sites: bool = True,
        refine_sigma_a: bool = True,
        refine_nu: bool = True,
        b_iterations: int = 5,
        min_b: float = 1.0,
        max_b: float = 200.0,
        m_history: int = 10,
        polish_geometry: bool = True,
        polish_steps: int = 10,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Refine coordinates, isotropic B-factors, scales, and solvent using L-BFGS."""
        # Ensure isotropic B-factors
        if any(sc.flags.use_u_aniso() for sc in self.xray_structure.scatterers()):
            self.convert_to_isotropic()

        # Ensure bulk solvent mask and initial scales are present
        if self.use_bulk_solvent and self.f_mask is None:
            self.compute_mask()
        if self.f_calc is None or self.f_model is None:
            self.refine_scale_and_solvent()

        # Build geometry restraints if not already present
        if self.restraints_manager is None:
            self.build_geometry_restraints()

        # Initial sigma_A and nu estimation if needed
        if refine_sigma_a and refine_nu and self.target_type == "intensity":
            self.refine_sigma_a_and_nu(nu_mode=self.nu_mode)
        elif self.sigma_a_per_refl is None:
            self.estimate_sigma_a()
        if refine_nu and self.target_type == "intensity" and (self.nu is None or self.estimate_nu_flag):
            self.estimate_nu(mode=self.nu_mode)

        uc = self.xray_structure.unit_cell()
        fract = np.asarray(uc.fractionalization_matrix(), dtype=np.float64).reshape(3, 3)

        # Initial evaluation and geometry audit
        sites_init = np.asarray(self.xray_structure.sites_cart(), dtype=np.float64)
        gst_init = self.compute_geometry_statistics(sites_init)
        val0, _ = self.compute_target_and_gradients(compute_curvature=True)
        summ0 = self.summary()

        tgt_desc = "ml_f Amplitude" if self.target_type == "amplitude" else "ml_i Intensity"

        if verbose:
            print("=" * 75)
            print(f"  L-BFGS Refinement Pipeline ({tgt_desc})")
            print(f"  Macrocycles: {macrocycles} | Max iter/cycle: {max_iterations_per_cycle} | Weight mode: {xray_weight_mode}")
            print(f"  Initial: NLL = {val0:.6f}, R_work = {summ0['r_work']*100:.2f}%, R_free = {summ0['r_free']*100:.2f}%, CC_free = {summ0.get('cc_free_i', float('nan')):.4f}")
            if gst_init.get("geom_available"):
                acc_tag = "[ACCEPTABLE]" if gst_init["is_acceptable"] else "[OUT-OF-TARGET]"
                print(f"           Stereo {acc_tag}: Bonds = {gst_init['bond_rms']:.4f} Å, Angles = {gst_init['angle_rms']:.2f}°, Planar = {gst_init['planarity_rms']:.4f} Å, E_geom = {gst_init['target']:.0f}")
            print("=" * 75)

        history: Dict[str, list] = {
            "nll": [],
            "geom_energy": [],
            "adp_energy": [],
            "r_work": [],
            "r_free": [],
            "b_mean": [],
            "bond_rms": [],
            "angle_rms": [],
            "planarity_rms": [],
            "chirality_rms": [],
            "is_acceptable": [],
            "w_xray": [],
        }

        has_restraints = self.restraints_manager is not None

        # Phase 0: Pure Geometry Regularization (if requested and restraints are present)
        if regularize_geometry and has_restraints and (gst_init.get("bond_rms", 0.0) > 0.020 or gst_init.get("planarity_rms", 0.0) > 0.015):
            if verbose:
                print(f"\n  [Phase 0] Running {regularize_steps} steps of pure geometry regularization...")
            def fg_geom(x):
                s = x.reshape(-1, 3)
                e, g = self.compute_geometry_energy_and_gradients(s)
                return float(e), g.ravel()
            s0_flat = sites_init.ravel()
            res_reg = scipy.optimize.minimize(fg_geom, s0_flat, jac=True, method="L-BFGS-B", options={"maxiter": regularize_steps, "maxcor": m_history})
            s_reg = res_reg.x.reshape(-1, 3)
            new_frac = (s_reg @ fract.T).astype(np.float64)
            self.xray_structure.set_sites_frac(flex.vec3_double([tuple(r) for r in new_frac]))
            self.compute_f_calc()
            fmod_c = self._assemble_f_model(self.k_total, self.k_sol, self.b_sol)
            c_arr = np.ascontiguousarray(fmod_c, dtype=np.complex128)
            self.f_model = self.i_obs.customized_copy(data=flex.complex_double(c_arr))
            self.compute_target_and_gradients(compute_curvature=True)
            gst_reg = self.compute_geometry_statistics()
            if verbose:
                print(f"           Post-reg Bonds = {gst_reg.get('bond_rms', 0.0):.4f} Å, Angles = {gst_reg.get('angle_rms', 0.0):.2f}°, Planar = {gst_reg.get('planarity_rms', 0.0):.4f} Å")

        # Determine X-ray weight w_xray
        sites_now = np.asarray(self.xray_structure.sites_cart(), dtype=np.float64)
        if has_restraints:
            if w_geom is not None and w_geom > 0:
                w_xray = 1.0 / float(w_geom)
            elif xray_weight_mode == "hessian":
                h_geom = self.estimate_geometry_curvature(n_shakes=3, spread=0.03, sites_cart=sites_now)
                h_xray = getattr(self, "h_xray_mean", 1.0)
                w_xray = xray_scale * (h_geom / max(h_xray, 1e-12))
            elif xray_weight_mode == "gradient":
                val, grads = self.compute_target_and_gradients()
                gx = np.asarray(grads.d_target_d_site_cart(), dtype=np.float64)
                _, gg = self.compute_geometry_energy_and_gradients(sites_now)
                rms_x = float(np.sqrt(np.mean(gx**2)))
                rms_g = float(np.sqrt(np.mean(gg**2)))
                w_xray = xray_scale * (rms_g / max(rms_x, 1e-12))
            else:
                w_xray = float(xray_scale)
        else:
            w_xray = 1.0

        if verbose:
            print(f"  Coupled Refinement Weight: w_xray = {w_xray:.4f} (Geometry scale = 1.0)\n")

        # Setup ADP restraints
        if refine_b and use_adp_restraints:
            if self.adp_prior is None:
                self.build_adp_restraints()
            if optimize_adp_weights and self.adp_prior is not None:
                self.optimize_adp_hyperparameters(maxiter=25, verbose=verbose)

        # Macrocycle Execution
        for cycle in range(1, macrocycles + 1):
            history["w_xray"].append(w_xray)

            # 1. Coordinate refinement with L-BFGS
            if refine_sites:
                s_in = np.asarray(self.xray_structure.sites_cart(), dtype=np.float64).ravel()
                def fg_joint(x):
                    s = x.reshape(-1, 3)
                    new_frac = (s @ fract.T).astype(np.float64)
                    self.xray_structure.set_sites_frac(flex.vec3_double([tuple(r) for r in new_frac]))
                    self.compute_f_calc()
                    fmod_c = self._assemble_f_model(self.k_total, self.k_sol, self.b_sol)
                    c_arr = np.ascontiguousarray(fmod_c, dtype=np.complex128)
                    self.f_model = self.i_obs.customized_copy(data=flex.complex_double(c_arr))
                    val_xray, grads_xray = self.compute_target_and_gradients()
                    g_xray = np.asarray(grads_xray.d_target_d_site_cart(), dtype=np.float64)
                    e_geom, g_geom = self.compute_geometry_energy_and_gradients(s)
                    f_tot = float(w_xray * val_xray + e_geom)
                    g_tot = (w_xray * g_xray + g_geom).ravel()
                    return f_tot, g_tot

                res = scipy.optimize.minimize(
                    fg_joint,
                    s_in,
                    jac=True,
                    method="L-BFGS-B",
                    options={"maxiter": max_iterations_per_cycle, "maxcor": m_history},
                )
                s_opt = res.x.reshape(-1, 3)
                new_frac = (s_opt @ fract.T).astype(np.float64)
                self.xray_structure.set_sites_frac(flex.vec3_double([tuple(r) for r in new_frac]))
                self.compute_f_calc()
                fmod_c = self._assemble_f_model(self.k_total, self.k_sol, self.b_sol)
                c_arr = np.ascontiguousarray(fmod_c, dtype=np.complex128)
                self.f_model = self.i_obs.customized_copy(data=flex.complex_double(c_arr))

            # 2. Isotropic B-factor refinement (preconditioned Newton steps with ADP prior)
            if refine_b:
                self.refine_b_iso(
                    max_iterations=b_iterations,
                    min_b=min_b,
                    max_b=max_b,
                    use_preconditioner=True,
                    use_adp_prior=use_adp_restraints,
                    w_adp=w_adp,
                    verbose=False,
                )

            # 3. Overall scale and bulk solvent refinement
            if refine_scales:
                self.refine_scale_and_solvent(recompute_mask=self.use_bulk_solvent)

            # 4. Periodic sigma_A and nu re-estimation
            if refine_sigma_a and refine_nu and self.target_type == "intensity":
                self.refine_sigma_a_and_nu(nu_mode=self.nu_mode)
            elif refine_sigma_a:
                self.estimate_sigma_a()
            elif refine_nu and self.target_type == "intensity" and (self.nu is not None or self.estimate_nu_flag):
                self.estimate_nu(mode=self.nu_mode)

            # Cycle statistics
            self.compute_target_and_gradients()
            gst_c = self.compute_geometry_statistics()
            summ_c = self.summary()
            b_vals_c = [sc.u_iso * 8.0 * np.pi**2 for sc in self.xray_structure.scatterers()]

            history["nll"].append(self.target_value)
            history["geom_energy"].append(gst_c.get("target", 0.0))
            if self.adp_prior is not None:
                X_c = np.asarray(self.xray_structure.sites_cart(), dtype=np.float64)
                beta_c = np.log(np.maximum(np.array(b_vals_c, dtype=np.float64), 1e-4))
                history["adp_energy"].append(float(self.adp_prior.energy_vec(beta_c, sites=X_c).item()))
            history["r_work"].append(summ_c["r_work"])
            history["r_free"].append(summ_c["r_free"])
            history["b_mean"].append(float(np.mean(b_vals_c)))
            history["bond_rms"].append(gst_c.get("bond_rms", float("nan")))
            history["angle_rms"].append(gst_c.get("angle_rms", float("nan")))
            history["planarity_rms"].append(gst_c.get("planarity_rms", float("nan")))
            history["chirality_rms"].append(gst_c.get("chirality_rms", float("nan")))
            history["is_acceptable"].append(gst_c.get("is_acceptable", False))

            if verbose:
                acc_tag = "[ACCEPTABLE]" if gst_c.get("is_acceptable") else "[OUT-OF-TARGET]"
                cc_free_str = f", CC_free = {summ_c['cc_free_i']:.4f}" if "cc_free_i" in summ_c else ""
                adp_str = f", ADP E = {history['adp_energy'][-1]:.1f}" if self.adp_prior is not None and history["adp_energy"] else ""
                print(
                    f"  Cycle {cycle:2d}/{macrocycles}: NLL = {self.target_value:.6f}, "
                    f"R_work = {summ_c['r_work']*100:.2f}%, R_free = {summ_c['r_free']*100:.2f}%{cc_free_str}{adp_str} | "
                    f"Bonds = {gst_c.get('bond_rms', 0.0):.4f} Å, Angles = {gst_c.get('angle_rms', 0.0):.2f}°, "
                    f"Planar = {gst_c.get('planarity_rms', 0.0):.4f} Å {acc_tag}"
                )

        # Optional Phase: Geometry Polish
        if polish_geometry and has_restraints and (gst_c.get("bond_rms", 0.0) > 0.012 or gst_c.get("planarity_rms", 0.0) > 0.010):
            if verbose:
                print(f"  [Polish] Running {polish_steps} steps of pure geometry polish...")
            def fg_geom_polish(x):
                s = x.reshape(-1, 3)
                e, g = self.compute_geometry_energy_and_gradients(s)
                return float(e), g.ravel()
            s_pol_in = np.asarray(self.xray_structure.sites_cart(), dtype=np.float64).ravel()
            res_pol = scipy.optimize.minimize(fg_geom_polish, s_pol_in, jac=True, method="L-BFGS-B", options={"maxiter": polish_steps, "maxcor": m_history})
            s_pol = res_pol.x.reshape(-1, 3)
            new_frac = (s_pol @ fract.T).astype(np.float64)
            self.xray_structure.set_sites_frac(flex.vec3_double([tuple(r) for r in new_frac]))
            self.compute_f_calc()
            fmod_c = self._assemble_f_model(self.k_total, self.k_sol, self.b_sol)
            c_arr = np.ascontiguousarray(fmod_c, dtype=np.complex128)
            self.f_model = self.i_obs.customized_copy(data=flex.complex_double(c_arr))

        if self.hierarchy is not None:
            self.hierarchy.adopt_xray_structure(self.xray_structure)

        summ_final = self.summary()
        gst_final = self.compute_geometry_statistics()
        if verbose:
            print("-" * 75)
            print(f"  Final:   NLL = {self.target_value:.6f}, R_work = {summ_final['r_work']*100:.2f}%, R_free = {summ_final['r_free']*100:.2f}%, CC_free = {summ_final.get('cc_free_i', float('nan')):.4f}")
            if gst_final.get("geom_available"):
                acc_tag = "[ACCEPTABLE]" if gst_final["is_acceptable"] else "[OUT-OF-TARGET]"
                print(f"           Stereo {acc_tag}: Bonds = {gst_final['bond_rms']:.4f} Å, Angles = {gst_final['angle_rms']:.2f}°, Planar = {gst_final['planarity_rms']:.4f} Å, Chiral = {gst_final['chirality_rms']:.4f} Å³")
            print("=" * 75 + "\n")

        return history
