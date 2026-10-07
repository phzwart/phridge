"""IntensityModel estimation and reporting methods."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from phridge.client.intensity.math_utils import golden_section_search_torch, isotonic_non_increasing, tv_denoise_1d
from phridge.d.rules.mli_quad import log_likelihood_normal, log_likelihood_t, normalize
from phridge.d.rules.mli_quad.target import IntensityLogLikelihood
from phridge.d.rules.amplitude import Observations
from phridge.d.rules.amplitude.maximum_likelihood import MaximumLikelihoodAmplitude
from phridge.shims.cctbx._flex import complex_double as flex_complex_double
from phridge.shims.cctbx._flex import double as flex_double


class IntensityStepsMixin:
    def estimate_sigma_a(
        self,
        overlap: Optional[int] = None,
        tv_lambda: Optional[float] = None,
        enforce_monotonic: Optional[bool] = None,
        interpolate: Optional[bool] = None,
    ) -> Dict[int, float]:
        """Estimate sigma_A per resolution bin by directly minimizing intensity NLL."""
        if self.f_model is None:
            self.refine_scale_and_solvent()

        overlap_val = self.overlap_bins if overlap is None else int(overlap)
        tv_lam_val = self.tv_norm if tv_lambda is None else float(tv_lambda)
        mono_val = self.enforce_monotonic if enforce_monotonic is None else bool(enforce_monotonic)
        interp_val = self.interpolate_sigma_a if interpolate is None else bool(interpolate)

        bins = list(self.binner.range_used())
        n_bins = len(bins)

        io_np = np.asarray(self.i_obs.data(), dtype=np.float64)
        si_np = np.asarray(self.i_obs.sigmas(), dtype=np.float64)
        eps_np = np.asarray(self.i_obs.epsilons().data().as_double(), dtype=np.float64)
        sw_np = np.asarray(self.mean_i_per_refl, dtype=np.float64)
        cen_np = np.asarray(self.i_obs.centric_flags().data(), dtype=bool)
        fc_abs_np = np.abs(np.asarray(self.f_model.data()))

        io_t = torch.as_tensor(io_np, dtype=self.float_dtype, device=self.torch_device)
        si_t = torch.as_tensor(si_np, dtype=self.float_dtype, device=self.torch_device)
        eps_t = torch.as_tensor(eps_np, dtype=self.float_dtype, device=self.torch_device)
        sw_t = torch.as_tensor(sw_np, dtype=self.float_dtype, device=self.torch_device)
        cen_t = torch.as_tensor(cen_np, device=self.torch_device)
        fc_abs_t = torch.as_tensor(fc_abs_np, dtype=self.float_dtype, device=self.torch_device)

        bin_sels = [np.asarray(self.binner.selection(b), dtype=bool) for b in bins]
        weights_list = [float(np.sum(s)) for s in bin_sels]

        if mono_val:
            u0 = torch.tensor(1.5, dtype=self.float_dtype, device=self.torch_device, requires_grad=True)
            raw_deltas = torch.full((n_bins - 1,), 0.1, dtype=self.float_dtype, device=self.torch_device, requires_grad=True)
            params = [u0, raw_deltas]
        else:
            u = torch.zeros(n_bins, dtype=self.float_dtype, device=self.torch_device, requires_grad=True)
            params = [u]

        opt = torch.optim.LBFGS(params, lr=1.0, max_iter=40, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            if mono_val:
                deltas = torch.nn.functional.softplus(raw_deltas)
                cum = torch.cat([torch.zeros(1, dtype=self.float_dtype, device=self.torch_device), torch.cumsum(deltas, dim=0)])
                logits = u0 - cum
                sa_bins = 0.01 + 0.989 * torch.sigmoid(logits)
            else:
                sa_bins = 0.01 + 0.989 * torch.sigmoid(u)

            sa_refl = torch.zeros(self.i_obs.size(), dtype=self.float_dtype, device=self.torch_device)
            for idx, sel_np in enumerate(bin_sels):
                sa_refl[sel_np] = sa_bins[idx]

            Ec, _, Zo, sZ = normalize(fc_abs_t, io_t, si_t, eps_t, sw_t, sa_refl)

            if overlap_val == 0:
                if self.nu is not None and self.target_type == "intensity":
                    ll = log_likelihood_t(
                        Ec,
                        sa_refl,
                        Zo,
                        sZ,
                        cen_t,
                        nu=float(self.nu),
                        n_u=10,
                    )
                else:
                    ll = log_likelihood_normal(Ec, sa_refl, Zo, sZ, cen_t)
                loss = -ll.mean()
                if tv_lam_val > 0.0 and n_bins > 1:
                    diffs = sa_bins[1:] - sa_bins[:-1]
                    loss = loss + tv_lam_val * torch.sqrt(diffs**2 + 1e-6).sum()
                loss.backward()
                return loss
            else:
                fit_mask = np.zeros(self.i_obs.size(), dtype=bool)
                for idx in range(n_bins):
                    lo = max(0, idx - overlap_val)
                    hi = min(n_bins, idx + overlap_val + 1)
                    for j in range(lo, hi):
                        fit_mask |= bin_sels[j]
                fit_sel_t = torch.as_tensor(fit_mask, device=self.torch_device)

                if self.nu is not None and self.target_type == "intensity":
                    ll = log_likelihood_t(
                        Ec[fit_sel_t],
                        sa_refl[fit_sel_t],
                        Zo[fit_sel_t],
                        sZ[fit_sel_t],
                        cen_t[fit_sel_t],
                        nu=float(self.nu),
                        n_u=10,
                    )
                else:
                    ll = log_likelihood_normal(
                        Ec[fit_sel_t],
                        sa_refl[fit_sel_t],
                        Zo[fit_sel_t],
                        sZ[fit_sel_t],
                        cen_t[fit_sel_t],
                    )
                loss = -ll.mean()
                if tv_lam_val > 0.0 and n_bins > 1:
                    diffs = sa_bins[1:] - sa_bins[:-1]
                    loss = loss + tv_lam_val * torch.sqrt(diffs**2 + 1e-6).sum()
                loss.backward()
                return loss

        opt.step(closure)

        if mono_val:
            deltas = torch.nn.functional.softplus(raw_deltas)
            cum = torch.cat([torch.zeros(1, dtype=self.float_dtype, device=self.torch_device), torch.cumsum(deltas, dim=0)])
            logits = u0 - cum
            sa_arr = (0.01 + 0.989 * torch.sigmoid(logits)).detach().cpu().numpy()
        else:
            sa_arr = (0.01 + 0.989 * torch.sigmoid(u)).detach().cpu().numpy()
        weights_arr = np.array(weights_list, dtype=np.float64)

        if tv_lam_val > 0.0:
            sa_arr = tv_denoise_1d(sa_arr, weights=weights_arr, lam=tv_lam_val, lower=0.01, upper=0.999)

        if mono_val:
            sa_arr = isotonic_non_increasing(sa_arr, weights=weights_arr)

        self.sigma_a_binned = {}
        for idx, i_bin in enumerate(bins):
            self.sigma_a_binned[i_bin] = float(sa_arr[idx])

        self._store_sigma_a_per_reflection(sa_arr, bins, interp_val, n_bins)
        return self.sigma_a_binned

    def _store_sigma_a_per_reflection(self, sa_arr, bins, interp_val: bool, n_bins: int) -> None:
        if interp_val and n_bins >= 2:
            try:
                sa_flex = flex_double(np.ascontiguousarray(sa_arr, dtype=np.float64))
                interp = self.binner.interpolate(sa_flex, 2.0)
                interp_np = np.clip(np.asarray(interp, dtype=np.float64), 0.01, 0.999)
                self.sigma_a_per_refl = flex_double(np.ascontiguousarray(interp_np, dtype=np.float64))
                return
            except Exception:
                pass
        sa_per_refl = np.full(self.i_obs.size(), 0.8, dtype=np.float64)
        for idx, i_bin in enumerate(bins):
            sel = np.asarray(self.binner.selection(i_bin), dtype=bool)
            sa_per_refl[sel] = sa_arr[idx]
        self.sigma_a_per_refl = flex_double(np.ascontiguousarray(sa_per_refl, dtype=np.float64))

    def estimate_nu(
        self,
        mode: Optional[str] = None,
        overlap: Optional[int] = None,
        tv_lambda: Optional[float] = None,
        bounds: Tuple[float, float] = (3.0, 30.0),
        prior_center: float = 7.0,
        prior_weight: float = 0.01,
        n_u: int = 10,
    ) -> Any:
        """Estimate or refine Student-t degrees of freedom 'nu' by directly minimizing intensity NLL."""
        if self.f_model is None:
            self.refine_scale_and_solvent()
        if self.sigma_a_per_refl is None:
            self.estimate_sigma_a()

        eff_mode = self.nu_mode if mode is None else str(mode)
        overlap_val = self.overlap_bins if overlap is None else int(overlap)
        tv_lam_val = self.tv_norm if tv_lambda is None else float(tv_lambda)

        io_t = torch.as_tensor(np.asarray(self.i_obs.data(), dtype=np.float64), dtype=self.float_dtype, device=self.torch_device)
        si_t = torch.as_tensor(np.asarray(self.i_obs.sigmas(), dtype=np.float64), dtype=self.float_dtype, device=self.torch_device)
        eps_t = torch.as_tensor(np.asarray(self.i_obs.epsilons().data().as_double(), dtype=np.float64), dtype=self.float_dtype, device=self.torch_device)
        sw_t = torch.as_tensor(np.asarray(self.mean_i_per_refl, dtype=np.float64), dtype=self.float_dtype, device=self.torch_device)
        cen_t = torch.as_tensor(np.asarray(self.i_obs.centric_flags().data(), dtype=bool), device=self.torch_device)
        fc_abs_t = torch.as_tensor(np.abs(np.asarray(self.f_model.data())), dtype=self.float_dtype, device=self.torch_device)
        sa_t = torch.as_tensor(np.asarray(self.sigma_a_per_refl, dtype=np.float64), dtype=self.float_dtype, device=self.torch_device)

        Ec, _, Zo, sZ = normalize(fc_abs_t, io_t, si_t, eps_t, sw_t, sa_t)

        Ec_cpu = Ec.cpu()
        sa_cpu = sa_t.cpu()
        Zo_cpu = Zo.cpu()
        sZ_cpu = sZ.cpu()
        cen_cpu = cen_t.cpu()

        work_sel = torch.as_tensor(~np.asarray(self.r_free_flags.data(), dtype=bool))
        bins = list(self.binner.range_used())
        n_bins = len(bins)

        if eff_mode == "global":
            def global_nll(nu_val: float) -> float:
                ll = log_likelihood_t(
                    Ec_cpu[work_sel],
                    sa_cpu[work_sel],
                    Zo_cpu[work_sel],
                    sZ_cpu[work_sel],
                    cen_cpu[work_sel],
                    nu=float(nu_val),
                    n_u=n_u,
                )
                prior = prior_weight * ((math.log(nu_val) - math.log(prior_center)) ** 2)
                return float(-ll.mean().item() + prior)

            nu_opt = golden_section_search_torch(global_nll, bounds[0], bounds[1], tol=0.1, max_iter=25)
            self.nu = float(nu_opt)
            self.nu_binned = {b: self.nu for b in bins}
            self.nu_per_refl = flex_double(self.i_obs.size(), self.nu)
            return self.nu

        bin_sels = [np.asarray(self.binner.selection(b), dtype=bool) for b in bins]
        weights_list = [float(np.sum(s)) for s in bin_sels]
        nu_arr = np.zeros(n_bins, dtype=np.float64)

        for idx in range(n_bins):
            lo = max(0, idx - overlap_val)
            hi = min(n_bins, idx + overlap_val + 1)
            pool_mask = np.zeros(self.i_obs.size(), dtype=bool)
            for j in range(lo, hi):
                pool_mask |= bin_sels[j]
            pool_sel = torch.as_tensor(pool_mask & ~np.asarray(self.r_free_flags.data(), dtype=bool))

            if pool_sel.sum() == 0:
                nu_arr[idx] = prior_center
                continue

            def bin_nu_nll(nu_val: float) -> float:
                ll = log_likelihood_t(
                    Ec_cpu[pool_sel],
                    sa_cpu[pool_sel],
                    Zo_cpu[pool_sel],
                    sZ_cpu[pool_sel],
                    cen_cpu[pool_sel],
                    nu=float(nu_val),
                    n_u=n_u,
                )
                prior = prior_weight * ((math.log(nu_val) - math.log(prior_center)) ** 2)
                return float(-ll.mean().item() + prior)

            nu_opt = golden_section_search_torch(bin_nu_nll, bounds[0], bounds[1], tol=0.2, max_iter=20)
            nu_arr[idx] = float(nu_opt)

        if tv_lam_val > 0.0 and n_bins > 1:
            weights_arr = np.array(weights_list, dtype=np.float64)
            nu_arr = tv_denoise_1d(nu_arr, weights=weights_arr, lam=tv_lam_val, lower=bounds[0], upper=bounds[1])

        self.nu_binned = {}
        for idx, i_bin in enumerate(bins):
            self.nu_binned[i_bin] = float(nu_arr[idx])

        self.nu = float(np.mean(nu_arr))

        nu_per_refl = np.full(self.i_obs.size(), self.nu, dtype=np.float64)
        for idx, i_bin in enumerate(bins):
            sel = np.asarray(self.binner.selection(i_bin), dtype=bool)
            nu_per_refl[sel] = nu_arr[idx]
        self.nu_per_refl = flex_double(np.ascontiguousarray(nu_per_refl, dtype=np.float64))

        return self.nu_binned

    refine_nu = estimate_nu

    def refine_sigma_a_and_nu(
        self,
        max_cycles: int = 2,
        nu_mode: Optional[str] = None,
        overlap: Optional[int] = None,
        tv_lambda: Optional[float] = None,
        enforce_monotonic: Optional[bool] = None,
        interpolate: Optional[bool] = None,
        bounds: Tuple[float, float] = (3.0, 30.0),
    ) -> Tuple[Dict[int, float], Any]:
        """Iteratively co-refine sigma_A and Student-t nu."""
        eff_mode = self.nu_mode if nu_mode is None else str(nu_mode)
        for _ in range(max_cycles):
            self.estimate_sigma_a(
                overlap=overlap,
                tv_lambda=tv_lambda,
                enforce_monotonic=enforce_monotonic,
                interpolate=interpolate,
            )
            self.estimate_nu(
                mode=eff_mode,
                overlap=overlap,
                tv_lambda=tv_lambda,
                bounds=bounds,
            )
        self.estimate_sigma_a(
            overlap=overlap,
            tv_lambda=tv_lambda,
            enforce_monotonic=enforce_monotonic,
            interpolate=interpolate,
        )
        return self.sigma_a_binned, (self.nu if eff_mode == "global" else self.nu_binned)

    def compute_target_and_gradients(self, compute_curvature: bool = False) -> Tuple[float, Any]:
        """Evaluate phridge ml_i target on F_model and backpropagate to atomic scatterer gradients."""
        if self.sigma_a_per_refl is None:
            self.estimate_sigma_a()

        io_np = np.asarray(self.i_obs.data(), dtype=np.float64)
        si_np = np.asarray(self.i_obs.sigmas(), dtype=np.float64)
        eps_np = np.asarray(self.i_obs.epsilons().data().as_double(), dtype=np.float64)
        cen_np = np.asarray(self.i_obs.centric_flags().data(), dtype=bool)
        sa_np = np.asarray(self.sigma_a_per_refl, dtype=np.float64)
        sw_np = np.asarray(self.mean_i_per_refl, dtype=np.float64)
        r_free_np = np.asarray(self.r_free_flags.data(), dtype=bool)

        nu_np = None
        if self.nu_per_refl is not None:
            nu_np = np.asarray(self.nu_per_refl, dtype=np.float64)
        elif self.nu is not None:
            nu_np = np.full(self.i_obs.size(), self.nu, dtype=np.float64)

        if self.target_type == "amplitude":
            f_obs = self.get_f_obs()
            fo_np = np.asarray(f_obs.data(), dtype=np.float64)
            si_np = np.asarray(f_obs.sigmas(), dtype=np.float64)
            beta_cctbx = np.maximum(sw_np * (1.0 - np.clip(sa_np, 0.0, 0.9999)**2), 1e-6)
            obs = Observations.from_numpy(
                device=self.device,
                dtype=self.float_dtype,
                data=fo_np,
                sigmas=si_np,
                epsilon=eps_np,
                centric=cen_np,
                alpha=sa_np,
                beta=beta_cctbx,
                r_free=r_free_np,
            )
            tgt_f = MaximumLikelihoodAmplitude(scale_factor=1.0)
            fmod_np = np.asarray(self.f_model.data(), dtype=np.complex128)
            fmod_t = torch.as_tensor(fmod_np, dtype=self.complex_dtype, device=self.torch_device)
            ev = tgt_f.evaluate(fmod_t, obs, compute_curvature=compute_curvature)
            self.target_eval = ev
            self.target_value = float(ev.value)
            self.target_value_test = float(ev.value_test) if ev.value_test is not None else float("nan")
            self.d_target_d_f_model = ev.d_target_d_f_calc
        else:
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

            fmod_np = np.asarray(self.f_model.data(), dtype=np.complex128)
            fmod_t = torch.as_tensor(fmod_np, dtype=self.complex_dtype, device=self.torch_device)

            tgt_kwargs: Dict[str, Any] = {}
            if self.nu is not None:
                tgt_kwargs["nu"] = float(self.nu)
            tgt = IntensityLogLikelihood(**tgt_kwargs)
            ev = tgt.evaluate(fmod_t, obs, compute_curvature=compute_curvature)
            self.target_eval = ev
            self.target_value = float(ev.value)
            self.target_value_test = float(ev.value_test) if ev.value_test is not None else float("nan")
            self.d_target_d_f_model = ev.d_target_d_f_calc

        self.d_target_d_f_calc = self.k_total * self.d_target_d_f_model

        c_arr = np.ascontiguousarray(self.d_target_d_f_calc, dtype=np.complex128)
        dtdf_flex = flex_complex_double(c_arr)
        self.gradients = self.sf.gradients(
            self.xray_structure,
            self.i_obs,
            dtdf_flex,
            d_min=self.d_min,
        )
        if compute_curvature:
            self.compute_hessian_preconditioners(ev)
        return self.target_value, self.gradients

    def write_pdb(self, file_name: str | Path = "output_refined.pdb") -> str:
        """Write the current atomic coordinates to a PDB file."""
        file_name_str = str(file_name)
        if self.hierarchy is not None:
            self.hierarchy.adopt_xray_structure(self.xray_structure)
            self.hierarchy.write_pdb_file(
                file_name=file_name_str,
                crystal_symmetry=self.xray_structure.crystal_symmetry(),
            )
        else:
            with open(file_name_str, "w") as f:
                f.write(self.xray_structure.as_pdb_file())
        return file_name_str

    def summary(self) -> Dict[str, Any]:
        """Compute and return refinement statistics and diagnostic summary."""
        io_np = np.asarray(self.i_obs.data(), dtype=np.float64)
        si_np = np.asarray(self.i_obs.sigmas(), dtype=np.float64)
        fc_amp = np.abs(np.asarray(self.f_model.data())) if self.f_model is not None else np.zeros_like(io_np)

        free_flags = np.asarray(self.r_free_flags.data(), dtype=bool)
        work_flags = ~free_flags

        i_calc = fc_amp**2
        r_intensity_num = np.sum(np.abs(io_np - i_calc))
        r_intensity_den = np.sum(io_np)
        r_intensity = float(r_intensity_num / max(r_intensity_den, 1e-12))

        r_intensity_free = float(
            np.sum(np.abs(io_np[free_flags] - i_calc[free_flags]))
            / max(np.sum(io_np[free_flags]), 1e-12)
        ) if np.sum(free_flags) > 0 else float("nan")

        def _calc_cc(obs, calc):
            if len(obs) < 2:
                return float("nan")
            c = np.corrcoef(obs, calc)[0, 1]
            return float(c) if np.isfinite(c) else float("nan")

        cc_work_i = _calc_cc(io_np[work_flags], i_calc[work_flags])
        cc_free_i = _calc_cc(io_np[free_flags], i_calc[free_flags]) if np.sum(free_flags) > 0 else float("nan")
        cc_total_i = _calc_cc(io_np, i_calc)

        fo_amp = np.sqrt(np.maximum(io_np, 0.0))
        r_work = float(
            np.sum(np.abs(fo_amp[work_flags] - fc_amp[work_flags]))
            / max(np.sum(fo_amp[work_flags]), 1e-12)
        )
        r_free = float(
            np.sum(np.abs(fo_amp[free_flags] - fc_amp[free_flags]))
            / max(np.sum(fo_amp[free_flags]), 1e-12)
        ) if np.sum(free_flags) > 0 else float("nan")

        mean_sa = float(np.mean(self.sigma_a_per_refl)) if self.sigma_a_per_refl is not None else float("nan")
        mean_nu = float(np.mean(self.nu_per_refl)) if self.nu_per_refl is not None else (self.nu if self.nu is not None else float("nan"))

        grad_rms = float("nan")
        grad_max = float("nan")
        if self.gradients is not None:
            g_cart = np.asarray(self.gradients.d_target_d_site_cart(), dtype=np.float64)
            norms = np.linalg.norm(g_cart, axis=1)
            grad_rms = float(np.sqrt(np.mean(norms**2)))
            grad_max = float(np.max(norms))

        b_vals = np.array([
            float(sc.u_iso * 8.0 * np.pi**2)
            for sc in self.xray_structure.scatterers()
            if sc.flags.use_u_iso() and sc.occupancy > 0.0
        ])
        b_mean = float(np.mean(b_vals)) if len(b_vals) > 0 else float("nan")
        b_min = float(np.min(b_vals)) if len(b_vals) > 0 else float("nan")
        b_max = float(np.max(b_vals)) if len(b_vals) > 0 else float("nan")
        b_std = float(np.std(b_vals)) if len(b_vals) > 0 else float("nan")

        grad_b_rms = float("nan")
        grad_b_max = float("nan")
        if self.gradients is not None and hasattr(self.gradients, "raw") and hasattr(self.gradients.raw, "d_u_iso"):
            g_u = np.asarray(self.gradients.raw.d_u_iso, dtype=np.float64)
            g_b = g_u / (8.0 * np.pi**2)
            grad_b_rms = float(np.sqrt(np.mean(g_b**2)))
            grad_b_max = float(np.max(np.abs(g_b)))

        gst = self.compute_geometry_statistics()

        adp_e = float("nan")
        if self.adp_prior is not None:
            try:
                X_s = np.asarray(self.xray_structure.sites_cart(), dtype=np.float64)
                beta_s = np.log(np.maximum(b_vals, 1e-4))
                adp_e = float(self.adp_prior.energy_vec(beta_s, sites=X_s).item())
            except Exception:
                pass

        return {
            "target_type": self.target_type,
            "n_refl": self.i_obs.size(),
            "n_work": int(np.sum(work_flags)),
            "n_free": int(np.sum(free_flags)),
            "d_max": float(self.i_obs.d_max_min()[0]),
            "d_min": float(self.i_obs.d_max_min()[1]),
            "mean_i_over_sig": float(np.mean(io_np / np.maximum(si_np, 1e-6))),
            "k_total": self.k_total,
            "k_sol": self.k_sol if self.use_bulk_solvent else None,
            "b_sol": self.b_sol if self.use_bulk_solvent else None,
            "mean_sigma_a": mean_sa,
            "sigma_a_binned": self.sigma_a_binned,
            "mean_nu": mean_nu,
            "nu_binned": self.nu_binned,
            "target_work": self.target_value,
            "target_test": self.target_value_test,
            "r_work": r_work,
            "r_free": r_free,
            "r_intensity": r_intensity,
            "r_intensity_free": r_intensity_free,
            "cc_work_i": cc_work_i,
            "cc_free_i": cc_free_i,
            "cc_total_i": cc_total_i,
            "grad_cart_rms": grad_rms,
            "grad_cart_max": grad_max,
            "b_mean": b_mean,
            "b_min": b_min,
            "b_max": b_max,
            "b_std": b_std,
            "grad_b_rms": grad_b_rms,
            "grad_b_max": grad_b_max,
            "bond_rms": gst["bond_rms"],
            "bond_max": gst["bond_max"],
            "angle_rms": gst["angle_rms"],
            "angle_max": gst["angle_max"],
            "planarity_rms": gst["planarity_rms"],
            "planarity_max": gst["planarity_max"],
            "chirality_rms": gst["chirality_rms"],
            "dihedral_rms": gst["dihedral_rms"],
            "geom_target": gst["target"],
            "geom_is_acceptable": gst["is_acceptable"],
            "adp_energy": adp_e,
        }

    def print_summary(self) -> None:
        """Print summary statistics to standard output."""
        s = self.summary()
        print("\n" + "=" * 60)
        print(" phridge Intensity Refinement & Map Generation Summary")
        print("=" * 60)
        print(f" Reflections:        {s['n_refl']} total ({s['n_work']} work, {s['n_free']} free)")
        target_desc = "Intensity Log-Likelihood (ml_i)" if s['target_type'] == 'intensity' else "Amplitude Maximum Likelihood (ml_f)"
        print(f" Target Function:    {target_desc}")
        print(f" Resolution:         {s['d_max']:.2f} - {s['d_min']:.2f} Å")
        print(f" Mean <I / sigma>:   {s['mean_i_over_sig']:.2f}")
        print(f" Overall Scale k:    {s['k_total']:.4e}")
        if self.use_bulk_solvent:
            print(f" Bulk Solvent:       k_sol = {s['k_sol']:.3f}, B_sol = {s['b_sol']:.1f} Å²")

        if not np.isnan(s.get("adp_energy", float("nan"))):
            print(f" ADP Restraints E:   {s['adp_energy']:.2f} (scale-invariant Student-t prior)")

        if not np.isnan(s["mean_nu"]):
            nu_desc = f"Student-t (mean nu = {s['mean_nu']:.2f})"
        else:
            nu_desc = "Gaussian / Normal"
        print(f" Noise Model:        {nu_desc}")

        sa_details = []
        if self.overlap_bins > 0:
            sa_details.append(f"overlap={self.overlap_bins}")
        else:
            sa_details.append("disjoint")
        if self.tv_norm > 0:
            sa_details.append(f"TV lam={self.tv_norm}")
        if self.enforce_monotonic:
            sa_details.append("monotonic")
        sa_mode_str = f" ({', '.join(sa_details)})" if sa_details else ""
        print(f" Mean Sigma_A:       {s['mean_sigma_a']:.4f}{sa_mode_str}")
        print(f" Target NLL (work):  {s['target_work']:.4f}")
        if not np.isnan(s['target_test']):
            print(f" Target NLL (free):  {s['target_test']:.4f}")
        print(f" R_work / R_free:    {s['r_work']*100:.2f}% / {s['r_free']*100:.2f}%")
        print(f" R_intensity (work/free): {s['r_intensity']*100:.2f}% / {s['r_intensity_free']*100:.2f}%")
        print(f" CC_intensity (work/free): {s['cc_work_i']:.4f} / {s['cc_free_i']:.4f}")
        if not np.isnan(s.get("b_mean", float("nan"))):
            print(f" Mean B_iso:         {s['b_mean']:.2f} Å² (min: {s['b_min']:.2f}, max: {s['b_max']:.2f}, std: {s['b_std']:.2f})")
        if not np.isnan(s.get("grad_b_rms", float("nan"))):
            print(f" B Gradient RMS:     {s['grad_b_rms']:.4e} (max: {s['grad_b_max']:.4e})")
        if not np.isnan(s['grad_cart_rms']):
            print(f" Site Gradient RMS:  {s['grad_cart_rms']:.4e} (max: {s['grad_cart_max']:.4e})")

        if s.get("bond_rms") is not None and np.isfinite(s.get("bond_rms", float("nan"))):
            print("\n Stereochemical Validation (CCTBX Restraints):")
            acc_str = "[ACCEPTABLE]" if s.get("geom_is_acceptable") else "[OUT-OF-TARGET]"
            print(f"   Validation Status:   {acc_str}")
            print(f"   Bond lengths RMSD:   {s['bond_rms']:.4f} Å  (max: {s['bond_max']:.4f} Å, target: <= 0.020 Å)")
            print(f"   Bond angles RMSD:    {s['angle_rms']:.2f}°  (max: {s['angle_max']:.2f}°, target: <= 2.20°)")
            print(f"   Planarity RMSD:      {s['planarity_rms']:.4f} Å  (max: {s['planarity_max']:.4f} Å, target: <= 0.010 Å)")
            print(f"   Chirality RMSD:      {s['chirality_rms']:.4f} Å³ (target: <= 0.100 Å³)")

        if self.sigma_a_binned:
            has_nu = bool(self.nu_binned)
            print("\n Parameter estimates per resolution bin:")
            if has_nu:
                print("   Bin | Resolution (Å) | Sigma_A |   Nu  ")
                print("   " + "-" * 42)
                for i_bin in self.binner.range_used():
                    d_range = self.binner.bin_d_range(i_bin)
                    sa_val = self.sigma_a_binned.get(i_bin, float("nan"))
                    nu_val = self.nu_binned.get(i_bin, float("nan"))
                    print(f"   {i_bin:3d} | {d_range[0]:6.2f} - {d_range[1]:5.2f} | {sa_val:7.4f} | {nu_val:6.2f}")
            else:
                print("   Bin | Resolution (Å) | Sigma_A")
                print("   " + "-" * 32)
                for i_bin in self.binner.range_used():
                    d_range = self.binner.bin_d_range(i_bin)
                    sa_val = self.sigma_a_binned.get(i_bin, float("nan"))
                    print(f"   {i_bin:3d} | {d_range[0]:6.2f} - {d_range[1]:5.2f} | {sa_val:7.4f}")
        print("=" * 60 + "\n")
