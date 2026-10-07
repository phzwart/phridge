"""Intensity-based crystallographic model state and likelihood evaluation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from phridge.shims.cctbx._flex import complex_double as flex_complex_double
from phridge.shims.cctbx._flex import double as flex_double
from phridge.shims.cctbx._flex import flex_mean

from phridge.sfcalc.client import StructureFactorServer
from phridge.d.map_coefficients import IntensityMapMixin
from phridge.tools.intensity.steps import IntensityStepsMixin
from phridge.tools.intensity.math_utils import (
    golden_section_search_torch,
    isotonic_non_increasing,
    tv_denoise_1d,
)
from phridge.legacy.intensity_refine import IntensityRefineMixin
from phridge.contrib.intensity_ll import register
from phridge.d.rules.mli_quad import log_likelihood_normal, log_likelihood_t, normalize
from phridge.d.rules.mli_quad.target import IntensityLogLikelihood
from phridge.d.rules.amplitude import Observations
from phridge.d.rules.amplitude.maximum_likelihood import MaximumLikelihoodAmplitude


class IntensityModel(IntensityStepsMixin, IntensityRefineMixin, IntensityMapMixin):
    """Drives intensity-based refinement, direct parameter estimation, and map generation.

    Parameters
    ----------
    xray_structure : cctbx.xray.structure
        Atomic model with crystal symmetry.
    i_obs : cctbx.miller.array
        Observed intensities (must have positive or estimated sigmas).
    hierarchy : iotbx.pdb.hierarchy.root, optional
        PDB hierarchy for atom naming and writing refined models.
    r_free_flags : cctbx.miller.array, optional
        Boolean test set reflection flags.
    sf : StructureFactorServer, optional
        Handle to the structure-factor engine (defaults to memory=True).
    use_bulk_solvent : bool, default False
        If True, compute real-space solvent mask via cctbx map gridding and
        refine k_sol, B_sol alongside k_total directly against ml_i NLL.
        Set ``dt_mask.enabled`` (or ``PHRIDGE_DT_MASK=1``) to modulate that
        mask by a nonlinear function of its distance transform.
    n_bins : int, default 10
        Number of resolution bins for Wilson normalization and sigmaA estimation.
    d_min : float, optional
        High-resolution cutoff in Angstroms.
    device : str, default "cpu"
        Compute device for phridge ("cpu" or "cuda").
    """

    def __init__(
        self,
        xray_structure: Any,
        i_obs: Any,
        hierarchy: Optional[Any] = None,
        r_free_flags: Optional[Any] = None,
        sf: Optional[StructureFactorServer] = None,
        use_bulk_solvent: bool = False,
        n_bins: int = 10,
        d_min: Optional[float] = None,
        overlap_bins: int = 1,
        tv_norm: float = 0.0,
        enforce_monotonic: bool = False,
        interpolate_sigma_a: bool = True,
        nu: Optional[float] = None,
        estimate_nu: bool = False,
        nu_mode: str = "global",
        target_type: str = "intensity",
        restraints_manager: Optional[Any] = None,
        device: str = "cpu",
        dt_mask: Optional[Any] = None,
    ) -> None:
        register()
        from phridge.worker.convert import resolve_device
        from phridge.sfcalc.ops import set_device

        resolved_device = resolve_device(device)
        self.device = resolved_device
        set_device(resolved_device)
        self.torch_device = torch.device(resolved_device)
        self.is_mps = self.torch_device.type == "mps"
        self.float_dtype = torch.float32 if self.is_mps else torch.float64
        self.complex_dtype = torch.complex64 if self.is_mps else torch.complex128
        self.sf = sf if sf is not None else StructureFactorServer(memory=True, device=resolved_device)
        self.use_bulk_solvent = use_bulk_solvent
        from phridge.r.components.mask import DistanceMaskOptions, options_from_env

        if dt_mask is None:
            self.dt_mask = options_from_env()
        elif isinstance(dt_mask, DistanceMaskOptions):
            self.dt_mask = dt_mask
        else:
            self.dt_mask = DistanceMaskOptions.model_validate(dt_mask)
        self.n_bins = int(n_bins)
        self.d_min = float(d_min) if d_min is not None else float(i_obs.d_min())
        self.overlap_bins: int = int(overlap_bins)
        self.tv_norm: float = float(tv_norm)
        self.enforce_monotonic: bool = bool(enforce_monotonic)
        self.interpolate_sigma_a: bool = bool(interpolate_sigma_a)

        # Target type: 'intensity' (ml_i) or 'amplitude' (ml_f)
        t_raw = str(target_type).lower().strip()
        if t_raw in ("amplitude", "ml_f", "f"):
            self.target_type = "amplitude"
        else:
            self.target_type = "intensity"
        self.f_obs: Optional[Any] = None

        # Student-t noise degrees of freedom (nu)
        if nu is not None:
            self.nu: Optional[float] = float(nu)
        elif self.target_type == "intensity" and bool(estimate_nu):
            self.nu = 7.0
        else:
            self.nu = None
        self.estimate_nu_flag: bool = bool(estimate_nu)
        self.nu_mode: str = str(nu_mode)
        self.nu_binned: Dict[int, float] = {}
        self.nu_per_refl: Optional[flex_double] = None

        # Align crystal symmetry
        cs = i_obs.crystal_symmetry()
        self.xray_structure = xray_structure.customized_copy(crystal_symmetry=cs)
        self.hierarchy = hierarchy
        self.restraints_manager = restraints_manager

        # ADP prior / B-factor restraints
        self.adp_prior: Optional[Any] = None
        self.adp_options: Optional[Any] = None
        self.adp_tables: Optional[Any] = None
        self.packed_restraints: Optional[Any] = None

        # Ensure refinement flags are set if none are enabled
        has_any_flags = any(sc.flags.grad_site() for sc in self.xray_structure.scatterers())
        if not has_any_flags:
            for sc in self.xray_structure.scatterers():
                if sc.occupancy > 0.0:
                    sc.flags.set_grad_site(True)
                    sc.flags.set_grad_u_iso(True)
                else:
                    sc.flags.set_grad_site(False)
                    sc.flags.set_grad_u_iso(False)
        if d_min is not None and float(i_obs.d_min()) < d_min:
            self.i_obs = i_obs.resolution_filter(d_min=d_min)
        else:
            self.i_obs = i_obs

        # Ensure sigmas exist
        if self.i_obs.sigmas() is None:
            mean_val = float(flex_mean(self.i_obs.data()))
            self.i_obs.set_sigmas(flex_double(self.i_obs.size(), max(mean_val * 0.1, 1.0)))

        # Align r_free_flags
        if r_free_flags is not None:
            self.r_free_flags = r_free_flags.common_set(self.i_obs)
        else:
            self.r_free_flags = self.i_obs.generate_r_free_flags(fraction=0.05)

        # Resolution binning and normalization factors (Sigma_wilson = <I/eps>_bin)
        self.binner = None
        self.mean_i_per_refl: Optional[flex_double] = None
        self.setup_bins(self.n_bins)

        # Structure factors & scale parameters
        self.k_total: float = 1.0
        self.k_sol: float = 0.35
        self.b_sol: float = 45.0
        self.f_calc: Optional[Any] = None
        self.f_mask: Optional[Any] = None
        self.f_mask_mod: Optional[Any] = None
        self.f_model: Optional[Any] = None

        # Sigma_A
        self.sigma_a_binned: Dict[int, float] = {}
        self.sigma_a_per_refl: Optional[flex_double] = None

        # Target and gradients
        self.target_eval: Optional[Any] = None
        self.target_value: float = float("nan")
        self.target_value_test: float = float("nan")
        self.d_target_d_f_model: Optional[np.ndarray] = None
        self.d_target_d_f_calc: Optional[np.ndarray] = None
        self.gradients: Optional[Any] = None

    def setup_bins(self, n_bins: int) -> None:
        """Setup resolution bins and compute normalization factors Sigma_wilson = <I/eps>_bin."""
        self.binner = self.i_obs.setup_binner(n_bins=n_bins)
        eps = self.i_obs.epsilons().data().as_double()
        io_data = self.i_obs.data()

        mean_i = flex_double(self.i_obs.size(), 1.0)
        for i_bin in self.binner.range_used():
            sel = self.binner.selection(i_bin)
            if sel.count(True) > 0:
                i_sel = io_data.select(sel)
                eps_sel = eps.select(sel)
                m = float(flex_mean(i_sel / eps_sel))
                mean_i.set_selected(sel, max(m, 1e-4))
        self.mean_i_per_refl = mean_i

        if self.nu is not None:
            self.nu_binned = {b: self.nu for b in self.binner.range_used()}
            self.nu_per_refl = flex_double(self.i_obs.size(), self.nu)

    def get_f_obs(self) -> Any:
        """Return amplitude Miller array, computing French-Wilson amplitudes if input was intensity."""
        if self.f_obs is None:
            if self.i_obs.is_xray_amplitude_array():
                self.f_obs = self.i_obs
            else:
                fw = self.i_obs.french_wilson()
                matches = self.i_obs.match_indices(fw)
                self.f_obs = fw.select(matches.permutation())
        return self.f_obs

    def compute_mask(self) -> Any:
        """Compute F_mask; optionally modulate the real-space mask by its distance transform."""
        from phridge.r.components.mask import build_f_masks

        built = build_f_masks(self.i_obs, self.xray_structure, self.dt_mask)
        self.f_mask = built.f_mask
        self.f_mask_mod = built.f_mask_mod
        return self.f_mask

    def compute_f_calc(self) -> Any:
        """Calculate atomic model structure factors using phridge."""
        self.f_calc = self.sf.f_calc(self.xray_structure, self.i_obs, d_min=self.d_min)
        return self.f_calc

    def _assemble_f_model(self, k: float, ksol: float, bsol: float) -> np.ndarray:
        """Assemble complex F_model = k * [ F_calc + ksol * exp(-bsol*s^2/4) * F_mask ]."""
        fc_np = np.asarray(self.f_calc.data(), dtype=np.complex128)
        if self.use_bulk_solvent and self.f_mask is not None:
            s2 = np.asarray(self.i_obs.d_star_sq().data(), dtype=np.float64)
            fm_np = np.asarray(self.f_mask.data(), dtype=np.complex128)
            f_sol = ksol * np.exp(-bsol * s2 / 4.0) * fm_np
            if self.f_mask_mod is not None:
                alpha = float(self.dt_mask.alpha) if self.dt_mask is not None else 0.0
                f_sol = f_sol + ksol * alpha * np.asarray(self.f_mask_mod.data(), dtype=np.complex128)
            return k * (fc_np + f_sol)
        return k * fc_np

    def refine_scale_and_solvent(self, max_iter: int = 50, recompute_mask: bool = False) -> Dict[str, float]:
        """Directly minimize target NLL wrt scale and bulk solvent parameters."""
        if self.f_calc is None:
            self.compute_f_calc()
        if self.use_bulk_solvent and (self.f_mask is None or recompute_mask):
            self.compute_mask()

        # Initial analytical overall scale
        fc_abs = np.abs(np.asarray(self.f_calc.data()))
        num = np.sum(np.asarray(self.i_obs.data()) * fc_abs**2)
        den = np.sum(fc_abs**4)
        k0 = float(np.sqrt(max(num / max(den, 1e-12), 1e-6)))

        work_sel = torch.as_tensor(~np.asarray(self.r_free_flags.data(), dtype=bool), device=self.torch_device)
        eps_t = torch.as_tensor(np.asarray(self.i_obs.epsilons().data().as_double(), dtype=np.float64), dtype=self.float_dtype, device=self.torch_device)
        sw_t = torch.as_tensor(np.asarray(self.mean_i_per_refl, dtype=np.float64), dtype=self.float_dtype, device=self.torch_device)
        cen_t = torch.as_tensor(np.asarray(self.i_obs.centric_flags().data(), dtype=bool), device=self.torch_device)

        fc_np = np.asarray(self.f_calc.data(), dtype=np.complex128)
        fc_t = torch.as_tensor(fc_np, dtype=self.complex_dtype, device=self.torch_device)

        fm_mod_t = None
        dt_alpha_t = None
        if self.use_bulk_solvent and self.f_mask is not None:
            fm_np = np.asarray(self.f_mask.data(), dtype=np.complex128)
            fm_t = torch.as_tensor(fm_np, dtype=self.complex_dtype, device=self.torch_device)
            s2_t = torch.as_tensor(np.asarray(self.i_obs.d_star_sq().data(), dtype=np.float64), dtype=self.float_dtype, device=self.torch_device)
            if self.f_mask_mod is not None:
                fm_mod_t = torch.as_tensor(
                    np.asarray(self.f_mask_mod.data(), dtype=np.complex128),
                    dtype=self.complex_dtype,
                    device=self.torch_device,
                )
                dt_alpha_t = torch.tensor(
                    float(self.dt_mask.alpha) if self.dt_mask is not None else 0.0,
                    dtype=self.float_dtype,
                    device=self.torch_device,
                )

        log_k = torch.tensor(math.log(max(k0, 1e-4)), dtype=self.float_dtype, device=self.torch_device, requires_grad=True)
        if self.use_bulk_solvent and self.f_mask is not None:
            logit_ksol = torch.tensor(0.0, dtype=self.float_dtype, device=self.torch_device, requires_grad=True)
            logit_bsol = torch.tensor(-0.5, dtype=self.float_dtype, device=self.torch_device, requires_grad=True)
            params = [log_k, logit_ksol, logit_bsol]
        else:
            params = [log_k]

        opt = torch.optim.LBFGS(params, lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe")

        if self.target_type == "amplitude":
            f_obs = self.get_f_obs()
            fo_t = torch.as_tensor(np.asarray(f_obs.data(), dtype=np.float64), dtype=self.float_dtype, device=self.torch_device)
            si_t = torch.as_tensor(np.asarray(f_obs.sigmas(), dtype=np.float64), dtype=self.float_dtype, device=self.torch_device)
            sa_init = torch.full_like(fo_t, 0.8)
            beta_init = sw_t * (1.0 - sa_init**2)
            obs_f = Observations(
                data=fo_t,
                sigmas=si_t,
                epsilon=eps_t,
                centric=cen_t,
                alpha=sa_init,
                beta=beta_init,
                r_free=torch.as_tensor(np.asarray(self.r_free_flags.data(), dtype=bool), device=self.torch_device),
            )
            tgt_f = MaximumLikelihoodAmplitude(scale_factor=1.0)

            def closure():
                opt.zero_grad()
                k = torch.exp(log_k)
                if self.use_bulk_solvent and self.f_mask is not None:
                    ksol = torch.sigmoid(logit_ksol)
                    bsol = 200.0 * torch.sigmoid(logit_bsol)
                    f_sol = ksol * torch.exp(-bsol * s2_t / 4.0) * fm_t
                    if fm_mod_t is not None and dt_alpha_t is not None:
                        f_sol = f_sol + ksol * dt_alpha_t * fm_mod_t
                    f_mod = k * (fc_t + f_sol)
                else:
                    f_mod = k * fc_t
                per_refl = tgt_f.per_reflection(f_mod, obs_f)
                loss = per_refl[work_sel].mean()
                loss.backward()
                return loss

        else:
            io_t = torch.as_tensor(np.asarray(self.i_obs.data(), dtype=np.float64), dtype=self.float_dtype, device=self.torch_device)
            si_t = torch.as_tensor(np.asarray(self.i_obs.sigmas(), dtype=np.float64), dtype=self.float_dtype, device=self.torch_device)
            sa_init = torch.full_like(io_t, 0.8)

            def closure():
                opt.zero_grad()
                k = torch.exp(log_k)
                if self.use_bulk_solvent and self.f_mask is not None:
                    ksol = torch.sigmoid(logit_ksol)
                    bsol = 200.0 * torch.sigmoid(logit_bsol)
                    f_sol = ksol * torch.exp(-bsol * s2_t / 4.0) * fm_t
                    if fm_mod_t is not None and dt_alpha_t is not None:
                        f_sol = f_sol + ksol * dt_alpha_t * fm_mod_t
                    f_mod = torch.abs(k * (fc_t + f_sol))
                else:
                    f_mod = k * torch.abs(fc_t)
                Ec, _, Zo, sZ = normalize(f_mod, io_t, si_t, eps_t, sw_t, sa_init)
                ll = log_likelihood_normal(Ec[work_sel], sa_init[work_sel], Zo[work_sel], sZ[work_sel], cen_t[work_sel])
                loss = -ll.mean()
                loss.backward()
                return loss

        opt.step(closure)
        self.k_total = float(torch.exp(log_k).item())
        if self.use_bulk_solvent and self.f_mask is not None:
            self.k_sol = float(torch.sigmoid(logit_ksol).item())
            self.b_sol = float((200.0 * torch.sigmoid(logit_bsol)).item())

        fmod_complex = self._assemble_f_model(self.k_total, self.k_sol, self.b_sol)
        c_arr = np.ascontiguousarray(fmod_complex, dtype=np.complex128)
        self.f_model = self.i_obs.customized_copy(data=flex_complex_double(c_arr))
        return {"k_total": self.k_total, "k_sol": self.k_sol, "b_sol": self.b_sol}

