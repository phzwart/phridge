"""Intensity-based crystallographic model, target evaluation, and map tool.

Bridges cctbx (for PDB/MTZ I/O, solvent mask gridding, and map export) with
phridge (for GPU/autograd structure factors, direct intensity likelihood
parameter estimation, and parameter gradients).

All estimation (normalization, scale factor, bulk solvent, and sigmaA) is
performed directly from observed intensities by minimizing the phridge
``ml_i`` intensity negative log-likelihood (NLL), without converting intensities
to amplitudes or relying on cctbx's legacy ``mmtbx.f_model``.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from cctbx.array_family import flex
import iotbx.ccp4_map
import iotbx.pdb
from iotbx.reflection_file_reader import any_reflection_file

from phridge.client import StructureFactorServer
from phridge.client.convert import crystal_from_cctbx
from phridge.contrib.intensity_ll import register
from phridge.contrib.intensity_ll.mli import log_likelihood_normal, log_likelihood_t, normalize
from phridge.contrib.intensity_ll.target import IntensityLogLikelihood
from phridge.sfcalc.targets import Observations
from phridge.sfcalc.targets.maximum_likelihood import MaximumLikelihoodAmplitude


def _fom_rice_woolfson(
    Ec: np.ndarray,
    sA: np.ndarray,
    Zo: np.ndarray,
    centric: np.ndarray,
) -> np.ndarray:
    """Figure of merit m = <cos(phi - phi_c)> under the Rice/Woolfson conditional distribution.

    For acentric: m = I_1(2X) / I_0(2X) where X = sA * Ec * sqrt(max(Zo, 0)) / (1 - sA^2).
    For centric:  m = tanh(X).
    """
    sA_clamp = np.clip(sA, 1e-4, 0.999)
    denom = np.maximum(1.0 - sA_clamp**2, 1e-6)
    zo_pos = np.maximum(Zo, 0.0)
    X = sA_clamp * Ec * np.sqrt(zo_pos) / denom
    X = np.clip(X, 0.0, 50.0)  # avoid overflow in exp/cosh

    # Centric: tanh(X)
    cen_m = np.tanh(X)

    # Acentric: I_1(2X) / I_0(2X)
    two_X = 2.0 * X
    # Ratio using exponentially scaled Bessel functions i1e(y)/i0e(y) via PyTorch
    two_X_t = torch.as_tensor(two_X, dtype=torch.float64)
    acen_m = (torch.special.i1e(two_X_t) / torch.special.i0e(two_X_t).clamp_min(1e-300)).cpu().numpy()

    fom = np.where(centric, cen_m, acen_m)
    return np.clip(fom, 0.0, 1.0)


def golden_section_search_torch(f, a: float, b: float, tol: float = 1e-2, max_iter: int = 20) -> float:
    """1D bounded scalar minimization using Golden Section Search in PyTorch."""
    invphi = (math.sqrt(5) - 1.0) / 2.0
    invphi2 = (3.0 - math.sqrt(5)) / 2.0
    c = a + invphi2 * (b - a)
    d = a + invphi * (b - a)
    yc = f(c)
    yd = f(d)
    for _ in range(max_iter):
        if (b - a) < tol:
            break
        if yc < yd:
            b = d
            d = c
            yd = yc
            c = a + invphi2 * (b - a)
            yc = f(c)
        else:
            a = c
            c = d
            yc = yd
            d = a + invphi * (b - a)
            yd = f(d)
    return float((a + b) / 2.0)


def tv_denoise_1d(
    y: np.ndarray,
    weights: Optional[np.ndarray] = None,
    lam: float = 0.05,
    lower: float = 0.01,
    upper: float = 0.999,
) -> np.ndarray:
    """1D Total Variation Denoising (TVD) using Huber-smoothed L1 difference penalty in PyTorch.

    Solves:
        min_{x in [lower, upper]^n} 0.5 * sum_i w_i (x_i - y_i)^2 + lam * sum_i sqrt((x_{i+1} - x_i)^2 + eps^2)
    """
    y_np = np.asarray(y, dtype=np.float64)
    n = len(y_np)
    if lam <= 0.0 or n <= 1:
        return np.clip(y_np, lower, upper)
    w_np = np.ones(n, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    w_np = w_np / max(float(np.mean(w_np)), 1e-12)

    y_t = torch.tensor(y_np, dtype=torch.float64)
    w_t = torch.tensor(w_np, dtype=torch.float64)

    span = upper - lower
    y_clip = np.clip(y_np, lower + 1e-5, upper - 1e-5)
    u_init = np.log((y_clip - lower) / (upper - y_clip))
    u = torch.tensor(u_init, dtype=torch.float64, requires_grad=True)

    opt = torch.optim.LBFGS([u], lr=1.0, max_iter=25, line_search_fn="strong_wolfe")
    eps = 1e-6

    def closure():
        opt.zero_grad()
        x = lower + span * torch.sigmoid(u)
        diffs = x[1:] - x[:-1]
        loss = 0.5 * (w_t * (x - y_t) ** 2).sum() + lam * torch.sqrt(diffs**2 + eps).sum()
        loss.backward()
        return loss

    opt.step(closure)
    x_opt = lower + span * torch.sigmoid(u)
    return np.asarray(x_opt.detach().cpu().numpy(), dtype=np.float64)


def isotonic_non_increasing(
    y: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Weighted isotonic regression enforcing non-increasing order: x_0 >= x_1 >= ... >= x_{n-1}
    using the Pool Adjacent Violators Algorithm (PAVA).
    """
    n = len(y)
    if n <= 1:
        return np.asarray(y, dtype=np.float64)
    w = [float(x) for x in (weights if weights is not None else np.ones(n))]
    v = [float(x) for x in y]
    blocks = [[i] for i in range(n)]

    i = 0
    while i < len(blocks) - 1:
        if v[i] < v[i + 1]:  # Violation of non-increasing order
            w_new = w[i] + w[i + 1]
            v_new = (w[i] * v[i] + w[i + 1] * v[i + 1]) / w_new
            w[i] = w_new
            v[i] = v_new
            blocks[i].extend(blocks[i + 1])
            del w[i + 1]
            del v[i + 1]
            del blocks[i + 1]
            if i > 0:
                i -= 1
        else:
            i += 1

    out = np.zeros(n, dtype=np.float64)
    for block_idx, block in enumerate(blocks):
        for idx in block:
            out[idx] = v[block_idx]
    return out


class IntensityModel:
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
        self.nu_per_refl: Optional[flex.double] = None

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
            # Default sigma to mean intensity * 0.1
            mean_val = float(flex.mean(self.i_obs.data()))
            self.i_obs.set_sigmas(flex.double(self.i_obs.size(), max(mean_val * 0.1, 1.0)))

        # Align r_free_flags
        if r_free_flags is not None:
            self.r_free_flags = r_free_flags.common_set(self.i_obs)
        else:
            self.r_free_flags = self.i_obs.generate_r_free_flags(fraction=0.05)

        # Resolution binning and normalization factors (Sigma_wilson = <I/eps>_bin)
        self.binner = None
        self.mean_i_per_refl: Optional[flex.double] = None
        self.setup_bins(self.n_bins)

        # Structure factors & scale parameters
        self.k_total: float = 1.0
        self.k_sol: float = 0.35
        self.b_sol: float = 45.0
        self.f_calc: Optional[Any] = None
        self.f_mask: Optional[Any] = None
        self.f_model: Optional[Any] = None

        # Sigma_A
        self.sigma_a_binned: Dict[int, float] = {}
        self.sigma_a_per_refl: Optional[flex.double] = None

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

        mean_i = flex.double(self.i_obs.size(), 1.0)
        for i_bin in self.binner.range_used():
            sel = self.binner.selection(i_bin)
            if sel.count(True) > 0:
                i_sel = io_data.select(sel)
                eps_sel = eps.select(sel)
                m = float(flex.mean(i_sel / eps_sel))
                mean_i.set_selected(sel, max(m, 1e-4))
        self.mean_i_per_refl = mean_i

        if self.nu is not None:
            self.nu_binned = {b: self.nu for b in self.binner.range_used()}
            self.nu_per_refl = flex.double(self.i_obs.size(), self.nu)

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
        """Compute real-space solvent mask via cctbx map gridding and FFT to reciprocal space."""
        from mmtbx.masks import manager as mask_manager

        # Gridding from miller array
        mm = mask_manager(miller_array=self.i_obs, xray_structure=self.xray_structure)
        f_masks = mm.shell_f_masks()
        self.f_mask = f_masks[0]
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

        if self.use_bulk_solvent and self.f_mask is not None:
            fm_np = np.asarray(self.f_mask.data(), dtype=np.complex128)
            fm_t = torch.as_tensor(fm_np, dtype=self.complex_dtype, device=self.torch_device)
            s2_t = torch.as_tensor(np.asarray(self.i_obs.d_star_sq().data(), dtype=np.float64), dtype=self.float_dtype, device=self.torch_device)

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
                    f_mod = k * (fc_t + f_sol)
                else:
                    f_mod = k * fc_t
                per_refl = tgt_f.per_reflection(f_mod, obs_f)
                loss = per_refl[work_sel].mean()
                loss.backward()
                return loss

        else:
            # Prepare normalized observation tensors for ml_i
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

        # Update F_model
        fmod_complex = self._assemble_f_model(self.k_total, self.k_sol, self.b_sol)
        self.f_model = self.i_obs.customized_copy(data=flex.complex_double(fmod_complex.tolist()))
        return {"k_total": self.k_total, "k_sol": self.k_sol, "b_sol": self.b_sol}

    def estimate_sigma_a(
        self,
        overlap: Optional[int] = None,
        tv_lambda: Optional[float] = None,
        enforce_monotonic: Optional[bool] = None,
        interpolate: Optional[bool] = None,
    ) -> Dict[int, float]:
        """Estimate sigma_A per resolution bin by directly minimizing intensity NLL.

        Supports:
        - Overlapping resolution bins (overlap >= 1) to smooth out sampling fluctuations
          and increase statistical power, especially on sparse free-reflection sets.
        - Total Variation (TV) regularization across resolution shells (tv_lambda > 0)
          to penalize spurious oscillations between adjacent bins.
        - Monotonic non-increasing constraint (enforce_monotonic=True) via weighted
          isotonic regression (PAVA), matching the physical decay of sigma_A with resolution.
        - Continuous resolution interpolation across reflections via cctbx binner.interpolate.
        """
        if self.f_model is None:
            self.refine_scale_and_solvent()

        overlap_val = self.overlap_bins if overlap is None else int(overlap)
        tv_lam_val = self.tv_norm if tv_lambda is None else float(tv_lambda)
        mono_val = self.enforce_monotonic if enforce_monotonic is None else bool(enforce_monotonic)
        interp_val = self.interpolate_sigma_a if interpolate is None else bool(interpolate)

        io_t = torch.as_tensor(np.asarray(self.i_obs.data(), dtype=np.float64), dtype=self.float_dtype, device=self.torch_device)
        si_t = torch.as_tensor(np.asarray(self.i_obs.sigmas(), dtype=np.float64), dtype=self.float_dtype, device=self.torch_device)
        eps_t = torch.as_tensor(np.asarray(self.i_obs.epsilons().data().as_double(), dtype=np.float64), dtype=self.float_dtype, device=self.torch_device)
        sw_t = torch.as_tensor(np.asarray(self.mean_i_per_refl, dtype=np.float64), dtype=self.float_dtype, device=self.torch_device)
        cen_t = torch.as_tensor(np.asarray(self.i_obs.centric_flags().data(), dtype=bool), device=self.torch_device)
        fc_abs_t = torch.as_tensor(np.abs(np.asarray(self.f_model.data())), dtype=self.float_dtype, device=self.torch_device)

        # Base normalization (sigmaA slot arbitrary here since we pass it inside the loop)
        dummy_sa = torch.full_like(fc_abs_t, 0.5)
        Ec, _, Zo, sZ = normalize(fc_abs_t, io_t, si_t, eps_t, sw_t, dummy_sa)

        free_flags = np.asarray(self.r_free_flags.data(), dtype=bool)
        n_free = int(np.sum(free_flags))
        # Use free set if sufficient reflections exist, else work set
        use_free = n_free >= 20

        bins = list(self.binner.range_used())
        n_bins = len(bins)
        bin_indices = torch.zeros(self.i_obs.size(), dtype=torch.int64, device=self.torch_device)
        weights_list: List[float] = []
        for idx, i_bin in enumerate(bins):
            sel = np.asarray(self.binner.selection(i_bin), dtype=bool)
            bin_indices[torch.as_tensor(sel, device=self.torch_device)] = idx
            weights_list.append(float(np.sum(sel & free_flags) if use_free else np.sum(sel)))

        fit_sel = free_flags if use_free else np.ones(self.i_obs.size(), dtype=bool)
        fit_sel_t = torch.as_tensor(fit_sel, device=self.torch_device)

        # PyTorch joint optimization across all resolution bins
        if mono_val:
            u0 = torch.tensor(1.5, dtype=self.float_dtype, device=self.torch_device, requires_grad=True)
            raw_deltas = torch.zeros(n_bins - 1, dtype=self.float_dtype, device=self.torch_device, requires_grad=True)
            params = [u0, raw_deltas]
        else:
            u = torch.full((n_bins,), 1.5, dtype=self.float_dtype, device=self.torch_device, requires_grad=True)
            params = [u]

        opt = torch.optim.LBFGS(params, lr=1.0, max_iter=25, line_search_fn="strong_wolfe")

        if self.target_type == "amplitude":
            f_obs = self.get_f_obs()
            fo_arr = np.asarray(f_obs.data(), dtype=np.float64)
            si_arr = np.asarray(f_obs.sigmas(), dtype=np.float64)
            fmod_t = torch.as_tensor(np.asarray(self.f_model.data(), dtype=np.complex128), dtype=self.complex_dtype, device=self.torch_device)
            tgt_f = MaximumLikelihoodAmplitude(scale_factor=1.0)
            fo_t = torch.as_tensor(fo_arr, dtype=self.float_dtype, device=self.torch_device)
            si_f_t = torch.as_tensor(si_arr, dtype=self.float_dtype, device=self.torch_device)

            def closure():
                opt.zero_grad()
                if mono_val:
                    deltas = torch.nn.functional.softplus(raw_deltas)
                    cum = torch.cat([torch.zeros(1, dtype=self.float_dtype, device=self.torch_device), torch.cumsum(deltas, dim=0)])
                    logits = u0 - cum
                    sa_bins = 0.01 + 0.989 * torch.sigmoid(logits)
                else:
                    sa_bins = 0.01 + 0.989 * torch.sigmoid(u)
                sa_refl = sa_bins[bin_indices]
                beta_refl = sw_t * (1.0 - sa_refl**2)
                obs_b = Observations(
                    data=fo_t,
                    sigmas=si_f_t,
                    epsilon=eps_t,
                    centric=cen_t,
                    alpha=sa_refl,
                    beta=beta_refl,
                )
                per_refl = tgt_f.per_reflection(fmod_t, obs_b)
                loss = per_refl[fit_sel_t].mean()
                if tv_lam_val > 0.0 and n_bins > 1:
                    diffs = sa_bins[1:] - sa_bins[:-1]
                    loss = loss + tv_lam_val * torch.sqrt(diffs**2 + 1e-6).sum()
                loss.backward()
                return loss

        else:
            dummy_sa = torch.full_like(fc_abs_t, 0.5)
            Ec, _, Zo, sZ = normalize(fc_abs_t, io_t, si_t, eps_t, sw_t, dummy_sa)

            def closure():
                opt.zero_grad()
                if mono_val:
                    deltas = torch.nn.functional.softplus(raw_deltas)
                    cum = torch.cat([torch.zeros(1, dtype=self.float_dtype, device=self.torch_device), torch.cumsum(deltas, dim=0)])
                    logits = u0 - cum
                    sa_bins = 0.01 + 0.989 * torch.sigmoid(logits)
                else:
                    sa_bins = 0.01 + 0.989 * torch.sigmoid(u)
                sa_refl = sa_bins[bin_indices]
                if self.nu is not None:
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

        # 1. Total Variation regularization
        if tv_lam_val > 0.0:
            sa_arr = tv_denoise_1d(sa_arr, weights=weights_arr, lam=tv_lam_val, lower=0.01, upper=0.999)

        # 2. Monotonic non-increasing enforcement
        if mono_val:
            sa_arr = isotonic_non_increasing(sa_arr, weights=weights_arr)

        self.sigma_a_binned = {}
        for idx, i_bin in enumerate(bins):
            self.sigma_a_binned[i_bin] = float(sa_arr[idx])

        # Assign per-reflection sigma_a
        if interp_val and n_bins >= 2:
            try:
                sa_flex = flex.double(sa_arr.tolist())
                interp = self.binner.interpolate(sa_flex, 2.0)
                interp_np = np.clip(np.asarray(interp, dtype=np.float64), 0.01, 0.999)
                self.sigma_a_per_refl = flex.double(interp_np.tolist())
            except Exception:
                sa_per_refl = np.full(self.i_obs.size(), 0.8, dtype=np.float64)
                for idx, i_bin in enumerate(bins):
                    sel = np.asarray(self.binner.selection(i_bin), dtype=bool)
                    sa_per_refl[sel] = sa_arr[idx]
                self.sigma_a_per_refl = flex.double(sa_per_refl.tolist())
        else:
            sa_per_refl = np.full(self.i_obs.size(), 0.8, dtype=np.float64)
            for idx, i_bin in enumerate(bins):
                sel = np.asarray(self.binner.selection(i_bin), dtype=bool)
                sa_per_refl[sel] = sa_arr[idx]
            self.sigma_a_per_refl = flex.double(sa_per_refl.tolist())

        return self.sigma_a_binned

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
        """Estimate or refine Student-t degrees of freedom 'nu' by directly minimizing intensity NLL.

        Parameters
        ----------
        mode : str, optional
            'global' estimates a single overall scalar nu for all reflections (default).
            'binned' estimates a resolution-dependent nu per resolution shell.
        overlap : int, optional
            Adjacent resolution bins to pool for sliding-window estimation (defaults to self.overlap_bins).
        tv_lambda : float, optional
            Total Variation regularization weight across resolution bins (defaults to self.tv_norm).
        bounds : tuple of float, default (3.0, 30.0)
            Search bounds for nu. Must be > 2 for finite variance.
        prior_center : float, default 7.0
            Honest central expectation for global degrees of freedom.
        prior_weight : float, default 0.01
            Regularization weight preventing infinite-variance boundary collapse (nu -> 2).
        n_u : int, default 10
            Number of Gauss quadrature nodes for log(lambda) integration.

        Returns
        -------
        Dict[int, float] if mode == 'binned', or float if mode == 'global'.
        """
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

        # For scalar quadrature optimization, evaluate on CPU to avoid MPS dispatch latency per scalar step
        Ec_c = Ec.detach().cpu()
        sa_c = sa_t.detach().cpu()
        Zo_c = Zo.detach().cpu()
        sZ_c = sZ.detach().cpu()
        cen_c = cen_t.detach().cpu()

        free_flags = np.asarray(self.r_free_flags.data(), dtype=bool)
        n_free = int(np.sum(free_flags))
        use_free = n_free >= 20

        bins = list(self.binner.range_used())
        n_bins = len(bins)

        if eff_mode == "global":
            fit_sel = free_flags if use_free else np.ones(self.i_obs.size(), dtype=bool)
            ec_fit = Ec_c[fit_sel]
            sa_fit = sa_c[fit_sel]
            zo_fit = Zo_c[fit_sel]
            sz_fit = sZ_c[fit_sel]
            cen_fit = cen_c[fit_sel]

            def global_nll(nu_val: float) -> float:
                with torch.no_grad():
                    ll = log_likelihood_t(ec_fit, sa_fit, zo_fit, sz_fit, cen_fit, nu=float(nu_val), n_u=n_u)
                    prior_loss = 0.5 * ((float(nu_val) - prior_center) / 4.0)**2 if prior_weight > 0 else 0.0
                    return -float(ll.sum().item()) / max(float(len(ec_fit)), 1.0) + prior_weight * prior_loss

            self.nu = golden_section_search_torch(global_nll, bounds[0], bounds[1])
            self.nu_binned = {b: self.nu for b in bins}
            self.nu_per_refl = flex.double(self.i_obs.size(), self.nu)
            return self.nu

        # mode == "binned"
        raw_nu_list: List[float] = []
        weights_list: List[float] = []

        for idx, i_bin in enumerate(bins):
            if overlap_val <= 0:
                bin_sel = np.asarray(self.binner.selection(i_bin), dtype=bool)
            else:
                bin_sel = np.zeros(self.i_obs.size(), dtype=bool)
                start_idx = max(0, idx - overlap_val)
                end_idx = min(n_bins - 1, idx + overlap_val)
                for j in range(start_idx, end_idx + 1):
                    bin_sel |= np.asarray(self.binner.selection(bins[j]), dtype=bool)

            fit_sel = (bin_sel & free_flags) if use_free else bin_sel
            if np.sum(fit_sel) < 10:
                fit_sel = bin_sel

            weights_list.append(float(np.sum(fit_sel)))
            ec_b = Ec_c[fit_sel]
            sa_b = sa_c[fit_sel]
            zo_b = Zo_c[fit_sel]
            sz_b = sZ_c[fit_sel]
            cen_b = cen_c[fit_sel]

            def bin_nu_nll(nu_val: float) -> float:
                with torch.no_grad():
                    ll = log_likelihood_t(ec_b, sa_b, zo_b, sz_b, cen_b, nu=float(nu_val), n_u=n_u)
                    prior_loss = 0.5 * ((float(nu_val) - prior_center) / 4.0)**2 if prior_weight > 0 else 0.0
                    return -float(ll.sum().item()) / max(float(len(ec_b)), 1.0) + prior_weight * prior_loss

            opt_nu = golden_section_search_torch(bin_nu_nll, bounds[0], bounds[1])
            raw_nu_list.append(opt_nu)

        nu_arr = np.array(raw_nu_list, dtype=np.float64)
        weights_arr = np.array(weights_list, dtype=np.float64)

        if tv_lam_val > 0.0:
            nu_arr = tv_denoise_1d(nu_arr, weights=weights_arr, lam=tv_lam_val, lower=bounds[0], upper=bounds[1])

        self.nu_binned = {}
        for idx, i_bin in enumerate(bins):
            self.nu_binned[i_bin] = float(nu_arr[idx])

        self.nu = float(np.mean(nu_arr))

        # Assign per-reflection nu using per-bin values (step) so torch.unique has at most n_bins values
        nu_per_refl = np.full(self.i_obs.size(), self.nu, dtype=np.float64)
        for idx, i_bin in enumerate(bins):
            sel = np.asarray(self.binner.selection(i_bin), dtype=bool)
            nu_per_refl[sel] = nu_arr[idx]
        self.nu_per_refl = flex.double(nu_per_refl.tolist())

        return self.nu_binned

    # Alias
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
        # Final pass for sigma_A with updated nu
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

        # Chain rule: F_model = k_total * F_calc + F_sol, so dQ/dF_calc = k_total * dQ/dF_model
        self.d_target_d_f_calc = self.k_total * self.d_target_d_f_model

        # Compute scatterer gradients via phridge
        dtdf_flex = flex.complex_double(self.d_target_d_f_calc.tolist())
        self.gradients = self.sf.gradients(
            self.xray_structure,
            self.i_obs,
            dtdf_flex,
            d_min=self.d_min,
        )
        if compute_curvature:
            self.compute_hessian_preconditioners(ev)
        return self.target_value, self.gradients

    def update_coordinates_step(self, step_scale: float = 0.001, max_shift_angstrom: float = 0.1) -> float:
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
        self.f_model = self.i_obs.customized_copy(data=flex.complex_double(fmod_complex.tolist()))
        new_target, _ = self.compute_target_and_gradients()
        return new_target

    def refine_coordinates(self, max_iterations: int = 5, step_scale: float = 0.001) -> List[float]:
        """Perform iterative coordinate refinement steps."""
        for sc in self.xray_structure.scatterers():
            sc.flags.set_grad_site(sc.occupancy > 0.0)
            sc.flags.set_grad_u_iso(False)

        history = [self.target_value]
        for it in range(max_iterations):
            val = self.update_coordinates_step(step_scale=step_scale)
            history.append(val)
        return history

    def convert_to_isotropic(self) -> None:
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
        self,
        fraction: float = 0.25,
        reset: bool = True,
        reset_to_value: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
        min_b: float = 1.0,
        max_b: float = 200.0,
        distribution: str = "uniform",
        update_model: bool = True,
    ) -> np.ndarray:
        """Reset and/or shake isotropic B-factors with fractional noise (e.g. +/- 25%).

        Parameters
        ----------
        fraction : float, default 0.25
            Fractional shake magnitude (e.g. 0.25 for +/- 25%).
        reset : bool, default True
            If True, resets all scatterers to a uniform base B-factor before shaking.
            If False, shakes each scatterer's existing B-factor individually by +/- fraction.
        reset_to_value : float, optional
            Base B-factor in Å² to reset all atoms to before shaking.
            If None and `reset=True`, defaults to the current mean isotropic B-factor.
        rng : np.random.Generator, optional
            NumPy random generator for reproducibility.
        min_b, max_b : float, default 1.0, 200.0
            Clamping bounds on B-factors in Å².
        distribution : str, default 'uniform'
            'uniform' for exact U(-fraction, +fraction) bounds,
            or 'normal' for Gaussian noise with standard deviation = fraction.
        update_model : bool, default True
            If True, updates `xray_structure`, `hierarchy`, and recomputes `f_calc` and `f_model`.

        Returns
        -------
        np.ndarray
            The array of new shaken isotropic B-factors in Å².
        """
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
            self.xray_structure.set_b_iso(values=flex.double(b_shaken.tolist()))
            if self.hierarchy is not None:
                self.hierarchy.adopt_xray_structure(self.xray_structure)
            self.compute_f_calc()
            fmod_c = self._assemble_f_model(self.k_total, self.k_sol, self.b_sol)
            self.f_model = self.i_obs.customized_copy(data=flex.complex_double(fmod_c.tolist()))
            self.compute_target_and_gradients()

        return b_shaken

    def shake_sites(
        self,
        rmsd: float = 0.10,
        rng: Optional[np.random.Generator] = None,
        update_model: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Shake Cartesian coordinates with random normal noise scaled to exact target RMSD.

        Parameters
        ----------
        rmsd : float, default 0.10
            Target Cartesian RMSD in Å.
        rng : np.random.Generator, optional
            NumPy random generator for reproducibility.
        update_model : bool, default True
            If True, updates `xray_structure`, `hierarchy`, and recomputes `f_calc` and `f_model`.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, float]
            (sites_perturbed, noise, actual_rmsd)
        """
        if rng is None:
            rng = np.random.default_rng()

        sites_orig = np.asarray(self.xray_structure.sites_cart(), dtype=np.float64)
        noise_raw = rng.normal(loc=0.0, scale=1.0, size=sites_orig.shape)
        curr_rmsd = float(np.sqrt(np.mean(noise_raw**2)))
        noise = noise_raw * (rmsd / max(curr_rmsd, 1e-12))
        actual_rmsd = float(np.sqrt(np.mean(noise**2)))
        sites_pert = sites_orig + noise

        if update_model:
            uc = self.xray_structure.unit_cell()
            fract = np.asarray(uc.fractionalization_matrix(), dtype=np.float64).reshape(3, 3)
            sites_frac = (sites_pert @ fract.T).astype(np.float64)
            self.xray_structure.set_sites_frac(flex.vec3_double([tuple(r) for r in sites_frac]))
            if self.hierarchy is not None:
                self.hierarchy.adopt_xray_structure(self.xray_structure)
            self.compute_f_calc()
            fmod_c = self._assemble_f_model(self.k_total, self.k_sol, self.b_sol)
            self.f_model = self.i_obs.customized_copy(data=flex.complex_double(fmod_c.tolist()))
            self.compute_target_and_gradients()

        return sites_pert, noise, actual_rmsd

    def shake(
        self,
        rmsd: Optional[float] = 0.10,
        b_fraction: Optional[float] = 0.25,
        reset_b: bool = True,
        reset_b_value: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, Any]:
        """Perform combined coordinate shake and isotropic B-factor reset with fractional shake.

        Parameters
        ----------
        rmsd : float, optional
            Target Cartesian coordinate shift in Å (e.g. 0.10 Å). If None or 0, sites are unchanged.
        b_fraction : float, optional
            Fractional B-factor shake magnitude (e.g. 0.25 for +/- 25%). If None or 0, B-factors unchanged.
        reset_b : bool, default True
            If True, resets all atoms to mean B (or reset_b_value) before applying fractional shake.
        reset_b_value : float, optional
            Explicit base B-factor to reset to (if None, uses model mean B).
        rng : np.random.Generator, optional
            Random generator for reproducibility.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing shaken sites, noise vector, actual coordinate RMSD,
            and shaken B-factors.
        """
        if rng is None:
            rng = np.random.default_rng()

        result: Dict[str, Any] = {}

        if rmsd is not None and rmsd > 0:
            sites_pert, noise, actual_rmsd = self.shake_sites(rmsd=rmsd, rng=rng, update_model=True)
            result["sites_perturbed"] = sites_pert
            result["noise"] = noise
            result["actual_rmsd"] = actual_rmsd

        if b_fraction is not None and b_fraction > 0:
            b_shaken = self.shake_b_iso(
                fraction=b_fraction,
                reset=reset_b,
                reset_to_value=reset_b_value,
                rng=rng,
                update_model=True,
            )
            result["b_shaken"] = b_shaken
            result["b_mean"] = float(np.mean(b_shaken))

        return result

    def update_b_iso_step(
        self,
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
            self.f_model = self.i_obs.customized_copy(data=flex.complex_double(fmod_c.tolist()))
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
                self.xray_structure.set_b_iso(values=flex.double(b_trial.tolist()))
                self.compute_f_calc()
                fmod_c = self._assemble_f_model(self.k_total, self.k_sol, self.b_sol)
                self.f_model = self.i_obs.customized_copy(data=flex.complex_double(fmod_c.tolist()))
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
                self.xray_structure.set_b_iso(values=flex.double(b_final.tolist()))
                self.gradients = best_grads
            else:
                self.xray_structure.set_b_iso(values=flex.double(b_curr.tolist()))
                self.compute_f_calc()
                fmod_c = self._assemble_f_model(self.k_total, self.k_sol, self.b_sol)
                self.f_model = self.i_obs.customized_copy(data=flex.complex_double(fmod_c.tolist()))
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
            self.xray_structure.set_b_iso(values=flex.double(b_trial.tolist()))
            self.compute_f_calc()
            fmod_c = self._assemble_f_model(self.k_total, self.k_sol, self.b_sol)
            self.f_model = self.i_obs.customized_copy(data=flex.complex_double(fmod_c.tolist()))
            t_trial, g_trial = self.compute_target_and_gradients()
            if t_trial < t_curr:
                best_t = t_trial
                best_b = b_trial
                best_grads = g_trial
                break
            alpha *= 0.5

        if best_t < t_curr:
            self.xray_structure.set_b_iso(values=flex.double(best_b.tolist()))
            self.target_value = best_t
            self.gradients = best_grads
        else:
            self.xray_structure.set_b_iso(values=flex.double(b_curr.tolist()))
            self.compute_f_calc()
            fmod_c = self._assemble_f_model(self.k_total, self.k_sol, self.b_sol)
            self.f_model = self.i_obs.customized_copy(data=flex.complex_double(fmod_c.tolist()))
            self.compute_target_and_gradients()

        if self.hierarchy is not None:
            self.hierarchy.adopt_xray_structure(self.xray_structure)

        return self.target_value

    def refine_b_iso(
        self,
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
                print(f"  Step {it+1:2d}/{max_iterations}: NLL = {val:.6f} (delta = {delta:+.6f}), B_iso mean = {b_vals.mean():.2f} Å²{adp_e_str} (min: {b_vals.min():.2f}, max: {b_vals.max():.2f})")
            if len(history) > 2 and abs(history[-1] - history[-2]) < 1e-7:
                if verbose:
                    print(f"  Converged after {it+1} steps (target change < 1e-7).")
                break
        return history

    def build_geometry_restraints(self) -> Any:
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
        self,
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
        self,
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
        self,
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
        self,
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
        self,
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
        self,
        ev: Any,
        damping_factor: float = 0.05,
        min_floor: float = 1e-6,
        w_xray: Optional[float] = None,
        h_geom: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute diagonal Hessian preconditioners for Cartesian sites and isotropic B-factors.

        Uses exact Gauss-Newton blocks from the structure factor engine:
          - For B-factors: H_B = H_{u_iso} / (8pi^2)^2, P_B = 1 / (H_B + lambda_B)
          - For Sites: H_cart = F^T H_frac F, P_sites = 1 / (diag(H_cart) + lambda_cart)
        When w_xray and h_geom are supplied, balances the combined objective:
          H_total = w_xray * H_cart + H_geom.
        """
        from phridge.sfcalc.client import _packed_xray
        from phridge.sfcalc.ops import _engine
        from phridge.models import SfEngineParams

        hkl = np.asarray(list(self.i_obs.indices()), dtype=np.int32)
        px, table = _packed_xray(self.xray_structure, None)
        params = SfEngineParams(d_min=self.d_min)
        eng = _engine(px, table, hkl, params)
        blocks = eng.gauss_newton_blocks(ev.curv_radial, ev.curv_tangential)

        # 1. B-factor curvature: H_B = H_u / (64 pi^4)
        # Note: B-factors have no geometry restraints, so H_B must NOT be scaled by w_xray.
        # This preserves the dimensional units of Å^4 and gives the natural Newton step: ΔB = - P_B * g_B.
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
        self,
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
        """Refine scale, solvent parameters, isotropic B-factors, and atomic coordinates with Adam.

        Uses:
          - CCTBX geometry restraints evaluated on the CCTBX side for chemical plausibility.
          - Bulk solvent model computed from CCTBX mask gridding.
          - Exact Gauss-Newton Hessian preconditioner on atomic sites and B values from the target.
          - Dynamic co-refinement of sigma_A across resolution bins and Student-t degrees of freedom nu.
        """
        import torch

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
        tgt_cls_desc = "MaximumLikelihoodAmplitude" if self.target_type == "amplitude" else "IntensityLogLikelihood"
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
            self.xray_structure.set_b_iso(values=flex.double(b_curr.tolist()))
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
            grads = self.sf.gradients(self.xray_structure, self.i_obs, flex.complex_double(dtdf_calc.tolist()), d_min=self.d_min)
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
                    rms_x = float(np.sqrt(np.mean(g_xray_cart**2)))
                    rms_g = float(np.sqrt(np.mean(g_geom**2)))
                    w_xray = xray_scale * (rms_g / max(rms_x, 1e-12))
                    if need_curv:
                        p_sites, p_B = self.compute_hessian_preconditioners(
                            ev, damping_factor=damping_factor, w_xray=w_xray, h_geom=rms_g
                        )
                elif w_geom is not None:
                    w_xray = 1.0 / max(float(w_geom), 1e-12)
                    if need_curv:
                        p_sites, p_B = self.compute_hessian_preconditioners(ev, damping_factor=damping_factor)
                else:
                    rms_x = float(np.sqrt(np.mean(g_xray_cart**2)))
                    rms_g = float(np.sqrt(np.mean(g_geom**2)))
                    w_xray = (rms_g / max(rms_x, 1e-12)) / max(geom_scale, 1e-12)
                    if need_curv:
                        p_sites, p_B = self.compute_hessian_preconditioners(ev, damping_factor=damping_factor)
            else:
                w_xray = 1.0
                if need_curv:
                    p_sites, p_B = self.compute_hessian_preconditioners(ev, damping_factor=damping_factor)

            # Intact geometry scale (w_geom = 1.0), X-ray reweighted by w_xray for coordinates
            g_sites_tot = w_xray * g_xray_cart + g_geom
            # B-factors have no geometry restraints, so their gradient is purely g_B
            g_B_tot = g_B

            # Step 6: Hessian preconditioners on sites and B values
            if use_preconditioner:
                p_g_sites = g_sites_tot * p_sites
                p_g_B = g_B_tot * p_B
            else:
                p_g_sites = g_sites_tot / max(w_xray, 1.0)
                p_g_B = g_B_tot

            if max_shift_angstrom is not None and max_shift_angstrom > 0:
                shift_norm = np.linalg.norm(p_g_sites, axis=1, keepdims=True)
                scale_clamp = np.minimum(1.0, max_shift_angstrom / np.maximum(shift_norm, 1e-12))
                p_g_sites = p_g_sites * scale_clamp

            # Step 7: Assign gradients to torch tensors and step
            if refine_sites:
                sites_cart_t.grad = torch.as_tensor(p_g_sites, dtype=self.float_dtype, device=self.torch_device)
            if refine_b:
                b_iso_t.grad = torch.as_tensor(p_g_B, dtype=self.float_dtype, device=self.torch_device)
            if refine_scales:
                k_tot_t.grad = d_scales[0]
                if self.use_bulk_solvent:
                    k_sol_t.grad = d_scales[1]
                    b_sol_t.grad = d_scales[2]

            opt.step()
            opt.zero_grad()

            with torch.no_grad():
                b_iso_t.clamp_(min=min_b, max=max_b)
                k_tot_t.clamp_(min=1e-5)
                if self.use_bulk_solvent:
                    k_sol_t.clamp_(min=0.0, max=1.0)
                    b_sol_t.clamp_(min=0.0, max=200.0)

            # Update model scale attributes
            self.k_total = float(k_tot_t.item())
            if self.use_bulk_solvent:
                self.k_sol = float(k_sol_t.item())
                self.b_sol = float(b_sol_t.item())

            # History recording
            b_mean = float(b_iso_t.detach().cpu().mean().item())
            history["nll"].append(self.target_value)
            history["geom_energy"].append(e_geom)
            history["b_mean"].append(b_mean)
            history["k_total"].append(self.k_total)
            if self.use_bulk_solvent:
                history["k_sol"].append(self.k_sol)
                history["b_sol"].append(self.b_sol)
            history["w_xray"].append(w_xray)

            if verbose:
                print(f"  Step {it+1:2d}/{max_iterations}: NLL = {self.target_value:.6f}, Geom E = {e_geom:.2f}, Mean B = {b_mean:.2f} Å², w_xray = {w_xray:.2e}, k = {self.k_total:.4e}")

        # Final pass: update structure factors, F_model, and hierarchy
        final_sites_frac = (sites_cart_t.detach().cpu().numpy() @ fract.T).astype(np.float64)
        self.xray_structure.set_sites_frac(flex.vec3_double([tuple(r) for r in final_sites_frac]))
        final_b = np.clip(b_iso_t.detach().cpu().numpy(), min_b, max_b).astype(np.float64)
        self.xray_structure.set_b_iso(values=flex.double(final_b.tolist()))
        self.compute_f_calc()
        fmod_complex = self._assemble_f_model(self.k_total, self.k_sol, self.b_sol)
        self.f_model = self.i_obs.customized_copy(data=flex.complex_double(fmod_complex.tolist()))

        if self.hierarchy is not None:
            self.hierarchy.adopt_xray_structure(self.xray_structure)

        if refine_sigma_a and refine_nu and self.target_type == "intensity":
            self.refine_sigma_a_and_nu(nu_mode=self.nu_mode)
        elif refine_sigma_a:
            self.estimate_sigma_a()
        elif refine_nu and self.target_type == "intensity" and (self.nu is not None or self.estimate_nu_flag):
            self.estimate_nu(mode=self.nu_mode)

        summ_final = self.summary()
        history["r_work"].append(summ_final["r_work"])
        history["r_free"].append(summ_final["r_free"])

        if verbose:
            print("-" * 65)
            print(f"  Final:   NLL = {self.target_value:.6f}, R_work = {summ_final['r_work']*100:.2f}%, R_free = {summ_final['r_free']*100:.2f}%, R_int = {summ_final['r_int']*100:.2f}%")
            if summ_final.get("bond_rms") is not None and np.isfinite(summ_final.get("bond_rms", float("nan"))):
                acc_tag = "[ACCEPTABLE]" if summ_final.get("geom_is_acceptable") else "[OUT-OF-TARGET]"
                print(f"           Stereochemistry {acc_tag}: Bonds = {summ_final['bond_rms']:.4f} Å, Angles = {summ_final['angle_rms']:.2f}°, Planar = {summ_final['planarity_rms']:.4f} Å, Chiral = {summ_final['chirality_rms']:.4f} Å³")
            print("=" * 65 + "\n")

        return history

    def refine_lbfgs(
        self,
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
        """Refine coordinates, isotropic B-factors, scales, and solvent using L-BFGS.

        Unlike coordinate-wise adaptive algorithms (e.g. Adam), L-BFGS uses a quasi-Newton
        Hessian approximation with Armijo-Wolfe line search along coupled search directions.
        This strictly prevents bond length distortions, maintains planarity, and respects
        stiff stereochemical potential energy surfaces.

        Parameters
        ----------
        macrocycles : int, default 3
            Number of macrocycles of coordinate and B-factor refinement.
        max_iterations_per_cycle : int, default 20
            Maximum L-BFGS iterations per coordinate refinement cycle.
        regularize_geometry : bool, default True
            Run an initial pure geometry L-BFGS regularization phase to eliminate severe
            starting clashes or stretched bonds before reciprocally coupled refinement.
        regularize_steps : int, default 15
            Number of pure geometry regularization steps.
        w_geom : float, optional
            Explicit geometry weight factor. If None, uses xray_weight_mode.
        xray_weight_mode : str, default "hessian"
            Weighting mode for X-ray vs geometry: "hessian", "gradient", or "fixed".
        xray_scale : float, default 1.0
            Multiplier on the Hessian/gradient ratio.
        refine_scales : bool, default True
            Refine overall scale and bulk solvent (k_sol, b_sol) via PyTorch L-BFGS.
        refine_b : bool, default True
            Refine isotropic atomic B-factors in each macrocycle.
        refine_sites : bool, default True
            Refine Cartesian coordinates via L-BFGS.
        refine_sigma_a : bool, default True
            Re-estimate sigma_A parameters across resolution shells each cycle.
        refine_nu : bool, default True
            Re-estimate Student-t degrees of freedom nu each cycle (intensity target).
        b_iterations : int, default 3
            Number of B-factor line-search steps per cycle.
        min_b, max_b : float, default 1.0, 200.0
            Bounds on refined isotropic B-factors.
        m_history : int, default 10
            L-BFGS correction vector memory size.
        polish_geometry : bool, default True
            Run a final pure geometry polish if residual strain remains after refinement.
        polish_steps : int, default 10
            Number of polish steps.
        verbose : bool, default True
            Print informative progress tables.
        """
        import scipy.optimize

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
            "nll": [val0],
            "geom_energy": [gst_init.get("target", 0.0)],
            "adp_energy": [],
            "r_work": [summ0["r_work"]],
            "r_free": [summ0["r_free"]],
            "b_mean": [float(np.mean([sc.u_iso * 8.0 * np.pi**2 for sc in self.xray_structure.scatterers()]))],
            "bond_rms": [gst_init.get("bond_rms", float("nan"))],
            "angle_rms": [gst_init.get("angle_rms", float("nan"))],
            "planarity_rms": [gst_init.get("planarity_rms", float("nan"))],
            "chirality_rms": [gst_init.get("chirality_rms", float("nan"))],
            "is_acceptable": [gst_init.get("is_acceptable", False)],
            "w_xray": [],
        }
        if self.adp_prior is not None:
            X_0 = np.asarray(self.xray_structure.sites_cart(), dtype=np.float64)
            b_vals_0 = np.array([float(sc.u_iso * 8.0 * np.pi**2) for sc in self.xray_structure.scatterers()])
            beta_0 = np.log(np.maximum(b_vals_0, 1e-4))
            history["adp_energy"].append(float(self.adp_prior.energy_vec(beta_0, sites=X_0).item()))

        # Phase 0: Pure geometry regularization if initial bonds are severely distorted
        has_restraints = self.restraints_manager is not None
        if regularize_geometry and has_restraints and (gst_init.get("bond_rms", 0.0) > 0.03 or not gst_init.get("is_acceptable", False)):
            if verbose:
                print(f"  [Phase 0] Pure Geometry Regularization (L-BFGS, {regularize_steps} steps)...")
            def fg_geom(x):
                s = x.reshape(-1, 3)
                e, g = self.compute_geometry_energy_and_gradients(s)
                return float(e), g.ravel()
            s0 = np.asarray(self.xray_structure.sites_cart(), dtype=np.float64).ravel()
            res_reg = scipy.optimize.minimize(fg_geom, s0, jac=True, method="L-BFGS-B", options={"maxiter": regularize_steps, "maxcor": m_history})
            s_reg = res_reg.x.reshape(-1, 3)
            new_frac = (s_reg @ fract.T).astype(np.float64)
            self.xray_structure.set_sites_frac(flex.vec3_double([tuple(r) for r in new_frac]))
            self.compute_f_calc()
            fmod_c = self._assemble_f_model(self.k_total, self.k_sol, self.b_sol)
            self.f_model = self.i_obs.customized_copy(data=flex.complex_double(fmod_c.tolist()))
            if refine_scales:
                self.refine_scale_and_solvent(recompute_mask=self.use_bulk_solvent)
            if refine_sigma_a:
                self.estimate_sigma_a()
            gst_post_reg = self.compute_geometry_statistics()
            if verbose:
                print(f"            Post-Regularization: Bonds = {gst_post_reg['bond_rms']:.4f} Å, Angles = {gst_post_reg['angle_rms']:.2f}°, Planar = {gst_post_reg['planarity_rms']:.4f} Å, E_geom = {gst_post_reg['target']:.0f}")

        # Initialize and optimize ADP restraints if requested
        if refine_b and use_adp_restraints:
            if self.adp_prior is None:
                self.build_adp_restraints()
            if optimize_adp_weights:
                self.optimize_adp_hyperparameters(verbose=verbose)

        # Compute X-ray weight w_xray
        if has_restraints:
            if w_geom is not None:
                w_xray = 1.0 / max(float(w_geom), 1e-12)
            elif xray_weight_mode == "hessian":
                h_geom = self.estimate_geometry_curvature(n_shakes=3, spread=0.03)
                self.compute_target_and_gradients(compute_curvature=True)
                h_xray = getattr(self, "h_xray_mean", 1.0)
                w_xray = xray_scale * (h_geom / max(h_xray, 1e-12))
            elif xray_weight_mode == "gradient":
                val_x, grads_x = self.compute_target_and_gradients()
                gx = np.asarray(grads_x.d_target_d_site_cart(), dtype=np.float64)
                _, gg = self.compute_geometry_energy_and_gradients()
                rms_gx = float(np.sqrt(np.mean(gx**2)))
                rms_gg = float(np.sqrt(np.mean(gg**2)))
                w_xray = xray_scale * (rms_gg / max(rms_gx, 1e-12))
            else:
                w_xray = xray_scale * 5000.0
        else:
            w_xray = 1.0

        if verbose:
            print(f"  Calculated w_xray = {w_xray:.2e}")

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
                    self.f_model = self.i_obs.customized_copy(data=flex.complex_double(fmod_c.tolist()))
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
                self.f_model = self.i_obs.customized_copy(data=flex.complex_double(fmod_c.tolist()))

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
            self.f_model = self.i_obs.customized_copy(data=flex.complex_double(fmod_c.tolist()))

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

    def compute_map_coefficients(
        self,
        weighted: bool = True,
        from_gradient: bool = True,
    ) -> Tuple[Any, Any, Any]:
        """Synthesize gradient difference map and sigmaA-weighted 2mFo-DFc and mFo-DFc maps.

        Parameters
        ----------
        weighted : bool, default True
            If True, weight the raw likelihood gradient by
            N_work * epsilon * Sigma_wilson * (1 - sigma_A^2).
            Centric reflections are scaled by -1.0 and acentric by -0.5 to place them
            on the identical physical Fourier difference scale.
            If False, returns raw gradient (-0.5 * dQ/dF_model).
        from_gradient : bool, default True
            If True, derives difference map (FOFCWT) and 2Fo-Fc map (2FOFCWT) directly
            from the likelihood gradient:
                F_diff = F_grad
                F_2fofc = D * F_model + 2 * F_grad
            This replaces the naive sqrt(max(Io, 0)) stand-in with the exact Bayesian
            conditional expectation under the likelihood target, properly handling
            negative/weak intensities and avoiding artificial noise baseline elevation.
            If False, uses the legacy/naive formula with sqrt(max(Io, 0)) and Rice-Woolfson FOM.

        Returns
        -------
        map_coeffs_grad : cctbx.miller.array
            Fourier coefficients for the likelihood gradient map (FGRAD).
        map_coeffs_2fofc : cctbx.miller.array
            Fourier coefficients for 2mFo-DFc map (2FOFCWT).
        map_coeffs_fofc : cctbx.miller.array
            Fourier coefficients for mFo-DFc map (FOFCWT).
        """
        if self.d_target_d_f_model is None or self.sigma_a_per_refl is None:
            self.compute_target_and_gradients()

        eps_np = np.asarray(self.i_obs.epsilons().data().as_double(), dtype=np.float64)
        sw_np = np.asarray(self.mean_i_per_refl, dtype=np.float64)
        sa_np = np.asarray(self.sigma_a_per_refl, dtype=np.float64)
        cen_np = np.asarray(self.i_obs.centric_flags().data(), dtype=bool)

        # 1. Gradient difference map:
        if weighted:
            n_work = float(self.i_obs.size() - self.r_free_flags.data().count(True))
            sigma_wilson = eps_np * sw_np
            var_weight = n_work * sigma_wilson * (1.0 - sa_np**2)
            # Centric reflections have 1 degree of freedom (variance = eps * Sigma_w / 2 in 1D),
            # while acentric have 2 (variance = eps * Sigma_w in 2D complex plane).
            # The likelihood gradient d(-ln L)/dF_c has an inherent factor of 2 for acentric
            # vs 1 for centric reflections. To put centric and acentric difference terms on the
            # identical physical Fourier scale: factor is -0.5 for acentric and -1.0 for centric.
            factor = np.where(cen_np, -1.0, -0.5)
            grad_f = factor * self.d_target_d_f_model * var_weight
        else:
            grad_f = -0.5 * self.d_target_d_f_model

        map_coeffs_grad = self.i_obs.customized_copy(
            data=flex.complex_double(grad_f.tolist()),
            sigmas=None,
            observation_type=None,
        )

        fmod_np = np.asarray(self.f_model.data(), dtype=np.complex128)

        # 2. Difference and 2Fo-Fc maps
        if from_gradient:
            # Exact Bayesian derivation directly from likelihood gradient
            f_fofc = grad_f
            f_2fofc = sa_np * fmod_np + 2.0 * grad_f
        else:
            # Legacy/naive synthesis using sqrt(max(Io, 0)) stand-in and Rice-Woolfson FOM
            phi_calc = np.angle(fmod_np)
            fmod_abs = np.abs(fmod_np)
            io_np = np.asarray(self.i_obs.data(), dtype=np.float64)
            fo_amp = np.sqrt(np.maximum(io_np, 0.0))
            denom = np.maximum(eps_np * sw_np, 1e-12)
            Ec = fmod_abs / np.sqrt(denom)
            Zo = io_np / denom
            m = _fom_rice_woolfson(Ec, sa_np, Zo, cen_np)
            D = sa_np
            phase_factor = np.exp(1j * phi_calc)
            f_2fofc = (2.0 * m * fo_amp - D * fmod_abs) * phase_factor
            f_fofc = (m * fo_amp - D * fmod_abs) * phase_factor

        map_coeffs_2fofc = self.i_obs.customized_copy(
            data=flex.complex_double(f_2fofc.tolist()),
            sigmas=None,
            observation_type=None,
        )
        map_coeffs_fofc = self.i_obs.customized_copy(
            data=flex.complex_double(f_fofc.tolist()),
            sigmas=None,
            observation_type=None,
        )

        return map_coeffs_grad, map_coeffs_2fofc, map_coeffs_fofc

    def compute_all_map_coefficients(
        self,
        newton_damping: float = 0.1,
        include_free: bool = True,
    ) -> Dict[str, Any]:
        """Compute all map coefficients and per-reflection posterior statistics.

        For intensity targets (ml_i), evaluates exact Bayesian posterior moments
        on quadrature nodes, yielding:
          - 'difference': exact Bayesian posterior difference map (mFo - DFc analogue)
          - 'model': exact Bayesian posterior 2mFo - DFc map
          - 'gradient': likelihood gradient map (d log L / d F_c^*)
          - 'newton': diagonal damped Newton step map (gradient / (curvature + mu))
          - 'gradient_weighted': Fisher-variance weighted gradient difference map
          - 'gradient_raw': unweighted gradient map (-0.5 * dQ/dF_model)
          - '2fofc': gradient-derived 2mFo - DFc map
          - 'fofc': gradient-derived mFo - DFc map
          - 'fom': figure of merit <E m(E)> / <E>
          - 'f_post': posterior mean amplitude <F>
          - 'robust_weight': Student-t scale mixture weight <lambda | Z_o>
          - 'curvature': radial curvature -d^2 log L / d|F_c|^2
          - 'stats': dictionary of summary statistics
        """
        if self.sigma_a_per_refl is None:
            self.estimate_sigma_a()
        if self.d_target_d_f_model is None:
            self.compute_target_and_gradients()

        grad_coeffs, coeffs_2fofc, coeffs_fofc = self.compute_map_coefficients(
            weighted=True, from_gradient=True
        )
        raw_grad_coeffs, _, _ = self.compute_map_coefficients(
            weighted=False, from_gradient=True
        )

        res: Dict[str, Any] = {
            "gradient_weighted": grad_coeffs,
            "gradient_raw": raw_grad_coeffs,
            "2fofc": coeffs_2fofc,
            "fofc": coeffs_fofc,
        }

        if self.target_type != "intensity":
            return res

        from phridge.contrib.intensity_ll.maps import (
            IntensityMapOptions,
            intensity_map_coefficients,
        )

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

        fmod_np = np.asarray(self.f_model.data(), dtype=np.complex128)
        fmod_t = torch.as_tensor(fmod_np, dtype=self.complex_dtype, device=self.torch_device)
        opts = IntensityMapOptions(
            newton_damping=newton_damping,
            include_free=include_free,
        )
        ms = intensity_map_coefficients(tgt, fmod_t, obs, opts)

        def _to_miller(c_data: np.ndarray) -> Any:
            return self.i_obs.customized_copy(
                data=flex.complex_double(c_data.tolist()),
                sigmas=None,
                observation_type=None,
            )

        def _to_real_miller(r_data: np.ndarray) -> Any:
            return self.i_obs.customized_copy(
                data=flex.double(r_data.tolist()),
                sigmas=None,
                observation_type=None,
            )

        res.update(
            difference=_to_miller(ms.difference),
            model=_to_miller(ms.model),
            gradient=_to_miller(ms.gradient),
            newton=_to_miller(ms.newton),
            fom=_to_real_miller(ms.fom),
            f_post=_to_real_miller(ms.f_post),
            robust_weight=_to_real_miller(ms.robust_weight),
            curvature=_to_real_miller(ms.curvature),
            stats=ms.stats,
        )
        if ms.d_loglik_d_nu is not None:
            res["d_loglik_d_nu"] = _to_real_miller(ms.d_loglik_d_nu)

        return res

    def write_maps(
        self,
        prefix: str = "output",
        resolution_factor: float = 0.25,
        write_ccp4: bool = True,
        write_mtz: bool = True,
        weighted_gradient: bool = True,
        from_gradient: bool = True,
        compute_all: bool = True,
    ) -> Dict[str, str]:
        """Export synthesized maps to MTZ and real-space CCP4 (.ccp4) formats."""
        grad_coeffs, coeffs_2fofc, coeffs_fofc = self.compute_map_coefficients(
            weighted=weighted_gradient,
            from_gradient=from_gradient,
        )
        raw_grad_coeffs, _, _ = self.compute_map_coefficients(
            weighted=False,
            from_gradient=from_gradient,
        )
        all_maps: Dict[str, Any] = {}
        if compute_all and self.target_type == "intensity":
            try:
                all_maps = self.compute_all_map_coefficients()
            except Exception as e:
                import warnings

                warnings.warn(f"Could not compute all posterior map coefficients: {e}")

        written_files = {}

        # Write MTZ
        if write_mtz:
            mtz_path = f"{prefix}_maps.mtz"
            mtz_dataset = coeffs_2fofc.as_mtz_dataset(column_root_label="2FOFCWT")
            mtz_dataset.add_miller_array(coeffs_fofc, column_root_label="FOFCWT")
            mtz_dataset.add_miller_array(grad_coeffs, column_root_label="FGRAD")
            mtz_dataset.add_miller_array(raw_grad_coeffs, column_root_label="FGRAD_RAW")
            if "newton" in all_maps:
                mtz_dataset.add_miller_array(all_maps["newton"], column_root_label="FNEWTON")
            if "difference" in all_maps:
                mtz_dataset.add_miller_array(all_maps["difference"], column_root_label="FDIFF_POST")
            if "model" in all_maps:
                mtz_dataset.add_miller_array(all_maps["model"], column_root_label="FMODEL_POST")
            if "fom" in all_maps:
                mtz_dataset.add_miller_array(all_maps["fom"], column_root_label="FOM")
            if "f_post" in all_maps:
                mtz_dataset.add_miller_array(all_maps["f_post"], column_root_label="F_POST")
            if "robust_weight" in all_maps and self.nu is not None:
                mtz_dataset.add_miller_array(all_maps["robust_weight"], column_root_label="ROBUST_WT")
            mtz_dataset.mtz_object().write(mtz_path)
            written_files["mtz"] = mtz_path

        # Write CCP4 maps
        if write_ccp4:
            map_targets = [
                ("gradient", grad_coeffs),
                ("gradient_raw", raw_grad_coeffs),
                ("2fofc", coeffs_2fofc),
                ("fofc", coeffs_fofc),
            ]
            if "newton" in all_maps:
                map_targets.append(("newton", all_maps["newton"]))
            if "difference" in all_maps:
                map_targets.append(("diff_post", all_maps["difference"]))
            if "model" in all_maps:
                map_targets.append(("model_post", all_maps["model"]))
            for name, coeffs in map_targets:
                ccp4_path = f"{prefix}_{name}.ccp4"
                fft_map = coeffs.fft_map(resolution_factor=resolution_factor)
                fft_map.apply_sigma_scaling()
                fft_map.as_ccp4_map(
                    file_name=ccp4_path,
                    labels=[f"phridge {name} map"],
                )
                written_files[name] = ccp4_path

        return written_files

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

        # Intensity R_int
        i_calc = fc_amp**2
        r_int_num = np.sum(np.abs(io_np - i_calc))
        r_int_den = np.sum(io_np)
        r_int = float(r_int_num / max(r_int_den, 1e-12))

        r_int_free = float(
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

        # Amplitude R_work and R_free
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

        # B-factor statistics
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
            "r_int": r_int,
            "r_int_free": r_int_free,
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
        print(f" R_intensity (work/free): {s['r_int']*100:.2f}% / {s['r_int_free']*100:.2f}%")
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
    # 1. Read MTZ first to determine crystal symmetry
    reader = any_reflection_file(str(mtz_path))
    miller_arrays = reader.as_miller_arrays()

    i_obs = None
    r_free = None

    for arr in miller_arrays:
        labels = [l.lower() for l in (arr.info().labels if arr.info() else [])]
        # Check for user-specified label
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

        # Check for free flags
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

    # Ensure observation type is set
    i_obs.set_observation_type_xray_intensity()

    # Merge anomalous if present
    if i_obs.anomalous_flag():
        i_obs = i_obs.as_non_anomalous_array().merge_equivalents().array()

    # Map to asymmetric unit
    i_obs = i_obs.map_to_asu()

    if r_free is not None:
        if r_free.anomalous_flag():
            r_free = r_free.as_non_anomalous_array().merge_equivalents().array()
        r_free = r_free.map_to_asu()

    mtz_cs = i_obs.crystal_symmetry()

    # 2. Read PDB with MTZ crystal symmetry fallback
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

    # Apply omit selection if specified
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
        print(f"Estimating sigma_A directly from intensities across {n_bins} resolution bins{sa_mode_msg}...")
        sa_bins = model.estimate_sigma_a(
            overlap=overlap_bins,
            tv_lambda=tv_norm,
            enforce_monotonic=enforce_monotonic,
            interpolate=interpolate_sigma_a,
        )
        for b_idx, sa_val in sorted(sa_bins.items()):
            d_range = model.binner.bin_d_range(b_idx)
            print(f"  Bin {b_idx} (d={d_range[0]:.2f}-{d_range[1]:.2f} Å): sigma_A = {sa_val:.4f}")

    tgt_desc = "intensity target (ml_i)" if model.target_type == "intensity" else "amplitude target (ml_f)"
    print(f"Evaluating {tgt_desc} and computing gradients...")
    target, grads = model.compute_target_and_gradients()
    print(f"  Target NLL: {target:.4f}")

    # Always write model PDB (especially useful if residues were omitted)
    model_pdb = f"{prefix}_model.pdb"
    model.write_pdb(model_pdb)
    files = {"model": model_pdb}

    if refine_mode == "adam" and max_iterations > 0:
        precond_lbl = "preconditioned " if use_preconditioner else ""
        print(f"Refining coordinates, B-factors, and scales with {precond_lbl}Adam ({max_iterations} steps)...")
        adam_history = model.refine_adam(
            max_iterations=max_iterations,
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
            use_preconditioner=use_preconditioner,
            preconditioner_interval=preconditioner_interval,
            sigma_a_interval=sigma_a_interval,
            min_b=min_b,
            max_b=max_b,
            damping_factor=damping_factor,
        )
        pdb_out = f"{prefix}_refined.pdb"
        model.write_pdb(pdb_out)
        files["refined"] = pdb_out
        print(f"  Wrote Adam-refined PDB: {pdb_out}")

    if refine_mode == "lbfgs" and (macrocycles > 0 or max_iterations > 0):
        eff_cycles = macrocycles if macrocycles > 0 else 3
        eff_iter = max_iterations if (max_iterations > 0 and max_iterations != 0) else lbfgs_max_iter
        print(f"Refining coordinates, B-factors, and scales with L-BFGS ({eff_cycles} cycles, {eff_iter} iter/cycle)...")
        lbfgs_history = model.refine_lbfgs(
            macrocycles=eff_cycles,
            max_iterations_per_cycle=eff_iter,
            regularize_geometry=regularize_geometry,
            w_geom=w_geom,
            xray_weight_mode=xray_weight_mode,
            xray_scale=xray_scale,
            refine_scales=refine_scales,
            refine_b=True,
            refine_sites=True,
            refine_sigma_a=refine_sigma_a,
            refine_nu=refine_nu,
            min_b=min_b,
            max_b=max_b,
            polish_geometry=polish_geometry,
        )
        pdb_out = f"{prefix}_refined.pdb"
        model.write_pdb(pdb_out)
        files["refined"] = pdb_out
        print(f"  Wrote L-BFGS refined PDB: {pdb_out}")

    do_refine_b = (refine_mode in ("b_iso", "both")) or (refine_b > 0)
    b_steps = max_iterations if refine_mode == "b_iso" else refine_b
    if do_refine_b and b_steps > 0:
        print(f"Refining isotropic B-factors ({b_steps} steps, sites fixed)...")
        b_history = model.refine_b_iso(
            max_iterations=b_steps,
            step_scale=b_step_scale,
            max_shift_b=max_shift_b,
            min_b=min_b,
            max_b=max_b,
        )
        b_out = f"{prefix}_refined.pdb" if refine_mode == "b_iso" else f"{prefix}_b_refined.pdb"
        model.write_pdb(b_out)
        files["b_refined"] = b_out
        if refine_mode == "b_iso":
            files["refined"] = b_out
        print(f"  Wrote B-refined PDB: {b_out}")

    do_refine_sites = (refine_mode in ("sites", "both")) and (max_iterations > 0) and (refine_mode != "b_iso")
    if do_refine_sites:
        print(f"Refining coordinates ({max_iterations} steps)...")
        history = model.refine_coordinates(max_iterations=max_iterations)
        pdb_out = f"{prefix}_refined.pdb"
        model.write_pdb(pdb_out)
        files["refined"] = pdb_out
        print(f"  Wrote refined PDB: {pdb_out}")

    print("Synthesizing and writing maps...")
    map_files = model.write_maps(prefix=prefix)
    files.update(map_files)
    for k, v in map_files.items():
        print(f"  Wrote {k} map: {v}")

    model.print_summary()

    if view:
        from phridge.client.viewer import open_static_viewer

        pdb_for_view = files.get("refined", files.get("model", str(pdb_path)))
        open_static_viewer(
            pdb_path=pdb_for_view,
            gradient_map=files.get("gradient"),
            map_2fofc=files.get("2fofc"),
            map_fofc=files.get("fofc"),
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
