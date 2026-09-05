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
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.optimize
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
    import scipy.special
    two_X = 2.0 * X
    # Ratio using exponentially scaled Bessel functions i1e(y)/i0e(y)
    acen_m = scipy.special.i1e(two_X) / np.maximum(scipy.special.i0e(two_X), 1e-300)

    fom = np.where(centric, cen_m, acen_m)
    return np.clip(fom, 0.0, 1.0)


def tv_denoise_1d(
    y: np.ndarray,
    weights: Optional[np.ndarray] = None,
    lam: float = 0.05,
    lower: float = 0.01,
    upper: float = 0.999,
) -> np.ndarray:
    """1D Total Variation Denoising (TVD) using Huber-smoothed L1 difference penalty.

    Solves:
        min_{x in [lower, upper]^n} 0.5 * sum_i w_i (x_i - y_i)^2 + lam * sum_i |x_{i+1} - x_i|
    """
    if lam <= 0.0:
        return np.clip(y, lower, upper)
    n = len(y)
    if n <= 1:
        return np.clip(y, lower, upper)
    if weights is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        mean_w = np.mean(w)
        w = w / mean_w if mean_w > 0 else np.ones(n, dtype=np.float64)

    eps = 1e-4

    def obj_and_grad(x: np.ndarray) -> Tuple[float, np.ndarray]:
        res = x - y
        f_data = 0.5 * float(np.sum(w * res**2))
        g_data = w * res

        diffs = np.diff(x)
        huber = np.sqrt(diffs**2 + eps**2)
        f_tv = float(lam * np.sum(huber))

        g_tv = np.zeros(n, dtype=np.float64)
        dh = diffs / huber
        g_tv[:-1] -= lam * dh
        g_tv[1:] += lam * dh
        return f_data + f_tv, g_data + g_tv

    bounds = [(lower, upper)] * n
    res = scipy.optimize.minimize(
        obj_and_grad,
        np.clip(y, lower, upper),
        jac=True,
        bounds=bounds,
        method="L-BFGS-B",
    )
    return np.asarray(res.x, dtype=np.float64)


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
        nu_mode: str = "binned",
        target_type: str = "intensity",
        device: str = "cpu",
    ) -> None:
        register()
        self.device = device
        self.sf = sf if sf is not None else StructureFactorServer(memory=True, device=device)
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
        self.nu: Optional[float] = float(nu) if nu is not None else None
        self.estimate_nu_flag: bool = bool(estimate_nu)
        self.nu_mode: str = str(nu_mode)
        self.nu_binned: Dict[int, float] = {}
        self.nu_per_refl: Optional[flex.double] = None

        # Align crystal symmetry
        cs = i_obs.crystal_symmetry()
        self.xray_structure = xray_structure.customized_copy(crystal_symmetry=cs)
        self.hierarchy = hierarchy

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

    def refine_scale_and_solvent(self, max_iter: int = 50) -> Dict[str, float]:
        """Directly minimize target NLL wrt scale and bulk solvent parameters."""
        if self.f_calc is None:
            self.compute_f_calc()
        if self.use_bulk_solvent and self.f_mask is None:
            self.compute_mask()

        # Initial analytical overall scale
        fc_abs = np.abs(np.asarray(self.f_calc.data()))
        num = np.sum(np.asarray(self.i_obs.data()) * fc_abs**2)
        den = np.sum(fc_abs**4)
        k0 = float(np.sqrt(max(num / max(den, 1e-12), 1e-6)))

        work_sel = torch.as_tensor(~np.asarray(self.r_free_flags.data(), dtype=bool))
        eps_t = torch.as_tensor(np.asarray(self.i_obs.epsilons().data().as_double(), dtype=np.float64))
        sw_t = torch.as_tensor(np.asarray(self.mean_i_per_refl, dtype=np.float64))
        cen_t = torch.as_tensor(np.asarray(self.i_obs.centric_flags().data(), dtype=bool))

        if self.target_type == "amplitude":
            f_obs = self.get_f_obs()
            fo_t = torch.as_tensor(np.asarray(f_obs.data(), dtype=np.float64))
            si_t = torch.as_tensor(np.asarray(f_obs.sigmas(), dtype=np.float64))
            sa_init = torch.full_like(fo_t, 0.8)
            beta_init = sw_t * (1.0 - sa_init**2)
            obs_f = Observations.from_numpy(
                data=fo_t,
                sigmas=si_t,
                epsilon=eps_t,
                centric=cen_t,
                alpha=sa_init,
                beta=beta_init,
                r_free=np.asarray(self.r_free_flags.data(), dtype=bool),
            )
            tgt_f = MaximumLikelihoodAmplitude(scale_factor=1.0)

            if not self.use_bulk_solvent:
                fc_t = torch.as_tensor(fc_abs, dtype=torch.float64)

                def nll_k_f(k_val: float) -> float:
                    fmod = float(k_val) * fc_t
                    with torch.no_grad():
                        t = tgt_f.per_reflection(fmod, obs_f)
                        return float(t[work_sel].mean().item())

                res = scipy.optimize.minimize_scalar(
                    nll_k_f,
                    bounds=(max(k0 * 0.1, 1e-4), k0 * 10.0),
                    method="bounded",
                    options={"maxiter": max_iter},
                )
                self.k_total = float(res.x)
            else:
                fc_np = np.asarray(self.f_calc.data(), dtype=np.complex128)
                fm_np = np.asarray(self.f_mask.data(), dtype=np.complex128)
                s2 = np.asarray(self.i_obs.d_star_sq().data(), dtype=np.float64)

                def nll_solvent_f(params: np.ndarray) -> float:
                    k, ksol, bsol = float(params[0]), float(params[1]), float(params[2])
                    f_sol = ksol * np.exp(-bsol * s2 / 4.0) * fm_np
                    fmod_t = torch.as_tensor(k * (fc_np + f_sol))
                    with torch.no_grad():
                        t = tgt_f.per_reflection(fmod_t, obs_f)
                        return float(t[work_sel].mean().item())

                init_p = [k0, 0.35, 45.0]
                bounds = [(k0 * 0.1, k0 * 10.0), (0.0, 1.0), (0.0, 150.0)]
                res = scipy.optimize.minimize(nll_solvent_f, init_p, bounds=bounds, method="L-BFGS-B", options={"maxiter": max_iter})
                self.k_total = float(res.x[0])
                self.k_sol = float(res.x[1])
                self.b_sol = float(res.x[2])
        else:
            # Prepare normalized observation tensors for ml_i
            io_t = torch.as_tensor(np.asarray(self.i_obs.data(), dtype=np.float64))
            si_t = torch.as_tensor(np.asarray(self.i_obs.sigmas(), dtype=np.float64))
            sa_init = torch.full_like(io_t, 0.8)

            if not self.use_bulk_solvent:
                fc_t = torch.as_tensor(fc_abs, dtype=torch.float64)

                def nll_k(k_val: float) -> float:
                    fmod = float(k_val) * fc_t
                    Ec, _, Zo, sZ = normalize(fmod, io_t, si_t, eps_t, sw_t, sa_init)
                    with torch.no_grad():
                        ll = log_likelihood_normal(Ec[work_sel], sa_init[work_sel], Zo[work_sel], sZ[work_sel], cen_t[work_sel])
                        return -float(ll.sum().item())

                res = scipy.optimize.minimize_scalar(
                    nll_k,
                    bounds=(max(k0 * 0.1, 1e-4), k0 * 10.0),
                    method="bounded",
                    options={"maxiter": max_iter},
                )
                self.k_total = float(res.x)
            else:
                fc_np = np.asarray(self.f_calc.data(), dtype=np.complex128)
                fm_np = np.asarray(self.f_mask.data(), dtype=np.complex128)
                s2 = np.asarray(self.i_obs.d_star_sq().data(), dtype=np.float64)

                def nll_solvent(params: np.ndarray) -> float:
                    k, ksol, bsol = float(params[0]), float(params[1]), float(params[2])
                    f_sol = ksol * np.exp(-bsol * s2 / 4.0) * fm_np
                    fmod_np = np.abs(k * (fc_np + f_sol))
                    fmod_t = torch.as_tensor(fmod_np, dtype=torch.float64)
                    Ec, _, Zo, sZ = normalize(fmod_t, io_t, si_t, eps_t, sw_t, sa_init)
                    with torch.no_grad():
                        ll = log_likelihood_normal(Ec[work_sel], sa_init[work_sel], Zo[work_sel], sZ[work_sel], cen_t[work_sel])
                        return -float(ll.sum().item())

                init_p = [k0, 0.35, 45.0]
                bounds = [(k0 * 0.1, k0 * 10.0), (0.0, 1.0), (0.0, 150.0)]
                res = scipy.optimize.minimize(nll_solvent, init_p, bounds=bounds, method="L-BFGS-B", options={"maxiter": max_iter})
                self.k_total = float(res.x[0])
                self.k_sol = float(res.x[1])
                self.b_sol = float(res.x[2])

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

        io_t = torch.as_tensor(np.asarray(self.i_obs.data(), dtype=np.float64))
        si_t = torch.as_tensor(np.asarray(self.i_obs.sigmas(), dtype=np.float64))
        eps_t = torch.as_tensor(np.asarray(self.i_obs.epsilons().data().as_double(), dtype=np.float64))
        sw_t = torch.as_tensor(np.asarray(self.mean_i_per_refl, dtype=np.float64))
        cen_t = torch.as_tensor(np.asarray(self.i_obs.centric_flags().data(), dtype=bool))
        fc_abs_t = torch.as_tensor(np.abs(np.asarray(self.f_model.data())), dtype=torch.float64)

        # Base normalization (sigmaA slot arbitrary here since we pass it inside the loop)
        dummy_sa = torch.full_like(fc_abs_t, 0.5)
        Ec, _, Zo, sZ = normalize(fc_abs_t, io_t, si_t, eps_t, sw_t, dummy_sa)

        free_flags = np.asarray(self.r_free_flags.data(), dtype=bool)
        n_free = int(np.sum(free_flags))
        # Use free set if sufficient reflections exist, else work set
        use_free = n_free >= 20

        bins = list(self.binner.range_used())
        n_bins = len(bins)

        raw_sa_list: List[float] = []
        weights_list: List[float] = []

        if self.target_type == "amplitude":
            f_obs = self.get_f_obs()
            fo_arr = np.asarray(f_obs.data(), dtype=np.float64)
            si_arr = np.asarray(f_obs.sigmas(), dtype=np.float64)
            cen_arr = np.asarray(self.i_obs.centric_flags().data(), dtype=bool)
            eps_arr = np.asarray(self.i_obs.epsilons().data().as_double(), dtype=np.float64)
            sw_arr = np.asarray(self.mean_i_per_refl, dtype=np.float64)
            fmod_t = torch.as_tensor(np.asarray(self.f_model.data(), dtype=np.complex128))
            tgt_f = MaximumLikelihoodAmplitude(scale_factor=1.0)

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
                if np.sum(fit_sel) < 5:
                    fit_sel = bin_sel

                weights_list.append(float(np.sum(fit_sel)))

                def bin_nll_f(sa_val: float) -> float:
                    sa_t = torch.full((int(np.sum(fit_sel)),), float(sa_val), dtype=torch.float64)
                    beta_t = torch.as_tensor(sw_arr[fit_sel] * (1.0 - sa_val**2), dtype=torch.float64)
                    obs_b = Observations.from_numpy(
                        data=fo_arr[fit_sel],
                        sigmas=si_arr[fit_sel],
                        epsilon=eps_arr[fit_sel],
                        centric=cen_arr[fit_sel],
                        alpha=sa_t,
                        beta=beta_t,
                    )
                    with torch.no_grad():
                        t = tgt_f.per_reflection(fmod_t[fit_sel], obs_b)
                        return float(t.sum().item())

                res = scipy.optimize.minimize_scalar(bin_nll_f, bounds=(0.01, 0.999), method="bounded")
                raw_sa_list.append(float(res.x))
        else:
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
                if np.sum(fit_sel) < 5:
                    fit_sel = bin_sel  # Fallback to work set if bin has too few free reflections

                weights_list.append(float(np.sum(fit_sel)))
                ec_b = Ec[torch.as_tensor(fit_sel)]
                zo_b = Zo[torch.as_tensor(fit_sel)]
                sz_b = sZ[torch.as_tensor(fit_sel)]
                cen_b = cen_t[torch.as_tensor(fit_sel)]

                nu_curr = self.nu_binned.get(i_bin, self.nu)

                def bin_nll(sa_val: float) -> float:
                    sa_t = torch.full_like(ec_b, float(sa_val))
                    with torch.no_grad():
                        if nu_curr is not None:
                            ll = log_likelihood_t(ec_b, sa_t, zo_b, sz_b, cen_b, nu=float(nu_curr), n_u=10)
                        else:
                            ll = log_likelihood_normal(ec_b, sa_t, zo_b, sz_b, cen_b)
                        return -float(ll.sum().item())

                res = scipy.optimize.minimize_scalar(bin_nll, bounds=(0.01, 0.999), method="bounded")
                raw_sa_list.append(float(res.x))

        sa_arr = np.array(raw_sa_list, dtype=np.float64)
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
        mode: str = "binned",
        overlap: Optional[int] = None,
        tv_lambda: Optional[float] = None,
        bounds: Tuple[float, float] = (2.05, 50.0),
        n_u: int = 10,
    ) -> Any:
        """Estimate or refine Student-t degrees of freedom 'nu' by directly minimizing intensity NLL.

        Parameters
        ----------
        mode : str, default "binned"
            'binned' estimates a resolution-dependent nu per resolution shell.
            'global' estimates a single overall scalar nu for all reflections.
        overlap : int, optional
            Adjacent resolution bins to pool for sliding-window estimation (defaults to self.overlap_bins).
        tv_lambda : float, optional
            Total Variation regularization weight across resolution bins (defaults to self.tv_norm).
        bounds : tuple of float, default (2.05, 50.0)
            Search bounds for nu. Must be > 2 for finite variance.
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

        overlap_val = self.overlap_bins if overlap is None else int(overlap)
        tv_lam_val = self.tv_norm if tv_lambda is None else float(tv_lambda)

        io_t = torch.as_tensor(np.asarray(self.i_obs.data(), dtype=np.float64))
        si_t = torch.as_tensor(np.asarray(self.i_obs.sigmas(), dtype=np.float64))
        eps_t = torch.as_tensor(np.asarray(self.i_obs.epsilons().data().as_double(), dtype=np.float64))
        sw_t = torch.as_tensor(np.asarray(self.mean_i_per_refl, dtype=np.float64))
        cen_t = torch.as_tensor(np.asarray(self.i_obs.centric_flags().data(), dtype=bool))
        fc_abs_t = torch.as_tensor(np.abs(np.asarray(self.f_model.data())), dtype=torch.float64)
        sa_t = torch.as_tensor(np.asarray(self.sigma_a_per_refl, dtype=np.float64))

        Ec, _, Zo, sZ = normalize(fc_abs_t, io_t, si_t, eps_t, sw_t, sa_t)

        free_flags = np.asarray(self.r_free_flags.data(), dtype=bool)
        n_free = int(np.sum(free_flags))
        use_free = n_free >= 20

        bins = list(self.binner.range_used())
        n_bins = len(bins)

        if mode == "global":
            fit_sel = free_flags if use_free else np.ones(self.i_obs.size(), dtype=bool)
            ec_fit = Ec[torch.as_tensor(fit_sel)]
            sa_fit = sa_t[torch.as_tensor(fit_sel)]
            zo_fit = Zo[torch.as_tensor(fit_sel)]
            sz_fit = sZ[torch.as_tensor(fit_sel)]
            cen_fit = cen_t[torch.as_tensor(fit_sel)]

            def global_nll(nu_val: float) -> float:
                with torch.no_grad():
                    ll = log_likelihood_t(ec_fit, sa_fit, zo_fit, sz_fit, cen_fit, nu=float(nu_val), n_u=n_u)
                    return -float(ll.sum().item())

            res = scipy.optimize.minimize_scalar(global_nll, bounds=bounds, method="bounded")
            self.nu = float(res.x)
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
            ec_b = Ec[torch.as_tensor(fit_sel)]
            sa_b = sa_t[torch.as_tensor(fit_sel)]
            zo_b = Zo[torch.as_tensor(fit_sel)]
            sz_b = sZ[torch.as_tensor(fit_sel)]
            cen_b = cen_t[torch.as_tensor(fit_sel)]

            def bin_nu_nll(nu_val: float) -> float:
                with torch.no_grad():
                    ll = log_likelihood_t(ec_b, sa_b, zo_b, sz_b, cen_b, nu=float(nu_val), n_u=n_u)
                    return -float(ll.sum().item())

            res = scipy.optimize.minimize_scalar(bin_nu_nll, bounds=bounds, method="bounded")
            raw_nu_list.append(float(res.x))

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
        nu_mode: str = "binned",
        overlap: Optional[int] = None,
        tv_lambda: Optional[float] = None,
        enforce_monotonic: Optional[bool] = None,
        interpolate: Optional[bool] = None,
        bounds: Tuple[float, float] = (2.05, 50.0),
    ) -> Tuple[Dict[int, float], Any]:
        """Iteratively co-refine sigma_A and Student-t nu."""
        for _ in range(max_cycles):
            self.estimate_sigma_a(
                overlap=overlap,
                tv_lambda=tv_lambda,
                enforce_monotonic=enforce_monotonic,
                interpolate=interpolate,
            )
            self.estimate_nu(
                mode=nu_mode,
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
        return self.sigma_a_binned, (self.nu_binned if nu_mode == "binned" else self.nu)

    def compute_target_and_gradients(self) -> Tuple[float, Any]:
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
            beta_cctbx = sw_np * (1.0 - sa_np**2)
            obs = Observations.from_numpy(
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
            fmod_t = torch.as_tensor(fmod_np)
            ev = tgt_f.evaluate(fmod_t, obs, compute_curvature=False)
            self.target_value = float(ev.value)
            self.target_value_test = float(ev.value_test) if ev.value_test is not None else float("nan")
            self.d_target_d_f_model = ev.d_target_d_f_calc
        else:
            obs = Observations.from_numpy(
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
            fmod_t = torch.as_tensor(fmod_np)

            tgt_kwargs: Dict[str, Any] = {}
            if self.nu is not None:
                tgt_kwargs["nu"] = float(self.nu)
            tgt = IntensityLogLikelihood(**tgt_kwargs)
            ev = tgt.evaluate(fmod_t, obs, compute_curvature=False)
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
        history = [self.target_value]
        for it in range(max_iterations):
            val = self.update_coordinates_step(step_scale=step_scale)
            history.append(val)
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

    def write_maps(
        self,
        prefix: str = "output",
        resolution_factor: float = 0.25,
        write_ccp4: bool = True,
        write_mtz: bool = True,
        weighted_gradient: bool = True,
        from_gradient: bool = True,
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
        written_files = {}

        # Write MTZ
        if write_mtz:
            mtz_path = f"{prefix}_maps.mtz"
            mtz_dataset = coeffs_2fofc.as_mtz_dataset(column_root_label="2FOFCWT")
            mtz_dataset.add_miller_array(coeffs_fofc, column_root_label="FOFCWT")
            mtz_dataset.add_miller_array(grad_coeffs, column_root_label="FGRAD")
            mtz_dataset.add_miller_array(raw_grad_coeffs, column_root_label="FGRAD_RAW")
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

    def write_pdb(self, file_name: str = "output_refined.pdb") -> str:
        """Write the current atomic coordinates to a PDB file."""
        if self.hierarchy is not None:
            self.hierarchy.adopt_xray_structure(self.xray_structure)
            self.hierarchy.write_pdb_file(file_name=file_name)
        else:
            with open(file_name, "w") as f:
                f.write(self.xray_structure.as_pdb_file())
        return file_name

    def summary(self) -> Dict[str, Any]:
        """Compute and return refinement statistics and diagnostic summary."""
        io_np = np.asarray(self.i_obs.data(), dtype=np.float64)
        si_np = np.asarray(self.i_obs.sigmas(), dtype=np.float64)
        fc_amp = np.abs(np.asarray(self.f_model.data())) if self.f_model is not None else np.zeros_like(io_np)

        free_flags = np.asarray(self.r_free_flags.data(), dtype=bool)
        work_flags = ~free_flags

        # Intensity R_int
        r_int_num = np.sum(np.abs(io_np - fc_amp**2))
        r_int_den = np.sum(io_np)
        r_int = float(r_int_num / max(r_int_den, 1e-12))

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
            "grad_cart_rms": grad_rms,
            "grad_cart_max": grad_max,
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
        print(f" R_intensity:        {s['r_int']*100:.2f}%")
        if not np.isnan(s['grad_cart_rms']):
            print(f" Site Gradient RMS:  {s['grad_cart_rms']:.4e} (max: {s['grad_cart_max']:.4e})")
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
    use_bulk_solvent: bool = False,
    n_bins: int = 10,
    d_min: Optional[float] = None,
    overlap_bins: int = 1,
    tv_norm: float = 0.0,
    enforce_monotonic: bool = False,
    interpolate_sigma_a: bool = True,
    nu: Optional[float] = None,
    estimate_nu: bool = False,
    nu_mode: str = "binned",
    target: str = "intensity",
    device: str = "cpu",
    omit: Optional[str] = None,
    omit_mode: str = "delete",
) -> IntensityModel:
    """Instantiate an IntensityModel from input PDB and MTZ files."""
    # 1. Read PDB
    pdb_inp = iotbx.pdb.input(file_name=str(pdb_path))
    xray_structure = pdb_inp.xray_structure_simple()
    hierarchy = pdb_inp.construct_hierarchy()

    # Apply omit selection if specified
    if omit:
        hierarchy, xray_structure, n_omitted = apply_omit(
            hierarchy=hierarchy,
            xray_structure=xray_structure,
            omit_spec=omit,
            omit_mode=omit_mode,
        )
        print(f"Applied omit selection '{omit}' ({omit_mode} mode): {n_omitted} atoms omitted.")

    # 2. Read MTZ
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
        device=device,
    )


def run_intensity_pipeline(
    pdb_path: str | Path,
    mtz_path: str | Path,
    prefix: str = "output",
    intensity_label: Optional[str] = None,
    use_bulk_solvent: bool = False,
    n_bins: int = 10,
    d_min: Optional[float] = None,
    overlap_bins: int = 1,
    tv_norm: float = 0.0,
    enforce_monotonic: bool = False,
    interpolate_sigma_a: bool = True,
    nu: Optional[float] = None,
    estimate_nu: bool = False,
    nu_mode: str = "binned",
    target: str = "intensity",
    max_iterations: int = 0,
    device: str = "cpu",
    view: bool = False,
    omit: Optional[str] = None,
    omit_mode: str = "delete",
) -> Dict[str, Any]:
    """Execute the full intensity refinement and map generation workflow."""
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
    )

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
        )
        for b_idx, sa_val in sorted(sa_bins.items()):
            d_range = model.binner.bin_d_range(b_idx)
            nu_str = f", nu = {model.nu_binned[b_idx]:.2f}" if b_idx in model.nu_binned else ""
            print(f"  Bin {b_idx} (d={d_range[0]:.2f}-{d_range[1]:.2f} Å): sigma_A = {sa_val:.4f}{nu_str}")
        if nu_mode == "global" and model.nu is not None:
            print(f"  Global fitted nu = {model.nu:.2f}")
    else:
        print(f"Estimating sigma_A directly from intensities across {n_bins} resolution bins{sa_mode_msg}...")
        sa_bins = model.estimate_sigma_a()
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

    if max_iterations > 0:
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
    parser.add_argument("--bulk-solvent", action="store_true", help="Enable cctbx map-gridded bulk solvent modeling")
    parser.add_argument("--n-bins", type=int, default=10, help="Number of resolution bins (default: 10)")
    parser.add_argument("--d-min", type=float, default=None, help="High resolution limit in Angstroms")
    parser.add_argument("--overlap-bins", type=int, default=1, help="Number of adjacent resolution bins to include on each side for overlapping sigma_A estimation (default: 1; set to 0 for disjoint bins)")
    parser.add_argument("--tv-norm", type=float, default=0.0, help="Total Variation regularization weight lambda_TV across resolution bins (default: 0.0; e.g. 0.05 or 0.1)")
    parser.add_argument("--enforce-monotonic", action="store_true", default=False, help="Enforce monotonic non-increasing sigma_A across resolution via isotonic regression")
    parser.add_argument("--no-interpolate-sigma-a", action="store_true", default=False, help="Disable continuous resolution interpolation of sigma_A across reflections")
    parser.add_argument("--nu", type=float, default=None, help="Fixed degrees of freedom for Student-t noise (e.g. 4.0; None uses Normal noise unless --estimate-nu is set)")
    parser.add_argument("--estimate-nu", action="store_true", default=False, help="Estimate / refine Student-t degrees of freedom nu directly from intensities")
    parser.add_argument("--nu-mode", default="binned", choices=["binned", "global"], help="Nu estimation mode: 'binned' (per resolution shell) or 'global' (single scalar, default: binned)")
    parser.add_argument("--target", default="intensity", choices=["intensity", "amplitude", "ml_i", "ml_f"], help="Refinement target: 'intensity' (ml_i) or 'amplitude' (ml_f) (default: intensity)")
    parser.add_argument("--refine", type=int, default=0, dest="max_iterations", help="Number of coordinate refinement steps")
    parser.add_argument("--omit", default=None, help="Residues or atom selection to omit (e.g. '45-52', '45:52', or 'resseq 45:52')")
    parser.add_argument("--omit-mode", default="delete", choices=["delete", "zero_occ"], help="Omit treatment: 'delete' (remove atoms) or 'zero_occ' (set occupancy to 0.0)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Compute device for phridge")
    parser.add_argument("--view", action="store_true", help="Launch browser-based 3D viewer (Mol*) to inspect maps and model")

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
            omit=opts.omit,
            omit_mode=opts.omit_mode,
            device=opts.device,
            view=opts.view,
        )
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
