"""Hierarchical, scale-invariant ADP prior / restraint term for phridge.

Math and parameterisation (see docs/geometry_curvature.md and Prompt):
1. Log-space parameterisation:
   - Isotropic atoms carry beta_i = log(B_i) where B_i = 8 pi^2 u_iso,i.
   - Anisotropic atoms carry symmetric 3x3 S_i with U_i = exp(S_i) / (8 pi^2).
     6 independent Voigt entries (00, 11, 22, 01, 02, 12).
   - Positivity is strictly guaranteed: B_i > 0 and U_i is strictly positive definite.
2. Penalise ratios, not differences:
   All similarity terms act on (beta_i - beta_j) or (S_i - S_j).
3. Heavy-tailed Student-t similarity:
   psi_nu(r) = 0.5 * (nu + 1) * log(1 + r^2 / nu); Gaussian limit psi_inf(r) = 0.5 * r^2.
   IRLS form: rho = delta / tau with frozen weight W = 0.5 * w * omega(rho_0),
   omega(rho) = (nu + 1) / (nu + rho^2).
4. Level anchor:
   lambda * (bar_beta - log(B_Wilson))^2.
5. Hirshfeld rigid bond on anisotropic bonded pairs:
   w_h * (b_hat^T (U_i - U_j) b_hat)^2.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch

from phridge.packing_geometry import PackedRestraints

Tensor = torch.Tensor

EIGHT_PI_SQ = 8.0 * (math.pi**2)
NORM_WEIGHTS_6 = [1.0, 1.0, 1.0, math.sqrt(2.0), math.sqrt(2.0), math.sqrt(2.0)]


def vec6_to_sym_mat(v: Tensor) -> Tensor:
    """Map (..., 6) Voigt vector (00, 11, 22, 01, 02, 12) to (..., 3, 3) symmetric matrix."""
    s00 = v[..., 0]
    s11 = v[..., 1]
    s22 = v[..., 2]
    s01 = v[..., 3]
    s02 = v[..., 4]
    s12 = v[..., 5]
    row0 = torch.stack([s00, s01, s02], dim=-1)
    row1 = torch.stack([s01, s11, s12], dim=-1)
    row2 = torch.stack([s02, s12, s22], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def sym_mat_to_vec6(M: Tensor) -> Tensor:
    """Map (..., 3, 3) symmetric matrix to (..., 6) Voigt vector (00, 11, 22, 01, 02, 12)."""
    return torch.stack(
        [
            M[..., 0, 0],
            M[..., 1, 1],
            M[..., 2, 2],
            M[..., 0, 1],
            M[..., 0, 2],
            M[..., 1, 2],
        ],
        dim=-1,
    )


def psi_nu(r: Tensor, nu: float) -> Tensor:
    """Student-t negative log-density kernel; Gaussian limit for nu >= 1e6 or inf."""
    if math.isinf(nu) or nu >= 1e6:
        return 0.5 * r * r
    return 0.5 * (nu + 1.0) * torch.log1p((r * r) / nu)


def omega_nu(r: Tensor, nu: float) -> Tensor:
    """IRLS weight (nu + 1) / (nu + r^2); 1.0 for Gaussian limit."""
    if math.isinf(nu) or nu >= 1e6:
        return torch.ones_like(r)
    return (nu + 1.0) / (nu + r * r)


@dataclass
class ADPPriorTables:
    pair_i: Tensor  # (P, 2) int64
    pair_w: Tensor  # (P,) float64
    pair_tau: Tensor  # (P,) float64
    pair_class: Tensor  # (P,) int64 (0=1-2, 1=1-3, 2=sphere)
    rigid_i: Tensor  # (R, 2) int64
    rigid_w: Tensor  # (R,) float64
    is_aniso: Tensor  # (N,) bool
    wilson_b: Optional[float]
    nu: float
    level_weight: float
    w_iso: float
    tau_iso: float
    n_sites: int
    param_dims: Tensor  # (N,) int64 (1 for iso, 6 for aniso)
    param_offsets: Tensor  # (N+1,) int64
    n_params: int
    b_hat_rigid: Optional[Tensor] = None  # (R, 3) float64 cached unit bond vectors

    @classmethod
    def from_packed(
        cls,
        packed: PackedRestraints,
        *,
        sites: Optional[Any] = None,
        device: str = "cpu",
        dtype: Any = torch.float64,
        is_aniso: Optional[Any] = None,
        wilson_b: Optional[float] = None,
        nu: Optional[float] = None,
        level_weight: Optional[float] = None,
        w_iso: float = 0.05,
        tau_iso: float = 0.5,
    ) -> "ADPPriorTables":
        def t_int(a: np.ndarray) -> Tensor:
            return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.long, device=device)

        def t_f(a: np.ndarray) -> Tensor:
            return torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=device)

        n_sites = int(packed.n_sites)

        # Anisotropic flags
        if is_aniso is not None:
            aniso_arr = np.asarray(is_aniso, dtype=bool).reshape(-1)
        elif packed.adp_is_aniso is not None and packed.adp_is_aniso.size == n_sites:
            aniso_arr = np.asarray(packed.adp_is_aniso, dtype=bool)
        else:
            aniso_arr = np.zeros(n_sites, dtype=bool)

        aniso_t = torch.as_tensor(np.ascontiguousarray(aniso_arr), dtype=torch.bool, device=device)
        dims = torch.where(aniso_t, 6, 1)
        offsets = torch.zeros(n_sites + 1, dtype=torch.long, device=device)
        torch.cumsum(dims, dim=0, out=offsets[1:])
        n_params = int(offsets[-1].item())

        eff_wilson = float(wilson_b if wilson_b is not None else (packed.wilson_b if packed.wilson_b is not None else 20.0))
        eff_nu = float(nu if nu is not None else (packed.adp_nu if packed.adp_nu is not None else 4.0))
        eff_level_w = float(
            level_weight
            if level_weight is not None
            else (packed.adp_level_weight if packed.adp_level_weight is not None else 1.0)
        )

        p_pair_i = t_int(packed.adp_pair_i_seqs) if packed.adp_pair_i_seqs is not None else torch.zeros((0, 2), dtype=torch.long, device=device)
        p_pair_w = t_f(packed.adp_pair_weight) if packed.adp_pair_weight is not None else torch.zeros((0,), dtype=dtype, device=device)
        p_pair_tau = t_f(packed.adp_pair_tau) if packed.adp_pair_tau is not None else torch.zeros((0,), dtype=dtype, device=device)
        p_pair_cls = t_int(packed.adp_pair_class) if packed.adp_pair_class is not None else torch.zeros((0,), dtype=torch.long, device=device)

        p_rigid_i = t_int(packed.rigid_bond_i_seqs) if packed.rigid_bond_i_seqs is not None else torch.zeros((0, 2), dtype=torch.long, device=device)
        p_rigid_w = t_f(packed.rigid_bond_weight) if packed.rigid_bond_weight is not None else torch.zeros((0,), dtype=dtype, device=device)

        b_hat_rigid = None
        if p_rigid_i.numel() > 0 and sites is not None:
            s_arr = np.asarray(sites, dtype=np.float64)
            s_t = torch.as_tensor(s_arr, dtype=dtype, device=device)
            p1 = s_t[p_rigid_i[:, 0]]
            p2 = s_t[p_rigid_i[:, 1]]
            b_vec = p2 - p1
            b_norm = torch.linalg.norm(b_vec, dim=-1, keepdim=True).clamp_min(1e-12)
            b_hat_rigid = b_vec / b_norm

        return cls(
            pair_i=p_pair_i,
            pair_w=p_pair_w,
            pair_tau=p_pair_tau,
            pair_class=p_pair_cls,
            rigid_i=p_rigid_i,
            rigid_w=p_rigid_w,
            is_aniso=aniso_t,
            wilson_b=eff_wilson,
            nu=eff_nu,
            level_weight=eff_level_w,
            w_iso=float(w_iso),
            tau_iso=float(tau_iso),
            n_sites=n_sites,
            param_dims=dims,
            param_offsets=offsets,
            n_params=n_params,
            b_hat_rigid=b_hat_rigid,
        )


class ADPPrior:
    """Evaluates hierarchical, scale-invariant ADP prior energy, gradients, and curvatures."""

    def __init__(self, tables: ADPPriorTables) -> None:
        self.t = tables

    # -- packing / unpacking -----------------------------------------------------------
    def pack_params(self, beta: Tensor, S: Optional[Tensor] = None) -> Tensor:
        """Pack beta (N,) and S (N, 6) or (N, 3, 3) into 1D parameter vector (M,)."""
        t = self.t
        n = t.n_sites
        dev = beta.device
        dtype = beta.dtype
        vec = torch.zeros(t.n_params, dtype=dtype, device=dev)

        if S is not None and S.ndim == 3 and S.shape[-2:] == (3, 3):
            S6 = sym_mat_to_vec6(S)
        else:
            S6 = S

        for i in range(n):
            o = int(t.param_offsets[i].item())
            if bool(t.is_aniso[i].item()):
                if S6 is not None:
                    vec[o : o + 6] = S6[i]
                else:
                    vec[o : o + 3] = beta[i]
                    vec[o + 3 : o + 6] = 0.0
            else:
                vec[o] = beta[i]
        return vec

    def unpack_params(self, adp_vec: Any) -> tuple[Tensor, Tensor]:
        """Unpack 1D parameter vector (M,) into beta (N,) and S (N, 6).

        For isotropic atoms, beta[i] = adp_vec[offset[i]] and S[i] = [beta[i], beta[i], beta[i], 0, 0, 0].
        For anisotropic atoms, S[i] = adp_vec[offset[i]:offset[i]+6] and beta[i] = mean(diag(S[i])).
        """
        t = self.t
        n = t.n_sites
        if not torch.is_tensor(adp_vec):
            adp_vec = torch.as_tensor(np.asarray(adp_vec, dtype=np.float64), dtype=torch.float64, device=t.param_dims.device)
        dev = adp_vec.device
        dtype = adp_vec.dtype

        # Vectorized gather using param_offsets
        aniso = t.is_aniso
        offsets = t.param_offsets

        # S6 for all atoms: shape (N, 6)
        S6 = torch.zeros((n, 6), dtype=dtype, device=dev)
        beta = torch.zeros(n, dtype=dtype, device=dev)

        iso_indices = torch.where(~aniso)[0]
        aniso_indices = torch.where(aniso)[0]

        if iso_indices.numel() > 0:
            iso_off = offsets[iso_indices]
            b_iso = adp_vec[iso_off]
            beta[iso_indices] = b_iso
            S6[iso_indices, 0] = b_iso
            S6[iso_indices, 1] = b_iso
            S6[iso_indices, 2] = b_iso

        if aniso_indices.numel() > 0:
            aniso_off = offsets[aniso_indices]
            for idx in aniso_indices:
                i = int(idx.item())
                o = int(offsets[i].item())
                s_val = adp_vec[o : o + 6]
                S6[i] = s_val
                beta[i] = (s_val[0] + s_val[1] + s_val[2]) / 3.0

        return beta, S6

    # -- energy evaluation -------------------------------------------------------------
    def energy_vec(self, adp_vec: Any, sites: Optional[Any] = None) -> Tensor:
        if sites is not None and not torch.is_tensor(sites):
            sites = torch.as_tensor(np.asarray(sites, dtype=np.float64), dtype=torch.float64, device=self.t.param_dims.device)
        beta, S6 = self.unpack_params(adp_vec)
        return self.energy(beta, S6, sites=sites)

    def gradient(self, adp_vec: Any, sites: Optional[Any] = None) -> Tensor:
        """Exact gradient d E_ADP / d adp_vec."""
        if not torch.is_tensor(adp_vec):
            adp_vec = torch.as_tensor(np.asarray(adp_vec, dtype=np.float64), dtype=torch.float64, device=self.t.param_dims.device)
        if sites is not None and not torch.is_tensor(sites):
            sites = torch.as_tensor(np.asarray(sites, dtype=np.float64), dtype=torch.float64, device=self.t.param_dims.device)
        x = adp_vec.detach().clone().requires_grad_(True)
        e = self.energy_vec(x, sites=sites)
        (g,) = torch.autograd.grad(e, x)
        return g

    def energy(
        self,
        beta: Tensor,
        S: Optional[Tensor] = None,
        sites: Optional[Tensor] = None,
    ) -> Tensor:
        """Exact E_ADP(beta, S)."""
        t = self.t
        dev = beta.device
        dtype = beta.dtype
        e = torch.zeros((), dtype=dtype, device=dev)

        if S is not None and S.ndim == 3 and S.shape[-2:] == (3, 3):
            S6 = sym_mat_to_vec6(S)
        else:
            S6 = S

        # 1. Pair similarity terms
        if t.pair_i.numel() > 0:
            i_idx = t.pair_i[:, 0]
            j_idx = t.pair_i[:, 1]
            w_ij = t.pair_w
            tau_ij = t.pair_tau
            aniso_i = t.is_aniso[i_idx]
            aniso_j = t.is_aniso[j_idx]

            # Case A: both isotropic
            m_ii = (~aniso_i) & (~aniso_j)
            if m_ii.any():
                ii_idx = torch.where(m_ii)[0]
                bi = beta[i_idx[ii_idx]]
                bj = beta[j_idx[ii_idx]]
                r_ii = (bi - bj) / tau_ij[ii_idx]
                e = e + (w_ij[ii_idx] * psi_nu(r_ii, t.nu)).sum()

            # Case B: both anisotropic
            m_aa = aniso_i & aniso_j
            if m_aa.any():
                aa_idx = torch.where(m_aa)[0]
                si = S6[i_idx[aa_idx]]
                sj = S6[j_idx[aa_idx]]
                diff = si - sj
                nw = torch.as_tensor(NORM_WEIGHTS_6, dtype=dtype, device=dev)
                scaled_diff = (diff * nw) / tau_ij[aa_idx, None]
                r_aa = torch.linalg.norm(scaled_diff, dim=-1)
                e = e + (w_ij[aa_idx] * psi_nu(r_aa, t.nu)).sum()

            # Case C: mixed iso-aniso (i is iso, j is aniso)
            m_ia = (~aniso_i) & aniso_j
            if m_ia.any():
                ia_idx = torch.where(m_ia)[0]
                bi = beta[i_idx[ia_idx]]
                sj = S6[j_idx[ia_idx]]
                bar_sj = (sj[:, 0] + sj[:, 1] + sj[:, 2]) / 3.0
                r_mean = (bi - bar_sj) / tau_ij[ia_idx]
                e = e + (w_ij[ia_idx] * psi_nu(r_mean, t.nu)).sum()

                # Deviatoric penalty on sj
                dev_sj = sj.clone()
                dev_sj[:, 0] = dev_sj[:, 0] - bar_sj
                dev_sj[:, 1] = dev_sj[:, 1] - bar_sj
                dev_sj[:, 2] = dev_sj[:, 2] - bar_sj
                nw = torch.as_tensor(NORM_WEIGHTS_6, dtype=dtype, device=dev)
                r_dev = torch.linalg.norm((dev_sj * nw) / t.tau_iso, dim=-1)
                e = e + (t.w_iso * psi_nu(r_dev, t.nu)).sum()

            # Case D: mixed iso-aniso (i is aniso, j is iso)
            m_ai = aniso_i & (~aniso_j)
            if m_ai.any():
                ai_idx = torch.where(m_ai)[0]
                si = S6[i_idx[ai_idx]]
                bj = beta[j_idx[ai_idx]]
                bar_si = (si[:, 0] + si[:, 1] + si[:, 2]) / 3.0
                r_mean = (bar_si - bj) / tau_ij[ai_idx]
                e = e + (w_ij[ai_idx] * psi_nu(r_mean, t.nu)).sum()

                # Deviatoric penalty on si
                dev_si = si.clone()
                dev_si[:, 0] = dev_si[:, 0] - bar_si
                dev_si[:, 1] = dev_si[:, 1] - bar_si
                dev_si[:, 2] = dev_si[:, 2] - bar_si
                nw = torch.as_tensor(NORM_WEIGHTS_6, dtype=dtype, device=dev)
                r_dev = torch.linalg.norm((dev_si * nw) / t.tau_iso, dim=-1)
                e = e + (t.w_iso * psi_nu(r_dev, t.nu)).sum()

        # 2. Hirshfeld rigid bond term
        if t.rigid_i.numel() > 0:
            b_hat = t.b_hat_rigid
            if b_hat is None and sites is not None:
                p1 = sites[t.rigid_i[:, 0]]
                p2 = sites[t.rigid_i[:, 1]]
                b_vec = p2 - p1
                b_hat = b_vec / torch.linalg.norm(b_vec, dim=-1, keepdim=True).clamp_min(1e-12)

            if b_hat is not None:
                r_idx_i = t.rigid_i[:, 0]
                r_idx_j = t.rigid_i[:, 1]
                w_h = t.rigid_w

                # Project atom i along b_hat
                u_proj_i = self._along_bond_projection(beta, S6, r_idx_i, b_hat)
                u_proj_j = self._along_bond_projection(beta, S6, r_idx_j, b_hat)
                rho_rigid = u_proj_i - u_proj_j
                e = e + (w_h * rho_rigid * rho_rigid).sum()

        # 3. Level anchor term
        if t.level_weight > 0.0 and t.wilson_b is not None and t.wilson_b > 0.0:
            log_b_wilson = math.log(float(t.wilson_b))
            bar_beta = beta.mean()
            rho_anchor = bar_beta - log_b_wilson
            e = e + t.level_weight * (rho_anchor * rho_anchor)

        return e

    def _along_bond_projection(self, beta: Tensor, S6: Tensor, atom_idx: Tensor, b_hat: Tensor) -> Tensor:
        """Compute b_hat^T U_i b_hat for atoms atom_idx."""
        t = self.t
        k = atom_idx.shape[0]
        proj = torch.zeros(k, dtype=beta.dtype, device=beta.device)

        aniso_flags = t.is_aniso[atom_idx]
        iso_k = torch.where(~aniso_flags)[0]
        aniso_k = torch.where(aniso_flags)[0]

        if iso_k.numel() > 0:
            b_val = beta[atom_idx[iso_k]]
            proj[iso_k] = torch.exp(b_val) / EIGHT_PI_SQ

        if aniso_k.numel() > 0:
            s_vec = S6[atom_idx[aniso_k]]
            s_mat = vec6_to_sym_mat(s_vec)
            # U = exp(S) / (8 pi^2)
            u_mat = torch.linalg.matrix_exp(s_mat) / EIGHT_PI_SQ
            b_sub = b_hat[aniso_k]  # (K_sub, 3)
            # b^T U b = sum_{a,c} b_a U_{ac} b_c
            u_b = torch.einsum("kab,kb->ka", u_mat, b_sub)
            proj[aniso_k] = (b_sub * u_b).sum(dim=-1)

        return proj

    # -- residuals and IRLS weights ----------------------------------------------------
    def residuals_vec(self, adp_vec: Tensor, sites: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        beta, S6 = self.unpack_params(adp_vec)
        return self.residuals(beta, S6, sites=sites)

    def residuals(
        self,
        beta: Tensor,
        S: Optional[Tensor] = None,
        sites: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Concatenated (rho, w) in IRLS form so that E_quad = sum w rho^2.

        Grad w.r.t. parameters with w frozen equals exact grad(E_ADP).
        Gauss-Newton curvature is 2 J^T W J.
        """
        t = self.t
        dev = beta.device
        dtype = beta.dtype

        if S is not None and S.ndim == 3 and S.shape[-2:] == (3, 3):
            S6 = sym_mat_to_vec6(S)
        else:
            S6 = S

        rhos: list[Tensor] = []
        weights: list[Tensor] = []

        # 1. Pair similarity terms
        if t.pair_i.numel() > 0:
            i_idx = t.pair_i[:, 0]
            j_idx = t.pair_i[:, 1]
            w_ij = t.pair_w
            tau_ij = t.pair_tau
            aniso_i = t.is_aniso[i_idx]
            aniso_j = t.is_aniso[j_idx]

            # Case A: both isotropic
            m_ii = (~aniso_i) & (~aniso_j)
            if m_ii.any():
                ii_idx = torch.where(m_ii)[0]
                bi = beta[i_idx[ii_idx]]
                bj = beta[j_idx[ii_idx]]
                r_ii = (bi - bj) / tau_ij[ii_idx]
                omega_ii = omega_nu(r_ii.detach(), t.nu)
                w_eff = 0.5 * w_ij[ii_idx] * omega_ii
                rhos.append(r_ii)
                weights.append(w_eff)

            # Case B: both anisotropic
            m_aa = aniso_i & aniso_j
            if m_aa.any():
                aa_idx = torch.where(m_aa)[0]
                si = S6[i_idx[aa_idx]]
                sj = S6[j_idx[aa_idx]]
                diff = si - sj
                nw = torch.as_tensor(NORM_WEIGHTS_6, dtype=dtype, device=dev)
                scaled_diff = (diff * nw) / tau_ij[aa_idx, None]  # (K, 6)
                r_norm = torch.linalg.norm(scaled_diff.detach(), dim=-1)  # (K,)
                omega_aa = omega_nu(r_norm, t.nu)  # (K,)
                w_eff = 0.5 * w_ij[aa_idx] * omega_aa  # (K,)
                for m in range(6):
                    rhos.append(scaled_diff[:, m])
                    weights.append(w_eff)

            # Case C: mixed iso-aniso (i is iso, j is aniso)
            m_ia = (~aniso_i) & aniso_j
            if m_ia.any():
                ia_idx = torch.where(m_ia)[0]
                bi = beta[i_idx[ia_idx]]
                sj = S6[j_idx[ia_idx]]
                bar_sj = (sj[:, 0] + sj[:, 1] + sj[:, 2]) / 3.0
                r_mean = (bi - bar_sj) / tau_ij[ia_idx]
                omega_mean = omega_nu(r_mean.detach(), t.nu)
                rhos.append(r_mean)
                weights.append(0.5 * w_ij[ia_idx] * omega_mean)

                # Deviatoric penalty on sj
                dev_sj = sj.clone()
                dev_sj[:, 0] = dev_sj[:, 0] - bar_sj
                dev_sj[:, 1] = dev_sj[:, 1] - bar_sj
                dev_sj[:, 2] = dev_sj[:, 2] - bar_sj
                nw = torch.as_tensor(NORM_WEIGHTS_6, dtype=dtype, device=dev)
                scaled_dev = (dev_sj * nw) / t.tau_iso
                r_dev_norm = torch.linalg.norm(scaled_dev.detach(), dim=-1)
                omega_dev = omega_nu(r_dev_norm, t.nu)
                w_eff_dev = 0.5 * t.w_iso * omega_dev
                for m in range(6):
                    rhos.append(scaled_dev[:, m])
                    weights.append(w_eff_dev)

            # Case D: mixed iso-aniso (i is aniso, j is iso)
            m_ai = aniso_i & (~aniso_j)
            if m_ai.any():
                ai_idx = torch.where(m_ai)[0]
                si = S6[i_idx[ai_idx]]
                bj = beta[j_idx[ai_idx]]
                bar_si = (si[:, 0] + si[:, 1] + si[:, 2]) / 3.0
                r_mean = (bar_si - bj) / tau_ij[ai_idx]
                omega_mean = omega_nu(r_mean.detach(), t.nu)
                rhos.append(r_mean)
                weights.append(0.5 * w_ij[ai_idx] * omega_mean)

                # Deviatoric penalty on si
                dev_si = si.clone()
                dev_si[:, 0] = dev_si[:, 0] - bar_si
                dev_si[:, 1] = dev_si[:, 1] - bar_si
                dev_si[:, 2] = dev_si[:, 2] - bar_si
                nw = torch.as_tensor(NORM_WEIGHTS_6, dtype=dtype, device=dev)
                scaled_dev = (dev_si * nw) / t.tau_iso
                r_dev_norm = torch.linalg.norm(scaled_dev.detach(), dim=-1)
                omega_dev = omega_nu(r_dev_norm, t.nu)
                w_eff_dev = 0.5 * t.w_iso * omega_dev
                for m in range(6):
                    rhos.append(scaled_dev[:, m])
                    weights.append(w_eff_dev)

        # 2. Hirshfeld rigid bond term
        if t.rigid_i.numel() > 0:
            b_hat = t.b_hat_rigid
            if b_hat is None and sites is not None:
                p1 = sites[t.rigid_i[:, 0]]
                p2 = sites[t.rigid_i[:, 1]]
                b_vec = p2 - p1
                b_hat = b_vec / torch.linalg.norm(b_vec, dim=-1, keepdim=True).clamp_min(1e-12)

            if b_hat is not None:
                r_idx_i = t.rigid_i[:, 0]
                r_idx_j = t.rigid_i[:, 1]
                u_proj_i = self._along_bond_projection(beta, S6, r_idx_i, b_hat)
                u_proj_j = self._along_bond_projection(beta, S6, r_idx_j, b_hat)
                rho_rigid = u_proj_i - u_proj_j
                rhos.append(rho_rigid)
                weights.append(t.rigid_w)

        # 3. Level anchor term
        if t.level_weight > 0.0 and t.wilson_b is not None and t.wilson_b > 0.0:
            log_b_wilson = math.log(float(t.wilson_b))
            bar_beta = beta.mean()
            rho_anchor = bar_beta - log_b_wilson
            rhos.append(rho_anchor.reshape(1))
            weights.append(torch.tensor([t.level_weight], dtype=dtype, device=dev))

        if not rhos:
            z = torch.zeros(0, dtype=dtype, device=dev)
            return z, z

        return torch.cat([r.reshape(-1) for r in rhos]), torch.cat([w.reshape(-1) for w in weights])

    def energy_from_residuals(self, adp_vec: Tensor, sites: Optional[Tensor] = None) -> Tensor:
        rho, w = self.residuals_vec(adp_vec, sites=sites)
        return (w * rho * rho).sum()

    # -- Gauss-Newton & full second derivatives ----------------------------------------
    def gn_hvp(self, adp_vec: Tensor, v: Tensor, sites: Optional[Tensor] = None) -> Tensor:
        """(2 J^T W J) v by double vjp with frozen IRLS weights."""
        x = adp_vec.detach().clone().requires_grad_(True)
        rho, w = self.residuals_vec(x, sites=sites)
        if rho.numel() == 0:
            return torch.zeros_like(adp_vec)
        w_detached = w.detach()
        u = torch.zeros_like(rho, requires_grad=True)
        (vjp,) = torch.autograd.grad(rho, x, grad_outputs=u, create_graph=True)
        (Jv,) = torch.autograd.grad((vjp * v).sum(), u, create_graph=True)
        (hv,) = torch.autograd.grad(rho, x, grad_outputs=2.0 * w_detached * Jv)
        return hv

    def full_hvp(self, adp_vec: Tensor, v: Tensor, sites: Optional[Tensor] = None) -> Tensor:
        """Exact Hessian-vector product nabla^2 E(x) v by double backward on exact E."""
        x = adp_vec.detach().clone().requires_grad_(True)
        e = self.energy_vec(x, sites=sites)
        (g,) = torch.autograd.grad(e, x, create_graph=True)
        (hv,) = torch.autograd.grad((g * v).sum(), x)
        return hv

    def hvp(self, adp_vec: Tensor, v: Tensor, sites: Optional[Tensor] = None, hessian: str = "gn") -> Tensor:
        if hessian == "gn":
            return self.gn_hvp(adp_vec, v, sites=sites)
        if hessian == "full":
            return self.full_hvp(adp_vec, v, sites=sites)
        raise ValueError("hessian must be 'gn' or 'full'")

    def gn_sparse_coo(
        self,
        adp_vec: Any,
        sites: Optional[Any] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """COO triplets (rows, cols, vals, M) of exact 2 J^T W J over ADP parameters (M x M)."""
        t = self.t
        M = t.n_params
        if not torch.is_tensor(adp_vec):
            x = torch.as_tensor(np.asarray(adp_vec, dtype=np.float64), dtype=torch.float64, device=t.param_dims.device).requires_grad_(True)
        else:
            x = adp_vec.detach().clone().requires_grad_(True)

        if not t.is_aniso.any() and t.rigid_i.numel() == 0 and M > 60:
            i_idx = t.pair_i[:, 0].cpu().numpy()
            j_idx = t.pair_i[:, 1].cpu().numpy()
            bi = x.detach().cpu().numpy()[i_idx]
            bj = x.detach().cpu().numpy()[j_idx]
            tau = t.pair_tau.cpu().numpy()
            w_ij = t.pair_w.cpu().numpy()
            r = (bi - bj) / tau
            nu = float(t.nu)
            omega = np.ones_like(r) if math.isinf(nu) else (nu + 1.0) / (nu + r * r)
            w_eff = 0.5 * w_ij * omega
            k = (2.0 * w_eff) / (tau * tau)

            rows_l = [i_idx, j_idx, i_idx, j_idx]
            cols_l = [i_idx, j_idx, j_idx, i_idx]
            vals_l = [k, k, -k, -k]

            if t.level_weight > 0.0 and t.wilson_b is not None and t.wilson_b > 0.0:
                lam = float(t.level_weight)
                diag_idx = np.arange(M)
                rows_l.append(diag_idx)
                cols_l.append(diag_idx)
                vals_l.append(np.full(M, 2.0 * lam / (M * M)))

            r_all = np.concatenate(rows_l).astype(np.int64)
            c_all = np.concatenate(cols_l).astype(np.int64)
            v_all = np.concatenate(vals_l).astype(np.float64)
            return r_all, c_all, v_all, M

        rho, w = self.residuals_vec(x, sites=sites)
        if rho.numel() == 0:
            e = np.zeros(0, dtype=np.int64)
            return e, e, np.zeros(0, dtype=np.float64), M

        # Compute J^T W J via torch.autograd.functional.jacobian or batch vjp
        # For small-to-moderate models (and benchmark verification), computing J is exact and clean.
        w_d = w.detach()

        # Compute Jacobian J: (K, M)
        J = torch.autograd.functional.jacobian(lambda z: self.residuals_vec(z, sites=sites)[0], x)
        H_dense = 2.0 * (J.T @ (w_d[:, None] * J))  # (M, M)

        H_np = H_dense.detach().cpu().numpy().astype(np.float64)

        # Find non-zeros above round-off floor
        tol = 1e-16 * max(float(np.abs(H_np).max()), 1.0)
        nz_r, nz_c = np.where(np.abs(H_np) > tol)
        nz_v = H_np[nz_r, nz_c]

        return nz_r.astype(np.int64), nz_c.astype(np.int64), nz_v.astype(np.float64), M

    def gn_blocks(self, adp_vec: Tensor, sites: Optional[Tensor] = None) -> list[Tensor]:
        """Per-atom Gauss-Newton diagonal blocks (d_i, d_i)."""
        t = self.t
        n = t.n_sites
        x = adp_vec.detach().clone().requires_grad_(True)
        rho, w = self.residuals_vec(x, sites=sites)
        if rho.numel() == 0:
            return [torch.zeros((int(t.param_dims[i]), int(t.param_dims[i])), dtype=adp_vec.dtype, device=adp_vec.device) for i in range(n)]

        w_d = w.detach()
        J = torch.autograd.functional.jacobian(lambda z: self.residuals_vec(z, sites=sites)[0], x)
        H_dense = 2.0 * (J.T @ (w_d[:, None] * J))

        blocks: list[Tensor] = []
        for i in range(n):
            o = int(t.param_offsets[i].item())
            d = int(t.param_dims[i].item())
            blocks.append(H_dense[o : o + d, o : o + d].clone())
        return blocks

    def gn_diagonal(self, adp_vec: Tensor, sites: Optional[Tensor] = None) -> Tensor:
        """Exact diagonal of 2 J^T W J as an (M,) tensor."""
        t = self.t
        M = t.n_params
        x = adp_vec.detach().clone().requires_grad_(True)

        if not t.is_aniso.any() and t.rigid_i.numel() == 0:
            if t.pair_i.numel() > 0:
                i_idx = t.pair_i[:, 0].cpu().numpy()
                j_idx = t.pair_i[:, 1].cpu().numpy()
                bi = x.detach().cpu().numpy()[i_idx]
                bj = x.detach().cpu().numpy()[j_idx]
                tau = t.pair_tau.cpu().numpy()
                w_ij = t.pair_w.cpu().numpy()
                r = (bi - bj) / tau
                nu = float(t.nu)
                omega = np.ones_like(r) if math.isinf(nu) else (nu + 1.0) / (nu + r * r)
                w_eff = 0.5 * w_ij * omega
                k = (2.0 * w_eff) / (tau * tau)
                d = np.bincount(i_idx, weights=k, minlength=M) + np.bincount(j_idx, weights=k, minlength=M)
            else:
                d = np.zeros(M, dtype=np.float64)
            if t.level_weight > 0.0 and t.wilson_b is not None and t.wilson_b > 0.0:
                d += (2.0 * float(t.level_weight)) / (M * M)
            return torch.as_tensor(d, dtype=adp_vec.dtype, device=adp_vec.device)

        rho, w = self.residuals_vec(x, sites=sites)
        if rho.numel() == 0:
            return torch.zeros(M, dtype=adp_vec.dtype, device=adp_vec.device)

        w_d = w.detach()
        J = torch.autograd.functional.jacobian(lambda z: self.residuals_vec(z, sites=sites)[0], x)
        # diag(2 J^T W J) = 2 * sum_r w_r J_{r, m}^2
        diag = 2.0 * (w_d[:, None] * (J * J)).sum(dim=0)
        return diag

    def with_hyperparameters(
        self,
        *,
        tau_12: Optional[float] = None,
        tau_13: Optional[float] = None,
        tau_sphere: Optional[float] = None,
        w_sphere: Optional[float] = None,
        nu: Optional[float] = None,
        level_weight: Optional[float] = None,
    ) -> "ADPPrior":
        """Return a copy of the prior with updated hyperparameters."""
        t = self.t
        p_tau = t.pair_tau.clone()
        p_w = t.pair_w.clone()

        if tau_12 is not None:
            p_tau = torch.where(t.pair_class == 0, float(tau_12), p_tau)
        if tau_13 is not None:
            p_tau = torch.where(t.pair_class == 1, float(tau_13), p_tau)
        if tau_sphere is not None:
            p_tau = torch.where(t.pair_class == 2, float(tau_sphere), p_tau)
        if w_sphere is not None:
            # Re-scale sphere class weights
            m_sph = t.pair_class == 2
            if m_sph.any():
                cur_max = p_w[m_sph].max()
                if cur_max > 0:
                    scale = float(w_sphere) / float(cur_max)
                    p_w[m_sph] = p_w[m_sph] * scale

        new_nu = float(nu if nu is not None else t.nu)
        new_lambda = float(level_weight if level_weight is not None else t.level_weight)

        new_tables = ADPPriorTables(
            pair_i=t.pair_i,
            pair_w=p_w,
            pair_tau=p_tau,
            pair_class=t.pair_class,
            rigid_i=t.rigid_i,
            rigid_w=t.rigid_w,
            is_aniso=t.is_aniso,
            wilson_b=t.wilson_b,
            nu=new_nu,
            level_weight=new_lambda,
            w_iso=t.w_iso,
            tau_iso=t.tau_iso,
            n_sites=t.n_sites,
            param_dims=t.param_dims,
            param_offsets=t.param_offsets,
            n_params=t.n_params,
            b_hat_rigid=t.b_hat_rigid,
        )
        return ADPPrior(new_tables)
