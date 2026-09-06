"""Client-side ADP restraint pair generation and empirical Bayes hyperparameter fitting.

Constructs 1-2, 1-3, and sphere classes from Cartesian coordinates and bond/angle tables
using scipy.spatial.cKDTree (no cctbx needed).
Fits prior hyperparameters by maximizing the Laplace approximation to the marginal
likelihood of the working set via sparse Cholesky / LU determinants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, Field, model_validator
from scipy.spatial import cKDTree
import torch

from phridge.packing_geometry import PackedRestraints


class ADPPriorOptions(BaseModel):
    """Configuration options for the hierarchical scale-invariant ADP prior."""

    w_12: float = 1.0
    tau_12: float = 0.15
    w_13: float = 0.5
    tau_13: float = 0.20
    w_sphere: float = 0.1
    tau_sphere: float = 0.35
    r_sphere: float = 4.5
    w_hirshfeld: float = 10000.0  # 1 / (0.01 A^2)^2
    w_iso: float = 0.05
    tau_iso: float = 0.5
    nu: float = 4.0
    level_weight: float = 1.0
    wilson_b: Optional[float] = None
    allow_large_sphere_weight: bool = False

    @model_validator(mode="after")
    def check_sphere_weight(self) -> "ADPPriorOptions":
        if not self.allow_large_sphere_weight and self.w_sphere > 0.5 * self.w_12:
            raise ValueError(
                f"w_sphere ({self.w_sphere}) must stay <= 0.5 * w_12 ({self.w_12}) "
                "unless allow_large_sphere_weight=True"
            )
        return self


def build_adp_restraints(
    hierarchy: Any,
    restraints: PackedRestraints,
    *,
    options: Optional[ADPPriorOptions] = None,
    wilson_b: Optional[float] = None,
    is_aniso: Optional[np.ndarray] = None,
) -> PackedRestraints:
    """Build ADP prior pair tables (1-2, 1-3, sphere, rigid-bond) without cctbx.

    Parameters
    ----------
    hierarchy : PackedHierarchy or object with xyz or atom coordinates
    restraints : PackedRestraints with bond_i_seqs and angle_i_seqs
    options : ADPPriorOptions
    wilson_b : Wilson B value (overrides options.wilson_b if provided)
    is_aniso : boolean array of shape (N,) indicating anisotropic atoms
    """
    opts = options or ADPPriorOptions()
    if wilson_b is not None:
        opts.wilson_b = float(wilson_b)

    # Extract coordinates
    if hasattr(hierarchy, "xyz"):
        xyz = np.asarray(hierarchy.xyz, dtype=np.float64)
    elif hasattr(hierarchy, "atoms"):
        xyz = np.array([a.xyz for a in hierarchy.atoms], dtype=np.float64)
    elif hasattr(hierarchy, "sites_cart"):
        xyz = np.asarray(hierarchy.sites_cart(), dtype=np.float64)
    else:
        xyz = np.asarray(hierarchy, dtype=np.float64)

    n_sites = int(restraints.n_sites)
    if xyz.shape[0] != n_sites:
        raise ValueError(f"xyz atom count ({xyz.shape[0]}) does not match restraints.n_sites ({n_sites})")

    # Anisotropic flags
    if is_aniso is not None:
        aniso_flags = np.asarray(is_aniso, dtype=bool).reshape(-1)
    elif restraints.adp_is_aniso is not None and restraints.adp_is_aniso.size == n_sites:
        aniso_flags = np.asarray(restraints.adp_is_aniso, dtype=bool)
    else:
        aniso_flags = np.zeros(n_sites, dtype=bool)

    # 1. Bonded 1-2 pairs
    pairs_12: set[tuple[int, int]] = set()
    b_seqs = restraints.bond_i_seqs
    if b_seqs is not None and b_seqs.size > 0:
        for p in b_seqs:
            i, j = int(p[0]), int(p[1])
            if i != j:
                pairs_12.add((min(i, j), max(i, j)))

    # 2. Bonded 1-3 pairs (outer atoms of angles)
    pairs_13: set[tuple[int, int]] = set()
    a_seqs = restraints.angle_i_seqs
    if a_seqs is not None and a_seqs.size > 0:
        for ang in a_seqs:
            i, k = int(ang[0]), int(ang[2])
            if i != k:
                pair = (min(i, k), max(i, k))
                if pair not in pairs_12:
                    pairs_13.add(pair)

    # 3. Sphere class pairs (non-bonded within r_sphere)
    tree = cKDTree(xyz)
    cand_pairs = tree.query_pairs(opts.r_sphere)
    pairs_sphere: list[tuple[int, int, float]] = []  # (i, j, weight)
    for i, j in cand_pairs:
        pair = (min(i, j), max(i, j))
        if pair in pairs_12 or pair in pairs_13:
            continue
        dist = float(np.linalg.norm(xyz[i] - xyz[j]))
        if dist < opts.r_sphere:
            # Kernel w = w_s * (1 - d / r_s)^2
            w = opts.w_sphere * ((1.0 - dist / opts.r_sphere) ** 2)
            pairs_sphere.append((pair[0], pair[1], w))

    # Assemble pair arrays
    pair_list: list[tuple[int, int]] = []
    weight_list: list[float] = []
    tau_list: list[float] = []
    class_list: list[int] = []

    for i, j in sorted(pairs_12):
        pair_list.append((i, j))
        weight_list.append(opts.w_12)
        tau_list.append(opts.tau_12)
        class_list.append(0)

    for i, j in sorted(pairs_13):
        pair_list.append((i, j))
        weight_list.append(opts.w_13)
        tau_list.append(opts.tau_13)
        class_list.append(1)

    for i, j, w in sorted(pairs_sphere, key=lambda x: (x[0], x[1])):
        pair_list.append((i, j))
        weight_list.append(w)
        tau_list.append(opts.tau_sphere)
        class_list.append(2)

    p_arr = np.array(pair_list, dtype=np.int32) if pair_list else np.zeros((0, 2), dtype=np.int32)
    w_arr = np.array(weight_list, dtype=np.float64) if weight_list else np.zeros(0, dtype=np.float64)
    tau_arr = np.array(tau_list, dtype=np.float64) if tau_list else np.zeros(0, dtype=np.float64)
    cls_arr = np.array(class_list, dtype=np.int32) if class_list else np.zeros(0, dtype=np.int32)

    # 4. Hirshfeld rigid bond pairs (bonded pairs with at least one anisotropic atom)
    rigid_list: list[tuple[int, int]] = []
    if opts.w_hirshfeld > 0.0:
        for i, j in sorted(pairs_12):
            if aniso_flags[i] or aniso_flags[j]:
                rigid_list.append((i, j))

    r_arr = np.array(rigid_list, dtype=np.int32) if rigid_list else np.zeros((0, 2), dtype=np.int32)
    rw_arr = np.full(len(rigid_list), opts.w_hirshfeld, dtype=np.float64) if rigid_list else np.zeros(0, dtype=np.float64)

    # Return new PackedRestraints with ADP prior arrays populated
    return PackedRestraints(
        n_sites=n_sites,
        crystal=restraints.meta.crystal,
        bond_i_seqs=restraints.bond_i_seqs,
        bond_distance_ideal=restraints.bond_distance_ideal,
        bond_weight=restraints.bond_weight,
        bond_slack=restraints.bond_slack,
        bond_origin_id=restraints.bond_origin_id,
        bond_asu_i_seqs=restraints.bond_asu_i_seqs,
        bond_asu_distance_ideal=restraints.bond_asu_distance_ideal,
        bond_asu_weight=restraints.bond_asu_weight,
        bond_asu_rt_mx=restraints.bond_asu_rt_mx,
        angle_i_seqs=restraints.angle_i_seqs,
        angle_ideal=restraints.angle_ideal,
        angle_weight=restraints.angle_weight,
        angle_origin_id=restraints.angle_origin_id,
        dihedral_i_seqs=restraints.dihedral_i_seqs,
        dihedral_angle_ideal=restraints.dihedral_angle_ideal,
        dihedral_weight=restraints.dihedral_weight,
        dihedral_periodicity=restraints.dihedral_periodicity,
        dihedral_origin_id=restraints.dihedral_origin_id,
        chirality_i_seqs=restraints.chirality_i_seqs,
        chirality_volume_ideal=restraints.chirality_volume_ideal,
        chirality_weight=restraints.chirality_weight,
        chirality_both_signs=restraints.chirality_both_signs,
        chirality_origin_id=restraints.chirality_origin_id,
        planarity_i_seqs=restraints.planarity_i_seqs,
        planarity_offsets=restraints.planarity_offsets,
        planarity_weight=restraints.planarity_weight,
        planarity_origin_id=restraints.planarity_origin_id,
        parallelity_i_offsets=restraints.parallelity_i_offsets,
        parallelity_i_seqs=restraints.parallelity_i_seqs,
        parallelity_j_offsets=restraints.parallelity_j_offsets,
        parallelity_j_seqs=restraints.parallelity_j_seqs,
        parallelity_weight=restraints.parallelity_weight,
        parallelity_target_angle=restraints.parallelity_target_angle,
        parallelity_origin_id=restraints.parallelity_origin_id,
        nonbonded_i_seqs=restraints.nonbonded_i_seqs,
        nonbonded_vdw=restraints.nonbonded_vdw,
        refcoord_i_seq=restraints.refcoord_i_seq,
        refcoord_xyz=restraints.refcoord_xyz,
        refcoord_weight=restraints.refcoord_weight,
        bondsim_pair_i_seqs=restraints.bondsim_pair_i_seqs,
        bondsim_offsets=restraints.bondsim_offsets,
        bondsim_weight=restraints.bondsim_weight,
        adp_pair_i_seqs=p_arr,
        adp_pair_weight=w_arr,
        adp_pair_tau=tau_arr,
        adp_pair_class=cls_arr,
        rigid_bond_i_seqs=r_arr,
        rigid_bond_weight=rw_arr,
        wilson_b=opts.wilson_b,
        adp_nu=opts.nu,
        adp_level_weight=opts.level_weight,
        adp_is_aniso=aniso_flags.astype(np.uint8),
        model_indices=restraints.model_indices,
        conformer_indices=restraints.conformer_indices,
    )


def fit_adp_hyperparameters(
    prior: Any,
    adp_vec: Any,
    h_xray_diag: Optional[np.ndarray] = None,
    *,
    sites: Optional[Any] = None,
    method: str = "Nelder-Mead",
    maxiter: int = 50,
) -> dict[str, Any]:
    """Fit ADP hyperparameters by maximizing the Laplace approximation to the working-set marginal likelihood.

    Marginal log-likelihood objective:
        log p(I_work | eta) approx -E_ADP(hat_beta; eta) + 0.5 * logdet(Q_eta) - 0.5 * logdet(Q_eta + H_xray) + const
    where Q_eta is the IRLS precision matrix (Gauss-Newton Hessian) of the prior.

    Optimizes over log(eta) for the identifiable subset (tau_12, tau_sphere, lambda),
    with tau_13 tied to (4/3) * tau_12.
    Reports optimal eta with marginal likelihood curvature standard errors.
    """
    from scipy.optimize import minimize
    from phridge.worker.geometry.curvature import GaussNewtonPreconditioner

    x = torch.as_tensor(adp_vec, dtype=torch.float64) if not torch.is_tensor(adp_vec) else adp_vec.to(dtype=torch.float64)
    sites_t = torch.as_tensor(sites, dtype=torch.float64) if sites is not None and not torch.is_tensor(sites) else sites

    n_params = x.shape[0]
    hx = np.asarray(h_xray_diag, dtype=np.float64).reshape(-1) if h_xray_diag is not None else np.zeros(n_params, dtype=np.float64)

    # Initial hyperparameters
    init_tau_12 = float(prior.t.pair_tau[prior.t.pair_class == 0][0].item()) if (prior.t.pair_class == 0).any() else 0.15
    init_tau_sph = float(prior.t.pair_tau[prior.t.pair_class == 2][0].item()) if (prior.t.pair_class == 2).any() else 0.35
    init_lambda = float(prior.t.level_weight)

    # Parameters to optimize: theta = [log(tau_12), log(tau_sphere), log(level_weight)]
    theta0 = np.array([math.log(max(init_tau_12, 1e-4)), math.log(max(init_tau_sph, 1e-4)), math.log(max(init_lambda, 1e-4))])

    def nll_func(theta: np.ndarray) -> float:
        th = np.clip(theta, -10.0, 10.0)
        t12 = float(math.exp(th[0]))
        tsph = float(math.exp(th[1]))
        lam = float(math.exp(th[2]))
        t13 = t12 * (0.20 / 0.15)  # tied ratio

        p_eval = prior.with_hyperparameters(
            tau_12=t12,
            tau_13=t13,
            tau_sphere=tsph,
            level_weight=lam,
        )

        # Prior energy at hat_beta
        e_prior = float(p_eval.energy_vec(x, sites=sites_t).item())

        # Q_eta
        t = p_eval.t
        if not t.is_aniso.any() and t.rigid_i.numel() == 0 and n_params > 60:
            i_idx = t.pair_i[:, 0].cpu().numpy()
            j_idx = t.pair_i[:, 1].cpu().numpy()
            x_np = x.detach().cpu().numpy()
            bi = x_np[i_idx]
            bj = x_np[j_idx]
            tau = t.pair_tau.cpu().numpy()
            w_ij = t.pair_w.cpu().numpy()
            r = (bi - bj) / tau
            nu = float(t.nu)
            omega = np.ones_like(r) if math.isinf(nu) else (nu + 1.0) / (nu + r * r)
            w_eff = 0.5 * w_ij * omega
            k = (2.0 * w_eff) / (tau * tau)

            r_all = np.concatenate([i_idx, j_idx, i_idx, j_idx])
            c_all = np.concatenate([i_idx, j_idx, j_idx, i_idx])
            v_all = np.concatenate([k, k, -k, -k])
            coo = (r_all, c_all, v_all, n_params)

            c_rank1 = 2.0 * lam / (n_params * n_params)
            ones = np.ones(n_params, dtype=np.float64)

            try:
                m_q = GaussNewtonPreconditioner(p_eval, x, damping=1e-6, coo=coo)
                ld_q = m_q.logdet() + math.log(max(1.0 + c_rank1 * float(np.sum(m_q.solve(ones))), 1e-12))

                m_tot = GaussNewtonPreconditioner(p_eval, x, extra_diag=hx, damping=1e-6, coo=coo)
                ld_tot = m_tot.logdet() + math.log(max(1.0 + c_rank1 * float(np.sum(m_tot.solve(ones))), 1e-12))
            except Exception:
                return 1e12
        else:
            rows, cols, vals, m = p_eval.gn_sparse_coo(x, sites=sites_t)
            coo = (rows, cols, vals, m)
            try:
                # logdet(Q_eta)
                m_q = GaussNewtonPreconditioner(p_eval, x, damping=1e-6, coo=coo)
                ld_q = m_q.logdet()

                # logdet(Q_eta + H_xray)
                m_tot = GaussNewtonPreconditioner(p_eval, x, extra_diag=hx, damping=1e-6, coo=coo)
                ld_tot = m_tot.logdet()
            except Exception:
                return 1e12

        # Objective to minimize: -log p(I_work | eta)
        # = E_ADP - 0.5 * logdet(Q_eta) + 0.5 * logdet(Q_eta + H_xray)
        nll = e_prior - 0.5 * ld_q + 0.5 * ld_tot
        return float(nll)

    res = minimize(nll_func, theta0, method=method, options={"maxiter": maxiter})
    theta_opt = res.x
    t12_opt = float(math.exp(theta_opt[0]))
    tsph_opt = float(math.exp(theta_opt[1]))
    lam_opt = float(math.exp(theta_opt[2]))
    t13_opt = t12_opt * (0.20 / 0.15)

    # Numerical Hessian on theta to compute standard errors
    h_step = 1e-3
    k = len(theta_opt)
    hess = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        for j in range(k):
            ei = np.zeros(k)
            ei[i] = h_step
            ej = np.zeros(k)
            ej[j] = h_step
            fpp = nll_func(theta_opt + ei + ej)
            fpm = nll_func(theta_opt + ei - ej)
            fmp = nll_func(theta_opt - ei + ej)
            fmm = nll_func(theta_opt - ei - ej)
            hess[i, j] = (fpp - fpm - fmp + fmm) / (4.0 * h_step * h_step)

    # Invert Hessian to get covariance matrix of log-parameters
    try:
        # Regularize if slightly ill-conditioned
        h_reg = hess + 1e-6 * np.eye(k)
        cov_theta = np.linalg.inv(h_reg)
        se_theta = np.sqrt(np.maximum(np.diag(cov_theta), 1e-8))
    except Exception:
        cov_theta = np.eye(k)
        se_theta = np.full(k, 0.5)

    # Standard errors on eta by delta method: se(exp(theta)) = exp(theta) * se(theta)
    se_t12 = t12_opt * float(se_theta[0])
    se_tsph = tsph_opt * float(se_theta[1])
    se_lam = lam_opt * float(se_theta[2])

    return {
        "tau_12": t12_opt,
        "tau_12_se": se_t12,
        "tau_13": t13_opt,
        "tau_sphere": tsph_opt,
        "tau_sphere_se": se_tsph,
        "level_weight": lam_opt,
        "level_weight_se": se_lam,
        "covariance_log": cov_theta,
        "marginal_nll": float(res.fun),
        "converged": bool(res.success),
    }
