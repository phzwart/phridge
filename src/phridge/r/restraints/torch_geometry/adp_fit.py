"""ADP hyperparameter fit. Torch evaluation, scipy outer loop."""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import torch

def _hessian_se(nll_func, theta_opt, k: int, h_step: float):
    hess = np.zeros((k, k))
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
    try:
        h_reg = hess + 1e-6 * np.eye(k)
        cov_theta = np.linalg.inv(h_reg)
        se_theta = np.sqrt(np.maximum(np.diag(cov_theta), 1e-8))
    except Exception:
        cov_theta = np.eye(k)
        se_theta = np.full(k, 0.5)
    return cov_theta, se_theta


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
    from phridge.numeric.optimize import minimize
    from phridge.r.restraints.torch_geometry.curvature import GaussNewtonPreconditioner

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
    cov_theta, se_theta = _hessian_se(nll_func, theta_opt, k, h_step)

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
