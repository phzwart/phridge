"""Per-dataset nuisance fit: k_sol, B_sol, traceless U_d, α(s), β(s).

Anisotropic scale ``A_d(h)`` reuses ``aniso_rice``'s traceless Laue-restricted
tensor (``M = I + A``), not Wilson ``B``. ``α`` and ``β`` are cubic B-splines
in s²; ``β`` is free (not tied to ``1 - σ_A²``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from phridge.contrib.intensity_ll.aniso_rice import (
    A_MAX,
    LaueClass,
    a_from_params,
    a_from_params_torch,
    laue_from_crystal,
    n_free,
    quadratic_form,
    quadratic_form_torch,
)
from phridge.contrib.multixtal.reflection_model import RiceMomentQuasiLikelihood, sigma_a_from_alpha_beta
from phridge.contrib.multixtal.residuals import sigma_p_shell
from phridge.contrib.multixtal.spline import eval_log_spline, open_uniform_knots
from phridge.contrib.multixtal.structure import apply_bulk_solvent


@dataclass
class DatasetNuisance:
    k_sol: float
    b_sol: float
    u_params: np.ndarray
    alpha_coef: np.ndarray
    beta_coef: np.ndarray
    knots: np.ndarray
    laue: LaueClass
    nll: float
    sigma_a: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    scale_a: np.ndarray
    f_c_abs2: np.ndarray
    sigma_p: np.ndarray


def anisotropic_scale(s_hat_h: np.ndarray, u_params: np.ndarray, laue: LaueClass) -> np.ndarray:
    a = a_from_params(u_params, laue, a_max=A_MAX)
    m = np.eye(3) + a
    return np.maximum(quadratic_form(s_hat_h, m), 1e-6)


def _aniso_scale_torch(s_hat_h: Any, u_params: Any, laue: LaueClass) -> Any:
    import torch

    a = a_from_params_torch(u_params, laue, a_max=A_MAX)
    eye = torch.eye(3, dtype=u_params.dtype, device=u_params.device)
    return quadratic_form_torch(s_hat_h, eye + a).clamp(min=1e-6)


def _init_alpha_beta(
    i_bar: np.ndarray,
    observed: np.ndarray,
    f_calc_abs2: np.ndarray,
    epsilon: np.ndarray,
    s_sq: np.ndarray,
    n_knots: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Least-squares slope/intercept → log-spline coefficients."""
    knots = open_uniform_knots(float(np.min(s_sq)), float(np.max(s_sq)), n_knots)
    obs = np.asarray(observed, dtype=bool)
    i = np.asarray(i_bar, dtype=np.float64)
    fc2 = np.asarray(f_calc_abs2, dtype=np.float64)
    eps = np.asarray(epsilon, dtype=np.float64)
    if int(obs.sum()) < 8:
        alpha = np.ones_like(i)
        beta = np.full_like(i, max(float(np.nanmedian(np.abs(i[obs]))) if obs.any() else 1.0, 1e-3))
        return knots, np.zeros(n_knots), np.full(n_knots, np.log(max(float(np.median(beta)), 1e-3))), alpha, beta
    x = (eps * fc2)[obs]
    y = i[obs]
    x = np.clip(x, 0.0, None)
    a_mat = np.stack([x, np.ones_like(x)], axis=1)
    coef, *_ = np.linalg.lstsq(a_mat, y, rcond=None)
    slope = max(float(coef[0]), 1e-6)
    intercept = max(float(coef[1]), 1e-3)
    alpha = np.full_like(i, np.sqrt(slope))
    beta = np.full_like(i, intercept)
    alpha_coef = np.full(n_knots, np.log(max(float(np.sqrt(slope)), 1e-4)))
    beta_coef = np.full(n_knots, np.log(max(intercept, 1e-4)))
    return knots, alpha_coef, beta_coef, alpha, beta


def fit_dataset_nuisance(
    i_bar: np.ndarray,
    sig2: np.ndarray,
    observed: np.ndarray,
    f_calc: np.ndarray,
    f_mask: np.ndarray,
    s_sq: np.ndarray,
    s_hat_h: np.ndarray,
    epsilon: np.ndarray,
    centric: np.ndarray,
    crystal: Any,
    n_knots: int = 8,
    max_iter: int = 40,
    device: str = "cpu",
    mode_mean: Optional[np.ndarray] = None,
) -> DatasetNuisance:
    """L-BFGS fit of k_sol, B_sol, U_d, α(s), β(s) on all reflections of one dataset."""
    import torch

    model = RiceMomentQuasiLikelihood()
    laue = laue_from_crystal(crystal)
    n_u = n_free(laue)
    i_t = torch.as_tensor(np.asarray(i_bar, dtype=np.float64), dtype=torch.float64, device=device)
    sig_t = torch.as_tensor(np.asarray(sig2, dtype=np.float64), dtype=torch.float64, device=device)
    obs = torch.as_tensor(np.asarray(observed, dtype=bool), dtype=torch.bool, device=device)
    f_calc_t = torch.as_tensor(np.asarray(f_calc, dtype=np.complex128), dtype=torch.complex128, device=device)
    f_mask_t = torch.as_tensor(np.asarray(f_mask, dtype=np.complex128), dtype=torch.complex128, device=device)
    ss = torch.as_tensor(np.asarray(s_sq, dtype=np.float64), dtype=torch.float64, device=device)
    sh = torch.as_tensor(np.asarray(s_hat_h, dtype=np.float64), dtype=torch.float64, device=device)
    eps = torch.as_tensor(np.asarray(epsilon, dtype=np.float64), dtype=torch.float64, device=device)
    cen = torch.as_tensor(np.asarray(centric, dtype=bool), dtype=torch.bool, device=device)
    extra = None
    if mode_mean is not None:
        extra = torch.as_tensor(np.asarray(mode_mean, dtype=np.float64), dtype=torch.float64, device=device)

    fc2_init = np.abs(np.asarray(f_calc, dtype=np.complex128)) ** 2
    knots, a0, b0, _, _ = _init_alpha_beta(i_bar, observed, fc2_init, epsilon, s_sq, n_knots)
    knots_t = torch.as_tensor(knots, dtype=torch.float64, device=device)

    raw_k = torch.nn.Parameter(torch.tensor(-1.0, dtype=torch.float64, device=device))
    raw_b = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64, device=device))
    u_p = torch.nn.Parameter(torch.zeros(max(n_u, 1), dtype=torch.float64, device=device))
    a_coef = torch.nn.Parameter(torch.as_tensor(a0, dtype=torch.float64, device=device))
    b_coef = torch.nn.Parameter(torch.as_tensor(b0, dtype=torch.float64, device=device))
    params = [raw_k, raw_b, a_coef, b_coef]
    if n_u > 0:
        params.append(u_p)

    opt = torch.optim.LBFGS(params, lr=0.8, max_iter=max(8, max_iter // 4), line_search_fn="strong_wolfe")

    def _unpack() -> tuple[Any, Any, Any, Any, Any]:
        k_sol = 0.6 * torch.sigmoid(raw_k)
        b_sol = 150.0 * torch.sigmoid(raw_b)
        u_use = u_p[:n_u] if n_u > 0 else u_p[:0]
        scale = _aniso_scale_torch(sh, u_use, laue) if n_u > 0 else torch.ones_like(ss)
        alpha = eval_log_spline(ss, knots_t, a_coef)
        beta = eval_log_spline(ss, knots_t, b_coef)
        return k_sol, b_sol, scale, alpha, beta

    def closure() -> Any:
        opt.zero_grad()
        k_sol, b_sol, scale, alpha, beta = _unpack()
        fc = apply_bulk_solvent(f_calc_t, f_mask_t, ss, k_sol, b_sol)
        fc2 = (fc.real**2 + fc.imag**2)
        nll = model.nll(i_t, sig_t, fc2, eps, cen, scale, alpha, beta)
        if extra is not None:
            # Mode contribution already in I units is treated as a mean shift.
            mean, var = model.moments(i_t, sig_t, fc2, eps, cen, scale, alpha, beta)
            var = var.clamp(min=1e-12)
            nll = 0.5 * ((i_t - mean - extra) ** 2 / var + var.log())
        loss = nll[obs].mean()
        loss.backward()
        return loss

    last = float("nan")
    steps = max(1, int(np.ceil(max_iter / 8)))
    for _ in range(steps):
        last = float(opt.step(closure).detach().cpu())

    with torch.no_grad():
        k_sol, b_sol, scale, alpha, beta = _unpack()
        fc = apply_bulk_solvent(f_calc_t, f_mask_t, ss, k_sol, b_sol)
        fc2 = (fc.real**2 + fc.imag**2).cpu().numpy()
        alpha_np = alpha.cpu().numpy()
        beta_np = beta.cpu().numpy()
        scale_np = scale.cpu().numpy()
        u_np = (u_p[:n_u].detach().cpu().numpy() if n_u > 0 else np.zeros(0, dtype=np.float64))
        k_v = float(k_sol.cpu())
        b_v = float(b_sol.cpu())
        a_c = a_coef.detach().cpu().numpy()
        b_c = b_coef.detach().cpu().numpy()

    sp = sigma_p_shell(fc2, np.asarray(s_sq, dtype=np.float64))
    sa = sigma_a_from_alpha_beta(alpha_np, beta_np, sp)
    return DatasetNuisance(
        k_sol=k_v,
        b_sol=b_v,
        u_params=u_np,
        alpha_coef=a_c,
        beta_coef=b_c,
        knots=knots,
        laue=laue,
        nll=last,
        sigma_a=sa,
        alpha=alpha_np,
        beta=beta_np,
        scale_a=scale_np,
        f_c_abs2=fc2,
        sigma_p=sp,
    )
