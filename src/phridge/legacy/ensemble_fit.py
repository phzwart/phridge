"""Initialisation, generalised EM, and the Fisher-identity L-BFGS path."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from phridge.r.generators.perturbation_modes.gauge import fix_gauge, reconstruction
from phridge.r.generators.perturbation_modes.moments import difference_coefficients, fisher_mu_gradient, posterior_f_moments
from phridge.r.generators.perturbation_modes.mstep import (
    als_known_f,
    expand_shells,
    initialize_from_residuals,
    solve_modes_torch,
    solve_scores_torch,
    update_shell_scales,
    work_weights,
)


@dataclass
class EnsembleResult:
    """Arrays returned by ``em_fit`` / ``direct_fit`` / ``als_known_f``."""

    delta0: np.ndarray
    modes: np.ndarray
    scores: np.ndarray
    score_cov: np.ndarray
    sigma2: np.ndarray
    d: np.ndarray
    shell_index: np.ndarray
    loglik_history: np.ndarray
    fom: np.ndarray
    heldout: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    mean_f: Optional[np.ndarray] = None
    loglik_unprojected: Optional[float] = None
    loglik_projected: Optional[float] = None


def _shells_or_one(shell_index: Optional[np.ndarray], n_h: int) -> np.ndarray:
    if shell_index is None:
        return np.zeros(n_h, dtype=np.int32)
    return np.asarray(shell_index, dtype=np.int32).reshape(-1)


def _broadcast_model(f_mu: np.ndarray, n_data: int) -> np.ndarray:
    mu = np.asarray(f_mu, dtype=np.complex128)
    if mu.ndim == 1:
        return np.broadcast_to(mu, (n_data, mu.shape[0])).copy()
    return mu


def phased_observed(i_obs: np.ndarray, phase: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """``|F| exp(iφ)`` with negative intensities floored at 0. Missing entries are 0."""
    amp = np.sqrt(np.maximum(np.asarray(i_obs, dtype=np.float64), 0.0))
    field = amp * np.exp(1j * np.asarray(phase, dtype=np.float64))
    return np.where(np.asarray(mask, dtype=bool), field, 0.0).astype(np.complex128)


def known_f_target(
    f_obs: np.ndarray,
    f_mu: np.ndarray,
    f_var: Optional[np.ndarray],
    d_refl: np.ndarray,
) -> np.ndarray:
    """``(F_obs - F_var) / D - F_mu``."""
    var = 0.0 if f_var is None else np.asarray(f_var, dtype=np.complex128)
    scale = np.maximum(np.asarray(d_refl, dtype=np.float64), 1e-6)
    mu = _broadcast_model(f_mu, f_obs.shape[0])
    return (np.asarray(f_obs, dtype=np.complex128) - var) / scale - mu


def fit_known_f(
    f_obs: np.ndarray,
    f_mu: np.ndarray,
    mask: np.ndarray,
    *,
    f_var: Optional[np.ndarray] = None,
    rank: int = 1,
    n_sweeps: int = 8,
    ridge: float = 1e-8,
    roles: Optional[np.ndarray] = None,
    shell_index: Optional[np.ndarray] = None,
) -> EnsembleResult:
    """No E-step. ``f_obs`` is already the phased complex observation."""
    observed = np.asarray(mask, dtype=bool)
    n_data, n_h = observed.shape
    d_table = np.ones((n_data, 1), dtype=np.float64)
    sigma_table = np.ones((n_data, 1), dtype=np.float64)
    shells = _shells_or_one(shell_index, n_h)
    d_refl = np.ones((n_data, n_h), dtype=np.float64)
    sigma_refl = np.ones((n_data, n_h), dtype=np.float64)
    weights = work_weights(observed, d_refl, sigma_refl, roles, include_calibration=False)
    target = known_f_target(f_obs, f_mu, f_var, d_refl)
    delta0, modes, scores, cov = als_known_f(target, weights, rank, n_sweeps, ridge)
    history = np.array([_known_f_loglik(target, weights, delta0, modes, scores)], dtype=np.float64)
    return EnsembleResult(
        delta0=delta0,
        modes=modes,
        scores=scores,
        score_cov=cov,
        sigma2=sigma_table,
        d=d_table,
        shell_index=shells,
        loglik_history=history,
        fom=observed.astype(np.float64),
        stats={"likelihood": "known_f", "k": int(modes.shape[0])},
        mean_f=np.asarray(f_obs, dtype=np.complex128),
    )


def _known_f_loglik(
    target: np.ndarray,
    weights: np.ndarray,
    delta0: np.ndarray,
    modes: np.ndarray,
    scores: np.ndarray,
) -> float:
    pred = reconstruction(delta0, modes, scores)
    resid = np.asarray(target, dtype=np.complex128) - pred
    w = np.asarray(weights, dtype=np.float64)
    rss = float(np.sum(w * np.abs(resid) ** 2))
    return -0.5 * rss


def _mu_field(
    f_mu: np.ndarray,
    f_var: np.ndarray,
    delta0: np.ndarray,
    modes: np.ndarray,
    scores: np.ndarray,
    d_refl: np.ndarray,
) -> np.ndarray:
    ens = _broadcast_model(f_mu, scores.shape[0] if scores.ndim == 2 else f_var.shape[0])
    ens = ens + reconstruction(delta0, modes, scores)
    return np.asarray(f_var, dtype=np.complex128) + np.asarray(d_refl, dtype=np.float64) * ens


def _e_step(
    mu: np.ndarray,
    sigma_refl: np.ndarray,
    i_obs: np.ndarray,
    sig_i: np.ndarray,
    epsilon: np.ndarray,
    sigma_wilson: np.ndarray,
    centric: np.ndarray,
    mask: np.ndarray,
    nu: Optional[float],
    hl: Optional[np.ndarray],
    hl_weight: float,
) -> dict[str, np.ndarray]:
    return posterior_f_moments(
        mu,
        sigma_refl,
        i_obs,
        sig_i,
        epsilon,
        sigma_wilson,
        centric,
        nu=nu,
        hl=hl,
        hl_weight=hl_weight,
        mask=mask,
    )


def init_from_difference_maps(
    i_obs: np.ndarray,
    sig_i: np.ndarray,
    mask: np.ndarray,
    f_mu: np.ndarray,
    epsilon: np.ndarray,
    centric: np.ndarray,
    sigma_wilson: np.ndarray,
    d_refl: np.ndarray,
    sigma_refl: np.ndarray,
    *,
    f_var: Optional[np.ndarray] = None,
    rank: int = 1,
    nu: Optional[float] = None,
    roles: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One E-step at the reference (z = 0) and an SVD of the difference coefficients.

    K is an argument. It is not chosen from the singular-value spectrum.
    """
    n_data, n_h = np.asarray(i_obs).shape
    var = np.zeros((n_data, n_h), dtype=np.complex128) if f_var is None else np.asarray(f_var, dtype=np.complex128)
    scores = np.zeros((n_data, 0), dtype=np.float64)
    modes = np.zeros((0, n_h), dtype=np.complex128)
    delta0 = np.zeros(n_h, dtype=np.complex128)
    mu = _mu_field(f_mu, var, delta0, modes, scores, d_refl)
    moments = _e_step(mu, sigma_refl, i_obs, sig_i, epsilon, sigma_wilson, centric, mask, nu, None, 0.0)
    weights = work_weights(mask, d_refl, sigma_refl, roles, include_calibration=False)
    diff = difference_coefficients(moments["mean_f"], mu, weights > 0)
    target = known_f_target(diff + mu, f_mu, var, d_refl)
    # difference_coefficients is E[F]-μ; the ALS target is (E[F]-F_var)/D - F_mu.
    target = (moments["mean_f"] - var) / np.maximum(d_refl, 1e-6) - _broadcast_model(f_mu, n_data)
    target = np.where(weights > 0, target, 0.0)
    return initialize_from_residuals(target, weights, rank)


def _maybe_project(
    hkl: Optional[np.ndarray],
    cell: Optional[tuple[float, ...]],
    delta0: np.ndarray,
    modes: np.ndarray,
    scores: np.ndarray,
    projection: Optional[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    if not projection or not projection.get("enabled"):
        return delta0, modes
    if hkl is None or cell is None:
        return delta0, modes
    from phridge.r.generators.perturbation_modes.projection import project_modes

    return project_modes(
        hkl,
        cell,
        delta0,
        modes,
        scores,
        grid=tuple(projection.get("grid", (16, 16, 16))),
        support=bool(projection.get("support", False)),
        support_mask=projection.get("support_mask"),
        support_dilate=int(projection.get("support_dilate", 0)),
        smooth_sigma=float(projection.get("smooth_sigma", 0.0)),
        zero_f000=bool(projection.get("zero_f000", True)),
        positivity=bool(projection.get("positivity", False)),
    )


def em_fit(
    i_obs: np.ndarray,
    sig_i: np.ndarray,
    mask: np.ndarray,
    f_mu: np.ndarray,
    epsilon: np.ndarray,
    centric: np.ndarray,
    sigma_wilson: np.ndarray,
    *,
    f_var: Optional[np.ndarray] = None,
    rank: int = 1,
    n_sweeps: int = 20,
    ridge: float = 1e-8,
    shell_index: Optional[np.ndarray] = None,
    roles: Optional[np.ndarray] = None,
    hl: Optional[np.ndarray] = None,
    hl_weight: float = 0.0,
    nu: Optional[float] = None,
    sigma_a0: float = 0.9,
    dof_correct: bool = False,
    hkl: Optional[np.ndarray] = None,
    cell: Optional[tuple[float, ...]] = None,
    projection: Optional[dict[str, Any]] = None,
    d_init: Optional[np.ndarray] = None,
    sigma2_init: Optional[np.ndarray] = None,
) -> EnsembleResult:
    """Generalised EM against the marginal intensity likelihood."""
    observed = np.asarray(mask, dtype=bool)
    n_data, n_h = observed.shape
    shells = _shells_or_one(shell_index, n_h)
    n_shell = int(shells.max()) + 1 if np.any(shells >= 0) else 1
    shells = np.where(shells < 0, 0, shells)
    var = np.zeros((n_data, n_h), dtype=np.complex128) if f_var is None else np.asarray(f_var, dtype=np.complex128)
    wilson = np.asarray(sigma_wilson, dtype=np.float64)
    if wilson.ndim == 1 and wilson.shape[0] == n_data:
        wilson = np.broadcast_to(wilson[:, None], (n_data, n_h)).copy()
    elif wilson.ndim == 0 or wilson.size == 1:
        wilson = np.full((n_data, n_h), float(np.asarray(wilson).reshape(-1)[0]))
    eps = np.asarray(epsilon, dtype=np.float64)
    if eps.ndim == 1:
        eps = np.broadcast_to(eps, (n_data, n_h)).copy()
    cen = np.asarray(centric, dtype=bool)
    if cen.ndim == 1:
        cen = np.broadcast_to(cen, (n_data, n_h)).copy()
    if d_init is None:
        d_table = np.full((n_data, n_shell), float(sigma_a0), dtype=np.float64)
    else:
        d_table = np.asarray(d_init, dtype=np.float64)
        if d_table.ndim == 1:
            d_table = np.broadcast_to(d_table, (n_data, n_shell)).copy()
    if sigma2_init is None:
        scale2 = eps * wilson
        sigma_table = np.mean((1.0 - float(sigma_a0) ** 2) * scale2, axis=1)
        sigma_table = np.broadcast_to(sigma_table[:, None], (n_data, n_shell)).copy()
    else:
        sigma_table = np.asarray(sigma2_init, dtype=np.float64)
        if sigma_table.ndim == 1:
            sigma_table = np.broadcast_to(sigma_table, (n_data, n_shell)).copy()
    d_refl = expand_shells(d_table, shells)
    sigma_refl = expand_shells(sigma_table, shells)
    delta0, modes, scores = init_from_difference_maps(
        i_obs,
        sig_i,
        observed,
        f_mu,
        eps,
        cen,
        wilson,
        d_refl,
        sigma_refl,
        f_var=var,
        rank=rank,
        nu=nu,
        roles=roles,
    )
    history: list[float] = []
    fom = np.zeros((n_data, n_h), dtype=np.float64)
    mean_f = np.zeros((n_data, n_h), dtype=np.complex128)
    loglik_unprojected = None
    loglik_projected = None
    cov = np.zeros((n_data, modes.shape[0], modes.shape[0]), dtype=np.float64)
    weight_hl = float(hl_weight) if hl is not None else 0.0
    for _sweep in range(int(n_sweeps)):
        mu = _mu_field(f_mu, var, delta0, modes, scores, d_refl)
        moments = _e_step(mu, sigma_refl, i_obs, sig_i, eps, wilson, cen, observed, nu, hl, weight_hl)
        train = moments["log_lik_hl"] if weight_hl > 0.0 and moments["log_lik_hl"] is not None else moments["log_lik"]
        history.append(float(np.sum(train)))
        fom = moments["fom"]
        mean_f = moments["mean_f"]
        weights = work_weights(observed, d_refl, sigma_refl, roles, include_calibration=False)
        target = (mean_f - var) / np.maximum(d_refl, 1e-6) - _broadcast_model(f_mu, n_data)
        target = np.where(weights > 0, target, 0.0)
        unprojected_ll = None
        delta0, modes = solve_modes_torch(target, scores, weights, ridge)
        scores, cov = solve_scores_torch(target, delta0, modes, weights, ridge)
        delta0, modes, scores = fix_gauge(delta0, modes, scores)
        if projection and projection.get("enabled"):
            mu_before = _mu_field(f_mu, var, delta0, modes, scores, d_refl)
            before = _e_step(mu_before, sigma_refl, i_obs, sig_i, eps, wilson, cen, observed, nu, hl, weight_hl)
            unprojected_ll = float(np.sum(before["log_lik"]))
            delta0, modes = _maybe_project(hkl, cell, delta0, modes, scores, projection)
            delta0, modes, scores = fix_gauge(delta0, modes, scores)
        calib = work_weights(observed, np.ones_like(d_refl), np.ones_like(sigma_refl), roles, include_calibration=True) > 0
        ens = _broadcast_model(f_mu, n_data) + reconstruction(delta0, modes, scores)
        d_table, sigma_table = update_shell_scales(
            mean_f,
            moments["mean_intensity"],
            ens,
            var,
            shells,
            calib,
            n_data,
            int(modes.shape[0]),
            dof_correct=dof_correct,
        )
        d_refl = expand_shells(d_table, shells)
        sigma_refl = expand_shells(sigma_table, shells)
        # Keep σ² inside the Wilson scale so σ_A stays in (0, 1).
        sigma_refl = np.minimum(sigma_refl, 0.95 * eps * wilson)
        for shell in range(n_shell):
            sel = shells == shell
            if np.any(sel):
                sigma_table[:, shell] = np.mean(sigma_refl[:, sel], axis=1)
        sigma_refl = expand_shells(sigma_table, shells)
        if unprojected_ll is not None:
            mu_after = _mu_field(f_mu, var, delta0, modes, scores, d_refl)
            after = _e_step(mu_after, sigma_refl, i_obs, sig_i, eps, wilson, cen, observed, nu, hl, weight_hl)
            loglik_unprojected = unprojected_ll
            loglik_projected = float(np.sum(after["log_lik"]))
    mu = _mu_field(f_mu, var, delta0, modes, scores, d_refl)
    final = _e_step(mu, sigma_refl, i_obs, sig_i, eps, wilson, cen, observed, nu, hl, weight_hl)
    train = final["log_lik_hl"] if weight_hl > 0.0 and final["log_lik_hl"] is not None else final["log_lik"]
    history.append(float(np.sum(train)))
    return EnsembleResult(
        delta0=delta0,
        modes=modes,
        scores=scores,
        score_cov=cov,
        sigma2=sigma_table,
        d=d_table,
        shell_index=shells.astype(np.int32),
        loglik_history=np.asarray(history, dtype=np.float64),
        fom=final["fom"],
        stats={
            "likelihood": "intensity",
            "k": int(modes.shape[0]),
            "dof_correct": bool(dof_correct),
            "hl_weight": weight_hl,
            "loglik_comparable_across_hl_weight": False,
        },
        mean_f=final["mean_f"],
        loglik_unprojected=loglik_unprojected,
        loglik_projected=loglik_projected,
    )


def negloglik_and_grad(
    delta0: np.ndarray,
    modes: np.ndarray,
    scores: np.ndarray,
    log_sigma2: np.ndarray,
    *,
    i_obs: np.ndarray,
    sig_i: np.ndarray,
    mask: np.ndarray,
    f_mu: np.ndarray,
    f_var: np.ndarray,
    epsilon: np.ndarray,
    centric: np.ndarray,
    sigma_wilson: np.ndarray,
    d_refl: np.ndarray,
    nu: Optional[float] = None,
) -> tuple[float, dict[str, np.ndarray]]:
    """Marginal negative log-likelihood and the Fisher gradient through μ and log σ²."""
    sigma_refl = np.exp(np.asarray(log_sigma2, dtype=np.float64))
    mu = _mu_field(f_mu, f_var, delta0, modes, scores, d_refl)
    moments = _e_step(mu, sigma_refl, i_obs, sig_i, epsilon, sigma_wilson, centric, mask, nu, None, 0.0)
    nll = float(-np.sum(moments["log_lik"]))
    g_mu = fisher_mu_gradient(moments["mean_f"], mu, sigma_refl)
    # d nll / d (Re μ, Im μ) = -g_mu. Chain rule into Δ0, modes, and z.
    d_real = np.asarray(d_refl, dtype=np.float64)
    g_delta0 = -np.sum(np.conj(g_mu) * d_real, axis=0)
    # Real gradient of nll w.r.t. complex parameters, Wirtinger-combined as d(nll)/d(re, im).
    # μ = F_var + D * (F_mu + Δ0 + z @ modes). ∂μ/∂Δ0 = D.
    grad_delta0 = -d_real * g_mu
    grad_delta0 = np.sum(grad_delta0, axis=0)
    loads = np.asarray(modes, dtype=np.complex128)
    z = np.asarray(scores, dtype=np.float64)
    if loads.shape[0] == 0:
        grad_modes = np.zeros_like(loads)
        grad_scores = np.zeros_like(z)
    else:
        # ∂μ/∂Δ_k = D * z_k, so the gradient w.r.t. the complex mode is conj? 
        # nll decreases when μ moves toward E[F], direction g_mu.
        # d nll / d Re(Δ_k,h) = sum_n -Re_part: use -D * z_nk * g_mu_nh
        grad_modes = -(z.T.astype(np.complex128) @ (g_mu * d_real))
        grad_scores = -np.real(np.conj(loads) @ (g_mu * d_real).T).T
    resid2 = (
        moments["mean_intensity"]
        - 2.0 * np.real(np.conj(mu) * moments["mean_f"])
        + np.abs(mu) ** 2
    )
    # d L / d log σ² = -1 + E[|F-μ|²]/σ², so d nll / d log σ² = 1 - that.
    grad_log_sigma = np.where(np.asarray(mask, dtype=bool), 1.0 - resid2 / np.maximum(sigma_refl, 1e-18), 0.0)
    _ = g_delta0
    return nll, {
        "delta0": np.sum(grad_delta0, axis=0) if False else grad_delta0,
        "modes": grad_modes,
        "scores": grad_scores,
        "log_sigma2": grad_log_sigma,
        "mu": -g_mu,
    }


def direct_fit(
    i_obs: np.ndarray,
    sig_i: np.ndarray,
    mask: np.ndarray,
    f_mu: np.ndarray,
    epsilon: np.ndarray,
    centric: np.ndarray,
    sigma_wilson: np.ndarray,
    *,
    f_var: Optional[np.ndarray] = None,
    rank: int = 1,
    max_iter: int = 8,
    shell_index: Optional[np.ndarray] = None,
    nu: Optional[float] = None,
    sigma_a0: float = 0.9,
    ridge: float = 1e-8,
) -> EnsembleResult:
    """L-BFGS on (Δ0, Δ, z, log σ²) using the Fisher gradient as ``.grad``."""
    import torch

    base = em_fit(
        i_obs,
        sig_i,
        mask,
        f_mu,
        epsilon,
        centric,
        sigma_wilson,
        f_var=f_var,
        rank=rank,
        n_sweeps=1,
        ridge=ridge,
        shell_index=shell_index,
        nu=nu,
        sigma_a0=sigma_a0,
        dof_correct=False,
    )
    n_data, n_h = np.asarray(i_obs).shape
    var = np.zeros((n_data, n_h), dtype=np.complex128) if f_var is None else np.asarray(f_var, dtype=np.complex128)
    d_refl = expand_shells(base.d, base.shell_index)
    sigma_refl = expand_shells(base.sigma2, base.shell_index)
    delta0_p = torch.nn.Parameter(torch.tensor(np.stack([base.delta0.real, base.delta0.imag]), dtype=torch.float64))
    modes_p = torch.nn.Parameter(
        torch.tensor(np.stack([base.modes.real, base.modes.imag], axis=0), dtype=torch.float64)
        if base.modes.size
        else torch.zeros((2, 0, n_h), dtype=torch.float64)
    )
    scores_p = torch.nn.Parameter(torch.tensor(base.scores, dtype=torch.float64))
    log_s_p = torch.nn.Parameter(torch.tensor(np.log(np.maximum(sigma_refl, 1e-6)), dtype=torch.float64))
    opt = torch.optim.LBFGS([delta0_p, modes_p, scores_p, log_s_p], lr=0.25, max_iter=2, line_search_fn="strong_wolfe")
    history = [float(base.loglik_history[-1])]

    def _unpack() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        delta = (delta0_p[0] + 1j * delta0_p[1]).detach().cpu().numpy()
        if modes_p.shape[1] == 0:
            modes = np.zeros((0, n_h), dtype=np.complex128)
        else:
            modes = (modes_p[0] + 1j * modes_p[1]).detach().cpu().numpy()
        return delta, modes, scores_p.detach().cpu().numpy(), log_s_p.detach().cpu().numpy()

    def closure() -> torch.Tensor:
        opt.zero_grad()
        delta, modes, scores, log_s = _unpack()
        _nll, grads = negloglik_and_grad(
            delta,
            modes,
            scores,
            log_s,
            i_obs=i_obs,
            sig_i=sig_i,
            mask=mask,
            f_mu=f_mu,
            f_var=var,
            epsilon=epsilon if np.asarray(epsilon).ndim == 2 else np.broadcast_to(epsilon, (n_data, n_h)),
            centric=centric if np.asarray(centric).ndim == 2 else np.broadcast_to(centric, (n_data, n_h)),
            sigma_wilson=sigma_wilson if np.asarray(sigma_wilson).ndim == 2 else np.broadcast_to(sigma_wilson, (n_data, n_h)),
            d_refl=d_refl,
            nu=nu,
        )
        delta0_p.grad = torch.tensor(np.stack([grads["delta0"].real, grads["delta0"].imag]), dtype=torch.float64)
        if modes_p.shape[1] == 0:
            modes_p.grad = torch.zeros_like(modes_p)
        else:
            modes_p.grad = torch.tensor(np.stack([grads["modes"].real, grads["modes"].imag]), dtype=torch.float64)
        scores_p.grad = torch.tensor(grads["scores"], dtype=torch.float64)
        log_s_p.grad = torch.tensor(grads["log_sigma2"], dtype=torch.float64)
        return torch.tensor(_nll, dtype=torch.float64)

    for _step in range(int(max_iter)):
        loss = opt.step(closure)
        history.append(float(-loss.detach()))
    delta, modes, scores, log_s = _unpack()
    delta, modes, scores = fix_gauge(delta, modes, scores)
    sigma_refl = np.exp(log_s)
    return EnsembleResult(
        delta0=delta,
        modes=modes,
        scores=scores,
        score_cov=base.score_cov,
        sigma2=sigma_refl,
        d=d_refl,
        shell_index=base.shell_index,
        loglik_history=np.asarray(history, dtype=np.float64),
        fom=base.fom,
        stats={"likelihood": "intensity", "route": "direct", "k": int(modes.shape[0])},
    )
