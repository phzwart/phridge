"""E-step moments in F units, on top of ``intensity_ll.maps.posterior_moments``.

Centric reflections use a two-point phase distribution. With the centric axis
rotated onto the real line and Hendrickson–Lattman projection ``A`` on that axis,

    w(φ=0)  ∝ exp(+κ + A),
    w(φ=π)  ∝ exp(-κ - A),
    κ = |μ| r / σ²_Δ     (half the acentric concentration),
    <e^{iφ}> = tanh(κ + A).

Acentric reflections keep the von Mises conditional phase,

    v(r) = κ(r) + (A + iB),   κ(r) = 2 r |μ| / σ²_Δ,
    <e^{iφ} | r> = e^{i arg v} I1(|v|) / I0(|v|),

and the amplitude nodes are reweighted by ``I0(|v|)/I0(κ)``. ``hl is None``
does not enter that branch: ``E_mean``, ``I_mean`` and ``log_lik`` stay on the
original ``mli`` path. The HL normaliser is ``log_lik_hl`` and is not comparable
across ``hl_weight`` values.

F and E are related by ``F = sqrt(ε Σ_W) E``, the same scale
``intensity_map_coefficients`` uses.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def fisher_mu_gradient(mean_f: np.ndarray, mu: np.ndarray, sigma2: np.ndarray) -> np.ndarray:
    """``dL/d(Re μ, Im μ) = 2(E[F] - μ) / σ²``, returned as one complex array."""
    resid = np.asarray(mean_f, dtype=np.complex128) - np.asarray(mu, dtype=np.complex128)
    scale = 2.0 / np.maximum(np.asarray(sigma2, dtype=np.float64), 1e-18)
    return resid * scale


def _scale(epsilon: np.ndarray, sigma_wilson: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(np.asarray(epsilon, dtype=np.float64) * np.asarray(sigma_wilson, dtype=np.float64), 1e-18))


def normalized_prior(
    mu: np.ndarray,
    sigma2: np.ndarray,
    epsilon: np.ndarray,
    sigma_wilson: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(Ec, sigma_A, phase, scale)`` matching a circular normal of variance ``sigma2``."""
    scale = _scale(epsilon, sigma_wilson)
    variance = np.asarray(sigma2, dtype=np.float64)
    ratio = np.clip(variance / np.maximum(scale**2, 1e-18), 1e-4, 1.0 - 1e-4)
    sigma_a = np.sqrt(1.0 - ratio)
    mu_c = np.asarray(mu, dtype=np.complex128)
    amplitude = np.abs(mu_c)
    phase = np.ones_like(mu_c)
    nonzero = amplitude > 1e-12
    phase = np.where(nonzero, mu_c / np.maximum(amplitude, 1e-12), 1.0 + 0j)
    ec = amplitude / (scale * sigma_a)
    return ec, sigma_a, phase, scale


def posterior_f_moments(
    mu: np.ndarray,
    sigma2: np.ndarray,
    i_obs: np.ndarray,
    sig_i: np.ndarray,
    epsilon: np.ndarray,
    sigma_wilson: np.ndarray,
    centric: np.ndarray,
    *,
    nu: Optional[float] = None,
    hl: Optional[np.ndarray] = None,
    hl_weight: float = 1.0,
    mask: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """Marginal intensity moments. ``mean_f`` is ``E[F | I]`` in F units."""
    import torch

    from phridge.contrib.intensity_ll.maps import posterior_moments

    mu_c = np.asarray(mu, dtype=np.complex128)
    flat = mu_c.reshape(-1)
    ec, sigma_a, phase, scale = normalized_prior(flat, np.asarray(sigma2).reshape(-1), np.asarray(epsilon).reshape(-1), np.asarray(sigma_wilson).reshape(-1))
    zo = np.asarray(i_obs, dtype=np.float64).reshape(-1) / np.maximum(scale**2, 1e-18)
    sz = np.asarray(sig_i, dtype=np.float64).reshape(-1) / np.maximum(scale**2, 1e-18)
    sz = np.maximum(sz, 1e-6)
    cen = np.asarray(centric, dtype=bool).reshape(-1)
    if cen.size == 1 and flat.size > 1:
        cen = np.broadcast_to(cen, flat.shape)
    hl_arg = None
    if hl is not None and float(hl_weight) != 0.0:
        hl_lab = np.asarray(hl, dtype=np.complex128).reshape(-1) * float(hl_weight)
        hl_arg = torch.as_tensor(hl_lab * np.conj(phase))
    post = posterior_moments(
        torch.as_tensor(ec, dtype=torch.float64),
        torch.as_tensor(sigma_a, dtype=torch.float64),
        torch.as_tensor(zo, dtype=torch.float64),
        torch.as_tensor(sz, dtype=torch.float64),
        torch.as_tensor(cen),
        None if nu is None else float(nu),
        hl=hl_arg,
    )
    e_complex = post.e_complex.detach().cpu().numpy().astype(np.complex128)
    mean_f = (scale * e_complex * phase).reshape(mu_c.shape)
    mean_i = (scale**2 * post.I_mean.detach().cpu().numpy()).reshape(mu_c.shape).astype(np.float64)
    log_lik = post.log_lik.detach().cpu().numpy().astype(np.float64).reshape(mu_c.shape)
    log_lik_hl = None
    if post.log_lik_hl is not None:
        log_lik_hl = post.log_lik_hl.detach().cpu().numpy().astype(np.float64).reshape(mu_c.shape)
    fom = post.fom.detach().cpu().numpy().astype(np.float64).reshape(mu_c.shape)
    if mask is not None:
        keep = np.asarray(mask, dtype=bool)
        log_lik = np.where(keep, log_lik, 0.0)
        if log_lik_hl is not None:
            log_lik_hl = np.where(keep, log_lik_hl, 0.0)
        mean_f = np.where(keep, mean_f, 0.0)
        mean_i = np.where(keep, mean_i, 0.0)
    return {
        "mean_f": mean_f,
        "mean_intensity": mean_i,
        "log_lik": log_lik,
        "log_lik_hl": log_lik_hl,
        "fom": fom,
        "scale": scale.reshape(mu_c.shape),
        "sigma_a": sigma_a.reshape(mu_c.shape),
        "ec": ec.reshape(mu_c.shape),
    }


def difference_coefficients(mean_f: np.ndarray, mu: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """σ_A-weighted difference, here ``E[F] - μ`` on observed reflections."""
    diff = np.asarray(mean_f, dtype=np.complex128) - np.asarray(mu, dtype=np.complex128)
    return np.where(np.asarray(mask, dtype=bool), diff, 0.0)


def fixed_grid_posterior(
    ec: np.ndarray,
    sigma_a: float,
    zo: np.ndarray,
    sz: np.ndarray,
    *,
    n_nodes: int = 64,
) -> dict[str, np.ndarray]:
    """Acentric posterior on one fixed Gauss–Legendre window, for the gradient check.

    Nodes do not move with ``ec``, matching the frozen-node Fisher identity.
    """
    import torch

    from phridge.contrib.intensity_ll.mli import _legendre, acen_E_log

    ec_t = torch.as_tensor(np.asarray(ec, dtype=np.float64))
    zo_t = torch.as_tensor(np.asarray(zo, dtype=np.float64))
    sz_t = torch.as_tensor(np.asarray(sz, dtype=np.float64))
    sa = torch.full_like(ec_t, float(sigma_a))
    nodes_x, weights = _legendre(int(n_nodes))
    lo = torch.zeros_like(ec_t)
    hi = torch.full_like(ec_t, 6.0)
    half = 0.5 * (hi - lo)
    mid = 0.5 * (hi + lo)
    nodes = mid[:, None] + half[:, None] * torch.as_tensor(nodes_x, dtype=torch.float64)
    log_w = torch.log(torch.as_tensor(weights, dtype=torch.float64).clamp_min(1e-300)) + torch.log(half)[:, None]
    terms = acen_E_log(nodes, ec_t[:, None], sa[:, None], zo_t[:, None], sz_t[:, None]) + log_w
    log_lik = torch.logsumexp(terms, dim=1)
    posterior = torch.softmax(terms, dim=1)
    a = 1.0 - float(sigma_a) ** 2
    kappa = (2.0 * float(sigma_a) / a) * ec_t[:, None] * nodes
    fom = torch.special.i1e(kappa.abs()) / torch.special.i0e(kappa.abs())
    em = (posterior * nodes * fom).sum(dim=1)
    return {
        "log_lik": log_lik.detach().cpu().numpy().astype(np.float64),
        "em": em.detach().cpu().numpy().astype(np.float64),
        "nodes_fixed": True,
    }
