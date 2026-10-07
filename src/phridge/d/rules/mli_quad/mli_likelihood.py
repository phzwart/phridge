"""Intensity likelihood given quadrature terms."""

from __future__ import annotations

import math
from typing import Any, Optional, Union

import torch

from phridge.d.rules.mli_quad.mli_quadrature import (
    _loggamma_rule,
    _logsumexp_rows,
    quadrature_terms_normal,
)

Tensor = torch.Tensor

def log_likelihood_normal(
    Ec: Tensor,
    sA: Tensor,
    Zo: Tensor,
    sZ: Tensor,
    centric: Union[Tensor, bool],
    *,
    snr_strong: float = 5.0,
    n_hermite: int = 7,
    n_legendre: int = 24,
    k_window: float = 8.0,
    differentiate_window: bool = False,
    return_stats: bool = False,
) -> Union[Tensor, tuple[Tensor, dict[str, Any]]]:
    """log L per reflection with normal noise on intensities.

    Ec, sA, Zo, sZ: tensors (N,) (sA may be (N,) or scalar); centric: bool tensor (N,).
    Differentiable in Ec and sA.
    """
    _, terms, stats = quadrature_terms_normal(
        Ec,
        sA,
        Zo,
        sZ,
        centric,
        snr_strong=snr_strong,
        n_hermite=n_hermite,
        n_legendre=n_legendre,
        k_window=k_window,
        differentiate_window=differentiate_window,
    )
    out = _logsumexp_rows(terms)
    if return_stats:
        return out, stats
    return out


def quadrature_terms_t(
    Ec: Tensor,
    sA: Tensor,
    Zo: Tensor,
    sZ: Tensor,
    centric: Union[Tensor, bool],
    nu: Union[float, Tensor],
    *,
    n_u: int = 12,
    snr_strong: float = 5.0,
    n_hermite: int = 7,
    n_legendre: int = 24,
    k_window: float = 8.0,
    differentiate_window: bool = False,
) -> tuple[Tensor, Tensor, Tensor, dict[str, Any]]:
    """Student-t quadrature: one batched normal-noise solve per distinct ν.

    Returns ``(E_nodes, log_terms, u_nodes, stats)`` with shapes
    ``(N, n_u, J)``, ``(N, n_u, J)``, ``(N, n_u)``. Unused (u, E) slots stay
    at E = 1 and log_term = -inf. ``log L = logsumexp`` over the last two axes.
    """
    Ec, sA, Zo, sZ = torch.broadcast_tensors(Ec, sA, Zo, sZ)
    nu_t = torch.as_tensor(nu, dtype=Ec.dtype, device=Ec.device).expand_as(Ec)
    nu_t = torch.nan_to_num(nu_t, nan=200.0, posinf=200.0, neginf=2.5).clamp(2.05, 500.0)
    centric_t = torch.as_tensor(centric, device=Ec.device).bool().expand_as(Ec)
    J = max(int(n_legendre), int(n_hermite))
    n_u = int(n_u)
    nodes = torch.ones(Ec.shape[0], n_u, J, dtype=Ec.dtype, device=Ec.device)
    terms = torch.full((Ec.shape[0], n_u, J), -math.inf, dtype=Ec.dtype, device=Ec.device)
    u_nodes = torch.zeros(Ec.shape[0], n_u, dtype=Ec.dtype, device=Ec.device)
    stats: dict[str, Any] = {}
    q_kw = dict(
        snr_strong=snr_strong,
        n_hermite=n_hermite,
        n_legendre=n_legendre,
        k_window=k_window,
        differentiate_window=differentiate_window,
    )
    for nu_val in torch.unique(nu_t.detach()).tolist():
        sel = nu_t == nu_val
        if not bool(sel.any()):
            continue
        u, w = _loggamma_rule(float(nu_val), n_u)
        u_t = torch.as_tensor(u, dtype=Ec.dtype).to(device=Ec.device)
        w_t = torch.as_tensor(w, dtype=Ec.dtype).to(device=Ec.device)
        finite = torch.isfinite(w_t) & (w_t > 0)
        if not bool(finite.any()):
            continue
        u_t, w_t = u_t[finite], w_t[finite]
        k = int(u_t.numel())
        n_sel = int(sel.sum())
        Ec_e = Ec[sel].repeat_interleave(k)
        sA_e = sA[sel].repeat_interleave(k)
        Zo_e = Zo[sel].repeat_interleave(k)
        sZ_e = (sZ[sel].unsqueeze(1) * torch.exp(-0.5 * u_t)).reshape(-1)
        cen_e = centric_t[sel].repeat_interleave(k)
        X, T, stats = quadrature_terms_normal(Ec_e, sA_e, Zo_e, sZ_e, cen_e, **q_kw)
        # repeat_interleave is (n_sel, k) in C order, matching view below
        nodes[sel, :k, :] = X.view(n_sel, k, J)
        terms[sel, :k, :] = T.view(n_sel, k, J) + torch.log(w_t).view(1, k, 1)
        u_nodes[sel, :k] = u_t
    return nodes, terms, u_nodes, stats


def log_likelihood_t(
    Ec: Tensor,
    sA: Tensor,
    Zo: Tensor,
    sZ: Tensor,
    centric: Union[Tensor, bool],
    nu: Union[float, Tensor],
    *,
    n_u: int = 12,
    snr_strong: float = 5.0,
    n_hermite: int = 7,
    n_legendre: int = 24,
    k_window: float = 8.0,
    differentiate_window: bool = False,
) -> Tensor:
    """Student-t noise via the Gamma scale mixture, integrated in log(lambda).

    nu: float or (N,) tensor with few distinct values (grouped internally).
    """
    _, terms, _, _ = quadrature_terms_t(
        Ec,
        sA,
        Zo,
        sZ,
        centric,
        nu,
        n_u=n_u,
        snr_strong=snr_strong,
        n_hermite=n_hermite,
        n_legendre=n_legendre,
        k_window=k_window,
        differentiate_window=differentiate_window,
    )
    return _logsumexp_rows(terms.reshape(terms.shape[0], -1))


# ---------------------------------------------------------------- convenience: from raw data
def normalize(
    f_calc_abs: Tensor,
    i_obs: Tensor,
    sig_i: Tensor,
    epsilon: Tensor,
    sigma_wilson: Tensor,
    sigma_a: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Raw amplitudes/intensities -> normalized (E_C, sigma_A, Z_o, sigma_Z).

    sigma_wilson: per-reflection expected <|F|^2>/epsilon (Wilson scale for the bin).
    """
    denom = epsilon * sigma_wilson
    return f_calc_abs / torch.sqrt(denom), sigma_a, i_obs / denom, sig_i / denom
