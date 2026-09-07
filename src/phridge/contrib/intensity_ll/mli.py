"""Intensity-based crystallographic likelihood by adaptive quadrature — PyTorch.

Per reflection, in normalized units (E_C model amplitude, sigma_A, Z_o observed
normalized intensity, sigma_Z its s.d., optional Student-t nu):

    L = int_0^inf  f_Rice/Woolfson(E | E_C, sigma_A) * f_noise(Z_o | E^2, sigma_Z[, nu]) dE

Algorithm (all vectorized over reflections, GPU-ready, differentiable in E_C
and sigma_A through autograd):

  * Newton on the analytic log-integrand in E (unimodal; ~4 iterations,
    start = max(sqrt(max(Z_o,0)), Rice mode)) gives mode E0 and curvature.
  * strong acentric reflections (Z_o/sigma_Z >= snr_strong): Gauss-Hermite in
    I = E^2 about the Laplace point (7 nodes ~ 1e-6 in log L).
  * everything else (weak, negative, centric): Gauss-Legendre on the window
    [max(0, E0 - k sigma), E0 + k sigma] (24 nodes ~ 1e-5 or better).
  * Student-t noise = Gamma scale mixture of normals; the mixture is
    integrated in u = log(lambda) with a Gauss rule for the log-gamma
    weight, tabulated once per nu (10-16 nodes).
  * Bessel functions only through i0e / i1e; all sums via logsumexp.

The mode and window are computed without gradient tracking (detached): the
quadrature nodes are treated as fixed, so the returned gradient is the
gradient of the quadrature approximation with frozen nodes. With 24-node
Legendre windows the omitted term is of the order of the quadrature error.
Set differentiate_window=True to track them (roughly doubles cost).
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Callable, Union

import numpy as np
import torch

LOG2PI = math.log(2.0 * math.pi)

Tensor = torch.Tensor
LogIntegrand = Callable[[Tensor], tuple[Tensor, Tensor, Tensor]]


# ---------------------------------------------------------------- rules
@lru_cache(maxsize=None)
def _legendre(n: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.legendre.leggauss(n)
    return x, w


@lru_cache(maxsize=None)
def _hermite(n: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.hermite.hermgauss(n)  # weight exp(-x^2)
    return x, w


def _golub_welsch(t: np.ndarray, pdf: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    pdf = pdf / pdf.sum()
    alpha: list[float] = []
    beta: list[float] = []
    p_prev = np.zeros_like(t)
    p = np.ones_like(t)
    norm_prev = None
    for k in range(n):
        norm = np.sum(pdf * p * p)
        alpha.append(float(np.sum(pdf * t * p * p) / norm))
        if k > 0:
            beta.append(float(norm / norm_prev))
        p_next = (t - alpha[-1]) * p - (beta[-1] if k > 0 else 0.0) * p_prev
        p_prev, p, norm_prev = p, p_next, norm
    J = np.diag(alpha) + np.diag(np.sqrt(beta), 1) + np.diag(np.sqrt(beta), -1)
    ev, V = np.linalg.eigh(J)
    return ev, V[0] ** 2


@lru_cache(maxsize=None)
def _loggamma_rule(nu: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss rule for the density of u = log(lambda), lambda ~ Gamma(nu/2, rate nu/2)."""
    u = np.linspace(-14.0, 6.0, 200001)
    lp = (nu / 2) * u - (nu / 2) * np.exp(u)
    pdf = np.exp(lp - lp.max())
    return _golub_welsch(u, pdf, n)


# ---------------------------------------------------------------- log integrands (torch)
def _ratio(x: Tensor) -> Tensor:
    return torch.special.i1e(x) / torch.special.i0e(x)


def log_normal_noise(I: Tensor, Zo: Tensor, sZ: Tensor) -> Tensor:
    return -(Zo - I) ** 2 / (2 * sZ**2) - 0.5 * torch.log(2 * math.pi * sZ**2)


def acen_E(
    E: Tensor, Ec: Tensor, sA: Tensor, Zo: Tensor, sZ: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Acentric log-integrand in E with first and second derivatives."""
    a = 1 - sA**2
    c = 2 * sA * Ec / a
    x = c * E
    r = _ratio(x)
    lg = (
        torch.log(2 * E / a)
        - (E**2 + sA**2 * Ec**2) / a
        + x
        + torch.log(torch.special.i0e(x))
        + log_normal_noise(E**2, Zo, sZ)
    )
    g = 1 / E - 2 * E / a + c * r + 2 * E * (Zo - E**2) / sZ**2
    drdx = 1 - torch.where(x > 0, r / x.clamp_min(1e-300), torch.full_like(x, 0.5)) - r**2
    H = -1 / E**2 - 2 / a + c**2 * drdx + (2 * (Zo - E**2) - 4 * E**2) / sZ**2
    return lg, g, H


def cen_E(
    E: Tensor, Ec: Tensor, sA: Tensor, Zo: Tensor, sZ: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Centric (Woolfson) log-integrand in E with derivatives."""
    a = 1 - sA**2
    k = sA * Ec / a
    kE = k * E
    lg = (
        0.5 * torch.log(2 / (math.pi * a))
        - (E**2 + sA**2 * Ec**2) / (2 * a)
        + kE.abs()
        + torch.log1p(torch.exp(-2 * kE.abs()))
        - math.log(2.0)
        + log_normal_noise(E**2, Zo, sZ)
    )
    t = torch.tanh(kE)
    g = -E / a + k * t + 2 * E * (Zo - E**2) / sZ**2
    H = -1 / a + k**2 * (1 - t**2) + (2 * (Zo - E**2) - 4 * E**2) / sZ**2
    return lg, g, H


def acen_I(
    I: Tensor, Ec: Tensor, sA: Tensor, Zo: Tensor, sZ: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Acentric log-integrand in I = E^2 (dI measure) with derivatives."""
    a = 1 - sA**2
    c = 2 * sA * Ec / a
    s = torch.sqrt(I)
    x = c * s
    r = _ratio(x)
    lg = (
        -torch.log(a)
        - (I + sA**2 * Ec**2) / a
        + x
        + torch.log(torch.special.i0e(x))
        + log_normal_noise(I, Zo, sZ)
    )
    dxdI = c / (2 * s.clamp_min(1e-300))
    g = -1 / a + r * dxdI + (Zo - I) / sZ**2
    drdx = 1 - r / x.clamp_min(1e-300) - r**2
    H = drdx * dxdI**2 - r * c / (4 * s.clamp_min(1e-300) ** 3) - 1 / sZ**2
    return lg, g, H


# ---------------------------------------------------------------- Newton mode search
def newton_mode(
    fun: LogIntegrand,
    x0: Tensor,
    lower: float = 0.0,
    iters: int = 12,
    tol: float = 1e-10,
) -> tuple[Tensor, Tensor, Tensor]:
    """Damped Newton on a log-integrand; keeps x > lower. Returns (x, H, iterations_used)."""
    x = x0.clone()
    active = torch.ones_like(x, dtype=torch.bool)
    used = torch.zeros_like(x, dtype=torch.int32)
    for _ in range(iters):
        _, g, H = fun(x)
        step = torch.where(H < 0, -g / H, torch.sign(g) * 0.25 * (x - lower + 0.1))
        step = torch.maximum(step, -0.5 * (x - lower))
        step = torch.minimum(step, 2.0 * (x - lower) + 1.0)
        conv = step.abs() < tol * (1 + x.abs())
        used = used + (active & ~conv).to(used.dtype)
        x = torch.where(active, x + step, x)
        active = active & ~conv
        if not bool(active.any()):
            break
    _, g, H = fun(x)
    return x, H, used


# ---------------------------------------------------------------- quadratures
def _logsumexp_rows(terms: Tensor) -> Tensor:
    return torch.logsumexp(terms, dim=1)


def _gl_window_terms(
    fun_cols: Callable[[Tensor], Tensor], lo: Tensor, hi: Tensor, n_gl: int
) -> tuple[Tensor, Tensor]:
    """Gauss-Legendre nodes X (N, n) on [lo, hi] and log(W_j f(X_j)) so that log int f = logsumexp(terms)."""
    x, w = _legendre(n_gl)
    x = torch.as_tensor(x, dtype=lo.dtype, device=lo.device)
    w = torch.as_tensor(w, dtype=lo.dtype, device=lo.device)
    half = 0.5 * (hi - lo)
    X = lo[:, None] + half[:, None] * (x[None, :] + 1.0)
    logW = torch.log(half)[:, None] + torch.log(w)[None, :]
    return X, logW + fun_cols(X)


def _gl_window(fun_cols: Callable[[Tensor], Tensor], lo: Tensor, hi: Tensor, n_gl: int) -> Tensor:
    return _logsumexp_rows(_gl_window_terms(fun_cols, lo, hi, n_gl)[1])


def _aghq_I_terms(
    fun_cols: Callable[[Tensor], Tensor], I0: Tensor, H: Tensor, n: int
) -> tuple[Tensor, Tensor]:
    """Adaptive Gauss-Hermite nodes X (N, n) in I about the Laplace point and log-terms (dI measure)."""
    x, w = _hermite(n)
    x = torch.as_tensor(x, dtype=I0.dtype, device=I0.device)
    w = torch.as_tensor(w, dtype=I0.dtype, device=I0.device)
    sig = 1 / torch.sqrt((-H).clamp_min(1e-300))
    X = I0[:, None] + math.sqrt(2.0) * sig[:, None] * x[None, :]
    X = X.clamp_min(1e-300)
    # int f = sig*sqrt2 * sum w_j f(X_j) e^{x_j^2}
    terms = (
        torch.log(w)[None, :]
        + x[None, :] ** 2
        + fun_cols(X)
        + torch.log(math.sqrt(2.0) * sig)[:, None]
    )
    return X, terms


def _aghq_I(fun_cols: Callable[[Tensor], Tensor], I0: Tensor, H: Tensor, n: int) -> Tensor:
    return _logsumexp_rows(_aghq_I_terms(fun_cols, I0, H, n)[1])


# ---------------------------------------------------------------- main evaluator
def quadrature_terms_normal(
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
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """Quadrature nodes and log-terms of the intensity likelihood with normal noise.

    Returns ``(E_nodes, log_terms, stats)`` with shapes (N, J), J = max(n_legendre, n_hermite):
    ``log L_h = logsumexp_j log_terms[h, j]`` and, since every term is W_j x (prior x noise)
    at the node, ``softmax_j(log_terms)`` is the posterior of the true amplitude on the nodes
    (strong reflections are integrated in I; their nodes are reported as E = sqrt(I) and the
    dI measure is already inside the terms, so posterior averages of any f(E) are the weighted
    node sums). Unused columns carry E = 1 and log_term = -inf.
    """
    Ec, sA, Zo, sZ = torch.broadcast_tensors(Ec, sA, Zo, sZ)
    centric = torch.as_tensor(centric, device=Ec.device).bool().expand_as(Ec)
    col = lambda v: v[:, None]  # noqa: E731

    ctx = torch.enable_grad() if differentiate_window else torch.no_grad()
    with ctx:
        # Newton in E for everything (unimodal); start = max(observed amplitude, Rice mode)
        Ec_, sA_, Zo_, sZ_ = (v if differentiate_window else v.detach() for v in (Ec, sA, Zo, sZ))
        a_ = 1 - sA_**2
        x0 = torch.maximum(torch.sqrt(Zo_.clamp_min(0.0)), torch.sqrt(sA_**2 * Ec_**2 + a_ / 2))

        def fun_E(E: Tensor) -> tuple[Tensor, Tensor, Tensor]:
            la, ga, Ha = acen_E(E, Ec_, sA_, Zo_, sZ_)
            lc, gc, Hc = cen_E(E, Ec_, sA_, Zo_, sZ_)
            return (
                torch.where(centric, lc, la),
                torch.where(centric, gc, ga),
                torch.where(centric, Hc, Ha),
            )

        E0, H_E, it_E = newton_mode(fun_E, x0, iters=20)
        sig_E = 1 / torch.sqrt((-H_E).clamp_min(1e-300))
        # boundary-dominated: gradient negative at the origin, or no negative curvature
        g_origin = fun_E(torch.full_like(E0, 1e-4))[1]
        E_max = torch.minimum(
            torch.sqrt(Zo_.clamp_min(0.0) + 9.0 * sZ_),
            sA_ * Ec_ + 6.0 * torch.sqrt(a_),
        )
        # boundary-dominated also when the Laplace width exceeds the physical support
        # (flat-topped centric integrands at the origin): integrate [0, E_max] instead.
        boundary = (H_E >= 0) | (g_origin < 0) | (E0 <= 1e-4) | (sig_E > E_max)
        lo = torch.where(boundary, torch.zeros_like(E0), (E0 - k_window * sig_E).clamp_min(0.0))
        hi_curv = torch.where(H_E < 0, torch.minimum(E_max, E0 + k_window * sig_E), E_max)
        # interior: +/- k sigma window, capped by the physical cutoff but never below E0 + 3 sigma
        hi_int = torch.minimum(E0 + k_window * sig_E, torch.maximum(E_max, E0 + 3.0 * sig_E))
        hi = torch.where(boundary, hi_curv, hi_int)
        hi = torch.where(hi <= lo, E_max, hi)
        lo = torch.where(hi <= lo, torch.zeros_like(lo), lo)

        strong = (~centric) & (Zo_ / sZ_ >= snr_strong) & (~boundary)
        # mode in I for the strong branch: a couple of Newton steps from E0^2
        I0, H_I, it_I = newton_mode(
            lambda I: acen_I(I, Ec_, sA_, Zo_, sZ_),
            (E0**2).clamp_min(1e-6),
            iters=6,
        )
        strong = strong & (H_I < 0)

    # --- quadrature with gradient tracking on the integrand
    J = max(int(n_legendre), int(n_hermite))
    nodes = torch.ones(Ec.shape[0], J, dtype=Ec.dtype, device=Ec.device)
    terms = torch.full((Ec.shape[0], J), -math.inf, dtype=Ec.dtype, device=Ec.device)
    weak = ~strong
    if bool(weak.any()):
        w = weak

        def cols_E(X: Tensor) -> Tensor:
            la = acen_E(X, col(Ec[w]), col(sA[w]), col(Zo[w]), col(sZ[w]))[0]
            lc = cen_E(X, col(Ec[w]), col(sA[w]), col(Zo[w]), col(sZ[w]))[0]
            return torch.where(col(centric[w]), lc, la)

        X, T = _gl_window_terms(cols_E, lo[w], hi[w], n_legendre)
        nodes[w, : n_legendre] = X
        terms[w, : n_legendre] = T
    if bool(strong.any()):
        s = strong

        def cols_I(X: Tensor) -> Tensor:
            return acen_I(X, col(Ec[s]), col(sA[s]), col(Zo[s]), col(sZ[s]))[0]

        X, T = _aghq_I_terms(cols_I, I0[s], H_I[s], n_hermite)
        nodes[s, : n_hermite] = torch.sqrt(X)
        terms[s, : n_hermite] = T
    stats = {
        "n_strong": int(strong.sum()),
        "n_weak": int(weak.sum()),
        "n_boundary": int(boundary.sum()),
        "newton_iters_mean": float(it_E.float().mean()) if it_E.numel() else 0.0,
        "newton_iters_max": int(it_E.max()) if it_E.numel() else 0,
    }
    return nodes, terms, stats


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
    Ec, sA, Zo, sZ = torch.broadcast_tensors(Ec, sA, Zo, sZ)
    nu_t = torch.as_tensor(nu, dtype=Ec.dtype, device=Ec.device).expand_as(Ec)
    centric_t = torch.as_tensor(centric, device=Ec.device).bool().expand_as(Ec)
    out = torch.empty_like(Ec)
    for nu_val in torch.unique(nu_t.detach()).tolist():
        sel = nu_t == nu_val
        u, w = _loggamma_rule(float(nu_val), n_u)
        terms = []
        for uj, wj in zip(u, w):
            ll = log_likelihood_normal(
                Ec[sel],
                sA[sel],
                Zo[sel],
                sZ[sel] * math.exp(-uj / 2),
                centric_t[sel],
                snr_strong=snr_strong,
                n_hermite=n_hermite,
                n_legendre=n_legendre,
                k_window=k_window,
                differentiate_window=differentiate_window,
            )
            assert isinstance(ll, torch.Tensor)
            terms.append(ll + math.log(wj))
        out[sel] = torch.logsumexp(torch.stack(terms, dim=1), dim=1)
    return out


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
