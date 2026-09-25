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
from typing import Any, Callable, Optional, Union

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
    """Discrete Golub–Welsch quadrature from samples ``(t, pdf)``.

    Hardened against underflow / non-finite weights: clamps norms, forces
    non-negative Jacobi β, and falls back to fewer nodes if ``eigh`` fails.
    """
    pdf = np.asarray(pdf, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    pdf = np.where(np.isfinite(pdf), np.maximum(pdf, 0.0), 0.0)
    s = float(pdf.sum())
    if not np.isfinite(s) or s <= 0.0:
        t0 = float(np.nanmean(t)) if np.any(np.isfinite(t)) else 0.0
        return np.array([t0], dtype=np.float64), np.array([1.0], dtype=np.float64)
    pdf = pdf / s
    n = max(1, int(n))

    def _build(n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
        alpha: list[float] = []
        beta: list[float] = []
        p_prev = np.zeros_like(t)
        p = np.ones_like(t)
        norm_prev = 1.0
        for k in range(n_nodes):
            norm = float(np.sum(pdf * p * p))
            if not np.isfinite(norm) or norm <= 1e-300:
                break
            num = float(np.sum(pdf * t * p * p))
            if not np.isfinite(num):
                break
            alpha.append(num / norm)
            if k > 0:
                beta.append(max(norm / max(norm_prev, 1e-300), 0.0))
            b_km1 = beta[-1] if k > 0 else 0.0
            p_next = (t - alpha[-1]) * p - b_km1 * p_prev
            if not np.all(np.isfinite(p_next)):
                break
            p_prev, p, norm_prev = p, p_next, norm
        n_use = len(alpha)
        if n_use <= 0:
            t0 = float(np.sum(pdf * t))
            return np.array([t0], dtype=np.float64), np.array([1.0], dtype=np.float64)
        if n_use == 1:
            return np.array([alpha[0]], dtype=np.float64), np.array([1.0], dtype=np.float64)
        beta_arr = np.maximum(np.asarray(beta[: n_use - 1], dtype=np.float64), 0.0)
        J = np.diag(alpha[:n_use]) + np.diag(np.sqrt(beta_arr), 1) + np.diag(np.sqrt(beta_arr), -1)
        if not np.all(np.isfinite(J)):
            raise np.linalg.LinAlgError("non-finite Jacobi matrix")
        ev, V = np.linalg.eigh(J)
        w = np.asarray(V[0] ** 2, dtype=np.float64)
        w = np.where(np.isfinite(w), np.maximum(w, 0.0), 0.0)
        w_sum = float(w.sum())
        if w_sum <= 0.0 or not np.isfinite(w_sum):
            raise np.linalg.LinAlgError("Golub-Welsch weights vanished")
        return ev, w / w_sum

    last_exc: Optional[Exception] = None
    for n_try in (n, max(1, n // 2), max(1, min(8, n)), 1):
        try:
            return _build(int(n_try))
        except Exception as exc:  # noqa: BLE001 — fall through to coarser rule
            last_exc = exc
            continue
    raise RuntimeError(f"_golub_welsch failed for n={n}: {last_exc}")


@lru_cache(maxsize=None)
def _loggamma_rule(nu: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss rule for the density of u = log(lambda), lambda ~ Gamma(nu/2, rate nu/2)."""
    nu_f = float(nu)
    if not np.isfinite(nu_f) or nu_f < 1.05:
        # Invalid / missing ν → near-Gaussian mixture (large ν)
        nu_f = 200.0
    n = max(1, int(n))
    # Large ν → λ→1 (u→0); single-node rule matches the normal-noise limit.
    if nu_f >= 199.0:
        return np.array([0.0], dtype=np.float64), np.array([1.0], dtype=np.float64)
    nu_f = min(nu_f, 198.0)
    # Concentrate the grid around the mode of log-Gamma as ν grows
    u_lo, u_hi = -14.0, 6.0
    if nu_f >= 40.0:
        u_lo, u_hi = -6.0, 3.0
    if nu_f >= 120.0:
        u_lo, u_hi = -3.0, 1.5
    u = np.linspace(u_lo, u_hi, 200001)
    lp = (nu_f / 2.0) * u - (nu_f / 2.0) * np.exp(u)
    lp = np.where(np.isfinite(lp), lp, -np.inf)
    m = float(np.max(lp))
    if not np.isfinite(m):
        return np.array([0.0], dtype=np.float64), np.array([1.0], dtype=np.float64)
    pdf = np.exp(lp - m)
    return _golub_welsch(u, pdf, n)


# ---------------------------------------------------------------- log integrands (torch)
def _ratio(x: Tensor) -> Tensor:
    return torch.special.i1e(x) / torch.special.i0e(x)


def log_normal_noise(I: Tensor, Zo: Tensor, sZ: Tensor) -> Tensor:
    return -(Zo - I) ** 2 / (2 * sZ**2) - 0.5 * torch.log(2 * math.pi * sZ**2)


def _acen_log(E: Tensor, Ec: Tensor, sA: Tensor, Zo: Tensor, sZ: Tensor, i0e: Tensor, x: Tensor, a: Tensor) -> Tensor:
    return (
        torch.log(2 * E / a)
        - (E**2 + sA**2 * Ec**2) / a
        + x
        + torch.log(i0e)
        + log_normal_noise(E**2, Zo, sZ)
    )


def acen_E_log(E: Tensor, Ec: Tensor, sA: Tensor, Zo: Tensor, sZ: Tensor) -> Tensor:
    """Acentric log-integrand in E (no derivatives; quadrature only)."""
    a = 1 - sA**2
    x = (2 * sA * Ec / a) * E
    return _acen_log(E, Ec, sA, Zo, sZ, torch.special.i0e(x), x, a)


def acen_E(
    E: Tensor, Ec: Tensor, sA: Tensor, Zo: Tensor, sZ: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Acentric log-integrand in E with first and second derivatives."""
    a = 1 - sA**2
    c = 2 * sA * Ec / a
    x = c * E
    i0e = torch.special.i0e(x)
    r = torch.special.i1e(x) / i0e
    lg = _acen_log(E, Ec, sA, Zo, sZ, i0e, x, a)
    g = 1 / E - 2 * E / a + c * r + 2 * E * (Zo - E**2) / sZ**2
    drdx = 1 - torch.where(x > 0, r / x.clamp_min(1e-300), torch.full_like(x, 0.5)) - r**2
    H = -1 / E**2 - 2 / a + c**2 * drdx + (2 * (Zo - E**2) - 4 * E**2) / sZ**2
    return lg, g, H


def cen_E_log(E: Tensor, Ec: Tensor, sA: Tensor, Zo: Tensor, sZ: Tensor) -> Tensor:
    """Centric log-integrand in E (no derivatives; quadrature only)."""
    a = 1 - sA**2
    kE = (sA * Ec / a) * E
    return (
        0.5 * torch.log(2 / (math.pi * a))
        - (E**2 + sA**2 * Ec**2) / (2 * a)
        + kE.abs()
        + torch.log1p(torch.exp(-2 * kE.abs()))
        - math.log(2.0)
        + log_normal_noise(E**2, Zo, sZ)
    )


def cen_E(
    E: Tensor, Ec: Tensor, sA: Tensor, Zo: Tensor, sZ: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Centric (Woolfson) log-integrand in E with derivatives."""
    a = 1 - sA**2
    k = sA * Ec / a
    kE = k * E
    lg = cen_E_log(E, Ec, sA, Zo, sZ)
    t = torch.tanh(kE)
    g = -E / a + k * t + 2 * E * (Zo - E**2) / sZ**2
    H = -1 / a + k**2 * (1 - t**2) + (2 * (Zo - E**2) - 4 * E**2) / sZ**2
    return lg, g, H


def acen_I_log(I: Tensor, Ec: Tensor, sA: Tensor, Zo: Tensor, sZ: Tensor) -> Tensor:
    """Acentric log-integrand in I = E^2 (no derivatives; quadrature only)."""
    a = 1 - sA**2
    s = torch.sqrt(I)
    x = (2 * sA * Ec / a) * s
    return (
        -torch.log(a)
        - (I + sA**2 * Ec**2) / a
        + x
        + torch.log(torch.special.i0e(x))
        + log_normal_noise(I, Zo, sZ)
    )


def acen_I(
    I: Tensor, Ec: Tensor, sA: Tensor, Zo: Tensor, sZ: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Acentric log-integrand in I = E^2 (dI measure) with derivatives."""
    a = 1 - sA**2
    c = 2 * sA * Ec / a
    s = torch.sqrt(I)
    x = c * s
    i0e = torch.special.i0e(x)
    r = torch.special.i1e(x) / i0e
    lg = (
        -torch.log(a)
        - (I + sA**2 * Ec**2) / a
        + x
        + torch.log(i0e)
        + log_normal_noise(I, Zo, sZ)
    )
    dxdI = c / (2 * s.clamp_min(1e-300))
    g = -1 / a + r * dxdI + (Zo - I) / sZ**2
    drdx = 1 - r / x.clamp_min(1e-300) - r**2
    H = drdx * dxdI**2 - r * c / (4 * s.clamp_min(1e-300) ** 3) - 1 / sZ**2
    return lg, g, H


# ---------------------------------------------------------------- Newton mode search
def _newton_step(x: Tensor, g: Tensor, H: Tensor, lower: float, tol: float) -> tuple[Tensor, Tensor]:
    step = torch.where(H < 0, -g / H, torch.sign(g) * 0.25 * (x - lower + 0.1))
    step = torch.maximum(step, -0.5 * (x - lower))
    step = torch.minimum(step, 2.0 * (x - lower) + 1.0)
    conv = step.abs() < tol * (1 + x.abs())
    return step, conv


def newton_mode(
    fun: LogIntegrand,
    x0: Tensor,
    lower: float = 0.0,
    iters: int = 12,
    tol: float = 1e-10,
) -> tuple[Tensor, Tensor, Tensor]:
    """Damped Newton on a log-integrand; keeps x > lower. Returns (x, H, iterations_used).

    ``fun`` may close over full-batch parameters, so this path always evaluates the
    whole vector. Prefer :func:`_newton_on` when the integrand takes explicit params.
    """
    x = x0.clone()
    active = torch.ones_like(x, dtype=torch.bool)
    used = torch.zeros_like(x, dtype=torch.int32)
    H = torch.zeros_like(x)
    for _ in range(iters):
        if not bool(active.any()):
            break
        _, g, H = fun(x)
        step, conv = _newton_step(x, g, H, lower, tol)
        used = used + (active & ~conv).to(used.dtype)
        x = torch.where(active, x + step, x)
        active = active & ~conv
    _, _, H = fun(x)
    return x, H, used


def _newton_on(
    integrand: Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]],
    x0: Tensor,
    Ec: Tensor,
    sA: Tensor,
    Zo: Tensor,
    sZ: Tensor,
    *,
    lower: float = 0.0,
    iters: int = 12,
    tol: float = 1e-10,
    compact_after: int = 4,
    peel_frac: float = 0.90,
    min_active: int = 256,
) -> tuple[Tensor, Tensor, Tensor]:
    """Newton that peels stragglers instead of waiting on the last row.

    Full-batch SIMD for ``compact_after`` steps (gather is not worth it while
    most rows are still moving). After that, when at least ``peel_frac`` of the
    *current* working set has converged — or the remainder is ``<= min_active``
    — gather the unconverged tail and continue only on those. Repeat until the
    list is empty or ``iters`` is hit. The window is tolerant of an imprecise
    centric mode, so a few leftovers at the cap are fine.

    Peeling on the first converged row would copy ~99% of the batch to drop 1%;
    waiting for a high converged fraction (or a small absolute tail) is cheaper.
    """
    if x0.numel() == 0:
        z = x0.clone()
        return z, z, torch.zeros(0, dtype=torch.int32, device=x0.device)
    x = x0.clone()
    used = torch.zeros(x.shape[0], dtype=torch.int32, device=x.device)
    H_out = torch.zeros_like(x)
    idx: Optional[Tensor] = None
    x_w, Ec_w, sA_w, Zo_w, sZ_w = x, Ec, sA, Zo, sZ
    for it in range(iters):
        _, g, H = integrand(x_w, Ec_w, sA_w, Zo_w, sZ_w)
        step, conv = _newton_step(x_w, g, H, lower, tol)
        still = ~conv
        if idx is None:
            used = used + still.to(used.dtype)
            x_w = x_w + step
            x = x_w
            H_out = H
        else:
            used[idx] = used[idx] + still.to(used.dtype)
            x_w = x_w + step
            x[idx] = x_w
            H_out[idx] = H
        if not bool(still.any()):
            break
        n_still = int(still.sum())
        n_work = x_w.shape[0]
        warmed = it + 1 >= compact_after
        tail = n_still <= min_active
        peeled = n_still <= n_work * (1.0 - peel_frac)
        if warmed and n_still < n_work and (tail or peeled):
            if idx is None:
                idx = still.nonzero(as_tuple=True)[0]
            else:
                idx = idx[still]
            x_w = x_w[still]
            Ec_w = Ec_w[still]
            sA_w = sA_w[still]
            Zo_w = Zo_w[still]
            sZ_w = sZ_w[still]
    return x, H_out, used


# ---------------------------------------------------------------- quadratures
_GAUSS_T: dict[tuple[str, int, torch.dtype, torch.device], tuple[Tensor, Tensor]] = {}


def _gauss_t(kind: str, n: int, ref: Tensor) -> tuple[Tensor, Tensor]:
    """Cached Gauss nodes/weights as tensors matching ``ref``'s dtype and device."""
    key = (kind, int(n), ref.dtype, ref.device)
    hit = _GAUSS_T.get(key)
    if hit is not None:
        return hit
    raw = _hermite(int(n)) if kind == "h" else _legendre(int(n))
    # CPU numpy → tensor, then .to(device): as_tensor(..., device=mps) is unreliable
    xt = torch.as_tensor(raw[0], dtype=ref.dtype).to(device=ref.device)
    wt = torch.as_tensor(raw[1], dtype=ref.dtype).to(device=ref.device)
    _GAUSS_T[key] = (xt, wt)
    return xt, wt


def _logsumexp_rows(terms: Tensor) -> Tensor:
    return torch.logsumexp(terms, dim=1)


def _gl_window_terms(
    fun_cols: Callable[[Tensor], Tensor], lo: Tensor, hi: Tensor, n_gl: int
) -> tuple[Tensor, Tensor]:
    """Gauss-Legendre nodes X (N, n) on [lo, hi] and log(W_j f(X_j)) so that log int f = logsumexp(terms)."""
    x, w = _gauss_t("l", n_gl, lo)
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
    x, w = _gauss_t("h", n, I0)
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


def _rice_start(Ec: Tensor, sA: Tensor, Zo: Tensor) -> Tensor:
    a = 1 - sA**2
    return torch.maximum(torch.sqrt(Zo.clamp_min(0.0)), torch.sqrt(sA**2 * Ec**2 + a / 2))


# Last Newton modes, keyed by (N, device, dtype) so a normal-noise batch and a
# Student-t (N × n_u) expansion can coexist. Refine reuses the same Miller order,
# so the next evaluate starts at last cycle's E0 / I0.
_WARM: dict[tuple[int, str, torch.dtype], dict[str, Tensor]] = {}


def clear_mode_cache() -> None:
    """Drop cached Newton modes (tests / new dataset)."""
    _WARM.clear()


def _warm_key(Ec: Tensor) -> tuple[int, str, torch.dtype]:
    return (int(Ec.shape[0]), str(Ec.device), Ec.dtype)


def _warm_x0(Ec: Tensor, sA: Tensor, Zo: Tensor, sZ: Tensor) -> tuple[Tensor, Tensor, int]:
    """Rice start, overwritten by the last mode where inputs have not jumped."""
    rice = _rice_start(Ec, sA, Zo)
    hit = _WARM.get(_warm_key(Ec))
    if hit is None:
        return rice, torch.zeros_like(rice), 0
    jump = (
        ((Ec - hit["Ec"]).abs() > (hit["Ec"].abs() + 0.2))
        | ((sA - hit["sA"]).abs() > 0.25)
        | ((sZ - hit["sZ"]).abs() > (hit["sZ"].abs() + 0.05))
        | ((Zo - hit["Zo"]).abs() > 2.0 * (hit["Zo"].abs() + hit["sZ"] + 0.05))
    )
    last_e = hit["E0"]
    last_i = hit["I0"]
    if last_e.device != Ec.device or last_e.dtype != Ec.dtype:
        last_e = last_e.to(device=Ec.device, dtype=Ec.dtype)
        last_i = last_i.to(device=Ec.device, dtype=Ec.dtype)
    use = (last_e > 1e-4) & ~jump
    n_use = int(use.sum())
    return torch.where(use, last_e, rice), torch.where(use & (last_i > 0), last_i, torch.zeros_like(rice)), n_use


def _store_warm(Ec: Tensor, sA: Tensor, Zo: Tensor, sZ: Tensor, E0: Tensor, I0: Tensor) -> None:
    _WARM[_warm_key(Ec)] = {
        "E0": E0.detach(),
        "I0": I0.detach(),
        "Ec": Ec.detach(),
        "sA": sA.detach(),
        "Zo": Zo.detach(),
        "sZ": sZ.detach(),
    }


def _grad_at(
    integrand: Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]],
    e_val: float,
    Ec: Tensor,
    sA: Tensor,
    Zo: Tensor,
    sZ: Tensor,
) -> Tensor:
    e = torch.full((Ec.shape[0],), e_val, dtype=Ec.dtype, device=Ec.device)
    return integrand(e, Ec, sA, Zo, sZ)[1]


def _newton_E_split(
    x0: Tensor,
    Ec: Tensor,
    sA: Tensor,
    Zo: Tensor,
    sZ: Tensor,
    centric: Tensor,
    *,
    iters_acen: int,
    iters_cen: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Independent Newton solves for the acentric and centric subsets."""
    E0 = torch.zeros_like(x0)
    H_E = torch.zeros_like(x0)
    it_E = torch.zeros(x0.shape[0], dtype=torch.int32, device=x0.device)
    ac, cen = ~centric, centric
    if bool(ac.any()):
        E0[ac], H_E[ac], it_E[ac] = _newton_on(
            acen_E, x0[ac], Ec[ac], sA[ac], Zo[ac], sZ[ac], iters=iters_acen
        )
    if bool(cen.any()):
        E0[cen], H_E[cen], it_E[cen] = _newton_on(
            cen_E, x0[cen], Ec[cen], sA[cen], Zo[cen], sZ[cen], iters=iters_cen
        )
    return E0, H_E, it_E


def _g_origin_split(
    Ec: Tensor, sA: Tensor, Zo: Tensor, sZ: Tensor, centric: Tensor, need: Tensor
) -> Tensor:
    """∂log f / ∂E at E=1e-4 on ``need``; +1 elsewhere (interior, not a boundary vote)."""
    g = torch.ones_like(Ec)
    if not bool(need.any()):
        return g
    ac = need & ~centric
    cen = need & centric
    if bool(ac.any()):
        g[ac] = _grad_at(acen_E, 1e-4, Ec[ac], sA[ac], Zo[ac], sZ[ac])
    if bool(cen.any()):
        g[cen] = _grad_at(cen_E, 1e-4, Ec[cen], sA[cen], Zo[cen], sZ[cen])
    return g


def _legendre_window(
    E0: Tensor,
    H_E: Tensor,
    Ec: Tensor,
    sA: Tensor,
    Zo: Tensor,
    sZ: Tensor,
    k_window: float,
    g_origin: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    a = 1 - sA**2
    sig_E = 1 / torch.sqrt((-H_E).clamp_min(1e-300))
    E_max = torch.minimum(
        torch.sqrt(Zo.clamp_min(0.0) + 9.0 * sZ),
        sA * Ec + 6.0 * torch.sqrt(a),
    )
    boundary = (H_E >= 0) | (g_origin < 0) | (E0 <= 1e-4) | (sig_E > E_max)
    lo = torch.where(boundary, torch.zeros_like(E0), (E0 - k_window * sig_E).clamp_min(0.0))
    hi_curv = torch.where(H_E < 0, torch.minimum(E_max, E0 + k_window * sig_E), E_max)
    hi_int = torch.minimum(E0 + k_window * sig_E, torch.maximum(E_max, E0 + 3.0 * sig_E))
    hi = torch.where(boundary, hi_curv, hi_int)
    hi = torch.where(hi <= lo, E_max, hi)
    lo = torch.where(hi <= lo, torch.zeros_like(lo), lo)
    return lo, hi, boundary


# ---------------------------------------------------------------- posterior mode
def posterior_mode_E(
    Ec: Tensor,
    sA: Tensor,
    Zo: Tensor,
    sZ: Tensor,
    centric: Union[Tensor, bool],
    *,
    iters: int = 20,
) -> tuple[Tensor, Tensor]:
    """Maximum a posteriori (MAP) normalized amplitude E_mode = argmax_E ln p(E | Zo, Ec).

    Returns ``(E_mode, H_mode)`` where E_mode >= 0 and H_mode is the negative second derivative
    (curvature) at the mode. If the mode lies at the origin (e.g. gradient at origin <= 0
    or non-negative curvature), E_mode is clipped to 0.0.
    """
    with torch.no_grad():
        Ec, sA, Zo, sZ = torch.broadcast_tensors(Ec, sA, Zo, sZ)
        centric_t = torch.as_tensor(centric, device=Ec.device).bool().expand_as(Ec)
        Ec_, sA_, Zo_, sZ_ = Ec.detach(), sA.detach(), Zo.detach(), sZ.detach()
        x0, _I_warm, _ = _warm_x0(Ec_, sA_, Zo_, sZ_)
        E0, H_E, _ = _newton_E_split(
            x0, Ec_, sA_, Zo_, sZ_, centric_t, iters_acen=min(iters, 12), iters_cen=iters
        )
        g_origin = _g_origin_split(Ec_, sA_, Zo_, sZ_, centric_t, torch.ones_like(centric_t))
        boundary = (H_E >= 0) | (g_origin <= 0) | (E0 <= 1e-4)
        E_mode = torch.where(boundary, torch.zeros_like(E0), E0)
        _store_warm(Ec_, sA_, Zo_, sZ_, E_mode, torch.zeros_like(E_mode))
        return E_mode, H_E


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
        Ec_, sA_, Zo_, sZ_ = (v if differentiate_window else v.detach() for v in (Ec, sA, Zo, sZ))
        x0, I_warm, n_warm = _warm_x0(Ec_, sA_, Zo_, sZ_)
        E0, H_E, it_E = _newton_E_split(
            x0, Ec_, sA_, Zo_, sZ_, centric, iters_acen=12, iters_cen=20
        )
        need_g0 = (E0 <= 0.05) | (H_E >= 0)
        g_origin = _g_origin_split(Ec_, sA_, Zo_, sZ_, centric, need_g0)
        lo, hi, boundary = _legendre_window(E0, H_E, Ec_, sA_, Zo_, sZ_, k_window, g_origin)
        strong_cand = (~centric) & (Zo_ / sZ_ >= snr_strong) & (~boundary)
        I0 = torch.zeros_like(x0)
        H_I = torch.ones_like(x0)
        if bool(strong_cand.any()):
            s0 = strong_cand
            I_start = (E0[s0] ** 2).clamp_min(1e-6)
            I_prev = I_warm[s0]
            I_start = torch.where(I_prev > 0, I_prev, I_start)
            I0[s0], H_I[s0], _ = _newton_on(
                acen_I,
                I_start,
                Ec_[s0],
                sA_[s0],
                Zo_[s0],
                sZ_[s0],
                iters=6,
                compact_after=2,
            )
        strong = strong_cand & (H_I < 0)
        weak = ~strong
        _store_warm(Ec_, sA_, Zo_, sZ_, E0, I0)

    # --- quadrature with gradient tracking on the integrand (log-only; no i1e / Hessian)
    J = max(int(n_legendre), int(n_hermite))
    nodes = torch.ones(Ec.shape[0], J, dtype=Ec.dtype, device=Ec.device)
    terms = torch.full((Ec.shape[0], J), -math.inf, dtype=Ec.dtype, device=Ec.device)
    if bool(weak.any()):
        w_ac = weak & ~centric
        w_cen = weak & centric
        if bool(w_ac.any()):
            wa = w_ac

            def cols_ac(X: Tensor) -> Tensor:
                return acen_E_log(X, col(Ec[wa]), col(sA[wa]), col(Zo[wa]), col(sZ[wa]))

            X, T = _gl_window_terms(cols_ac, lo[wa], hi[wa], n_legendre)
            nodes[wa, : n_legendre] = X
            terms[wa, : n_legendre] = T
        if bool(w_cen.any()):
            wc = w_cen

            def cols_cen(X: Tensor) -> Tensor:
                return cen_E_log(X, col(Ec[wc]), col(sA[wc]), col(Zo[wc]), col(sZ[wc]))

            X, T = _gl_window_terms(cols_cen, lo[wc], hi[wc], n_legendre)
            nodes[wc, : n_legendre] = X
            terms[wc, : n_legendre] = T
    if bool(strong.any()):
        s = strong

        def cols_I(X: Tensor) -> Tensor:
            return acen_I_log(X, col(Ec[s]), col(sA[s]), col(Zo[s]), col(sZ[s]))

        X, T = _aghq_I_terms(cols_I, I0[s], H_I[s], n_hermite)
        nodes[s, : n_hermite] = torch.sqrt(X)
        terms[s, : n_hermite] = T
    stats = {
        "n_strong": int(strong.sum()),
        "n_weak": int(weak.sum()),
        "n_boundary": int(boundary.sum()),
        "n_warm": n_warm,
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
