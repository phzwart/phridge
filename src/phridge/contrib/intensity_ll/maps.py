"""Map coefficients from the intensity likelihood (``ml_i``) — posterior, not F_obs, based.

There is no |F_obs| in an intensity target. What replaces it is the posterior of the
true amplitude for each reflection,

    p(E | Z_o, E_C) ∝ p_prior(E | E_C, sigma_A) · p_noise(Z_o | E², sigma_Z[, nu]),

and every map coefficient below is a posterior average on the same quadrature nodes
that ``mli`` uses for log L (so the maps are exactly consistent with the gradient the
refinement sees). Fisher's identity gives the score in closed form:

    acentric:  d log L / d E_C = (2 sigma_A / a) ( <E m(E)> - sigma_A E_C ),   m = I1/I0(2 sigma_A E_C E / a)
    centric:   d log L / d E_C = (  sigma_A / a) ( <E m(E)> - sigma_A E_C ),   m = tanh( sigma_A E_C E / a)

with a = 1 - sigma_A² and <.> the posterior average. In F units (F = sqrt(eps Sigma) E,
D|F_c| = sqrt(eps Sigma) sigma_A E_C) the four maps are

    difference   (mFo-DFc analogue)      sqrt(eps Sigma) ( <E m> - sigma_A E_C ) e^{i phi_c}
    model        (2mFo-DFc analogue)     sqrt(eps Sigma) ( 2<E m> - sigma_A E_C ) e^{i phi_c}   [centric: <E m>]
    gradient     d log L / d F_c^*       (2 sigma_A / a) ( <E m> - sigma_A E_C ) / sqrt(eps Sigma) e^{i phi_c}
    newton       gradient / (curv_+ + mu)  the damped diagonal Newton step; curvature = -d² log L / d|F_c|²

For Student-t noise (Gamma scale mixture) the posterior also yields the robust weight
<lambda | Z_o> (~ (nu+1)/(nu + r²) for well-measured reflections) and the exact derivative
of log L with respect to nu,

    d log L / d nu = ½[log(nu/2) + 1 - psi(nu/2)] + ½ <log lambda - lambda>,

which is what a refinement of nu needs. See ``maps.md`` next to this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np
import torch
from pydantic import BaseModel, Field

from phridge.contrib.intensity_ll.mli import (
    _loggamma_rule,
    _ratio,
    normalize,
    posterior_mode_E,
    quadrature_terms_normal,
)
from phridge.sfcalc.targets.base import Observations

Tensor = torch.Tensor


class IntensityMapOptions(BaseModel):
    """Options for the ``ml_i_maps`` op (map-specific; the likelihood options live in the target spec)."""

    model_config = {"extra": "forbid"}

    newton_damping: float = Field(
        default=0.1,
        gt=0.0,
        description=(
            "Levenberg-style damping of the Newton map: gradient / (max(curvature, 0) + damping * median positive "
            "curvature). Bounds the re-inflation of reflections with vanishing or negative curvature."
        ),
    )
    include_free: bool = Field(default=True, description="If False, free reflections get zero coefficients.")
    differentiate_window: bool = Field(
        default=False,
        description="Track the Newton window when forming the curvature (roughly 2x cost).",
    )
    bin_size: Optional[int] = Field(
        default=None,
        ge=1,
        description="Equal-count resolution shells for S_post/S_prior bins; None skips shell arrays.",
    )


# ---------------------------------------------------------------- posterior on the nodes
@dataclass
class PosteriorMoments:
    """Per-reflection posterior summaries in normalized units (all tensors (N,))."""

    log_lik: Tensor
    E_mean: Tensor  # <E>
    I_mean: Tensor  # <E^2>
    Em_mean: Tensor  # <E m(E)>
    score: Tensor  # d log L / d E_C (closed form via Fisher's identity)
    fom: Tensor  # <E m(E)> / <E>
    lambda_mean: Tensor  # <lambda>  (ones for normal noise)
    log_lambda_mean: Tensor  # <log lambda>  (zeros for normal noise)
    d_loglik_d_nu: Optional[Tensor]  # None for normal noise
    stats: dict[str, Any] = field(default_factory=dict)


def _fom_at(E: Tensor, Ec: Tensor, sA: Tensor, centric: Tensor) -> Tensor:
    """Figure of merit m(E) = <cos(phase error)> given the true amplitude E."""
    a = 1 - sA**2
    x = sA * Ec * E / a
    m_acen = _ratio(2 * x)
    m_cen = torch.tanh(x)
    return torch.where(centric, m_cen, m_acen)


def _score_from_Em(Em: Tensor, Ec: Tensor, sA: Tensor, centric: Tensor) -> Tensor:
    a = 1 - sA**2
    pref = torch.where(centric, sA / a, 2 * sA / a)
    return pref * (Em - sA * Ec)


@dataclass
class PosteriorNodes:
    """Frozen quadrature nodes and the joint posterior on them.

    This is the single object both :func:`posterior_moments` and the Rice surrogate
    fit (:mod:`phridge.contrib.intensity_ll.surrogate`) read, so no consumer can
    drift onto a different set of nodes than the one the likelihood integrated on.
    """

    nodes: Tensor  # (N, n_u, J) amplitude nodes E
    terms: Tensor  # (N, n_u, J) log integrand + log quadrature weight
    p: Tensor  # (N, n_u, J) joint posterior on the (u, E) nodes
    u_nodes: Tensor  # (N, n_u) log(lambda) nodes; zeros for normal noise
    log_lik: Tensor  # (N,)
    Ec: Tensor  # (N,)
    sA: Tensor  # (N,)
    centric: Tensor  # (N,) bool
    stats: dict[str, Any]


def posterior_node_cache(
    Ec: Tensor,
    sA: Tensor,
    Zo: Tensor,
    sZ: Tensor,
    centric: Union[Tensor, bool],
    nu: Optional[Union[float, Tensor]] = None,
    *,
    n_u: int = 12,
    snr_strong: float = 5.0,
    n_hermite: int = 7,
    n_legendre: int = 24,
    k_window: float = 8.0,
) -> PosteriorNodes:
    """Build the frozen nodes and the joint posterior ``p(E, u | Z_o, E_C)`` on them.

    Same arguments as ``log_likelihood_normal`` / ``log_likelihood_t``. Everything is
    computed under ``no_grad`` (the window is frozen) and returned detached.
    """
    with torch.no_grad():
        Ec, sA, Zo, sZ = torch.broadcast_tensors(Ec, sA, Zo, sZ)
        centric_t = torch.as_tensor(centric, device=Ec.device).bool().expand_as(Ec)
        q_kw = dict(snr_strong=snr_strong, n_hermite=n_hermite, n_legendre=n_legendre, k_window=k_window)

        if nu is None:
            nodes, terms, stats = quadrature_terms_normal(Ec, sA, Zo, sZ, centric_t, **q_kw)
            nodes = nodes[:, None, :]  # (N, 1, J)
            terms = terms[:, None, :]
            u_nodes = torch.zeros(Ec.shape[0], 1, dtype=Ec.dtype, device=Ec.device)
        else:
            nu_t = torch.as_tensor(nu, dtype=Ec.dtype, device=Ec.device).expand_as(Ec)
            nu_t = torch.nan_to_num(nu_t, nan=200.0, posinf=200.0, neginf=2.5).clamp(2.05, 500.0)
            J = max(int(n_legendre), int(n_hermite))
            nodes = torch.ones(Ec.shape[0], n_u, J, dtype=Ec.dtype, device=Ec.device)
            terms = torch.full((Ec.shape[0], n_u, J), -math.inf, dtype=Ec.dtype, device=Ec.device)
            u_nodes = torch.zeros(Ec.shape[0], n_u, dtype=Ec.dtype, device=Ec.device)
            stats = {}
            for nu_val in torch.unique(nu_t).tolist():
                sel = nu_t == nu_val
                u, w = _loggamma_rule(float(nu_val), n_u)
                for k, (uk, wk) in enumerate(zip(u, w)):
                    X, T, st = quadrature_terms_normal(
                        Ec[sel], sA[sel], Zo[sel], sZ[sel] * math.exp(-uk / 2), centric_t[sel], **q_kw
                    )
                    nodes[sel, k, :] = X
                    terms[sel, k, :] = T + math.log(wk)
                    u_nodes[sel, k] = float(uk)
                stats = st  # last group; only used for diagnostics

        flat_terms = terms.reshape(terms.shape[0], -1)
        log_lik = torch.logsumexp(flat_terms, dim=1)
        p = torch.softmax(flat_terms, dim=1).reshape(terms.shape)  # joint posterior on (u, E) nodes

    return PosteriorNodes(
        nodes=nodes,
        terms=terms,
        p=p,
        u_nodes=u_nodes,
        log_lik=log_lik,
        Ec=Ec,
        sA=sA,
        centric=centric_t,
        stats=stats,
    )


def posterior_moments(
    Ec: Tensor,
    sA: Tensor,
    Zo: Tensor,
    sZ: Tensor,
    centric: Union[Tensor, bool],
    nu: Optional[Union[float, Tensor]] = None,
    *,
    n_u: int = 12,
    snr_strong: float = 5.0,
    n_hermite: int = 7,
    n_legendre: int = 24,
    k_window: float = 8.0,
    cache: Optional[PosteriorNodes] = None,
) -> PosteriorMoments:
    """Posterior averages of the true amplitude on the ``mli`` quadrature nodes.

    Same arguments as ``log_likelihood_normal`` / ``log_likelihood_t``. Nodes are frozen
    (no gradient through the window); the moments are returned detached. Pass ``cache``
    to reuse nodes already built by :func:`posterior_node_cache`.
    """
    with torch.no_grad():
        if cache is None:
            cache = posterior_node_cache(
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
            )
        Ec, sA, centric_t = cache.Ec, cache.sA, cache.centric
        nodes, p, u_nodes = cache.nodes, cache.p, cache.u_nodes
        stats = cache.stats
        log_lik = cache.log_lik

        E = nodes
        m = _fom_at(E, Ec[:, None, None], sA[:, None, None], centric_t[:, None, None])
        E_mean = (p * E).sum(dim=(1, 2))
        I_mean = (p * E**2).sum(dim=(1, 2))
        Em_mean = (p * E * m).sum(dim=(1, 2))
        p_u = p.sum(dim=2)  # (N, n_u) marginal over the scale mixture
        lambda_mean = (p_u * torch.exp(u_nodes)).sum(dim=1)
        log_lambda_mean = (p_u * u_nodes).sum(dim=1)
        score = _score_from_Em(Em_mean, Ec, sA, centric_t)
        fom = Em_mean / E_mean.clamp_min(1e-300)

        d_nu: Optional[Tensor] = None
        if nu is not None:
            nu_t = torch.as_tensor(nu, dtype=Ec.dtype, device=Ec.device).expand_as(Ec)
            half = nu_t / 2
            d_nu = 0.5 * (torch.log(half) + 1.0 - torch.special.digamma(half)) + 0.5 * (log_lambda_mean - lambda_mean)

    return PosteriorMoments(
        log_lik=log_lik,
        E_mean=E_mean,
        I_mean=I_mean,
        Em_mean=Em_mean,
        score=score,
        fom=fom,
        lambda_mean=lambda_mean,
        log_lambda_mean=log_lambda_mean,
        d_loglik_d_nu=d_nu,
        stats=stats,
    )


# ---------------------------------------------------------------- map coefficients in F units
@dataclass
class IntensityMapSet:
    """Per-reflection complex coefficients (N,) in F units plus the weights behind them."""

    difference: np.ndarray  # (<F m> - D|Fc|) e^{i phi_c}
    model: np.ndarray  # (2<F m> - D|Fc|) e^{i phi_c}; centric <F m> e^{i phi_c}
    gradient: np.ndarray  # d log L / d F_c^*
    newton: np.ndarray  # gradient / (curvature_+ + damping)
    fom: np.ndarray  # <F m> / <F>
    robust_weight: np.ndarray  # <lambda | Z_o>   (ones for normal noise)
    curvature: np.ndarray  # -d^2 log L / d|F_c|^2
    f_post: np.ndarray  # <F>  (robust, model-conditioned French-Wilson amplitude)
    d_loglik_d_nu: Optional[np.ndarray]
    stats: dict[str, Any]
    f_mode: Optional[np.ndarray] = None  # F_mode = scale * E_mode (posterior mode amplitude)


def _radial_curvature(target: Any, f_calc: Tensor, obs: Observations, differentiate_window: bool) -> Tensor:
    """-d^2 log L / d|F_c|^2 per reflection via double autograd on the target's own per_reflection."""
    amp = f_calc.abs().clone().requires_grad_(True)
    phase = f_calc / f_calc.abs().clamp(min=1e-300)
    prev = target.differentiate_window
    target.differentiate_window = bool(differentiate_window)
    try:
        nll = target.per_reflection(amp * phase, obs)  # -log L per reflection
    finally:
        target.differentiate_window = prev
    (g1,) = torch.autograd.grad(nll.sum(), amp, create_graph=True)
    (g2,) = torch.autograd.grad(g1.sum(), amp)
    return g2.detach()


def intensity_map_coefficients(
    target: Any,
    f_calc: Tensor,
    obs: Observations,
    options: Optional[IntensityMapOptions] = None,
) -> IntensityMapSet:
    """Map coefficients for an ``IntensityLogLikelihood`` target at the current F_calc.

    ``target`` supplies sigma_A / Sigma / sigma(I) / nu exactly as it does for the NLL, so the
    maps and the refinement target share one posterior.
    """
    opts = options or IntensityMapOptions()
    f_calc = f_calc.detach()
    fo = obs.data
    fc = f_calc.abs()
    eps = obs.epsilon if obs.epsilon is not None else torch.ones_like(fo)
    centric = obs.centric if obs.centric is not None else torch.zeros_like(fo, dtype=torch.bool)
    sA = target._sigma_a(obs)
    sW = target._sigma_wilson(obs)
    sig = target._sigma(obs)

    ok = (sA > 0) & (sA < 1.0 - 1e-6) & (sW > 0) & (sig > 0) & (eps > 0) & (fc > 0)
    if not opts.include_free and obs.r_free is not None:
        ok = ok & ~obs.r_free
    sA_s = torch.where(ok, sA, torch.full_like(sA, 0.5))
    sW_s = torch.where(ok, sW, torch.ones_like(sW))
    sig_s = torch.where(ok, sig, torch.ones_like(sig))
    eps_s = torch.where(ok, eps, torch.ones_like(eps))
    fc_s = torch.where(ok, fc, torch.ones_like(fc))
    fo_s = torch.where(ok, fo, torch.zeros_like(fo))

    Ec, sA_n, Zo, sZ = normalize(fc_s, fo_s, sig_s, eps_s, sW_s, sA_s)
    post = posterior_moments(
        Ec,
        sA_n,
        Zo,
        sZ,
        centric,
        target.nu,
        n_u=target.n_u,
        snr_strong=target.snr_strong,
        n_hermite=target.n_hermite,
        n_legendre=target.n_legendre,
        k_window=target.k_window,
    )
    scale = torch.sqrt(eps_s * sW_s)  # F = scale * E
    phase = f_calc / fc.clamp(min=1e-300)

    diff_E = post.Em_mean - sA_n * Ec
    model_E = torch.where(centric, post.Em_mean, 2 * post.Em_mean - sA_n * Ec)
    grad_F = post.score / scale  # d log L / d|F_c|
    curv_F = _radial_curvature(target, f_calc, obs, opts.differentiate_window)  # already in F units
    pos = curv_F[ok & (curv_F > 0)]
    mu = opts.newton_damping * (pos.median() if pos.numel() else torch.tensor(1.0, dtype=curv_F.dtype))
    newton_F = grad_F / (curv_F.clamp_min(0.0) + mu)

    def _c(real_amp: Tensor) -> np.ndarray:
        z = torch.where(ok, real_amp, torch.zeros_like(real_amp)).to(phase.dtype) * phase
        return z.cpu().numpy().astype(np.complex128)

    def _r(v: Tensor, fill: float = 0.0) -> np.ndarray:
        return torch.where(ok, v, torch.full_like(v, fill)).cpu().numpy().astype(np.float64)

    stats = dict(post.stats)
    stats.update(
        n_ok=int(ok.sum()),
        n_total=int(ok.numel()),
        nu=None if target.nu is None else float(target.nu),
        information_weight_acentric="2 sigma_A / (1 - sigma_A^2) / sqrt(eps Sigma)",
        newton_damping_abs=float(mu),
    )
    f_mode_E, _ = posterior_mode_E(Ec, sA_n, Zo, sZ, centric)

    return IntensityMapSet(
        difference=_c(scale * diff_E),
        model=_c(scale * model_E),
        gradient=_c(grad_F),
        newton=_c(newton_F),
        fom=_r(post.fom),
        robust_weight=_r(post.lambda_mean, 1.0),
        curvature=_r(curv_F),
        f_post=_r(scale * post.E_mean),
        d_loglik_d_nu=None if post.d_loglik_d_nu is None else _r(post.d_loglik_d_nu),
        stats=stats,
        f_mode=_r(scale * f_mode_E),
    )
