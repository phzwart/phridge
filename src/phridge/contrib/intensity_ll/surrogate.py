"""Per-reflection Rice surrogate for the exact marginal intensity likelihood (``ml_i``).

What this is
------------
The exact target integrates a Rice/Woolfson prior on the true normalized amplitude
``E`` against the intensity noise model (:mod:`phridge.contrib.intensity_ll.mli`).
That marginal costs one adaptive quadrature per evaluation, and a refinement macro
cycle asks for it hundreds of times (bulk-solvent scaling, weight scans, minimizer
line searches, simulated annealing). This module builds a **surrogate** that speaks
the stock amplitude-likelihood interface (``ml_f``: Rice for acentrics, Woolfson /
``tanh`` for centrics) and agrees with the exact marginal to second order at the
current model, so those inner evaluations become closed-form.

Theory: EM with the curvature deficit repaired
----------------------------------------------
Write ``t = E_C`` for the normalized model amplitude and let ``q`` be the exact
posterior ``p(E | Z_o, t_0)`` at the checkpoint model ``t_0``. The EM / variational
bound (the Q-function, equivalently the ELBO) is

    Q(t) = E_q[log p(E | t)]  <=  log L(t) + const,

and by **Fisher's identity** its gradient at ``t_0`` is the exact gradient:

    Q'(t_0) = E_q[d/dt log p(E | t_0)] = d/dt log L(t_0).

Its curvature, however, is wrong. **Louis' identity** says

    d2/dt2 log L(t_0) = E_q[d2/dt2 log p(E | t_0)] + Var_q(d/dt log p(E | t_0)),
                        \\________ Q''(t_0) _______/   \\___ missing information ___/

so the EM bound understates the curvature by exactly the missing information, the
posterior variance of the complete-data score. An inner loop run on ``Q`` therefore
takes systematically short steps and converges at the EM rate.

We instead fit the Rice family to match **both** the exact score and the exact
curvature at ``t_0``. That trades the global lower-bound guarantee of EM for
Newton-quality inner convergence, and recovers safety a different way: every block
is adjudicated by an exact evaluation, and the exact NLL is the sole arbiter of
whether the block is kept (see
:mod:`phridge.client.intensity.interleaved`).

The parameterization, and why it is what it is
----------------------------------------------
The surrogate is ``m(t; F_p, alpha_p, beta_p)``, the acentric Rice log-likelihood
exactly as ``ml_f`` consumes it (centrics: the Woolfson / ``tanh`` analog, identical
procedure throughout):

    acentric:  m = log(2 F_p / beta_p) - (F_p^2 + alpha_p^2 t^2) / beta_p
                     + log I0(2 alpha_p F_p t / beta_p)
    centric:   m = 1/2 log(2 / (pi beta_p)) - (F_p^2 + alpha_p^2 t^2) / (2 beta_p)
                     + log cosh(alpha_p F_p t / beta_p)

* ``alpha_p`` is **shell-wise and pinned**: ``alpha_p(s) = sigma_A(s)``. It is never
  fitted per reflection. Two reasons, both fatal if ignored. Identifiability:
  ``alpha_p`` and ``F_p`` enter the Bessel argument only through their product, so
  the pair is not determined by two derivative conditions. Meaning: ``alpha_p`` is
  the model-quality parameter, and a per-reflection ``alpha_p`` would no longer be
  ``sigma_A``.

* ``F_p`` and ``beta_p`` are **per-reflection**, solved so that ``m'(t_0)`` and
  ``m''(t_0)`` equal the exact score and curvature. Per-reflection ``beta_p`` is not
  a liberty: with a shell-median ``beta_p`` instead, the maximum relative gradient
  error over ``t`` in ``[0.75, 1.25] t_0`` degrades by a factor 4-5 on a mixed shell.
  With per-reflection ``(F_p, beta_p)`` that error has p90 < 0.08 on a mixed shell
  including ~12% negative intensities. A per-reflection ``beta_p`` is also the
  standard variance-inflation convention, so it is interface-legal: cctbx target
  functors take ``alpha`` / ``beta`` as per-reflection flex arrays.

Derivatives used (``r(X) = I1(X)/I0(X)``, ``X = 2 alpha_p F_p t / beta_p``):

    m'  = -2 alpha_p^2 t / beta_p + (2 alpha_p F_p / beta_p) r(X)
    m'' = -2 alpha_p^2 / beta_p   + (2 alpha_p F_p / beta_p)^2 r'(X),
    r'(X) = 1 - r/X - r^2

Note the structure: the surrogate is the Rice *prior* with ``E`` replaced by the
constant ``F_p`` and ``1 - sigma_A^2`` replaced by ``beta_p``. That is why the
closed-form initialization below is exactly the Jensen'd EM bound written in
``ml_f`` form.

Initialization and fallback ladder (per reflection, in order)
------------------------------------------------------------
1. Closed-form EM / Jensen values ``F_p = <E>_post``, ``beta_p = 1 - sigma_A^2``.
   Free from quantities the checkpoint already computed, and a provable lower bound
   (exact-gradient-approximate: it is the prior derivative evaluated at ``<E>_post``
   rather than averaged over ``E``).
2. Newton on ``(F_p, log beta_p)`` from that initialization. The log
   parameterization keeps ``beta_p > 0``. Vectorized over the reflection batch;
   there is no Python loop over reflections.
3. On fit failure: keep the initialization values and set ``FIT_FALLBACK_INIT``.
4. If the initialization is also unusable (should not occur; guarded anyway): set
   ``FIT_EXACT_ROUTE`` and the controller routes that reflection through the exact
   target inside the inner loop.

Per-shell fit-failure counts are emitted in the telemetry dict.

Student-t caveat
----------------
With Student-t intensity noise the robust down-weighting is frozen into
``(F_p, beta_p)`` at the checkpoint. A reflection that becomes an outlier partway
through a block is therefore over-trusted until the next refresh, because Rice tails
are lighter than the t-marginal's. The block accept/reject is the containment; a
rising rejection rate is the signal to shorten blocks, not to widen the tolerance.

Hard rule
---------
``F_p`` and ``beta_p`` are internal arrays. They must never appear in any printed
report, output file, or user-visible label. ``F_p`` is one paste away from being
mistaken for an observed amplitude and ``beta_p`` for an experimental variance. Same
discipline as the S-family naming (see :mod:`phridge.contrib.intensity_ll.rint`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np
import torch
from pydantic import BaseModel, Field

from phridge.contrib.intensity_ll.maps import (
    PosteriorNodes,
    posterior_moments,
    posterior_node_cache,
)
from phridge.contrib.intensity_ll.mli import _ratio, normalize
from phridge.sfcalc.targets.base import Observations

Tensor = torch.Tensor

# Mask bit flags (per reflection). FIT_OK is the absence of every other bit.
FIT_OK = 0
FIT_FALLBACK_INIT = 1
FIT_EXACT_ROUTE = 2

_SERIES_X = 0.3  # below this Bessel-ratio argument the series beats the cancelling form
_SA_LO, _SA_HI = 1e-4, 1.0 - 1e-4


class SurrogateFitOptions(BaseModel):
    """Options for the Rice surrogate fit (``ml_i_surrogate_fit`` op JSON)."""

    model_config = {"extra": "forbid"}

    curvature: str = Field(
        default="louis",
        description=(
            "How the exact second derivative in t is obtained: 'louis' (posterior "
            "expectation of the complete-data second derivative plus the posterior "
            "variance of the complete-data score, on the existing quadrature nodes) or "
            "'finite_difference' (central differences of the analytic score, two extra "
            "vectorized score evaluations). 'louis' is the shipped route; the two are "
            "cross-checked against each other in the tests."
        ),
    )
    fd_rel_step: float = Field(
        default=1e-3,
        gt=0.0,
        lt=0.5,
        description="Relative step in t for the finite-difference curvature route.",
    )
    newton_iters: int = Field(default=30, ge=1, le=200, description="Max Newton iterations.")
    backtrack_steps: int = Field(
        default=8, ge=0, le=32, description="Max step halvings per Newton iteration."
    )
    rel_tol: float = Field(
        default=1e-9, gt=0.0, description="Relative residual tolerance on (m' - g1, m'' - g2)."
    )
    beta_bounds: tuple[float, float] = Field(
        default=(1e-6, 1.0e3),
        description="Bounds on beta_p in normalized units (Wilson-scaled units are Sigma_W times these).",
    )
    f_p_floor: float = Field(
        default=1e-8, gt=0.0, description="Positivity floor on F_p in normalized units."
    )

    def model_post_init(self, _context: Any) -> None:
        if self.curvature not in ("louis", "finite_difference"):
            raise ValueError("curvature must be 'louis' or 'finite_difference'")
        lo, hi = self.beta_bounds
        if not (0.0 < lo < hi):
            raise ValueError("beta_bounds must satisfy 0 < lo < hi")


# ---------------------------------------------------------------- Bessel ratio helpers
def _ratio_prime(x: Tensor) -> Tensor:
    """``r'(X) = 1 - r/X - r^2`` with the small-X series (the closed form cancels)."""
    small = x < _SERIES_X
    xs = torch.where(small, torch.ones_like(x), x)
    r = _ratio(xs)
    direct = 1.0 - r / xs - r * r
    x2 = torch.where(small, x, torch.zeros_like(x)) ** 2
    series = 0.5 - 0.1875 * x2 + (5.0 / 96.0) * x2**2 - 0.01253256 * x2**3
    return torch.where(small, series, direct)


# ---------------------------------------------------------------- the surrogate itself
def rice_log_derivatives(
    t: Tensor,
    f_p: Tensor,
    alpha_p: Tensor,
    beta_p: Tensor,
    centric: Tensor,
) -> tuple[Tensor, Tensor]:
    """First and second derivative in ``t`` of the ``ml_f`` log-likelihood.

    Acentric Rice and centric Woolfson, matching
    :class:`phridge.sfcalc.targets.maximum_likelihood.MaximumLikelihoodAmplitude`
    term for term in normalized units.
    """
    c_a = 2.0 * alpha_p * f_p / beta_p
    x = c_a * t
    r = _ratio(x)
    rp = _ratio_prime(x)
    q_a = 2.0 * alpha_p**2 / beta_p
    m1_a = -q_a * t + c_a * r
    m2_a = -q_a + c_a**2 * rp

    c_c = alpha_p * f_p / beta_p
    y = c_c * t
    s = torch.tanh(y)
    q_c = alpha_p**2 / beta_p
    m1_c = -q_c * t + c_c * s
    m2_c = -q_c + c_c**2 * (1.0 - s * s)

    return torch.where(centric, m1_c, m1_a), torch.where(centric, m2_c, m2_a)


def rice_log_likelihood(
    t: Tensor,
    f_p: Tensor,
    alpha_p: Tensor,
    beta_p: Tensor,
    centric: Tensor,
) -> Tensor:
    """The surrogate log-likelihood itself (used for the predicted-delta diagnostic)."""
    from phridge.sfcalc.targets.maximum_likelihood import ln_cosh, ln_i0

    x = 2.0 * alpha_p * f_p * t / beta_p
    acen = (
        torch.log(2.0 * f_p / beta_p)
        - (f_p**2 + (alpha_p * t) ** 2) / beta_p
        + ln_i0(x)
    )
    cen = (
        0.5 * torch.log(2.0 / (math.pi * beta_p))
        - (f_p**2 + (alpha_p * t) ** 2) / (2.0 * beta_p)
        + ln_cosh(0.5 * x)
    )
    return torch.where(centric, cen, acen)


# ---------------------------------------------------------------- exact score / curvature
def _prior_dt_terms(
    E: Tensor, t: Tensor, sA: Tensor, centric: Tensor
) -> tuple[Tensor, Tensor]:
    """``d/dt`` and ``d2/dt2`` of the complete-data log prior ``log p(E | t)``.

    This is the only ``t``-dependent piece of the complete-data log density: for
    Student-t noise the complete data is ``(E, lambda)`` and ``log p(lambda)`` carries
    no ``t``, so Fisher's and Louis' identities may be taken against the joint
    posterior on the ``(u, E)`` nodes without further terms.
    """
    a = 1.0 - sA**2
    k_a = 2.0 * sA * E / a
    x = k_a * t
    d1_a = -2.0 * sA**2 * t / a + k_a * _ratio(x)
    d2_a = -2.0 * sA**2 / a + k_a**2 * _ratio_prime(x)

    k_c = sA * E / a
    y = k_c * t
    s = torch.tanh(y)
    d1_c = -(sA**2) * t / a + k_c * s
    d2_c = -(sA**2) / a + k_c**2 * (1.0 - s * s)

    return torch.where(centric, d1_c, d1_a), torch.where(centric, d2_c, d2_a)


def exact_score_and_curvature(cache: PosteriorNodes) -> tuple[Tensor, Tensor]:
    """Exact ``(d/dt log L, d2/dt2 log L)`` at the checkpoint model, via Louis' identity.

    ``g1`` is Fisher's identity, the posterior mean of the complete-data score; it is
    the same quantity ``maps._score_from_Em`` returns as
    ``(2 sigma_A / (1 - sigma_A^2)) * Delta``. ``g2`` adds the missing information,
    the posterior variance of the complete-data score, to the posterior mean of the
    complete-data second derivative. Both are evaluated on the nodes the likelihood
    itself integrated on, so no extra quadrature is performed.
    """
    with torch.no_grad():
        t = cache.Ec[:, None, None]
        sA = cache.sA.clamp(_SA_LO, _SA_HI)[:, None, None]
        cen = cache.centric[:, None, None]
        d1, d2 = _prior_dt_terms(cache.nodes, t, sA, cen)
        p = cache.p
        g1 = (p * d1).sum(dim=(1, 2))
        mean_d2 = (p * d2).sum(dim=(1, 2))
        var_d1 = (p * d1**2).sum(dim=(1, 2)) - g1**2
        return g1, mean_d2 + var_d1.clamp_min(0.0)


def curvature_by_finite_difference(
    Ec: Tensor,
    sA: Tensor,
    Zo: Tensor,
    sZ: Tensor,
    centric: Union[Tensor, bool],
    nu: Optional[Union[float, Tensor]] = None,
    *,
    rel_step: float = 1e-3,
    **quad_kwargs: Any,
) -> Tensor:
    """``d2/dt2 log L`` by central differences of the analytic score.

    The independent route to the same number as :func:`exact_score_and_curvature`;
    two extra vectorized score evaluations. Shipped for the cross-check, and
    available through ``SurrogateFitOptions(curvature="finite_difference")``.
    """
    with torch.no_grad():
        h = rel_step * Ec.abs().clamp_min(1e-3)
        kw = dict(quad_kwargs)
        s_plus = posterior_moments(Ec + h, sA, Zo, sZ, centric, nu, **kw).score
        s_minus = posterior_moments((Ec - h).clamp_min(0.0), sA, Zo, sZ, centric, nu, **kw).score
        return (s_plus - s_minus) / (2.0 * h)


# ---------------------------------------------------------------- vectorized Newton fit
@dataclass
class RiceSurrogate:
    """Fitted per-reflection surrogate arrays in **normalized** units.

    Internal object. ``f_p`` and ``beta_p`` must never reach a printed report, an
    output file, or a user-visible label; see the module docstring.
    """

    f_p: Tensor  # (N,)
    beta_p: Tensor  # (N,)
    alpha_p: Tensor  # (N,) = sigma_A, shell-wise and pinned
    t0: Tensor  # (N,) E_C at the checkpoint
    centric: Tensor  # (N,) bool
    mask: Tensor  # (N,) int8 bit flags
    score: Tensor  # (N,) exact g1 at t0
    curvature: Tensor  # (N,) exact g2 at t0
    telemetry: dict[str, Any] = field(default_factory=dict)

    @property
    def exact_route(self) -> Tensor:
        """Reflections the inner loop must still send through the exact target."""
        return (self.mask & FIT_EXACT_ROUTE) != 0

    def amplitude_units(
        self, epsilon: Tensor, sigma_wilson: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Convert to the units ``ml_f`` consumes: ``(F_p, alpha_p, beta_p)``.

        With ``S = sqrt(epsilon Sigma_W)`` so that ``F = S E``, substituting
        ``F_p -> S F_p``, ``beta_p -> Sigma_W beta_p`` into the ``ml_f`` form (whose
        residual scale is ``eb = epsilon beta``) reproduces the normalized expressions
        term for term, so ``d/d|F_c| = (1/S) d/dt`` follows automatically.
        """
        scale = torch.sqrt(epsilon * sigma_wilson)
        return self.f_p * scale, self.alpha_p, self.beta_p * sigma_wilson

    def predicted_log_likelihood(self, t: Tensor) -> Tensor:
        """Surrogate log-likelihood at a new ``t`` (diagnostic only, never an arbiter)."""
        return rice_log_likelihood(t, self.f_p, self.alpha_p, self.beta_p, self.centric)


def _residual(
    theta: Tensor,
    t0: Tensor,
    alpha_p: Tensor,
    centric: Tensor,
    g1: Tensor,
    g2: Tensor,
) -> tuple[Tensor, Tensor]:
    f_p = theta[:, 0]
    beta_p = torch.exp(theta[:, 1])
    m1, m2 = rice_log_derivatives(t0, f_p, alpha_p, beta_p, centric)
    return m1 - g1, m2 - g2


def _scaled_norm(r1: Tensor, r2: Tensor, s1: Tensor, s2: Tensor) -> Tensor:
    return torch.sqrt((r1 / s1) ** 2 + (r2 / s2) ** 2)


def _newton_solve(
    theta0: Tensor,
    t0: Tensor,
    sA: Tensor,
    centric: Tensor,
    g1: Tensor,
    g2: Tensor,
    s1: Tensor,
    s2: Tensor,
    pool: Tensor,
    opts: SurrogateFitOptions,
) -> tuple[Tensor, Tensor]:
    """Damped Newton with backtracking from one initialization; returns (theta, residual norm).

    Vectorized over reflections, and compressed onto the still-unconverged subset at
    every iteration. That matters: without the compression a handful of hard
    reflections keep the whole batch iterating and the fit costs an order of magnitude
    more than the exact evaluation it is supposed to replace.
    """
    log_lo, log_hi = math.log(opts.beta_bounds[0]), math.log(opts.beta_bounds[1])
    theta = theta0.clone()
    r1, r2 = _residual(theta, t0, sA, centric, g1, g2)
    best_norm = _scaled_norm(r1, r2, s1, s2)
    best_norm = torch.where(
        torch.isfinite(best_norm), best_norm, torch.full_like(best_norm, math.inf)
    )
    idx = (pool & (best_norm > opts.rel_tol)).nonzero().flatten()

    for _ in range(opts.newton_iters):
        if idx.numel() == 0:
            break
        th_i = theta[idx]
        t_i, a_i, c_i = t0[idx], sA[idx], centric[idx]
        g1_i, g2_i, s1_i, s2_i = g1[idx], g2[idx], s1[idx], s2[idx]
        base_i = best_norm[idx]

        th = th_i.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            rr1, rr2 = _residual(th, t_i, a_i, c_i, g1_i, g2_i)
            # Each residual element depends only on its own row of theta, so the
            # gradient of the sum is exactly the per-reflection partial derivative.
            (j1,) = torch.autograd.grad(rr1.sum(), th, retain_graph=True)
            (j2,) = torch.autograd.grad(rr2.sum(), th)
        jac = torch.stack([j1, j2], dim=1)  # (n, 2, 2): rows = residual components
        rhs = -torch.stack([rr1.detach(), rr2.detach()], dim=1).unsqueeze(-1)
        det = jac[:, 0, 0] * jac[:, 1, 1] - jac[:, 0, 1] * jac[:, 1, 0]
        solvable = torch.isfinite(det) & (det.abs() > 0) & torch.isfinite(jac).all(dim=1).all(dim=1)
        eye = torch.eye(2, dtype=jac.dtype, device=jac.device).expand_as(jac)
        delta = torch.linalg.solve(torch.where(solvable[:, None, None], jac, eye), rhs).squeeze(-1)
        delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)

        lam = torch.ones_like(t_i)
        taken = torch.zeros_like(solvable)
        for _ in range(opts.backtrack_steps + 1):
            trial = th_i + lam[:, None] * delta
            trial = torch.stack(
                [trial[:, 0].clamp_min(opts.f_p_floor), trial[:, 1].clamp(log_lo, log_hi)], dim=1
            )
            q1, q2 = _residual(trial, t_i, a_i, c_i, g1_i, g2_i)
            tn = _scaled_norm(q1, q2, s1_i, s2_i)
            take = solvable & torch.isfinite(tn) & (tn < base_i) & ~taken
            th_i = torch.where(take[:, None], trial, th_i)
            base_i = torch.where(take, tn, base_i)
            taken = taken | take
            if bool((taken | ~solvable).all()):
                break
            lam = torch.where(taken, lam, lam * 0.5)

        theta[idx] = th_i
        best_norm[idx] = base_i
        # A reflection that could not improve its residual is done (converged or stuck).
        idx = idx[taken & (base_i > opts.rel_tol)]

    return theta, best_norm


def fit_rice_surrogate(
    t0: Tensor,
    sA: Tensor,
    centric: Tensor,
    g1: Tensor,
    g2: Tensor,
    e_mean: Tensor,
    *,
    options: Optional[SurrogateFitOptions] = None,
    shell_index: Optional[Tensor] = None,
) -> RiceSurrogate:
    """Solve ``m'(t_0) = g1`` and ``m''(t_0) = g2`` for ``(F_p, log beta_p)``.

    Damped Newton with backtracking, fully vectorized over the reflection batch.
    ``alpha_p`` is pinned to ``sigma_A`` and never fitted. See the module docstring
    for the initialization / fallback ladder.

    The EM / Jensen initialization converges for the overwhelming majority of
    reflections. A minority -- weak reflections at small ``t_0`` whose exact
    log-likelihood is locally **convex** -- live on the far side of the family, where
    ``F_p^2 > beta_p`` produces the positive curvature, and are unreachable by Newton
    from the EM start. Those get additional vectorized starts, including one placed
    analytically on the ``X = 1`` convex branch. Whatever still fails drops to the
    fallback ladder.
    """
    opts = options or SurrogateFitOptions()
    with torch.no_grad():
        sA_c = sA.clamp(_SA_LO, _SA_HI)
        beta_lo, beta_hi = opts.beta_bounds
        log_lo, log_hi = math.log(beta_lo), math.log(beta_hi)

        # 1. closed-form EM / Jensen initialization
        f_init = torch.nan_to_num(e_mean, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(opts.f_p_floor)
        beta_init = (1.0 - sA_c**2).clamp(beta_lo, beta_hi)
        init_usable = (
            torch.isfinite(f_init) & (f_init > 0) & torch.isfinite(beta_init) & (beta_init > 0)
        )

        # The Rice family is not globally concave in t -- at X = 0 the curvature is
        # 2 alpha^2 (F_p^2/beta - 1) / beta, positive whenever F_p^2 > beta -- so a
        # positive exact curvature is matchable too. Only non-finite targets are
        # refused outright; reachability is decided by the residual after Newton.
        matchable = torch.isfinite(g1) & torch.isfinite(g2)

        # Residual scales: the two components differ by ~1/t in magnitude.
        s1 = (g1.abs() + 2.0 * sA_c**2 * t0.abs() / beta_init).clamp_min(1e-8)
        s2 = (g2.abs() + 2.0 * sA_c**2 / beta_init).clamp_min(1e-8)

        pool = matchable & init_usable

        def _clamped(f_p: Tensor, beta: Tensor) -> Tensor:
            return torch.stack(
                [
                    torch.nan_to_num(f_p, nan=1.0, posinf=1.0, neginf=1.0).clamp_min(opts.f_p_floor),
                    torch.log(
                        torch.nan_to_num(beta, nan=1.0, posinf=1.0, neginf=1.0).clamp(beta_lo, beta_hi)
                    ).clamp(log_lo, log_hi),
                ],
                dim=1,
            )

        starts = [_clamped(f_init, beta_init)]

        # Convex branch: put X = 2 alpha F_p t / beta at 1, where r'(1) is known, and
        # solve the curvature condition for beta in closed form. This lands inside the
        # F_p^2 > beta region that positive curvature requires.
        r_prime_1 = float(_ratio_prime(torch.ones((), dtype=t0.dtype)))
        t_safe = t0.abs().clamp_min(1e-6)
        denom = r_prime_1 / t_safe**2 - g2
        beta_cvx = (2.0 * sA_c**2 / torch.where(denom.abs() > 1e-12, denom, torch.full_like(denom, 1e-12))).abs()
        starts.append(_clamped(beta_cvx / (2.0 * sA_c * t_safe), beta_cvx))
        # Two scale sweeps, cheap insurance against a bad EM start.
        starts.append(_clamped(f_init * 4.0, beta_init * 16.0))
        starts.append(_clamped(f_init * 0.25, beta_init * 0.25))

        theta = starts[0]
        best_norm = torch.full_like(t0, math.inf)
        remaining = pool
        for start in starts:
            th, nrm = _newton_solve(start, t0, sA_c, centric, g1, g2, s1, s2, remaining, opts)
            better = torch.isfinite(nrm) & (nrm < best_norm)
            theta = torch.where(better[:, None], th, theta)
            best_norm = torch.where(better, nrm, best_norm)
            # Later starts are only for what the earlier ones could not reach.
            remaining = remaining & (best_norm > opts.rel_tol)
            if not bool(remaining.any()):
                break

        f_p = theta[:, 0]
        beta_p = torch.exp(theta[:, 1])
        fit_ok = (
            (best_norm <= opts.rel_tol)
            & matchable
            & torch.isfinite(f_p)
            & (f_p > 0)
            & torch.isfinite(beta_p)
            & (beta_p > beta_lo)
            & (beta_p < beta_hi)
        )

        mask = torch.zeros_like(f_p, dtype=torch.int8)
        fallback = ~fit_ok & init_usable
        exact_route = ~fit_ok & ~init_usable
        f_p = torch.where(fit_ok, f_p, f_init)
        beta_p = torch.where(fit_ok, beta_p, beta_init)
        mask = torch.where(fallback, torch.full_like(mask, FIT_FALLBACK_INIT), mask)
        mask = torch.where(exact_route, torch.full_like(mask, FIT_EXACT_ROUTE), mask)
        # Keep the arrays usable even on the exact-routed reflections: ml_f still sees
        # them, and the controller adds the exact contribution on top.
        f_p = torch.where(torch.isfinite(f_p) & (f_p > 0), f_p, torch.full_like(f_p, 1.0))
        beta_p = torch.where(
            torch.isfinite(beta_p) & (beta_p > 0), beta_p, torch.full_like(beta_p, 1.0)
        )

        telemetry: dict[str, Any] = {
            "n_total": int(f_p.numel()),
            "n_fit_ok": int(fit_ok.sum()),
            "n_fallback_init": int(fallback.sum()),
            "n_exact_route": int(exact_route.sum()),
            "n_non_finite_target": int((~matchable).sum()),
            "curvature_route": opts.curvature,
            "residual_p90": float(
                torch.quantile(best_norm[torch.isfinite(best_norm)], 0.9).item()
            )
            if bool(torch.isfinite(best_norm).any())
            else float("nan"),
        }
        if shell_index is not None:
            fail = fallback | exact_route
            n_shell = int(shell_index.max().item()) + 1 if shell_index.numel() else 0
            counts = torch.zeros(n_shell, dtype=torch.int64, device=f_p.device)
            counts.scatter_add_(0, shell_index.long(), fail.long())
            totals = torch.zeros(n_shell, dtype=torch.int64, device=f_p.device)
            totals.scatter_add_(0, shell_index.long(), torch.ones_like(fail, dtype=torch.int64))
            telemetry["fit_failures_per_shell"] = [int(v) for v in counts.tolist()]
            telemetry["n_per_shell"] = [int(v) for v in totals.tolist()]

        return RiceSurrogate(
            f_p=f_p,
            beta_p=beta_p,
            alpha_p=sA_c,
            t0=t0,
            centric=centric,
            mask=mask,
            score=g1,
            curvature=g2,
            telemetry=telemetry,
        )


# ---------------------------------------------------------------- top-level entry point
@dataclass
class SurrogateFitResult:
    """Surrogate in ``ml_f`` (amplitude) units plus the exact quantities behind it."""

    f_p: np.ndarray  # (N,) F units — internal, never printed
    alpha_p: np.ndarray  # (N,) = sigma_A
    beta_p: np.ndarray  # (N,) F units — internal, never printed
    mask: np.ndarray  # (N,) int8
    score: np.ndarray  # (N,) d log L / d E_C
    curvature: np.ndarray  # (N,) d2 log L / d E_C^2
    e_c: np.ndarray  # (N,) E_C at the checkpoint
    e_mean: np.ndarray  # (N,) <E>_post
    rho2: np.ndarray  # (N,) visible fraction 1 - Var_post(E)/(...)  (refresh heuristic)
    ok: np.ndarray  # (N,) bool — reflections the surrogate covers at all
    telemetry: dict[str, Any]


def fit_surrogate_for_observations(
    target: Any,
    f_calc: Tensor,
    obs: Observations,
    options: Optional[SurrogateFitOptions] = None,
    *,
    shell_index: Optional[Tensor] = None,
) -> SurrogateFitResult:
    """Fit the surrogate at the current model, reusing the exact target's own nuisance arrays.

    ``target`` is an :class:`~phridge.contrib.intensity_ll.target.IntensityLogLikelihood`,
    so ``sigma_A``, ``Sigma_W``, ``sigma(I)`` and ``nu`` come from exactly the same
    place the exact NLL reads them: the surrogate and the exact target share one
    posterior at the checkpoint.
    """
    opts = options or SurrogateFitOptions()
    with torch.no_grad():
        fo = obs.data
        fc = f_calc.abs()
        eps = obs.epsilon if obs.epsilon is not None else torch.ones_like(fo)
        centric = (
            obs.centric if obs.centric is not None else torch.zeros_like(fo, dtype=torch.bool)
        )
        sA = target._sigma_a(obs)
        sW = target._sigma_wilson(obs)
        sig = target._sigma(obs)

        ok = (sA > 0) & (sA < 1.0 - 1e-6) & (sW > 0) & (sig > 0) & (eps > 0) & (fc > 0)
        sA_s = torch.where(ok, sA, torch.full_like(sA, 0.5))
        sW_s = torch.where(ok, sW, torch.ones_like(sW))
        sig_s = torch.where(ok, sig, torch.ones_like(sig))
        eps_s = torch.where(ok, eps, torch.ones_like(eps))
        fc_s = torch.where(ok, fc, torch.ones_like(fc))
        fo_s = torch.where(ok, fo, torch.zeros_like(fo))

        Ec, sA_n, Zo, sZ = normalize(fc_s, fo_s, sig_s, eps_s, sW_s, sA_s)
        nu = target._nu(obs)
        quad_kw = dict(
            n_u=target.n_u,
            snr_strong=target.snr_strong,
            n_hermite=target.n_hermite,
            n_legendre=target.n_legendre,
            k_window=target.k_window,
        )
        cache = posterior_node_cache(Ec, sA_n, Zo, sZ, centric, nu, **quad_kw)
        post = posterior_moments(Ec, sA_n, Zo, sZ, centric, nu, cache=cache, **quad_kw)

        g1, g2 = exact_score_and_curvature(cache)
        if opts.curvature == "finite_difference":
            g2 = curvature_by_finite_difference(
                Ec, sA_n, Zo, sZ, centric, nu, rel_step=opts.fd_rel_step, **quad_kw
            )

        fit = fit_rice_surrogate(
            Ec,
            sA_n,
            cache.centric,
            g1,
            g2,
            post.E_mean,
            options=opts,
            shell_index=shell_index,
        )
        f_p_F, alpha_p, beta_p_F = fit.amplitude_units(eps_s, sW_s)

        # Visible fraction per reflection: the share of the residual that is model
        # signal rather than posterior latitude. The EM contraction rate per
        # reflection is the missing-information fraction 1 - rho2, so low-rho2
        # reflections are the ones whose surrogates go stale fastest.
        v_vis = (post.E_mean - sA_n * Ec) ** 2
        v_lat = (post.I_mean - post.E_mean**2).clamp_min(0.0)
        rho2 = v_vis / (v_vis + v_lat).clamp_min(1e-300)

        telemetry = dict(fit.telemetry)
        telemetry["n_masked_out"] = int((~ok).sum())

        def _np(v: Tensor, dtype: Any = np.float64) -> np.ndarray:
            return v.detach().cpu().numpy().astype(dtype)

        return SurrogateFitResult(
            f_p=_np(f_p_F),
            alpha_p=_np(alpha_p),
            beta_p=_np(beta_p_F),
            mask=_np(fit.mask, np.int8),
            score=_np(fit.score),
            curvature=_np(fit.curvature),
            e_c=_np(Ec),
            e_mean=_np(post.E_mean),
            rho2=_np(rho2),
            ok=_np(ok, bool),
            telemetry=telemetry,
        )
