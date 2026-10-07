"""The S family of integrated agreement statistics for ``ml_i``.

``S_post`` is the expected residual under the per-reflection posterior;
``S_prior`` is the same functional under the prior only. **Same residual
functional, different measure** — that one sentence defines the family.

Reporting-layer only — does not affect the likelihood, gradients, or map
coefficients. Expectations use the same quadrature nodes / weights as
``posterior_moments`` (Rice/Woolfson × Gaussian or Student-t mixture).

Why S and not R
---------------
These are not the crystallographic R factor: the functional is an expectation
under a probability distribution, and point-estimate variants of it are biased
low by posterior shrinkage. Named with the letter R they get pasted into R
columns and compared against deposited values no matter what the caption says —
the field already paid for that lesson with Rmerge / Rmeas / Rpim, and NMR
deliberately picked "Q-factor" for its R-like statistic.

**Never write a bare ``S``.** Capital S alone is the small-molecule
goodness-of-fit (``_refine_ls_goodness_of_fit``) and lowercase ``s`` is
sin(theta)/lambda, which appears throughout this package as ``sigma_A(s)``.
Every user-visible occurrence carries its subscript: ``S_post``, ``S_prior``.

Definitions (normalized amplitude units)
---------------------------------------
Per reflection the posterior is ``p(E | Z_o, E_C)`` on the frozen nodes.
Over a reflection set (work and free handled separately):

.. math::

    k_S = \\frac{\\sum_h \\langle E_h \\rangle E_{C,h}}
                {\\sum_h E_{C,h}^2}

    S_{\\mathrm{post}} = \\frac{\\sum_h \\langle |E_h - k_S E_{C,h}| \\rangle}
                               {\\sum_h \\langle E_h \\rangle}

``S_prior`` is the same pair of expectations under the **prior only**
(``prior_only=True``), implemented by evaluating the identical node machinery
with ``sigma_Z = 1e6`` (flat likelihood) — the sigma_A-implied no-data floor.

``s_vis`` is the visible-numerator (plug-in) variant, which evaluates the
residual at the posterior mean instead of integrating over it. It is a
**debugging quantity only** and is never printed in the default report.

Do not compare any of these row-for-row against deposited legacy R values.

**Scale convention for shells:** one scale convention for the whole family.
Overall work and free each get their own ``k_S``; resolution shells reuse the
**parent set's** ``k_S`` (not re-fit per shell).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import torch
from pydantic import BaseModel, Field

from phridge.d.maps import _fom_at
from phridge.d.rules.mli_quad import normalize, quadrature_terms_normal, quadrature_terms_t

Tensor = torch.Tensor

PRIOR_ONLY_SIGMA_Z = 1e6


class IntegratedSOptions(BaseModel):
    """Options for integrated-R reporting (maps JSON / env)."""

    model_config = {"extra": "forbid"}

    bin_size: Optional[int] = Field(default=None, ge=1, description="Equal-count shell size; None → no shells")
    prior_only_sigma_z: float = Field(default=PRIOR_ONLY_SIGMA_Z, gt=0.0)
    min_free_per_shell: int = Field(
        default=30, ge=0, description="Blank free-set shell statistics below this count"
    )
    score_test: bool = Field(
        default=False,
        description="Compute xi / omega; costs n_predictive_nodes extra posterior solves",
    )
    n_predictive_nodes: int = Field(
        default=12, ge=2, le=64, description="Gauss-Hermite nodes for the xi reference variance"
    )


@dataclass
class AmplitudeMomentsCache:
    """Cached quadrature nodes/weights for pass-2 absdev without re-finding modes."""

    nodes: Tensor  # (N, n_u, J)
    p: Tensor  # joint posterior weights, same shape
    E_mean: Tensor  # (N,)
    E_sq_mean: Tensor  # (N,) <E^2>; same quantity maps.py reports as I_mean
    Ec: Tensor  # (N,)
    sA: Tensor  # (N,)
    centric: Tensor  # (N,) bool
    prior_only: bool

    def variance(self) -> Tensor:
        """``Var_post(E) = <E^2> - <E>^2``, clamped at 0."""
        return (self.E_sq_mean - self.E_mean**2).clamp_min(0.0)


def posterior_amplitude_cache(
    Ec: Tensor,
    sA: Tensor,
    Zo: Tensor,
    sZ: Tensor,
    centric: Union[Tensor, bool],
    nu: Optional[Union[float, Tensor]] = None,
    *,
    prior_only: bool = False,
    n_u: int = 12,
    snr_strong: float = 5.0,
    n_hermite: int = 7,
    n_legendre: int = 32,
    k_window: float = 8.0,
    prior_only_sigma_z: float = PRIOR_ONLY_SIGMA_Z,
) -> AmplitudeMomentsCache:
    """Pass 1: build node weights and ``<E>``. Reuse cache for ``absdev_given_k``.

    When ``prior_only=True``, ``sZ`` is replaced by ``prior_only_sigma_z`` (default
    ``1e6``) so the noise factor is flat — same code path, no duplicate integrator.

    Reporting defaults match map-coefficient quadrature (stable); callers may raise
    ``n_legendre`` for extra accuracy.
    """
    with torch.no_grad():
        Ec, sA, Zo, sZ = torch.broadcast_tensors(Ec, sA, Zo, sZ)
        # Keep Rice width away from 0 (sA→1 makes log-integrands explode)
        sA = sA.clamp(1e-4, 1.0 - 1e-4)
        Ec = torch.nan_to_num(Ec, nan=0.0, posinf=1e3, neginf=0.0).clamp_min(0.0)
        Zo = torch.nan_to_num(Zo, nan=0.0, posinf=1e6, neginf=-1e6)
        sZ = torch.nan_to_num(sZ, nan=1.0, posinf=1e6, neginf=1.0).clamp_min(1e-12)
        if prior_only:
            sZ = torch.full_like(sZ, float(prior_only_sigma_z))
        centric_t = torch.as_tensor(centric, device=Ec.device).bool().expand_as(Ec)
        q_kw = dict(snr_strong=snr_strong, n_hermite=n_hermite, n_legendre=n_legendre, k_window=k_window)

        if nu is None:
            nodes, terms, _stats = quadrature_terms_normal(Ec, sA, Zo, sZ, centric_t, **q_kw)
            nodes = nodes[:, None, :]
            terms = terms[:, None, :]
        else:
            nodes, terms, _u_nodes, _stats = quadrature_terms_t(
                Ec, sA, Zo, sZ, centric_t, nu, n_u=n_u, **q_kw
            )

        flat_terms = terms.reshape(terms.shape[0], -1)
        flat_terms = torch.where(
            torch.isfinite(flat_terms), flat_terms, torch.full_like(flat_terms, -math.inf)
        )
        all_bad = torch.isneginf(flat_terms).all(dim=1)
        # Fallback: plug-in amplitude so <E> stays on the physical scale (not 0).
        # Keep nodes/p consistent with E_mean — never leave stale padding mass in absdev.
        fallback = torch.sqrt(Zo.clamp_min(0.0)).clamp_min(1e-6)
        fallback = torch.where(Ec > 0, 0.5 * (fallback + Ec), fallback)
        if bool(all_bad.any()):
            flat_terms = flat_terms.clone()
            flat_terms[all_bad] = -math.inf
            flat_terms[all_bad, 0] = 0.0
            nodes_flat = nodes.reshape(nodes.shape[0], -1).clone()
            nodes_flat[all_bad] = 0.0
            nodes_flat[all_bad, 0] = fallback[all_bad]
            nodes = nodes_flat.reshape(nodes.shape)
        p = torch.softmax(flat_terms, dim=1).reshape(terms.shape)
        # Replace non-finite softmax rows (all -inf → nan) with a delta at fallback
        p_flat = p.reshape(p.shape[0], -1)
        bad_p = ~torch.isfinite(p_flat).all(dim=1) | (p_flat.sum(dim=1) < 0.5)
        if bool(bad_p.any()):
            p_flat = p_flat.clone()
            nodes_flat = nodes.reshape(nodes.shape[0], -1).clone()
            p_flat[bad_p] = 0.0
            p_flat[bad_p, 0] = 1.0
            nodes_flat[bad_p] = 0.0
            nodes_flat[bad_p, 0] = fallback[bad_p]
            p = p_flat.reshape(p.shape)
            nodes = nodes_flat.reshape(nodes.shape)
        E_mean = (p * nodes).sum(dim=(1, 2))
        E_mean = torch.where(torch.isfinite(E_mean) & (E_mean > 0), E_mean, fallback)
        E_mean = E_mean.clamp_min(1e-12)
        E_sq_mean = (p * nodes**2).sum(dim=(1, 2))
        # Degenerate rows fall back to zero variance rather than a negative one
        E_sq_mean = torch.where(
            torch.isfinite(E_sq_mean) & (E_sq_mean >= E_mean**2), E_sq_mean, E_mean**2
        )

    return AmplitudeMomentsCache(
        nodes=nodes,
        p=p,
        E_mean=E_mean,
        E_sq_mean=E_sq_mean,
        Ec=Ec,
        sA=sA,
        centric=centric_t,
        prior_only=bool(prior_only),
    )


def absdev_given_k(cache: AmplitudeMomentsCache, k: Union[float, Tensor]) -> Tensor:
    """Pass 2: ``<|E - k E_C|>`` on cached nodes (no mode re-find).

    When ``E_C = 0`` the residual is ``<E>`` (Wilson-only prior).
    Non-finite ``k`` returns NaN residuals (never ``k=0``, which forces ``R≈1``).
    """
    with torch.no_grad():
        k_t = torch.as_tensor(k, dtype=cache.E_mean.dtype, device=cache.E_mean.device)
        if not torch.isfinite(k_t).all():
            return torch.full_like(cache.E_mean, float("nan"))
        if k_t.ndim == 0:
            target = k_t * cache.Ec
        else:
            target = k_t.reshape(-1) * cache.Ec
        resid = (cache.nodes - target[:, None, None]).abs()
        out = (cache.p * resid).sum(dim=(1, 2))
        zero_ec = cache.Ec.abs() < 1e-15
        if bool(zero_ec.any()):
            out = torch.where(zero_ec, cache.E_mean, out)
        # Plug-in |<E>-k Ec|; triangle inequality <|E-t|> ≤ <E> + |t|
        plug = (cache.E_mean - target).abs()
        out = torch.where(torch.isfinite(out), out, plug)
        out = torch.minimum(out, cache.E_mean + target.abs())
        return out.clamp_min(0.0)


def sqdev_given_k(cache: AmplitudeMomentsCache, k: Union[float, Tensor]) -> Tensor:
    """Pass 2: ``<(E - k E_C)^2>`` on cached nodes — the L2 companion to
    :func:`absdev_given_k`.

    Per reflection this equals ``(<E> - k E_C)^2 + Var_post(E)`` exactly, which is
    the identity behind the ``v_vis`` / ``v_lat`` split. ``S_post`` itself is L1 and
    admits no such decomposition, so the split is reported from this quantity.
    """
    with torch.no_grad():
        k_t = torch.as_tensor(k, dtype=cache.E_mean.dtype, device=cache.E_mean.device)
        if not torch.isfinite(k_t).all():
            return torch.full_like(cache.E_mean, float("nan"))
        if k_t.ndim == 0:
            target = k_t * cache.Ec
        else:
            target = k_t.reshape(-1) * cache.Ec
        resid = (cache.nodes - target[:, None, None]) ** 2
        out = (cache.p * resid).sum(dim=(1, 2))
        # Fallback is the exact decomposition, so the identity survives bad rows
        plug = (cache.E_mean - target) ** 2 + cache.variance()
        out = torch.where(torch.isfinite(out), out, plug)
        return out.clamp_min(0.0)


def delta_given_cache(cache: AmplitudeMomentsCache) -> Tensor:
    """``Delta = <E m>_post - sigma_A E_C`` — the difference-map numerator.

    Identical expression to ``maps.intensity_map_coefficients`` (``Em_mean - sA*Ec``),
    evaluated on the cached nodes with the shared :func:`~...maps._fom_at` so the
    figure-of-merit factor cannot drift between the map path and the score test.
    """
    with torch.no_grad():
        m = _fom_at(
            cache.nodes,
            cache.Ec[:, None, None],
            cache.sA[:, None, None],
            cache.centric[:, None, None],
        )
        Em_mean = (cache.p * cache.nodes * m).sum(dim=(1, 2))
        Em_mean = torch.where(torch.isfinite(Em_mean), Em_mean, cache.sA * cache.Ec)
        return Em_mean - cache.sA * cache.Ec


def predictive_delta_variance(
    Ec: Tensor,
    sA: Tensor,
    sZ: Tensor,
    centric: Tensor,
    nu: Optional[Union[float, Tensor]] = None,
    *,
    n_z: int = 12,
    **cache_kw: Any,
) -> Tensor:
    """``E_model[|Delta|^2]``: the variance of the score numerator under the
    predictive distribution of ``Z_o`` at the claimed ``sigma_A``.

    Evaluated by Gauss-Hermite quadrature on a rule matched to the exact predictive
    mean and variance of ``Z_o`` — **never** by Monte Carlo. A sampled reference
    variance is inflated by Jensen's inequality, and that inflation cancels in
    ``omega = xi_free / xi_work`` but not in ``xi_work`` or ``xi_free`` separately,
    which would leave both individually uninterpretable.

    With ``a = 1 - sigma_A^2`` the true intensity is noncentral chi-squared about
    ``sigma_A^2 E_C^2``, so ``<I> = sigma_A^2 E_C^2 + a`` and ``Var(I)`` is
    ``a^2 + 2 a sigma_A^2 E_C^2`` for acentrics (2 dof) or twice that for centrics
    (1 dof); measurement noise adds ``sigma_Z^2``.

    Costs ``n_z`` full posterior solves per reflection — the reason the score test
    is opt-in rather than part of the default report.
    """
    with torch.no_grad():
        a = (1.0 - sA**2).clamp_min(1e-12)
        mu_c = (sA * Ec) ** 2
        mu_I = mu_c + a
        var_I = torch.where(centric, 2.0 * a**2 + 4.0 * a * mu_c, a**2 + 2.0 * a * mu_c)
        sd = torch.sqrt(2.0 * (var_I + sZ**2).clamp_min(1e-30))
        x, w = np.polynomial.hermite.hermgauss(int(n_z))
        acc = torch.zeros_like(Ec)
        norm = 1.0 / math.sqrt(math.pi)
        for x_i, w_i in zip(x.tolist(), w.tolist()):
            cache = posterior_amplitude_cache(
                Ec, sA, mu_I + sd * float(x_i), sZ, centric, nu=nu, **cache_kw
            )
            acc = acc + (norm * float(w_i)) * delta_given_cache(cache) ** 2
        return acc.clamp_min(0.0)


def scale_k(E_mean: Tensor, Ec: Tensor, mask: Optional[Tensor] = None) -> float:
    """LS scale ``k = sum(<E> E_C) / sum(E_C^2)`` on the masked set."""
    with torch.no_grad():
        if mask is None:
            mask = torch.ones_like(E_mean, dtype=torch.bool)
        else:
            mask = mask.bool()
        finite = mask & torch.isfinite(E_mean) & torch.isfinite(Ec) & (E_mean > 1e-12) & (Ec > 0)
        if not bool(finite.any()):
            warnings.warn("scale_k: empty reflection set; returning nan", stacklevel=2)
            return float("nan")
        num = (E_mean[finite] * Ec[finite]).sum()
        den = (Ec[finite] ** 2).sum().clamp_min(1e-300)
        k = float((num / den).item())
        return k if math.isfinite(k) else float("nan")


def aggregate_s(E_mean: Tensor, absdev: Tensor, mask: Optional[Tensor] = None) -> float:
    """``R = sum <|E-k Ec|> / sum <E>`` on the masked set."""
    with torch.no_grad():
        if mask is None:
            mask = torch.ones_like(E_mean, dtype=torch.bool)
        else:
            mask = mask.bool()
        finite = (
            mask
            & torch.isfinite(E_mean)
            & torch.isfinite(absdev)
            & (E_mean > 1e-12)
            & (absdev >= 0)
        )
        if not bool(finite.any()):
            warnings.warn("aggregate_s: empty reflection set; returning nan", stacklevel=2)
            return float("nan")
        den = E_mean[finite].sum().clamp_min(1e-300)
        r = float((absdev[finite].sum() / den).item())
        # Guard against numerical blow-ups from degenerate shells
        if not math.isfinite(r) or r < 0.0 or r > 5.0:
            return float("nan")
        return r


def equal_count_shell_indices(d_spacings: Tensor, bin_size: int) -> Tensor:
    """Shell id per reflection (0..n_bins-1), matching ``stats_report`` equal-count bins.

    Reflections sorted by descending d (low → high resolution); each shell has
    up to ``bin_size`` reflections. Invalid / non-finite d get shell id ``-1``.
    """
    d = d_spacings.reshape(-1)
    n = int(d.numel())
    out = torch.full((n,), -1, dtype=torch.int64, device=d.device)
    valid = torch.isfinite(d)
    if not bool(valid.any()):
        return out
    idx = torch.where(valid)[0]
    order = idx[torch.argsort(-d[idx])]
    bs = max(1, int(bin_size))
    # shell = position_in_sorted // bin_size
    pos = torch.arange(order.numel(), device=d.device, dtype=torch.int64)
    shells = pos // bs
    out[order] = shells
    return out


def _mask_all_ones(n: int, device: Any, dtype_bool: Any = torch.bool) -> Tensor:
    return torch.ones(n, dtype=dtype_bool, device=device)


from phridge.d.rules.mli_quad.rint_report import IntegratedSResult, integrated_s_report

# Re-export normalize for callers assembling Ec/Zo from raw intensities
__all__ = [
    "PRIOR_ONLY_SIGMA_Z",
    "AmplitudeMomentsCache",
    "IntegratedSOptions",
    "IntegratedSResult",
    "absdev_given_k",
    "aggregate_s",
    "delta_given_cache",
    "predictive_delta_variance",
    "equal_count_shell_indices",
    "integrated_s_report",
    "normalize",
    "posterior_amplitude_cache",
    "scale_k",
    "sqdev_given_k",
]
