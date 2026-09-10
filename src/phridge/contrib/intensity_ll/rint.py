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

from phridge.contrib.intensity_ll.maps import _fom_at
from phridge.contrib.intensity_ll.mli import _loggamma_rule, normalize, quadrature_terms_normal

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
            nu_t = torch.as_tensor(nu, dtype=Ec.dtype, device=Ec.device).expand_as(Ec)
            J = max(int(n_legendre), int(n_hermite))
            nodes = torch.ones(Ec.shape[0], n_u, J, dtype=Ec.dtype, device=Ec.device)
            terms = torch.full((Ec.shape[0], n_u, J), -math.inf, dtype=Ec.dtype, device=Ec.device)
            nu_t = torch.nan_to_num(nu_t, nan=200.0, posinf=200.0, neginf=2.5)
            nu_t = nu_t.clamp(2.05, 500.0)
            for nu_val in torch.unique(nu_t).tolist():
                sel = nu_t == nu_val
                if not bool(sel.any()):
                    continue
                u, w = _loggamma_rule(float(nu_val), n_u)
                for k, (uk, wk) in enumerate(zip(u, w)):
                    wk_f = float(wk)
                    if not math.isfinite(wk_f) or wk_f <= 0.0:
                        continue
                    X, T, _st = quadrature_terms_normal(
                        Ec[sel],
                        sA[sel],
                        Zo[sel],
                        sZ[sel] * math.exp(-uk / 2),
                        centric_t[sel],
                        **q_kw,
                    )
                    nodes[sel, k, : X.shape[-1]] = X
                    terms[sel, k, : T.shape[-1]] = T + math.log(wk_f)

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


@dataclass
class IntegratedSResult:
    """Scalars + optional per-shell arrays for work / free / all."""

    s_post_work: float
    s_post_free: float
    s_post_all: float
    s_prior_work: float
    s_prior_free: float
    s_prior_all: float
    # k_S: least squares on posterior means; converges to sigma_A in the no-data limit
    k_s_work: float
    k_s_free: float
    k_s_all: float
    k_s_prior_work: float
    k_s_prior_free: float
    k_s_prior_all: float
    # Visible-numerator (plug-in) variant: residual at the posterior mean rather than
    # integrated over it, sum |<E>-k_S Ec| / sum <E>. Internal — never printed in the
    # default report; kept on the object so a suspect S_post can be diagnosed.
    s_vis_work: float = float("nan")
    s_vis_free: float = float("nan")
    s_vis_all: float = float("nan")
    # Visible / latent L2 split: v_vis = Σ(<E>-k E_C)², v_lat = Σ Var_post(E).
    # rho2 = v_vis/(v_vis+v_lat) is descriptive, not a calibrated test.
    v_vis_work: float = float("nan")
    v_vis_free: float = float("nan")
    v_vis_all: float = float("nan")
    v_lat_work: float = float("nan")
    v_lat_free: float = float("nan")
    v_lat_all: float = float("nan")
    rho2_work: float = float("nan")
    rho2_free: float = float("nan")
    rho2_all: float = float("nan")
    # Score test on the map coefficients (opt-in). 1-xi_work estimates the effective
    # fitted parameters per work reflection; xi_free > 1 is model error.
    xi_work: float = float("nan")
    xi_free: float = float("nan")
    xi_all: float = float("nan")
    omega: float = float("nan")  # xi_free / xi_work
    # Free-set shell statistics below this count are blanked (reported so the
    # client legend stays in sync with the worker instead of duplicating it)
    min_free_per_shell: int = 30
    # Reflections dropped because E_C is a normalization blow-up (tiny Wilson Σ)
    n_ec_outliers: int = 0
    ec_median: float = float("nan")
    ec_max: float = float("nan")
    # Per-shell (length = n_bins); NaN where empty
    bin_d_max: Optional[np.ndarray] = None
    bin_d_min: Optional[np.ndarray] = None
    s_post_work_bins: Optional[np.ndarray] = None
    s_post_free_bins: Optional[np.ndarray] = None
    s_prior_work_bins: Optional[np.ndarray] = None
    s_prior_free_bins: Optional[np.ndarray] = None
    v_vis_work_bins: Optional[np.ndarray] = None
    v_vis_free_bins: Optional[np.ndarray] = None
    v_lat_work_bins: Optional[np.ndarray] = None
    v_lat_free_bins: Optional[np.ndarray] = None
    rho2_work_bins: Optional[np.ndarray] = None
    rho2_free_bins: Optional[np.ndarray] = None
    xi_work_bins: Optional[np.ndarray] = None
    xi_free_bins: Optional[np.ndarray] = None
    omega_bins: Optional[np.ndarray] = None
    n_work_bins: Optional[np.ndarray] = None
    n_free_bins: Optional[np.ndarray] = None

    def as_s_values(self) -> dict[str, Any]:
        def _r(x: float) -> float:
            v = float(x)
            return v if math.isfinite(v) and 0.0 <= v <= 5.0 else float("nan")

        def _s_list(arr: Optional[np.ndarray]) -> list[float]:
            if arr is None:
                return []
            return [_r(float(v)) for v in np.asarray(arr).tolist()]

        def _f(x: float) -> float:
            v = float(x)
            return v if math.isfinite(v) else float("nan")

        def _f_list(arr: Optional[np.ndarray]) -> list[float]:
            if arr is None:
                return []
            return [_f(float(v)) for v in np.asarray(arr).tolist()]

        out: dict[str, Any] = {
            "s_post_work": _r(self.s_post_work),
            "s_post_free": _r(self.s_post_free),
            "s_post_all": _r(self.s_post_all),
            "s_prior_work": _r(self.s_prior_work),
            "s_prior_free": _r(self.s_prior_free),
            "s_prior_all": _r(self.s_prior_all),
            "s_vis_work": _r(self.s_vis_work),
            "s_vis_free": _r(self.s_vis_free),
            "s_vis_all": _r(self.s_vis_all),
            "k_s_work": self.k_s_work,
            "k_s_free": self.k_s_free,
            "k_s_all": self.k_s_all,
            "k_s_prior_work": self.k_s_prior_work,
            "k_s_prior_free": self.k_s_prior_free,
            "k_s_prior_all": self.k_s_prior_all,
            "v_vis_work": _f(self.v_vis_work),
            "v_vis_free": _f(self.v_vis_free),
            "v_vis_all": _f(self.v_vis_all),
            "v_lat_work": _f(self.v_lat_work),
            "v_lat_free": _f(self.v_lat_free),
            "v_lat_all": _f(self.v_lat_all),
            "rho2_work": _f(self.rho2_work),
            "rho2_free": _f(self.rho2_free),
            "rho2_all": _f(self.rho2_all),
            "xi_work": _f(self.xi_work),
            "xi_free": _f(self.xi_free),
            "xi_all": _f(self.xi_all),
            "omega": _f(self.omega),
            "min_free_per_shell": int(self.min_free_per_shell),
            "n_ec_outliers": int(self.n_ec_outliers),
            "ec_median": self.ec_median,
            "ec_max": self.ec_max,
        }
        if self.bin_d_max is not None:
            out["bin_d_max"] = self.bin_d_max.tolist()
            out["bin_d_min"] = self.bin_d_min.tolist()  # type: ignore[union-attr]
            out["s_post_work_bins"] = _s_list(self.s_post_work_bins)
            out["s_post_free_bins"] = _s_list(self.s_post_free_bins)
            out["s_prior_work_bins"] = _s_list(self.s_prior_work_bins)
            out["s_prior_free_bins"] = _s_list(self.s_prior_free_bins)
            out["v_vis_work_bins"] = _f_list(self.v_vis_work_bins)
            out["v_vis_free_bins"] = _f_list(self.v_vis_free_bins)
            out["v_lat_work_bins"] = _f_list(self.v_lat_work_bins)
            out["v_lat_free_bins"] = _f_list(self.v_lat_free_bins)
            out["rho2_work_bins"] = _f_list(self.rho2_work_bins)
            out["rho2_free_bins"] = _f_list(self.rho2_free_bins)
            out["xi_work_bins"] = _f_list(self.xi_work_bins)
            out["xi_free_bins"] = _f_list(self.xi_free_bins)
            out["omega_bins"] = _f_list(self.omega_bins)
            out["n_work_bins"] = self.n_work_bins.tolist()  # type: ignore[union-attr]
            out["n_free_bins"] = self.n_free_bins.tolist()  # type: ignore[union-attr]
        return out


def _fill_shell_r(
    E_mean: Tensor,
    absdev: Tensor,
    shell_ids: Tensor,
    set_mask: Tensor,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    r_bins = np.full(n_bins, np.nan)
    n_bins_count = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        m = (shell_ids == b) & set_mask
        n_bins_count[b] = int(m.sum().item())
        if n_bins_count[b] == 0:
            continue
        r_bins[b] = aggregate_s(E_mean, absdev, m)
    return r_bins, n_bins_count


def integrated_s_report(
    Ec: Tensor,
    sA: Tensor,
    Zo: Tensor,
    sZ: Tensor,
    centric: Union[Tensor, bool],
    *,
    work_mask: Optional[Tensor] = None,
    free_mask: Optional[Tensor] = None,
    nu: Optional[Union[float, Tensor]] = None,
    d_spacings: Optional[Tensor] = None,
    bin_size: Optional[int] = None,
    n_u: int = 12,
    snr_strong: float = 5.0,
    n_hermite: int = 7,
    n_legendre: int = 32,
    k_window: float = 8.0,
    prior_only_sigma_z: float = PRIOR_ONLY_SIGMA_Z,
    chunk_size: Optional[int] = 4096,
    ec_outlier_ratio: float = 50.0,
    min_free_per_shell: int = 30,
    score_test: bool = False,
    n_predictive_nodes: int = 12,
) -> IntegratedSResult:
    """Compute overall (and optional per-shell) ``S_post`` / ``S_prior``.

    Shells reuse the parent set's ``k`` (work shells use ``k_s_work``, etc.).
    Empty free set → ``s_post_free`` / ``s_prior_free`` are NaN (warning only if a
    free mask was supplied explicitly).

    For large ``N``, reflections are processed in chunks of ``chunk_size`` so the
    node cache stays bounded; each chunk re-runs the mode finder (reporting only).

    Alongside the R values it reports the visible / latent L2 split
    ``v_vis = sum (<E> - k E_C)^2`` and ``v_lat = sum Var_post(E)``, whose sum is
    exactly ``sum <(E - k E_C)^2>``. The share ``rho2 = v_vis / (v_vis + v_lat)``
    is **descriptive, not a calibrated test**: in the weak-data limit the posterior
    mean carries the Rice Jacobian lift, so ``rho2`` does not go to zero even when
    the data carry no information.

    ``min_free_per_shell`` blanks every free-set shell statistic below that count
    (silently — a sparse free set is expected, not an error).

    ``score_test`` additionally reports ``xi = sum |Delta|^2 / sum E_model[|Delta|^2]``
    on the map coefficients and ``omega = xi_free / xi_work``. It is off by default
    because the reference variance costs ``n_predictive_nodes`` extra posterior solves
    per reflection. Free-set values are the honest ones; work-set values are
    optimistically biased by fitting, so ``1 - xi_work`` estimates the effective fitted
    parameters per work reflection while ``xi_free > 1`` is model error. Under the null
    ``omega`` has predicted value ``(1 + p_eff/N_work) / (1 - p_eff/N_work)``.

    ``ec_outlier_ratio`` drops reflections with ``E_C`` above that multiple of the
    median ``E_C``. These come from a near-zero Wilson :math:`\\Sigma` (normalized
    units are meaningless there), and because ``k = \\sum\\langle E\\rangle E_C /
    \\sum E_C^2`` is least-squares, a handful of them drive ``k`` to zero and pin
    every ``S_post`` at 1. Set to 0 to disable.
    """
    Ec, sA, Zo, sZ = torch.broadcast_tensors(
        torch.as_tensor(Ec, dtype=torch.float64),
        torch.as_tensor(sA, dtype=torch.float64),
        torch.as_tensor(Zo, dtype=torch.float64),
        torch.as_tensor(sZ, dtype=torch.float64),
    )
    # Collapse nearly-constant per-reflection ν to a scalar (Student-t mixture cost).
    # Keep a full-length tensor when ν varies (bin-wise fit); chunks must slice it.
    nu_per_refl: Optional[Tensor] = None
    nu_scalar: Optional[float] = None
    if nu is not None and not isinstance(nu, (int, float)):
        nu_t = torch.as_tensor(nu, dtype=torch.float64).reshape(-1)
        if nu_t.numel() == Ec.numel():
            uniq = torch.unique(nu_t)
            if uniq.numel() == 1 or float((uniq.max() - uniq.min()).item()) < 1e-6:
                nu_scalar = float(uniq[0].item())
            else:
                nu_per_refl = nu_t
        else:
            nu_scalar = float(nu_t.reshape(-1)[0].item())
    elif nu is not None:
        nu_scalar = float(nu)

    n = int(Ec.numel())
    device = Ec.device
    centric_t = torch.as_tensor(centric, device=device).bool().expand_as(Ec)
    explicit_free = free_mask is not None
    if work_mask is None and free_mask is None:
        work = _mask_all_ones(n, device)
        free = torch.zeros(n, dtype=torch.bool, device=device)
    elif free_mask is None:
        work = work_mask.bool()
        free = torch.zeros(n, dtype=torch.bool, device=device)
    elif work_mask is None:
        free = free_mask.bool()
        work = ~free
    else:
        work = work_mask.bool()
        free = free_mask.bool()

    # A near-zero Wilson Σ makes E_C = |F_c|/sqrt(εΣ) explode. The least-squares k
    # weights by E_C², so a few such reflections collapse k to ~0 and pin S_post at 1
    # (the prior-only floor is immune, since <E>_prior scales with E_C).
    keep = torch.isfinite(Ec) & (Ec >= 0)
    n_ec_outliers = 0
    ec_median = float("nan")
    ec_max = float("nan")
    considered = keep & (work | free) & (Ec > 0)
    if bool(considered.any()):
        ec_median = float(Ec[considered].median().item())
        ec_max = float(Ec[considered].max().item())
        if ec_outlier_ratio > 0 and math.isfinite(ec_median) and ec_median > 0:
            bad = keep & (Ec > ec_outlier_ratio * ec_median)
            n_ec_outliers = int((bad & (work | free)).sum().item())
            if n_ec_outliers:
                warnings.warn(
                    f"integrated_s_report: dropped {n_ec_outliers} reflections with "
                    f"E_C > {ec_outlier_ratio}x median ({ec_median:.3g}); "
                    "check the Wilson Sigma for a near-zero floor",
                    stacklevel=2,
                )
                keep = keep & ~bad
    work = work & keep
    free = free & keep

    kw_base = dict(
        n_u=n_u,
        snr_strong=snr_strong,
        n_hermite=n_hermite,
        n_legendre=n_legendre,
        k_window=k_window,
        prior_only_sigma_z=prior_only_sigma_z,
    )
    cs = int(chunk_size) if chunk_size is not None and chunk_size > 0 else n
    cs = max(1, min(cs, n))

    def _nu_chunk(sl: slice) -> Optional[Union[float, Tensor]]:
        if nu_per_refl is not None:
            return nu_per_refl[sl]
        return nu_scalar

    # Single pass: collect <E> first (need k), then absdev on a second cache pass.
    # Both passes use identical inputs; we keep caches short-lived per chunk for memory.
    E_mean = torch.empty(n, dtype=torch.float64, device=device)
    E_mean_prior = torch.empty(n, dtype=torch.float64, device=device)
    var_post = torch.empty(n, dtype=torch.float64, device=device)
    delta_obs = torch.empty(n, dtype=torch.float64, device=device)
    ref_var = torch.empty(n, dtype=torch.float64, device=device)
    for start in range(0, n, cs):
        stop = min(start + cs, n)
        sl = slice(start, stop)
        nu_c = _nu_chunk(sl)
        post = posterior_amplitude_cache(
            Ec[sl], sA[sl], Zo[sl], sZ[sl], centric_t[sl], nu=nu_c, prior_only=False, **kw_base
        )
        prior = posterior_amplitude_cache(
            Ec[sl], sA[sl], Zo[sl], sZ[sl], centric_t[sl], nu=nu_c, prior_only=True, **kw_base
        )
        E_mean[sl] = post.E_mean
        E_mean_prior[sl] = prior.E_mean
        var_post[sl] = post.variance()
        if score_test:
            delta_obs[sl] = delta_given_cache(post)
            ref_var[sl] = predictive_delta_variance(
                Ec[sl],
                sA[sl],
                sZ[sl],
                centric_t[sl],
                nu=nu_c,
                n_z=int(n_predictive_nodes),
                **kw_base,
            )

    def _scale(E_m: Tensor, mask: Tensor) -> float:
        if not bool(mask.any()):
            if explicit_free and mask is free:
                warnings.warn("scale_k: empty free set; returning nan", stacklevel=3)
            return float("nan")
        return scale_k(E_m, Ec, mask)

    k_w = _scale(E_mean, work)
    k_f = _scale(E_mean, free)
    k_a = scale_k(E_mean, Ec, keep)
    ki_w = _scale(E_mean_prior, work)
    ki_f = _scale(E_mean_prior, free)
    ki_a = scale_k(E_mean_prior, Ec, keep)

    abs_w = torch.empty(n, dtype=torch.float64, device=device)
    abs_f = torch.empty(n, dtype=torch.float64, device=device)
    abs_a = torch.empty(n, dtype=torch.float64, device=device)
    absi_w = torch.empty(n, dtype=torch.float64, device=device)
    absi_f = torch.empty(n, dtype=torch.float64, device=device)
    absi_a = torch.empty(n, dtype=torch.float64, device=device)

    for start in range(0, n, cs):
        stop = min(start + cs, n)
        sl = slice(start, stop)
        nu_c = _nu_chunk(sl)
        post = posterior_amplitude_cache(
            Ec[sl], sA[sl], Zo[sl], sZ[sl], centric_t[sl], nu=nu_c, prior_only=False, **kw_base
        )
        prior = posterior_amplitude_cache(
            Ec[sl], sA[sl], Zo[sl], sZ[sl], centric_t[sl], nu=nu_c, prior_only=True, **kw_base
        )
        # Prefer pass-1 <E> for plug-in consistency (same k, same mean)
        post.E_mean = E_mean[sl]
        prior.E_mean = E_mean_prior[sl]
        abs_w[sl] = absdev_given_k(post, k_w)
        abs_f[sl] = absdev_given_k(post, k_f)
        abs_a[sl] = absdev_given_k(post, k_a)
        absi_w[sl] = absdev_given_k(prior, ki_w)
        absi_f[sl] = absdev_given_k(prior, ki_f)
        absi_a[sl] = absdev_given_k(prior, ki_a)

    def _agg(E_m: Tensor, absdev: Tensor, mask: Tensor) -> float:
        if not bool(mask.any()):
            if explicit_free and mask is free:
                warnings.warn("aggregate_s: empty free set; returning nan", stacklevel=3)
            return float("nan")
        return aggregate_s(E_m, absdev, mask)

    s_post_w = _agg(E_mean, abs_w, work)
    s_post_f = _agg(E_mean, abs_f, free)
    s_post_a = aggregate_s(E_mean, abs_a, keep)
    s_prior_w = _agg(E_mean_prior, absi_w, work)
    s_prior_f = _agg(E_mean_prior, absi_f, free)
    s_prior_a = aggregate_s(E_mean_prior, absi_a, keep)

    def _plug(E_m: Tensor, k: float, mask: Tensor) -> float:
        if not math.isfinite(k) or not bool(mask.any()):
            return float("nan")
        return aggregate_s(E_m, (E_m - k * Ec).abs(), mask)

    def _split(k: float, mask: Tensor) -> tuple[float, float, float]:
        """``(v_vis, v_lat, rho2)`` on the masked set; NaN when empty or ``k`` is bad."""
        if not math.isfinite(k) or not bool(mask.any()):
            return float("nan"), float("nan"), float("nan")
        sel = mask & torch.isfinite(E_mean) & torch.isfinite(var_post) & torch.isfinite(Ec)
        if not bool(sel.any()):
            return float("nan"), float("nan"), float("nan")
        v_vis = float(((E_mean[sel] - k * Ec[sel]) ** 2).sum().item())
        v_lat = float(var_post[sel].sum().item())
        total = v_vis + v_lat
        rho2 = v_vis / total if total > 0 and math.isfinite(total) else float("nan")
        return v_vis, v_lat, rho2

    vv_w, vl_w, rho2_w = _split(k_w, work)
    vv_f, vl_f, rho2_f = _split(k_f, free)
    vv_a, vl_a, rho2_a = _split(k_a, keep)

    def _xi(mask: Tensor) -> float:
        """``sum |Delta|^2 / sum E_model[|Delta|^2]`` on the masked set."""
        if not score_test or not bool(mask.any()):
            return float("nan")
        sel = mask & torch.isfinite(delta_obs) & torch.isfinite(ref_var) & (ref_var > 0)
        if not bool(sel.any()):
            return float("nan")
        den = float(ref_var[sel].sum().item())
        if not math.isfinite(den) or den <= 0:
            return float("nan")
        val = float((delta_obs[sel] ** 2).sum().item()) / den
        return val if math.isfinite(val) else float("nan")

    def _omega(xi_f: float, xi_w: float) -> float:
        if not (math.isfinite(xi_f) and math.isfinite(xi_w)) or xi_w <= 0:
            return float("nan")
        return xi_f / xi_w

    xi_w = _xi(work)
    xi_f = _xi(free)
    xi_a = _xi(keep)

    result = IntegratedSResult(
        s_post_work=s_post_w,
        s_post_free=s_post_f,
        s_post_all=s_post_a,
        s_prior_work=s_prior_w,
        s_prior_free=s_prior_f,
        s_prior_all=s_prior_a,
        k_s_work=k_w,
        k_s_free=k_f,
        k_s_all=k_a,
        k_s_prior_work=ki_w,
        k_s_prior_free=ki_f,
        k_s_prior_all=ki_a,
        s_vis_work=_plug(E_mean, k_w, work),
        s_vis_free=_plug(E_mean, k_f, free) if bool(free.any()) else float("nan"),
        s_vis_all=_plug(E_mean, k_a, keep),
        v_vis_work=vv_w,
        v_vis_free=vv_f,
        v_vis_all=vv_a,
        v_lat_work=vl_w,
        v_lat_free=vl_f,
        v_lat_all=vl_a,
        rho2_work=rho2_w,
        rho2_free=rho2_f,
        rho2_all=rho2_a,
        xi_work=xi_w,
        xi_free=xi_f,
        xi_all=xi_a,
        omega=_omega(xi_f, xi_w),
        min_free_per_shell=int(min_free_per_shell),
        n_ec_outliers=n_ec_outliers,
        ec_median=ec_median,
        ec_max=ec_max,
    )

    if d_spacings is not None and bin_size is not None and int(bin_size) >= 1:
        d = torch.as_tensor(d_spacings, dtype=torch.float64, device=device).reshape(-1)
        shells = equal_count_shell_indices(d, int(bin_size))
        n_bins = int(shells.max().item()) + 1 if bool((shells >= 0).any()) else 0
        if n_bins > 0:
            d_max = np.full(n_bins, np.nan)
            d_min = np.full(n_bins, np.nan)
            for b in range(n_bins):
                in_b = shells == b
                if bool(in_b.any()):
                    d_max[b] = float(d[in_b].max().item())
                    d_min[b] = float(d[in_b].min().item())
            rw, nw = _fill_shell_r(E_mean, abs_w, shells, work, n_bins)
            rf, nf = _fill_shell_r(E_mean, abs_f, shells, free, n_bins)
            riw, _ = _fill_shell_r(E_mean_prior, absi_w, shells, work, n_bins)
            rif, _ = _fill_shell_r(E_mean_prior, absi_f, shells, free, n_bins)
            vvw = np.full(n_bins, np.nan)
            vlw = np.full(n_bins, np.nan)
            r2w = np.full(n_bins, np.nan)
            vvf = np.full(n_bins, np.nan)
            vlf = np.full(n_bins, np.nan)
            r2f = np.full(n_bins, np.nan)
            xiw = np.full(n_bins, np.nan)
            xif = np.full(n_bins, np.nan)
            omg = np.full(n_bins, np.nan)
            for b in range(n_bins):
                in_b = shells == b
                vvw[b], vlw[b], r2w[b] = _split(k_w, in_b & work)
                vvf[b], vlf[b], r2f[b] = _split(k_f, in_b & free)
                xiw[b] = _xi(in_b & work)
                xif[b] = _xi(in_b & free)

            # A sparse free set is expected, not an error: blank the free-set shell
            # statistics below the threshold rather than printing noise, and do it
            # silently. Counts stay real so the reader can see why.
            thin = nf < max(int(min_free_per_shell), 0)
            for arr in (rf, rif, vvf, vlf, r2f, xif):
                arr[thin] = np.nan
            for b in range(n_bins):
                omg[b] = _omega(float(xif[b]), float(xiw[b]))

            result.bin_d_max = d_max
            result.bin_d_min = d_min
            result.s_post_work_bins = rw
            result.s_post_free_bins = rf
            result.s_prior_work_bins = riw
            result.s_prior_free_bins = rif
            result.v_vis_work_bins = vvw
            result.v_vis_free_bins = vvf
            result.v_lat_work_bins = vlw
            result.v_lat_free_bins = vlf
            result.rho2_work_bins = r2w
            result.rho2_free_bins = r2f
            result.xi_work_bins = xiw
            result.xi_free_bins = xif
            result.omega_bins = omg
            result.n_work_bins = nw
            result.n_free_bins = nf

    return result


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
