"""Independent per-shell ``sigma_A`` and ``beta`` for stage 2 of the nuisance fit.

In normalized units (``Z_o = I/(epsilon Sigma_W)``, ``E_C = |F_c|/sqrt(epsilon Sigma_W)``)
the Rice first moment is exactly

    E[Z_o | E_C] = sigma_A^2 * E_C^2 + beta

so within a resolution shell ``sigma_A^2`` is the **slope** and ``beta`` the **intercept**
of observed against model intensity. The classical constraint ``beta = 1 - sigma_A^2``
forces that line through ``(1, 1)``, which is precisely the statement that the Wilson
normalization is exact. When it is not -- residual anisotropy, an inadequate bulk-solvent
model, tNCS -- the constrained fit can only reach the data by tilting the slope, and the
normalization error is laundered into ``sigma_A``. Measured: with ``beta`` inflated 1.5x,
the constrained fit returns ``sigma_A ~ 0.79`` against a true ``0.85``, while the free fit
recovers both. Where the normalization is right, the two agree, so freeing ``beta`` costs
nothing and removes a bias that is otherwise invisible.

``sigma_A`` is also no longer forced to be monotone. It is commonly depressed at low
resolution where the solvent model is poor, and can dip mid-range (ice rings, detector
artifacts); a cumulative-drop parameterization cannot represent that and pushes the
structure into neighbouring shells instead. Smoothness across shells replaces it, as a
second-difference penalty on the logits -- a statement that the profile is smooth, which
is defensible, rather than that it never rises, which is not.

Exactness of the free fit
-------------------------
The likelihood integrands (:mod:`phridge.d.rules.mli_quad`) take ``(E_C, sigma_A)``
and use them **only** through the product ``sigma_A * E_C`` and through the prior variance
``a = 1 - sigma_A^2``. A free ``(sigma_A, beta)`` pair is therefore reached exactly, with no
change to the quadrature, target or gradients, by transforming the inputs:

    sigma_A_eff = sqrt(1 - beta)
    E_C_eff     = sigma_A * E_C / sqrt(1 - beta)

which gives ``a = beta`` and ``sigma_A_eff * E_C_eff = sigma_A * E_C``, hence a Rice prior
with ``E[E^2] = sigma_A^2 E_C^2 + beta``. Verified to machine precision for both parities
(the prior integrates to 1 and its second moment is the target) in
``tests/contrib/test_free_beta.py::test_the_free_beta_transform_is_exact``. See
:func:`rice_inputs`. Note this is a reparameterization at the call site, not an
approximation, and the constrained case ``beta = 1 - sigma_A^2`` makes it the identity.

Identifiability
---------------
With ``beta`` free, ``sigma_A`` and an overall ``F_c`` scale ``k`` are exactly degenerate:
only ``sigma_A^2 k^2`` enters the slope. Stage 2 must not refine a joint scale when beta is
free, and the op raises rather than reporting one of infinitely many split-ups.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np
from pydantic import BaseModel, Field

__all__ = [
    "BETA_MAX",
    "BETA_MIN",
    "BETA_LO",
    "BETA_SPAN",
    "BOUND_MARGIN",
    "BetaMode",
    "ShellErrors",
    "ShellFitOptions",
    "ShellTable",
    "SigmaAShape",
    "SIGMA_A_LO",
    "SIGMA_A_SPAN",
    "beta_from_logit",
    "describe_shells",
    "logit_from_beta",
    "logit_from_sigma_a",
    "moment_init_shells",
    "normalize_beta_mode",
    "normalize_sigma_a_shape",
    "rice_inputs",
    "second_difference_penalty",
    "shell_errors_from_hessian",
    "sigma_a_from_logit",
]

# sigma_A in (0.01, 0.999) and beta in (0.001, 0.999) via a sigmoid. The bounds are open
# so the logit stays finite; the margins keep the sigmoid off its flat tails, where the
# gradient collapses and a parameter initialized on a bound can never leave it.
SIGMA_A_LO = 0.01
SIGMA_A_SPAN = 0.989
BETA_LO = 0.001
BETA_SPAN = 0.998

# beta must stay strictly below 1 for sqrt(1 - beta) in the input transform.
BETA_MIN = BETA_LO
BETA_MAX = BETA_LO + BETA_SPAN

# A parameter this close to its bound is not being determined by the data; report it as
# pinned rather than as a fitted value.
BOUND_MARGIN = 0.02

BetaMode = Literal["free", "constrained"]
SigmaAShape = Literal["free", "monotone"]


class ShellFitOptions(BaseModel):
    """Options for the per-shell ``sigma_A`` / ``beta`` fit (stage 2).

    The smoothness strengths are regularization, so they are tuning choices: pick them on
    the tune set and never on the free/audit set, or the reported held-out statistics stop
    being held out.
    """

    model_config = {"extra": "forbid"}

    beta_mode: BetaMode = Field(
        default="free",
        description=(
            "'free' (default) fits beta per shell independently of sigma_A, so a Wilson "
            "normalization error cannot be laundered into sigma_A. 'constrained' restores "
            "the classical beta = Sigma_W (1 - sigma_A^2) and is kept so previously "
            "reported runs reproduce exactly."
        ),
    )
    sigma_a_shape: SigmaAShape = Field(
        default="free",
        description=(
            "'free' (default) fits one independent logit per shell, regularized by "
            "smoothness. 'monotone' restores the cumulative-drop parameterization, which "
            "cannot represent a low-resolution or mid-range dip in sigma_A."
        ),
    )
    lambda_u: float = Field(
        default=1.0,
        ge=0.0,
        description="Second-difference smoothness on the sigma_A logits across shells.",
    )
    lambda_v: float = Field(
        default=1.0,
        ge=0.0,
        description="Second-difference smoothness on the beta logits across shells.",
    )
    lambda_consistency: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Strength of an optional soft prior pulling beta toward 1 - sigma_A^2. Zero "
            "(default) because beta is meant to be independent; raise it to pull weak data "
            "toward the constrained solution when the joint fit is poorly determined."
        ),
    )

    @property
    def beta_free(self) -> bool:
        return self.beta_mode == "free"

    @property
    def monotone(self) -> bool:
        return self.sigma_a_shape == "monotone"


def normalize_beta_mode(value: Any) -> BetaMode:
    """Coerce op JSON into a valid ``beta_mode``. Unset or unrecognized -> free."""
    text = str(value or "").strip().lower()
    if text in ("constrained", "constrain", "classical", "tied", "derived"):
        return "constrained"
    return "free"


def normalize_sigma_a_shape(value: Any) -> SigmaAShape:
    """Coerce op JSON into a valid ``sigma_a_shape``. Unset or unrecognized -> free."""
    text = str(value or "").strip().lower()
    if text in ("monotone", "monotonic", "decreasing", "cumulative"):
        return "monotone"
    return "free"


def _clamp(x: Any, lo: float, hi: float) -> Any:
    """Clamp a numpy array or a torch tensor without importing torch."""
    if hasattr(x, "clamp"):
        return x.clamp(lo, hi)
    return np.clip(x, lo, hi)


def sigma_a_from_logit(u: Any) -> Any:
    """``sigma_A = 0.01 + 0.989 * sigmoid(u)``, numpy side."""
    return SIGMA_A_LO + SIGMA_A_SPAN / (1.0 + np.exp(-np.asarray(u, dtype=np.float64)))


def beta_from_logit(v: Any) -> Any:
    """``beta = 0.001 + 0.998 * sigmoid(v)`` in normalized units, numpy side."""
    return BETA_LO + BETA_SPAN / (1.0 + np.exp(-np.asarray(v, dtype=np.float64)))


def logit_from_sigma_a(sigma_a: Any) -> np.ndarray:
    """Inverse of :func:`sigma_a_from_logit`, clipped off the bounds."""
    frac = (np.asarray(sigma_a, dtype=np.float64) - SIGMA_A_LO) / SIGMA_A_SPAN
    frac = np.clip(frac, 1e-4, 1.0 - 1e-4)
    return np.log(frac / (1.0 - frac))


def logit_from_beta(beta: Any) -> np.ndarray:
    """Inverse of :func:`beta_from_logit`, clipped off the bounds."""
    frac = (np.asarray(beta, dtype=np.float64) - BETA_LO) / BETA_SPAN
    frac = np.clip(frac, 1e-4, 1.0 - 1e-4)
    return np.log(frac / (1.0 - frac))


def rice_inputs(e_c: Any, sigma_a: Any, beta: Any) -> tuple[Any, Any]:
    """Map a free ``(sigma_A, beta)`` pair onto the inputs the existing quadrature takes.

    Returns ``(E_C_eff, sigma_A_eff)`` such that calling the unmodified likelihood with
    them is *exactly* the Rice model with ``E[Z_o | E_C] = sigma_A^2 E_C^2 + beta``. The
    integrands use ``(E_C, sigma_A)`` only via the product ``sigma_A E_C`` and via
    ``a = 1 - sigma_A^2``, so setting ``sigma_A_eff = sqrt(1 - beta)`` makes ``a = beta``
    and rescaling ``E_C`` restores the product. Differentiable, so autodiff reaches both
    parameters through it; works on numpy arrays and torch tensors alike.

    When ``beta = 1 - sigma_A^2`` this is the identity, which is why the constrained path
    does not route through it: ``sqrt(1 - (1 - sigma_A^2))`` is not bit-identical to
    ``sigma_A``, and the constrained path has to reproduce old output exactly.
    """
    b = _clamp(beta, BETA_MIN, BETA_MAX)
    sa_eff = (1.0 - b) ** 0.5
    return sigma_a * e_c / sa_eff, sa_eff


def second_difference_penalty(x: Any) -> Any:
    """``sum_k (x[k-1] - 2 x[k] + x[k+1])^2`` -- curvature, not slope.

    Penalizing the second difference leaves any straight trend through the shells free and
    charges only for kinks, so a genuinely falling sigma_A is not pulled flat. Fewer than
    three shells have no second difference and the penalty is zero.
    """
    if x.shape[0] < 3:
        return x.sum() * 0.0
    d2 = x[:-2] - 2.0 * x[1:-1] + x[2:]
    return (d2**2).sum()


@dataclass(frozen=True)
class ShellErrors:
    """Standard errors and correlation of ``(sigma_A, beta)`` per shell."""

    sigma_a_se: np.ndarray
    beta_se: np.ndarray
    correlation: np.ndarray
    ok: bool

    def as_json(self) -> dict[str, Any]:
        return {
            "sigma_a_se": [_finite_or_none(v) for v in self.sigma_a_se],
            "beta_se": [_finite_or_none(v) for v in self.beta_se],
            "sigma_a_beta_correlation": [_finite_or_none(v) for v in self.correlation],
            "errors_ok": bool(self.ok),
        }


def _finite_or_none(value: Any) -> Optional[float]:
    v = float(value)
    return v if math.isfinite(v) else None


def shell_errors_from_hessian(
    hessian: np.ndarray, sigma_a: np.ndarray, beta: np.ndarray
) -> ShellErrors:
    """Delta-method standard errors from the Hessian of the tune NLL in logit space.

    ``hessian`` is the ``(2K, 2K)`` second derivative of the **unpenalized** tune-set NLL
    with respect to ``[u, v]`` at the optimum -- unpenalized because these answer "how well
    do the data pin this shell", and the smoothness penalty is not data.

    A Hessian that is not positive definite means some direction is not determined; those
    shells get NaN rather than a number that looks like an error bar.
    """
    k = len(sigma_a)
    h = np.asarray(hessian, dtype=np.float64)
    nan = np.full(k, np.nan, dtype=np.float64)
    if h.shape != (2 * k, 2 * k) or not np.all(np.isfinite(h)):
        return ShellErrors(nan, nan.copy(), nan.copy(), False)
    with np.errstate(all="ignore"):
        try:
            cov = np.linalg.inv(h)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(h)
        if not np.all(np.isfinite(cov)):
            return ShellErrors(nan, nan.copy(), nan.copy(), False)
        var_u = np.diag(cov)[:k]
        var_v = np.diag(cov)[k:]
        cov_uv = np.array([cov[i, k + i] for i in range(k)], dtype=np.float64)
        # dsigma_A/du = SPAN * s(1-s) with s = (sigma_A - LO)/SPAN; likewise for beta.
        s_a = np.clip((np.asarray(sigma_a) - SIGMA_A_LO) / SIGMA_A_SPAN, 1e-12, 1 - 1e-12)
        s_b = np.clip((np.asarray(beta) - BETA_LO) / BETA_SPAN, 1e-12, 1 - 1e-12)
        j_u = SIGMA_A_SPAN * s_a * (1.0 - s_a)
        j_v = BETA_SPAN * s_b * (1.0 - s_b)
        bad = (var_u <= 0.0) | (var_v <= 0.0)
        se_a = np.where(bad, np.nan, j_u * np.sqrt(np.abs(var_u)))
        se_b = np.where(bad, np.nan, j_v * np.sqrt(np.abs(var_v)))
        denom = np.sqrt(np.abs(var_u) * np.abs(var_v))
        corr = np.where(denom > 0.0, cov_uv / np.maximum(denom, 1e-300), np.nan)
        corr = np.where(bad, np.nan, np.clip(corr, -1.0, 1.0))
    ok = bool(np.all(np.isfinite(se_a)) and np.all(np.isfinite(se_b)))
    return ShellErrors(se_a, se_b, corr, ok)


def moment_init_shells(
    z_obs: np.ndarray,
    e_c_sq: np.ndarray,
    sigma_z: np.ndarray,
    shell_idx: np.ndarray,
    n_shells: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-shell moment estimate of ``(sigma_A, beta)`` by weighted regression.

    Regresses ``Z_o`` on ``E_C^2`` within each shell weighted by ``1/sigma_Z^2``: the slope
    is ``sigma_A^2`` and the intercept is ``beta``, straight from the Rice first moment.
    Noisy, but it starts the ML fit near the answer in exactly the case that matters --
    a normalization that is off, where the constrained solution is a poor start and the
    joint fit would otherwise have to walk a long way with two correlated parameters.

    Shells that cannot support a regression (too few reflections, no spread in ``E_C^2``,
    or a fit that lands outside the bounds) fall back to a neutral ``(0.7, 0.5)``.
    """
    z = np.asarray(z_obs, dtype=np.float64)
    x = np.asarray(e_c_sq, dtype=np.float64)
    w = 1.0 / np.maximum(np.asarray(sigma_z, dtype=np.float64), 1e-12) ** 2
    idx = np.asarray(shell_idx, dtype=np.int64)
    sigma_a = np.full(int(n_shells), 0.7, dtype=np.float64)
    beta = np.full(int(n_shells), 0.5, dtype=np.float64)
    finite = np.isfinite(z) & np.isfinite(x) & np.isfinite(w)
    for k in range(int(n_shells)):
        sel = finite & (idx == k)
        if int(np.count_nonzero(sel)) < 8:
            continue
        xk, zk, wk = x[sel], z[sel], w[sel]
        sw = wk.sum()
        if not np.isfinite(sw) or sw <= 0.0:
            continue
        # weighted 2x2 normal equations for z = slope * x + intercept
        mx = float((wk * xk).sum() / sw)
        mz = float((wk * zk).sum() / sw)
        vxx = float((wk * (xk - mx) ** 2).sum() / sw)
        vxz = float((wk * (xk - mx) * (zk - mz)).sum() / sw)
        if not np.isfinite(vxx) or vxx <= 1e-12:
            continue
        slope = vxz / vxx
        intercept = mz - slope * mx
        if not (np.isfinite(slope) and np.isfinite(intercept)):
            continue
        sigma_a[k] = float(np.clip(math.sqrt(max(slope, 0.0)), 0.02, 0.98))
        beta[k] = float(np.clip(intercept, 0.02, 0.98))
    return sigma_a, beta


@dataclass(frozen=True)
class ShellTable:
    """Per-shell stage-2 result in reportable form."""

    centers_s_sq: np.ndarray
    sigma_a: np.ndarray
    beta: np.ndarray
    errors: ShellErrors
    beta_mode: str
    sigma_a_shape: str
    lambda_u: float
    lambda_v: float
    lambda_consistency: float

    @property
    def consistency(self) -> np.ndarray:
        """``sigma_A^2 + beta``, which is 1 exactly when the Wilson normalization is right."""
        return np.asarray(self.sigma_a, dtype=np.float64) ** 2 + np.asarray(
            self.beta, dtype=np.float64
        )

    @property
    def consistency_error(self) -> np.ndarray:
        return np.abs(self.consistency - 1.0)

    @property
    def at_bound(self) -> np.ndarray:
        """Shells where either parameter sits within :data:`BOUND_MARGIN` of a bound."""
        sa = np.asarray(self.sigma_a, dtype=np.float64)
        b = np.asarray(self.beta, dtype=np.float64)
        return (
            (sa <= SIGMA_A_LO + BOUND_MARGIN)
            | (sa >= SIGMA_A_LO + SIGMA_A_SPAN - BOUND_MARGIN)
            | (b <= BETA_LO + BOUND_MARGIN)
            | (b >= BETA_LO + BETA_SPAN - BOUND_MARGIN)
        )

    def as_json(self) -> dict[str, Any]:
        err = self.consistency_error
        out: dict[str, Any] = {
            "beta_mode": self.beta_mode,
            "sigma_a_shape": self.sigma_a_shape,
            "lambda_u": float(self.lambda_u),
            "lambda_v": float(self.lambda_v),
            "lambda_consistency": float(self.lambda_consistency),
            "bin_beta": [float(v) for v in self.beta],
            "bin_consistency": [float(v) for v in self.consistency],
            "consistency_mean_abs_error": float(np.mean(err)) if err.size else 0.0,
            "consistency_max_abs_error": float(np.max(err)) if err.size else 0.0,
            "n_at_bound": int(np.count_nonzero(self.at_bound)),
            "bins_at_bound": [int(i) for i in np.flatnonzero(self.at_bound)],
        }
        out.update(self.errors.as_json())
        return out


def describe_shells(
    centers_s_sq: Any,
    sigma_a: Any,
    beta: Any,
    errors: ShellErrors,
    options: ShellFitOptions,
) -> ShellTable:
    """Bundle the stage-2 shell result for reporting."""
    return ShellTable(
        centers_s_sq=np.asarray(centers_s_sq, dtype=np.float64),
        sigma_a=np.asarray(sigma_a, dtype=np.float64),
        beta=np.asarray(beta, dtype=np.float64),
        errors=errors,
        beta_mode=options.beta_mode,
        sigma_a_shape=options.sigma_a_shape,
        lambda_u=float(options.lambda_u),
        lambda_v=float(options.lambda_v),
        lambda_consistency=float(options.lambda_consistency),
    )


class ShellFitResult:
    """One LBFGS at a fixed ν. Arrays are copies, so a later refit cannot overwrite them."""

    def __init__(self, sigma_a: np.ndarray, beta: np.ndarray, nll_mean: float, n_tune: float) -> None:
        self.sigma_a = np.array(sigma_a, dtype=np.float64, copy=True)
        self.beta = np.array(beta, dtype=np.float64, copy=True)
        self.nll_mean = float(nll_mean)
        self.nll_sum = float(nll_mean) * float(n_tune)


class _ShellSession:
    """Per-shell σ_A / β LBFGS, including the second-difference penalty and the Rice tensor."""

    def __init__(
        self,
        *,
        fc: np.ndarray,
        io: np.ndarray,
        sig: np.ndarray,
        eps: np.ndarray,
        centric: np.ndarray,
        s_sq: np.ndarray,
        sigma_w: np.ndarray,
        tune: np.ndarray,
        s_cart: Optional[np.ndarray],
        shell: ShellFitOptions,
        n_bins: Optional[int],
        sphericity: float,
        sigma_a_tensor: bool,
        space_group_number: int,
    ) -> None:
        import torch

        from phridge.d.rules.mli_quad import log_likelihood_normal, log_likelihood_t, normalize
        from phridge.d.rules.mli_quad.aniso_rice import (
            laue_from_number,
            modulation_wanted,
            unit_directions,
        )
        from phridge.q.error_models.wilson_aniso import tensor_identifiability
        from phridge.sfcalc.ops import _DEVICE

        self.torch = torch
        self.normalize = normalize
        self.log_t = log_likelihood_t
        self.log_n = log_likelihood_normal
        self.shell = shell
        self.free_beta = bool(shell.beta_free and not shell.monotone or shell.beta_free)
        self.monotone = bool(shell.monotone)
        dev = _DEVICE["device"]
        self.device = dev
        self.dtype = torch.float32 if str(dev).startswith("mps") else torch.float64
        self.tune = tune
        self.n_tune = max(float(tune.sum()), 1.0)
        s_sq_t = s_sq[tune]
        self._build_shells(torch, fc, io, sig, eps, sigma_w, s_sq_t, n_bins)
        self._tensors(torch, fc, io, sig, eps, centric, sigma_w)
        self._rice(s_cart, sphericity, sigma_a_tensor, space_group_number, laue_from_number, modulation_wanted, unit_directions, tensor_identifiability)
        self.current_nu: Optional[float] = None
        self.opt: Any = None

    def _build_shells(
        self, torch: Any, fc: np.ndarray, io: np.ndarray, sig: np.ndarray, eps: np.ndarray,
        sigma_w: np.ndarray, s_sq_t: np.ndarray, n_bins: Optional[int],
    ) -> None:
        n_tune = int(self.tune.sum())
        self.n_sa = max(3, int(n_bins)) if n_bins is not None else int(min(20, max(6, n_tune // 250)))
        s_lo = float(s_sq_t.min())
        s_hi = float(s_sq_t.max())
        if s_hi - s_lo < 1e-8:
            self.edges = np.array([s_lo - 1e-6, s_hi + 1e-6], dtype=np.float64)
            self.n_sa = 1
        else:
            self.edges = np.linspace(s_lo, s_hi, self.n_sa + 1)
        idx = np.clip(np.digitize(s_sq_t, self.edges[1:-1], right=False), 0, self.n_sa - 1)
        self.shell_idx = torch.as_tensor(idx, dtype=torch.long, device=self.device)
        denom = np.maximum(eps[self.tune] * sigma_w[self.tune], 1e-30)
        sa0, b0 = moment_init_shells(
            z_obs=io[self.tune] / denom,
            e_c_sq=(fc[self.tune] ** 2) / denom,
            sigma_z=sig[self.tune] / denom,
            shell_idx=idx,
            n_shells=self.n_sa,
        )
        self.u_sa = torch.as_tensor(logit_from_sigma_a(sa0), dtype=self.dtype, device=self.device).clone().requires_grad_(True)
        self.v_beta = torch.as_tensor(logit_from_beta(b0), dtype=self.dtype, device=self.device).clone().requires_grad_(True)
        self.params: list[Any] = [self.u_sa, self.v_beta]

    def _tensors(
        self, torch: Any, fc: np.ndarray, io: np.ndarray, sig: np.ndarray, eps: np.ndarray,
        centric: np.ndarray, sigma_w: np.ndarray,
    ) -> None:
        tune = self.tune
        self.fc_t = torch.as_tensor(fc[tune], dtype=self.dtype, device=self.device)
        self.io_t = torch.as_tensor(io[tune], dtype=self.dtype, device=self.device)
        self.si_t = torch.as_tensor(sig[tune], dtype=self.dtype, device=self.device)
        self.eps_t = torch.as_tensor(eps[tune], dtype=self.dtype, device=self.device)
        self.cen_t = torch.as_tensor(centric[tune], device=self.device)
        self.sw_t = torch.as_tensor(sigma_w[tune], dtype=self.dtype, device=self.device)

    def _rice(
        self, s_cart: Optional[np.ndarray], sphericity: float, enabled: bool, space_group_number: int,
        laue_from_number: Any, modulation_wanted: Any, unit_directions: Any, tensor_identifiability: Any,
    ) -> None:
        import torch

        self.lam_sph = max(0.0, float(sphericity))
        self.laue = laue_from_number(int(space_group_number))
        ident = float(tensor_identifiability(s_cart[self.tune])) if s_cart is not None else 0.0
        self.do_rice, _why = modulation_wanted(
            enabled=bool(enabled),
            beta_free=bool(self.shell.beta_free),
            sigma_a_free=not self.monotone,
            bins_mode=True,
            laue=self.laue,
            s_cart=None if s_cart is None else s_cart[self.tune],
            identifiability=ident,
        )
        self.u_m_a = None
        self.u_m_b = None
        self.s_hat = None
        if self.do_rice and s_cart is not None and self.laue.n_free > 0:
            hat = unit_directions(s_cart)[self.tune]
            self.s_hat = torch.as_tensor(hat, dtype=self.dtype, device=self.device)
            self.u_m_a = torch.zeros(self.laue.n_free, dtype=self.dtype, device=self.device, requires_grad=True)
            self.u_m_b = torch.zeros(self.laue.n_free, dtype=self.dtype, device=self.device, requires_grad=True)
            self.params = list(self.params) + [self.u_m_a, self.u_m_b]

    def _sa_bins(self) -> Any:
        return SIGMA_A_LO + SIGMA_A_SPAN * self.torch.sigmoid(self.u_sa)

    def _beta_bins(self) -> Any:
        if self.shell.beta_free:
            return BETA_LO + BETA_SPAN * self.torch.sigmoid(self.v_beta)
        sa = self._sa_bins()
        return 1.0 - sa ** 2

    def _quad(self, u_m: Any) -> Any:
        from phridge.d.rules.mli_quad.aniso_rice import a_from_params_torch, quadratic_form_torch

        torch = self.torch
        a = a_from_params_torch(u_m, self.laue)
        m = torch.eye(3, dtype=self.dtype, device=self.device) + a
        return torch.clamp(quadratic_form_torch(self.s_hat, m), 1e-6, 1e6)

    def _sa_tune(self) -> Any:
        sa = self._sa_bins()[self.shell_idx]
        if self.u_m_a is None:
            return sa
        return self.torch.clamp(sa * self.torch.sqrt(self._quad(self.u_m_a)), 1e-4, 0.9999)

    def _beta_tune(self) -> Any:
        beta = self._beta_bins()[self.shell_idx]
        if self.u_m_b is None:
            return beta
        return self.torch.clamp(beta * self._quad(self.u_m_b), BETA_LO, BETA_LO + BETA_SPAN)

    def _nll_per_refl(self, nu_val: Optional[float]) -> Any:
        sa_t = self._sa_tune()
        ec, sa_n, zo, sz = self.normalize(self.fc_t, self.io_t, self.si_t, self.eps_t, self.sw_t, sa_t)
        if self.shell.beta_free:
            ec, sa_n = rice_inputs(ec, sa_t, self._beta_tune())
        if nu_val is not None and float(nu_val) < 199.0:
            ll = self.log_t(ec, sa_n, zo, sz, self.cen_t, nu=float(nu_val), n_u=10)
        else:
            ll = self.log_n(ec, sa_n, zo, sz, self.cen_t)
        return -ll

    def _penalty(self) -> Any:
        from phridge.d.rules.mli_quad.aniso_rice import a_from_params_torch, frobenius_sq

        pen = None
        if self.n_sa > 1 and self.shell.lambda_u > 0.0:
            pen = self.shell.lambda_u * second_difference_penalty(self.u_sa)
        if self.v_beta is not None and self.n_sa > 1 and self.shell.lambda_v > 0.0:
            term = self.shell.lambda_v * second_difference_penalty(self.v_beta)
            pen = term if pen is None else pen + term
        if self.do_rice and self.lam_sph > 0.0 and self.u_m_a is not None and self.u_m_b is not None:
            term = self.lam_sph * (
                frobenius_sq(a_from_params_torch(self.u_m_a, self.laue))
                + frobenius_sq(a_from_params_torch(self.u_m_b, self.laue))
            )
            pen = term if pen is None else pen + term
        return pen

    def _closure(self) -> Any:
        self.opt.zero_grad()
        loss = self._nll_per_refl(self.current_nu).sum()
        pen = self._penalty()
        if pen is not None:
            loss = loss + pen
        loss.backward()
        return loss

    def lbfgs(self, nu_val: Optional[float]) -> ShellFitResult:
        self.current_nu = None if nu_val is None else float(nu_val)
        self.opt = self.torch.optim.LBFGS(self.params, max_iter=40, line_search_fn="strong_wolfe")
        try:
            self.opt.step(self._closure)
        except Exception:
            pass
        with self.torch.no_grad():
            nll_i = self._nll_per_refl(self.current_nu)
            mean = float(nll_i.mean().item())
            sa = self._sa_bins().detach().cpu().numpy().astype(np.float64)
            beta = self._beta_bins().detach().cpu().numpy().astype(np.float64)
        result = ShellFitResult(sa, beta, mean, self.n_tune)
        result.edges = np.array(self.edges, dtype=np.float64, copy=True)
        result.laue = self.laue
        result.u_m_a = None if self.u_m_a is None else self.u_m_a.detach().cpu().numpy().astype(np.float64)
        result.u_m_b = None if self.u_m_b is None else self.u_m_b.detach().cpu().numpy().astype(np.float64)
        return result


class SigmaABeta:
    """Per-shell σ_A and β with Σ_W frozen. Logit shells, smoothness inside the objective."""

    def __init__(
        self,
        shell: Optional[ShellFitOptions] = None,
        *,
        n_bins: Optional[int] = None,
        sphericity: float = 1.0,
        sigma_a_tensor: bool = True,
        space_group_number: int = 1,
    ) -> None:
        self.shell = shell or ShellFitOptions()
        self.n_bins = n_bins
        self.sphericity = float(sphericity)
        self.sigma_a_tensor = bool(sigma_a_tensor)
        self.space_group_number = int(space_group_number)
        self.sigma_a_shell: Optional[np.ndarray] = None
        self.beta_shell: Optional[np.ndarray] = None
        self.last: Optional[ShellFitResult] = None

    def forward(self, location: Any, sigma_w: Any) -> Any:
        from phridge.contracts.tensors import Predictive
        import torch

        if self.sigma_a_shell is None or self.beta_shell is None:
            raise RuntimeError("SigmaABeta.forward needs fit() first")
        _ = (location, sigma_w)
        return Predictive({
            "sigma_a": torch.as_tensor(self.sigma_a_shell, dtype=torch.float64),
            "beta": torch.as_tensor(self.beta_shell, dtype=torch.float64),
        })

    def adjoint(self, cotangent: Any) -> Any:
        from phridge.contracts.tensors import Predictive

        return Predictive({"location": cotangent.fields["sigma_a"]})

    def fit(self, location: Any, obs: Any, mask: Any, nu: Optional[float], sigma_w: Any) -> Any:
        """One per-shell LBFGS at fixed ν. ν ≥ 199 is the Gaussian node."""
        import torch

        from phridge.contracts.tensors import Predictive
        from phridge.q.error_models.wilson_aniso import reciprocal_cartesian

        def _np(value: Any, dtype: Any) -> np.ndarray:
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            return np.asarray(value, dtype=dtype)

        fc = np.abs(_np(location, np.complex128)).reshape(-1)
        io = _np(obs.data["i_obs"], np.float64).reshape(-1)
        sig = _np(obs.data["sig_i"], np.float64).reshape(-1)
        eps = _np(obs.data["epsilon"], np.float64).reshape(-1)
        centric = _np(obs.data["centric"], bool).reshape(-1)
        s_sq = _np(obs.data["s_sq"], np.float64).reshape(-1)
        hkl = _np(obs.index["hkl"], np.int32).reshape(-1, 3)
        cell = _np(obs.data["unit_cell"], np.float64).reshape(-1)
        tune = _np(mask, bool).reshape(-1)
        sw = _np(sigma_w, np.float64).reshape(-1)
        s_cart = reciprocal_cartesian(tuple(float(x) for x in cell), hkl)
        session = _ShellSession(
            fc=fc, io=io, sig=sig, eps=eps, centric=centric, s_sq=s_sq, sigma_w=sw, tune=tune,
            s_cart=s_cart, shell=self.shell, n_bins=self.n_bins, sphericity=self.sphericity,
            sigma_a_tensor=self.sigma_a_tensor, space_group_number=self.space_group_number,
        )
        self.last = session.lbfgs(nu)
        self.sigma_a_shell = self.last.sigma_a
        self.beta_shell = self.last.beta
        return Predictive({
            "sigma_a": torch.as_tensor(self.sigma_a_shell, dtype=torch.float64),
            "beta": torch.as_tensor(self.beta_shell, dtype=torch.float64),
        })


def _shell_standard_errors(
    *,
    sigma_a: np.ndarray,
    beta: np.ndarray,
    free_beta: bool,
    shell_idx: Any,
    fc_t: Any,
    io_t: Any,
    si_t: Any,
    eps_t: Any,
    sw_t: Any,
    cen_t: Any,
    nu_tune: Any,
    nu_scalar: Optional[float],
    dtype: Any,
    device: Any,
) -> ShellErrors:
    """Hessian standard errors of per-shell (σ_A, β) from the unpenalized tune NLL."""
    import torch

    from phridge.d.rules.mli_quad import log_likelihood_normal, log_likelihood_t, normalize

    k = int(len(sigma_a))
    u = torch.as_tensor(logit_from_sigma_a(sigma_a), dtype=dtype, device=device).clone().requires_grad_(True)
    v = torch.as_tensor(logit_from_beta(beta), dtype=dtype, device=device).clone().requires_grad_(True)

    def _nll(u_vec: Any, v_vec: Any) -> Any:
        sa_bins = SIGMA_A_LO + SIGMA_A_SPAN * torch.sigmoid(u_vec)
        sa_t = sa_bins[shell_idx]
        ec, sa_n, zo, sz = normalize(fc_t, io_t, si_t, eps_t, sw_t, sa_t)
        if free_beta:
            b_bins = BETA_LO + BETA_SPAN * torch.sigmoid(v_vec)
            ec, sa_n = rice_inputs(ec, sa_t, b_bins[shell_idx])
        if nu_tune is not None:
            ll = log_likelihood_t(ec, sa_n, zo, sz, cen_t, nu=nu_tune, n_u=10)
        elif nu_scalar is not None and float(nu_scalar) < 199.0:
            ll = log_likelihood_t(ec, sa_n, zo, sz, cen_t, nu=float(nu_scalar), n_u=10)
        else:
            ll = log_likelihood_normal(ec, sa_n, zo, sz, cen_t)
        return -ll.sum()

    params = [u, v] if free_beta else [u]
    grads = torch.autograd.grad(_nll(u, v), params, create_graph=True)

    def _hvp(seeds: list[Any]) -> list[Any]:
        dot = sum((g * s).sum() for g, s in zip(grads, seeds))
        out = torch.autograd.grad(dot, params, retain_graph=True, allow_unused=True)
        return [torch.zeros_like(p) if o is None else o.detach() for p, o in zip(params, out)]

    ones = torch.ones(k, dtype=dtype, device=device)
    if free_beta:
        zeros = torch.zeros(k, dtype=dtype, device=device)
        col_u, col_v = _hvp([ones, zeros]), _hvp([zeros, ones])
        hess = np.zeros((2 * k, 2 * k), dtype=np.float64)
        h_uu = col_u[0].cpu().numpy().astype(np.float64)
        h_vu = col_u[1].cpu().numpy().astype(np.float64)
        h_vv = col_v[1].cpu().numpy().astype(np.float64)
        for i in range(k):
            hess[i, i] = h_uu[i]
            hess[k + i, k + i] = h_vv[i]
            hess[i, k + i] = hess[k + i, i] = h_vu[i]
        return shell_errors_from_hessian(hess, sigma_a, beta)
    h_uu = _hvp([ones])[0].cpu().numpy().astype(np.float64)
    with np.errstate(all="ignore"):
        var_u = np.where(h_uu > 0.0, 1.0 / np.maximum(h_uu, 1e-300), np.nan)
        frac = np.clip((np.asarray(sigma_a) - SIGMA_A_LO) / SIGMA_A_SPAN, 1e-12, 1 - 1e-12)
        se_sa = SIGMA_A_SPAN * frac * (1.0 - frac) * np.sqrt(var_u)
        se_beta = 2.0 * np.asarray(sigma_a, dtype=np.float64) * se_sa
    corr = np.where(np.isfinite(se_sa), -1.0, np.nan)
    return ShellErrors(se_sa, se_beta, corr, bool(np.all(np.isfinite(se_sa))))
