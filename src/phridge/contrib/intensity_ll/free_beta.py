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
The likelihood integrands (:mod:`phridge.contrib.intensity_ll.mli`) take ``(E_C, sigma_A)``
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
