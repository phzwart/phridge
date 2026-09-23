"""Anisotropic Wilson normalization ``Sigma_W`` for the intensity likelihood.

The nuisance stack needs an expected intensity ``Sigma_W`` to normalize by. The
isotropic form

    Sigma_W(s) = Sigma_0 * exp(-0.5 * B_W * s^2),   s^2 = 1/d^2

cannot describe real data: at 1.85 A a modest 10 A^2 spread in B between principal
axes is already a factor ~4 difference in expected intensity between directions
(20 A^2 -> ~19x). This module replaces the scalar exponent with a symmetric tensor,

    Sigma_W(h) = Sigma_0 * exp(-0.5 * s_cart^T B s_cart)

where ``s_cart = O^-T h`` is the metric-correct Cartesian reciprocal vector
(``|s_cart| = 1/d``) and ``B`` is a positive semi-definite 3x3 tensor in A^2.

**The default is binned** (``wilson_model="binned"``):

    Sigma_W(h) = S_k * exp(-0.5 * s_cart^T B s_cart),   tr(B) = 0

``S_k`` is the mean of ``I/(epsilon A)`` over resolution bin ``k``, recomputed for every
trial ``B``, so each bin keeps its observed mean intensity exactly and the only thing
fitted is the global direction dependence. A single smooth curve (``"anisotropic"``,
``"isotropic"``) cannot follow structure between shells, and because stage 2 freezes
``Sigma_W`` that misfit is paid for in sigma_A and beta. Both curve forms remain
available but must be asked for. Where the reflection directions cannot determine a
tensor -- see :func:`tensor_identifiability` -- it is dropped and the fit says so in
``sigma_wilson_params["wilson_fallback"]`` rather than reporting an unconstrained one.


Why exactly one anisotropy carrier is free
------------------------------------------
An anisotropic ``Sigma_W`` and ``f_model``'s ``k_anisotropic`` are **degenerate**:
both absorb the same directional falloff, and fitting both leaves a well-fitting
pair of numbers that individually mean nothing. Exactly one of them may be free.

The settled decision is that **the normalization carries all data anisotropy and
the model-side anisotropic scale is constrained to isotropic**. The reason is that
``Sigma_W`` is the *prior*, and three downstream quantities are only interpretable
if the prior actually describes the observations' directional falloff:

* ``sigma_Z = sigma_I / (epsilon Sigma_W)`` and hence the reported ``data%``, which
  is otherwise too high in weak directions and too low in strong ones;
* the prior variance ``beta = Sigma_W (1 - sigma_A^2)``, mis-priced the same way,
  which biases the per-reflection posterior weights and so the map coefficients
  *directionally*;
* the per-shell ``sigma_A(s)``, a scalar, which silently absorbs whatever is left
  over as "model error".

With the prior carrying it, ``k_anisotropic`` has nothing legitimate left to carry.
:func:`phridge.client.intensity.engine.IntensityFModel._assert_single_anisotropy_carrier`
enforces this by measuring the anisotropy actually applied to ``F_calc`` and raising
if both carriers are live -- a flag would only record the intent, not the outcome.


Parameterization
----------------
``B = L L^T`` with ``L`` lower triangular and log-diagonal, so positive
semi-definiteness is automatic and LBFGS stays unconstrained -- the same trick the
isotropic fit already uses via ``log``/``softplus`` for ``Sigma_0`` and ``B_W``.

The tensor is carried in the **Cartesian** reciprocal frame rather than the
reciprocal-lattice one. Both are valid; Cartesian is better conditioned (entries are
O(B) ~ 20 rather than O(B/a^2) ~ 3e-3) and, more usefully, it makes the isotropic
start exact and trivial: ``B = B_W * I`` gives back ``exp(-0.5 B_W s^2)`` for any
unit cell, so the anisotropic fit begins at the isotropic solution and can only
improve the likelihood.


Reporting
---------
Raw components of ``B`` are basis-dependent and not comparable between data sets.
:class:`WilsonTensor` therefore reports the eigen-decomposition: ``b_iso`` (the
trace/3 gauge-fixed isotropic part), ``delta_b_aniso`` (the eigenvalue spread, which
is the comparable anisotropy), and the principal directions as direction cosines
against ``a*``, ``b*``, ``c*``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Sequence

import numpy as np
from pydantic import BaseModel, Field

# A B tensor whose eigenvalue spread is below this is isotropic for every practical
# purpose: at 1.5 A, 0.5 A^2 is a 6% intensity difference between principal axes.
ANISO_NEGLIGIBLE_DELTA_B = 0.5

# Below this relative eigenvalue of the normalized 6-parameter design Gram matrix the
# reflection directions do not span enough of the sphere to determine a tensor.
# Calibrated on real and degenerate sets: a full resolution sphere gives 0.40, a 14 deg
# cone about one axis 1.6e-4, and a line / plane / pair of axes numerically zero. So 1e-3
# separates "usable" from "cannot be determined" by a factor of 400 either way.
MIN_TENSOR_IDENTIFIABILITY = 1e-3

WilsonModel = Literal["binned", "isotropic", "anisotropic"]

# Models whose Sigma_W carries directional falloff, so k_anisotropic must be held isotropic.
_ANISOTROPIC_MODELS = ("binned", "anisotropic")


class WilsonOptions(BaseModel):
    """Options controlling the Wilson normalization fit (stage 1 of the nuisance fit)."""

    model_config = {"extra": "forbid"}

    wilson_model: WilsonModel = Field(
        default="binned",
        description=(
            "'binned' (default): Sigma_W(h) = S_k * exp(-0.5 s^T B s) with S_k the mean "
            "of I/(eps A) in resolution bin k and B a global traceless tensor, so every "
            "bin keeps its observed mean intensity and only the direction dependence is "
            "fitted. 'anisotropic' fits Sigma_0 and a PSD 3x3 B tensor (one smooth curve); "
            "'isotropic' fits Sigma_0 and a scalar B_W. Both anisotropic forms require the "
            "model-side k_anisotropic to be constrained to isotropic, since the two are "
            "degenerate; the tensor is dropped on reflection sets that cannot determine it."
        ),
    )
    max_iter: int = Field(
        default=60, ge=1, le=1000, description="LBFGS iterations for the stage-1 fit."
    )

    @property
    def anisotropic(self) -> bool:
        return self.wilson_model == "anisotropic"

    @property
    def binned(self) -> bool:
        return self.wilson_model == "binned"

    @property
    def carries_anisotropy(self) -> bool:
        return wilson_carries_anisotropy(self.wilson_model)


def wilson_carries_anisotropy(model: Any) -> bool:
    """Whether this Sigma_W form owns the data anisotropy (so k_anisotropic may not)."""
    return str(model) in _ANISOTROPIC_MODELS


def normalize_wilson_model(value: Any) -> WilsonModel:
    """Coerce op JSON into a valid ``wilson_model``. Unset or unrecognized -> binned.

    A single smooth curve cannot follow what real data do between shells (ice rings,
    detector gaps, the solvent hump), and freezing such a curve before sigma_A and beta
    are fitted costs more likelihood than the model explains. Per-bin means keep the
    observed intensity level in every shell; the global tensor adds only the directions.
    """
    text = str(value or "").strip().lower()
    if text in ("iso", "isotropic", "scalar"):
        return "isotropic"
    if text in ("anisotropic", "aniso", "tensor", "curve"):
        return "anisotropic"
    return "binned"


def traceless(b_cart: np.ndarray) -> np.ndarray:
    """``B - tr(B)/3 I``: the directional part only, the isotropic falloff removed."""
    b = 0.5 * (np.asarray(b_cart, dtype=np.float64) + np.asarray(b_cart, dtype=np.float64).T)
    return b - np.trace(b) / 3.0 * np.eye(3)


def binned_sigma_w(
    *,
    i_over_eps: np.ndarray,
    bin_id: np.ndarray,
    s_cart: Optional[np.ndarray],
    b_cart: Optional[np.ndarray],
    valid: Optional[np.ndarray] = None,
    floor: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``Sigma_W = S_k A(h)`` with ``S_k`` chosen so each bin's mean ``I/(eps Sigma_W)`` is 1.

    ``A(h) = exp(-0.5 s^T B s)`` with ``B`` made traceless, so the bins own all isotropic
    falloff and the tensor only redistributes intensity between directions inside a bin.
    Returns ``(sigma_w, bin_means)``. Numpy twin of the torch fit in the nuisance op.
    """
    y = np.asarray(i_over_eps, dtype=np.float64)
    idx = np.asarray(bin_id, dtype=np.int64)
    ok = np.isfinite(y) if valid is None else (np.asarray(valid, dtype=bool) & np.isfinite(y))
    if s_cart is not None and b_cart is not None:
        s = np.asarray(s_cart, dtype=np.float64)
        a = np.exp(-0.5 * np.einsum("ni,ij,nj->n", s, traceless(b_cart), s))
    else:
        a = np.ones_like(y)
    n_bins = int(idx.max()) + 1 if idx.size else 0
    sums = np.bincount(idx[ok], weights=(y / a)[ok], minlength=n_bins)
    counts = np.bincount(idx[ok], minlength=n_bins)
    means = sums / np.maximum(counts, 1)
    if floor is None:
        pos = means[means > 0]
        floor = 1e-3 * float(np.median(pos)) if pos.size else 1e-8
    means = np.maximum(means, float(floor))
    return np.maximum(means[idx] * a, 1e-12), means


def equal_count_bins(s_sq: np.ndarray, per_bin: int = 500, min_bins: int = 6, max_bins: int = 60) -> np.ndarray:
    """Resolution bins with equal reflection counts, used when the caller supplies none."""
    s = np.asarray(s_sq, dtype=np.float64)
    n = s.size
    n_bins = int(min(max_bins, max(min_bins, n // max(int(per_bin), 1))))
    n_bins = max(1, min(n_bins, n))
    order = np.argsort(s, kind="stable")
    ids = np.empty(n, dtype=np.int64)
    ids[order] = (np.arange(n) * n_bins) // max(n, 1)
    return ids


@dataclass
class WilsonTensor:
    """A fitted Wilson normalization, in the inspectable eigen-form.

    ``b_cart`` is kept for reconstruction but is deliberately not the primary output:
    its components depend on the basis, so they cannot be compared between data sets.
    ``eigenvalues`` and ``delta_b_aniso`` can.
    """

    sigma_0: float = float("nan")
    b_cart: tuple[float, ...] = (float("nan"),) * 6  # B11,B22,B33,B12,B13,B23 (A^2)
    eigenvalues: tuple[float, ...] = (float("nan"),) * 3  # ascending, A^2
    # Rows follow ``eigenvalues``; each is |cos| against a*, b*, c*.
    direction_cosines: tuple[tuple[float, ...], ...] = ()
    b_iso: float = float("nan")  # trace / 3
    delta_b_aniso: float = float("nan")  # max - min eigenvalue
    model: WilsonModel = "isotropic"
    nll_per_refl: float = float("nan")
    n_params: int = 2

    @property
    def is_valid(self) -> bool:
        return math.isfinite(self.sigma_0) and all(np.isfinite(e) for e in self.eigenvalues)

    @property
    def is_effectively_isotropic(self) -> bool:
        return (
            math.isfinite(self.delta_b_aniso)
            and self.delta_b_aniso < ANISO_NEGLIGIBLE_DELTA_B
        )

    def as_json(self) -> dict[str, Any]:
        """Flat, JSON-safe form for the op result and the report header."""
        return {
            "sigma_0": float(self.sigma_0),
            "wilson_model": str(self.model),
            "b_cart": [float(x) for x in self.b_cart],
            "b_eigenvalues": [float(x) for x in self.eigenvalues],
            "b_direction_cosines": [[float(x) for x in row] for row in self.direction_cosines],
            "b_iso": float(self.b_iso),
            "delta_b_aniso": float(self.delta_b_aniso),
            "n_wilson_params": int(self.n_params),
        }


def orthogonalization(unit_cell: Sequence[float]) -> np.ndarray:
    """Cartesian-from-fractional matrix ``O`` (``x_cart = O x_frac``)."""
    from phridge.sfcalc.engine.cell import orthogonalization_matrix

    return orthogonalization_matrix(unit_cell)


def reciprocal_cartesian(unit_cell: Sequence[float], hkl: Any) -> np.ndarray:
    """``s_cart = O^-T h`` for integer Miller triples, shape (N, 3), ``|s_cart| = 1/d``.

    Metric-correct by construction: ``sum(s_cart**2)`` reproduces the ``s_sq`` the
    isotropic path uses to machine precision, for triclinic cells too. Raw integer
    indices would not.
    """
    from phridge.sfcalc.engine.cell import reciprocal_cartesian as _rc

    return _rc(unit_cell, np.asarray(hkl, dtype=np.float64))


def reciprocal_metric(unit_cell: Sequence[float]) -> np.ndarray:
    """``G`` with ``h^T G h = 1/d^2``; equals ``O^-1 O^-T``."""
    o_inv = np.linalg.inv(orthogonalization(unit_cell))
    return o_inv @ o_inv.T


def cholesky_params_from_b(b_cart: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a PSD ``B`` into (log-diagonal, off-diagonal) Cholesky parameters.

    Inverse of :func:`b_from_cholesky_params`. A jitter is added if ``B`` is singular
    (which happens when the moment fit returns ``B_W = 0``), because the whole point of
    the Cholesky form is that the optimizer never has to see a boundary.
    """
    b = np.asarray(b_cart, dtype=np.float64)
    b = 0.5 * (b + b.T)
    for jitter in (0.0, 1e-6, 1e-3, 1.0):
        try:
            low = np.linalg.cholesky(b + jitter * np.eye(3))
        except np.linalg.LinAlgError:
            continue
        diag = np.clip(np.diag(low), 1e-8, None)
        return np.log(diag), np.array([low[1, 0], low[2, 0], low[2, 1]], dtype=np.float64)
    raise ValueError("B tensor is not positive semi-definite even with jitter")


def b_from_cholesky_params(log_diag: Any, off_diag: Any) -> np.ndarray:
    """Rebuild ``B = L L^T`` from the unconstrained parameters (numpy)."""
    ld = np.asarray(log_diag, dtype=np.float64).ravel()
    od = np.asarray(off_diag, dtype=np.float64).ravel()
    low = np.zeros((3, 3), dtype=np.float64)
    low[0, 0], low[1, 1], low[2, 2] = np.exp(ld[0]), np.exp(ld[1]), np.exp(ld[2])
    low[1, 0], low[2, 0], low[2, 1] = od[0], od[1], od[2]
    return low @ low.T


def isotropic_cholesky_params(b_wilson: float) -> tuple[np.ndarray, np.ndarray]:
    """Parameters for ``B = B_W * I``, the exact isotropic starting point.

    ``B_W = 0`` is a legitimate moment-fit result but a Cholesky boundary, so it is
    floored -- the fit is free to walk back down.
    """
    b = max(float(b_wilson), 1e-3)
    return np.full(3, 0.5 * math.log(b)), np.zeros(3, dtype=np.float64)


def tensor_identifiability(s_cart: Any) -> float:
    """How well these reflection directions determine 6 tensor components, in [0, 1].

    The quadratic form ``s^T B s`` is linear in ``B``'s six components with design row
    ``[sx^2, sy^2, sz^2, 2 sx sy, 2 sx sz, 2 sy sz]`` on unit directions. The returned
    number is the smallest eigenvalue of that design's normalized Gram matrix divided by
    the largest -- zero when the directions lie in a plane or along a line, where six
    components are simply not determined and a fit would return confident nonsense in the
    unconstrained directions.

    Direction only, deliberately: whether the *magnitudes* span a useful resolution range
    is a separate question that the isotropic fit already faces.
    """
    s = np.asarray(s_cart, dtype=np.float64)
    if s.ndim != 2 or s.shape[1] != 3 or s.shape[0] < 6:
        return 0.0
    norm = np.linalg.norm(s, axis=1)
    keep = norm > 0
    if int(np.count_nonzero(keep)) < 6:
        return 0.0
    u = s[keep] / norm[keep, None]
    design = np.column_stack(
        [
            u[:, 0] ** 2, u[:, 1] ** 2, u[:, 2] ** 2,
            2.0 * u[:, 0] * u[:, 1], 2.0 * u[:, 0] * u[:, 2], 2.0 * u[:, 1] * u[:, 2],
        ]
    )
    with np.errstate(all="ignore"):
        gram = design.T @ design / design.shape[0]
        eig = np.linalg.eigvalsh(gram)
    if not np.all(np.isfinite(eig)) or eig[-1] <= 0.0:
        return 0.0
    return float(max(eig[0], 0.0) / eig[-1])


def sigma_w_from_b(
    *,
    s_cart: np.ndarray,
    sigma_0: float,
    b_cart: np.ndarray,
    floor: float = 1e-12,
) -> np.ndarray:
    """``Sigma_W(h) = Sigma_0 exp(-0.5 s^T B s)`` (numpy; the fit uses torch)."""
    s = np.asarray(s_cart, dtype=np.float64)
    quad = np.einsum("ni,ij,nj->n", s, np.asarray(b_cart, dtype=np.float64), s)
    return np.maximum(float(sigma_0) * np.exp(-0.5 * quad), floor)


def describe_wilson_tensor(
    *,
    sigma_0: float,
    b_cart: np.ndarray,
    unit_cell: Sequence[float],
    model: WilsonModel = "anisotropic",
    nll_per_refl: float = float("nan"),
    n_params: Optional[int] = None,
) -> WilsonTensor:
    """Build the reportable eigen-form of a fitted ``B``.

    Direction cosines are taken against the reciprocal axes ``a*``, ``b*``, ``c*``
    because that is the frame a crystallographer can act on -- "the weak direction is
    along c*" is actionable, three Cartesian components are not. Absolute values: an
    eigenvector's sign is arbitrary and would otherwise flip between runs.
    """
    b = np.asarray(b_cart, dtype=np.float64)
    b = 0.5 * (b + b.T)
    eigvals, eigvecs = np.linalg.eigh(b)  # ascending, orthonormal columns

    o_inv = np.linalg.inv(orthogonalization(unit_cell))
    # Rows of O^-1 are a*, b*, c* in Cartesian coordinates (s_cart = O^-T h).
    axes = o_inv / np.maximum(np.linalg.norm(o_inv, axis=1, keepdims=True), 1e-30)
    cosines = tuple(
        tuple(float(abs(np.dot(axes[j], eigvecs[:, i]))) for j in range(3)) for i in range(3)
    )

    return WilsonTensor(
        sigma_0=float(sigma_0),
        b_cart=(
            float(b[0, 0]), float(b[1, 1]), float(b[2, 2]),
            float(b[0, 1]), float(b[0, 2]), float(b[1, 2]),
        ),
        eigenvalues=tuple(float(x) for x in eigvals),
        direction_cosines=cosines,
        b_iso=float(np.trace(b) / 3.0),
        delta_b_aniso=float(eigvals[-1] - eigvals[0]),
        model=model,
        nll_per_refl=float(nll_per_refl),
        n_params=int(n_params if n_params is not None else (2 if model == "isotropic" else 7)),
    )


def describe_isotropic_wilson(
    *,
    sigma_0: float,
    b_wilson: float,
    unit_cell: Sequence[float],
    nll_per_refl: float = float("nan"),
) -> WilsonTensor:
    """The same reportable form for an isotropic fit, so the header never branches."""
    return describe_wilson_tensor(
        sigma_0=sigma_0,
        b_cart=float(b_wilson) * np.eye(3),
        unit_cell=unit_cell,
        model="isotropic",
        nll_per_refl=nll_per_refl,
        n_params=2,
    )


@dataclass
class DataFracSpread:
    """Spread of ``data%`` inside one resolution shell.

    Under an isotropic normalization this spread contains the anisotropy: reflections
    at the same resolution but in different directions get different ``sigma_Z``, so
    ``data%`` fans out. Under a correct anisotropic normalization it should collapse to
    whatever genuine per-reflection ``sigma_I`` variation exists. That makes ``p90-p10``
    the direct diagnostic for whether the tensor is doing its job.
    """

    p10: float = float("nan")
    p50: float = float("nan")
    p90: float = float("nan")

    @property
    def spread(self) -> float:
        return self.p90 - self.p10


def data_frac_spread(data_frac: Any) -> DataFracSpread:
    """10th / 50th / 90th percentile of a ``data%`` slice, NaN-safe."""
    arr = np.asarray(data_frac, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return DataFracSpread()
    p10, p50, p90 = (float(x) for x in np.percentile(arr, [10.0, 50.0, 90.0]))
    return DataFracSpread(p10=p10, p50=p50, p90=p90)
