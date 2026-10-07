"""Resolution-binned intensity diagnostics printed during mli_quad refinement.

Per bin (default 500 reflections, low→high resolution):
  - mean I/σ, fraction negative, fraction with I/σ < 1..5
  - mean σ_A(s) and Wilson Σ_W(s) from the current nuisance fit
  - mean data-fraction vs Wilson prior:  σ_Z^{-2} / (1 + σ_Z^{-2})
    with σ_Z = σ_I / (ε Σ_W)  (model-free; ≈0 = prior-dominated)
  - S_post / S_prior when ``ml_i_maps`` bins are available (expected residual
    under the posterior / under the prior — never write a bare ``S``)

The header also carries the agreement statistics, each tagged with the amplitudes
behind it: the French-Wilson R (the conventional one, model-free ``F_obs``), the
direct intensity R, and the posterior point estimates — which are printed under
their own heading and never labelled R, because shrinkage biases them low and they
improve as the data get worse.

It also carries the overall anisotropic B of the global scale — the PDB REMARK 3
quantity — recovered from the applied ``k_anisotropic`` array by
:func:`fit_aniso_scale`, since ``b_cart`` on the fmodel is ``None`` by construction.

A second table splits ``CC_I = corr(I_obs, I_calc)`` by **fixed** I/sigma bin
(``<0``, ``0-1``, ``1-2``, ``2-3``, ``3-5``, ``5-9``, ``>9``) and resolution shell, with
work/free CC and the population share in each cell, and mean sigma_A per shell. Reading
down a column holds I/sigma fixed and varies resolution, which is the sigma_A dependence;
reading across a row holds resolution and sigma_A fixed and varies I/sigma.

Enable/disable with ``PHRIDGE_STATS_REPORT`` (default on). Bin size:
``PHRIDGE_STATS_BIN_SIZE`` (default 500). CC table geometry:
``PHRIDGE_CC_SHELLS`` (default 8) and ``PHRIDGE_CC_ISIG_EDGES`` (default ``0,1,2,3,5,9``).
"""

from __future__ import annotations

import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, TextIO

import numpy as np

from phridge.q.error_models.wilson_aniso import data_frac_spread


def stats_report_enabled(default: bool = True) -> bool:
    raw = os.environ.get("PHRIDGE_STATS_REPORT")
    if raw is None or not str(raw).strip():
        return bool(default)
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def stats_bin_size(default: int = 500) -> int:
    raw = os.environ.get("PHRIDGE_STATS_BIN_SIZE")
    if raw is None or not str(raw).strip():
        return int(default)
    try:
        n = int(str(raw).strip())
        return n if n >= 1 else int(default)
    except ValueError:
        return int(default)


@dataclass
class ResolutionBinStats:
    bin: int
    n: int
    d_max: float
    d_min: float
    mean_isig: float
    frac_neg: float
    frac_lt_1: float
    frac_lt_2: float
    frac_lt_3: float
    frac_lt_4: float
    frac_lt_5: float
    mean_sigma_a: float = float("nan")
    mean_sigma_wilson: float = float("nan")
    mean_data_frac: float = float("nan")  # σ_Z^{-2}/(1+σ_Z^{-2})
    # Spread of data% inside the shell. Resolution is fixed here, so under an isotropic
    # Σ_W this spread carries the anisotropy; under a correct tensor it should collapse
    # to whatever genuine σ_I variation exists.
    data_frac_p10: float = float("nan")
    data_frac_p90: float = float("nan")
    s_post: float = float("nan")  # S_post (work; parent-set k_S)
    s_prior: float = float("nan")  # S_prior — σ_A-implied no-data floor
    s_post_free: float = float("nan")
    s_prior_free: float = float("nan")
    rho2: float = float("nan")  # visible share v_vis/(v_vis+v_lat), work set
    rho2_free: float = float("nan")
    n_work: int = 0
    n_free: int = 0


@dataclass
class AnisoScale:
    """Overall anisotropic B of the global scale, as PDB REMARK 3 reports it.

    Recovered from the applied ``k_anisotropic`` array rather than read off the scaling
    object, because ``k_anisotropic`` is what actually multiplies F_calc inside
    ``f_model()`` -- a B fitted from it describes the anisotropy the target really sees,
    whatever the scaling internals happen to store.

    ``log_rms_residual`` is the honesty check. The fit assumes the Debye-Waller form
    ``k(h) = exp(c - 2 pi^2 h^T U* h)``; if a scaling method applied something else
    (a per-bin curve, say), the reported B is only that array's best anisotropic
    projection and the residual is how much of it was left behind.

    ``b_iso_equiv`` is reported but is **not** an independent quantity: the isotropic
    part of the scale is degenerate with ``k_isotropic``, which carries its own
    resolution dependence. ``anisotropy`` -- the spread of the principal values -- is
    the part that only the anisotropic scale can supply, and the number to watch.
    """

    b_cart: tuple[float, ...] = (float("nan"),) * 6  # B11,B22,B33,B12,B13,B23 (Å²)
    principal_b: tuple[float, ...] = (float("nan"),) * 3  # ascending eigenvalues (Å²)
    b_iso_equiv: float = float("nan")  # trace(B_cart) / 3
    anisotropy: float = float("nan")  # max - min principal value
    log_scale_offset: float = float("nan")  # the fitted h-independent constant c
    log_rms_residual: float = float("nan")
    n_refl: int = 0
    source: str = "k_anisotropic"

    @property
    def is_valid(self) -> bool:
        return bool(self.n_refl) and all(np.isfinite(b) for b in self.b_cart)


def fit_aniso_scale(
    *,
    miller_indices: Any,
    k_anisotropic: Any,
    unit_cell: Sequence[float],
    min_n: int = 50,
) -> AnisoScale:
    """Recover the overall anisotropic B from an applied ``k_anisotropic`` array.

    ``log k(h) = c - 2 pi^2 (h^T U* h)`` is linear in the six components of ``U*`` and
    in ``c``, so this is a 7-parameter linear least squares -- no optimizer, no starting
    point, no chance of landing in a local minimum. ``c`` has to be fitted even though it
    is not reported: without it the isotropic part of ``U*`` would absorb any overall
    normalization of the array and the whole tensor would shift.

    Returns an empty :class:`AnisoScale` rather than raising when the array is unusable
    (too few reflections, non-positive values, a rank-deficient set of indices). This is
    reporting-only and must never cost a macro cycle.
    """
    from phridge.transforms.fourier.ten_eyck.cell import orthogonalization_matrix

    out = AnisoScale()
    hkl = np.asarray(miller_indices, dtype=np.float64)
    k = _as_numpy(k_anisotropic)
    if hkl.ndim != 2 or hkl.shape[1] != 3 or hkl.shape[0] != k.size:
        raise ValueError(f"miller_indices {hkl.shape} incompatible with k {k.shape}")

    good = np.isfinite(k) & (k > 0) & np.all(np.isfinite(hkl), axis=1)
    hkl, k = hkl[good], k[good]
    if k.size < max(7, int(min_n)):
        return out

    h, kk, ll = hkl[:, 0], hkl[:, 1], hkl[:, 2]
    tp2 = 2.0 * np.pi**2
    design = np.column_stack(
        [
            -tp2 * h * h,
            -tp2 * kk * kk,
            -tp2 * ll * ll,
            -2.0 * tp2 * h * kk,
            -2.0 * tp2 * h * ll,
            -2.0 * tp2 * kk * ll,
            np.ones_like(h),
        ]
    )
    log_k = np.log(k)
    # LAPACK's gelsd leaves the FPU error flags set even on a clean solve, and numpy
    # then attributes them to whichever operation checks them next -- here the residual
    # matmul, which reports a spurious divide-by-zero on exact input. Ignore the flags
    # and validate the values instead, which is the only check that means anything.
    with np.errstate(all="ignore"):
        try:
            params, *_ = np.linalg.lstsq(design, log_k, rcond=None)
        except np.linalg.LinAlgError:
            return out
        if not np.all(np.isfinite(params)):
            return out
        resid = log_k - design @ params

    u11, u22, u33, u12, u13, u23, c = (float(x) for x in params)
    u_star = np.array(
        [[u11, u12, u13], [u12, u22, u23], [u13, u23, u33]], dtype=np.float64
    )
    # u_cart = O U* O^T, then B = 8 pi^2 U (cctbx adptbx conventions; see sfcalc.cell).
    o = orthogonalization_matrix(unit_cell)
    b = 8.0 * np.pi**2 * (o @ u_star @ o.T)
    if not np.all(np.isfinite(b)):
        return out
    principal = np.sort(np.linalg.eigvalsh(b))

    out.b_cart = (
        float(b[0, 0]), float(b[1, 1]), float(b[2, 2]),
        float(b[0, 1]), float(b[0, 2]), float(b[1, 2]),
    )
    out.principal_b = tuple(float(x) for x in principal)
    out.b_iso_equiv = float(np.trace(b) / 3.0)
    out.anisotropy = float(principal[-1] - principal[0])
    out.log_scale_offset = c
    out.log_rms_residual = float(np.sqrt(np.mean(resid**2)))
    out.n_refl = int(k.size)
    return out


def b_field(val: float, decimals: int = 5) -> str:
    """Format one B component, rounding a signed zero to an unsigned one.

    A symmetry-fixed component comes back as -1e-17 rather than exactly zero, which
    prints as ``-0.00000``. Harmless to a human, but this string goes into a deposited
    PDB field; rounding before the sign is decided keeps it ``0.00000``.
    """
    v = round(float(val), decimals)
    return f"{v + 0.0:.{decimals}f}"  # -0.0 + 0.0 is +0.0 in IEEE 754


def format_aniso_scale_remark_3(aniso: AnisoScale, prefix: str = "REMARK   3  ") -> list[str]:
    """The mandated PDB ``OVERALL ANISOTROPIC B VALUE`` block, exact field spelling."""
    if not aniso.is_valid:
        return []
    lines = [prefix + "OVERALL ANISOTROPIC B VALUE."]
    for name, val in zip(("B11", "B22", "B33", "B12", "B13", "B23"), aniso.b_cart):
        lines.append(prefix + f" {name} (A**2) : {b_field(val)}")
    return lines


def format_aniso_scale_summary(aniso: AnisoScale) -> str:
    """One-line summary for the scaling log: the tensor, then what to read from it."""
    if not aniso.is_valid:
        return "overall anisotropic B: unavailable"
    b = [b_field(x, 3) for x in aniso.b_cart]
    return (
        f"overall anisotropic B (Cartesian, Å²): "
        f"B11={b[0]} B22={b[1]} B33={b[2]} "
        f"B12={b[3]} B13={b[4]} B23={b[5]} | "
        f"principal {aniso.principal_b[0]:.2f}/{aniso.principal_b[1]:.2f}/"
        f"{aniso.principal_b[2]:.2f} | anisotropy={aniso.anisotropy:.3f} | "
        f"B_iso equiv={b_field(aniso.b_iso_equiv, 3)} "
        f"(fitted from {aniso.source} on {aniso.n_refl} refl, "
        f"log-RMS resid={aniso.log_rms_residual:.4f})"
    )


# Fixed I/sigma bin boundaries, not quantiles: the bins must mean the same thing in
# every run and every shell, so a table can be compared across data sets. These are the
# interior edges; the outer bins are I/sigma < 0 (genuine negatives, which mli_quad keeps)
# and I/sigma > 9.
DEFAULT_ISIG_EDGES: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 5.0, 9.0)


@dataclass
class CCIsigCell:
    """``CC_I`` and population share on one (resolution shell, I/sigma bin) cell."""

    cc_work: float = float("nan")
    cc_free: float = float("nan")
    n_work: int = 0
    n_free: int = 0
    frac: float = float("nan")  # share of the parent row's reflections in this bin

    @property
    def n(self) -> int:
        return self.n_work + self.n_free


@dataclass
class CCIsigRow:
    """One resolution shell of the ``CC_I`` table."""

    shell: int
    d_max: float
    d_min: float
    n_work: int
    n_free: int
    mean_sigma_a: float = float("nan")
    cells: list[CCIsigCell] = field(default_factory=list)
    marginal: CCIsigCell = field(default_factory=CCIsigCell)

    @property
    def n(self) -> int:
        return self.n_work + self.n_free


@dataclass
class CCIsigTable:
    """``CC_I`` split by fixed I/sigma bin and resolution shell.

    The I/sigma bins are **fixed** (:data:`DEFAULT_ISIG_EDGES`), not quantiles, so a
    column means the same thing in every row, in every run, and across data sets.
    Quantile columns would fill the table more evenly but could not be compared between
    refinements, and the population imbalance the fixed bins expose -- how much of a
    shell sits below I/sigma of 1, say -- is itself the diagnostic.

    ``mean_sigma_a`` on each row is what makes the resolution axis readable as a sigma_A
    axis: at fixed I/sigma, moving down a column varies sigma_A.
    """

    edges: list[float] = field(default_factory=list)  # interior edges
    labels: list[str] = field(default_factory=list)
    rows: list[CCIsigRow] = field(default_factory=list)
    column_marginal: list[CCIsigCell] = field(default_factory=list)
    overall: CCIsigCell = field(default_factory=CCIsigCell)
    min_n: int = 30

    @property
    def n_bins(self) -> int:
        return len(self.labels)


@dataclass
class IntensityStatsReport:
    """Full report payload from one scale / nuisance update."""

    n_refl: int
    bin_size: int
    bins: list[ResolutionBinStats] = field(default_factory=list)
    mean_sigma_a: float = float("nan")
    mean_data_frac: float = float("nan")
    frac_neg_all: float = float("nan")
    sigma_a_params: dict[str, float] = field(default_factory=dict)
    sigma_wilson_params: dict[str, float] = field(default_factory=dict)
    nu: Optional[float] = None
    nu_params: dict[str, Any] = field(default_factory=dict)
    label: str = ""
    # Overall S family (filled by merge from ml_i_maps)
    s_post_work: float = float("nan")
    s_post_free: float = float("nan")
    s_prior_work: float = float("nan")
    s_prior_free: float = float("nan")
    k_s_work: float = float("nan")
    k_s_prior_work: float = float("nan")
    rho2_work: float = float("nan")
    rho2_free: float = float("nan")
    s_report_version: Optional[int] = None

    # -- Agreement statistics, each tagged with the amplitudes it came from --------
    # r_fw_* is the conventional R (model-free French-Wilson F_obs) and the only one
    # that may carry the letter R in printed output. The posterior point estimates are
    # kept because they are useful for watching shrinkage, but they are printed under
    # their own heading and never labelled R -- they are biased low and improve as the
    # data get worse. r_intensity_* is a genuine point-estimate R on intensities.
    r_fw_work: float = float("nan")
    r_fw_free: float = float("nan")
    r_post_work: float = float("nan")
    r_post_free: float = float("nan")
    r_mode_work: float = float("nan")
    r_mode_free: float = float("nan")
    r_intensity_work: float = float("nan")
    r_intensity_free: float = float("nan")
    cc_isig: Optional[CCIsigTable] = None
    aniso_scale: Optional[AnisoScale] = None

    # -- Deprecated R-family aliases -------------------------------------------
    # Read-only bridges for callers written against the old names. They exist on
    # this object only: the printed report and the worker payload use S names
    # exclusively. Remove in the next minor version.
    @property
    def r_int_work(self) -> float:
        _warn_renamed("r_int_work", "s_post_work")
        return self.s_post_work

    @property
    def r_int_free(self) -> float:
        _warn_renamed("r_int_free", "s_post_free")
        return self.s_post_free

    @property
    def r_inf_work(self) -> float:
        _warn_renamed("r_inf_work", "s_prior_work")
        return self.s_prior_work

    @property
    def r_inf_free(self) -> float:
        _warn_renamed("r_inf_free", "s_prior_free")
        return self.s_prior_free

    @property
    def k_int_work(self) -> float:
        _warn_renamed("k_int_work", "k_s_work")
        return self.k_s_work

    @property
    def k_inf_work(self) -> float:
        _warn_renamed("k_inf_work", "k_s_prior_work")
        return self.k_s_prior_work

    @property
    def r_int_version(self) -> Optional[int]:
        _warn_renamed("r_int_version", "s_report_version")
        return self.s_report_version


def _warn_renamed(old: str, new: str) -> None:
    warnings.warn(
        f"{old!r} is deprecated: the integrated R statistics were renamed to the "
        f"S family (S_post / S_prior) because they are not the crystallographic "
        f"R factor. Use {new!r} instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def _fmt_r(x: float) -> str:
    return f"{x:.4f}" if np.isfinite(x) else "n/a"


def _as_numpy(x: Any, dtype: Any = np.float64) -> np.ndarray:
    if x is None:
        raise ValueError("expected array, got None")
    if hasattr(x, "as_numpy_array"):
        return np.asarray(x.as_numpy_array(), dtype=dtype)
    return np.asarray(x, dtype=dtype)


def compute_intensity_stats_report(
    *,
    intensities: Any,
    sigmas: Any,
    d_spacings: Any,
    epsilon: Optional[Any] = None,
    sigma_a: Optional[Any] = None,
    sigma_wilson: Optional[Any] = None,
    bin_size: Optional[int] = None,
    sigma_a_params: Optional[dict[str, float]] = None,
    sigma_wilson_params: Optional[dict[str, float]] = None,
    nu: Optional[float] = None,
    nu_params: Optional[dict[str, Any]] = None,
    label: str = "",
) -> IntensityStatsReport:
    """Build resolution-ordered bins of intensity / σ_A diagnostics."""
    io = _as_numpy(intensities)
    sig = _as_numpy(sigmas)
    d = _as_numpy(d_spacings)
    n = int(io.size)
    if sig.size != n or d.size != n:
        raise ValueError("intensities, sigmas, and d_spacings must have the same length")

    if epsilon is None:
        eps = np.ones(n, dtype=np.float64)
    else:
        eps = _as_numpy(epsilon)
        if eps.size != n:
            raise ValueError("epsilon length mismatch")

    sa = None if sigma_a is None else _as_numpy(sigma_a)
    sw = None if sigma_wilson is None else _as_numpy(sigma_wilson)
    if sa is not None and sa.size != n:
        raise ValueError("sigma_a length mismatch")
    if sw is not None and sw.size != n:
        raise ValueError("sigma_wilson length mismatch")

    valid = np.isfinite(io) & np.isfinite(sig) & (sig > 0) & np.isfinite(d) & (eps > 0)
    if sw is not None:
        valid &= np.isfinite(sw) & (sw > 0)
    if sa is not None:
        valid &= np.isfinite(sa)

    io = io[valid]
    sig = sig[valid]
    d = d[valid]
    eps = eps[valid]
    if sa is not None:
        sa = sa[valid]
    if sw is not None:
        sw = sw[valid]

    n_use = int(io.size)
    bs = int(bin_size) if bin_size is not None else stats_bin_size()
    if n_use == 0:
        return IntensityStatsReport(
            n_refl=0,
            bin_size=bs,
            sigma_a_params=dict(sigma_a_params or {}),
            sigma_wilson_params=dict(sigma_wilson_params or {}),
            nu=nu,
            nu_params=dict(nu_params or {}),
            label=label,
        )

    order = np.argsort(-d)  # low → high resolution
    io, sig, d, eps = io[order], sig[order], d[order], eps[order]
    if sa is not None:
        sa = sa[order]
    if sw is not None:
        sw = sw[order]

    snr = io / sig
    # Normalized measurement noise vs Wilson scale: σ_Z = σ_I / (ε Σ_W)
    if sw is not None:
        sigma_z = sig / np.maximum(eps * sw, 1e-300)
        inv_var = 1.0 / np.maximum(sigma_z * sigma_z, 1e-300)
        data_frac = inv_var / (1.0 + inv_var)
    else:
        data_frac = np.full(n_use, np.nan)

    rows: list[ResolutionBinStats] = []
    n_bins = (n_use + bs - 1) // bs
    for b in range(n_bins):
        lo = b * bs
        hi = min(lo + bs, n_use)
        snr_b = snr[lo:hi]
        io_b = io[lo:hi]
        d_b = d[lo:hi]
        df_b = data_frac[lo:hi]
        _spread_b = data_frac_spread(df_b)
        rows.append(
            ResolutionBinStats(
                bin=b + 1,
                n=hi - lo,
                d_max=float(d_b.max()),
                d_min=float(d_b.min()),
                mean_isig=float(np.mean(snr_b)),
                frac_neg=float(np.mean(io_b < 0.0)),
                frac_lt_1=float(np.mean(snr_b < 1.0)),
                frac_lt_2=float(np.mean(snr_b < 2.0)),
                frac_lt_3=float(np.mean(snr_b < 3.0)),
                frac_lt_4=float(np.mean(snr_b < 4.0)),
                frac_lt_5=float(np.mean(snr_b < 5.0)),
                mean_sigma_a=float(np.mean(sa[lo:hi])) if sa is not None else float("nan"),
                mean_sigma_wilson=float(np.mean(sw[lo:hi])) if sw is not None else float("nan"),
                mean_data_frac=float(np.nanmean(df_b)) if np.any(np.isfinite(df_b)) else float("nan"),
                data_frac_p10=_spread_b.p10,
                data_frac_p90=_spread_b.p90,
            )
        )

    return IntensityStatsReport(
        n_refl=n_use,
        bin_size=bs,
        bins=rows,
        mean_sigma_a=float(np.mean(sa)) if sa is not None else float("nan"),
        mean_data_frac=float(np.nanmean(data_frac)) if np.any(np.isfinite(data_frac)) else float("nan"),
        frac_neg_all=float(np.mean(io < 0.0)),
        sigma_a_params=dict(sigma_a_params or {}),
        sigma_wilson_params=dict(sigma_wilson_params or {}),
        nu=nu,
        nu_params=dict(nu_params or {}),
        label=label,
    )


def cc_shell_count(default: int = 8) -> int:
    return _env_int("PHRIDGE_CC_SHELLS", default, lo=1, hi=40)


def cc_isig_edges(default: Sequence[float] = DEFAULT_ISIG_EDGES) -> tuple[float, ...]:
    """Interior I/sigma bin edges, overridable with ``PHRIDGE_CC_ISIG_EDGES``.

    Comma-separated and strictly increasing, e.g. ``0,1,2,3,5,9``. Anything malformed
    falls back to the default rather than producing a table with meaningless columns.
    """
    raw = os.environ.get("PHRIDGE_CC_ISIG_EDGES")
    if raw is None or not str(raw).strip():
        return tuple(float(e) for e in default)
    try:
        vals = [float(part) for part in str(raw).replace(" ", "").split(",") if part]
    except ValueError:
        return tuple(float(e) for e in default)
    if len(vals) < 1 or any(b <= a for a, b in zip(vals, vals[1:])):
        return tuple(float(e) for e in default)
    return tuple(vals)


def isig_bin_labels(edges: Sequence[float]) -> list[str]:
    """Human labels for the fixed bins, e.g. ``<0``, ``0-1``, ..., ``>9``."""
    e = [float(x) for x in edges]
    labels = [f"<{e[0]:g}"]
    labels += [f"{lo:g}-{hi:g}" for lo, hi in zip(e, e[1:])]
    labels.append(f">{e[-1]:g}")
    return labels


def _env_int(name: str, default: int, *, lo: int, hi: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return int(default)
    try:
        n = int(str(raw).strip())
    except ValueError:
        return int(default)
    return n if lo <= n <= hi else int(default)


def _cc(x: np.ndarray, y: np.ndarray, min_n: int) -> float:
    """Pearson CC, or NaN when the sample is too small or degenerate.

    Same functional as ``cc_intensity_*`` in the worker so the marginals of this table
    are directly comparable with the banner value.
    """
    if x.size < max(2, min_n) or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    c = np.corrcoef(x, y)[0, 1]
    return float(c) if np.isfinite(c) else float("nan")


def compute_cc_isig_table(
    *,
    intensities: Any,
    sigmas: Any,
    d_spacings: Any,
    i_calc: Any,
    r_free: Optional[Any] = None,
    sigma_a: Optional[Any] = None,
    edges: Optional[Sequence[float]] = None,
    n_shells: Optional[int] = None,
    min_n: int = 30,
) -> CCIsigTable:
    """``CC_I`` = corr(I_obs, I_calc) on a resolution x fixed-I/sigma grid.

    Resolution shells are an equal-count split; the I/sigma bins are fixed boundaries
    (see :class:`CCIsigTable`). Each cell also carries ``frac``, its share of the
    shell's reflections, because with fixed bins the populations are deliberately
    uneven and a CC means little without knowing how many reflections it came from.
    Cells thinner than ``min_n`` get a NaN ``CC`` but keep their counts and fraction.
    """
    io = _as_numpy(intensities)
    sig = _as_numpy(sigmas)
    d = _as_numpy(d_spacings)
    ic = _as_numpy(i_calc)
    n = int(io.size)
    for name, arr in (("sigmas", sig), ("d_spacings", d), ("i_calc", ic)):
        if arr.size != n:
            raise ValueError(f"{name} length {arr.size} != intensities length {n}")
    if r_free is None:
        free = np.zeros(n, dtype=bool)
    else:
        free = _as_numpy(r_free, dtype=bool)
        if free.size != n:
            raise ValueError("r_free length mismatch")
    sa = None if sigma_a is None else _as_numpy(sigma_a)
    if sa is not None and sa.size != n:
        raise ValueError("sigma_a length mismatch")

    valid = (
        np.isfinite(io) & np.isfinite(sig) & (sig > 0) & np.isfinite(d) & np.isfinite(ic)
    )
    io, sig, d, ic, free = io[valid], sig[valid], d[valid], ic[valid], free[valid]
    if sa is not None:
        sa = sa[valid]

    e = tuple(float(x) for x in (edges if edges is not None else cc_isig_edges()))
    ns = int(n_shells) if n_shells is not None else cc_shell_count()
    table = CCIsigTable(
        edges=list(e), labels=isig_bin_labels(e), min_n=int(min_n)
    )
    nb = table.n_bins
    if io.size == 0:
        return table

    snr = io / sig
    bin_index = np.digitize(snr, np.asarray(e, dtype=np.float64), right=False)

    order = np.argsort(-d)  # low -> high resolution
    shell_index = np.empty(io.size, dtype=np.int64)
    bounds = np.linspace(0, io.size, ns + 1).astype(int)
    for s in range(ns):
        shell_index[order[bounds[s] : bounds[s + 1]]] = s

    work = ~free

    def make_cell(sel: np.ndarray, denom: int) -> CCIsigCell:
        w, f = sel & work, sel & free
        n_sel = int(np.count_nonzero(sel))
        return CCIsigCell(
            cc_work=_cc(io[w], ic[w], min_n),
            cc_free=_cc(io[f], ic[f], min_n),
            n_work=int(np.count_nonzero(w)),
            n_free=int(np.count_nonzero(f)),
            frac=(n_sel / denom) if denom else float("nan"),
        )

    for s in range(ns):
        in_shell = shell_index == s
        n_shell = int(np.count_nonzero(in_shell))
        if n_shell == 0:
            continue
        row = CCIsigRow(
            shell=s + 1,
            d_max=float(d[in_shell].max()),
            d_min=float(d[in_shell].min()),
            n_work=int(np.count_nonzero(in_shell & work)),
            n_free=int(np.count_nonzero(in_shell & free)),
            mean_sigma_a=float(np.mean(sa[in_shell])) if sa is not None else float("nan"),
        )
        # Fractions are row-normalized: "what share of THIS shell sits in this I/sigma
        # bin", which is the question the resolution axis makes interesting.
        row.cells = [make_cell(in_shell & (bin_index == b), n_shell) for b in range(nb)]
        row.marginal = make_cell(in_shell, n_shell)
        table.rows.append(row)

    total = int(io.size)
    table.column_marginal = [make_cell(bin_index == b, total) for b in range(nb)]
    table.overall = make_cell(np.ones(total, dtype=bool), total)
    return table


