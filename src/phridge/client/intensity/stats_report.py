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

from phridge.contrib.intensity_ll.wilson import data_frac_spread


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
    from phridge.sfcalc.engine.cell import orthogonalization_matrix

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


def format_wilson_normalization(sw_params: Optional[dict[str, Any]]) -> list[str]:
    """Render the fitted Wilson normalization in inspectable form.

    Eigenvalues and directions, not raw ``B`` components: the components depend on the
    basis and cannot be compared between data sets, the eigen-form can. ``ΔB_aniso`` is
    the number that says whether the anisotropic fit found anything.
    """
    if not sw_params:
        return []
    p = dict(sw_params)
    s0 = p.get("sigma_0")
    if s0 is None:
        return []
    model = str(p.get("wilson_model", "isotropic"))
    eig = p.get("b_eigenvalues") or []
    d_b = p.get("delta_b_aniso")
    b_iso = p.get("b_iso", p.get("b_wilson"))
    out = []
    if model == "binned":
        # The bins carry the isotropic falloff, so there is no Σ₀ or B_iso to print.
        head = f"  Σ_W [binned]: {int(p.get('n_bins', 0))} bins (mean I/ε fixed per bin)"
    else:
        head = f"  Σ_W [{model}]: Σ₀={float(s0):.4g}"
        if b_iso is not None and np.isfinite(float(b_iso)):
            head += f"  B_iso={float(b_iso):.3f}"
    if d_b is not None and np.isfinite(float(d_b)):
        head += f"  ΔB_aniso={float(d_b):.3f} Å²"
    if p.get("wilson_nll") is not None and np.isfinite(float(p["wilson_nll"])):
        head += f"  NLL/refl={float(p['wilson_nll']):.4f}"
    head += f"  ({p.get('method', 'moment_plot')})"
    out.append(head)

    if len(eig) == 3 and all(np.isfinite(float(e)) for e in eig):
        cos = p.get("b_direction_cosines") or []
        parts = []
        for i, e in enumerate(eig):
            axis = ""
            if i < len(cos) and len(cos[i]) == 3:
                names = ("a*", "b*", "c*")
                j = int(np.argmax([abs(float(c)) for c in cos[i]]))
                axis = f"≈{names[j]}({abs(float(cos[i][j])):.2f})"
            parts.append(f"{float(e):.2f}{axis}")
        label = "traceless B eigenvalues" if model == "binned" else "eigenvalues"
        out.append(f"    {label} (Å²) = " + ", ".join(parts))
    if model in ("anisotropic", "binned"):
        # The constraint is part of the result: an anisotropic Σ_W is only interpretable
        # because the model-side k_aniso was held isotropic. Say so where the number is.
        out.append(
            "    all data anisotropy is carried here; the overall k_anisotropic is "
            "constrained isotropic (the two are degenerate)"
        )
    if p.get("wilson_fallback"):
        out.append(f"    NOTE: fell back to isotropic — {p['wilson_fallback']}")
    spread = p.get("data_frac_spread") or {}
    if spread.get("mean_spread") is not None and np.isfinite(float(spread["mean_spread"])):
        out.append(
            f"    mean within-shell data% spread (p90−p10) = "
            f"{100.0 * float(spread['mean_spread']):.1f} pts "
            "— shrinks when the normalization absorbs the anisotropy"
        )
    return out


def format_shell_nuisance(
    sa_params: Optional[dict[str, Any]],
    nu_params: Optional[dict[str, Any]] = None,
) -> list[str]:
    """Render the per-shell σ_A and β fit, with errors and the normalization check.

    ``σ_A² + β`` is the column to read: it is 1 exactly when the Wilson normalization is
    right, so a systematic departure says the normalization is off — which is precisely the
    error that a constrained β would have hidden inside σ_A instead of showing here.

    Shells where either parameter sits on a bound are marked ``*``: the sigmoid's gradient
    collapses there, so the value is pinned rather than fitted and its error bar is
    meaningless however small it prints.
    """
    if not sa_params:
        return []
    p = dict(sa_params)
    sa = p.get("bin_sigma_a")
    beta = p.get("bin_beta")
    if not isinstance(sa, (list, tuple)) or not isinstance(beta, (list, tuple)):
        return []
    if len(sa) == 0 or len(beta) != len(sa):
        return []
    centers = p.get("bin_centers_s2") or []
    sa_se = p.get("sigma_a_se") or []
    b_se = p.get("beta_se") or []
    corr = p.get("sigma_a_beta_correlation") or []
    at_bound = set(int(i) for i in (p.get("bins_at_bound") or []))

    mode = str(p.get("beta_mode", "constrained"))
    shape = str(p.get("sigma_a_shape", "monotone"))
    head = f"  β: {mode}"
    if mode == "free":
        head += (
            f" (smooth λ_u={float(p.get('lambda_u', 0.0)):.3g}"
            f" λ_v={float(p.get('lambda_v', 0.0)):.3g}"
        )
        if float(p.get("lambda_consistency", 0.0)) > 0.0:
            head += f" λ_c={float(p['lambda_consistency']):.3g}"
        head += ")"
    head += f"  σ_A: {shape}"
    if p.get("consistency_mean_abs_error") is not None:
        head += (
            f"  |σ_A²+β−1|: mean={float(p['consistency_mean_abs_error']):.4f}"
            f" max={float(p['consistency_max_abs_error']):.4f}"
        )
    if at_bound:
        head += f"  at bound: {len(at_bound)} shell(s)"
    out = [head]
    if p.get("beta_fallback"):
        out.append(f"    NOTE: β not fitted — {p['beta_fallback']}")
    if p.get("sigma_a_tensor_fallback"):
        out.append(f"    NOTE: σ_A/β tensor — {p['sigma_a_tensor_fallback']}")
    m_a, m_b = p.get("M_A"), p.get("M_beta")
    if isinstance(m_a, dict) or isinstance(m_b, dict):
        def _eig(block: Any) -> str:
            if not isinstance(block, dict):
                return "."
            ev = block.get("eigenvalues") or []
            dlt = block.get("delta_aniso")
            if not ev:
                return "."
            evs = "/".join(f"{float(x):.3f}" for x in ev)
            extra = f"  Δ={float(dlt):.3f}" if dlt is not None else ""
            return f"{evs}{extra}"

        out.append(
            f"    M_A [{p.get('laue', '?')}]: {_eig(m_a)}"
            f"   M_β: {_eig(m_b)}"
            f"   λ_sph={float(p.get('lambda_sphericity', 0.0)):.3g}"
            f"   n={int(p.get('n_tensor_params', 0))}"
        )

    def _num(seq: Any, i: int, width: int, dec: int) -> str:
        try:
            v = seq[i]
        except (IndexError, TypeError):
            return f"{'.':>{width}}"
        if v is None or not np.isfinite(float(v)):
            return f"{'.':>{width}}"
        return f"{float(v):>{width}.{dec}f}"

    nu_bins: list[Any] = []
    np_ = dict(nu_params or {})
    if np_.get("mode") in ("bins", "grid_bins") and isinstance(np_.get("bin_nu"), (list, tuple)):
        nu_bins = list(np_["bin_nu"])
    nu_hdr = f" {'ν':>6}" if nu_bins else ""
    out.append(
        f"    {'shell':>5} {'d(Å)':>7} {'σ_A':>7} {'±':>6} "
        f"{'β':>7} {'±':>6} {'σ_A²+β':>7} {'corr':>6}{nu_hdr}"
    )
    for i in range(len(sa)):
        try:
            s_sq = float(centers[i])
            d_txt = f"{1.0 / math.sqrt(s_sq):>7.2f}" if s_sq > 0 else f"{'.':>7}"
        except (IndexError, TypeError, ValueError):
            d_txt = f"{'.':>7}"
        cons = float(sa[i]) ** 2 + float(beta[i])
        flag = "*" if i in at_bound else " "
        nu_txt = f" {_num(nu_bins, i, 6, 2)}" if nu_bins else ""
        out.append(
            f"    {i + 1:>5} {d_txt} {float(sa[i]):>7.3f} {_num(sa_se, i, 6, 3)} "
            f"{float(beta[i]):>7.3f} {_num(b_se, i, 6, 3)} {cons:>7.3f} "
            f"{_num(corr, i, 6, 2)}{flag}{nu_txt}"
        )
    return out


def _nu_col_label(v: float) -> str:
    if not np.isfinite(float(v)):
        return "G"
    if abs(float(v) - round(float(v))) < 1e-6:
        return str(int(round(float(v))))
    return f"{float(v):g}"


def format_nu_grid_profiles(nu_params: Optional[dict[str, Any]]) -> list[str]:
    """ν-grid NLL and the per-shell σ_A / β that were refit at each node.

    Printed on the cycle that searched ν. Later cycles hold ν and only show the
    selected σ_A/β table.
    """
    p = dict(nu_params or {})
    grid = p.get("grid")
    nll = p.get("grid_nll")
    if p.get("mode") not in ("grid", "grid_bins") or not isinstance(grid, (list, tuple)) or not isinstance(nll, (list, tuple)):
        return []
    if len(grid) == 0 or len(nll) != len(grid):
        return []

    g_nll = p.get("grid_nll_gaussian")
    chosen = p.get("nu")
    try:
        ref = float(g_nll) if g_nll is not None else float(min(nll))
    except (TypeError, ValueError):
        ref = float(nll[0])

    def _mean_row(rows: Any) -> list[float]:
        out: list[float] = []
        if not isinstance(rows, (list, tuple)):
            return out
        for row in rows:
            if not isinstance(row, (list, tuple)) or not row:
                out.append(float("nan"))
                continue
            vals = [float(x) for x in row if x is not None and np.isfinite(float(x))]
            out.append(float(np.mean(vals)) if vals else float("nan"))
        return out

    mean_sa = _mean_row(p.get("grid_sigma_a"))
    mean_b = _mean_row(p.get("grid_beta"))
    g_sa = p.get("gaussian_sigma_a") or []
    g_b = p.get("gaussian_beta") or []
    g_mean_sa = (
        float(np.mean([float(x) for x in g_sa if np.isfinite(float(x))]))
        if isinstance(g_sa, (list, tuple)) and g_sa
        else float("nan")
    )
    g_mean_b = (
        float(np.mean([float(x) for x in g_b if np.isfinite(float(x))]))
        if isinstance(g_b, (list, tuple)) and g_b
        else float("nan")
    )

    def _num(v: Any, width: int, dec: int) -> str:
        try:
            x = float(v)
        except (TypeError, ValueError):
            return f"{'.':>{width}}"
        if not np.isfinite(x):
            return f"{'.':>{width}}"
        return f"{x:>{width}.{dec}f}"

    out = [
        "  ν grid"
        + (" per shell" if p.get("mode") == "grid_bins" else "")
        + " (σ_A,β refit at each node; NLL per tune reflection)",
        f"    {'ν':>6} {'NLL':>8} {'Δ vs G':>8} {'⟨σ_A⟩':>7} {'⟨β⟩':>7}",
    ]
    for i, (nu_i, nll_i) in enumerate(zip(grid, nll)):
        mark = (
            "*"
            if p.get("mode") != "grid_bins"
            and chosen is not None
            and abs(float(nu_i) - float(chosen)) < 1e-6
            else " "
        )
        sa_i = mean_sa[i] if i < len(mean_sa) else float("nan")
        b_i = mean_b[i] if i < len(mean_b) else float("nan")
        out.append(
            f"    {_nu_col_label(float(nu_i)):>6} {_num(nll_i, 8, 4)} "
            f"{_num(float(nll_i) - ref, 8, 4)} {_num(sa_i, 7, 3)} {_num(b_i, 7, 3)}{mark}"
        )
    if g_nll is not None:
        out.append(
            f"    {'G':>6} {_num(g_nll, 8, 4)} {_num(0.0, 8, 4)} "
            f"{_num(g_mean_sa, 7, 3)} {_num(g_mean_b, 7, 3)}"
        )
    bin_nu = p.get("bin_nu") if p.get("mode") == "grid_bins" else None
    if isinstance(bin_nu, (list, tuple)) and bin_nu:
        labels = []
        for v in bin_nu:
            labels.append("G" if float(v) >= 199.0 else _nu_col_label(float(v)))
        out.append("    selected ν(s): " + " ".join(labels))
    elif chosen is not None:
        out.append(f"    selected ν={float(chosen):g}  (*)")

    centers = p.get("bin_centers_s2") or []
    nll_shell = p.get("grid_nll_shell")
    sa_shell = p.get("grid_sigma_a")
    n_shells = 0
    for block in (sa_shell, nll_shell, p.get("grid_beta")):
        if isinstance(block, (list, tuple)) and block and isinstance(block[0], (list, tuple)):
            n_shells = max(n_shells, len(block[0]))
    if n_shells == 0:
        return out

    labels = [_nu_col_label(float(v)) for v in grid] + (["G"] if g_nll is not None else [])
    col_w = max(6, max(len(x) for x in labels))

    def _d_txt(i: int) -> str:
        try:
            s_sq = float(centers[i])
            return f"{1.0 / math.sqrt(s_sq):>7.2f}" if s_sq > 0 else f"{'.':>7}"
        except (IndexError, TypeError, ValueError):
            return f"{'.':>7}"

    def _matrix_block(title: str, rows_by_nu: Any, gaussian_row: Any, dec: int) -> list[str]:
        if not isinstance(rows_by_nu, (list, tuple)) or len(rows_by_nu) != len(grid):
            return []
        head = f"    {'shell':>5} {'d(Å)':>7} " + " ".join(f"{lab:>{col_w}}" for lab in labels)
        lines = [f"  {title}", head]
        n_sh = min(n_shells, max((len(r) for r in rows_by_nu if isinstance(r, (list, tuple))), default=0))
        n_tune = p.get("grid_n_shell") or []
        for k in range(n_sh):
            cells = []
            for row in rows_by_nu:
                try:
                    cells.append(_num(row[k], col_w, dec))
                except (IndexError, TypeError):
                    cells.append(f"{'.':>{col_w}}")
            if g_nll is not None:
                try:
                    cells.append(_num(gaussian_row[k], col_w, dec) if gaussian_row else f"{'.':>{col_w}}")
                except (IndexError, TypeError):
                    cells.append(f"{'.':>{col_w}}")
            n_txt = ""
            try:
                n_txt = f"  n={int(n_tune[k])}" if n_tune else ""
            except (IndexError, TypeError, ValueError):
                n_txt = ""
            lines.append(
                f"    {k + 1:>5} {_d_txt(k)} " + " ".join(cells) + n_txt
            )
        return lines

    out.extend(
        _matrix_block(
            "per-shell NLL (tune, per reflection)",
            nll_shell,
            p.get("gaussian_nll_shell"),
            4,
        )
    )
    out.extend(
        _matrix_block(
            "per-shell σ_A (α) at each ν",
            p.get("grid_sigma_a"),
            p.get("gaussian_sigma_a"),
            3,
        )
    )
    out.extend(
        _matrix_block(
            "per-shell β at each ν",
            p.get("grid_beta"),
            p.get("gaussian_beta"),
            3,
        )
    )
    return out


def format_cc_isig_table(table: CCIsigTable) -> list[str]:
    """Render the ``CC_I`` table: a CC line and a population line per shell."""
    if not table.rows or table.n_bins == 0:
        return []

    def _v(x: float) -> str:
        # Four characters, always. A negative CC drops its leading zero ("-.05") so a
        # weak cell -- exactly what this table exists to show -- cannot shift a column.
        if not np.isfinite(x):
            return f"{'.':>4}"
        s = f"{x:.2f}"
        return s if not s.startswith("-") else "-" + s[2:]

    def cc_cell(c: CCIsigCell) -> str:
        return f"{_v(c.cc_work)}/{_v(c.cc_free)}"

    def frac_cell(c: CCIsigCell) -> str:
        return f"{100.0 * c.frac:8.1f}%" if np.isfinite(c.frac) else f"{'.':>9}"

    lead = f"{'shell':>5} {'d_max':>7} {'d_min':>7} {'σ_A':>6} {'Nw':>6} {'Nf':>5}"
    header = (
        lead + " " + " ".join(lbl.center(9) for lbl in table.labels) + " " + "all".center(9)
    )
    width = len(header)
    blank = f"{'':>5} {'':>7} {'':>7} {'':>6} {'':>6} {'':>5} "
    lines = ["=" * width]
    lines.append("CC_I = corr(I_obs, I_calc) by I/σ bin and resolution shell")
    lines.append(
        "  per shell: first line CC_I work/free, second line % of that shell's reflections"
    )
    lines.append("=" * width)
    lines.append(header)
    lines.append("-" * width)
    for row in table.rows:
        sa = f"{row.mean_sigma_a:6.3f}" if np.isfinite(row.mean_sigma_a) else f"{'n/a':>6}"
        lines.append(
            f"{row.shell:5d} {row.d_max:7.3f} {row.d_min:7.3f} {sa} "
            f"{row.n_work:6d} {row.n_free:5d} "
            + " ".join(cc_cell(c) for c in row.cells)
            + " "
            + cc_cell(row.marginal)
        )
        lines.append(
            blank + " ".join(frac_cell(c) for c in row.cells) + " " + frac_cell(row.marginal)
        )
    lines.append("-" * width)
    lines.append(
        f"{'all':>5} {'':>7} {'':>7} {'':>6} "
        f"{table.overall.n_work:6d} {table.overall.n_free:5d} "
        + " ".join(cc_cell(c) for c in table.column_marginal)
        + " "
        + cc_cell(table.overall)
    )
    lines.append(
        blank
        + " ".join(frac_cell(c) for c in table.column_marginal)
        + " "
        + frac_cell(table.overall)
    )
    lines.append("-" * width)
    lines.append(
        "  I/σ bins are FIXED boundaries, not quantiles, so a column means the same thing\n"
        "  in every shell, in every run, and across data sets. The populations are\n"
        "  therefore uneven by design — which is why every CC sits directly above its own\n"
        "  share of the shell (the % line; each % line sums to 100).\n"
        "  Down a column: fixed I/σ, varying resolution — that isolates the σ_A dependence,\n"
        "  read against the σ_A column. Across a row: fixed resolution and σ_A, varying I/σ.\n"
        "  CC_I is model-free on the observation side (raw I_obs), unlike the posterior CC\n"
        f"  of <F> vs |F_c|, which is shrunk toward the model. Cells with fewer than "
        f"{table.min_n}\n"
        "  reflections show '.' for CC but still report their population share.\n"
        "  Within-cell CC is attenuated by range restriction — a narrow I/σ band selects on\n"
        "  I_obs itself — so compare cells with each other, never with the 'all' column.\n"
        "  The <0 column is the extreme of that: conditioning on I_obs<0 means a larger\n"
        "  I_calc needs more negative noise to get there, so the CC is driven negative by\n"
        "  selection alone. Read that column as a population count, not as model quality."
    )
    lines.append("=" * width)
    return lines


def format_intensity_stats_report(report: IntensityStatsReport) -> str:
    """Render a multi-line ASCII report suitable for phenix.refine logs."""
    lines: list[str] = []
    border = "=" * 96
    lines.append(border)
    title = "mli_quad intensity / σ_A resolution report"
    if report.label:
        title = f"{title} [{report.label}]"
    lines.append(title)
    lines.append(
        f"  N={report.n_refl}  bin_size={report.bin_size}  bins={len(report.bins)}  "
        f"⟨σ_A⟩={report.mean_sigma_a:.4f}  ⟨data_frac⟩={report.mean_data_frac:.3f}  "
        f"neg_all={100.0 * report.frac_neg_all:.1f}%"
    )
    sa_p = report.sigma_a_params
    sw_p = report.sigma_wilson_params
    bits = []
    if sa_p:
        mode = sa_p.get("mode", "read")
        bits.append(f"σ_A[{mode}]")
        if "n_bins" in sa_p:
            bits.append(f"n_bins={sa_p['n_bins']}")
        if "k" in sa_p:
            bits.append(f"k≈{sa_p['k']:.4f}")
        if "b_delta" in sa_p:
            bits.append(f"BΔ≈{sa_p['b_delta']:.3f}")
        bin_sa = sa_p.get("bin_sigma_a")
        if isinstance(bin_sa, (list, tuple)) and len(bin_sa) >= 2:
            bits.append(f"σ_A: {float(bin_sa[0]):.3f}→{float(bin_sa[-1]):.3f}")
    if report.nu is not None:
        bits.append(f"ν={float(report.nu):.2f}")
    if bits:
        lines.append("  " + "  ".join(bits))
    lines.extend(format_shell_nuisance(sa_p, nu_params=report.nu_params))
    lines.extend(format_nu_grid_profiles(report.nu_params))
    lines.extend(format_wilson_normalization(sw_p))
    # The overall anisotropic B of the global scale. B_iso_equiv is shown but is shared
    # with k_isotropic; the anisotropy is the part only this tensor can supply.
    an = report.aniso_scale
    if an is not None and an.is_valid:
        b = [b_field(x, 3) for x in an.b_cart]
        lines.append(
            f"  Aniso B (scale, Å²): B11={b[0]} B22={b[1]} B33={b[2]} "
            f"B12={b[3]} B13={b[4]} B23={b[5]}"
        )
        lines.append(
            f"    principal {an.principal_b[0]:.2f}/{an.principal_b[1]:.2f}/"
            f"{an.principal_b[2]:.2f}  anisotropy={an.anisotropy:.3f}  "
            f"B_iso equiv={b_field(an.b_iso_equiv, 3)} (shared with k_iso)  "
            f"log-RMS resid={an.log_rms_residual:.4f}"
        )
    if (
        np.isfinite(report.s_post_work)
        or np.isfinite(report.k_s_work)
        or report.s_report_version is not None
    ):
        lines.append(
            f"  S_post: work={_fmt_r(report.s_post_work)} free={_fmt_r(report.s_post_free)}  "
            f"S_prior={_fmt_r(report.s_prior_work)}/{_fmt_r(report.s_prior_free)}  "
            f"k_S={_fmt_r(report.k_s_work)}"
            + (
                f"  latent={_fmt_r(1.0 - report.rho2_work)}"
                if np.isfinite(report.rho2_work)
                else ""
            )
            + (f"  ver={report.s_report_version}" if report.s_report_version is not None else "")
        )

    # The conventional R factor, named for the amplitudes it uses, then the posterior
    # point estimates under their own heading. The posterior ones deliberately do not
    # carry the letter R: they are biased low by shrinkage and improve as the data get
    # worse, so a reader who skims must not be able to mistake them for an R factor.
    if np.isfinite(report.r_fw_work) or np.isfinite(report.r_fw_free):
        lines.append(
            f"  R (French-Wilson amplitudes, model-free F_obs — the conventional R): "
            f"work={_fmt_r(report.r_fw_work)} free={_fmt_r(report.r_fw_free)}"
        )
    if np.isfinite(report.r_intensity_work) or np.isfinite(report.r_intensity_free):
        lines.append(
            f"  Direct Intensity R (I_obs vs I_calc): "
            f"work={_fmt_r(report.r_intensity_work)} free={_fmt_r(report.r_intensity_free)}"
        )
    if any(
        np.isfinite(x)
        for x in (report.r_post_work, report.r_post_free, report.r_mode_work, report.r_mode_free)
    ):
        lines.append(
            "  Shrunken |F| agreement — posterior point estimates, biased low, NOT R factors:"
        )
        lines.append(
            f"    mean <F>: work={_fmt_r(report.r_post_work)} free={_fmt_r(report.r_post_free)}"
            f"   mode: work={_fmt_r(report.r_mode_work)} free={_fmt_r(report.r_mode_free)}"
            "   (fall as the data weaken)"
        )
    lines.append(border)

    header = (
        f"{'bin':>4} {'n':>5} {'d_max':>7} {'d_min':>7} "
        f"{'<I/σ>':>7} {'neg%':>6} "
        f"{'<1%':>5} {'<2%':>5} {'<3%':>5} {'<4%':>5} {'<5%':>5} "
        f"{'σ_A':>6} {'data%':>6} {'d%spr':>6} {'S_post':>7} {'S_prior':>7} {'1-ρ²':>6}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for r in report.bins:
        sa_s = f"{r.mean_sigma_a:6.3f}" if np.isfinite(r.mean_sigma_a) else f"{'n/a':>6}"
        df_s = f"{100.0 * r.mean_data_frac:5.1f}%" if np.isfinite(r.mean_data_frac) else f"{'n/a':>6}"
        spread = r.data_frac_p90 - r.data_frac_p10
        sp_s = f"{100.0 * spread:5.1f}%" if np.isfinite(spread) else f"{'n/a':>6}"
        ri_s = f"{r.s_post:7.4f}" if np.isfinite(r.s_post) else f"{'n/a':>7}"
        rf_s = f"{r.s_prior:7.4f}" if np.isfinite(r.s_prior) else f"{'n/a':>7}"
        lat_s = f"{1.0 - r.rho2:6.3f}" if np.isfinite(r.rho2) else f"{'n/a':>6}"
        lines.append(
            f"{r.bin:4d} {r.n:5d} {r.d_max:7.3f} {r.d_min:7.3f} "
            f"{r.mean_isig:7.2f} {100.0 * r.frac_neg:5.1f}% "
            f"{100.0 * r.frac_lt_1:4.1f}% {100.0 * r.frac_lt_2:4.1f}% "
            f"{100.0 * r.frac_lt_3:4.1f}% {100.0 * r.frac_lt_4:4.1f}% "
            f"{100.0 * r.frac_lt_5:4.1f}% "
            f"{sa_s} {df_s} {sp_s} {ri_s} {rf_s} {lat_s}"
        )
    lines.append("-" * len(header))
    lines.append(
        "  neg% / <k% = fraction I<0 / I/σ<k.  "
        "data% ≈ measurement precision vs Wilson prior: σ_Z^{-2}/(1+σ_Z^{-2}), "
        "σ_Z=σ_I/(ε Σ_W).  d%spr = p90−p10 of data% inside the shell; resolution is "
        "fixed there, so under an isotropic Σ_W this carries the anisotropy and it "
        "should shrink under the tensor fit.  S_post / S_prior = expected residual under the posterior / "
        "under the prior (σ_A floor). Same functional, different measure. NOT the "
        "crystallographic R factor; do not compare with deposited R values.  "
        "1-ρ² = latent share of the L2 residual, "
        "Σ Var_post(E) / Σ <(E-k E_C)²> (descriptive, not a calibrated test)."
    )
    lines.append(border)
    if report.cc_isig is not None:
        cc_lines = format_cc_isig_table(report.cc_isig)
        if cc_lines:
            lines.append("")
            lines.extend(cc_lines)
    return "\n".join(lines)


def print_intensity_stats_report(
    report: IntensityStatsReport,
    out: Optional[TextIO] = None,
    extra_streams: Optional[Sequence[TextIO]] = None,
) -> None:
    text = format_intensity_stats_report(report)
    streams: list[TextIO] = [out or sys.stdout]
    if extra_streams:
        for s in extra_streams:
            if s is not None and s not in streams:
                streams.append(s)
    for stream in streams:
        try:
            print(text, file=stream)
            if hasattr(stream, "flush"):
                stream.flush()
        except Exception:
            pass


def _finite_r(v: Any, *, hi: float = 5.0) -> float:
    """Accept only finite R in ``[0, hi]``; else NaN (guards stale-worker blowups)."""
    try:
        if v is None:
            return float("nan")
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x if np.isfinite(x) and 0.0 <= x <= hi else float("nan")


def merge_sstat_bins_into_report(report: IntensityStatsReport, r_values: dict[str, Any]) -> IntensityStatsReport:
    """Copy per-shell S_post / S_prior from ``ml_i_maps`` ``r_values`` into the report."""
    if not report.bins:
        return report
    r_iw = r_values.get("s_post_work_bins") or []
    r_if = r_values.get("s_prior_work_bins") or []
    r_iwf = r_values.get("s_post_free_bins") or []
    r_iff = r_values.get("s_prior_free_bins") or []
    rho2_w = r_values.get("rho2_work_bins") or []
    rho2_f = r_values.get("rho2_free_bins") or []
    n_w = r_values.get("n_work_bins") or []
    n_f = r_values.get("n_free_bins") or []
    for i, row in enumerate(report.bins):
        if i < len(r_iw):
            row.s_post = _finite_r(r_iw[i])
        if i < len(r_if):
            row.s_prior = _finite_r(r_if[i])
        if i < len(r_iwf):
            row.s_post_free = _finite_r(r_iwf[i])
        if i < len(r_iff):
            row.s_prior_free = _finite_r(r_iff[i])
        if i < len(rho2_w):
            row.rho2 = _finite_r(rho2_w[i], hi=1.0)
        if i < len(rho2_f):
            row.rho2_free = _finite_r(rho2_f[i], hi=1.0)
        if i < len(n_w):
            row.n_work = int(n_w[i])
        if i < len(n_f):
            row.n_free = int(n_f[i])
    return report


def report_from_fmodel(
    fmodel: Any,
    *,
    bin_size: Optional[int] = None,
    label: str = "",
    sigma_a_params: Optional[dict[str, float]] = None,
    sigma_wilson_params: Optional[dict[str, float]] = None,
) -> IntensityStatsReport:
    """Build a report from an IntensityFModel (or compatible) after nuisance fit."""
    i_obs = fmodel.i_obs() if callable(getattr(fmodel, "i_obs", None)) else fmodel._i_obs
    sa = getattr(fmodel, "sigma_a", None)
    sw = getattr(fmodel, "sigma_wilson", None)
    nu = getattr(fmodel, "nu", None)
    if hasattr(nu, "item"):
        try:
            nu = float(nu.item())
        except Exception:
            nu = float(nu) if nu is not None else None
    elif nu is not None:
        try:
            nu = float(nu)
        except Exception:
            nu = None

    return compute_intensity_stats_report(
        intensities=i_obs.data(),
        sigmas=i_obs.sigmas(),
        d_spacings=i_obs.d_spacings().data(),
        epsilon=i_obs.epsilons().data().as_double() if hasattr(i_obs, "epsilons") else None,
        sigma_a=sa,
        sigma_wilson=sw,
        bin_size=bin_size,
        sigma_a_params=sigma_a_params or getattr(fmodel, "_sigma_a_params", None),
        sigma_wilson_params=sigma_wilson_params or getattr(fmodel, "_sigma_wilson_params", None),
        nu=nu,
        nu_params=getattr(fmodel, "_nu_params", None),
        label=label,
    )
