"""Resolution-binned intensity diagnostics printed during mli_quad refinement.

Per bin (default 500 reflections, low→high resolution):
  - mean I/σ, fraction negative, fraction with I/σ < 1..5
  - mean σ_A(s) and Wilson Σ_W(s) from the current nuisance fit
  - mean data-fraction vs Wilson prior:  σ_Z^{-2} / (1 + σ_Z^{-2})
    with σ_Z = σ_I / (ε Σ_W)  (model-free; ≈0 = prior-dominated)
  - S_post / S_prior when ``ml_i_maps`` bins are available (expected residual
    under the posterior / under the prior — never write a bare ``S``)

Enable/disable with ``PHRIDGE_STATS_REPORT`` (default on). Bin size:
``PHRIDGE_STATS_BIN_SIZE`` (default 500).
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, TextIO

import numpy as np


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
    s_post: float = float("nan")  # S_post (work; parent-set k_S)
    s_prior: float = float("nan")  # S_prior — σ_A-implied no-data floor
    s_post_free: float = float("nan")
    s_prior_free: float = float("nan")
    rho2: float = float("nan")  # visible share v_vis/(v_vis+v_lat), work set
    rho2_free: float = float("nan")
    n_work: int = 0
    n_free: int = 0


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
        label=label,
    )


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
    if sw_p:
        if "sigma_0" in sw_p:
            bits.append(f"Σ_W: Σ0={sw_p['sigma_0']:.3g}")
        if "b_wilson" in sw_p:
            bits.append(f"B_W={sw_p['b_wilson']:.3f}")
    if report.nu is not None:
        bits.append(f"ν={float(report.nu):.2f}")
    if bits:
        lines.append("  " + "  ".join(bits))
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
    lines.append(border)

    header = (
        f"{'bin':>4} {'n':>5} {'d_max':>7} {'d_min':>7} "
        f"{'<I/σ>':>7} {'neg%':>6} "
        f"{'<1%':>5} {'<2%':>5} {'<3%':>5} {'<4%':>5} {'<5%':>5} "
        f"{'σ_A':>6} {'data%':>6} {'S_post':>7} {'S_prior':>7} {'1-ρ²':>6}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for r in report.bins:
        sa_s = f"{r.mean_sigma_a:6.3f}" if np.isfinite(r.mean_sigma_a) else f"{'n/a':>6}"
        df_s = f"{100.0 * r.mean_data_frac:5.1f}%" if np.isfinite(r.mean_data_frac) else f"{'n/a':>6}"
        ri_s = f"{r.s_post:7.4f}" if np.isfinite(r.s_post) else f"{'n/a':>7}"
        rf_s = f"{r.s_prior:7.4f}" if np.isfinite(r.s_prior) else f"{'n/a':>7}"
        lat_s = f"{1.0 - r.rho2:6.3f}" if np.isfinite(r.rho2) else f"{'n/a':>6}"
        lines.append(
            f"{r.bin:4d} {r.n:5d} {r.d_max:7.3f} {r.d_min:7.3f} "
            f"{r.mean_isig:7.2f} {100.0 * r.frac_neg:5.1f}% "
            f"{100.0 * r.frac_lt_1:4.1f}% {100.0 * r.frac_lt_2:4.1f}% "
            f"{100.0 * r.frac_lt_3:4.1f}% {100.0 * r.frac_lt_4:4.1f}% "
            f"{100.0 * r.frac_lt_5:4.1f}% "
            f"{sa_s} {df_s} {ri_s} {rf_s} {lat_s}"
        )
    lines.append("-" * len(header))
    lines.append(
        "  neg% / <k% = fraction I<0 / I/σ<k.  "
        "data% ≈ measurement precision vs Wilson prior: σ_Z^{-2}/(1+σ_Z^{-2}), "
        "σ_Z=σ_I/(ε Σ_W).  S_post / S_prior = expected residual under the posterior / "
        "under the prior (σ_A floor). Same functional, different measure. NOT the "
        "crystallographic R factor; do not compare with deposited R values.  "
        "1-ρ² = latent share of the L2 residual, "
        "Σ Var_post(E) / Σ <(E-k E_C)²> (descriptive, not a calibrated test)."
    )
    lines.append(border)
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
        label=label,
    )
