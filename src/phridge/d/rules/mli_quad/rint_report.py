"""Integrated S report for the intensity likelihood."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import torch

from phridge.d.rules.mli_quad.rint import (
    PRIOR_ONLY_SIGMA_Z,
    _mask_all_ones,
    absdev_given_k,
    aggregate_s,
    delta_given_cache,
    equal_count_shell_indices,
    posterior_amplitude_cache,
    predictive_delta_variance,
    scale_k,
)

Tensor = torch.Tensor


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



def _nu_for_report(nu, Ec):
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
    return nu_per_refl, nu_scalar


def _masks_for_report(Ec, centric, work_mask, free_mask, ec_outlier_ratio):
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
    return work, free, keep, explicit_free, n_ec_outliers, ec_median, ec_max, centric_t, n, device


def _nu_chunk(nu_per_refl, nu_scalar, sl):
    if nu_per_refl is not None:
        return nu_per_refl[sl]
    return nu_scalar


def _scale_k_or_nan(E_m, Ec, mask, free, explicit_free):
    if not bool(mask.any()):
        if explicit_free and mask is free:
            warnings.warn("scale_k: empty free set; returning nan", stacklevel=2)
        return float("nan")
    return scale_k(E_m, Ec, mask)


def _agg_s_or_nan(E_m, absdev, mask, free, explicit_free):
    if not bool(mask.any()):
        if explicit_free and mask is free:
            warnings.warn("aggregate_s: empty free set; returning nan", stacklevel=2)
        return float("nan")
    return aggregate_s(E_m, absdev, mask)


def _plug_s(E_m, Ec, k, mask):
    if not math.isfinite(k) or not bool(mask.any()):
        return float("nan")
    return aggregate_s(E_m, (E_m - k * Ec).abs(), mask)


def _split_latent(k, mask, E_mean, var_post, Ec):
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


def _xi_of(mask, score_test, delta_obs, ref_var):
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


def _omega_of(xi_f, xi_w):
    if not (math.isfinite(xi_f) and math.isfinite(xi_w)) or xi_w <= 0:
        return float("nan")
    return xi_f / xi_w


def _posterior_passes(
    Ec, sA, Zo, sZ, centric_t, work, free, keep, explicit_free,
    nu_per_refl, nu_scalar, n, device, score_test, n_predictive_nodes,
    n_u, snr_strong, n_hermite, n_legendre, k_window, prior_only_sigma_z, chunk_size,
):
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

    E_mean = torch.empty(n, dtype=torch.float64, device=device)
    E_mean_prior = torch.empty(n, dtype=torch.float64, device=device)
    var_post = torch.empty(n, dtype=torch.float64, device=device)
    delta_obs = torch.empty(n, dtype=torch.float64, device=device)
    ref_var = torch.empty(n, dtype=torch.float64, device=device)
    for start in range(0, n, cs):
        stop = min(start + cs, n)
        sl = slice(start, stop)
        nu_c = _nu_chunk(nu_per_refl, nu_scalar, sl)
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

    k_w = _scale_k_or_nan(E_mean, Ec, work, free, explicit_free)
    k_f = _scale_k_or_nan(E_mean, Ec, free, free, explicit_free)
    k_a = scale_k(E_mean, Ec, keep)
    ki_w = _scale_k_or_nan(E_mean_prior, Ec, work, free, explicit_free)
    ki_f = _scale_k_or_nan(E_mean_prior, Ec, free, free, explicit_free)
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
        nu_c = _nu_chunk(nu_per_refl, nu_scalar, sl)
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

    return (
        E_mean, E_mean_prior, var_post, delta_obs, ref_var,
        abs_w, abs_f, abs_a, absi_w, absi_f, absi_a,
        k_w, k_f, k_a, ki_w, ki_f, ki_a,
    )


def _reduce_report(
    Ec, E_mean, E_mean_prior, var_post, delta_obs, ref_var,
    abs_w, abs_f, abs_a, absi_w, absi_f, absi_a,
    k_w, k_f, k_a, work, free, keep, explicit_free, score_test,
):
    s_post_w = _agg_s_or_nan(E_mean, abs_w, work, free, explicit_free)
    s_post_f = _agg_s_or_nan(E_mean, abs_f, free, free, explicit_free)
    s_post_a = aggregate_s(E_mean, abs_a, keep)
    s_prior_w = _agg_s_or_nan(E_mean_prior, absi_w, work, free, explicit_free)
    s_prior_f = _agg_s_or_nan(E_mean_prior, absi_f, free, free, explicit_free)
    s_prior_a = aggregate_s(E_mean_prior, absi_a, keep)
    vv_w, vl_w, rho2_w = _split_latent(k_w, work, E_mean, var_post, Ec)
    vv_f, vl_f, rho2_f = _split_latent(k_f, free, E_mean, var_post, Ec)
    vv_a, vl_a, rho2_a = _split_latent(k_a, keep, E_mean, var_post, Ec)
    xi_w = _xi_of(work, score_test, delta_obs, ref_var)
    xi_f = _xi_of(free, score_test, delta_obs, ref_var)
    xi_a = _xi_of(keep, score_test, delta_obs, ref_var)
    return dict(
        s_post_w=s_post_w, s_post_f=s_post_f, s_post_a=s_post_a,
        s_prior_w=s_prior_w, s_prior_f=s_prior_f, s_prior_a=s_prior_a,
        vv_w=vv_w, vl_w=vl_w, rho2_w=rho2_w,
        vv_f=vv_f, vl_f=vl_f, rho2_f=rho2_f,
        vv_a=vv_a, vl_a=vl_a, rho2_a=rho2_a,
        xi_w=xi_w, xi_f=xi_f, xi_a=xi_a,
        s_vis_w=_plug_s(E_mean, Ec, k_w, work),
        s_vis_f=_plug_s(E_mean, Ec, k_f, free) if bool(free.any()) else float("nan"),
        s_vis_a=_plug_s(E_mean, Ec, k_a, keep),
    )


def _integrated_result(scalars, k_w, k_f, k_a, ki_w, ki_f, ki_a, min_free_per_shell, n_ec_outliers, ec_median, ec_max):
    return IntegratedSResult(
        s_post_work=scalars["s_post_w"],
        s_post_free=scalars["s_post_f"],
        s_post_all=scalars["s_post_a"],
        s_prior_work=scalars["s_prior_w"],
        s_prior_free=scalars["s_prior_f"],
        s_prior_all=scalars["s_prior_a"],
        k_s_work=k_w,
        k_s_free=k_f,
        k_s_all=k_a,
        k_s_prior_work=ki_w,
        k_s_prior_free=ki_f,
        k_s_prior_all=ki_a,
        s_vis_work=scalars["s_vis_w"],
        s_vis_free=scalars["s_vis_f"],
        s_vis_all=scalars["s_vis_a"],
        v_vis_work=scalars["vv_w"],
        v_vis_free=scalars["vv_f"],
        v_vis_all=scalars["vv_a"],
        v_lat_work=scalars["vl_w"],
        v_lat_free=scalars["vl_f"],
        v_lat_all=scalars["vl_a"],
        rho2_work=scalars["rho2_w"],
        rho2_free=scalars["rho2_f"],
        rho2_all=scalars["rho2_a"],
        xi_work=scalars["xi_w"],
        xi_free=scalars["xi_f"],
        xi_all=scalars["xi_a"],
        omega=_omega_of(scalars["xi_f"], scalars["xi_w"]),
        min_free_per_shell=int(min_free_per_shell),
        n_ec_outliers=n_ec_outliers,
        ec_median=ec_median,
        ec_max=ec_max,
    )


def _attach_shells(
    result, d_spacings, bin_size, device, E_mean, E_mean_prior, var_post, Ec,
    abs_w, abs_f, absi_w, absi_f, delta_obs, ref_var, work, free,
    k_w, k_f, score_test, min_free_per_shell,
):
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
                vvw[b], vlw[b], r2w[b] = _split_latent(k_w, in_b & work, E_mean, var_post, Ec)
                vvf[b], vlf[b], r2f[b] = _split_latent(k_f, in_b & free, E_mean, var_post, Ec)
                xiw[b] = _xi_of(in_b & work, score_test, delta_obs, ref_var)
                xif[b] = _xi_of(in_b & free, score_test, delta_obs, ref_var)

            # A sparse free set is expected, not an error: blank the free-set shell
            # statistics below the threshold rather than printing noise, and do it
            # silently. Counts stay real so the reader can see why.
            thin = nf < max(int(min_free_per_shell), 0)
            for arr in (rf, rif, vvf, vlf, r2f, xif):
                arr[thin] = np.nan
            for b in range(n_bins):
                omg[b] = _omega_of(float(xif[b]), float(xiw[b]))

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
    nu_per_refl, nu_scalar = _nu_for_report(nu, Ec)
    (
        work, free, keep, explicit_free, n_ec_outliers, ec_median, ec_max, centric_t, n, device,
    ) = _masks_for_report(Ec, centric, work_mask, free_mask, ec_outlier_ratio)
    (
        E_mean, E_mean_prior, var_post, delta_obs, ref_var,
        abs_w, abs_f, abs_a, absi_w, absi_f, absi_a,
        k_w, k_f, k_a, ki_w, ki_f, ki_a,
    ) = _posterior_passes(
        Ec, sA, Zo, sZ, centric_t, work, free, keep, explicit_free,
        nu_per_refl, nu_scalar, n, device, score_test, n_predictive_nodes,
        n_u, snr_strong, n_hermite, n_legendre, k_window, prior_only_sigma_z, chunk_size,
    )
    scalars = _reduce_report(
        Ec, E_mean, E_mean_prior, var_post, delta_obs, ref_var,
        abs_w, abs_f, abs_a, absi_w, absi_f, absi_a,
        k_w, k_f, k_a, work, free, keep, explicit_free, score_test,
    )
    result = _integrated_result(
        scalars, k_w, k_f, k_a, ki_w, ki_f, ki_a, min_free_per_shell, n_ec_outliers, ec_median, ec_max,
    )
    _attach_shells(
        result, d_spacings, bin_size, device, E_mean, E_mean_prior, var_post, Ec,
        abs_w, abs_f, absi_w, absi_f, delta_obs, ref_var, work, free,
        k_w, k_f, score_test, min_free_per_shell,
    )
    return result




