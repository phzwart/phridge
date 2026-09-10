"""Oracle and property tests for the S family (S_post / S_prior)."""

from __future__ import annotations

import math
from pathlib import Path
import re
import sys

import numpy as np
import pytest
import torch
from scipy.special import i0e

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from phridge.contrib.intensity_ll.mli import posterior_mode_E  # noqa: E402
from phridge.contrib.intensity_ll.rint import (  # noqa: E402
    PRIOR_ONLY_SIGMA_Z,
    absdev_given_k,
    aggregate_s,
    delta_given_cache,
    integrated_s_report,
    posterior_amplitude_cache,
    predictive_delta_variance,
    scale_k,
    sqdev_given_k,
)
from phridge.contrib.intensity_ll.mli import log_likelihood_normal


# --------------------------------------------------------------------------- numpy oracle
def _rice_log_prior_acentric(E: np.ndarray, Ec: float, sa: float) -> np.ndarray:
    """log p(E|E_C) acentric Rice, up to additive constant in E (normalized later)."""
    a2 = max(1.0 - sa * sa, 1e-12)
    # p(E) = (2E/s) exp(-(E^2 + sa^2 Ec^2)/s) I0(2 sa E Ec / s)
    # i0e(x) = exp(-|x|) I0(x) → I0(x) = i0e(x) exp(|x|)
    s = a2
    x = 2.0 * sa * E * Ec / s
    log_i0 = np.log(np.maximum(i0e(x), 1e-300)) + np.abs(x)
    return np.log(np.maximum(2.0 * E / s, 1e-300)) - (E * E + (sa * Ec) ** 2) / s + log_i0


def _woolfson_log_prior_centric(E: np.ndarray, Ec: float, sa: float) -> np.ndarray:
    """log p(E|E_C) centric Woolfson (E >= 0)."""
    a2 = max(1.0 - sa * sa, 1e-12)
    # p(E) ∝ exp(-(E^2 + sa^2 Ec^2)/(2s)) * cosh(sa E Ec / s) / sqrt(s)  (E>0 folded)
    s = a2
    z = sa * E * Ec / s
    # cosh(z) = exp(|z|) * (exp(-2|z|) + 1)/2 ≈ careful in log space
    log_cosh = np.abs(z) + np.log1p(np.exp(-2.0 * np.abs(z))) - math.log(2.0)
    return -0.5 * math.log(s) - (E * E + (sa * Ec) ** 2) / (2.0 * s) + log_cosh


def oracle_moments(
    Zo: float,
    sZ: float,
    Ec: float,
    sa: float,
    *,
    centric: bool = False,
    k: float | None = None,
    prior_only: bool = False,
    n_grid: int = 8000,
) -> tuple[float, float]:
    """Dense-grid ``(<E>, <|E - k Ec|>)`` oracle. If ``k`` is None, returns ``(<E>, <E>)``."""
    E = np.linspace(1e-6, 12.0, n_grid)
    dE = E[1] - E[0]
    if centric:
        log_prior = _woolfson_log_prior_centric(E, Ec, sa)
    else:
        log_prior = _rice_log_prior_acentric(E, Ec, sa)
    if prior_only or sZ >= 0.5 * PRIOR_ONLY_SIGMA_Z:
        log_post = log_prior
    else:
        # N(Zo; E^2, sZ^2)
        log_lik = -0.5 * math.log(2.0 * math.pi * sZ * sZ) - 0.5 * ((Zo - E * E) / sZ) ** 2
        log_post = log_prior + log_lik
    m = np.max(log_post)
    w = np.exp(log_post - m)
    w_sum = np.sum(w) * dE
    if w_sum <= 0 or not np.isfinite(w_sum):
        return float("nan"), float("nan")
    E_mean = float(np.sum(w * E) * dE / w_sum)
    if k is None:
        return E_mean, E_mean
    absdev = float(np.sum(w * np.abs(E - k * Ec)) * dE / w_sum)
    return E_mean, absdev


def _torch_batch_moments(Zo, sZ, Ec, sa, centric=False, nu=None, prior_only=False):
    Zo_t = torch.as_tensor(Zo, dtype=torch.float64)
    sZ_t = torch.as_tensor(sZ, dtype=torch.float64)
    Ec_t = torch.as_tensor(Ec, dtype=torch.float64)
    sa_t = torch.as_tensor(sa, dtype=torch.float64)
    cen = torch.as_tensor(centric, dtype=torch.bool) if np.ndim(centric) else torch.full_like(Zo_t, bool(centric), dtype=torch.bool)
    # Dense GL for oracle agreement (production reporting uses lighter defaults)
    cache = posterior_amplitude_cache(
        Ec_t, sa_t, Zo_t, sZ_t, cen, nu=nu, prior_only=prior_only, snr_strong=1e9, n_legendre=128
    )
    k = scale_k(cache.E_mean, cache.Ec, None)
    absdev = absdev_given_k(cache, k)
    return cache.E_mean.numpy(), absdev.numpy(), k


# --------------------------------------------------------------------------- tests
def test_oracle_agreement_batch():
    rng = np.random.default_rng(0)
    n = 50
    Zo = rng.normal(1.0, 1.5, size=n)
    sZ = rng.uniform(0.15, 0.8, size=n)
    Ec = rng.uniform(0.2, 2.0, size=n)
    sa = rng.choice([0.6, 0.9, 0.99], size=n)
    E_t, abs_t, k = _torch_batch_moments(Zo, sZ, Ec, sa)
    for i in range(n):
        E_o, abs_o = oracle_moments(float(Zo[i]), float(sZ[i]), float(Ec[i]), float(sa[i]), k=k)
        assert abs(E_t[i] - E_o) / max(E_o, 1e-6) < 1e-3
        assert abs(abs_t[i] - abs_o) / max(abs_o, 1e-6) < 1e-3


def test_strong_data_limit_matches_conventional_r():
    rng = np.random.default_rng(1)
    n = 200
    Ec = rng.uniform(0.5, 2.5, size=n)
    sa = np.full(n, 0.85)
    # Positive Z_o near E_true^2 with tiny noise
    E_true = np.sqrt(np.maximum(rng.exponential(1.0, size=n), 1e-6))  # rough
    Zo = E_true**2
    sZ = np.full(n, 1e-3)
    Zo_t = torch.as_tensor(Zo, dtype=torch.float64)
    sZ_t = torch.as_tensor(sZ, dtype=torch.float64)
    Ec_t = torch.as_tensor(Ec, dtype=torch.float64)
    sa_t = torch.as_tensor(sa, dtype=torch.float64)
    cen = torch.zeros(n, dtype=torch.bool)
    rep = integrated_s_report(Ec_t, sa_t, Zo_t, sZ_t, cen)
    # Conventional amplitude R on sqrt(Z_o) with LS scale
    Fo = np.sqrt(np.maximum(Zo, 0.0))
    k_conv = float(np.sum(Fo * Ec) / max(np.sum(Ec * Ec), 1e-300))
    r_conv = float(np.sum(np.abs(Fo - k_conv * Ec)) / max(np.sum(Fo), 1e-300))
    assert abs(rep.s_post_all - r_conv) < 1e-3


def test_prior_only_matches_large_sigma_z_and_oracle():
    Zo = np.array([0.5, 1.2, -0.3, 2.0])
    Ec = np.array([0.8, 1.1, 0.4, 1.5])
    sa = np.array([0.7, 0.9, 0.95, 0.6])
    sZ = np.ones_like(Zo) * 0.4
    Zo_t = torch.as_tensor(Zo, dtype=torch.float64)
    Ec_t = torch.as_tensor(Ec, dtype=torch.float64)
    sa_t = torch.as_tensor(sa, dtype=torch.float64)
    sZ_t = torch.as_tensor(sZ, dtype=torch.float64)
    cen = torch.zeros(4, dtype=torch.bool)

    c_prior = posterior_amplitude_cache(
        Ec_t, sa_t, Zo_t, sZ_t, cen, prior_only=True, snr_strong=1e9, n_legendre=128
    )
    c_flat = posterior_amplitude_cache(
        Ec_t,
        sa_t,
        Zo_t,
        torch.full_like(sZ_t, PRIOR_ONLY_SIGMA_Z),
        cen,
        prior_only=False,
        snr_strong=1e9,
        n_legendre=128,
    )
    np.testing.assert_allclose(c_prior.E_mean.numpy(), c_flat.E_mean.numpy(), rtol=1e-4, atol=1e-5)
    k = scale_k(c_prior.E_mean, c_prior.Ec)
    abs_p = absdev_given_k(c_prior, k).numpy()
    for i in range(4):
        E_o, abs_o = oracle_moments(float(Zo[i]), float(sZ[i]), float(Ec[i]), float(sa[i]), k=k, prior_only=True)
        assert abs(c_prior.E_mean[i].item() - E_o) / max(E_o, 1e-6) < 1e-3
        assert abs(abs_p[i] - abs_o) / max(abs_o, 1e-6) < 1e-3


def _simulate_ensemble(n, sa_true, sa_assumed, sZ_base, noise_mult, rng, noise_draw=None):
    """Acentric ensemble: Wilson E_C, Rice E_true, Z_o = E^2 + sZ*noise."""
    # Wilson-like E_C amplitudes: Rayleigh with scale 1/sqrt(2) → <E_C^2>=1
    Ec = rng.rayleigh(scale=1.0 / math.sqrt(2.0), size=n)
    a2 = 1.0 - sa_true**2
    # Sample E from Rice: use rejection via polar method approximation —
    # E^2 ~ noncentral chi2: Z = |sa Ec + sqrt(a2/2)*(g1+i g2)|^2 for acentric
    g1 = rng.normal(size=n)
    g2 = rng.normal(size=n)
    re = sa_true * Ec + math.sqrt(a2 / 2.0) * g1
    im = math.sqrt(a2 / 2.0) * g2
    E_true = np.sqrt(re * re + im * im)
    if noise_draw is None:
        noise_draw = rng.normal(size=n)
    sZ = sZ_base * noise_mult
    Zo = E_true**2 + sZ * noise_draw
    return Ec, Zo, np.full(n, sZ), np.full(n, sa_assumed), noise_draw


def test_noise_ramp_flatness_and_mode_pathology():
    rng = np.random.default_rng(42)
    n = 4000
    sa = 0.9
    sZ_base = 0.3
    # Fixed Ec, E_true, noise draw; only sZ multiplier changes
    Ec0, Zo0, _, _, noise = _simulate_ensemble(n, sa, sa, sZ_base, 1.0, rng)
    # Rebuild Zo from E_true implicitly: Zo0 = E^2 + sZ_base * noise → E^2 = Zo0 - sZ_base*noise
    E2 = Zo0 - sZ_base * noise
    multipliers = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    s_posts = []
    s_priors = []
    mode_rs = []
    for m in multipliers:
        sZ = sZ_base * m
        Zo = E2 + sZ * noise
        Ec_t = torch.as_tensor(Ec0, dtype=torch.float64)
        Zo_t = torch.as_tensor(Zo, dtype=torch.float64)
        sZ_t = torch.full((n,), sZ, dtype=torch.float64)
        sa_t = torch.full((n,), sa, dtype=torch.float64)
        cen = torch.zeros(n, dtype=torch.bool)
        rep = integrated_s_report(Ec_t, sa_t, Zo_t, sZ_t, cen)
        s_posts.append(rep.s_post_all)
        s_priors.append(rep.s_prior_all)
        # Plug-in mode R
        Em, _ = posterior_mode_E(Ec_t, sa_t, Zo_t, sZ_t, cen)
        Em_np = Em.detach().cpu().numpy()
        k_m = float(np.sum(Em_np * Ec0) / max(np.sum(Ec0 * Ec0), 1e-300))
        mode_rs.append(float(np.sum(np.abs(Em_np - k_m * Ec0)) / max(np.sum(Em_np), 1e-300)))

    assert max(s_posts) - min(s_posts) < 0.01
    for ri, rf in zip(s_posts, s_priors):
        assert abs(ri - rf) < 0.02
    assert mode_rs[0] - mode_rs[-1] > 0.1


def test_inflated_sigma_a_reversion():
    rng = np.random.default_rng(7)
    n = 4000
    sZ_base = 0.3
    Ec0, Zo0, _, _, noise = _simulate_ensemble(n, sa_true=0.7, sa_assumed=0.9, sZ_base=sZ_base, noise_mult=1.0, rng=rng)
    E2 = Zo0 - sZ_base * noise
    multipliers = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    s_posts = []
    s_priors = []
    for m in multipliers:
        sZ = sZ_base * m
        Zo = E2 + sZ * noise
        Ec_t = torch.as_tensor(Ec0, dtype=torch.float64)
        Zo_t = torch.as_tensor(Zo, dtype=torch.float64)
        sZ_t = torch.full((n,), sZ, dtype=torch.float64)
        sa_t = torch.full((n,), 0.9, dtype=torch.float64)
        cen = torch.zeros(n, dtype=torch.bool)
        rep = integrated_s_report(Ec_t, sa_t, Zo_t, sZ_t, cen)
        s_posts.append(rep.s_post_all)
        s_priors.append(rep.s_prior_all)
    for i in range(len(s_posts) - 1):
        assert s_posts[i] >= s_posts[i + 1] - 1e-6
    assert s_posts[0] > 0.35
    assert abs(s_posts[-1] - s_priors[-1]) < 0.02


def test_student_t_consistency():
    rng = np.random.default_rng(3)
    n = 80
    Zo = rng.normal(1.0, 1.2, size=n)
    Zo[0] = -0.5
    sZ = rng.uniform(0.2, 0.6, size=n)
    Ec = rng.uniform(0.3, 2.0, size=n)
    sa = np.full(n, 0.85)
    Ec_t = torch.as_tensor(Ec, dtype=torch.float64)
    Zo_t = torch.as_tensor(Zo, dtype=torch.float64)
    sZ_t = torch.as_tensor(sZ, dtype=torch.float64)
    sa_t = torch.as_tensor(sa, dtype=torch.float64)
    cen = torch.zeros(n, dtype=torch.bool)
    g = integrated_s_report(Ec_t, sa_t, Zo_t, sZ_t, cen, nu=None)
    t200 = integrated_s_report(Ec_t, sa_t, Zo_t, sZ_t, cen, nu=200.0, n_u=24)
    # Finite-n_u Gamma mixture → Gaussian as ν→∞; ν=200 is within a few 1e-3 of normal.
    assert abs(g.s_post_all - t200.s_post_all) < 5e-3
    t1000 = integrated_s_report(Ec_t, sa_t, Zo_t, sZ_t, cen, nu=1000.0, n_u=32)
    assert abs(g.s_post_all - t1000.s_post_all) < 1e-3
    t4 = integrated_s_report(Ec_t, sa_t, Zo_t, sZ_t, cen, nu=4.0)
    assert math.isfinite(t4.s_post_all) and math.isfinite(t4.s_prior_all)


def test_chunked_per_reflection_nu():
    """Bin-wise ν + chunk_size must slice ν; full-length expand used to raise RuntimeError."""
    rng = np.random.default_rng(11)
    n = 5000
    Ec = torch.as_tensor(rng.uniform(0.4, 2.0, size=n), dtype=torch.float64)
    Zo = torch.as_tensor(rng.normal(1.0, 0.8, size=n), dtype=torch.float64)
    sZ = torch.as_tensor(rng.uniform(0.25, 0.5, size=n), dtype=torch.float64)
    sa = torch.full((n,), 0.75, dtype=torch.float64)
    cen = torch.zeros(n, dtype=torch.bool)
    nu = torch.zeros(n, dtype=torch.float64)
    for i, v in enumerate((4.0, 8.0, 16.0, 40.0, 80.0)):
        nu[i * 1000 : (i + 1) * 1000] = v
    d = torch.linspace(30.0, 1.5, n)
    work = torch.ones(n, dtype=torch.bool)
    free = torch.zeros(n, dtype=torch.bool)
    free[::25] = True
    work = work & ~free
    rep = integrated_s_report(
        Ec,
        sa,
        Zo,
        sZ,
        cen,
        work_mask=work,
        free_mask=free,
        nu=nu,
        d_spacings=d,
        bin_size=500,
        chunk_size=1024,
        n_u=8,
        n_legendre=32,
    )
    assert math.isfinite(rep.s_post_work) and math.isfinite(rep.s_prior_work)
    assert math.isfinite(rep.s_post_free)
    assert rep.s_post_work_bins is not None
    assert np.isfinite(rep.s_post_work_bins).sum() >= 5


def test_centric_smoke_and_oracle():
    Zo = np.array([0.4, 1.5, -0.2, 2.2, 0.9])
    sZ = np.array([0.3, 0.25, 0.5, 0.2, 0.35])
    Ec = np.array([0.7, 1.2, 0.5, 1.8, 0.9])
    sa = np.array([0.8, 0.9, 0.7, 0.95, 0.85])
    Ec_t = torch.as_tensor(Ec, dtype=torch.float64)
    Zo_t = torch.as_tensor(Zo, dtype=torch.float64)
    sZ_t = torch.as_tensor(sZ, dtype=torch.float64)
    sa_t = torch.as_tensor(sa, dtype=torch.float64)
    cen = torch.ones(5, dtype=torch.bool)
    rep = integrated_s_report(Ec_t, sa_t, Zo_t, sZ_t, cen)
    assert math.isfinite(rep.s_post_all) and math.isfinite(rep.s_prior_all)
    E_t, abs_t, k = _torch_batch_moments(Zo, sZ, Ec, sa, centric=True)
    for i in range(5):
        E_o, abs_o = oracle_moments(
            float(Zo[i]), float(sZ[i]), float(Ec[i]), float(sa[i]), centric=True, k=k
        )
        assert abs(E_t[i] - E_o) / max(E_o, 1e-6) < 5e-3
        assert abs(abs_t[i] - abs_o) / max(abs_o, 1e-6) < 5e-3


def test_empty_free_returns_nan():
    Ec_t = torch.ones(10, dtype=torch.float64)
    Zo_t = torch.ones(10, dtype=torch.float64)
    sZ_t = torch.full((10,), 0.3, dtype=torch.float64)
    sa_t = torch.full((10,), 0.8, dtype=torch.float64)
    cen = torch.zeros(10, dtype=torch.bool)
    work = torch.ones(10, dtype=torch.bool)
    free = torch.zeros(10, dtype=torch.bool)
    with pytest.warns(UserWarning):
        rep = integrated_s_report(Ec_t, sa_t, Zo_t, sZ_t, cen, work_mask=work, free_mask=free)
    assert math.isnan(rep.s_post_free)
    assert math.isfinite(rep.s_post_work)


def test_ec_outliers_do_not_collapse_k():
    """A near-zero Wilson Sigma gives huge E_C; LS k then collapses and pins S_post at 1."""
    n_good, n_bad = 2000, 20
    Ec = torch.cat(
        [torch.full((n_good,), 1.0, dtype=torch.float64), torch.full((n_bad,), 1e4, dtype=torch.float64)]
    )
    sa = torch.full((n_good + n_bad,), 0.9, dtype=torch.float64)
    Zo = torch.cat(
        [torch.full((n_good,), 0.81, dtype=torch.float64), torch.full((n_bad,), 0.81, dtype=torch.float64)]
    )
    sZ = torch.full((n_good + n_bad,), 0.05, dtype=torch.float64)
    cen = torch.zeros(n_good + n_bad, dtype=torch.bool)

    with pytest.warns(UserWarning, match="E_C"):
        rep = integrated_s_report(Ec, sa, Zo, sZ, cen)
    assert rep.n_ec_outliers == n_bad
    # k must stay on the physical scale, not collapse toward 0
    assert 0.1 < rep.k_s_all < 3.0
    assert rep.s_post_all < 0.5

    # Disabling the guard reproduces the pathology (k ~ 0, R ~ 1)
    raw = integrated_s_report(Ec, sa, Zo, sZ, cen, ec_outlier_ratio=0.0)
    assert raw.k_s_all < 1e-3
    assert raw.s_post_all > 0.9


def test_nan_k_does_not_force_r_equals_one():
    """Legacy bug: non-finite k → absdev with k=0 → R≡1 (looks fitted, is not)."""
    Ec_t = torch.ones(20, dtype=torch.float64) * 1.2
    Zo_t = torch.ones(20, dtype=torch.float64)
    sZ_t = torch.full((20,), 0.3, dtype=torch.float64)
    sa_t = torch.full((20,), 0.9, dtype=torch.float64)
    cen = torch.zeros(20, dtype=torch.bool)
    cache = posterior_amplitude_cache(Ec_t, sa_t, Zo_t, sZ_t, cen, nu=30.0)
    ad = absdev_given_k(cache, float("nan"))
    assert torch.isnan(ad).all()
    assert math.isnan(aggregate_s(cache.E_mean, ad))
    # Empty work set → nan k → nan R (not 1.0)
    work = torch.zeros(20, dtype=torch.bool)
    free = torch.ones(20, dtype=torch.bool)
    rep = integrated_s_report(Ec_t, sa_t, Zo_t, sZ_t, cen, work_mask=work, free_mask=free)
    assert math.isnan(rep.k_s_work)
    assert math.isnan(rep.s_post_work)
    assert math.isfinite(rep.s_post_free)


def test_l2_identity_visible_plus_latent():
    """``<(E-kE_C)^2> = (<E>-kE_C)^2 + Var(E)`` — algebraic, must hold to precision."""
    rng = np.random.default_rng(5)
    n = 400
    Ec = torch.as_tensor(rng.uniform(0.3, 2.0, size=n), dtype=torch.float64)
    Zo = torch.as_tensor(rng.normal(1.0, 1.2, size=n), dtype=torch.float64)
    sZ = torch.as_tensor(rng.uniform(0.2, 0.6, size=n), dtype=torch.float64)
    sa = torch.as_tensor(rng.choice([0.6, 0.9, 0.99], size=n), dtype=torch.float64)
    cen = torch.zeros(n, dtype=torch.bool)

    cache = posterior_amplitude_cache(Ec, sa, Zo, sZ, cen)
    k = scale_k(cache.E_mean, cache.Ec)
    lhs = float(sqdev_given_k(cache, k).sum().item())
    v_vis = float(((cache.E_mean - k * cache.Ec) ** 2).sum().item())
    v_lat = float(cache.variance().sum().item())
    assert abs(lhs - (v_vis + v_lat)) / (v_vis + v_lat) < 1e-10

    # The report must expose exactly those sums
    rep = integrated_s_report(Ec, sa, Zo, sZ, cen)
    assert abs(rep.v_vis_all - v_vis) / v_vis < 1e-10
    assert abs(rep.v_lat_all - v_lat) / v_lat < 1e-10
    assert abs(rep.rho2_all - v_vis / (v_vis + v_lat)) < 1e-12


def test_rho2_decreases_with_noise():
    """The latent share grows as the data weaken and the posterior widens."""
    rng = np.random.default_rng(42)
    n, sa, sZ_base = 4000, 0.9, 0.3
    Ec0, Zo0, _, _, noise = _simulate_ensemble(n, sa, sa, sZ_base, 1.0, rng)
    E2 = Zo0 - sZ_base * noise
    rho2s = []
    for m in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        sZ = sZ_base * m
        rep = integrated_s_report(
            torch.as_tensor(Ec0, dtype=torch.float64),
            torch.full((n,), sa, dtype=torch.float64),
            torch.as_tensor(E2 + sZ * noise, dtype=torch.float64),
            torch.full((n,), sZ, dtype=torch.float64),
            torch.zeros(n, dtype=torch.bool),
        )
        rho2s.append(rep.rho2_all)
    for prev, nxt in zip(rho2s[:-1], rho2s[1:]):
        assert nxt < prev
    assert rho2s[0] > 0.9
    assert rho2s[-1] < 0.3


def test_thin_free_shells_are_blanked_silently(recwarn):
    """A sparse free set leaves too few per shell to report; blank it without noise."""
    rng = np.random.default_rng(9)
    n = 2000
    Ec = torch.as_tensor(rng.uniform(0.3, 2.0, size=n), dtype=torch.float64)
    Zo = torch.as_tensor(rng.normal(1.0, 1.0, size=n), dtype=torch.float64)
    sZ = torch.as_tensor(rng.uniform(0.2, 0.6, size=n), dtype=torch.float64)
    sa = torch.full((n,), 0.85, dtype=torch.float64)
    cen = torch.zeros(n, dtype=torch.bool)
    d = torch.linspace(30.0, 1.8, n)
    free = torch.zeros(n, dtype=torch.bool)
    free[::25] = True  # 80 free total → 20 per 500-reflection shell
    kw = dict(work_mask=~free, free_mask=free, d_spacings=d, bin_size=500)

    rep = integrated_s_report(Ec, sa, Zo, sZ, cen, min_free_per_shell=30, **kw)
    assert rep.n_free_bins.tolist() == [20, 20, 20, 20]  # counts stay honest
    assert np.isnan(rep.s_post_free_bins).all()
    assert np.isnan(rep.s_prior_free_bins).all()
    assert np.isnan(rep.rho2_free_bins).all()
    assert np.isfinite(rep.s_post_work_bins).all()
    assert np.isfinite(rep.rho2_work_bins).all()
    assert [str(w.message) for w in recwarn] == []

    # Threshold is the only thing suppressing them
    kept = integrated_s_report(Ec, sa, Zo, sZ, cen, min_free_per_shell=0, **kw)
    assert np.isfinite(kept.s_post_free_bins).all()
    assert np.isfinite(kept.rho2_free_bins).all()


def _score_report(Ec, sa, Zo, sZ, *, free=None, n=None, **kw):
    """``integrated_s_report`` with the score test on, from plain arrays."""
    n = int(n if n is not None else len(Ec))
    return integrated_s_report(
        torch.as_tensor(Ec, dtype=torch.float64),
        torch.as_tensor(sa, dtype=torch.float64),
        torch.as_tensor(Zo, dtype=torch.float64),
        torch.as_tensor(sZ, dtype=torch.float64),
        torch.zeros(n, dtype=torch.bool),
        work_mask=(~free if free is not None else None),
        free_mask=free,
        score_test=True,
        **kw,
    )


def test_xi_is_calibrated_on_the_no_fit_ensemble():
    """With sigma_A truthful and nothing fitted, xi is 1 on both sets and omega is 1."""
    rng = np.random.default_rng(42)
    n, sa, sZ = 4000, 0.9, 0.3
    Ec, Zo, sZ_arr, sa_arr, _ = _simulate_ensemble(n, sa, sa, sZ, 1.0, rng)
    free = torch.zeros(n, dtype=torch.bool)
    free[::10] = True
    rep = _score_report(Ec, sa_arr, Zo, sZ_arr, free=free, n=n)
    assert 0.9 < rep.xi_work < 1.1
    assert 0.9 < rep.xi_free < 1.1
    assert 0.9 < rep.omega < 1.1


def test_xi_detects_inflated_sigma_a_and_loses_power_with_noise():
    """An over-claimed sigma_A shows up as xi >> 1 while the data are informative."""
    rng = np.random.default_rng(42)
    n, sa_true, sa_claim, sZ_base = 4000, 0.7, 0.9, 0.3
    Ec0, Zo0, _, _, noise = _simulate_ensemble(n, sa_true, sa_true, sZ_base, 1.0, rng)
    E2 = Zo0 - sZ_base * noise
    xis = []
    for m in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        sZ = sZ_base * m
        rep = _score_report(
            Ec0, np.full(n, sa_claim), E2 + sZ * noise, np.full(n, sZ), n=n
        )
        xis.append(rep.xi_all)
    assert xis[0] > 1.5
    # xi peaks where the data are sharp enough to expose the error but not so sharp
    # that the reference variance shrinks with it; past the peak it decays back
    # toward the null as the observations stop constraining anything.
    peak = int(np.argmax(xis))
    assert peak <= 2
    for prev, nxt in zip(xis[peak:-1], xis[peak + 1 :]):
        assert nxt < prev
    assert xis[-1] < 1.2


def test_predictive_reference_variance_matches_a_monte_carlo_oracle():
    """The quadrature is the production path; Monte Carlo is only allowed as an oracle."""
    rng = np.random.default_rng(7)
    m = 60_000
    ratios = []
    quad_tot = mc_tot = 0.0
    for Ec0 in (0.4, 1.0, 2.0):
        for sA0 in (0.7, 0.95):
            for sZ0 in (0.1, 0.5, 2.0):
                a2 = 1.0 - sA0**2
                re = sA0 * Ec0 + math.sqrt(a2 / 2.0) * rng.normal(size=m)
                im = math.sqrt(a2 / 2.0) * rng.normal(size=m)
                Zo = torch.as_tensor(re * re + im * im + sZ0 * rng.normal(size=m))
                Ec = torch.full((m,), Ec0, dtype=torch.float64)
                sA = torch.full((m,), sA0, dtype=torch.float64)
                sZ = torch.full((m,), sZ0, dtype=torch.float64)
                cen = torch.zeros(m, dtype=torch.bool)
                cache = posterior_amplitude_cache(Ec, sA, Zo, sZ, cen)
                mc = float((delta_given_cache(cache) ** 2).mean().item())
                quad = float(
                    predictive_delta_variance(Ec[:1], sA[:1], sZ[:1], cen[:1], n_z=12)[0].item()
                )
                ratios.append(quad / mc)
                quad_tot += quad
                mc_tot += mc
    # Per reflection the Gaussian moment-match is only approximate: the true
    # predictive of Z_o is a skewed noncentral chi-squared plus noise.
    assert min(ratios) > 0.75
    assert max(ratios) < 1.30
    # Aggregated — which is how xi uses it — the errors cancel.
    assert 0.95 < quad_tot / mc_tot < 1.05


def test_fitting_p_directions_on_work_splits_xi_and_s_post():
    """Fit p = 0.2 N_work directions on the work set only; xi and S_post must split
    apart while their midpoint stays put."""
    rng = np.random.default_rng(11)
    n, sa, sZ0 = 2500, 0.9, 0.3
    Ec0, Zo, sZ_arr, sa_arr, _ = _simulate_ensemble(n, sa, sa, sZ0, 1.0, rng)
    free = torch.zeros(n, dtype=torch.bool)
    free[::5] = True  # 20% free → N_work = 2000
    work = ~free
    n_work = int(work.sum().item())
    p = int(round(0.2 * n_work))

    Ec_t = torch.as_tensor(Ec0, dtype=torch.float64)
    sA_t = torch.as_tensor(sa_arr, dtype=torch.float64)
    Zo_t = torch.as_tensor(Zo, dtype=torch.float64)
    sZ_t = torch.as_tensor(sZ_arr, dtype=torch.float64)
    cen = torch.zeros(n, dtype=torch.bool)
    pre = _score_report(Ec0, sa_arr, Zo, sZ_arr, free=free, n=n)

    # Gauss-Newton on the work-set NLL over a linear perturbation of E_C. Because
    # E_C,h depends on beta only through row h, the per-reflection score is
    # s_h * U[h,:], so the empirical-Fisher Hessian is U^T diag(s^2) U.
    U = torch.as_tensor(rng.normal(size=(n, p)) / math.sqrt(p))
    Uw = U[work]
    beta = torch.zeros(p, dtype=torch.float64)
    eye = torch.eye(p, dtype=torch.float64)
    for _ in range(3):
        Ec_b = (Ec_t + U @ beta).clamp_min(1e-6)
        Ec_v = Ec_b.detach().requires_grad_(True)
        nll = -log_likelihood_normal(Ec_v, sA_t, Zo_t, sZ_t, cen).sum()
        (s_all,) = torch.autograd.grad(nll, Ec_v)
        s = s_all[work]
        hess = Uw.T @ (s[:, None] ** 2 * Uw)
        # Damping keeps the step inside the linear regime the p/N algebra assumes
        hess = hess + 0.5 * float(hess.diagonal().mean()) * eye
        beta = beta - torch.linalg.solve(hess, Uw.T @ s)
    Ec_fit = (Ec_t + U @ beta).clamp_min(1e-6).detach()

    post = _score_report(Ec_fit.numpy(), sa_arr, Zo, sZ_arr, free=free, n=n)
    assert post.xi_work < 1.0 < post.xi_free
    assert post.omega > 1.3
    assert post.s_post_work < post.s_post_free
    mid_pre = 0.5 * (pre.s_post_work + pre.s_post_free)
    mid_post = 0.5 * (post.s_post_work + post.s_post_free)
    assert abs(mid_post - mid_pre) < 0.02


def test_ops_never_fuses_device_and_dtype_in_to():
    """MPS has no float64: a fused ``.to(dtype=float64, device="cpu")`` raises for
    small tensors but silently yields uninitialized memory past ~1k elements."""
    src = (ROOT / "src" / "phridge" / "contrib" / "intensity_ll" / "ops.py").read_text()
    fused = re.compile(r"\.to\([^)]*dtype=[^)]*device=[^)]*\)|\.to\([^)]*device=[^)]*dtype=[^)]*\)")
    offenders = [ln for ln in src.splitlines() if fused.search(ln) and not ln.lstrip().startswith("#")]
    assert offenders == [], f"use .cpu().double() instead: {offenders}"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
def test_mps_to_float64_cpu_roundtrip_is_exact():
    """Guards the conversion actually used by ml_i_maps at production sizes."""
    x = torch.full((42734,), 3.5, dtype=torch.float32, device="mps")
    y = x.detach().cpu().double()
    assert y.dtype is torch.float64
    assert bool((y == 3.5).all())

