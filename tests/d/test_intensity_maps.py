"""Map coefficients from the intensity likelihood posterior (phridge.d.maps)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from pydantic import ValidationError

from phridge.d.maps import IntensityMapOptions

torch = pytest.importorskip("torch")

from phridge.d.maps import (  # noqa: E402
    intensity_map_coefficients,
    posterior_moments,
)
from phridge.d.rules.mli_quad import log_likelihood_normal, log_likelihood_t  # noqa: E402
from phridge.d.rules.mli_quad.target import IntensityLogLikelihood  # noqa: E402
from phridge.d.rules.amplitude import Observations  # noqa: E402


def _grid(n=1500, seed=1):
    rng = np.random.default_rng(seed)
    sA = rng.choice([0.3, 0.6, 0.8, 0.9, 0.95, 0.99], n)
    Et = np.sqrt(-np.log(rng.random(n)))
    Ec = np.clip(np.abs(rng.normal(sA * Et, np.sqrt((1 - sA**2) / 2))), 1e-3, None)
    sZ = np.exp(rng.uniform(math.log(0.02), math.log(3), n))
    Zo = Et**2 + sZ * rng.normal(size=n)
    cen = rng.random(n) < 0.3
    T = lambda v: torch.as_tensor(v, dtype=torch.float64)  # noqa: E731
    return T(Ec), T(sA), T(Zo), T(sZ), torch.as_tensor(cen)


def test_map_options_accept_reject():
    opts = IntensityMapOptions.model_validate({"newton_damping": 0.3, "include_free": False})
    assert opts.newton_damping == 0.3 and opts.include_free is False
    with pytest.raises(ValidationError):
        IntensityMapOptions.model_validate({"newton_damping": 0.0})
    with pytest.raises(ValidationError):
        IntensityMapOptions.model_validate({"bogus": 1})


def test_evaluate_lifts_cpu_f_to_mps_when_worker_is_mps():
    """C++ stamp leaves F on CPU; ml_i evaluate must still run quadrature on Metal."""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    from phridge.sfcalc.ops import set_device

    n = 64
    rng = np.random.default_rng(0)
    io = rng.normal(1.0, 0.5, n)
    sig = np.full(n, 0.2)
    fc = torch.randn(n, dtype=torch.complex64)  # CPU, as after stamp_cpp
    obs = Observations.from_numpy(
        device="cpu",
        dtype=torch.float32,
        data=io,
        sigmas=sig,
        epsilon=np.ones(n),
        centric=np.zeros(n, dtype=bool),
        alpha=np.full(n, 0.8),
        beta=np.full(n, 10.0),
    )
    set_device("mps")
    try:
        ev = IntensityLogLikelihood().evaluate(fc, obs, compute_curvature=True)
    finally:
        set_device("cpu")
    assert np.isfinite(ev.value)
    assert np.isfinite(ev.d_target_d_f_calc).all()
    assert np.isfinite(ev.curv_radial).all()


# ------------------------------------------------------------------ posterior identities
@pytest.mark.parametrize("nu", [None, 4.0])
def test_posterior_score_matches_autograd(nu):
    """Fisher identity on frozen nodes: closed-form score == autograd of the quadrature log L."""
    Ec, sA, Zo, sZ, cen = _grid()
    Ec_ = Ec.clone().requires_grad_(True)
    Zo_ = Zo.clone().requires_grad_(True)
    ll = log_likelihood_normal(Ec_, sA, Zo_, sZ, cen) if nu is None else log_likelihood_t(Ec_, sA, Zo_, sZ, cen, nu)
    gE, gZ = torch.autograd.grad(ll.sum(), (Ec_, Zo_))
    pm = posterior_moments(Ec, sA, Zo, sZ, cen, nu)
    assert torch.allclose(pm.log_lik, ll.detach(), atol=1e-12)
    assert torch.allclose(pm.score, gE, atol=1e-10, rtol=1e-9)
    assert float(pm.fom.min()) >= 0.0 and float(pm.fom.max()) <= 1.0 + 1e-12
    if nu is None:
        # <I>_post = Z_o + sigma_Z^2 d log L / d Z_o for normal noise
        assert torch.allclose(pm.I_mean, Zo + sZ**2 * gZ, atol=1e-10, rtol=1e-9)
        assert torch.allclose(pm.lambda_mean, torch.ones_like(Ec))
    else:
        assert float(pm.lambda_mean.min()) > 0.0
        assert pm.d_loglik_d_nu is not None


def test_d_loglik_d_nu_matches_finite_difference_in_aggregate():
    Ec, sA, Zo, sZ, cen = _grid(n=800, seed=2)
    nu, h = 5.0, 1e-3
    pm = posterior_moments(Ec, sA, Zo, sZ, cen, nu)
    fd = (log_likelihood_t(Ec, sA, Zo, sZ, cen, nu + h) - log_likelihood_t(Ec, sA, Zo, sZ, cen, nu - h)) / (2 * h)
    # per-reflection FD is noisy across quadrature-branch switches; the sum is what a nu step uses
    assert abs(float(pm.d_loglik_d_nu.sum() - fd.sum())) < 1e-3 * max(1.0, abs(float(fd.sum())))
    assert float((pm.d_loglik_d_nu - fd).abs().median()) < 1e-5


def test_flat_top_centric_window_regression():
    """Centric, weak, mode at the origin with tiny curvature: window must not stretch to E ~ 60."""
    t = lambda v: torch.tensor([v], dtype=torch.float64)  # noqa: E731
    Ec, sA, Zo = 0.77, 0.60, 0.127
    y = np.linspace(-14, 3, 400001)
    E = np.exp(y)
    a = 1 - sA**2
    k = sA * Ec / a
    for sZ in (0.4936, 0.4924, 0.5552):
        lg = (0.5 * np.log(2 / (np.pi * a)) - (E**2 + sA**2 * Ec**2) / (2 * a) + np.abs(k * E)
              + np.log1p(np.exp(-2 * np.abs(k * E))) - np.log(2) - (Zo - E**2) ** 2 / (2 * sZ**2)
              - 0.5 * np.log(2 * np.pi * sZ**2) + y)
        m = lg.max()
        ref = m + np.log(np.trapezoid(np.exp(lg - m), y))
        got = float(log_likelihood_normal(t(Ec), t(sA), t(Zo), t(sZ), torch.tensor([True])))
        assert abs(got - ref) < 5e-5


# ------------------------------------------------------------------ map coefficients
def _synthetic(n=2000, seed=3, sA=0.85, Sigma=100.0):
    rng = np.random.default_rng(seed)
    ft = np.sqrt(Sigma / 2) * (rng.normal(size=n) + 1j * rng.normal(size=n))
    fc = sA * ft + np.sqrt(Sigma * (1 - sA**2) / 2) * (rng.normal(size=n) + 1j * rng.normal(size=n))
    sig = np.exp(rng.uniform(np.log(0.5), np.log(150), n))
    io = np.abs(ft) ** 2 + sig * rng.normal(size=n)
    obs = Observations.from_numpy(
        data=io, sigmas=sig, epsilon=np.ones(n), centric=np.zeros(n, bool),
        alpha=np.full(n, sA), beta=np.full(n, Sigma),
    )
    return fc, io, sig, obs


@pytest.mark.parametrize("nu", [None, 5.0])
def test_gradient_map_is_target_gradient(nu):
    fc, io, sig, obs = _synthetic()
    tgt = IntensityLogLikelihood(nu=nu) if nu else IntensityLogLikelihood()
    fct = torch.as_tensor(fc)
    ev = tgt.evaluate(fct, obs, compute_curvature=True)
    ms = intensity_map_coefficients(tgt, fct, obs)
    n = len(fc)
    # target value is the mean NLL: gradient map = -n * d_target_d_f_calc, curvature = n * curv_radial
    assert np.max(np.abs(ms.gradient + n * ev.d_target_d_f_calc)) < 1e-10 * np.max(np.abs(ms.gradient))
    assert np.max(np.abs(ms.curvature - n * ev.curv_radial)) < 1e-10 * np.max(np.abs(ms.curvature))
    for arr in (ms.difference, ms.model, ms.newton, ms.fom, ms.robust_weight, ms.f_post):
        assert np.isfinite(arr).all()
    # coefficients carry the model phase
    ph = fc / np.abs(fc)
    assert np.allclose(np.imag(ms.difference * np.conj(ph)), 0.0, atol=1e-9 * np.abs(ms.difference).max())
    if nu is None:
        assert np.allclose(ms.robust_weight, 1.0)
        assert ms.d_loglik_d_nu is None
    else:
        assert ms.robust_weight.min() < 0.9 and ms.robust_weight.max() > 1.0
        assert ms.d_loglik_d_nu is not None


def test_difference_map_reduces_to_mfo_dfc_for_strong_data_and_vanishes_for_noise():
    from scipy.special import i0e, i1e

    fc, io, sig, obs = _synthetic()
    sA, Sigma = 0.85, 100.0
    tgt = IntensityLogLikelihood()
    ms = intensity_map_coefficients(tgt, torch.as_tensor(fc), obs)
    Fo = np.sqrt(np.clip(io, 0, None))
    x = 2 * sA * Fo * np.abs(fc) / ((1 - sA**2) * Sigma)
    m = i1e(x) / i0e(x)
    mfodfc = (m * Fo - sA * np.abs(fc)) * fc / np.abs(fc)
    strong = io / sig > 30
    noisy = sig > 100
    rel = np.abs(ms.difference[strong] - mfodfc[strong]) / np.abs(mfodfc[strong])
    assert np.median(rel) < 5e-3
    ratio_strong = np.median(np.abs(ms.difference[strong]) / np.abs(mfodfc[strong]))
    ratio_noisy = np.median(np.abs(ms.difference[noisy]) / np.abs(mfodfc[noisy]))
    assert abs(ratio_strong - 1.0) < 0.02
    assert ratio_noisy < 0.4
    # model map: 2<Fm> - D|Fc| == difference + <Fm>
    Em_F = ms.difference + sA * np.abs(fc) * fc / np.abs(fc)
    assert np.allclose(ms.model, ms.difference + Em_F, atol=1e-9 * np.abs(ms.model).max())


def test_include_free_masks_coefficients():
    fc, io, sig, obs = _synthetic(n=300)
    obs.r_free = torch.as_tensor(np.arange(300) % 10 == 0)
    tgt = IntensityLogLikelihood()
    ms = intensity_map_coefficients(tgt, torch.as_tensor(fc), obs, IntensityMapOptions(include_free=False))
    free = obs.r_free.numpy()
    assert np.all(ms.difference[free] == 0) and np.all(ms.difference[~free] != 0)
