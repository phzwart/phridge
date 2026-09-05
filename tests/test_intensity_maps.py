"""Map coefficients from the intensity likelihood posterior (phridge.contrib.intensity_ll.maps)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from pydantic import ValidationError

from phridge.contrib.intensity_ll import MAPS_OP_NAME, register
from phridge.contrib.intensity_ll.maps import IntensityMapOptions
from phridge.ops import get_implementation, get_op

torch = pytest.importorskip("torch")

from phridge.contrib.intensity_ll.maps import (  # noqa: E402
    intensity_map_coefficients,
    posterior_moments,
)
from phridge.contrib.intensity_ll.mli import log_likelihood_normal, log_likelihood_t  # noqa: E402
from phridge.contrib.intensity_ll.target import IntensityLogLikelihood  # noqa: E402
from phridge.sfcalc.targets import Observations  # noqa: E402


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


# ------------------------------------------------------------------ registration / options
def test_register_adds_maps_op():
    register()
    spec = get_op(MAPS_OP_NAME)
    assert spec.outputs["difference"] == "MillerArray"
    assert spec.inputs["maps"] == "json"
    assert get_implementation(MAPS_OP_NAME) is not None


def test_map_options_accept_reject():
    opts = IntensityMapOptions.model_validate({"newton_damping": 0.3, "include_free": False})
    assert opts.newton_damping == 0.3 and opts.include_free is False
    with pytest.raises(ValidationError):
        IntensityMapOptions.model_validate({"newton_damping": 0.0})
    with pytest.raises(ValidationError):
        IntensityMapOptions.model_validate({"bogus": 1})


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


# ------------------------------------------------------------------ op round trip (memory bridge)
def test_ml_i_maps_op_round_trip_memory_bridge():
    from phridge.client import Bridge
    from phridge.models import CrystalSymmetry, ObservationType
    from phridge.packing import PackedMiller

    register()
    fc, io, sig, obs = _synthetic(n=200, seed=5)
    n = len(fc)
    rng = np.random.default_rng(0)
    hkl = rng.integers(-8, 9, size=(n, 3))
    crystal = CrystalSymmetry(unit_cell=[20.0, 30.0, 40.0, 90.0, 90.0, 90.0], space_group_hall="P 1")
    f_obs = PackedMiller(crystal=crystal, hkl=hkl, data=io, sigmas=sig, observation_type=ObservationType.intensity)
    f_calc = PackedMiller(crystal=crystal, hkl=hkl, data=fc, observation_type=ObservationType.complex)
    bridge = Bridge(memory=True, timeout=5)
    out = bridge.call(
        MAPS_OP_NAME,
        f_calc=f_calc,
        f_obs=f_obs,
        target={"name": "ml_i", "nu": 6.0},
        alpha=np.full(n, 0.85),
        beta=np.full(n, 100.0),
        epsilon=np.ones(n),
        centric=np.zeros(n, bool),
        maps={"newton_damping": 0.2},
    )
    assert set(out) >= {"difference", "model", "gradient", "newton", "fom", "robust_weight", "stats", "d_loglik_d_nu"}
    diff = np.asarray(out["difference"].data)
    assert diff.shape == (n,) and np.iscomplexobj(diff)
    assert out["stats"]["nu"] == 6.0 and out["stats"]["n_ok"] == n
    # identical to the in-process computation
    tgt = IntensityLogLikelihood(nu=6.0)
    ms = intensity_map_coefficients(tgt, torch.as_tensor(fc), obs, IntensityMapOptions(newton_damping=0.2))
    np.testing.assert_allclose(diff, ms.difference, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(out["robust_weight"]), ms.robust_weight, rtol=1e-12)
    with pytest.raises(Exception, match="ml_i"):
        bridge.call(MAPS_OP_NAME, f_calc=f_calc, f_obs=f_obs, target={"name": "ml_f"})


# ------------------------------------------------------------------ cctbx client path
def test_remote_intensity_maps_cctbx_round_trip():
    cctbx = pytest.importorskip("cctbx")  # noqa: F841
    from cctbx import sgtbx
    from cctbx.array_family import flex
    from cctbx.development import random_structure

    from phridge.client import Bridge
    from phridge.contrib.intensity_ll.client import RemoteIntensityMaps

    flex.set_random_seed(0)
    xs = random_structure.xray_structure(
        space_group_info=sgtbx.space_group_info("P212121"),
        elements=["C", "N", "O", "S"] * 4, volume_per_atom=50, random_u_iso=True,
    )
    fc = xs.structure_factors(d_min=2.5, algorithm="direct").f_calc()
    xs2 = xs.deep_copy_scatterers()
    xs2.shake_sites_in_place(rms_difference=0.3)
    fc2 = xs2.structure_factors(d_min=2.5, algorithm="direct").f_calc()
    i_true = fc.amplitudes().data() ** 2
    n = i_true.size()
    rng = np.random.default_rng(0)
    sig = flex.double(0.05 * np.asarray(i_true) + 0.5)
    i_obs = fc.customized_copy(data=i_true + sig * flex.double(rng.normal(size=n)), sigmas=sig)
    i_obs.set_observation_type_xray_intensity()
    beta = flex.double(np.full(n, float(flex.mean(i_true))))
    alpha = flex.double(np.full(n, 0.9))
    r_free = flex.bool(rng.random(n) < 0.1)

    maps = RemoteIntensityMaps(
        Bridge(memory=True, timeout=10), i_obs, {"name": "ml_i", "nu": 8.0},
        alpha=alpha, beta=beta, r_free_flags=r_free, maps={"include_free": False},
    )
    res = maps(fc2)
    assert res.difference.is_complex_array() and res.difference.size() == n
    assert res.difference.indices().all_eq(i_obs.indices())
    assert res.model.size() == n and res.newton.size() == n and res.gradient.size() == n
    assert res.fom.size() == n and 0.0 <= flex.min(res.fom) and flex.max(res.fom) <= 1.0 + 1e-12
    assert res.robust_weight.size() == n and flex.min(res.robust_weight) > 0
    assert res.d_loglik_d_nu_total is not None
    free = np.asarray(r_free)
    diff = np.asarray(res.difference.data())
    assert np.all(diff[free] == 0) and np.all(diff[~free] != 0)
    assert res.stats["nu"] == 8.0
    # coefficients are D|Fc| e^{i phi_c}-consistent: phase equals the model phase where nonzero
    ph = np.asarray(fc2.data()) / np.abs(np.asarray(fc2.data()))
    assert np.allclose(np.imag(diff[~free] * np.conj(ph[~free])), 0.0, atol=1e-9 * np.abs(diff).max())
