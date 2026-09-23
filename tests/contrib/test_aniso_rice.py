"""Laue-class σ_A / β modulation: identity, planted direction, fallbacks."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from phridge.contrib.intensity_ll import aniso_rice as ar
from phridge.models import CrystalSymmetry, ObservationType
from phridge.packing import PackedMiller


def test_laue_from_number_and_hall():
    assert ar.laue_from_number(1).n_free == 5
    assert ar.laue_from_number(4).n_free == 3
    assert ar.laue_from_number(19).n_free == 2
    assert ar.laue_from_number(96).n_free == 1
    assert ar.laue_from_number(221).n_free == 0
    assert ar.laue_from_hall("P 1").name == "-1"
    assert ar.laue_from_hall("P 21 21 21").name == "mmm"
    crystal = CrystalSymmetry(
        unit_cell=[50.0, 60.0, 70.0, 90.0, 90.0, 90.0],
        space_group_hall="P 21 21 21",
        space_group_number=19,
    )
    assert ar.laue_from_crystal(crystal).name == "mmm"


def test_a_zero_is_the_identity_modulation():
    laue = ar.LAUE_TRICLINIC
    a = ar.a_from_params(np.zeros(laue.n_free), laue)
    np.testing.assert_allclose(a, 0.0)
    rng = np.random.default_rng(0)
    s = rng.normal(size=(200, 3))
    s_hat = ar.unit_directions(s)
    q = ar.quadratic_form(s_hat, np.eye(3) + a)
    np.testing.assert_allclose(q, 1.0, atol=1e-12)


def test_tetragonal_is_traceless_and_xx_equals_yy():
    a = ar.a_from_params(np.array([0.4]), ar.LAUE_TETRAGONAL)
    assert abs(np.trace(a)) < 1e-12
    assert a[0, 0] == pytest.approx(a[1, 1])
    assert a[2, 2] == pytest.approx(-2.0 * a[0, 0])


def test_cubic_and_unidentifiable_are_not_fitted():
    s_line = np.stack([np.linspace(0.1, 1.0, 20), np.zeros(20), np.zeros(20)], axis=1)
    ok, why = ar.modulation_wanted(
        enabled=True,
        beta_free=True,
        sigma_a_free=True,
        bins_mode=True,
        laue=ar.LAUE_TRICLINIC,
        s_cart=s_line,
    )
    assert ok is False and why and "identifiability" in why
    ok, why = ar.modulation_wanted(
        enabled=True,
        beta_free=True,
        sigma_a_free=True,
        bins_mode=True,
        laue=ar.LAUE_CUBIC,
        s_cart=np.random.default_rng(0).normal(size=(80, 3)),
    )
    assert ok is False and why is None
    ok, why = ar.modulation_wanted(
        enabled=True,
        beta_free=False,
        sigma_a_free=False,
        bins_mode=True,
        laue=ar.LAUE_TRICLINIC,
        s_cart=np.random.default_rng(1).normal(size=(80, 3)),
    )
    assert ok is False and why is None


def _directional_problem(seed=0, n=4000, k_z=0.35):
    rng = np.random.default_rng(seed)
    hkl = rng.integers(-12, 13, size=(n, 3)).astype(np.int32)
    hkl[np.all(hkl == 0, axis=1)] = (1, 0, 0)
    crystal = CrystalSymmetry(
        unit_cell=[60.0, 60.0, 60.0, 90.0, 90.0, 90.0],
        space_group_hall="P 1",
        space_group_number=1,
    )
    from phridge.contrib.intensity_ll.wilson import reciprocal_cartesian

    s_cart = reciprocal_cartesian(crystal.unit_cell, hkl)
    s_hat = ar.unit_directions(s_cart)
    a = np.diag([-0.5 * k_z, -0.5 * k_z, k_z])
    quad = ar.quadratic_form(s_hat, np.eye(3) + a)
    ss = np.sum(s_cart**2, axis=1)
    sa = np.clip(0.82 * np.sqrt(quad), 0.15, 0.98)
    sw = 150.0 * np.exp(-8.0 * ss)
    fc = np.sqrt(sw) * (0.7 + 0.5 * rng.random(n))
    i_true = (sa * fc) ** 2
    sig = 0.08 * sw + 0.5
    io = i_true + rng.normal(0.0, sig)
    f_obs = PackedMiller(
        crystal=crystal, hkl=hkl, data=io, sigmas=sig, observation_type=ObservationType.intensity
    )
    f_calc = PackedMiller(
        crystal=crystal, hkl=hkl, data=fc.astype(np.complex128), observation_type=ObservationType.complex
    )
    tune = rng.random(n) >= 0.15
    return dict(f_calc=f_calc, f_obs=f_obs, ss=ss, tune=tune, n=n)


def _fit(p, **kw):
    from phridge.contrib.intensity_ll.ops import ml_i_nuisance_fit

    common = dict(
        f_calc=p["f_calc"],
        f_obs=p["f_obs"],
        tune_mask=p["tune"],
        s_sq=p["ss"],
        epsilon=np.ones(p["n"]),
        centric=np.zeros(p["n"], dtype=bool),
        fit_nu=False,
        fit_scale=False,
        sigma_a_mode="bins",
        n_sigma_a_bins=8,
        fit_sigma_wilson=True,
        wilson_model="isotropic",
        beta_mode="free",
        sigma_a_shape="free",
    )
    common.update(kw)
    return ml_i_nuisance_fit(**common)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch required")
def test_a_zero_reproduces_the_scalar_shell_nll():
    p = _directional_problem(seed=1, n=2500, k_z=0.0)
    off = _fit(p, sigma_a_tensor=False)
    # A huge sphericity prior holds M at I; the shells can still move.
    on = _fit(p, sigma_a_tensor=True, sphericity=1e6)
    assert on["sigma_a_params"]["sigma_a_tensor"] is True
    ev = on["sigma_a_params"]["M_A"]["eigenvalues"]
    assert max(ev) - min(ev) < 0.05
    # Same shells, same M ≈ I → the per-reflection σ_A cannot wander far.
    rms = float(np.sqrt(np.mean((np.asarray(on["sigma_a"]) - np.asarray(off["sigma_a"])) ** 2)))
    assert rms < 0.05


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch required")
def test_recovers_a_planted_z_stretch():
    p = _directional_problem(seed=2, n=5000, k_z=0.40)
    out = _fit(p, sigma_a_tensor=True, sphericity=0.1)
    params = out["sigma_a_params"]
    assert params["sigma_a_tensor"] is True
    assert params["n_tensor_params"] == 10
    ev = np.asarray(params["M_A"]["eigenvalues"], dtype=np.float64)
    # Planted M eigenvalues are 0.8, 0.8, 1.4 — the largest should be along z
    # (direction cosines vs a*, b*, c*; c* is the third axis).
    assert ev[-1] > ev[0] + 0.05
    cos = np.asarray(params["M_A"]["direction_cosines"], dtype=np.float64)
    # Row of the largest eigenvalue: |cos| against a*, b*, c*.
    assert cos[-1, 2] > 0.6


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch required")
def test_cubic_records_no_tensor_params():
    p = _directional_problem(seed=3, n=2000)
    p["f_obs"].meta.crystal = CrystalSymmetry(
        unit_cell=[60.0, 60.0, 60.0, 90.0, 90.0, 90.0],
        space_group_hall="P 2 3",
        space_group_number=195,
    )
    p["f_calc"].meta.crystal = p["f_obs"].meta.crystal
    out = _fit(p, sigma_a_tensor=True)
    assert out["sigma_a_params"]["n_tensor_params"] == 0
    assert out["sigma_a_params"]["laue"] == "m-3m"
    assert "sigma_a_tensor_fallback" not in out["sigma_a_params"]


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch required")
def test_constrained_monotone_is_bit_identical_with_or_without_the_flag():
    p = _directional_problem(seed=4, n=2000)
    a = _fit(p, sigma_a_tensor=True, beta_mode="constrained", sigma_a_shape="monotone")
    b = _fit(p, sigma_a_tensor=False, beta_mode="constrained", sigma_a_shape="monotone")
    np.testing.assert_allclose(a["sigma_a"], b["sigma_a"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(a["beta"], b["beta"], rtol=1e-10, atol=1e-12)
    assert a["sigma_a_params"]["n_tensor_params"] == 0
    assert b["sigma_a_params"]["n_tensor_params"] == 0
    assert "sigma_a_tensor_fallback" not in a["sigma_a_params"]
