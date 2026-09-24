"""v1 spatial σ_A: inverse-mask mix, gauge, and nuisance-fit direction."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from phridge.contrib.intensity_ll import local_sigma_a as LSA
from phridge.models import CrystalSymmetry, ObservationType
from phridge.packing import PackedMiller

requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch required for ml_i_nuisance_fit"
)


def test_u_zero_is_the_identity():
    rng = np.random.default_rng(0)
    fa = rng.normal(size=40) + 1j * rng.normal(size=40)
    fb = rng.normal(size=40) + 1j * rng.normal(size=40)
    w_mol, w_sol = LSA.channel_weights(0.0, 0.5)
    assert w_mol == 1.0 and w_sol == 1.0
    shell = np.repeat(np.arange(4), 10)
    f_eff, n_k = LSA.mix_f_eff(fa, fb, w_mol, w_sol, shell, "shell")
    np.testing.assert_array_equal(f_eff, fa + fb)
    np.testing.assert_array_equal(n_k, np.ones(4))


def test_envelope_coefficients_are_minus_f_mask():
    hkl = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 2, 2]], dtype=np.int32)
    f_mask = np.array([10 + 0j, 1 + 2j, -3 + 0.5j, 0.2 - 0.1j], dtype=np.complex128)
    # s_sq = 1/d^2; d=20, 15, 4 Å
    s_sq = np.array([0.0, 1.0 / 20.0**2, 1.0 / 15.0**2, 1.0 / 4.0**2])
    h, c, keep = LSA.envelope_coefficients(hkl, f_mask, s_sq, d_min=10.0, blur_b=0.0)
    assert keep[0] is np.False_ or keep[0] == False
    assert not keep[3]  # 4 Å is inside the cutoff
    assert keep[1] and keep[2]
    np.testing.assert_allclose(c, -f_mask[keep])
    assert np.all(np.any(h != 0, axis=1))
    h0, c0, _ = LSA.phi_coefficients(hkl, f_mask, s_sq, u=0.0, d_min=10.0)
    np.testing.assert_allclose(c0, 0.0)


def test_per_shell_rms_gauge():
    rng = np.random.default_rng(1)
    n, n_shells = 200, 4
    fa = rng.normal(size=n) + 1j * rng.normal(size=n)
    fb = 0.4 * (rng.normal(size=n) + 1j * rng.normal(size=n))
    shell = np.repeat(np.arange(n_shells), n // n_shells)
    w_mol, w_sol = LSA.channel_weights(0.8, 0.45)
    f_model = fa + fb
    f_eff, n_k = LSA.mix_f_eff(fa, fb, w_mol, w_sol, shell, "shell")
    for k in range(n_shells):
        sel = shell == k
        rms_e = float(np.sqrt(np.mean(np.abs(f_eff[sel]) ** 2)))
        rms_m = float(np.sqrt(np.mean(np.abs(f_model[sel]) ** 2)))
        assert rms_e == pytest.approx(rms_m, abs=1e-12)
    assert n_k.shape == (n_shells,)


def test_cli_scripts_accept_spatial_sigma_a_flag():
    root = Path(__file__).resolve().parents[2]
    driver = (root / "scripts" / "phenix_refine_mli.py").read_text()
    wrapper = (root / "scripts" / "run_phenix_intensity.sh").read_text()
    assert "--spatial-sigmaa" in driver.lower()
    assert "PHRIDGE_SPATIAL_SIGMA_A" in driver
    assert "--spatial-sigmaA" in wrapper
    assert "PHRIDGE_SPATIAL_SIGMA_A" in wrapper


def test_pydantic_accept_reject():
    LSA.LocalSigmaAOptions()
    LSA.LocalSigmaAOptions(enabled=True, d_min=20.0, lambda_u=0.5)
    with pytest.raises(ValidationError):
        LSA.LocalSigmaAOptions(residual=True)
    with pytest.raises(ValidationError):
        LSA.LocalSigmaAOptions(d_min=0.0)
    with pytest.raises(ValidationError):
        LSA.LocalSigmaAOptions(not_a_field=1)  # type: ignore[call-arg]


def test_v1_jacobian_atom_channel_drops_with_w_mol():
    rng = np.random.default_rng(2)
    fa = rng.normal(size=30) + 1j * rng.normal(size=30)
    fb = 0.1 * (rng.normal(size=30) + 1j * rng.normal(size=30))
    full = LSA.apply_frozen_mix(fa, fb, 1.0, 1.0, 1.0)
    half = LSA.apply_frozen_mix(fa, fb, 0.4, 1.0, 1.0)
    # Holding F_bulk and the gauge, shrinking w_mol must shrink the atom term.
    np.testing.assert_allclose(full - fb, fa)
    np.testing.assert_allclose(half - fb, 0.4 * fa)
    assert float(np.mean(np.abs(half - fb))) < float(np.mean(np.abs(full - fb)))


def _split_problem(*, d_mol=1.0, d_sol=0.35, seed=0, n=4000):
    """Intensities from D_mol F_atoms + D_sol F_bulk, Rice-drawn."""
    rng = np.random.default_rng(seed)
    d = 1.0 / np.sqrt(rng.uniform(1.0 / 25.0**2, 1.0 / 2.5**2, n))
    ss = 1.0 / d**2
    amp_a = 40.0 * np.exp(-8.0 * ss) * np.sqrt(rng.exponential(1.0, n))
    fa = amp_a * np.exp(2j * np.pi * rng.random(n))
    amp_b = 70.0 * np.exp(-40.0 * ss) * np.sqrt(rng.exponential(1.0, n))
    fb = -amp_b * np.exp(1j * (np.angle(fa) + rng.normal(0.0, 0.6, n)))
    f_true = d_mol * fa + d_sol * fb
    f_model = fa + fb
    sigma_a = 0.88
    sw = np.empty(n)
    order = np.argsort(ss)
    for chunk in np.array_split(order, 24):
        sw[chunk] = np.mean(np.abs(f_true[chunk]) ** 2)
    noise = np.sqrt(0.5 * (1.0 - sigma_a**2) * sw) * (
        rng.normal(size=n) + 1j * rng.normal(size=n)
    )
    i_true = np.abs(sigma_a * f_true + noise) ** 2
    sig = 0.04 * i_true + 0.02 * sw
    i_obs = i_true + rng.normal(0.0, sig)
    crystal = CrystalSymmetry(unit_cell=[80.0, 80.0, 80.0, 90.0, 90.0, 90.0], space_group_hall="P 1")
    hkl = np.stack([np.arange(n), np.zeros(n), np.zeros(n)], axis=1).astype(np.int32)
    hkl[0] = [1, 0, 0]
    f_obs = PackedMiller(
        crystal=crystal, hkl=hkl, data=i_obs, sigmas=sig, observation_type=ObservationType.intensity
    )
    f_calc = PackedMiller(
        crystal=crystal, hkl=hkl, data=f_model, observation_type=ObservationType.complex
    )
    tune = rng.random(n) >= 0.1
    return dict(
        f_calc=f_calc,
        f_obs=f_obs,
        fa=fa,
        fb=fb,
        ss=ss,
        tune=tune,
        n=n,
        sw=sw,
    )


def _fit(p, **kw):
    from phridge.contrib.intensity_ll.ops import ml_i_nuisance_fit

    opts = dict(
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
        sigma_a_tensor=False,
    )
    opts.update(kw)
    return ml_i_nuisance_fit(p["f_calc"], p["f_obs"], **opts)


@requires_torch
def test_missing_split_falls_back():
    p = _split_problem(seed=3, n=1500)
    out = _fit(
        p,
        local_sigma_a={"enabled": True, "lambda_u": 1.0},
    )
    loc = out["sigma_a_params"]["local_sigma_a"]
    assert loc["enabled"] is False
    assert "f_atoms" in loc["fallback"]
    assert np.all(np.isfinite(out["sigma_a"]))


@requires_torch
def test_enabled_plus_fit_scale_plus_free_beta_raises():
    p = _split_problem(seed=4, n=800)
    with pytest.raises(ValueError, match="not identifiable"):
        _fit(
            p,
            fit_scale=True,
            beta_mode="free",
            f_atoms=p["fa"],
            f_bulk=p["fb"],
            local_sigma_a={"enabled": True},
        )


@requires_torch
def test_recovers_planted_channel_direction():
    p_hi = _split_problem(d_mol=1.0, d_sol=0.30, seed=5, n=5000)
    high_mol = _fit(
        p_hi,
        f_atoms=p_hi["fa"],
        f_bulk=p_hi["fb"],
        local_sigma_a={"enabled": True, "lambda_u": 0.05, "e_bar": 0.5},
    )
    p_lo = _split_problem(d_mol=0.30, d_sol=1.0, seed=6, n=5000)
    low_mol = _fit(
        p_lo,
        f_atoms=p_lo["fa"],
        f_bulk=p_lo["fb"],
        local_sigma_a={"enabled": True, "lambda_u": 0.05, "e_bar": 0.5},
    )
    u_hi = float(high_mol["sigma_a_params"]["local_sigma_a"]["u"])
    u_lo = float(low_mol["sigma_a_params"]["local_sigma_a"]["u"])
    assert high_mol["sigma_a_params"]["local_sigma_a"]["enabled"] is True
    assert u_hi > 0.0
    assert u_lo < 0.0
    sa = np.asarray(high_mol["sigma_a"])
    assert float(np.median(sa[sa > 0])) > 0.5
