"""σ_A nuisance fit should vary with resolution (not stay flat)."""

from __future__ import annotations

import numpy as np
import pytest

from phridge.models import CrystalSymmetry, ObservationType
from phridge.packing import PackedMiller


def _packed_arrays(n: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    hkl = np.zeros((n, 3), dtype=np.int32)
    hkl[:, 0] = np.linspace(1, 40, n).astype(np.int32)
    crystal = CrystalSymmetry(unit_cell=[50.0, 50.0, 50.0, 90.0, 90.0, 90.0], space_group_hall="P 1")
    s2 = (hkl[:, 0].astype(np.float64) / 50.0) ** 2

    sa_true = np.clip(0.95 * np.exp(-8.0 * s2), 0.05, 0.99)
    sw = 200.0 * np.exp(-10.0 * s2)
    fc = np.sqrt(sw) * (0.8 + 0.4 * rng.random(n))
    i_true = (sa_true * fc) ** 2
    sig = 0.15 * sw + 1.0
    io = i_true + rng.normal(0.0, sig)

    f_obs = PackedMiller(
        crystal=crystal,
        hkl=hkl,
        data=io,
        sigmas=sig,
        observation_type=ObservationType.intensity,
    )
    f_calc = PackedMiller(
        crystal=crystal,
        hkl=hkl,
        data=fc.astype(np.complex128),
        observation_type=ObservationType.complex,
    )
    tune = np.ones(n, dtype=bool)
    return f_calc, f_obs, tune, s2, sa_true


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None,
    reason="torch required for ml_i_nuisance_fit",
)
def test_nuisance_fit_binned_sigma_a_not_flat():
    from phridge.contrib.intensity_ll.ops import ml_i_nuisance_fit

    f_calc, f_obs, tune, s2, _ = _packed_arrays()
    out = ml_i_nuisance_fit(
        f_calc,
        f_obs,
        tune_mask=tune,
        s_sq=s2,
        epsilon=np.ones(len(s2)),
        centric=np.zeros(len(s2), dtype=bool),
        fit_nu=False,
        fit_scale=False,
        sigma_a_mode="bins",
        n_sigma_a_bins=10,
        fit_sigma_wilson=True,
    )
    sa = np.asarray(out["sigma_a"], dtype=np.float64)
    params = out["sigma_a_params"]
    assert params["mode"] == "bins"
    assert params["n_bins"] >= 6
    order = np.argsort(s2)
    sa_sorted = sa[order]
    assert sa_sorted[:100].mean() > sa_sorted[-100:].mean() + 0.05
    bin_sa = params["bin_sigma_a"]
    assert bin_sa[0] > bin_sa[-1]
    # Must not collapse to the σ_A → 1 ceiling
    assert sa.max() < 0.995


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None,
    reason="torch required for ml_i_nuisance_fit",
)
def test_nuisance_fit_binned_nu_with_tv():
    from phridge.contrib.intensity_ll.ops import ml_i_nuisance_fit

    f_calc, f_obs, tune, s2, _ = _packed_arrays(n=3000, seed=3)
    common = dict(
        f_calc=f_calc,
        f_obs=f_obs,
        tune_mask=tune,
        s_sq=s2,
        epsilon=np.ones(len(s2)),
        centric=np.zeros(len(s2), dtype=bool),
        fit_nu=True,
        nu_mode="bins",
        fit_scale=False,
        sigma_a_mode="bins",
        n_sigma_a_bins=10,
        fit_sigma_wilson=True,
        nu_bounds=(3.0, 30.0),
    )
    out = ml_i_nuisance_fit(**common, tv_norm=0.2)
    np_ = out["nu_params"]
    assert np_["mode"] == "bins"
    assert len(np_["bin_nu"]) == np_["n_bins"]
    assert np_["tv_norm"] == 0.2
    nu_arr = np.asarray(out["nu_per_refl"], dtype=np.float64)
    assert nu_arr.shape == (len(s2),)
    assert np.all(np.isfinite(nu_arr))
    assert float(out["nu"]) == pytest.approx(float(np.mean(nu_arr)), rel=1e-5)
    assert 3.0 <= nu_arr.min() and nu_arr.max() <= 30.0

    out_none = ml_i_nuisance_fit(**common, tv_norm=0.0)
    out_tv = ml_i_nuisance_fit(**common, tv_norm=1.0)

    def _nu_tv(params: dict) -> float:
        bins = np.asarray(params["bin_nu"], dtype=np.float64)
        return float(np.sum(np.sqrt(np.diff(bins) ** 2 + 1e-6)))

    assert _nu_tv(out_tv["nu_params"]) <= _nu_tv(out_none["nu_params"]) + 1e-6


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None,
    reason="torch required for ml_i_nuisance_fit",
)
def test_nuisance_fit_global_nu_still_works():
    from phridge.contrib.intensity_ll.ops import ml_i_nuisance_fit

    f_calc, f_obs, tune, s2, _ = _packed_arrays(n=1200, seed=4)
    out = ml_i_nuisance_fit(
        f_calc,
        f_obs,
        tune_mask=tune,
        s_sq=s2,
        fit_nu=True,
        nu_mode="global",
        fit_scale=False,
        sigma_a_mode="bins",
        n_sigma_a_bins=8,
        nu_bounds=(3.0, 30.0),
    )
    assert out["nu_params"]["mode"] == "global"
    assert out["nu"] is not None
    assert 3.0 <= float(out["nu"]) <= 30.0
    nu_arr = np.asarray(out["nu_per_refl"], dtype=np.float64)
    np.testing.assert_allclose(nu_arr, float(out["nu"]))


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None,
    reason="torch required for ml_i_nuisance_fit",
)
def test_nuisance_fit_read_mode_still_works():
    from phridge.contrib.intensity_ll.ops import ml_i_nuisance_fit

    f_calc, f_obs, tune, s2, _ = _packed_arrays(n=800)
    out = ml_i_nuisance_fit(
        f_calc,
        f_obs,
        tune_mask=tune,
        s_sq=s2,
        fit_nu=False,
        fit_scale=False,
        sigma_a_mode="read",
    )
    assert out["sigma_a_params"]["mode"] == "read"
    assert "b_delta" in out["sigma_a_params"]
    sa = np.asarray(out["sigma_a"], dtype=np.float64)
    assert sa.min() < sa.max()


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None,
    reason="torch required for ml_i_nuisance_fit",
)
def test_nuisance_fit_tv_norm_smooths_bin_sigma_a():
    from phridge.contrib.intensity_ll.ops import ml_i_nuisance_fit

    f_calc, f_obs, tune, s2, _ = _packed_arrays(n=3000, seed=1)
    common = dict(
        f_calc=f_calc,
        f_obs=f_obs,
        tune_mask=tune,
        s_sq=s2,
        epsilon=np.ones(len(s2)),
        centric=np.zeros(len(s2), dtype=bool),
        fit_nu=False,
        fit_scale=False,
        sigma_a_mode="bins",
        n_sigma_a_bins=12,
        fit_sigma_wilson=True,
    )
    out_none = ml_i_nuisance_fit(**common, tv_norm=0.0)
    out_tv = ml_i_nuisance_fit(**common, tv_norm=0.5)

    def _tv_energy(params: dict) -> float:
        bins = np.asarray(params["bin_sigma_a"], dtype=np.float64)
        return float(np.sum(np.sqrt(np.diff(bins) ** 2 + 1e-6)))

    assert out_tv["sigma_a_params"]["tv_norm"] == 0.5
    assert _tv_energy(out_tv["sigma_a_params"]) <= _tv_energy(out_none["sigma_a_params"]) + 1e-9


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None,
    reason="torch required for ml_i_nuisance_fit",
)
def test_intensity_only_wilson_ml_positive_sigma0():
    from phridge.contrib.intensity_ll.ops import ml_i_nuisance_fit

    f_calc, f_obs, tune, s2, _ = _packed_arrays(n=2500, seed=2)
    common = dict(
        f_calc=f_calc,
        f_obs=f_obs,
        tune_mask=tune,
        s_sq=s2,
        epsilon=np.ones(len(s2)),
        centric=np.zeros(len(s2), dtype=bool),
        fit_nu=False,
        fit_scale=False,
        sigma_a_mode="bins",
        n_sigma_a_bins=8,
    )
    moment = ml_i_nuisance_fit(**common, fit_sigma_wilson=False)
    ml_w = ml_i_nuisance_fit(**common, fit_sigma_wilson=True)

    swp = ml_w["sigma_wilson_params"]
    assert swp["fitted"] is True
    assert swp["method"] == "intensity_ml"
    assert swp["sigma_0"] > 0.0
    assert swp["b_wilson"] >= 0.0
    assert moment["sigma_wilson_params"]["method"] == "moment_plot"
    # Synthetic generator uses Σ₀=200, B_W=10
    assert 20.0 < swp["sigma_0"] < 2000.0
    # Classic residual β = Σ (1 - σ_A²)
    beta = np.asarray(ml_w["beta"], dtype=np.float64)
    sa = np.asarray(ml_w["sigma_a"], dtype=np.float64)
    sw = np.asarray(ml_w["sigma_wilson"], dtype=np.float64)
    np.testing.assert_allclose(beta, sw * (1.0 - sa**2), rtol=1e-6)
    assert sa.max() < 0.995
