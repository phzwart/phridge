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
