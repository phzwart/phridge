"""Unit tests for synthetic crystallographic intensity generation with Student-t noise."""

from __future__ import annotations

import numpy as np
import pytest

from phridge.contrib.intensity_ll.synthetic import (
    SyntheticIntensityResult,
    compute_base_sigmas,
    generate_synthetic_series,
    sample_student_t,
)


def test_sample_student_t():
    rng = np.random.default_rng(42)
    t, w = sample_student_t(nu=8.0, size=(10000,), rng=rng)

    assert t.shape == (10000,)
    assert w.shape == (10000,)
    assert np.all(w > 0.0)

    # Theoretical mean = 0, variance = nu / (nu - 2) = 8 / 6 = 1.333...
    assert abs(np.mean(t)) < 0.05
    emp_var = np.var(t)
    assert abs(emp_var - 1.333) < 0.15

    # Reproducibility
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    t1, _ = sample_student_t(nu=5.0, size=100, rng=rng1)
    t2, _ = sample_student_t(nu=5.0, size=100, rng=rng2)
    np.testing.assert_allclose(t1, t2)


def test_compute_base_sigmas():
    i_true = np.array([0.0, 10.0, 100.0, 1000.0])
    sig_exp = np.array([1.0, 2.0, 5.0, 20.0])

    # Direct mode with sig_exp
    sig_direct = compute_base_sigmas(i_true, sig_exp=sig_exp, mode="direct")
    np.testing.assert_allclose(sig_direct, sig_exp)

    # Poisson mode
    sig_poiss = compute_base_sigmas(i_true, mode="poisson")
    assert np.all(sig_poiss >= 1.0)
    assert sig_poiss[3] > sig_poiss[2] > sig_poiss[1]

    # SNR mode
    sig_snr = compute_base_sigmas(i_true, mode="snr")
    assert len(sig_snr) == 4
    assert np.all(sig_snr > 0.0)


def test_synthetic_intensity_result_dataclass():
    res = SyntheticIntensityResult(
        multiplier=2.0,
        i_synth=np.array([1.0, 2.0]),
        sig_synth=np.array([0.1, 0.2]),
        stats={"mean_i_over_sig": 10.0},
    )
    assert res.multiplier == 2.0
    assert len(res.i_synth) == 2
    assert res.stats["mean_i_over_sig"] == 10.0


def test_generate_synthetic_series_in_memory():
    class DummyMillerArray:
        def __init__(self, data, sigmas, d_spacings):
            self._data = data
            self._sigmas = sigmas
            self._d = d_spacings

        def data(self):
            return self._data

        def sigmas(self):
            return self._sigmas

        def d_spacings(self):
            class D:
                def __init__(self, d):
                    self._d = d
                def data(self):
                    return self._d
            return D(self._d)

    class DummyModel:
        def __init__(self):
            n = 500
            self.i_obs = DummyMillerArray(
                data=np.linspace(10.0, 500.0, n),
                sigmas=np.linspace(1.0, 25.0, n),
                d_spacings=np.linspace(1.5, 10.0, n),
            )
            self.f_calc = DummyMillerArray(
                data=np.sqrt(np.linspace(10.0, 500.0, n)),
                sigmas=None,
                d_spacings=np.linspace(1.5, 10.0, n),
            )
            self.n_bins = 5

    model = DummyModel()
    results, summary_rows = generate_synthetic_series(
        model=model,
        multipliers=[0.0, 1.0, 3.0],
        nu=7.0,
        mode="direct",
        write_mtz=False,
    )

    assert len(results) == 3
    assert len(summary_rows) == 3

    # Zero noise multiplier has zero noise R factor
    assert summary_rows[0]["multiplier"] == 0.0
    assert summary_rows[0]["r_noise_pct"] == 0.0
    assert summary_rows[0]["corr_true_synth"] == 1.0

    # Multiplier 3 has higher noise than multiplier 1
    assert summary_rows[2]["r_noise_pct"] > summary_rows[1]["r_noise_pct"]

    # Check shell stats structure
    res1 = results[1.0]
    assert len(res1.shell_stats) == 5
    assert "mean_i_over_sig" in res1.shell_stats[0]
    assert "r_noise_percent" in res1.shell_stats[0]
