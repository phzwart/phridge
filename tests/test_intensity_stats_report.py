"""Tests for resolution-binned intensity / σ_A stats reports."""

from __future__ import annotations

import numpy as np

from phridge.client.intensity.stats_report import (
    compute_intensity_stats_report,
    format_intensity_stats_report,
    stats_report_enabled,
)


def test_compute_intensity_stats_report_bins_and_sigma_a():
    n = 1200
    rng = np.random.default_rng(0)
    # Fake resolution: d from 10 → 1.5 Å
    d = np.linspace(10.0, 1.5, n)
    s2 = 1.0 / np.maximum(d * d, 1e-6)
    sigma_wilson = 100.0 * np.exp(-0.5 * 20.0 * s2)
    sigma_a = np.clip(np.sqrt(0.9) * np.exp(-0.25 * 30.0 * s2), 1e-4, 0.999)
    # Intensities: mix of strong and weak / negative at high res
    sig = 0.05 * sigma_wilson + 1.0
    io = rng.normal(loc=sigma_wilson, scale=sig)
    io[-200:] = rng.normal(loc=0.0, scale=sig[-200:])  # weak outer

    report = compute_intensity_stats_report(
        intensities=io,
        sigmas=sig,
        d_spacings=d,
        epsilon=np.ones(n),
        sigma_a=sigma_a,
        sigma_wilson=sigma_wilson,
        bin_size=500,
        sigma_a_params={"k": 0.9, "b_delta": 30.0},
        sigma_wilson_params={"sigma_0": 100.0, "b_wilson": 20.0},
        nu=7.0,
        label="unit-test",
    )
    assert report.n_refl == n
    assert report.bin_size == 500
    assert len(report.bins) == 3
    assert report.bins[0].n == 500
    assert report.bins[-1].n == 200
    # Low-res bin should have higher σ_A than outermost
    assert report.bins[0].mean_sigma_a > report.bins[-1].mean_sigma_a
    assert 0.0 <= report.bins[0].mean_data_frac <= 1.0
    text = format_intensity_stats_report(report)
    assert "σ_A" in text
    assert "data%" in text
    assert "unit-test" in text


def test_stats_report_enabled_env(monkeypatch):
    monkeypatch.delenv("PHRIDGE_STATS_REPORT", raising=False)
    assert stats_report_enabled(default=True) is True
    monkeypatch.setenv("PHRIDGE_STATS_REPORT", "0")
    assert stats_report_enabled(default=True) is False
    monkeypatch.setenv("PHRIDGE_STATS_REPORT", "1")
    assert stats_report_enabled(default=False) is True
