"""Tests for modularized intensity package and compatibility shim."""

from __future__ import annotations

import numpy as np

import phridge.client.intensity as intensity
import phridge.client.intensity_tool as intensity_tool
from phridge.client.intensity.cli import parse_omit_selection
from phridge.client.intensity.maps import IntensityMapMixin
from phridge.client.intensity.math_utils import (
    isotonic_non_increasing,
    tv_denoise_1d,
)
from phridge.client.intensity.model import IntensityModel
from phridge.client.intensity.refine import IntensityRefineMixin


def test_intensity_package_exports():
    """Verify clean exports and identity with backward-compat shim."""
    assert intensity.IntensityModel is IntensityModel
    assert intensity.IntensityModel is intensity_tool.IntensityModel
    assert intensity.IntensityMapMixin is IntensityMapMixin
    assert intensity.IntensityRefineMixin is IntensityRefineMixin
    assert intensity.tv_denoise_1d is tv_denoise_1d
    assert intensity.isotonic_non_increasing is isotonic_non_increasing
    assert intensity.parse_omit_selection is parse_omit_selection
    assert issubclass(IntensityModel, IntensityMapMixin)
    assert issubclass(IntensityModel, IntensityRefineMixin)


def test_tv_denoise_1d():
    y = np.array([0.9, 0.5, 0.85, 0.4, 0.8, 0.3])
    raw_tv = np.sum(np.abs(np.diff(y)))

    denoised_light = tv_denoise_1d(y, lam=0.05)
    denoised_heavy = tv_denoise_1d(y, lam=0.5)

    tv_light = np.sum(np.abs(np.diff(denoised_light)))
    tv_heavy = np.sum(np.abs(np.diff(denoised_heavy)))

    assert tv_light < raw_tv
    assert tv_heavy < tv_light
    assert np.all(denoised_heavy >= 0.01)
    assert np.all(denoised_heavy <= 0.999)


def test_isotonic_non_increasing():
    y = np.array([0.5, 0.8, 0.6, 0.9, 0.4, 0.7])
    mono = isotonic_non_increasing(y)

    assert len(mono) == len(y)
    diffs = np.diff(mono)
    assert np.all(diffs <= 1e-9)
    assert np.min(mono) >= np.min(y) - 1e-9
    assert np.max(mono) <= np.max(y) + 1e-9


def test_parse_omit_selection():
    assert parse_omit_selection("45-52") == "resseq 45:52"
    assert parse_omit_selection("45:52") == "resseq 45:52"
    assert parse_omit_selection("52") == "resseq 52"
    assert parse_omit_selection("chain A and resseq 10:20") == "chain A and resseq 10:20"
