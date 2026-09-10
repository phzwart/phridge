"""Guard: no posterior-derived statistic may be labelled an R factor.

The S family (``S_post`` / ``S_prior``) is a different functional from the
crystallographic R factor, and its point-estimate variants are biased low by
posterior shrinkage. If either ever regains the letter R in printed output it
will be pasted into an R column and compared against deposited values, so the
naming is asserted here rather than left to review.
"""

from __future__ import annotations

import io
import re
import warnings

import numpy as np
import pytest

from phridge.client.intensity.engine import IntensityFModelInfo
from phridge.client.intensity.stats_report import (
    IntensityStatsReport,
    ResolutionBinStats,
    format_intensity_stats_report,
)

# The two sanctioned R lines: the legacy French-Wilson amplitude bridge (which
# fills the mandated PDB fields) and the direct-intensity R, a genuine point
# estimate on intensities that carries no posterior shrinkage.
_SANCTIONED = re.compile(
    r"Direct Intensity R|DIRECT INTENSITY R|R VALUE|FREE R VALUE", re.IGNORECASE
)
# "R_work", "R-FREE", "R value", "R_int", "R_plug", ...
_R_LABEL = re.compile(r"R[-_ ]?(value|work|free|int|inf|plug)", re.IGNORECASE)
# Only lines that actually report a number are labels; prose captions such as
# "do not compare with deposited R values" are warnings, not labels.
_HAS_NUMBER = re.compile(r"[=:]\s*-?\d")


def _offending_lines(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        if not _HAS_NUMBER.search(line) or _SANCTIONED.search(line):
            continue
        if _R_LABEL.search(line):
            out.append(line)
    return out


class _FakeFModel:
    """Duck-typed stand-in: ``IntensityFModelInfo`` probes everything with hasattr."""

    target_name = "mli_quad"
    scale_factor = 1.0
    nu = 7.0

    def __init__(self, r_values: dict) -> None:
        self._rv = r_values

    def inferred_r_values(self) -> dict:
        return dict(self._rv)

    # Legacy French-Wilson amplitude R — the deliberate bridge
    def r_work(self) -> float:
        return 0.1912

    def r_free(self) -> float:
        return 0.2233

    def r_all(self) -> float:
        return 0.1935

    # Shrunken amplitude diagnostics
    def r_post_work(self) -> float:
        return 0.1120

    def r_post_free(self) -> float:
        return 0.1284

    def r_post_all(self) -> float:
        return 0.1131

    def r_mode_work(self) -> float:
        return 0.1043

    def r_mode_free(self) -> float:
        return 0.1201

    def r_mode_all(self) -> float:
        return 0.1054

    # Direct intensity R
    def r_intensity_work(self) -> float:
        return 0.2814

    def r_intensity_free(self) -> float:
        return 0.3120

    def r_intensity_all(self) -> float:
        return 0.2831

    # S family
    def s_post_work(self) -> float:
        return 0.3967

    def s_post_free(self) -> float:
        return 0.3949

    def s_post_all(self) -> float:
        return 0.3962

    def s_prior_work(self) -> float:
        return 0.4361

    def s_prior_free(self) -> float:
        return 0.4358

    def s_prior_all(self) -> float:
        return 0.4360

    def cc_work(self) -> float:
        return 0.9412

    def cc_free(self) -> float:
        return 0.9310

    def cc_intensity_work(self) -> float:
        return 0.9388

    def cc_intensity_free(self) -> float:
        return 0.9280

    def target_w(self) -> float:
        return 12450.21

    def target_t(self) -> float:
        return 1238.94


def _r_values() -> dict:
    n = 4
    return {
        "s_post_work": 0.3967,
        "s_post_free": 0.3949,
        "s_prior_work": 0.4361,
        "s_prior_free": 0.4358,
        "k_s_work": 0.7412,
        "k_s_prior_work": 0.9045,
        "s_vis_work": 0.2871,
        "rho2_work": 0.6120,
        "rho2_free": 0.6008,
        "n_ec_outliers": 0,
        "min_free_per_shell": 30,
        "s_report_version": 7,
        "r_intensity_work": 0.2814,
        "r_intensity_free": 0.3120,
        "r_intensity_all": 0.2831,
        "bin_d_max": [30.0, 4.2, 3.1, 2.4],
        "bin_d_min": [4.2, 3.1, 2.4, 1.8],
        "s_post_work_bins": [0.31, 0.35, 0.41, 0.48],
        "s_post_free_bins": [0.33, 0.37, 0.43, 0.50],
        "s_prior_work_bins": [0.42, 0.43, 0.44, 0.45],
        "s_prior_free_bins": [0.42, 0.43, 0.44, 0.45],
        "rho2_work_bins": [0.7, 0.65, 0.6, 0.55],
        "omega_bins": [float("nan")] * n,
        "n_work_bins": [900] * n,
        "n_free_bins": [100] * n,
    }


def _banner_and_bins() -> str:
    info = IntensityFModelInfo(_FakeFModel(_r_values()))
    buf = io.StringIO()
    info.show_all(out=buf)
    return buf.getvalue()


def _stats_report_text() -> str:
    report = IntensityStatsReport(
        n_refl=4000,
        bin_size=1000,
        s_post_work=0.3967,
        s_post_free=0.3949,
        s_prior_work=0.4361,
        s_prior_free=0.4358,
        k_s_work=0.7412,
        k_s_prior_work=0.9045,
        rho2_work=0.6120,
        rho2_free=0.6008,
        s_report_version=7,
    )
    for i in range(4):
        report.bins.append(
            ResolutionBinStats(
                bin=i + 1,
                n=1000,
                d_max=30.0 - 7.0 * i,
                d_min=23.0 - 7.0 * i,
                mean_isig=8.0 - i,
                frac_neg=0.01,
                frac_lt_1=0.05,
                frac_lt_2=0.10,
                frac_lt_3=0.15,
                frac_lt_4=0.20,
                frac_lt_5=0.25,
                mean_sigma_a=0.9 - 0.1 * i,
                mean_sigma_wilson=100.0,
                mean_data_frac=0.8,
                s_post=0.31 + 0.05 * i,
                s_prior=0.42,
                rho2=0.7 - 0.05 * i,
                n_work=900,
                n_free=100,
            )
        )
    return format_intensity_stats_report(report)


def test_default_report_uses_s_post_and_s_prior():
    """(a) The S names actually reach the printed output."""
    banner = _banner_and_bins()
    stats = _stats_report_text()
    for text, what in ((banner, "engine banner"), (stats, "stats report")):
        assert "S_post" in text, f"S_post missing from {what}"
        assert "S_prior" in text, f"S_prior missing from {what}"


def test_no_posterior_statistic_is_labelled_r():
    """(b) Only the French-Wilson and direct-intensity lines may carry an R label."""
    for text in (_banner_and_bins(), _stats_report_text()):
        offenders = _offending_lines(text)
        assert offenders == [], "posterior statistic labelled with R:\n" + "\n".join(offenders)


def test_guard_regex_would_catch_a_regression():
    """The guard is only worth having if it fails on the pre-rename output."""
    assert _offending_lines("| Integrated R_int:  work= 0.2700   free= 0.2900")
    assert _offending_lines("  Inferred R-values (<F>):  r_work=0.1120 r_free=0.1284")
    assert _offending_lines("REMARK   3   POSTERIOR MEAN R-WORK / R-FREE : 0.1120 / 0.1284")
    # ...and does not fire on the two sanctioned lines or on prose captions
    assert not _offending_lines("| Direct Intensity R: r_work= 0.2814   r_free= 0.3120")
    assert not _offending_lines("REMARK   3   R VALUE            (WORKING SET) : 0.1912")
    assert not _offending_lines("  NOT the crystallographic R factor; do not compare with deposited R values.")


def test_never_prints_a_bare_s():
    """Bare capital S is the small-molecule goodness-of-fit; always subscript it.

    ``k_S`` and ``S_post`` do not trip this: the character after S is a word
    character, so there is no word boundary.
    """
    bare_s = re.compile(r"\bS\b")
    for text in (_banner_and_bins(), _stats_report_text()):
        offenders = [ln for ln in text.splitlines() if bare_s.search(ln)]
        assert offenders == [], "bare S printed:\n" + "\n".join(offenders)


def test_s_vis_is_internal():
    """s_vis is a debugging quantity: on the object, never in the default report."""
    banner = _banner_and_bins()
    assert "s_vis" not in banner
    assert "R_plug" not in banner
    assert "S_vis" not in banner


def test_deprecated_r_aliases_warn_and_return_the_new_fields():
    """(c) Old names still resolve, but say so."""
    report = IntensityStatsReport(
        n_refl=10,
        bin_size=10,
        s_post_work=0.3967,
        s_post_free=0.3949,
        s_prior_work=0.4361,
        s_prior_free=0.4358,
        k_s_work=0.7412,
        k_s_prior_work=0.9045,
        s_report_version=7,
    )
    pairs = [
        ("r_int_work", "s_post_work"),
        ("r_int_free", "s_post_free"),
        ("r_inf_work", "s_prior_work"),
        ("r_inf_free", "s_prior_free"),
        ("k_int_work", "k_s_work"),
        ("k_inf_work", "k_s_prior_work"),
        ("r_int_version", "s_report_version"),
    ]
    for old, new in pairs:
        with pytest.warns(DeprecationWarning, match=new):
            value = getattr(report, old)
        assert value == getattr(report, new)


def test_remark_3_deposits_the_legacy_r_not_the_posterior_mean():
    """The mandated PDB fields are the one place a wrong number does real damage."""
    info = IntensityFModelInfo(_FakeFModel(_r_values()))
    buf = io.StringIO()
    info.show_remark_3(out=buf)
    text = buf.getvalue()
    # r_work()/r_free() are the French-Wilson legacy R (0.1912 / 0.2233 in the
    # fake), never the shrunken posterior mean (0.1120 / 0.1284).
    assert "0.1912" in text and "0.2233" in text
    assert "0.1120" not in text and "0.1284" not in text
    assert "S_POST" in text and "S_PRIOR" in text
    assert _offending_lines(text) == []


def test_french_wilson_r_is_wired_to_r_work():
    """r_work/r_free/r_all must route through the French-Wilson bridge."""
    from phridge.client.intensity.engine import IntensityFModel

    calls = []

    class _Probe(IntensityFModel):
        def __init__(self):  # bypass the heavy constructor
            pass

        def _r_french_wilson(self, which="work"):
            calls.append(which)
            return 0.1234

    probe = _Probe()
    assert probe.r_work() == 0.1234
    assert probe.r_free() == 0.1234
    assert probe.r_all() == 0.1234
    assert calls == ["work", "free", "all"]


def test_deprecated_aliases_are_not_in_printed_output():
    """The bridge is for callers only — the report must never print the old names."""
    text = _banner_and_bins() + _stats_report_text()
    for old in ("r_int_", "r_inf_", "k_int", "k_inf", "R_int", "R_inf"):
        assert old not in text, f"{old!r} leaked into printed output"
