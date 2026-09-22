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
    compute_cc_isig_table,
    format_intensity_stats_report,
)

# The sanctioned R lines: anything that names French-Wilson explicitly (the
# amplitude bridge that fills the mandated PDB fields), and the direct-intensity R,
# a genuine point estimate on intensities that carries no posterior shrinkage.
# Naming French-Wilson is what earns a line the right to an R label, so the guard
# doubles as the check that every reported R says which amplitudes it used.
_SANCTIONED = re.compile(
    r"Direct Intensity R|DIRECT INTENSITY R|R VALUE|FREE R VALUE|French-?Wilson",
    re.IGNORECASE,
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
        "s_report_version": 9,
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


def _cc_isig_fixture():
    """A small but fully populated CC_I table."""
    rng = np.random.default_rng(4)
    n = 6000
    d = 2.0 + 20.0 * rng.random(n) ** 2
    sig = 1.0 + 8.0 * (2.0 / d) ** 2
    i_calc = np.abs(rng.normal(0.0, 30.0, n))
    i_obs = i_calc + rng.normal(0.0, sig * 3.0, n)
    return compute_cc_isig_table(
        intensities=i_obs,
        sigmas=sig,
        d_spacings=d,
        i_calc=i_calc,
        r_free=rng.random(n) < 0.1,
        # populated so the guard sees the σ_A column rendered, not the n/a fallback
        sigma_a=np.clip(0.95 - 0.5 * (2.0 / d), 0.1, 0.95),
        n_shells=4,
    )


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
        s_report_version=9,
        # French-Wilson R (the conventional one) alongside the shrunken point
        # estimates, so the guard below sees every agreement line the report can print.
        r_fw_work=0.1912,
        r_fw_free=0.2233,
        r_post_work=0.1120,
        r_post_free=0.1284,
        r_mode_work=0.1043,
        r_mode_free=0.1201,
        r_intensity_work=0.2814,
        r_intensity_free=0.3120,
        cc_isig=_cc_isig_fixture(),
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
        s_report_version=9,
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


def test_remark_3_says_the_r_values_are_french_wilson():
    """The provenance has to be in the deposited text, not only in the source.

    A reader of REMARK 3 cannot see which amplitudes went into the mandated R fields,
    and under ``mli_quad`` the refinement target never forms ``F_obs`` at all — so the
    only thing keeping that R interpretable is the label saying French-Wilson.
    """
    info = IntensityFModelInfo(_FakeFModel(_r_values()))
    buf = io.StringIO()
    info.show_remark_3(out=buf)
    text = buf.getvalue()
    for line in text.splitlines():
        if "R VALUE" in line and _HAS_NUMBER.search(line):
            assert "FRENCH-WILSON" in line, line
    assert "FREE R VALUE" in text
    assert "FRENCH-WILSON POSTERIOR AMPLITUDES" in text


class _FakeCctbxInfo:
    """Stand-in for ``mmtbx.f_model.f_model_info.info``, which needs cctbx."""

    def __init__(self, r_work, r_free):
        self.r_work = r_work
        self.r_free = r_free

    def show_remark_3(self, out):
        print(f"REMARK   3   R VALUE            (WORKING SET) : {self.r_work}", file=out)
        print(f"REMARK   3   FREE R VALUE                     : {self.r_free}", file=out)


def _remark_3_with_cctbx(r_work, r_free) -> str:
    info = IntensityFModelInfo(_FakeFModel(_r_values()))
    info._cctbx_info = _FakeCctbxInfo(r_work, r_free)
    buf = io.StringIO()
    info.show_remark_3(out=buf)
    return buf.getvalue()


def test_cctbx_r_fields_agreeing_with_french_wilson_raise_no_warning():
    """cctbx delegates to r_work()/r_free(), so the normal case is silent.

    Both conventions count as agreement: cctbx reports R as a fraction in some versions
    and as a percentage in others, and neither is a provenance problem.
    """
    for r_work, r_free in ((0.1912, 0.2233), (19.12, 22.33)):
        text = _remark_3_with_cctbx(r_work, r_free)
        assert "WARNING" not in text, text
        assert "FRENCH-WILSON" in text


def test_cctbx_r_fields_computed_from_the_scaffold_are_flagged():
    """A cctbx version computing R itself would use the sqrt(max(I,0)) scaffold.

    That silently puts two different numbers under one REMARK 3 field name, and the
    wrong one is the one a depositor reads first — so it has to be called out.
    """
    text = _remark_3_with_cctbx(0.4410, 0.4602)
    assert "WARNING" in text
    assert "R_WORK 0.4410 vs 0.1912" in text
    assert "DO NOT DEPOSIT THEM" in text
    # the French-Wilson values are still printed, and still labelled
    assert "0.1912" in text and "FRENCH-WILSON" in text


def _aniso_fixture():
    """An AnisoScale with a known tensor, built the way the engine builds it."""
    from phridge.client.intensity.stats_report import fit_aniso_scale
    from phridge.sfcalc.engine.cell import orthogonalization_matrix

    cell = (78.1, 78.1, 37.0, 90.0, 90.0, 90.0)
    b = np.diag([-1.5, -1.5, 3.0])
    o_inv = np.linalg.inv(orthogonalization_matrix(cell))
    u_star = o_inv @ (b / (8 * np.pi**2)) @ o_inv.T
    rng = np.random.default_rng(7)
    hkl = rng.integers(-25, 26, size=(4000, 3)).astype(np.float64)
    hkl = hkl[np.any(hkl != 0, axis=1)]
    k = np.exp(-2 * np.pi**2 * np.einsum("ni,ij,nj->n", hkl, u_star, hkl))
    return fit_aniso_scale(miller_indices=hkl, k_anisotropic=k, unit_cell=cell)


@pytest.mark.parametrize("with_cctbx", [False, True])
def test_remark_3_carries_the_overall_anisotropic_b_value(with_cctbx):
    """The mandated field must be written, in both branches, with its provenance.

    cctbx cannot supply this one: ``b_cart`` on the fmodel is ``None`` by construction,
    so a cctbx-written block would carry nothing or zeros for a field describing the
    applied scale. The values come from ``k_anisotropic`` instead and say so.
    """
    fmodel = _FakeFModel(_r_values())
    fmodel.aniso_scale = _aniso_fixture
    info = IntensityFModelInfo(fmodel)
    if with_cctbx:
        info._cctbx_info = _FakeCctbxInfo(0.1912, 0.2233)
    buf = io.StringIO()
    info.show_remark_3(out=buf)
    text = buf.getvalue()

    assert "REMARK   3  OVERALL ANISOTROPIC B VALUE." in text
    for name in ("B11", "B22", "B33", "B12", "B13", "B23"):
        assert f"REMARK   3   {name} (A**2) : " in text, name
    assert "-1.50000" in text and "3.00000" in text
    assert "FITTED FROM THE APPLIED K_ANISOTROPIC" in text
    assert "ANISOTROPY" in text
    # the B fields must not trip the guard on posterior-derived R naming
    assert _offending_lines(text) == []


def test_aniso_b_banner_lines_stay_inside_the_panel():
    """A large anisotropy must not push the panel border out of alignment."""
    from phridge.client.intensity.stats_report import AnisoScale

    fmodel = _FakeFModel(_r_values())
    huge = AnisoScale(
        b_cart=(-1234.567, 890.123, -0.0, 456.789, -0.0, 12.5),
        principal_b=(-1300.0, 5.0, 900.0),
        b_iso_equiv=-114.8,
        anisotropy=2200.0,
        log_rms_residual=0.01,
        n_refl=5000,
    )
    fmodel.aniso_scale = lambda: huge
    buf = io.StringIO()
    IntensityFModelInfo(fmodel).show_rfactors_targets_scales_overall(out=buf)
    lines = buf.getvalue().splitlines()
    border = len(lines[0])
    aniso_lines = [ln for ln in lines if "B11=" in ln or "B12=" in ln]
    assert len(aniso_lines) == 2
    # exactly the border width, so the closing pipes line up with every other row
    for ln in aniso_lines:
        assert len(ln) == border, (len(ln), border, ln)
        assert ln.endswith("|")
    # and the structural zeros do not read as measured negatives
    assert "-0.000" not in "".join(aniso_lines)


def test_remark_3_omits_the_anisotropic_b_when_it_is_unavailable():
    """No tensor means no block — never a row of zeros in a deposited field."""
    fmodel = _FakeFModel(_r_values())
    fmodel.aniso_scale = lambda: None
    buf = io.StringIO()
    IntensityFModelInfo(fmodel).show_remark_3(out=buf)
    assert "ANISOTROPIC B" not in buf.getvalue()


def test_every_reported_r_names_its_amplitudes():
    """No R may be printed with a number and no statement of where F_obs came from.

    ``mli_quad`` has three different things a reader could mistake for "the R factor"
    — French-Wilson amplitude R, direct intensity R, and the shrunken posterior-mean
    diagnostic. Each printed R line must therefore identify itself.
    """
    text = _banner_and_bins()
    unlabelled = []
    for line in text.splitlines():
        if not (_R_LABEL.search(line) and _HAS_NUMBER.search(line)):
            continue
        if not _SANCTIONED.search(line):
            unlabelled.append(line)
    assert unlabelled == [], "R reported without naming its amplitudes:\n" + "\n".join(unlabelled)
    # and the French-Wilson R is actually present, not merely permitted
    assert "French-Wilson" in text


def test_r_factors_reports_french_wilson_not_the_posterior_mean():
    """``r_factors()`` is what Phenix asks for "the R factors"; it must be FW.

    This is the one caller that used to prefer ``r_post_*``, the statistic whose own
    docstring says it improves as the data get worse. It must return the French-Wilson
    values and say so in the formatted string.
    """
    from phridge.client.intensity.engine import IntensityFModel

    class _Probe(IntensityFModel):
        def __init__(self):  # bypass the heavy constructor
            pass

        def _r_french_wilson(self, which="work"):
            return {"work": 0.1912, "free": 0.2233, "all": 0.1935}[which]

        def inferred_r_values(self):
            # the shrunken diagnostics are present and must be ignored
            return {
                "r_post_work": 0.1120,
                "r_post_free": 0.1284,
                "r_post_all": 0.1131,
                "cc_post_work": 0.9412,
                "cc_post_free": 0.9310,
            }

    text = _Probe().r_factors(prefix="  ")
    assert "0.1912" in text and "0.2233" in text and "0.1935" in text
    assert "0.1120" not in text and "0.1284" not in text
    assert "French-Wilson" in text

    values = _Probe().r_factors(as_string=False)
    values = values if isinstance(values, dict) else values.__dict__
    assert values["r_work"] == 0.1912
    assert values["r_free"] == 0.2233
    assert values["r_source"] == "french_wilson"


def test_french_wilson_aliases_match_the_plain_names():
    """The explicit aliases exist so a report never has to print a bare ``r_work``."""
    from phridge.client.intensity.engine import IntensityFModel

    class _Probe(IntensityFModel):
        def __init__(self):
            pass

        def _r_french_wilson(self, which="work"):
            return {"work": 0.19, "free": 0.22, "all": 0.20}[which]

    probe = _Probe()
    assert probe.r_work_french_wilson() == probe.r_work()
    assert probe.r_free_french_wilson() == probe.r_free()
    assert probe.r_all_french_wilson() == probe.r_all()


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
