"""Tests for resolution-binned intensity / σ_A stats reports."""

from __future__ import annotations

import numpy as np
import pytest

from phridge.client.intensity.stats_report import (
    AnisoScale,
    IntensityStatsReport,
    compute_cc_isig_table,
    compute_intensity_stats_report,
    fit_aniso_scale,
    format_aniso_scale_remark_3,
    format_aniso_scale_summary,
    format_cc_isig_table,
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


# ============================================================ agreement statistics
def test_report_header_names_the_amplitudes_behind_every_number():
    """The conventional R is labelled French-Wilson; the shrunken ones are not R."""
    report = IntensityStatsReport(
        n_refl=10,
        bin_size=10,
        r_fw_work=0.1912,
        r_fw_free=0.2233,
        r_post_work=0.1120,
        r_post_free=0.1284,
        r_mode_work=0.1043,
        r_mode_free=0.1201,
        r_intensity_work=0.2814,
        r_intensity_free=0.3120,
    )
    text = format_intensity_stats_report(report)
    assert "R (French-Wilson amplitudes, model-free F_obs" in text
    assert "0.1912" in text and "0.2233" in text
    # posterior point estimates present, but never under an R label
    assert "0.1120" in text and "0.1043" in text
    assert "NOT R factors" in text
    for line in text.splitlines():
        if "0.1120" in line or "0.1043" in line:
            assert "R_work" not in line and "R work" not in line and "r_work" not in line

    # absent values print nothing rather than a row of n/a
    bare = format_intensity_stats_report(IntensityStatsReport(n_refl=10, bin_size=10))
    assert "French-Wilson" not in bare
    assert "NOT R factors" not in bare


# ============================================================ CC_I table
def _cc_problem(n=12000, seed=0, model_quality=0.9):
    """Synthetic I_obs / I_calc whose agreement degrades with resolution."""
    rng = np.random.default_rng(seed)
    d = 2.0 + 25.0 * rng.random(n) ** 2
    sigma_wilson = 100.0 * np.exp(-2.0 * 20.0 / (d * d))
    e_true = np.abs(rng.normal(0, 1, n) + 1j * rng.normal(0, 1, n)) / np.sqrt(2)
    sa = np.clip(model_quality - 0.5 * (2.0 / d), 0.05, 0.99)
    e_c = np.abs(
        sa * e_true
        + np.sqrt(1 - sa**2) * np.abs(rng.normal(0, 1, n) + 1j * rng.normal(0, 1, n)) / np.sqrt(2)
    )
    sig = 0.05 * sigma_wilson + 0.4 * sigma_wilson * (2.0 / d) ** 2
    i_obs = sigma_wilson * e_true**2 + rng.normal(0.0, sig)
    return dict(
        intensities=i_obs,
        sigmas=sig,
        d_spacings=d,
        i_calc=sigma_wilson * e_c**2,
        r_free=rng.random(n) < 0.1,
    )


def test_cc_isig_bins_are_the_fixed_boundaries_not_quantiles():
    """The columns must be <0, 0-1, 1-2, 2-3, 3-5, 5-9, >9 -- and provably so.

    Fixed bins are the whole point: a column has to mean the same I/sigma band in every
    shell and every run, so the table can be compared across data sets. The check is
    that each reflection lands in the bin its own I/sigma says it should.
    """
    p = _cc_problem()
    t = compute_cc_isig_table(**p, n_shells=6)
    assert t.labels == ["<0", "0-1", "1-2", "2-3", "3-5", "5-9", ">9"]
    assert t.n_bins == 7
    assert t.edges == [0.0, 1.0, 2.0, 3.0, 5.0, 9.0]
    assert len(t.column_marginal) == 7
    assert len(t.rows) == 6

    snr = p["intensities"] / p["sigmas"]
    expected = [
        int(np.count_nonzero(snr < 0)),
        int(np.count_nonzero((snr >= 0) & (snr < 1))),
        int(np.count_nonzero((snr >= 1) & (snr < 2))),
        int(np.count_nonzero((snr >= 2) & (snr < 3))),
        int(np.count_nonzero((snr >= 3) & (snr < 5))),
        int(np.count_nonzero((snr >= 5) & (snr < 9))),
        int(np.count_nonzero(snr >= 9)),
    ]
    assert [c.n for c in t.column_marginal] == expected
    # unlike quantiles, the populations are deliberately uneven
    assert max(expected) > 3 * min(expected), expected

    # shells still hold equal counts and run low -> high resolution
    counts = [r.n for r in t.rows]
    assert max(counts) - min(counts) <= 2, counts
    d_mins = [r.d_min for r in t.rows]
    assert d_mins == sorted(d_mins, reverse=True), d_mins
    # every reflection lands in exactly one cell
    assert sum(c.n for r in t.rows for c in r.cells) == t.overall.n


def test_cc_isig_custom_edges_and_env_override(monkeypatch):
    from phridge.client.intensity.stats_report import cc_isig_edges

    p = _cc_problem(n=3000)
    t = compute_cc_isig_table(**p, edges=[0.0, 2.0, 6.0], n_shells=3)
    assert t.labels == ["<0", "0-2", "2-6", ">6"]

    monkeypatch.delenv("PHRIDGE_CC_ISIG_EDGES", raising=False)
    assert cc_isig_edges() == (0.0, 1.0, 2.0, 3.0, 5.0, 9.0)
    monkeypatch.setenv("PHRIDGE_CC_ISIG_EDGES", "0, 2, 4, 8")
    assert cc_isig_edges() == (0.0, 2.0, 4.0, 8.0)
    # malformed or non-increasing input must not produce meaningless columns
    for bad in ("", "abc", "3,1,2", "1,1", "0,,x"):
        monkeypatch.setenv("PHRIDGE_CC_ISIG_EDGES", bad)
        assert cc_isig_edges() == (0.0, 1.0, 2.0, 3.0, 5.0, 9.0), bad


def test_cc_isig_fractions_are_row_normalized():
    """Each row's population shares must sum to 1, and the columns to 1 overall."""
    t = compute_cc_isig_table(**_cc_problem(), n_shells=6)
    for row in t.rows:
        fracs = [c.frac for c in row.cells]
        assert all(np.isfinite(f) for f in fracs)
        assert sum(fracs) == pytest.approx(1.0, abs=1e-12)
        # and the fraction really is this cell's share of this shell
        assert fracs[0] == pytest.approx(row.cells[0].n / row.n, abs=1e-12)
        assert row.marginal.frac == pytest.approx(1.0, abs=1e-12)
    assert sum(c.frac for c in t.column_marginal) == pytest.approx(1.0, abs=1e-12)
    assert t.overall.frac == pytest.approx(1.0, abs=1e-12)


def test_cc_isig_reports_mean_sigma_a_per_shell():
    """The σ_A column is what makes reading down a column a σ_A scan."""
    p = _cc_problem()
    sa = np.clip(0.9 - 0.5 * (2.0 / p["d_spacings"]), 0.05, 0.99)
    t = compute_cc_isig_table(**p, sigma_a=sa, n_shells=6)
    vals = [r.mean_sigma_a for r in t.rows]
    assert all(np.isfinite(v) for v in vals)
    # σ_A falls with resolution, so it must fall down the table
    assert vals == sorted(vals, reverse=True), vals
    assert "σ_A" in "\n".join(format_cc_isig_table(t))

    # and without σ_A the column degrades to n/a rather than breaking
    t2 = compute_cc_isig_table(**p, n_shells=3)
    assert all(not np.isfinite(r.mean_sigma_a) for r in t2.rows)
    assert "n/a" in "\n".join(format_cc_isig_table(t2))


def test_cc_isig_recovers_the_expected_gradients():
    """CC_I must fall toward weak I/σ and toward high resolution (falling σ_A).

    The assertion is a trend, not a strict ordering. Cells hold a few hundred
    reflections at most, so the CC standard error is around 0.04 and adjacent cells
    invert routinely; requiring monotonicity would make this a noise detector.
    """
    p = _cc_problem()
    sa = np.clip(0.9 - 0.5 * (2.0 / p["d_spacings"]), 0.05, 0.99)
    t = compute_cc_isig_table(**p, sigma_a=sa, n_shells=6)

    cols = [c.cc_work for c in t.column_marginal]
    assert np.isfinite(cols[0]) and np.isfinite(cols[-1])
    assert cols[0] < cols[-1], cols  # negative-intensity bin agrees worst
    # Conditioning on I_obs<0 drives the CC negative by selection alone: a larger I_calc
    # needs more negative noise to land below zero. Pin that so the legend stays honest.
    assert cols[0] < 0.0, cols

    checked = 0
    for row in t.rows:
        vals = [c.cc_work for c in row.cells if np.isfinite(c.cc_work)]
        if len(vals) < 4:
            continue
        checked += 1
        assert vals[0] < vals[-1], (row.shell, vals)
        half = len(vals) // 2
        assert np.mean(vals[:half]) < np.mean(vals[half:]), (row.shell, vals)
    assert checked >= 3, f"only {checked} shells had enough populated cells"

    # the σ_A dependence: outermost shell agrees worse than innermost
    assert t.rows[-1].marginal.cc_work < t.rows[0].marginal.cc_work


def test_cc_isig_thin_cells_lose_their_cc_but_keep_their_population():
    """A cell too thin for a CC must still report how many reflections it holds.

    With fixed bins some cells are legitimately near-empty, and 'this bin holds 0.3% of
    the shell' is itself the answer -- suppressing the count too would hide it.
    """
    t = compute_cc_isig_table(**_cc_problem(n=600), n_shells=8, min_n=30)
    saw_thin = False
    for row in t.rows:
        for c in row.cells:
            if c.n_work < 30:
                assert not np.isfinite(c.cc_work), (c.n_work, c.cc_work)
            if c.n_free < 30:
                assert not np.isfinite(c.cc_free), (c.n_free, c.cc_free)
            if c.n < 30:
                saw_thin = True
                assert np.isfinite(c.frac)  # population share survives
    assert saw_thin, "fixture did not produce a thin cell"


def test_cc_isig_marginal_matches_a_direct_correlation():
    """The overall cell must be the plain Pearson CC, matching cc_intensity_work."""
    p = _cc_problem()
    t = compute_cc_isig_table(**p, n_shells=6)
    work = ~p["r_free"]
    expected = np.corrcoef(p["intensities"][work], p["i_calc"][work])[0, 1]
    assert t.overall.cc_work == pytest.approx(expected, abs=1e-12)


def test_cc_isig_formats_in_fixed_width_columns():
    """Negative CC in a weak cell must not shift the columns."""
    t = compute_cc_isig_table(**_cc_problem(n=3000), n_shells=5)
    t.rows[0].cells[0].cc_work = -0.05
    t.rows[0].cells[1].cc_free = float("nan")
    lines = format_cc_isig_table(t)
    body = [ln for ln in lines if ln[:5].strip().isdigit()]
    assert len(body) == len(t.rows)
    assert len({len(ln) for ln in lines if ln.strip()}) > 0
    assert len({len(ln) for ln in body}) == 1, [len(ln) for ln in body]
    # negatives drop the leading zero to hold the 4-char field
    assert "-.05" in body[0]
    joined = "\n".join(lines)
    assert "CC_I = corr(I_obs, I_calc)" in joined
    assert "FIXED boundaries, not quantiles" in joined
    assert "model-free" in joined
    assert "range restriction" in joined
    # the fixed bin labels appear as column headers
    for label in ("<0", "0-1", "3-5", ">9"):
        assert label in joined
    # every shell contributes a CC line and a % line
    assert joined.count("%") >= len(t.rows) * t.n_bins


def test_cc_isig_degrades_on_empty_and_flag_free_input():
    empty = compute_cc_isig_table(
        intensities=np.array([]), sigmas=np.array([]), d_spacings=np.array([]),
        i_calc=np.array([]),
    )
    assert empty.rows == []
    assert empty.labels  # the bin definition survives even with no data
    assert format_cc_isig_table(empty) == []

    p = _cc_problem(n=2000)
    p.pop("r_free")  # no test set at all
    t = compute_cc_isig_table(**p, n_shells=3)
    assert t.overall.n_free == 0
    assert not np.isfinite(t.overall.cc_free)
    assert np.isfinite(t.overall.cc_work)
    assert t.overall.frac == pytest.approx(1.0)


# ============================================================ overall anisotropic B
_ORTHO_CELL = (78.1, 78.1, 37.0, 90.0, 90.0, 90.0)
_MONO_CELL = (61.0, 74.5, 92.3, 90.0, 104.2, 90.0)  # non-diagonal O, so B12/B13/B23 mix


def _aniso_problem(b_cart_mat, cell, n=6000, seed=0, k_scale=1.0):
    """A ``k_anisotropic`` array generated from a known Cartesian B.

    Inverts the reporting path exactly (``u_cart = O U* O^T``), so a fit that does the
    conversion wrong cannot pass by luck -- the monoclinic cell has a non-diagonal O,
    which mixes the off-diagonal components.
    """
    from phridge.sfcalc.engine.cell import orthogonalization_matrix

    o_inv = np.linalg.inv(orthogonalization_matrix(cell))
    u_star = o_inv @ (np.asarray(b_cart_mat, dtype=np.float64) / (8 * np.pi**2)) @ o_inv.T
    rng = np.random.default_rng(seed)
    hkl = rng.integers(-25, 26, size=(n, 3)).astype(np.float64)
    hkl = hkl[np.any(hkl != 0, axis=1)]
    k = k_scale * np.exp(-2 * np.pi**2 * np.einsum("ni,ij,nj->n", hkl, u_star, hkl))
    return hkl, k


@pytest.mark.parametrize("cell", [_ORTHO_CELL, _MONO_CELL])
def test_aniso_b_recovers_a_known_tensor_exactly(cell):
    """The fit is linear least squares on an exact model, so it must be exact."""
    b_true = np.array([[3.1, -0.8, 1.4], [-0.8, -1.9, 0.3], [1.4, 0.3, -5.2]])
    hkl, k = _aniso_problem(b_true, cell)
    a = fit_aniso_scale(miller_indices=hkl, k_anisotropic=k, unit_cell=cell)

    assert a.is_valid
    assert a.b_cart == pytest.approx((3.1, -1.9, -5.2, -0.8, 1.4, 0.3), abs=1e-8)
    assert a.log_rms_residual < 1e-10
    assert a.n_refl == hkl.shape[0]

    eig = np.sort(np.linalg.eigvalsh(b_true))
    assert a.principal_b == pytest.approx(tuple(eig), abs=1e-8)
    assert a.anisotropy == pytest.approx(eig[-1] - eig[0], abs=1e-8)
    assert a.b_iso_equiv == pytest.approx(np.trace(b_true) / 3.0, abs=1e-8)


def test_aniso_b_does_not_let_an_overall_scale_leak_into_the_tensor():
    """The fitted constant must absorb normalization, not the isotropic part of B.

    Without the constant in the design matrix a ``k_anisotropic`` array that is not
    normalized to 1 would tilt the whole tensor, which is the failure this pins down.
    """
    b_true = np.diag([2.0, 2.0, -4.0])
    hkl, k = _aniso_problem(b_true, _ORTHO_CELL)
    plain = fit_aniso_scale(miller_indices=hkl, k_anisotropic=k, unit_cell=_ORTHO_CELL)
    scaled = fit_aniso_scale(
        miller_indices=hkl, k_anisotropic=7.3 * k, unit_cell=_ORTHO_CELL
    )
    assert scaled.b_cart == pytest.approx(plain.b_cart, abs=1e-8)
    assert scaled.log_scale_offset - plain.log_scale_offset == pytest.approx(
        np.log(7.3), abs=1e-8
    )


def test_aniso_b_residual_exposes_a_non_debye_waller_scale():
    """A scale that is not of Debye-Waller form must show up in the residual.

    The reported B is then only the best anisotropic projection of whatever was applied,
    and the residual is the only thing that tells a reader so.
    """
    b_true = np.diag([1.5, 1.5, -3.0])
    hkl, k = _aniso_problem(b_true, _ORTHO_CELL)
    clean = fit_aniso_scale(miller_indices=hkl, k_anisotropic=k, unit_cell=_ORTHO_CELL)

    rng = np.random.default_rng(11)
    lumpy = k * np.exp(rng.normal(0.0, 0.25, k.size))  # not expressible as a tensor
    dirty = fit_aniso_scale(miller_indices=hkl, k_anisotropic=lumpy, unit_cell=_ORTHO_CELL)
    assert clean.log_rms_residual < 1e-10
    assert dirty.log_rms_residual > 0.2
    assert dirty.is_valid  # still reported, just with the caveat attached


def test_aniso_b_degrades_on_unusable_input():
    hkl, k = _aniso_problem(np.diag([1.0, 1.0, -2.0]), _ORTHO_CELL, n=20)
    assert not fit_aniso_scale(
        miller_indices=hkl, k_anisotropic=k, unit_cell=_ORTHO_CELL
    ).is_valid  # too few reflections to trust 7 parameters

    hkl, k = _aniso_problem(np.diag([1.0, 1.0, -2.0]), _ORTHO_CELL)
    # a non-positive scale has no logarithm; those reflections must be dropped, not fatal
    k_bad = k.copy()
    k_bad[:5] = -1.0
    k_bad[5:10] = 0.0
    a = fit_aniso_scale(miller_indices=hkl, k_anisotropic=k_bad, unit_cell=_ORTHO_CELL)
    assert a.is_valid
    assert a.n_refl == k.size - 10

    assert not AnisoScale().is_valid
    assert format_aniso_scale_remark_3(AnisoScale()) == []
    assert "unavailable" in format_aniso_scale_summary(AnisoScale())

    with pytest.raises(ValueError):
        fit_aniso_scale(
            miller_indices=hkl[:10], k_anisotropic=k, unit_cell=_ORTHO_CELL
        )


def test_aniso_b_remark_3_uses_the_mandated_pdb_field_names():
    """Deposited output has to match the PDB spelling exactly, not paraphrase it."""
    hkl, k = _aniso_problem(np.diag([2.0, 2.0, -4.0]), _ORTHO_CELL)
    a = fit_aniso_scale(miller_indices=hkl, k_anisotropic=k, unit_cell=_ORTHO_CELL)
    lines = format_aniso_scale_remark_3(a)
    assert lines[0] == "REMARK   3  OVERALL ANISOTROPIC B VALUE."
    for i, name in enumerate(("B11", "B22", "B33", "B12", "B13", "B23")):
        assert lines[i + 1].startswith(f"REMARK   3   {name} (A**2) : ")
    assert lines[1].endswith("2.00000")  # five decimals, as the PDB format specifies
    assert len(lines) == 7
    # A symmetry-fixed component fits as -1e-17; a deposited field must not read
    # "-0.00000", which looks like a measured negative rather than a structural zero.
    for line in lines[1:]:
        assert "-0.00000" not in line, line
    assert "0.00000" in lines[4]


def test_aniso_b_appears_in_the_stats_report_header():
    hkl, k = _aniso_problem(np.diag([2.5, 2.5, -5.0]), _ORTHO_CELL)
    report = IntensityStatsReport(n_refl=100, bin_size=50)
    report.aniso_scale = fit_aniso_scale(
        miller_indices=hkl, k_anisotropic=k, unit_cell=_ORTHO_CELL
    )
    text = format_intensity_stats_report(report)
    assert "Aniso B (scale" in text
    assert "B11=" in text and "B23=" in text
    assert "anisotropy=" in text
    # the degeneracy has to be stated where the number is, not only in the docstring
    assert "shared with k_iso" in text

    # and an absent tensor prints nothing rather than a row of NaNs
    report.aniso_scale = None
    assert "Aniso B" not in format_intensity_stats_report(report)


# ============================================================ engine wiring
class _FakeMillerColumn:
    def __init__(self, values):
        self._v = np.asarray(values)

    def data(self):
        return self._v

    def as_numpy_array(self):
        return self._v


class _FakeUnitCell:
    def __init__(self, params):
        self._p = tuple(params)

    def parameters(self):
        return self._p


class _FakeIObs:
    """Enough of a cctbx miller array for the report builders."""

    def __init__(self, i, sig, d, hkl=None, cell=_ORTHO_CELL):
        self._i, self._sig, self._d = i, sig, d
        self._hkl, self._cell = hkl, cell

    def data(self):
        return self._i

    def sigmas(self):
        return self._sig

    def d_spacings(self):
        return _FakeMillerColumn(self._d)

    def indices(self):
        # flex.miller_index iterates as triples, which is what the engine relies on
        return [tuple(int(x) for x in row) for row in self._hkl]

    def unit_cell(self):
        return _FakeUnitCell(self._cell)

    def size(self):
        return int(self._i.size)


def _engine_probe(**overrides):
    """An IntensityFModel with the heavy constructor bypassed."""
    from phridge.client.intensity.engine import IntensityFModel

    p = _cc_problem(n=4000)
    f_model_amp = np.sqrt(np.maximum(p["i_calc"], 0.0))

    class _Probe(IntensityFModel):
        def __init__(self):
            self._i_obs = _FakeIObs(p["intensities"], p["sigmas"], p["d_spacings"])
            self._r_free_flags = _FakeMillerColumn(p["r_free"])

        def f_model(self):
            return _FakeMillerColumn(f_model_amp.astype(np.complex128))

        def r_work(self):
            return 0.1912

        def r_free(self):
            return 0.2233

        def inferred_r_values(self):
            return {
                "r_post_work": 0.1120,
                "r_post_free": 0.1284,
                "r_mode_work": 0.1043,
                "r_mode_free": 0.1201,
                "r_intensity_work": 0.2814,
                "r_intensity_free": 0.3120,
            }

    probe = _Probe()
    for name, value in overrides.items():
        setattr(probe, name, value)
    return probe, p


def test_engine_populates_the_report_agreement_fields():
    probe, _ = _engine_probe()
    report = IntensityStatsReport(n_refl=10, bin_size=10)
    probe._merge_agreement_stats_into_report(report)
    assert report.r_fw_work == 0.1912
    assert report.r_fw_free == 0.2233
    assert report.r_post_work == 0.1120
    assert report.r_mode_free == 0.1201
    assert report.r_intensity_work == 0.2814


def test_engine_builds_the_cc_isig_table_from_f_model():
    """I_calc must be |F_model|^2 on the observation reflection list."""
    probe, p = _engine_probe()
    report = IntensityStatsReport(n_refl=10, bin_size=10)
    probe._attach_cc_isig_table(report)
    assert report.cc_isig is not None
    table = report.cc_isig
    assert table.rows
    work = ~p["r_free"]
    expected = np.corrcoef(p["intensities"][work], p["i_calc"][work])[0, 1]
    assert table.overall.cc_work == pytest.approx(expected, abs=1e-9)
    assert format_intensity_stats_report(report).count("CC_I = corr(I_obs, I_calc)") == 1


def test_engine_report_helpers_never_raise_on_a_broken_fmodel(capsys):
    """Both helpers are reporting-only: a failure must not cost a macro cycle."""
    probe, _ = _engine_probe()

    def boom():
        raise RuntimeError("f_model unavailable")

    probe.f_model = boom
    probe.inferred_r_values = boom
    probe.r_work = boom
    report = IntensityStatsReport(n_refl=10, bin_size=10)
    probe._merge_agreement_stats_into_report(report)
    probe._attach_cc_isig_table(report)
    assert report.cc_isig is None
    assert not np.isfinite(report.r_fw_work)
    assert "CC_I table skipped" in capsys.readouterr().err


def test_engine_flags_a_reflection_count_mismatch():
    probe, p = _engine_probe()
    probe.f_model = lambda: _FakeMillerColumn(np.ones(7, dtype=np.complex128))
    report = IntensityStatsReport(n_refl=10, bin_size=10)
    probe._attach_cc_isig_table(report)
    assert report.cc_isig is None


def _aniso_probe(b_cart_mat, cell=_MONO_CELL, k_scale=1.0):
    """An engine probe whose k_anisotropic array encodes a known Cartesian B."""
    hkl, k = _aniso_problem(b_cart_mat, cell, n=5000, k_scale=k_scale)
    n = hkl.shape[0]
    probe, _ = _engine_probe()
    probe._i_obs = _FakeIObs(
        np.ones(n), np.ones(n), np.linspace(20.0, 2.0, n), hkl=hkl, cell=cell
    )
    probe.k_anisotropic = lambda: k
    return probe


def test_engine_recovers_the_aniso_b_from_the_applied_scale_array():
    """The engine must read the tensor off k_anisotropic, not off the None b_cart."""
    b_true = np.array([[2.2, 0.4, -1.1], [0.4, -0.7, 0.6], [-1.1, 0.6, -1.5]])
    probe = _aniso_probe(b_true)
    a = probe.aniso_scale()
    assert a.is_valid
    assert a.b_cart == pytest.approx((2.2, -0.7, -1.5, 0.4, -1.1, 0.6), abs=1e-8)
    assert a.source == "k_anisotropic"
    # and it lands on the report
    report = IntensityStatsReport(n_refl=10, bin_size=10)
    probe._merge_agreement_stats_into_report(report)
    assert report.aniso_scale is a


def test_engine_caches_the_aniso_b_until_asked_to_refresh():
    """Rescaling changes the tensor, so update_all_scales must refresh the cache."""
    probe = _aniso_probe(np.diag([1.0, 1.0, -2.0]))
    first = probe.aniso_scale()
    assert probe.aniso_scale() is first  # cached, not refitted per caller

    _, k2 = _aniso_problem(np.diag([4.0, 4.0, -8.0]), _MONO_CELL, n=5000)
    probe.k_anisotropic = lambda: k2
    assert probe.aniso_scale() is first  # still the stale one
    refreshed = probe.aniso_scale(refresh=True)
    assert refreshed is not first
    assert refreshed.anisotropy > first.anisotropy * 3


def test_engine_reports_the_aniso_b_in_the_scaling_log(capsys):
    probe = _aniso_probe(np.diag([3.0, 3.0, -6.0]))
    probe._report_aniso_scale()
    out = capsys.readouterr().out
    assert "global scaling: overall anisotropic B" in out
    assert "anisotropy=" in out
    assert "fitted from k_anisotropic" in out


def test_engine_warns_when_a_stored_b_cart_contradicts_the_applied_scale(capsys):
    """A deposited tensor that is not the applied one is exactly what must not slip out."""
    probe = _aniso_probe(np.diag([2.0, 2.0, -4.0]))
    assert probe._aniso_scale_disagreement() == ""  # nothing stored: the normal case

    fitted = probe.aniso_scale()
    probe.__dict__["b_cart"] = tuple(fitted.b_cart)  # agrees
    assert probe._aniso_scale_disagreement() == ""

    probe.__dict__["b_cart"] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # the b_cart=None failure
    assert "max component difference" in probe._aniso_scale_disagreement()
    probe._report_aniso_scale()
    assert "WARNING: stored b_cart disagrees" in capsys.readouterr().out


def test_engine_aniso_b_never_raises_on_a_broken_scale_array(capsys):
    probe, _ = _engine_probe()

    def boom():
        raise RuntimeError("k_anisotropic unavailable")

    probe.k_anisotropic = boom
    assert probe.aniso_scale() is None
    assert "anisotropic B unavailable" in capsys.readouterr().err
    # a report build must survive it too
    report = IntensityStatsReport(n_refl=10, bin_size=10)
    probe._merge_agreement_stats_into_report(report)
    assert report.aniso_scale is None
    probe._report_aniso_scale()  # must not raise


def test_cc_shell_env_knob(monkeypatch):
    from phridge.client.intensity.stats_report import cc_shell_count

    monkeypatch.delenv("PHRIDGE_CC_SHELLS", raising=False)
    assert cc_shell_count() == 8
    monkeypatch.setenv("PHRIDGE_CC_SHELLS", "12")
    assert cc_shell_count() == 12
    # out of range and unparseable fall back to the default
    for bad in ("0", "99", "abc", ""):
        monkeypatch.setenv("PHRIDGE_CC_SHELLS", bad)
        assert cc_shell_count() == 8, bad
