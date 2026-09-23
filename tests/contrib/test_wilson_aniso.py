"""Anisotropic Wilson normalization: recovery, isotropic limit, and what it buys.

The motivating claim is that an isotropic ``Sigma_W`` mis-scales ``sigma_Z`` per
direction, so ``data%`` fans out within a resolution shell. Test 4 is the one that
encodes that claim; the rest establish that the tensor is fitted correctly and does
not invent anisotropy where there is none.
"""

from __future__ import annotations

import importlib.util
import io
import math

import numpy as np
import pytest

from phridge.contrib.intensity_ll import wilson as W
from phridge.models import CrystalSymmetry, ObservationType
from phridge.packing import PackedMiller

requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch required for ml_i_nuisance_fit",
)

_CELL = (68.0, 82.0, 51.0, 90.0, 90.0, 90.0)
_D_MIN = 1.8


def _hkl_to_d_min(cell, d_min):
    """Every integer triple inside the resolution sphere, minus (0,0,0)."""
    a, b, c = cell[0], cell[1], cell[2]
    hmax, kmax, lmax = (int(math.floor(x / d_min)) for x in (a, b, c))
    h, k, l = np.meshgrid(
        np.arange(-hmax, hmax + 1),
        np.arange(-kmax, kmax + 1),
        np.arange(0, lmax + 1),
        indexing="ij",
    )
    hkl = np.stack([h.ravel(), k.ravel(), l.ravel()], axis=1)
    hkl = hkl[np.any(hkl != 0, axis=1)]
    s_cart = W.reciprocal_cartesian(cell, hkl)
    d = 1.0 / np.maximum(np.linalg.norm(s_cart, axis=1), 1e-12)
    return hkl[d >= d_min].astype(np.int32)


def _synthetic(
    b_cart,
    *,
    cell=_CELL,
    sigma_0=800.0,
    d_min=_D_MIN,
    seed=0,
    centric_frac=0.0,
    rel_sigma=0.06,
    floor_frac=0.02,
):
    """Intensities drawn from a known anisotropic Wilson distribution.

    Acentric reflections get ``Sigma * Exp(1)`` (variance 1 in Z), centrics get
    ``Sigma * chi2_1`` (variance 2). Measurement noise is a realistic mix of a relative
    term and a floor, so weak high-resolution data are genuinely noise-dominated and a
    good fraction of ``I_obs`` comes out negative.
    """
    rng = np.random.default_rng(seed)
    hkl = _hkl_to_d_min(cell, d_min)
    n = hkl.shape[0]
    s_cart = W.reciprocal_cartesian(cell, hkl)
    sw_true = W.sigma_w_from_b(s_cart=s_cart, sigma_0=sigma_0, b_cart=np.asarray(b_cart))

    centric = rng.random(n) < float(centric_frac)
    z = np.where(centric, rng.normal(0.0, 1.0, n) ** 2, rng.exponential(1.0, n))
    i_true = sw_true * z
    sig = rel_sigma * sw_true + floor_frac * sigma_0 * np.exp(-0.5 * np.trace(b_cart) / 3.0 * 0.25)
    sig = np.maximum(sig, 1e-3 * sigma_0)
    i_obs = i_true + rng.normal(0.0, sig)

    crystal = CrystalSymmetry(unit_cell=list(cell), space_group_hall="P 1")
    f_obs = PackedMiller(
        crystal=crystal,
        hkl=hkl,
        data=i_obs,
        sigmas=sig,
        observation_type=ObservationType.intensity,
    )
    f_calc = PackedMiller(
        crystal=crystal,
        hkl=hkl,
        data=np.sqrt(np.maximum(i_true, 0.0)).astype(np.complex128),
        observation_type=ObservationType.complex,
    )
    return dict(
        f_calc=f_calc,
        f_obs=f_obs,
        hkl=hkl,
        s_cart=s_cart,
        centric=centric,
        sigma_i=sig,
        sw_true=sw_true,
        n=n,
    )


def _fit(problem, wilson_model, **kw):
    from phridge.contrib.intensity_ll.ops import ml_i_nuisance_fit

    n = problem["n"]
    opts = dict(
        tune_mask=np.ones(n, dtype=bool),
        epsilon=np.ones(n),
        centric=problem["centric"],
        fit_nu=False,
        fit_scale=False,
        sigma_a_mode="bins",
        n_sigma_a_bins=8,
        fit_sigma_wilson=True,
        wilson_model=wilson_model,
    )
    opts.update(kw)
    return ml_i_nuisance_fit(problem["f_calc"], problem["f_obs"], **opts)


def _b_from(out):
    p = out["sigma_wilson_params"]
    b = np.asarray(p["b_cart"], dtype=np.float64)
    return np.array([[b[0], b[3], b[4]], [b[3], b[1], b[5]], [b[4], b[5], b[2]]])


# --------------------------------------------------------------------- 1. recovery
@requires_torch
def test_anisotropic_fit_recovers_planted_eigenvalues_and_directions():
    """Eigenvalues to better than 10%, principal directions to within 10 degrees."""
    b_true = np.diag([12.0, 27.0, 19.0])  # dB = 15 A^2 between extreme axes
    p = _synthetic(b_true, seed=1)
    out = _fit(p, "anisotropic")
    params = out["sigma_wilson_params"]

    assert params["wilson_model"] == "anisotropic"
    assert params["n_wilson_params"] == 7
    fitted = np.asarray(params["b_eigenvalues"], dtype=np.float64)
    truth = np.sort(np.linalg.eigvalsh(b_true))
    rel = np.abs(fitted - truth) / truth
    assert np.all(rel < 0.10), (fitted, truth, rel)
    assert params["delta_b_aniso"] == pytest.approx(15.0, rel=0.15)

    # Directions: the planted tensor is diagonal, so each fitted eigenvector must line
    # up with a Cartesian axis, in the order the eigenvalues sort.
    b_fit = _b_from(out)
    _, vecs = np.linalg.eigh(b_fit)
    expected_axis = np.argsort(np.diag(b_true))  # 12->x, 19->z, 27->y
    for i, axis in enumerate(expected_axis):
        cos = abs(vecs[axis, i])
        assert cos > math.cos(math.radians(10.0)), (i, axis, cos)


# --------------------------------------------------------- 2. isotropic limit
@requires_torch
def test_anisotropic_fit_does_not_invent_anisotropy():
    """Planted isotropic: recover B_W to 2% and report dB < 1 A^2."""
    b_w = 22.0
    p = _synthetic(np.eye(3) * b_w, seed=2)
    out = _fit(p, "anisotropic")
    params = out["sigma_wilson_params"]

    assert params["b_iso"] == pytest.approx(b_w, rel=0.02)
    assert params["delta_b_aniso"] < 1.0, params["b_eigenvalues"]
    # b_wilson stays meaningful for isotropic consumers: it is the gauge-fixed trace/3
    assert params["b_wilson"] == pytest.approx(params["b_iso"], rel=1e-9)


# ------------------------------------------------- 3. likelihood improvement
@requires_torch
def test_anisotropic_fit_improves_the_likelihood_beyond_its_parameter_cost():
    """6 extra parameters must buy far more than they cost at these N."""
    b_true = np.diag([12.0, 27.0, 19.0])
    p = _synthetic(b_true, seed=3)
    iso = _fit(p, "isotropic")["sigma_wilson_params"]
    ani = _fit(p, "anisotropic")["sigma_wilson_params"]

    gain = float(iso["wilson_nll"]) - float(ani["wilson_nll"])
    # AIC cost of 6 extra parameters is 6/N nats per reflection -- orders below this.
    cost = 6.0 / p["n"]
    assert gain > 0.01, (iso["wilson_nll"], ani["wilson_nll"], gain)
    assert gain > 100.0 * cost, (gain, cost)
    # and the isotropic fit is the anisotropic one's starting point, so it cannot win
    assert ani["wilson_nll"] <= iso["wilson_nll"] + 1e-9


# ----------------------------------------------- 4. data% direction-independence
@requires_torch
def test_data_frac_spread_across_directions_collapses_under_the_tensor():
    """The motivating test: an isotropic Sigma_W fans data% out by direction.

    Within a single resolution shell, reflections are binned by the angle between their
    reciprocal vector and the principal axis of the planted tensor. Under an isotropic
    normalization sigma_Z is mis-scaled per direction, so the shell's data% depends on
    that angle; under the tensor it must not.
    """
    b_true = np.diag([12.0, 27.0, 19.0])
    p = _synthetic(b_true, seed=4)
    s_cart, sig = p["s_cart"], p["sigma_i"]
    d_star = np.linalg.norm(s_cart, axis=1)

    def direction_spread(sigma_w):
        sz = sig / np.maximum(sigma_w, 1e-30)
        inv = 1.0 / np.maximum(sz, 1e-30) ** 2
        frac = inv / (1.0 + inv)
        # one mid-resolution shell, so resolution is held fixed
        lo, hi = np.percentile(d_star, [45.0, 55.0])
        in_shell = (d_star >= lo) & (d_star <= hi)
        axis = np.linalg.eigh(b_true)[1][:, -1]  # the weakest (largest B) direction
        cos = np.abs(s_cart[in_shell] @ axis) / np.maximum(d_star[in_shell], 1e-30)
        means = []
        for a, b in zip(np.arange(0.0, 1.0, 0.2), np.arange(0.2, 1.2, 0.2)):
            sel = (cos >= a) & (cos < b)
            if np.count_nonzero(sel) > 50:
                means.append(float(np.mean(frac[in_shell][sel])))
        assert len(means) >= 4, means
        return max(means) - min(means)

    sw_iso = np.asarray(_fit(p, "isotropic")["sigma_wilson"], dtype=np.float64)
    sw_ani = np.asarray(_fit(p, "anisotropic")["sigma_wilson"], dtype=np.float64)
    spread_iso = direction_spread(sw_iso)
    spread_ani = direction_spread(sw_ani)

    assert spread_ani * 3.0 < spread_iso, (spread_iso, spread_ani)

    # the op's own per-shell diagnostic must agree with that conclusion
    mean_iso = _fit(p, "isotropic")["sigma_wilson_params"]["data_frac_spread"]["mean_spread"]
    mean_ani = _fit(p, "anisotropic")["sigma_wilson_params"]["data_frac_spread"]["mean_spread"]
    assert mean_ani < mean_iso, (mean_iso, mean_ani)


# ---------------------------------------------------- 5. centric consistency
@requires_torch
def test_centrics_do_not_drive_the_tensor():
    """U must be the same with and without centrics.

    The Wilson mean is ``epsilon*Sigma`` for both parities -- only the variance differs
    (1 vs 2) -- so the centric factor belongs in the likelihood's centric branch and
    *not* in the normalization ``Z = I/(epsilon Sigma)``. If it leaked into the
    normalization, a tensor fit would absorb the acentric/centric imbalance into U and
    excluding centrics would move it. This pins that it does not.
    """
    b_true = np.diag([12.0, 27.0, 19.0])
    mixed = _synthetic(b_true, seed=5, centric_frac=0.25)
    with_cen = _fit(mixed, "anisotropic")

    keep = ~mixed["centric"]
    crystal = mixed["f_obs"].meta.crystal
    acentric_only = dict(mixed)
    acentric_only["f_obs"] = PackedMiller(
        crystal=crystal,
        hkl=mixed["hkl"][keep],
        data=np.asarray(mixed["f_obs"].data)[keep],
        sigmas=np.asarray(mixed["f_obs"].sigmas)[keep],
        observation_type=ObservationType.intensity,
    )
    acentric_only["f_calc"] = PackedMiller(
        crystal=crystal,
        hkl=mixed["hkl"][keep],
        data=np.asarray(mixed["f_calc"].data)[keep],
        observation_type=ObservationType.complex,
    )
    acentric_only["centric"] = mixed["centric"][keep]
    acentric_only["n"] = int(keep.sum())
    without_cen = _fit(acentric_only, "anisotropic")

    e_with = np.asarray(with_cen["sigma_wilson_params"]["b_eigenvalues"])
    e_without = np.asarray(without_cen["sigma_wilson_params"]["b_eigenvalues"])
    assert np.all(np.abs(e_with - e_without) / e_without < 0.10), (e_with, e_without)
    # and both still recover the planted tensor
    truth = np.sort(np.linalg.eigvalsh(b_true))
    assert np.all(np.abs(e_with - truth) / truth < 0.10), (e_with, truth)


# ------------------------------------------------------------------- 6. guard
def test_guard_raises_when_both_anisotropy_carriers_are_free():
    """Anisotropic Sigma_W plus an anisotropic scale is the one fatal combination.

    It fits well and means nothing, so it must raise rather than warn.
    """
    from phridge.client.intensity.stats_report import AnisoScale

    engine = pytest.importorskip("phridge.client.intensity.engine")
    probe = engine.IntensityFModel.__new__(engine.IntensityFModel)

    live = AnisoScale(
        b_cart=(-4.0, -4.0, 8.0, 0.0, 0.0, 0.0),
        principal_b=(-4.0, -4.0, 8.0),
        anisotropy=12.0,
        n_refl=5000,
    )
    probe._aniso_scale = live
    probe._wilson_model = "anisotropic"
    with pytest.raises(ValueError, match="both carry anisotropy"):
        probe._assert_single_anisotropy_carrier()

    # isotropic scale alongside the tensor is the sanctioned combination
    probe._aniso_scale = AnisoScale(
        b_cart=(0.1, 0.1, 0.1, 0.0, 0.0, 0.0),
        principal_b=(0.1, 0.1, 0.1),
        anisotropy=0.02,
        n_refl=5000,
    )
    probe._assert_single_anisotropy_carrier()

    # the binned model carries the anisotropy in its tensor too
    probe._wilson_model = "binned"
    probe._aniso_scale = live
    with pytest.raises(ValueError, match="both carry anisotropy"):
        probe._assert_single_anisotropy_carrier()

    # and an isotropic Wilson model never constrains the scale
    probe._wilson_model = "isotropic"
    probe._assert_single_anisotropy_carrier()


def test_wilson_options_reject_unknown_keys_and_models():
    with pytest.raises(Exception):
        W.WilsonOptions(wilson_model="isotropic", bogus=1)
    with pytest.raises(Exception):
        W.WilsonOptions(wilson_model="sort_of_anisotropic")
    assert W.WilsonOptions(wilson_model="anisotropic").anisotropic
    assert not W.WilsonOptions(wilson_model="isotropic").anisotropic


def test_binned_is_the_default_everywhere():
    """Per-bin means plus a global tensor; the single-curve forms must be asked for."""
    import inspect

    from phridge.contrib.intensity_ll.ops import ml_i_nuisance_fit

    assert W.WilsonOptions().wilson_model == "binned"
    assert W.WilsonOptions().carries_anisotropy
    for value in (None, "", "  ", "nonsense", "binned", "BINS"):
        assert W.normalize_wilson_model(value) == "binned", value
    for value in ("tensor", "aniso", "ANISOTROPIC", "curve"):
        assert W.normalize_wilson_model(value) == "anisotropic", value
    for value in ("isotropic", "ISO", " scalar "):
        assert W.normalize_wilson_model(value) == "isotropic", value
    assert (
        inspect.signature(ml_i_nuisance_fit).parameters["wilson_model"].default
        == "binned"
    )
    assert W.wilson_carries_anisotropy("binned")
    assert W.wilson_carries_anisotropy("anisotropic")
    assert not W.wilson_carries_anisotropy("isotropic")


def test_engine_wilson_model_defaults_to_binned(monkeypatch):
    engine = pytest.importorskip("phridge.client.intensity.engine")
    probe = engine.IntensityFModel.__new__(engine.IntensityFModel)
    monkeypatch.delenv("PHRIDGE_WILSON_MODEL", raising=False)
    assert probe.wilson_model == "binned"

    probe2 = engine.IntensityFModel.__new__(engine.IntensityFModel)
    monkeypatch.setenv("PHRIDGE_WILSON_MODEL", "isotropic")
    assert probe2.wilson_model == "isotropic"


def test_scaling_constraint_turns_off_both_b_cart_flags():
    """anisotropic_scaling alone is not enough — minimization_b_cart also fits the tensor.

    Both live in the real phenix.refine ``bulk_solvent_and_scale`` scope.
    """
    engine = pytest.importorskip("phridge.client.intensity.engine")

    class _Scope:
        anisotropic_scaling = True
        minimization_b_cart = True

    class _Params:
        def __init__(self):
            self.bulk_solvent_and_scale = _Scope()

    probe = engine.IntensityFModel.__new__(engine.IntensityFModel)
    probe._wilson_model = "anisotropic"
    buf = io.StringIO()
    params = probe._constrain_scaling_to_isotropic(_Params(), log=buf)
    assert params.bulk_solvent_and_scale.anisotropic_scaling is False
    assert params.bulk_solvent_and_scale.minimization_b_cart is False
    log = buf.getvalue()
    assert "anisotropic_scaling" in log and "minimization_b_cart" in log

    # an isotropic Wilson model must leave the scaling parameters alone
    probe._wilson_model = "isotropic"
    untouched = probe._constrain_scaling_to_isotropic(_Params())
    assert untouched.bulk_solvent_and_scale.anisotropic_scaling is True

    # and a params object with none of the flags must report that, not pretend success
    probe._wilson_model = "anisotropic"
    buf2 = io.StringIO()
    probe._constrain_scaling_to_isotropic(object(), log=buf2)
    assert "no known flag found" in buf2.getvalue()


def test_identifiability_separates_usable_from_degenerate_direction_sets():
    hkl = _hkl_to_d_min(_CELL, 2.5)
    full = W.tensor_identifiability(W.reciprocal_cartesian(_CELL, hkl))
    assert full > 0.1, full  # a real sphere is comfortably determined

    line = np.zeros((2000, 3), dtype=np.int64)
    line[:, 0] = np.arange(1, 2001)
    plane = hkl[hkl[:, 2] == 0]
    for label, subset in (("line", line), ("plane", plane)):
        val = W.tensor_identifiability(W.reciprocal_cartesian(_CELL, subset))
        assert val < W.MIN_TENSOR_IDENTIFIABILITY, (label, val)

    assert W.tensor_identifiability(np.zeros((3, 3))) == 0.0
    assert W.tensor_identifiability(np.zeros((100, 3))) == 0.0


@requires_torch
def test_degenerate_direction_set_falls_back_to_isotropic_and_says_so():
    """An unidentifiable tensor must not be reported as a confident one.

    Reflections along a single axis leave four of the six components unconstrained, so
    the fit has to decline rather than return numbers for them.
    """
    from phridge.contrib.intensity_ll.ops import ml_i_nuisance_fit

    n = 1500
    cell = (50.0, 50.0, 50.0, 90.0, 90.0, 90.0)
    hkl = np.zeros((n, 3), dtype=np.int32)
    hkl[:, 0] = np.linspace(1, 24, n).astype(np.int32)
    s2 = (hkl[:, 0].astype(np.float64) / 50.0) ** 2
    rng = np.random.default_rng(0)
    sw = 200.0 * np.exp(-0.5 * 18.0 * s2)
    i_obs = sw * rng.exponential(1.0, n) + rng.normal(0.0, 0.1 * sw)
    crystal = CrystalSymmetry(unit_cell=list(cell), space_group_hall="P 1")

    out = ml_i_nuisance_fit(
        PackedMiller(
            crystal=crystal, hkl=hkl,
            data=np.sqrt(np.maximum(i_obs, 0.0)).astype(np.complex128),
            observation_type=ObservationType.complex,
        ),
        PackedMiller(
            crystal=crystal, hkl=hkl, data=i_obs, sigmas=0.1 * sw,
            observation_type=ObservationType.intensity,
        ),
        tune_mask=np.ones(n, dtype=bool),
        epsilon=np.ones(n),
        centric=np.zeros(n, dtype=bool),
        fit_nu=False,
        fit_scale=False,
        fit_sigma_wilson=True,
        # the default, stated so the test still means something if the default moves
        wilson_model="anisotropic",
    )
    p = out["sigma_wilson_params"]
    assert p["wilson_model_requested"] == "anisotropic"
    assert p["wilson_model"] == "isotropic"
    assert "cannot determine a tensor" in p["wilson_fallback"]
    assert p["delta_b_aniso"] == 0.0
    assert p["n_wilson_params"] == 2
    assert np.isfinite(p["b_wilson"]) and p["b_wilson"] > 0.0


def test_reciprocal_vectors_match_the_isotropic_s_sq_metric():
    """The tensor basis has to be the same metric the isotropic path uses.

    Raw integer indices would fit a tensor in the wrong basis and the isotropic start
    would not be a start at all.
    """
    from phridge.contrib.intensity_ll.ops import _compute_s_sq

    for cell in (_CELL, (61.0, 74.5, 92.3, 90.0, 104.2, 90.0), (45.2, 58.9, 71.3, 71.1, 84.3, 66.8)):
        hkl = np.array([[1, 0, 0], [0, 3, 0], [0, 0, 5], [2, -3, 4], [-7, 2, 11]])
        s_cart = W.reciprocal_cartesian(cell, hkl)
        assert np.allclose(np.sum(s_cart**2, axis=1), _compute_s_sq(cell, hkl), rtol=1e-12)
        # B = B_W*I in this frame reproduces the isotropic exponent exactly
        iso = W.sigma_w_from_b(s_cart=s_cart, sigma_0=10.0, b_cart=np.eye(3) * 19.0)
        assert np.allclose(iso, 10.0 * np.exp(-0.5 * 19.0 * _compute_s_sq(cell, hkl)), rtol=1e-12)


# ------------------------------------------------------------ binned Σ_W (default)
def _bins_for(problem, per_bin=400):
    return W.equal_count_bins(np.sum(problem["s_cart"] ** 2, axis=1), per_bin=per_bin)


def _bin_mean_z(problem, sigma_w, bins):
    z = np.asarray(problem["f_obs"].data) / np.asarray(sigma_w)
    return np.bincount(bins, weights=z) / np.bincount(bins)


@requires_torch
def test_binned_keeps_every_bin_mean_intensity_fixed():
    """For any fitted B, each bin's mean I/(εΣ_W) is exactly 1."""
    b_true = np.diag([12.0, 27.0, 19.0])
    p = _synthetic(b_true, seed=6)
    bins = _bins_for(p)
    out = _fit(p, "binned", wilson_bins=bins.astype(np.float64))
    params = out["sigma_wilson_params"]

    assert params["wilson_model"] == "binned"
    assert params["n_bins"] == int(bins.max()) + 1
    np.testing.assert_allclose(_bin_mean_z(p, out["sigma_wilson"], bins), 1.0, atol=1e-9)
    # the tensor is traceless: the bins own all isotropic falloff
    assert abs(float(np.trace(_b_from(out)))) < 1e-9
    assert params["n_wilson_params"] == params["n_bins"] + 5


@requires_torch
def test_binned_tensor_recovers_the_directional_part_of_the_planted_b():
    """Only B - tr(B)/3 is identifiable once the bins carry the isotropic falloff."""
    b_true = np.diag([12.0, 27.0, 19.0])
    p = _synthetic(b_true, seed=7)
    out = _fit(p, "binned", wilson_bins=_bins_for(p).astype(np.float64))
    fitted = np.asarray(out["sigma_wilson_params"]["b_eigenvalues"], dtype=np.float64)
    truth = np.sort(np.linalg.eigvalsh(W.traceless(b_true)))  # -7.33, -0.33, 7.67
    assert np.all(np.abs(fitted - truth) < 1.5), (fitted, truth)
    assert out["sigma_wilson_params"]["delta_b_aniso"] == pytest.approx(15.0, rel=0.15)


@requires_torch
def test_binned_without_a_tensor_is_exactly_the_per_bin_mean():
    """fit_sigma_wilson=False keeps B = 0: Σ_W is the caller's own per-bin mean I/ε."""
    p = _synthetic(np.diag([12.0, 27.0, 19.0]), seed=8)
    bins = _bins_for(p)
    out = _fit(p, "binned", wilson_bins=bins.astype(np.float64), fit_sigma_wilson=False)
    i_obs = np.asarray(p["f_obs"].data)
    means = np.bincount(bins, weights=i_obs) / np.bincount(bins)
    np.testing.assert_allclose(out["sigma_wilson"], means[bins], rtol=1e-12)
    assert out["sigma_wilson_params"]["method"] == "bin_means"
    assert out["sigma_wilson_params"]["n_wilson_params"] == out["sigma_wilson_params"]["n_bins"]


@requires_torch
def test_binned_tensor_improves_on_bins_alone_and_on_the_single_curve():
    """The tensor starts at B = 0 (bins alone), so it cannot lose to that start; and the
    bins must do at least as well as one smooth curve on data drawn from that curve."""
    b_true = np.diag([12.0, 27.0, 19.0])
    p = _synthetic(b_true, seed=9)
    bins = _bins_for(p).astype(np.float64)
    bins_only = _fit(p, "binned", wilson_bins=bins, fit_sigma_wilson=False)
    binned = _fit(p, "binned", wilson_bins=bins)
    nll_bins = float(bins_only["sigma_wilson_params"]["wilson_nll"])
    nll_binned = float(binned["sigma_wilson_params"]["wilson_nll"])
    assert nll_binned < nll_bins - 0.01, (nll_bins, nll_binned)
    curve = float(_fit(p, "anisotropic")["sigma_wilson_params"]["wilson_nll"])
    # bins spend more parameters, so allow them a small per-reflection deficit only
    assert nll_binned < curve + 0.01, (curve, nll_binned)


def test_wilson_bins_accept_a_device_tensor():
    """The worker hands array inputs over as device tensors (MPS on a Mac), which
    np.asarray cannot read; the bins must go through the host copy like every input."""
    from phridge.contrib.intensity_ll.ops import _wilson_bin_ids

    class _DeviceTensor:
        def __init__(self, values):
            self._values = np.asarray(values)

        def __array__(self, *args, **kwargs):
            raise TypeError("can't convert mps:0 device type tensor to numpy")

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._values

    s_sq = np.linspace(0.01, 0.3, 12)
    ids = _wilson_bin_ids(_DeviceTensor([0, 0, 0, 2, 2, 2, 5, 5, 5, 7, 7, 7]), s_sq, W)
    assert ids.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]


def test_binned_sigma_w_numpy_twin_and_bins():
    rng = np.random.default_rng(0)
    s = rng.normal(size=(600, 3)) * 0.3
    s_sq = np.sum(s**2, axis=1)
    bins = W.equal_count_bins(s_sq, per_bin=100)
    assert bins.max() + 1 == 6
    counts = np.bincount(bins)
    assert counts.max() - counts.min() <= 1
    assert np.all(np.diff([s_sq[bins == k].max() for k in range(6)]) > 0)
    y = rng.exponential(50.0, 600)
    sw, means = W.binned_sigma_w(
        i_over_eps=y, bin_id=bins, s_cart=s, b_cart=np.diag([30.0, 0.0, 0.0])
    )
    np.testing.assert_allclose(np.bincount(bins, weights=y / sw) / counts, 1.0, rtol=1e-12)
    assert means.shape == (6,)
