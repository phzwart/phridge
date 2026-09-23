"""Free per-shell ``beta`` and a non-monotone ``sigma_A``.

The claim under test is that the constraint ``beta = 1 - sigma_A^2`` is not neutral. It
forces the line ``E[Z_o|E_C] = sigma_A^2 E_C^2 + beta`` through ``(1, 1)``, which asserts
that the Wilson normalization is exact; when it is not, the only way the constrained fit
can reach the data is by tilting the slope, and the normalization error is laundered into
``sigma_A``. Test 1 encodes exactly that. The rest establish that freeing ``beta`` does not
invent structure where the normalization is right (test 2), that dropping monotonicity buys
a real dip that the old parameterization cannot represent (test 3), and that the legacy path
still behaves exactly as before (tests 5 and 6).

Note that ``sigma_A`` is the *slope* and both ``Z_o`` and ``E_C^2`` are divided by the same
``Sigma_W``, so the slope is invariant to an error in ``Sigma_0`` while the intercept scales
as ``beta / c``. That is why these tests can assert a hard number for ``sigma_A`` but must
compare ``beta`` against the fit's own normalization.
"""

from __future__ import annotations

import importlib.util
import math

import numpy as np
import pytest

from phridge.contrib.intensity_ll import free_beta as FB
from phridge.contrib.intensity_ll import wilson as W
from phridge.models import CrystalSymmetry, ObservationType
from phridge.packing import PackedMiller

requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch required for ml_i_nuisance_fit",
)

_CELL = (68.0, 82.0, 51.0, 90.0, 90.0, 90.0)
_N_BINS = 6


def _hkl_to_d_min(cell, d_min):
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


def _planted(
    sigma_a_of_shell,
    beta_of_shell,
    *,
    d_min=3.1,
    sigma_0=500.0,
    b_wilson=14.0,
    rel_sigma=0.05,
    centric_frac=0.0,
    seed=0,
    n_bins=_N_BINS,
):
    """Data drawn from the Rice model with ``sigma_A`` and ``beta`` planted independently.

    Per reflection, ``E_true = sigma_A E_C + g`` with ``g`` complex normal of variance
    ``beta`` (real normal for centrics), so ``E[|E_true|^2] = sigma_A^2 E_C^2 + beta``
    exactly -- the identity the free fit is supposed to invert. ``beta`` is *not* tied to
    ``1 - sigma_A^2``, which is the whole point: the generator can put the pair anywhere.
    """
    rng = np.random.default_rng(seed)
    hkl = _hkl_to_d_min(_CELL, d_min)
    n = hkl.shape[0]
    s_cart = W.reciprocal_cartesian(_CELL, hkl)
    s_sq = np.sum(s_cart**2, axis=1)
    sw_true = sigma_0 * np.exp(-0.5 * b_wilson * s_sq)

    # the same equal-width s^2 shells the fit will build, so planted and fitted line up
    edges = np.linspace(s_sq.min(), s_sq.max(), n_bins + 1)
    shell = np.clip(np.digitize(s_sq, edges[1:-1], right=False), 0, n_bins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    sa_true = np.asarray([float(sigma_a_of_shell(k, n_bins)) for k in range(n_bins)])
    beta_true = np.asarray([float(beta_of_shell(k, n_bins)) for k in range(n_bins)])

    centric = rng.random(n) < float(centric_frac)
    e_c = np.sqrt(rng.exponential(1.0, n))  # Wilson-distributed model intensities
    sa_r, beta_r = sa_true[shell], beta_true[shell]
    g_re = rng.normal(0.0, 1.0, n)
    g_im = rng.normal(0.0, 1.0, n)
    # acentric: complex noise of total variance beta; centric: real noise of variance beta
    real = sa_r * e_c + np.where(centric, np.sqrt(beta_r), np.sqrt(0.5 * beta_r)) * g_re
    imag = np.where(centric, 0.0, np.sqrt(0.5 * beta_r) * g_im)
    z_true = real**2 + imag**2

    i_true = sw_true * z_true
    sig = np.maximum(rel_sigma * sw_true, 1e-6 * sigma_0)
    i_obs = i_true + rng.normal(0.0, sig)

    crystal = CrystalSymmetry(unit_cell=list(_CELL), space_group_hall="P 1")
    return dict(
        f_calc=PackedMiller(
            crystal=crystal,
            hkl=hkl,
            data=(e_c * np.sqrt(sw_true)).astype(np.complex128),
            observation_type=ObservationType.complex,
        ),
        f_obs=PackedMiller(
            crystal=crystal,
            hkl=hkl,
            data=i_obs,
            sigmas=sig,
            observation_type=ObservationType.intensity,
        ),
        centric=centric,
        n=n,
        n_bins=n_bins,
        shell=shell,
        centers=centers,
        sw_true=sw_true,
        sigma_a_true=sa_true,
        beta_true=beta_true,
    )


def _fit(problem, **kw):
    from phridge.contrib.intensity_ll.ops import ml_i_nuisance_fit

    opts = dict(
        tune_mask=np.ones(problem["n"], dtype=bool),
        epsilon=np.ones(problem["n"]),
        centric=problem["centric"],
        fit_nu=False,
        fit_scale=False,
        sigma_a_mode="bins",
        n_sigma_a_bins=problem["n_bins"],
        fit_sigma_wilson=True,
        wilson_model="isotropic",
    )
    opts.update(kw)
    return ml_i_nuisance_fit(problem["f_calc"], problem["f_obs"], **opts)


def _shells(out):
    p = out["sigma_a_params"]
    return np.asarray(p["bin_sigma_a"]), np.asarray(p["bin_beta"])


def _beta_expected(problem, out):
    """Planted ``beta`` expressed in the normalization the fit actually chose.

    Both ``Z_o`` and ``E_C^2`` are divided by ``Sigma_W``, so an error ``c`` in ``Sigma_0``
    leaves the slope alone and scales the intercept by ``1/c``. Comparing the fitted beta
    against the planted one without this correction would be testing stage 1, not beta.
    """
    sw_fit = np.asarray(out["sigma_wilson"], dtype=np.float64)
    c = float(np.median(sw_fit / problem["sw_true"]))
    return problem["beta_true"] / c, c


# --------------------------------------------------------------- 0: the transform is exact
def test_the_free_beta_transform_is_exact():
    """``rice_inputs`` must reproduce the Rice model with a free intercept, exactly.

    This is what lets beta be fitted without touching the quadrature, the target or the
    gradients: the prior it induces has to integrate to 1 and have second moment
    ``sigma_A^2 E_C^2 + beta`` for an arbitrary (not tied) pair.
    """
    torch = pytest.importorskip("torch")
    from phridge.contrib.intensity_ll.mli import acen_E, cen_E

    grid = torch.linspace(1e-7, 45.0, 1_200_001, dtype=torch.float64)
    huge = torch.tensor(1e12, dtype=torch.float64)
    zero = torch.tensor(0.0, dtype=torch.float64)
    flat = -0.5 * torch.log(2 * torch.pi * huge**2) - 0.5 * ((grid**2 - zero) / huge) ** 2
    d_e = grid[1] - grid[0]

    for centric in (False, True):
        for sigma_a, beta, e_c in [
            (0.85, 0.30, 1.4),
            (0.85, 1.5 * (1 - 0.85**2), 1.4),  # the laundering case
            (0.60, 0.90, 0.3),
            (0.95, 0.05, 3.0),
            (0.20, 0.98, 2.0),
        ]:
            ec_eff, sa_eff = FB.rice_inputs(
                torch.tensor(e_c, dtype=torch.float64),
                torch.tensor(sigma_a, dtype=torch.float64),
                torch.tensor(beta, dtype=torch.float64),
            )
            fun = cen_E if centric else acen_E
            dens = torch.exp(fun(grid, ec_eff, sa_eff, zero, huge)[0] - flat)
            norm = float((dens * d_e).sum())
            second = float((grid**2 * dens * d_e).sum())
            assert norm == pytest.approx(1.0, abs=2e-5), (centric, sigma_a, beta, norm)
            assert second == pytest.approx(sigma_a**2 * e_c**2 + beta, rel=1e-9), (
                centric, sigma_a, beta, second,
            )


# ------------------------------------------------------------------------- 1: laundering
@requires_torch
@pytest.mark.parametrize("regime,rel_sigma", [("strong", 0.03), ("weak", 0.35)])
def test_constrained_beta_launders_normalization_error_into_sigma_a(regime, rel_sigma):
    """beta inflated 1.5x: the constrained fit must bias sigma_A low, the free fit must not.

    This is the reason beta is fitted. Both fits see identical data; the only difference is
    whether the intercept is allowed to be what the data say it is.
    """
    sa_true = 0.85
    beta_true = 1.5 * (1.0 - sa_true**2)
    prob = _planted(
        lambda k, n: sa_true, lambda k, n: beta_true, rel_sigma=rel_sigma, seed=11
    )

    free = _fit(prob, beta_mode="free")
    tied = _fit(prob, beta_mode="constrained")
    sa_free, beta_free = _shells(free)
    sa_tied, _ = _shells(tied)
    beta_exp, _ = _beta_expected(prob, free)

    # the constrained fit cannot reach the inflated intercept, so it tilts the slope
    assert np.mean(sa_tied) < sa_true - 0.04, (regime, np.mean(sa_tied))
    # the free fit recovers the slope, which is invariant to the normalization error
    assert abs(np.mean(sa_free) - sa_true) < 0.04, (regime, np.mean(sa_free))
    # and the intercept, in the normalization the fit itself chose
    assert np.max(np.abs(beta_free - beta_exp)) < 0.06, (regime, beta_free, beta_exp)
    # the free fit must also be the better explanation of the same data
    assert free["tune_nll"] <= tied["tune_nll"] + 1e-9


@requires_torch
def test_free_and_constrained_agree_when_the_normalization_is_right():
    """With beta = 1 - sigma_A^2 truly holding, freeing beta must cost nothing."""
    sa_of = lambda k, n: 0.9 - 0.25 * k / max(n - 1, 1)  # noqa: E731
    prob = _planted(sa_of, lambda k, n: 1.0 - sa_of(k, n) ** 2, rel_sigma=0.05, seed=5)
    sa_free, _ = _shells(_fit(prob, beta_mode="free"))
    sa_tied, _ = _shells(_fit(prob, beta_mode="constrained"))
    assert np.max(np.abs(sa_free - sa_tied)) < 0.05, (sa_free, sa_tied)


# ----------------------------------------------------------------- 2: no invented structure
@requires_torch
def test_free_beta_invents_no_structure_under_correct_normalization():
    """Correct normalization, monotone truth: beta must land on 1 - sigma_A^2 everywhere.

    A free parameter that wanders when it has no reason to would show up here as a beta
    profile that departs from the constrained curve, or as a sigma_A profile that zig-zags.
    """
    sa_of = lambda k, n: 0.95 - 0.40 * k / max(n - 1, 1)  # noqa: E731
    prob = _planted(sa_of, lambda k, n: 1.0 - sa_of(k, n) ** 2, rel_sigma=0.05, seed=7)
    out = _fit(prob, beta_mode="free")
    sa, beta = _shells(out)
    beta_exp, _ = _beta_expected(prob, out)

    assert np.max(np.abs(beta - beta_exp)) < 0.05, (beta, beta_exp)
    # sigma_A^2 + beta is the normalization consistency check and should sit at 1
    assert out["sigma_a_params"]["consistency_max_abs_error"] < 0.08
    # the recovered profile must still be falling: no sign changes in the first difference
    # beyond what noise at this bin count can produce
    diffs = np.diff(sa)
    assert np.count_nonzero(diffs > 0.02) == 0, sa
    assert np.max(np.abs(sa - prob["sigma_a_true"])) < 0.06, (sa, prob["sigma_a_true"])


# ------------------------------------------------------------------ 3: non-monotone recovery
@requires_torch
def test_non_monotone_sigma_a_dip_is_recovered_only_without_the_constraint():
    """A low-resolution dip: the monotone parameterization cannot represent it.

    sigma_A is commonly depressed at low resolution where the bulk-solvent model is
    inadequate. A cumulative-drop profile can only fall, so it has to push that structure
    into neighbouring shells; the smoothed free fit should track it.
    """

    def sa_of(k, n):
        # 0.75 at the lowest shell, rising to 0.90, then falling away
        return float(np.interp(k, [0, 1, 2, n - 1], [0.75, 0.88, 0.90, 0.60]))

    prob = _planted(sa_of, lambda k, n: 1.0 - sa_of(k, n) ** 2, rel_sigma=0.04, seed=3)
    truth = prob["sigma_a_true"]

    free, _ = _shells(_fit(prob, beta_mode="free"))
    mono, _ = _shells(_fit(prob, beta_mode="constrained", sigma_a_shape="monotone"))

    # the monotone fit is non-increasing by construction, so it cannot show the rise
    assert np.all(np.diff(mono) <= 1e-9), mono
    assert abs(mono[0] - truth[0]) > 0.05, (mono[0], truth[0])
    # the free fit reproduces the dip: it rises out of shell 0 and tracks the profile
    assert free[1] > free[0] + 0.02, free
    assert np.max(np.abs(free - truth)) < 0.05, (free, truth)


# ------------------------------------------------------------------------- 4: scale guard
@requires_torch
def test_free_beta_with_a_joint_scale_raises():
    """sigma_A and an overall F_c scale enter the slope only as sigma_A^2 k^2."""
    prob = _planted(lambda k, n: 0.8, lambda k, n: 0.4, seed=1)
    with pytest.raises(ValueError, match="not identifiable"):
        _fit(prob, beta_mode="free", fit_scale=True)
    # the constrained form breaks the degeneracy, so it stays allowed
    _fit(prob, beta_mode="constrained", fit_scale=True)


# ------------------------------------------------------------- 5 & 6: the legacy path stands
@requires_torch
def test_constrained_monotone_path_keeps_its_exact_legacy_invariants():
    """The reproducible path must be reproducible: same parameterization, same counts.

    beta tied to Sigma_W (1 - sigma_A^2) with zero tolerance, sigma_A non-increasing with
    zero tolerance, and one shell parameter rather than two.
    """
    prob = _planted(lambda k, n: 0.85, lambda k, n: 0.4, seed=2)
    out = _fit(prob, beta_mode="constrained", sigma_a_shape="monotone")

    sigma_a = np.asarray(out["sigma_a"])
    beta = np.asarray(out["beta"])
    sw = np.asarray(out["sigma_wilson"])
    assert np.array_equal(beta, sw * (1.0 - np.clip(sigma_a, 0.0, 0.9999) ** 2))

    sa_bins, _ = _shells(out)
    assert np.all(np.diff(sa_bins) <= 0.0), sa_bins
    assert out["sigma_a_params"]["beta_mode"] == "constrained"
    # one parameter per shell, not two, plus Sigma_0 and the scalar B_W from stage 1
    n_wilson = 2
    assert out["p_theta"] == prob["n_bins"] + n_wilson

    free = _fit(prob, beta_mode="free")
    n_tensor = int(free["sigma_a_params"].get("n_tensor_params") or 0)
    assert free["p_theta"] == 2 * prob["n_bins"] + n_wilson + n_tensor
    assert n_tensor == 10  # P 1: two traceless triclinic tensors


@requires_torch
def test_constrained_monotone_path_reproduces_pre_change_output_bit_for_bit():
    """Golden values recorded from the code as it stood before beta was freed.

    Verified at the time by running this exact synthetic case through a git worktree of the
    previous commit: every output array (sigma_a, sigma_wilson, beta, p_theta, tune_nll,
    scale_k, bin_sigma_a, k, b_delta) compared equal with ``np.array_equal``. These
    constants pin that agreement so the reproducible path cannot drift silently.

    Stage 1 is pinned isotropic because the anisotropic Sigma_W default is a separate
    change; this test is about stage 2.
    """
    prob = _planted(lambda k, n: 0.85, lambda k, n: 0.4, seed=2)
    out = _fit(
        prob,
        beta_mode="constrained",
        sigma_a_shape="monotone",
        wilson_model="isotropic",
    )
    golden = [
        0.8070993818325408,
        0.8070993810065095,
        0.8070993810065039,
        0.8070993810065039,
        0.8070993810065039,
        0.8070993810065039,
    ]
    sa_bins, _ = _shells(out)
    np.testing.assert_allclose(sa_bins, golden, rtol=0.0, atol=1e-12)
    assert out["tune_nll"] == pytest.approx(0.8195873003991764, abs=1e-12)
    assert out["sigma_a_params"]["k"] == pytest.approx(0.6514094121544696, abs=1e-12)
    assert out["p_theta"] == prob["n_bins"] + 2


@requires_torch
def test_read_mode_falls_back_to_constrained_beta_and_says_so():
    """A Read-style sigma_A curve has no shells, so there is nowhere to hang a free beta."""
    prob = _planted(lambda k, n: 0.8, lambda k, n: 0.5, seed=4)
    out = _fit(prob, sigma_a_mode="read", beta_mode="free")
    sigma_a = np.asarray(out["sigma_a"])
    sw = np.asarray(out["sigma_wilson"])
    assert np.array_equal(
        np.asarray(out["beta"]), sw * (1.0 - np.clip(sigma_a, 0.0, 0.9999) ** 2)
    )
    assert "no resolution shells" in out["sigma_a_params"]["beta_fallback"]


# ------------------------------------------------------------------ options and reporting
def test_shell_options_reject_unknown_keys_and_values():
    with pytest.raises(Exception):
        FB.ShellFitOptions(beta_mode="free", bogus=1)
    with pytest.raises(Exception):
        FB.ShellFitOptions(beta_mode="sort_of_free")
    with pytest.raises(Exception):
        FB.ShellFitOptions(lambda_u=-1.0)
    assert FB.ShellFitOptions().beta_mode == "free"
    assert FB.ShellFitOptions().sigma_a_shape == "free"
    assert FB.ShellFitOptions().lambda_consistency == 0.0  # prior is off by default
    for value in (None, "", "nonsense", "free"):
        assert FB.normalize_beta_mode(value) == "free"
    for value in ("constrained", "CLASSICAL", " tied "):
        assert FB.normalize_beta_mode(value) == "constrained"
    for value in ("monotone", "MONOTONIC", " decreasing "):
        assert FB.normalize_sigma_a_shape(value) == "monotone"


def test_moment_initializer_recovers_slope_and_intercept():
    """The initializer is a weighted regression of Z_o on E_C^2; it must find the pair."""
    rng = np.random.default_rng(0)
    n, k = 60_000, 3
    shell = rng.integers(0, k, n)
    sa_true = np.array([0.9, 0.7, 0.4])
    beta_true = np.array([0.5, 0.8, 0.95])
    e_c = np.sqrt(rng.exponential(1.0, n))
    sa_r, b_r = sa_true[shell], beta_true[shell]
    z = (sa_r * e_c + np.sqrt(0.5 * b_r) * rng.normal(size=n)) ** 2 + (
        np.sqrt(0.5 * b_r) * rng.normal(size=n)
    ) ** 2
    sa_hat, beta_hat = FB.moment_init_shells(z, e_c**2, np.full(n, 0.05), shell, k)
    assert np.max(np.abs(sa_hat - sa_true)) < 0.05, (sa_hat, sa_true)
    assert np.max(np.abs(beta_hat - beta_true)) < 0.05, (beta_hat, beta_true)

    # a shell with no reflections, and one with no spread in E_C^2, must fall back quietly
    sa_e, beta_e = FB.moment_init_shells(z[:4], e_c[:4] ** 2, np.full(4, 0.1), shell[:4], k)
    assert np.all(np.isfinite(sa_e)) and np.all(np.isfinite(beta_e))


def test_second_difference_penalty_charges_curvature_not_slope():
    """A straight profile must be free, so a real trend in sigma_A is not pulled flat."""
    straight = np.linspace(-2.0, 2.0, 7)
    assert FB.second_difference_penalty(straight) == pytest.approx(0.0, abs=1e-24)
    kinked = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    assert FB.second_difference_penalty(kinked) > 1.0
    assert FB.second_difference_penalty(np.array([1.0, 2.0])) == pytest.approx(0.0)


# ------------------------------------------------- the fitted beta must reach target + maps
def _obs_for(prob, out, beta_residual):
    torch = pytest.importorskip("torch")
    from phridge.sfcalc.targets.base import Observations

    n = prob["n"]
    return Observations(
        data=torch.as_tensor(np.asarray(prob["f_obs"].data), dtype=torch.float64),
        sigmas=torch.as_tensor(np.asarray(prob["f_obs"].sigmas), dtype=torch.float64),
        epsilon=torch.ones(n, dtype=torch.float64),
        centric=torch.as_tensor(prob["centric"]),
        alpha=torch.as_tensor(out["sigma_a"], dtype=torch.float64),
        beta=torch.as_tensor(out["sigma_wilson"], dtype=torch.float64),
        beta_residual=(
            None
            if beta_residual is None
            else torch.as_tensor(beta_residual, dtype=torch.float64)
        ),
    )


def _laundering_problem(seed=11):
    sa = 0.85
    return _planted(lambda k, n: sa, lambda k, n: 1.5 * (1 - sa**2), rel_sigma=0.05, seed=seed)


@requires_torch
def test_the_fitted_beta_reaches_the_target_the_gradients_and_the_maps():
    """A β that only shows up in the report is a diagnostic; it has to change the model.

    The prior variance enters the NLL, the gradient w.r.t. F_calc and the posterior behind
    the map coefficients, so a correctly plumbed free β must move all three.
    """
    torch = pytest.importorskip("torch")
    from phridge.contrib.intensity_ll.maps import intensity_map_coefficients
    from phridge.contrib.intensity_ll.target import IntensityLogLikelihood

    prob = _laundering_problem()
    out = _fit(prob, beta_mode="free")
    br = np.asarray(out["beta_residual"])
    # the fit found a residual variance genuinely different from the tied value
    tied_value = 1.0 - np.asarray(out["sigma_a"]) ** 2
    assert np.median(np.abs(br - tied_value)) > 0.05, (br.mean(), tied_value.mean())

    fc = torch.as_tensor(np.asarray(prob["f_calc"].data), dtype=torch.complex128)
    tgt = IntensityLogLikelihood()

    def measure(beta_residual):
        obs = _obs_for(prob, out, beta_residual)
        nll = float(tgt.per_reflection(fc, obs).mean())
        amp = fc.abs().clone().requires_grad_(True)
        phase = fc / fc.abs().clamp(min=1e-300)
        grad = torch.autograd.grad(tgt.per_reflection(amp * phase, obs).sum(), amp)[0]
        maps = intensity_map_coefficients(tgt, fc, obs)
        return nll, float(grad.abs().mean()), float(np.mean(maps.fom)), maps

    nll_t, g_t, fom_t, maps_t = measure(None)
    nll_f, g_f, fom_f, maps_f = measure(br)

    # the free prior explains the same data better
    assert nll_f < nll_t - 1e-3, (nll_f, nll_t)
    # and it moves the gradient and the posterior, not just the printed table
    assert abs(g_f / g_t - 1.0) > 0.05, (g_f, g_t)
    assert abs(fom_f - fom_t) > 0.01, (fom_f, fom_t)
    assert np.mean(np.abs(maps_f.difference - maps_t.difference)) > 0.0
    # a larger residual variance means the phases are *less* determined than the tied
    # prior claimed: the constrained form was overconfident
    assert np.mean(br) > np.mean(tied_value)
    assert fom_f < fom_t


@requires_torch
def test_absent_beta_residual_reproduces_the_tied_prior_exactly():
    """Every consumer must fall back to 1 - σ_A² bit-for-bit when β is not supplied."""
    torch = pytest.importorskip("torch")
    from phridge.contrib.intensity_ll.maps import intensity_map_coefficients
    from phridge.contrib.intensity_ll.target import IntensityLogLikelihood
    from phridge.contrib.intensity_ll.mli import normalize

    prob = _laundering_problem(seed=6)
    out = _fit(prob, beta_mode="constrained")
    fc = torch.as_tensor(np.asarray(prob["f_calc"].data), dtype=torch.complex128)
    tgt = IntensityLogLikelihood()
    obs = _obs_for(prob, out, None)

    # normalized() must be the identity on the quadrature inputs, and report jac == 1
    fo, sig = obs.data, obs.sigmas
    eps = obs.epsilon
    sW = torch.as_tensor(out["sigma_wilson"], dtype=torch.float64)
    sA = torch.as_tensor(out["sigma_a"], dtype=torch.float64)
    plain = normalize(fc.abs(), fo, sig, eps, sW, sA)
    routed = tgt.normalized(fc.abs(), fo, sig, eps, sW, sA, obs)
    for a, b in zip(plain, routed[:4]):
        assert torch.equal(a, b)
    assert torch.equal(routed[4], torch.ones_like(routed[4]))

    # and a target built with an explicit scalar beta_residual must differ from it
    tgt_free = IntensityLogLikelihood(beta_residual=0.5)
    assert float(tgt_free.per_reflection(fc, obs).mean()) != float(
        tgt.per_reflection(fc, obs).mean()
    )
    assert np.any(
        intensity_map_coefficients(tgt_free, fc, obs).fom
        != intensity_map_coefficients(tgt, fc, obs).fom
    )


@requires_torch
def test_map_gradient_matches_finite_differences_under_a_free_beta():
    """The closed-form score needs the chain rule back to the real E_C.

    ``posterior_moments().score`` is ``d log L / d E_C`` in whatever parameterization it was
    handed, so under the reparameterization it must be multiplied by
    ``dE_C_eff/dE_C = σ_A/sqrt(1-β)``. Dropping that factor biases every map gradient by
    exactly that ratio — measured at ~6% here, which is small enough to go unnoticed
    forever and only detectable against the true likelihood, hence this test.
    """
    torch = pytest.importorskip("torch")
    from phridge.contrib.intensity_ll.maps import intensity_map_coefficients
    from phridge.contrib.intensity_ll.target import IntensityLogLikelihood

    prob = _laundering_problem(seed=13)
    out = _fit(prob, beta_mode="free")
    fc = torch.as_tensor(np.asarray(prob["f_calc"].data), dtype=torch.complex128)
    tgt = IntensityLogLikelihood()

    for label, br in (("tied", None), ("free", out["beta_residual"])):
        obs = _obs_for(prob, out, br)
        maps = intensity_map_coefficients(tgt, fc, obs)
        amp, phase = fc.abs(), fc / fc.abs().clamp(min=1e-300)
        h = 1e-5 * amp.clamp(min=1e-8)
        d_nll = (
            tgt.per_reflection((amp + h) * phase, obs)
            - tgt.per_reflection((amp - h) * phase, obs)
        ) / (2 * h)
        fd = -d_nll.detach().numpy()  # d log L / d|F_c|
        sel = np.isfinite(fd) & (np.abs(fd) > 1e-6)
        ratio = np.abs(np.asarray(maps.gradient))[sel] / np.abs(fd)[sel]
        assert np.percentile(ratio, 1) > 0.995, (label, np.percentile(ratio, 1))
        assert np.percentile(ratio, 99) < 1.005, (label, np.percentile(ratio, 99))


@requires_torch
def test_surrogate_is_fitted_against_the_free_beta_likelihood():
    """The surrogate approximates the exact target, so it must see the same prior.

    It lives in the true ``t = E_C`` coordinate (``amplitude_units`` maps it to ``ml_f``
    through ``F = S E``), while the quadrature runs in the effective pair — so the score it
    records has to match the exact score w.r.t. the real E_C.
    """
    torch = pytest.importorskip("torch")
    from phridge.contrib.intensity_ll.surrogate import fit_surrogate_for_observations
    from phridge.contrib.intensity_ll.target import IntensityLogLikelihood

    prob = _laundering_problem(seed=17)
    out = _fit(prob, beta_mode="free")
    fc = torch.as_tensor(np.asarray(prob["f_calc"].data), dtype=torch.complex128)
    tgt = IntensityLogLikelihood()

    scores = {}
    for label, br in (("tied", None), ("free", out["beta_residual"])):
        obs = _obs_for(prob, out, br)
        fit = fit_surrogate_for_observations(tgt, fc, obs)
        scores[label] = np.asarray(fit.score)
        # the recorded score is d log L / d E_C; check it against the exact one
        amp, phase = fc.abs(), fc / fc.abs().clamp(min=1e-300)
        h = 1e-5 * amp.clamp(min=1e-8)
        d_nll = (
            tgt.per_reflection((amp + h) * phase, obs)
            - tgt.per_reflection((amp - h) * phase, obs)
        ) / (2 * h)
        scale = torch.sqrt(obs.epsilon * torch.as_tensor(out["sigma_wilson"]))
        fd_dt = (-d_nll * scale).detach().numpy()  # chain to E_C: d/dE_C = S d/d|F_c|
        sel = np.isfinite(fd_dt) & (np.abs(fd_dt) > 1e-3)
        ratio = scores[label][sel] / fd_dt[sel]
        assert np.percentile(ratio, 5) > 0.98, (label, np.percentile(ratio, 5))
        assert np.percentile(ratio, 95) < 1.02, (label, np.percentile(ratio, 95))

    # and the two priors genuinely give different surrogates
    assert np.median(np.abs(scores["free"] - scores["tied"])) > 1e-6


@requires_torch
def test_engine_hands_the_fitted_beta_to_every_evaluation():
    """One plumbing point: if it is not in the common eval kwargs it reaches nothing."""
    engine = pytest.importorskip("phridge.client.intensity.engine")

    probe = engine.IntensityFModel.__new__(engine.IntensityFModel)
    probe._i_obs = None
    probe.sigma_a = np.array([0.8, 0.7])
    probe.sigma_wilson = np.array([100.0, 50.0])
    probe._r_free_flags = None

    class _Eps:
        def data(self):
            return self

        def as_double(self):
            return np.array([1.0, 1.0])

    class _Obs:
        def epsilons(self):
            return _Eps()

        def centric_flags(self):
            return _Eps()

    probe._i_obs = _Obs()
    assert "beta_residual" not in probe._common_eval_kwargs()  # absent -> tied prior

    probe.beta_residual = np.array([0.35, 0.40])
    kw = probe._common_eval_kwargs()
    assert "beta_residual" in kw
    np.testing.assert_allclose(kw["beta_residual"], [0.35, 0.40])


def test_engine_defaults_to_free_beta_and_fixes_the_stage_2_scale(monkeypatch):
    """The engine must keep production out of the degenerate state, not just detect it."""
    import io as _io

    engine = pytest.importorskip("phridge.client.intensity.engine")
    probe = engine.IntensityFModel.__new__(engine.IntensityFModel)
    monkeypatch.delenv("PHRIDGE_BETA_MODE", raising=False)
    monkeypatch.delenv("PHRIDGE_SIGMA_A_SHAPE", raising=False)
    assert probe.beta_mode == "free"
    assert probe.sigma_a_shape == "free"

    buf = _io.StringIO()
    assert probe._fix_scale_for_free_beta(True, log=buf) is False
    assert "σ_A²k²" in buf.getvalue()
    # nothing to say when no scale was requested
    quiet = _io.StringIO()
    assert probe._fix_scale_for_free_beta(False, log=quiet) is False
    assert quiet.getvalue() == ""

    # constrained beta breaks the degeneracy, so the scale may stay free
    tied = engine.IntensityFModel.__new__(engine.IntensityFModel)
    monkeypatch.setenv("PHRIDGE_BETA_MODE", "constrained")
    assert tied.beta_mode == "constrained"
    assert tied._fix_scale_for_free_beta(True) is True


def test_report_shows_beta_and_the_normalization_consistency_column():
    from phridge.client.intensity.stats_report import format_shell_nuisance

    params = {
        "mode": "bins",
        "n_bins": 3,
        "bin_centers_s2": [0.01, 0.04, 0.09],
        "bin_sigma_a": [0.90, 0.80, 0.995],
        "bin_beta": [0.30, 0.40, 0.05],
        "sigma_a_se": [0.011, 0.013, None],
        "beta_se": [0.012, 0.014, None],
        "sigma_a_beta_correlation": [-0.41, -0.39, None],
        "beta_mode": "free",
        "sigma_a_shape": "free",
        "lambda_u": 1.0,
        "lambda_v": 1.0,
        "lambda_consistency": 0.0,
        "consistency_mean_abs_error": 0.0567,
        "consistency_max_abs_error": 0.11,
        "bins_at_bound": [2],
    }
    lines = format_shell_nuisance(params)
    text = "\n".join(lines)
    assert "β: free" in text and "λ_u=1" in text and "σ_A: free" in text
    assert "|σ_A²+β−1|" in text and "at bound: 1 shell(s)" in text
    assert "σ_A²+β" in lines[1]
    # 0.90^2 + 0.30 = 1.11 shows up as the consistency for shell 1, and 10 A is 1/sqrt(0.01)
    assert "1.110" in lines[2] and "10.00" in lines[2]
    assert lines[-1].rstrip().endswith("*")  # the pinned shell is flagged
    assert "." in lines[-1]  # and its missing error bars print as dots

    # a fallback note must surface, and an empty/read-mode params must render nothing
    assert "β not fitted" in "\n".join(
        format_shell_nuisance({**params, "beta_fallback": "no resolution shells"})
    )
    tensor_line = "\n".join(
        format_shell_nuisance(
            {
                **params,
                "laue": "-1",
                "lambda_sphericity": 1.0,
                "n_tensor_params": 10,
                "M_A": {"eigenvalues": [0.9, 1.0, 1.1], "delta_aniso": 0.2},
                "M_beta": {"eigenvalues": [0.95, 1.0, 1.05], "delta_aniso": 0.1},
            }
        )
    )
    assert "M_A [-1]" in tensor_line and "M_β" in tensor_line and "λ_sph=1" in tensor_line
    assert format_shell_nuisance({}) == []
    assert format_shell_nuisance({"mode": "read", "k": 0.5}) == []


def test_bound_and_consistency_reporting():
    err = FB.ShellErrors(
        np.array([0.01, 0.02]), np.array([0.03, 0.04]), np.array([-0.5, 0.1]), True
    )
    table = FB.describe_shells(
        centers_s_sq=[0.01, 0.04],
        sigma_a=[0.9, 0.995],  # the second sits within 0.02 of the upper bound
        beta=[0.19, 0.5],
        errors=err,
        options=FB.ShellFitOptions(),
    )
    assert table.consistency[0] == pytest.approx(0.9**2 + 0.19)
    js = table.as_json()
    assert js["n_at_bound"] == 1 and js["bins_at_bound"] == [1]
    assert js["consistency_max_abs_error"] == pytest.approx(
        max(abs(0.9**2 + 0.19 - 1.0), abs(0.995**2 + 0.5 - 1.0))
    )
    assert js["beta_mode"] == "free" and js["errors_ok"] is True
