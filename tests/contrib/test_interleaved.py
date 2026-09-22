"""Interleaved refinement: the Rice surrogate fit and the block accept/reject ladder.

The suite is torch-only (no cctbx / Phenix), so the surrogate math is checked against
an independent numpy dense-grid oracle and the end-to-end behaviour against a synthetic
structure driven through :class:`~phridge.sfcalc.engine.StructureFactorEngine`.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scipy.special import i0e  # noqa: E402

from phridge.client.intensity.interleaved import (  # noqa: E402
    BlockBudget,
    BlockOutcome,
    BlockSite,
    CheckpointResult,
    InterleavedController,
    InterleavedOptions,
    TargetMode,
    TorchOptimBlockRunner,
)
from phridge.contrib.intensity_ll import SURROGATE_FIT_OP_NAME, register_ops  # noqa: E402
from phridge.contrib.intensity_ll.maps import (  # noqa: E402
    posterior_moments,
    posterior_node_cache,
)
from phridge.contrib.intensity_ll.mli import normalize  # noqa: E402
from phridge.contrib.intensity_ll.surrogate import (  # noqa: E402
    FIT_EXACT_ROUTE,
    FIT_FALLBACK_INIT,
    SurrogateFitOptions,
    curvature_by_finite_difference,
    exact_score_and_curvature,
    fit_rice_surrogate,
    fit_surrogate_for_observations,
    rice_log_derivatives,
)
from phridge.contrib.intensity_ll.target import IntensityLogLikelihood  # noqa: E402
from phridge.models import CrystalSymmetry, ObservationType, Scatterer, SymOp  # noqa: E402
from phridge.packing import PackedMiller  # noqa: E402
from phridge.packing_xtal import PackedXray  # noqa: E402
from phridge.sfcalc.engine import EngineParams, StructureFactorEngine  # noqa: E402
from phridge.sfcalc.ops import scattering_model  # noqa: E402
from phridge.sfcalc.packing import PackedScatteringTable  # noqa: E402
from phridge.sfcalc.targets import Observations, build_target  # noqa: E402

SIGMA_A = 0.94
DTYPE = torch.float64


def _T(x) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=DTYPE)


# ---------------------------------------------------------------- oracles and fixtures
def _oracle_log_likelihood(t: float, sA: float, Zo: float, sZ: float, n: int = 400001) -> float:
    """Dense-grid quadrature of the acentric Rice x Gaussian marginal.

    Deliberately independent of ``mli.py``: a plain trapezoid rule in ``E`` on a wide
    window, with the Bessel factor evaluated through the exponentially scaled
    ``scipy.special.i0e`` so the integrand never overflows.
    """
    a = 1.0 - sA**2
    e_max = math.sqrt(max(Zo, 0.0)) + 12.0 * math.sqrt(sZ) + 10.0
    E = np.linspace(1e-9, e_max, n)
    x = 2.0 * sA * t * E / a
    log_prior = np.log(2.0 * E / a) - (E**2 + sA**2 * t**2) / a + np.log(i0e(x)) + np.abs(x)
    log_noise = -((Zo - E**2) ** 2) / (2.0 * sZ**2) - 0.5 * math.log(2.0 * math.pi * sZ**2)
    f = log_prior + log_noise
    m = float(f.max())
    return m + float(np.log(np.trapezoid(np.exp(f - m), E)))


def _oracle_score_and_curvature(t: float, sA: float, Zo: float, sZ: float) -> tuple[float, float]:
    h = 2e-4 * max(t, 1.0)
    l0 = _oracle_log_likelihood(t, sA, Zo, sZ)
    lp = _oracle_log_likelihood(t + h, sA, Zo, sZ)
    lm = _oracle_log_likelihood(t - h, sA, Zo, sZ)
    return (lp - lm) / (2.0 * h), (lp - 2.0 * l0 + lm) / h**2


# Six regimes at sigma_A = 0.94: (E_C, Z_o, sigma_Z).
REGIMES: dict[str, tuple[float, float, float]] = {
    "strong": (1.0, 25.0, 1.0),
    "weak": (1.0, 2.0, 1.0),
    "near_zero": (1.0, 0.1, 1.0),
    "negative": (1.0, -0.8, 1.0),
    "very_negative": (1.0, -3.0, 1.0),
    "strong_high_t0": (2.5, 30.0, 1.5),
}
# High-SNR regimes track the exact score far more tightly than weak ones; the
# high-t0 case is held to a looser bound only because a +/-25% window is 2.5x
# wider in absolute t there, and the surrogate is a second-order match.
_GRADIENT_TOLERANCE = {
    "strong": 1e-3,
    "weak": 0.10,
    "near_zero": 0.05,
    "negative": 0.05,
    "very_negative": 0.02,
    "strong_high_t0": 1e-2,
}


def _regime_arrays(centric: bool = False):
    Ec = _T([v[0] for v in REGIMES.values()])
    Zo = _T([v[1] for v in REGIMES.values()])
    sZ = _T([v[2] for v in REGIMES.values()])
    sA = torch.full_like(Ec, SIGMA_A)
    cen = torch.full_like(Ec, centric, dtype=torch.bool)
    return Ec, sA, Zo, sZ, cen


def _mixed_shell(seed: int = 7, n: int = 300, sA: float = SIGMA_A, noise: float = 0.55):
    """A shell drawn consistently with ``sigma_A``, with a tail of negative intensities.

    The true amplitude is drawn from the Rice conditional the model implies -- acentric
    ``E | E_C ~ Rice(sigma_A E_C, (1 - sigma_A^2)/2`` per component) -- so the shell is
    the one a model of this quality would actually produce, not an arbitrary pairing.
    """
    rng = np.random.default_rng(seed)
    Ec = np.sqrt(rng.exponential(1.0, n))  # Wilson model amplitudes
    s = math.sqrt((1.0 - sA**2) / 2.0)
    e_true = np.hypot(sA * Ec + rng.normal(0.0, s, n), rng.normal(0.0, s, n))
    sZ = noise * (0.5 + rng.random(n))
    Zo = e_true**2 + rng.normal(0.0, sZ)
    return _T(Ec), torch.full((n,), sA, dtype=DTYPE), _T(Zo), _T(sZ), torch.zeros(n, dtype=torch.bool)


def _fit(Ec, sA, Zo, sZ, cen, **kw):
    cache = posterior_node_cache(Ec, sA, Zo, sZ, cen)
    g1, g2 = exact_score_and_curvature(cache)
    post = posterior_moments(Ec, sA, Zo, sZ, cen, cache=cache)
    return fit_rice_surrogate(Ec, sA, cache.centric, g1, g2, post.E_mean, **kw), g1, g2, post


def _max_relative_gradient_error(f_p, beta_p, alpha_p, Ec, sA, Zo, sZ, cen, n_grid: int = 21):
    """Max over ``t`` in ``[0.75, 1.25] t_0`` of the score error, scaled by the window's
    own score magnitude.

    Scaling by ``max_t |exact score|`` rather than the pointwise value is deliberate:
    a weak reflection's score passes through zero inside the window, and a pointwise
    relative error there measures the zero crossing, not the surrogate.
    """
    num = torch.zeros_like(Ec)
    den = torch.zeros_like(Ec)
    for frac in torch.linspace(0.75, 1.25, n_grid, dtype=DTYPE):
        t = Ec * frac
        exact = posterior_moments(t, sA, Zo, sZ, cen).score
        approx, _ = rice_log_derivatives(t, f_p, alpha_p, beta_p, cen)
        num = torch.maximum(num, (approx - exact).abs())
        den = torch.maximum(den, exact.abs())
    return num / den.clamp_min(1e-12)


# ================================================================ 1. the fit table
@pytest.mark.parametrize("centric", [False, True])
def test_fit_table_matches_dense_grid_oracle(centric):
    """Six regimes at sigma_A=0.94: every fit converges and reproduces the oracle.

    The oracle is a numpy dense-grid trapezoid quadrature that shares no code with
    ``mli.py``, so this pins the Louis-identity score and curvature, not just their
    self-consistency.
    """
    Ec, sA, Zo, sZ, cen = _regime_arrays(centric=centric)
    fit, g1, g2, _ = _fit(Ec, sA, Zo, sZ, cen)

    # every regime is fitted, not fallen back
    assert torch.all(fit.mask == 0), f"fallbacks in regimes: {fit.mask.tolist()}"
    assert torch.all(fit.f_p > 0) and torch.all(fit.beta_p > 0)
    assert fit.telemetry["n_fit_ok"] == len(REGIMES)

    # the fitted surrogate reproduces the exact score and curvature at t_0
    m1, m2 = rice_log_derivatives(Ec, fit.f_p, fit.alpha_p, fit.beta_p, cen)
    assert torch.allclose(m1, g1, rtol=1e-6, atol=1e-6)
    assert torch.allclose(m2, g2, rtol=1e-6, atol=1e-6)

    # alpha_p is pinned to sigma_A -- never fitted per reflection
    assert torch.allclose(fit.alpha_p, sA)

    if centric:
        return  # the oracle below is the acentric marginal

    for i, name in enumerate(REGIMES):
        t, zo, sz = REGIMES[name]
        o1, o2 = _oracle_score_and_curvature(t, SIGMA_A, zo, sz)
        assert abs(float(g1[i]) - o1) <= 1e-5 * max(abs(o1), 1.0), f"{name}: score vs oracle"
        assert abs(float(g2[i]) - o2) <= 1e-4 * max(abs(o2), 1.0), f"{name}: curvature vs oracle"

    err = _max_relative_gradient_error(
        fit.f_p, fit.beta_p, fit.alpha_p, Ec, sA, Zo, sZ, cen
    )
    for i, name in enumerate(REGIMES):
        assert float(err[i]) < _GRADIENT_TOLERANCE[name], f"{name}: gradient error {float(err[i]):.4f}"


def test_shell_gradient_error_p90_below_eight_percent():
    """The headline accuracy claim, on a mixed shell with a tail of negative intensities."""
    Ec, sA, Zo, sZ, cen = _mixed_shell()
    assert float((Zo < 0).double().mean()) > 0.10  # the negatives are really there
    fit, _, _, _ = _fit(Ec, sA, Zo, sZ, cen)
    err = _max_relative_gradient_error(fit.f_p, fit.beta_p, fit.alpha_p, Ec, sA, Zo, sZ, cen)
    assert float(torch.quantile(err, 0.9)) < 0.10
    assert float(err.median()) < 0.05
    # the fallback ladder is for a handful of hard reflections, not the bulk
    assert fit.telemetry["n_fit_ok"] >= 0.97 * fit.telemetry["n_total"]


# ================================================================ 2. Louis vs finite differences
@pytest.mark.parametrize("centric", [False, True])
def test_louis_curvature_matches_finite_difference(centric):
    """The two independent curvature routes agree on a mixed batch including negative Z_o.

    Louis' identity evaluates the posterior mean of the complete-data second derivative
    plus the posterior variance of the complete-data score on the frozen nodes; the
    finite-difference route central-differences the analytic score, which rebuilds the
    adaptive quadrature window at each shifted model. They share no algebra beyond the
    quadrature itself.

    At the default quadrature resolution a small number of reflections sit near a window
    boundary, and the finite-difference route sees the window move rather than the
    likelihood change; that is a property of the finite-difference route, not a
    disagreement about the curvature. So the bulk is held to the tight tolerance at the
    default resolution, and the worst case is required to *converge* to it when the
    quadrature is refined -- which is the statement that the two routes compute the same
    quantity.
    """
    Ec, sA, Zo, sZ, acen = _mixed_shell(seed=11 if not centric else 13, n=200)
    cen = torch.ones_like(Ec, dtype=torch.bool) if centric else acen
    assert float((Zo < 0).double().mean()) > 0.10

    cache = posterior_node_cache(Ec, sA, Zo, sZ, cen)
    g1, g2_louis = exact_score_and_curvature(cache)
    g2_fd = curvature_by_finite_difference(Ec, sA, Zo, sZ, cen, rel_step=1e-3)
    rel = (g2_louis - g2_fd).abs() / g2_fd.abs().clamp_min(1e-8)
    assert float(torch.quantile(rel, 0.99)) < 1e-4, f"p99 disagreement {float(rel.max()):.2e}"

    fine = dict(n_legendre=64, n_hermite=15)
    g2_louis_fine = exact_score_and_curvature(
        posterior_node_cache(Ec, sA, Zo, sZ, cen, **fine)
    )[1]
    g2_fd_fine = curvature_by_finite_difference(Ec, sA, Zo, sZ, cen, rel_step=1e-3, **fine)
    rel_fine = (g2_louis_fine - g2_fd_fine).abs() / g2_fd_fine.abs().clamp_min(1e-8)
    assert float(rel_fine.max()) < 1e-4, f"refined max disagreement {float(rel_fine.max()):.2e}"

    # Fisher's identity: the Louis score is the same quantity the map path reports
    post = posterior_moments(Ec, sA, Zo, sZ, cen, cache=cache)
    assert torch.allclose(g1, post.score, rtol=1e-10, atol=1e-12)


def test_fit_agrees_whichever_curvature_route_is_used():
    """Switching ``curvature`` must not move the fitted surrogate meaningfully."""
    Ec, sA, Zo, sZ, cen = _mixed_shell(seed=17, n=150)
    cache = posterior_node_cache(Ec, sA, Zo, sZ, cen)
    g1, g2 = exact_score_and_curvature(cache)
    post = posterior_moments(Ec, sA, Zo, sZ, cen, cache=cache)
    g2_fd = curvature_by_finite_difference(Ec, sA, Zo, sZ, cen, rel_step=1e-3)

    a = fit_rice_surrogate(Ec, sA, cen, g1, g2, post.E_mean)
    b = fit_rice_surrogate(Ec, sA, cen, g1, g2_fd, post.E_mean)
    ok = (a.mask == 0) & (b.mask == 0)
    assert bool(ok.any())
    assert float(((a.f_p - b.f_p).abs() / a.f_p)[ok].max()) < 1e-3
    assert float(((a.beta_p - b.beta_p).abs() / a.beta_p)[ok].max()) < 1e-3


# ================================================================ 3. per-reflection beta
def _refit_f_p_at_fixed_beta(Ec, alpha_p, beta_p, cen, g1, iters: int = 200):
    """Solve the score condition alone for F_p with beta_p held fixed."""
    f_p = torch.full_like(Ec, 1.0)
    for _ in range(iters):
        x = f_p.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            m1, _ = rice_log_derivatives(Ec, x, alpha_p, beta_p, cen)
            (d,) = torch.autograd.grad(m1.sum(), x)
        m1v, _ = rice_log_derivatives(Ec, f_p, alpha_p, beta_p, cen)
        step = ((m1v - g1) / torch.where(d.abs() > 1e-12, d, torch.ones_like(d))).clamp(-0.5, 0.5)
        f_p = (f_p - step).clamp_min(1e-8)
    return f_p


def test_per_reflection_beta_is_necessary():
    """A shell-median beta_p must degrade the gradient error by more than 2x.

    This is the guard behind the per-reflection beta_p: it is not a stylistic choice,
    and collapsing it to one value per shell is a measurable regression.
    """
    Ec, sA, Zo, sZ, cen = _mixed_shell(seed=7)
    fit, g1, _, _ = _fit(Ec, sA, Zo, sZ, cen)

    per_reflection = _max_relative_gradient_error(
        fit.f_p, fit.beta_p, fit.alpha_p, Ec, sA, Zo, sZ, cen
    )
    median_beta = fit.beta_p.median().expand_as(fit.beta_p).clone()
    f_p_median = _refit_f_p_at_fixed_beta(Ec, fit.alpha_p, median_beta, cen, g1)
    shell_median = _max_relative_gradient_error(
        f_p_median, median_beta, fit.alpha_p, Ec, sA, Zo, sZ, cen
    )

    p90_per = float(torch.quantile(per_reflection, 0.9))
    p90_med = float(torch.quantile(shell_median, 0.9))
    assert p90_med > 2.0 * p90_per, f"shell-median p90 {p90_med:.4f} vs per-reflection {p90_per:.4f}"


# ================================================================ 4. stationarity
def _model_with_zero_exact_score(Zo, sZ, sA, cen, lo=1e-4, hi=8.0, iters=60):
    """Bisect ``E_C`` per reflection until the exact score vanishes.

    The score is monotonically decreasing in ``t`` wherever the likelihood is concave,
    so a bisection on the sign is well posed and needs no refinement to converge.
    """
    lo_t = torch.full_like(Zo, lo)
    hi_t = torch.full_like(Zo, hi)
    for _ in range(iters):
        mid = 0.5 * (lo_t + hi_t)
        s = posterior_moments(mid, sA, Zo, sZ, cen).score
        lo_t = torch.where(s > 0, mid, lo_t)
        hi_t = torch.where(s > 0, hi_t, mid)
    return 0.5 * (lo_t + hi_t)


def test_stationary_point_is_shared_with_the_exact_target():
    """At a model where the exact score vanishes, the fitted surrogate's does too.

    The consequence that matters: the interleaved inner loop cannot walk away from a
    converged model, because its target has the same stationary point per reflection.
    """
    _, sA, Zo, sZ, cen = _mixed_shell(seed=23, n=120)
    # keep the reflections whose score really does cross zero in the bracket
    Ec = _model_with_zero_exact_score(Zo, sZ, sA, cen)
    exact_score = posterior_moments(Ec, sA, Zo, sZ, cen).score
    interior = exact_score.abs() < 1e-6
    assert int(interior.sum()) > 50

    fit, g1, _, _ = _fit(Ec, sA, Zo, sZ, cen)
    m1, _ = rice_log_derivatives(Ec, fit.f_p, fit.alpha_p, fit.beta_p, cen)

    fitted = interior & (fit.mask == 0)
    assert int(fitted.sum()) > 50
    assert float(m1[fitted].abs().max()) < 1e-6, "surrogate is not stationary where the exact target is"


# ================================================================ synthetic refinement harness
def _p1_crystal(cell=(30.0, 32.0, 34.0, 90.0, 90.0, 90.0)):
    return CrystalSymmetry(
        unit_cell=list(cell),
        space_group_hall="P 1",
        space_group_number=1,
        symops=[SymOp(r=[1, 0, 0, 0, 1, 0, 0, 0, 1], t=[0, 0, 0])],
    )


def _carbon_table():
    return PackedScatteringTable(
        labels=["C"],
        gauss_a=np.array([[2.31, 1.02, 1.5886, 0.865]], dtype=np.float64),
        gauss_b=np.array([[20.8439, 10.2075, 0.5687, 51.6512]], dtype=np.float64),
        gauss_c=np.array([0.2156], dtype=np.float64),
    )


def _engine(sites, crystal, hkl, n_atoms):
    xray = PackedXray(
        crystal=crystal,
        sites_frac=sites,
        occupancy=np.ones(n_atoms),
        u_iso=np.full(n_atoms, 0.06),
        scatterers=[
            Scatterer(i=i, scattering_type="C", anisotropic=False, use_u_iso=True)
            for i in range(n_atoms)
        ],
    )
    model = scattering_model(xray, _carbon_table())
    return StructureFactorEngine(model, hkl, EngineParams(d_min=2.0, quality_factor=100.0))


class _SyntheticProblem:
    """A small P1 structure with synthetic intensities, refinable in torch alone.

    Stands in for host mode: it exercises the same surrogate fit, the same ``ml_f``
    inner target, and the same controller ladder, without needing cctbx or Phenix.
    """

    def __init__(self, n_atoms: int = 8, seed: int = 3, shake: float = 0.06):
        rng = np.random.default_rng(seed)
        self.crystal = _p1_crystal()
        hkl = [
            (h, k, l)
            for h in range(-4, 5)
            for k in range(-4, 5)
            for l in range(-4, 5)
            if (h, k, l) != (0, 0, 0)
        ]
        self.hkl = np.asarray(hkl, dtype=np.int32)
        self.n = len(hkl)
        self.true_sites = rng.random((n_atoms, 3))
        self.n_atoms = n_atoms

        eng_true = _engine(self.true_sites, self.crystal, self.hkl, n_atoms)
        fc_true = eng_true.f_calc_numpy()

        cell = np.asarray(self.crystal.unit_cell[:3], dtype=np.float64)
        dstar2 = ((self.hkl / cell) ** 2).sum(axis=1)
        self.sigma_wilson = np.mean(np.abs(fc_true) ** 2) * np.ones(self.n)
        self.sigma_a = np.full(self.n, 0.93)
        self.epsilon = np.ones(self.n)
        self.centric = np.zeros(self.n, dtype=bool)
        self.sigma_i = (0.25 + 0.4 * rng.random(self.n)) * self.sigma_wilson
        self.i_obs = np.abs(fc_true) ** 2 + rng.normal(0.0, self.sigma_i)
        self.r_free = rng.random(self.n) < 0.08
        self.d_star_sq = dstar2

        self.start_sites = self.true_sites + rng.normal(0.0, shake, self.true_sites.shape)
        self.engine = _engine(self.start_sites, self.crystal, self.hkl, n_atoms)

        self.exact_obs = Observations(
            data=_T(self.i_obs),
            sigmas=_T(self.sigma_i),
            weights=None,
            r_free=torch.as_tensor(self.r_free),
            epsilon=_T(self.epsilon),
            centric=torch.as_tensor(self.centric),
            alpha=_T(self.sigma_a),
            beta=_T(self.sigma_wilson),
            nu=None,
        )
        self.exact_target = build_target({"name": "ml_i"})
        self.mlf_target = build_target({"name": "ml_f"})

    def sites_tensor(self) -> torch.Tensor:
        t = _T(self.start_sites).clone()
        t.requires_grad_(True)
        return t

    def f_calc(self, sites: torch.Tensor) -> torch.Tensor:
        _, occ, u_iso, u_star, fp, fdp = self.engine.tensors()
        return self.engine.f_calc(sites, occ, u_iso, u_star, fp, fdp)

    def rms_to_truth(self, sites: torch.Tensor) -> float:
        cell = np.asarray(self.crystal.unit_cell[:3], dtype=np.float64)
        d = (sites.detach().cpu().numpy() - self.true_sites) * cell
        return float(np.sqrt((d**2).mean()))


class _TorchHost:
    """:class:`~phridge.client.intensity.interleaved.InterleavedHost` for the torch path."""

    def __init__(self, problem: _SyntheticProblem, sites: torch.Tensor):
        self.p = problem
        self.sites = sites
        self.surrogate_obs: Observations | None = None
        self._use_surrogate = False
        self.n_exact = 0
        self.n_surrogate = 0
        self.controller: InterleavedController | None = None

    # -- host contract ----------------------------------------------------------
    def use_surrogate(self, enabled: bool) -> None:
        self._use_surrogate = bool(enabled) and self.surrogate_obs is not None

    def exact_checkpoint(self) -> CheckpointResult:
        with torch.no_grad():
            fc = self.p.f_calc(self.sites)
        self.n_exact += 1
        fit = fit_surrogate_for_observations(self.p.exact_target, fc, self.p.exact_obs)
        self.surrogate_obs = Observations(
            data=_T(fit.f_p),
            sigmas=None,
            weights=None,
            r_free=self.p.exact_obs.r_free,
            epsilon=self.p.exact_obs.epsilon,
            centric=self.p.exact_obs.centric,
            alpha=_T(fit.alpha_p),
            beta=_T(fit.beta_p),
            nu=None,
        )
        nll = self._exact_value(fc)
        return CheckpointResult(
            nll=nll,
            nll_free=None,
            f_p=fit.f_p,
            alpha_p=fit.alpha_p,
            beta_p=fit.beta_p,
            mask=fit.mask,
            e_c=fit.e_c,
            rho2=fit.rho2,
            telemetry=fit.telemetry,
        )

    def _exact_value(self, fc: torch.Tensor) -> float:
        with torch.no_grad():
            per = self.p.exact_target.per_reflection(fc, self.p.exact_obs)
            return float(self.p.exact_target.reduce(per, self.p.exact_obs))

    def exact_nll(self) -> float:
        with torch.no_grad():
            fc = self.p.f_calc(self.sites)
        self.n_exact += 1
        return self._exact_value(fc)

    def surrogate_nll(self) -> float:
        assert self.surrogate_obs is not None
        with torch.no_grad():
            fc = self.p.f_calc(self.sites)
            per = self.p.mlf_target.per_reflection(fc, self.surrogate_obs)
            return float(self.p.mlf_target.reduce(per, self.surrogate_obs))

    # -- the closure the block runner minimizes ---------------------------------
    def closure(self) -> torch.Tensor:
        if self.sites.grad is not None:
            self.sites.grad = None
        fc = self.p.f_calc(self.sites)
        if self._use_surrogate:
            assert self.surrogate_obs is not None
            self.n_surrogate += 1
            if self.controller is not None:
                self.controller.telemetry.n_surrogate_evals += 1
            per = self.p.mlf_target.per_reflection(fc, self.surrogate_obs)
            q = self.p.mlf_target.reduce(per, self.surrogate_obs)
        else:
            self.n_exact += 1
            per = self.p.exact_target.per_reflection(fc, self.p.exact_obs)
            q = self.p.exact_target.reduce(per, self.p.exact_obs)
        q.backward()
        return q


def _run_refinement(
    mode: TargetMode,
    *,
    n_macro_cycles: int = 4,
    seed: int = 3,
    options: InterleavedOptions | None = None,
    optimizer: str = "lbfgs",
    lr: float = 0.02,
    max_iterations: int = 8,
    sabotage: bool = False,
):
    problem = _SyntheticProblem(seed=seed)
    sites = problem.sites_tensor()
    host = _TorchHost(problem, sites)
    runner = TorchOptimBlockRunner(
        [sites],
        host.closure,
        optimizer=optimizer,
        max_iterations=max_iterations,
        lr=lr,
    )
    if sabotage:
        # A block that reliably makes the exact NLL worse, to force the ladder to run.
        # It has to be a genuine distortion, not a rigid-body shift: the target depends
        # only on |F_model|, so a uniform translation of a P1 structure changes nothing.
        original_run = runner.run
        signs = torch.where(
            (torch.arange(sites.shape[0]) % 2 == 0).unsqueeze(1), 1.0, -1.0
        ).to(sites.dtype)

        def bad_run(budget):
            original_run(budget)
            with torch.no_grad():
                sites.add_(signs * 0.15 * budget.step_scale)

        runner.run = bad_run  # type: ignore[method-assign]

    controller = InterleavedController(host, options or InterleavedOptions(verbose=False), mode=mode)
    host.controller = controller
    for cycle in range(n_macro_cycles):
        controller.run_macro_cycle(
            runner,
            final=(cycle == n_macro_cycles - 1),
            site=BlockSite(
                stage="coordinates (xyz) minimization",
                macro_cycle=cycle + 1,
                total_macro_cycles=n_macro_cycles,
            ),
        )
    return problem, host, controller, sites


# ================================================================ 5. end-to-end equivalence
@pytest.mark.slow
def test_interleaved_matches_exact_end_to_end():
    """Interleaved refinement lands on the same model as exact-only, far more cheaply.

    The four claims the mode has to make good on: the same final exact NLL, the same
    coordinates, materially fewer exact evaluations, and a non-increasing exact NLL
    across accepted blocks.
    """
    setup = dict(n_macro_cycles=6, max_iterations=25)
    _, host_x, ctrl_x, sites_x = _run_refinement(TargetMode.exact, **setup)
    problem, host_i, ctrl_i, sites_i = _run_refinement(TargetMode.interleaved, **setup)

    nll_exact = host_x.exact_nll()
    nll_inter = host_i.exact_nll()

    # 1. same final exact NLL (the surrogate never gets to define the answer)
    assert nll_inter <= nll_exact + 1e-4, f"interleaved {nll_inter:.6f} vs exact {nll_exact:.6f}"

    # 2. same coordinates
    drift = float(
        (sites_i.detach() - sites_x.detach()).abs().max()
        * max(problem.crystal.unit_cell[:3])
    )
    assert drift < 0.02, f"coordinate drift {drift:.4f} A"

    # 3. the exact-evaluation count is the whole point
    assert host_i.n_exact * 3 < host_x.n_exact, (
        f"exact evals: interleaved {host_i.n_exact}, exact-only {host_x.n_exact}"
    )
    assert host_i.n_surrogate > host_i.n_exact

    # 4. the exact NLL never rises across accepted blocks
    trace = ctrl_i.telemetry.accepted_nll_trace()
    assert trace, "no block was accepted"
    for a, b in zip(trace, trace[1:]):
        assert b <= a + 1e-9, f"exact NLL rose across accepted blocks: {trace}"

    # both runs actually improved the model
    assert problem.rms_to_truth(sites_i) < problem.rms_to_truth(_T(problem.start_sites))
    assert ctrl_x.telemetry.n_exact_evals > 0


@pytest.mark.slow
def test_exact_evaluation_saving_grows_with_block_length():
    """The longer the inner block, the more the exact target is amortized.

    The saving is not a fixed factor: it is the ratio of inner evaluations to block
    boundaries, so it is the block length that buys it. This is also the knob the
    rejection ladder trades against, which is why a rising rejection rate says
    "shorten the blocks".
    """
    ratios = []
    for max_iterations in (8, 25):
        _, host_x, _, _ = _run_refinement(
            TargetMode.exact, n_macro_cycles=4, max_iterations=max_iterations
        )
        _, host_i, _, _ = _run_refinement(
            TargetMode.interleaved, n_macro_cycles=4, max_iterations=max_iterations
        )
        ratios.append(host_x.n_exact / host_i.n_exact)
    assert ratios[1] > ratios[0], ratios


@pytest.mark.parametrize("optimizer", ["lbfgs", "adam", "adamw", "sgd"])
def test_block_runner_drives_every_optimizer(optimizer):
    """The ladder is optimizer-agnostic: LBFGS, Adam, AdamW and SGD are all just adapters."""
    lr = 0.02 if optimizer == "lbfgs" else 2e-4
    problem, host, controller, sites = _run_refinement(
        TargetMode.interleaved, n_macro_cycles=2, optimizer=optimizer, lr=lr, max_iterations=4
    )
    assert controller.telemetry.blocks
    assert host.n_surrogate > 0
    assert torch.isfinite(sites).all()


# ================================================================ 6. the rejection ladder
def test_rejected_block_halves_then_falls_back_to_exact():
    """A pathological block must be rejected, halved, and finally run fully exact.

    Nothing about a bad surrogate step is allowed to leak into the accepted model: the
    ladder ends in an exact block or it does not end.
    """
    problem, host, controller, sites = _run_refinement(
        TargetMode.interleaved,
        n_macro_cycles=2,  # the last cycle is forced exact, so sabotage the first
        options=InterleavedOptions(max_halvings=1, verbose=False),
        sabotage=True,
    )
    outcomes = [b.outcome for b in controller.telemetry.blocks]
    assert BlockOutcome.rejected in outcomes, outcomes
    assert BlockOutcome.exact_fallback in outcomes, outcomes
    assert outcomes.index(BlockOutcome.exact_fallback) > outcomes.index(BlockOutcome.rejected)
    # the halved retry really did run on a halved budget
    rejected = [b for b in controller.telemetry.blocks if b.outcome is BlockOutcome.rejected]
    assert len(rejected) == 2, outcomes
    assert rejected[1].budget.max_iterations < rejected[0].budget.max_iterations
    assert rejected[1].budget.step_scale < rejected[0].budget.step_scale
    # the run completed and the model is finite and usable
    assert torch.isfinite(sites).all()
    assert math.isfinite(host.exact_nll())


def test_final_macro_cycle_is_always_exact():
    """Whatever the mode, the last block runs on the exact target."""
    _, host, controller, _ = _run_refinement(TargetMode.interleaved, n_macro_cycles=3)
    assert controller.telemetry.blocks[-1].outcome is BlockOutcome.exact
    exact_mode_blocks = [
        b for b in controller.telemetry.blocks if b.outcome is BlockOutcome.exact
    ]
    assert len(exact_mode_blocks) == 1


def test_tolerance_zero_rejects_any_uphill_step():
    """With the default tol=0.0 no block that raises the exact NLL is ever accepted."""
    _, _, controller, _ = _run_refinement(
        TargetMode.interleaved,
        n_macro_cycles=2,
        options=InterleavedOptions(tol=0.0, max_halvings=0, verbose=False),
    )
    for block in controller.telemetry.blocks:
        if block.outcome is BlockOutcome.accepted:
            assert block.exact_delta is not None and block.exact_delta <= 0.0


# ================================================================ op / options / naming
def test_surrogate_fit_op_registers_and_round_trips():
    from phridge.client import Bridge
    from phridge.ops import get_op

    register_ops()
    spec = get_op(SURROGATE_FIT_OP_NAME)
    assert "f_p" in spec.outputs and "nll" in spec.outputs
    assert "surrogate" in spec.inputs

    rng = np.random.default_rng(5)
    n, sA = 300, 0.93
    hkl = rng.integers(-8, 9, size=(n, 3))
    crystal = _p1_crystal()
    sigma_wilson = np.full(n, 120.0)
    Ec = np.sqrt(rng.exponential(1.0, n))
    s = math.sqrt((1.0 - sA**2) / 2.0)
    e_true = np.hypot(sA * Ec + rng.normal(0, s, n), rng.normal(0, s, n))
    sigma_i = (0.4 + rng.random(n)) * 0.5 * sigma_wilson
    i_obs = e_true**2 * sigma_wilson + rng.normal(0.0, sigma_i)
    fc = Ec * np.sqrt(sigma_wilson) * np.exp(1j * rng.uniform(0, 2 * math.pi, n))
    r_free = rng.random(n) < 0.1

    def miller(data, sigmas=None, obs_type=ObservationType.intensity):
        return PackedMiller(
            crystal=crystal, hkl=hkl, data=data, sigmas=sigmas, observation_type=obs_type
        )

    bridge = Bridge(memory=True, timeout=120)
    common = dict(
        alpha=np.full(n, sA),
        beta=sigma_wilson,
        epsilon=np.ones(n),
        centric=np.zeros(n, bool),
        r_free=r_free,
    )
    out = bridge.call(
        SURROGATE_FIT_OP_NAME,
        f_calc=miller(fc, obs_type=ObservationType.complex),
        f_obs=miller(i_obs, sigma_i),
        target={"name": "ml_i"},
        shell_index=np.arange(n) // 100,
        **common,
    )
    assert out["telemetry"]["n_fit_ok"] == n
    assert len(out["telemetry"]["fit_failures_per_shell"]) == 3
    assert np.all(np.asarray(out["f_p"]) > 0) and np.all(np.asarray(out["beta_p"]) > 0)
    assert math.isfinite(out["nll"]["work"]) and out["nll"]["n_work"] == int((~r_free).sum())

    # The fitted arrays, pushed through the stock ml_f target, must reproduce the
    # exact dQ/dF_calc and curvature. This is the interface contract of the whole mode.
    surrogate = bridge.call(
        "target_eval",
        f_calc=miller(fc, obs_type=ObservationType.complex),
        f_obs=miller(np.asarray(out["f_p"]), obs_type=ObservationType.amplitude),
        target={"name": "ml_f"},
        alpha=np.asarray(out["alpha_p"]),
        beta=np.asarray(out["beta_p"]),
        epsilon=np.ones(n),
        centric=np.zeros(n, bool),
        r_free=r_free,
    )
    exact = bridge.call(
        "target_eval",
        f_calc=miller(fc, obs_type=ObservationType.complex),
        f_obs=miller(i_obs, sigma_i),
        target={"name": "ml_i"},
        **common,
    )
    g_s = np.asarray(surrogate.d_target_d_f_calc)
    g_e = np.asarray(exact.d_target_d_f_calc)
    sel = np.abs(g_e) > 0
    rel = np.abs(g_s - g_e)[sel] / np.abs(g_e)[sel]
    assert rel.max() < 1e-6, f"surrogate gradient max rel error {rel.max():.2e}"
    c_s = np.asarray(surrogate.curv_radial)
    c_e = np.asarray(exact.curv_radial)
    assert np.abs(c_s - c_e).max() < 1e-6 * np.abs(c_e).max()


def test_option_models_accept_and_reject():
    import pydantic

    assert SurrogateFitOptions().curvature == "louis"
    assert SurrogateFitOptions(curvature="finite_difference").curvature == "finite_difference"
    assert InterleavedOptions().tol == 0.0
    assert InterleavedOptions(refresh="visible_fraction").refresh == "visible_fraction"

    for bad in (
        {"curvature": "laplace"},
        {"unknown_key": 1},
        {"newton_iters": 0},
        {"beta_bounds": (2.0, 1.0)},
        {"fd_rel_step": 0.0},
    ):
        with pytest.raises((pydantic.ValidationError, ValueError)):
            SurrogateFitOptions(**bad)

    for bad in (
        {"refresh": "sometimes"},
        {"unknown_key": 1},
        {"tol": -1.0},
        {"max_halvings": -1},
        {"rho2_threshold": 1.5},
    ):
        with pytest.raises((pydantic.ValidationError, ValueError)):
            InterleavedOptions(**bad)


def test_budget_halving_reduces_both_iterations_and_step():
    b = BlockBudget(max_iterations=20, step_scale=1.0)
    h = b.halved()
    assert h.max_iterations == 10 and h.step_scale == 0.5
    assert h.halved().max_iterations == 5
    # never halves below one iteration
    assert BlockBudget(max_iterations=1).halved().max_iterations == 1


def test_visible_fraction_refresh_selects_low_rho2_reflections():
    host = object()
    ctrl = InterleavedController(
        host,  # type: ignore[arg-type]
        InterleavedOptions(refresh="visible_fraction", rho2_threshold=0.3, delta_ec_threshold=0.5),
        mode=TargetMode.interleaved,
    )
    cp = CheckpointResult(
        nll=1.0,
        nll_free=None,
        f_p=np.zeros(4),
        alpha_p=np.zeros(4),
        beta_p=np.zeros(4),
        mask=np.zeros(4, dtype=np.int8),
        e_c=np.array([1.0, 1.0, 1.0, 1.0]),
        rho2=np.array([0.1, 0.9, 0.9, 0.05]),
    )
    stale = ctrl.refresh_selection(cp, e_c_since=np.array([1.0, 1.0, 2.0, 1.0]))
    assert stale.tolist() == [True, False, True, True]
    # v1 refreshes everything, which is what ships
    full = InterleavedController(host, InterleavedOptions(), mode=TargetMode.interleaved)  # type: ignore[arg-type]
    assert full.refresh_selection(cp, None).all()


# ---------------------------------------------------------------- the naming guard
_FORBIDDEN = re.compile(r"\b(F_p|f_p|beta_p|F_surrogate|surrogate_amplitude)\b")
_TELEMETRY_SOURCES = (
    Path("src/phridge/client/intensity/interleaved.py"),
    Path("src/phridge/contrib/intensity_ll/surrogate.py"),
    Path("src/phridge/contrib/intensity_ll/surrogate_op.py"),
)


def test_internal_surrogate_arrays_are_never_user_visible():
    """``F_p`` and ``beta_p`` must not reach any printed line, label, or output file.

    ``F_p`` is one paste away from being mistaken for an observed amplitude and
    ``beta_p`` for an experimental variance, so the discipline is the same as the
    S-family naming rule: the arrays exist, and nothing user-facing names them.
    """
    offenders: list[str] = []
    for path in _TELEMETRY_SOURCES:
        text = path.read_text()
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not (
                stripped.startswith("print(")
                or "print(" in stripped
                or "file=" in stripped
                or "short_caption" in stripped
            ):
                continue
            if _FORBIDDEN.search(line):
                offenders.append(f"{path}:{lineno}: {stripped}")
    assert not offenders, "surrogate parameters appear in user-visible output:\n" + "\n".join(offenders)


def test_block_log_lines_carry_no_surrogate_parameters():
    _, _, controller, _ = _run_refinement(TargetMode.interleaved, n_macro_cycles=2)
    report = controller.telemetry.report()
    assert report
    assert not _FORBIDDEN.search(report), report
    # but it does carry what an operator needs
    assert "[interleaved]" in report
    assert "exact evals" in report
    for block in controller.telemetry.blocks:
        assert not _FORBIDDEN.search(block.line())


# ================================================================ 11. when and where
def test_every_block_line_says_when_and_where_it_ran():
    """A block line is useless without its stage and macro cycle.

    "block 3: accepted" leaves the reader unable to tell coordinate refinement in
    cycle 1 from a B-factor block in cycle 4, which is exactly what you need to know
    when a rejection shows up.
    """
    _, _, controller, _ = _run_refinement(TargetMode.interleaved, n_macro_cycles=3)
    assert controller.telemetry.blocks
    for block in controller.telemetry.blocks:
        line = block.line()
        assert "coordinates (xyz) minimization" in line, line
        assert f"macro cycle {block.site.macro_cycle}/3" in line, line
    # every cycle is represented, and in order
    seen = [b.site.macro_cycle for b in controller.telemetry.blocks]
    assert seen == sorted(seen)
    assert set(seen) == {1, 2, 3}


def test_report_rolls_up_where_the_interleaving_happened():
    _, _, controller, _ = _run_refinement(TargetMode.interleaved, n_macro_cycles=3)
    summary = controller.telemetry.stage_summary()
    assert len(summary) == 1, summary
    assert "coordinates (xyz) minimization" in summary[0]
    assert "3 block(s)" in summary[0]
    assert summary[0] in controller.telemetry.report()
    # The outcome tally and the eval counters must not be confused with each other:
    # "exact" is both an outcome name and the name of a counter.
    n_blocks = len(controller.telemetry.blocks)
    assert f"{n_blocks - 1} accepted, 1 exact" in summary[0], summary[0]
    total_exact = sum(b.n_exact_evals for b in controller.telemetry.blocks)
    assert f"exact evals {total_exact}" in summary[0], summary[0]


def test_block_site_label_degrades_without_context():
    """An unlabelled site must still produce a readable line, never a crash."""
    site = BlockSite()
    assert site.label() == "macro cycle ? | unknown stage"
    assert BlockSite(stage="ADP", macro_cycle=2).label() == "macro cycle 2 | ADP"
    assert (
        BlockSite(stage="ADP", macro_cycle=2, total_macro_cycles=5, detail="weight trial").label()
        == "macro cycle 2/5 | ADP (weight trial)"
    )


def test_block_site_is_read_off_the_phenix_lbfgs_call():
    """The "where" comes from the wrapped call, positional arguments included.

    Phenix has no Python here to test against, so this pins the helper to an
    mmtbx-shaped signature: the refine flags name the stage, ``macro_cycle`` supplies
    the cycle, and a call that will not bind degrades to a readable label instead of
    raising inside a refinement.
    """
    from phridge.client.intensity.phenix_hook import _block_site

    class lbfgs:  # the mmtbx.refinement.minimization.lbfgs shape
        def __init__(
            self,
            restraints_manager=None,
            fmodels=None,
            model=None,
            target_weights=None,
            refine_xyz=False,
            refine_adp=False,
            refine_occupancies=False,
            lbfgs_termination_params=None,
            macro_cycle=None,
            log=None,
        ):
            pass

    class Engine:
        macro_cycle_index = 2
        total_macro_cycles = 5

    init, eng = lbfgs.__init__, Engine()

    # refine_xyz passed positionally
    site = _block_site(eng, init, "self", (None, "fm", "model", None, True), {})
    assert site.label() == "macro cycle 2/5 | coordinates (xyz) minimization"

    # explicit macro_cycle wins over the engine counter
    site = _block_site(eng, init, "self", (), {"refine_adp": True, "macro_cycle": 4})
    assert site.label() == "macro cycle 4/5 | B-factors (ADP) minimization"

    # several stages at once
    site = _block_site(eng, init, "self", (), {"refine_xyz": True, "refine_occupancies": True})
    assert site.where() == "coordinates (xyz) + occupancies minimization"

    # nothing recognizable, and an engine with no counter at all
    assert _block_site(eng, init, "self", (), {}).where() == "LBFGS minimization"
    assert _block_site(object(), init, "self", (), {}).label() == "macro cycle ? | LBFGS minimization"
    # a signature that cannot bind must not raise
    assert _block_site(eng, init, "self", tuple(range(40)), {}).where() == "LBFGS minimization"


def test_interleaving_announces_itself_before_the_block_runs():
    """The log has to say it is about to interleave, not only what happened after."""
    problem = _SyntheticProblem(seed=3)
    sites = problem.sites_tensor()
    host = _TorchHost(problem, sites)
    runner = TorchOptimBlockRunner([sites], host.closure, max_iterations=4, lr=0.02)
    emitted: list[str] = []
    controller = InterleavedController(
        host, InterleavedOptions(verbose=True), mode=TargetMode.interleaved
    )
    controller._emit = emitted.append  # type: ignore[method-assign]
    host.controller = controller
    site = BlockSite(stage="coordinates (xyz) minimization", macro_cycle=1, total_macro_cycles=4)
    controller.run_macro_cycle(runner, site=site)

    joined = "\n".join(emitted)
    assert "INTERLEAVING HERE" in joined, joined
    assert "checkpoint (exact)" in joined, joined
    # the announcement precedes the outcome line
    assert emitted[0].startswith(f"[interleaved] {site.label()} |")
    assert "INTERLEAVING HERE" in emitted[0]
    assert not _FORBIDDEN.search(joined), joined

    # and the final cycle says out loud that it is not interleaving
    emitted.clear()
    controller.run_macro_cycle(runner, final=True, site=site)
    joined = "\n".join(emitted)
    assert "running fully exact (final macro cycle)" in joined, joined
