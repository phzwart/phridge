"""Torch FFT structure-factor engine and targets against cctbx reference implementations."""

import numpy as np
import pytest

cctbx = pytest.importorskip("cctbx")
from cctbx import sgtbx, xray  # noqa: E402
from cctbx.array_family import flex  # noqa: E402
from cctbx.development import random_structure  # noqa: E402

torch = pytest.importorskip("torch")

from phridge.client import Bridge  # noqa: E402
from phridge.client.convert import crystal_from_cctbx  # noqa: E402
from phridge.client.convert_xtal import scattering_table_from_cctbx, xray_from_cctbx  # noqa: E402
from phridge.client.xtal_engine import (  # noqa: E402
    RemoteRefinementTarget,
    RemoteStructureFactors,
    RemoteTargetFunctor,
)
from phridge.models import SfEngineParams  # noqa: E402
from phridge.worker.ops.xtal_ops import scattering_model  # noqa: E402
from phridge.worker.targets import LeastSquares, MaximumLikelihoodAmplitude, Observations  # noqa: E402
from phridge.worker.xtal.engine import EngineParams, StructureFactorEngine  # noqa: E402


def _structure(space_group="P21", elements=("C", "N", "O", "S"), n_repeat=3, aniso=False, anomalous=False, seed=0):
    import random

    random.seed(seed)
    flex.set_random_seed(seed)
    xs = random_structure.xray_structure(
        space_group_info=sgtbx.space_group_info(space_group),
        elements=list(elements) * n_repeat,
        volume_per_atom=50,
        random_u_iso=True,
        random_occupancy=True,
        use_u_aniso=aniso,
    )
    for sc in xs.scatterers():
        if anomalous:
            sc.fp = 0.3
            sc.fdp = 0.7
        sc.flags.set_grad_site(True)
        sc.flags.set_grad_occupancy(True)
        sc.flags.set_grad_u_iso(sc.flags.use_u_iso())
        sc.flags.set_grad_u_aniso(sc.flags.use_u_aniso())
        sc.flags.set_grad_fp(anomalous)
        sc.flags.set_grad_fdp(anomalous)
    return xs


def _engine(xs, hkl, d_min, quality_factor=1000):
    model = scattering_model(xray_from_cctbx(xs), scattering_table_from_cctbx(xs))
    return StructureFactorEngine(model, hkl, EngineParams(d_min=d_min, quality_factor=quality_factor))


def _rel(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.abs(a - b).max() / max(np.abs(a).max(), 1e-300)


# ----------------------------------------------------------------- F_calc
@pytest.mark.parametrize("space_group,aniso", [("P1", False), ("P21", True), ("C2", False), ("P4132", True), ("R3:H", False)])
def test_f_calc_matches_cctbx_direct(space_group, aniso):
    xs = _structure(space_group, aniso=aniso, anomalous=True)
    d_min = 1.6
    fc = xs.structure_factors(d_min=d_min, algorithm="direct").f_calc()
    hkl = np.array(list(fc.indices()))
    eng = _engine(xs, hkl, d_min)
    mine = eng.f_calc_numpy()
    ref = np.array(fc.data())
    assert _rel(ref, mine) < 3e-3
    r_factor = np.abs(np.abs(mine) - np.abs(ref)).sum() / np.abs(ref).sum()
    assert r_factor < 1e-3


def test_special_positions_and_centering_weights():
    xs = _structure("P4132", elements=("C", "O"), n_repeat=2)
    model = scattering_model(xray_from_cctbx(xs), scattering_table_from_cctbx(xs))
    ref = np.array([sc.multiplicity() for sc in xs.scatterers()])
    np.testing.assert_array_equal(model.multiplicity, ref)


# -------------------------------------------------------------- gradients
@pytest.mark.parametrize("space_group,aniso", [("P21", False), ("P4132", True), ("R3:H", True)])
def test_gradients_match_cctbx_direct(space_group, aniso):
    xs = _structure(space_group, elements=("C", "N", "O", "S", "Se"), aniso=aniso, anomalous=True)
    d_min = 1.6
    fc = xs.structure_factors(d_min=d_min, algorithm="direct").f_calc()
    hkl = np.array(list(fc.indices()))
    rng = np.random.default_rng(0)
    dtdf = rng.normal(size=len(hkl)) + 1j * rng.normal(size=len(hkl))
    ref = xray.structure_factors.gradients_direct(
        xray_structure=xs,
        u_iso_refinable_params=None,
        miller_set=fc,
        d_target_d_f_calc=flex.complex_double(dtdf),
        n_parameters=0,
    )._results
    mine = _engine(xs, hkl, d_min).gradients(dtdf)
    aniso_mask = np.array([sc.flags.use_u_aniso() for sc in xs.scatterers()])
    assert _rel(ref.d_target_d_site_frac(), mine["site_frac"]) < 1e-2
    assert _rel(ref.d_target_d_occupancy(), mine["occupancy"]) < 1e-2
    assert _rel(ref.d_target_d_fp(), mine["fp"]) < 1e-2
    assert _rel(ref.d_target_d_fdp(), mine["fdp"]) < 1e-2
    if aniso:
        assert _rel(ref.d_target_d_u_star(), mine["u_star"][aniso_mask]) < 3e-2
        assert np.all(mine["u_iso"][aniso_mask] == 0)
    else:
        assert _rel(ref.d_target_d_u_iso(), mine["u_iso"][~aniso_mask]) < 2e-2


def test_gradient_convention_finite_difference():
    """dQ/dp = sum_h Re[conj(G_h) dF_h/dp] for Q = sum_h Re[conj(G_h) F_h] (exact chain rule on the FFT model)."""
    xs = _structure("P21", n_repeat=1)
    d_min = 2.0
    fc = xs.structure_factors(d_min=d_min, algorithm="direct").f_calc()
    hkl = np.array(list(fc.indices()))
    eng = _engine(xs, hkl, d_min)
    rng = np.random.default_rng(1)
    g = rng.normal(size=len(hkl)) + 1j * rng.normal(size=len(hkl))
    grads = eng.gradients(g)

    def q_of(sites):
        params = list(eng.tensors())
        params[0] = torch.as_tensor(sites, dtype=torch.float64)
        with torch.no_grad():
            f = eng.f_calc(*params).numpy()
        return float(np.sum(np.real(np.conj(g) * f)))

    sites = eng.model.sites_frac.copy()
    h = 1e-5
    for j in range(3):
        plus = sites.copy()
        plus[0, j] += h
        minus = sites.copy()
        minus[0, j] -= h
        fd = (q_of(plus) - q_of(minus)) / (2 * h)
        assert abs(fd - grads["site_frac"][0, j]) < 1e-4 * max(1.0, abs(fd))


# ---------------------------------------------------------------- targets
def _obs_setup(seed=1):
    xs = _structure("P212121", n_repeat=4, seed=seed)
    fc = xs.structure_factors(d_min=2.0, algorithm="direct").f_calc()
    xs2 = xs.deep_copy_scatterers()
    xs2.shake_sites_in_place(rms_difference=0.3)
    fc2 = xs2.structure_factors(d_min=2.0, algorithm="direct").f_calc()
    f_obs = fc.amplitudes()
    f_obs = f_obs.customized_copy(data=f_obs.data() * 1.7)
    return xs2, fc2, f_obs


@pytest.mark.parametrize("obs_type", ["F", "I"])
def test_least_squares_matches_cctbx(obs_type):
    from cctbx.xray import ext

    _, fc2, f_obs = _obs_setup()
    rng = np.random.default_rng(0)
    n = f_obs.size()
    yobs = f_obs.data() if obs_type == "F" else flex.pow2(f_obs.data())
    w = flex.double(rng.random(n) + 0.5)
    cls = ext.targets_least_squares_residual if obs_type == "F" else ext.targets_least_squares_residual_for_intensity
    ref = cls(yobs, w, fc2.data(), True, 0.0)
    obs = Observations.from_numpy(data=np.array(yobs), weights=np.array(w))
    ev = LeastSquares(obs_type=obs_type, compute_scale_using_all_data=True).evaluate(torch.as_tensor(np.array(fc2.data())), obs)
    assert abs(ev.value - ref.target()) < 1e-12
    assert abs(ev.scale_factor - ref.scale_factor()) < 1e-12
    assert _rel(np.array(ref.derivatives()), ev.d_target_d_f_calc) < 1e-12


def test_ml_amplitude_matches_cctbx():
    from cctbx.xray import ext

    _, fc2, f_obs = _obs_setup()
    n = f_obs.size()
    rng = np.random.default_rng(0)
    r_free = flex.bool(rng.random(n) < 0.1)
    alpha = flex.double(rng.random(n) * 0.5 + 0.5)
    beta = flex.double(rng.random(n) * 200 + 50)
    eps = f_obs.epsilons().data().as_double()
    cen = f_obs.centric_flags().data()
    k = 1.3
    ref = ext.mlf_target_and_gradients(f_obs.data(), r_free, fc2.data(), alpha, beta, k, eps, cen, True)
    obs = Observations.from_numpy(
        data=np.array(f_obs.data()),
        r_free=np.array(r_free),
        epsilon=np.array(eps),
        centric=np.array(cen),
        alpha=np.array(alpha),
        beta=np.array(beta),
    )
    ev = MaximumLikelihoodAmplitude(scale_factor=k).evaluate(torch.as_tensor(np.array(fc2.data())), obs)
    assert abs(ev.value - ref.target_work()) < 1e-6
    assert abs(ev.value_test - ref.target_test()) < 1e-6
    assert _rel(np.array(ref.gradients_work()), ev.d_target_d_f_calc[~np.array(r_free)]) < 1e-5
    assert np.all(ev.d_target_d_f_calc[np.array(r_free)] == 0)


def test_amplitude_curvature_finite_difference():
    _, fc2, f_obs = _obs_setup()
    n = f_obs.size()
    rng = np.random.default_rng(0)
    obs = Observations.from_numpy(
        data=np.array(f_obs.data()),
        epsilon=np.array(f_obs.epsilons().data().as_double()),
        centric=np.array(f_obs.centric_flags().data()),
        alpha=rng.random(n) * 0.5 + 0.5,
        beta=rng.random(n) * 200 + 50,
    )
    tgt = MaximumLikelihoodAmplitude()
    fc = torch.as_tensor(np.array(fc2.data()))
    ev = tgt.evaluate(fc, obs)
    amp = fc.abs()
    phase = fc / amp
    h = 1e-3

    def gprime(scale):
        f = (amp * scale) * phase
        e = tgt.evaluate(f, obs, compute_curvature=False)
        # radial derivative g'(|F|) = Re[conj(G) e^{i phi}]
        return np.real(np.conj(e.d_target_d_f_calc) * phase.numpy())

    fd = (gprime(1 + h) - gprime(1 - h)) / (2 * h * amp.numpy())
    assert _rel(fd, ev.curv_radial) < 1e-4
    g1 = gprime(1.0)
    assert _rel(g1 / amp.numpy(), ev.curv_tangential) < 1e-10


# ----------------------------------------------------- bridge round trips
def _bridge():
    return Bridge(memory=True, timeout=2)


def test_crystal_symmetry_carries_symops():
    xs = _structure("C2", n_repeat=1)
    cs = crystal_from_cctbx(xs.crystal_symmetry())
    assert len(cs.symops) == xs.space_group().order_z()
    assert cs.model_dump()["symops"][0]["r"] == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


def test_structure_factor_server_f_calc_and_gradients():
    """cctbx-feel StructureFactorServer façade (memory Bridge)."""
    from phridge.client import StructureFactorServer

    sf = StructureFactorServer(memory=True, device="cpu")
    xs = _structure("P21", n_repeat=1, seed=11)
    d_min = 2.0
    ref_fc = xs.structure_factors(d_min=d_min, algorithm="direct").f_calc()
    params = SfEngineParams(d_min=d_min, quality_factor=1000)
    fc = sf.f_calc(xs, ref_fc, params=params)
    assert _rel(np.array(ref_fc.data()), np.array(fc.data())) < 3e-3
    rng = np.random.default_rng(0)
    dtdf = flex.complex_double(rng.normal(size=fc.size()) + 1j * rng.normal(size=fc.size()))
    grads = sf.gradients(xs, ref_fc, dtdf, params=params)
    assert grads.packed().size() == xs.n_parameters()
    f_obs = ref_fc.amplitudes()
    target, packed = sf.target_and_gradients(xs, f_obs, {"name": "ls", "obs_type": "F"}, params=params)
    assert target >= 0.0
    assert packed.size() == xs.n_parameters()


def test_remote_structure_factors_drop_in():
    bridge = _bridge()
    xs = _structure("P21", aniso=True, anomalous=True)
    d_min = 1.8
    ref_fc = xs.structure_factors(d_min=d_min, algorithm="direct").f_calc()
    engine = RemoteStructureFactors(bridge, xs, ref_fc, params=SfEngineParams(d_min=d_min, quality_factor=1000))
    fc = engine.f_calc()
    assert fc.indices().all_eq(ref_fc.indices())
    assert _rel(np.array(ref_fc.data()), np.array(fc.data())) < 3e-3

    rng = np.random.default_rng(0)
    dtdf = flex.complex_double(rng.normal(size=fc.size()) + 1j * rng.normal(size=fc.size()))
    ref = xray.structure_factors.gradients_direct(
        xray_structure=xs,
        u_iso_refinable_params=None,
        miller_set=ref_fc,
        d_target_d_f_calc=dtdf,
        n_parameters=xs.n_parameters(),
    )
    mine = engine.gradients(dtdf)
    packed_ref = np.array(ref.packed())
    packed_mine = np.array(mine.packed())
    assert packed_ref.shape == packed_mine.shape == (xs.n_parameters(),)
    assert _rel(packed_ref, packed_mine) < 3e-2  # u_cart entries dominate the scale


def test_remote_target_functor_and_refinement_target():
    bridge = _bridge()
    xs2, fc2, f_obs = _obs_setup()
    n = f_obs.size()
    rng = np.random.default_rng(0)
    r_free = flex.bool(rng.random(n) < 0.1)
    alpha = flex.double(np.full(n, 0.9))
    beta = flex.double(np.full(n, 100.0))
    functor = RemoteTargetFunctor(bridge, f_obs, {"name": "ml_f"}, r_free_flags=r_free, alpha=alpha, beta=beta)
    from cctbx.xray import ext

    ref = ext.mlf_target_and_gradients(
        f_obs.data(), r_free, fc2.data(), alpha, beta, 1.0,
        f_obs.epsilons().data().as_double(), f_obs.centric_flags().data(), True,
    )
    res = functor(fc2)
    assert abs(res.target_work() - ref.target_work()) < 1e-6
    assert _rel(np.array(ref.gradients_work()), np.array(res.gradients_work())) < 1e-5
    assert res.curvatures_work() is not None

    refiner = RemoteRefinementTarget(
        bridge, xs2, f_obs, {"name": "ml_f"}, params=SfEngineParams(d_min=2.0, quality_factor=1000),
        r_free_flags=r_free, alpha=alpha, beta=beta,
    )
    target, packed = refiner.target_and_gradients()
    assert abs(target - ref.target_work()) < 5e-3 * abs(ref.target_work())
    # chain rule check against cctbx: gradients_direct with the cctbx ML derivatives
    dtdf = flex.complex_double(n, 0j)
    work = ~np.array(r_free)
    gw = np.array(ref.gradients_work())
    full = np.zeros(n, dtype=np.complex128)
    full[work] = gw
    ref_packed = xray.structure_factors.gradients_direct(
        xray_structure=xs2, u_iso_refinable_params=None, miller_set=fc2,
        d_target_d_f_calc=flex.complex_double(full), n_parameters=xs2.n_parameters(),
    ).packed()
    assert _rel(np.array(ref_packed), np.array(packed)) < 2e-2


def test_lbfgs_refinement_with_remote_target():
    """scitbx.lbfgs driving site refinement on the remote LS target (the Phenix-side use pattern)."""
    import scitbx.lbfgs

    bridge = _bridge()
    xs = _structure("P21", n_repeat=2, seed=3)
    fc = xs.structure_factors(d_min=2.0, algorithm="direct").f_calc()
    f_obs = fc.amplitudes()
    xs2 = xs.deep_copy_scatterers()
    xs2.shake_sites_in_place(rms_difference=0.15)
    for sc in xs2.scatterers():
        sc.flags.set_grad_site(True)
        sc.flags.set_grad_occupancy(False)
        sc.flags.set_grad_u_iso(False)
        sc.flags.set_grad_u_aniso(False)
    refiner = RemoteRefinementTarget(bridge, xs2, f_obs, {"name": "ls", "obs_type": "F"}, params=SfEngineParams(d_min=2.0))
    rms_before = xs.rms_difference(xs2)
    t0, _ = refiner.target_and_gradients(xs2)

    class Minimizer:
        def __init__(self):
            self.x = xs2.sites_cart().as_double()
            self.n_calls = 0
            scitbx.lbfgs.run(target_evaluator=self, termination_params=scitbx.lbfgs.termination_parameters(max_iterations=15))

        def compute_functional_and_gradients(self):
            xs2.set_sites_cart(flex.vec3_double(self.x))
            self.n_calls += 1
            target, grad = refiner.target_and_gradients(xs2)
            return target, grad

    Minimizer()
    t1, _ = refiner.target_and_gradients(xs2)
    assert t1 < 0.2 * t0
    assert xs.rms_difference(xs2) < 0.5 * rms_before


def test_gauss_newton_hvp_matches_finite_difference_of_gradients():
    """For LS-on-F near the solution the GN Hessian ~ true Hessian; check (J^T H J) v against FD of the gradient."""
    xs = _structure("P21", n_repeat=1, seed=5)
    d_min = 2.0
    fc = xs.structure_factors(d_min=d_min, algorithm="direct").f_calc()
    hkl = np.array(list(fc.indices()))
    eng = _engine(xs, hkl, d_min)
    f_obs = np.abs(eng.f_calc_numpy()) * 1.0  # exact data: residual term of the Hessian vanishes
    obs = Observations.from_numpy(data=f_obs)
    tgt = LeastSquares(obs_type="F", scale_factor=1.0)

    def grad_at(sites):
        params = list(eng.tensors(requires_grad=True))
        params[0] = torch.tensor(sites, dtype=torch.float64, requires_grad=True)
        with torch.no_grad():
            f = eng.f_calc(*params)
        ev = tgt.evaluate(f, obs)
        return eng.gradients(ev.d_target_d_f_calc, params=tuple(params))["site_frac"], ev

    sites = eng.model.sites_frac.copy()
    _, ev = grad_at(sites)
    rng = np.random.default_rng(0)
    v_site = rng.normal(size=sites.shape) * 1e-2
    zeros = [np.zeros_like(eng.model.occupancy), np.zeros_like(eng.model.u_iso), np.zeros_like(eng.model.u_star), np.zeros_like(eng.model.fp), np.zeros_like(eng.model.fdp)]
    hv = eng.gauss_newton_hvp((v_site, *zeros), ev.curv_radial, ev.curv_tangential)["site_frac"]
    h = 1e-3
    fd = (grad_at(sites + h * v_site)[0] - grad_at(sites - h * v_site)[0]) / (2 * h)
    assert _rel(fd, hv) < 2e-2


def test_gauss_newton_hvp_op_round_trip():
    bridge = _bridge()
    xs2, fc2, f_obs = _obs_setup()
    refiner = RemoteRefinementTarget(bridge, xs2, f_obs, {"name": "ls", "obs_type": "F"}, params=SfEngineParams(d_min=2.0))
    out = refiner.compute()
    from phridge.packing_scattering import PackedSfGradients
    from phridge.client.xtal_engine import _miller_template

    n = xs2.scatterers().size()
    v = PackedSfGradients(
        d_site_frac=np.random.default_rng(0).normal(size=(n, 3)) * 1e-2,
        d_occupancy=np.zeros(n), d_u_iso=np.zeros(n), d_u_star=np.zeros((n, 6)), d_fp=np.zeros(n), d_fdp=np.zeros(n),
    )
    hv = bridge.call(
        "gauss_newton_hvp",
        xray=xray_from_cctbx(xs2), table=scattering_table_from_cctbx(xs2), params=refiner.params,
        target=out["target"], hkl=_miller_template(f_obs), v=v,
    )
    assert hv.d_site_frac.shape == (n, 3)
    # J^T H J is symmetric positive semidefinite for a convex-in-|F| LS target: v . Hv >= 0
    assert float((hv.d_site_frac * v.d_site_frac).sum()) > 0


def _exact_gn_diagonal_sites(eng, curv_radial, curv_tangential):
    """Exact diag(J^T H J) for sites via e_i^T (J^T H J) e_i."""
    sites = eng.model.sites_frac
    zeros = [
        np.zeros_like(eng.model.occupancy),
        np.zeros_like(eng.model.u_iso),
        np.zeros_like(eng.model.u_star),
        np.zeros_like(eng.model.fp),
        np.zeros_like(eng.model.fdp),
    ]
    diag = np.zeros_like(sites)
    for i in range(sites.shape[0]):
        for j in range(3):
            v = np.zeros_like(sites)
            v[i, j] = 1.0
            hv = eng.gauss_newton_hvp((v, *zeros), curv_radial, curv_tangential)
            diag[i, j] = hv["site_frac"][i, j]
    return diag


def test_gauss_newton_diagonal_hutchinson_matches_exact():
    """Hutchinson diag ≈ exact GN diagonal on a tiny P1 LS-on-F problem (exact data)."""
    xs = _structure("P1", elements=("C", "N", "O"), n_repeat=1, seed=7)
    d_min = 2.0
    fc = xs.structure_factors(d_min=d_min, algorithm="direct").f_calc()
    hkl = np.array(list(fc.indices()))
    eng = _engine(xs, hkl, d_min, quality_factor=1000)
    f_obs = np.abs(eng.f_calc_numpy())
    obs = Observations.from_numpy(data=f_obs)
    tgt = LeastSquares(obs_type="F", scale_factor=1.0)
    with torch.no_grad():
        f = eng.f_calc(*eng.tensors())
    ev = tgt.evaluate(f, obs)
    exact = _exact_gn_diagonal_sites(eng, ev.curv_radial, ev.curv_tangential)
    estimates = [
        eng.gauss_newton_diagonal(ev.curv_radial, ev.curv_tangential, n_probes=64, seed=s)["site_frac"]
        for s in range(4)
    ]
    mean_est = np.mean(estimates, axis=0)
    # loose tolerance: Hutchinson with m=64, averaged over seeds
    assert _rel(exact, mean_est) < 0.25
    assert np.corrcoef(exact.ravel(), mean_est.ravel())[0, 1] > 0.95


def test_gauss_newton_diagonal_op_round_trip():
    bridge = _bridge()
    xs = _structure("P1", elements=("C", "N", "O"), n_repeat=1, seed=8)
    fc = xs.structure_factors(d_min=2.0, algorithm="direct").f_calc()
    f_obs = fc.amplitudes()
    refiner = RemoteRefinementTarget(
        bridge, xs, f_obs, {"name": "ls", "obs_type": "F"}, params=SfEngineParams(d_min=2.0, quality_factor=1000)
    )
    out = refiner.compute()
    from phridge.client.xtal_engine import _miller_template

    diag = bridge.call(
        "gauss_newton_diagonal",
        xray=xray_from_cctbx(xs),
        table=scattering_table_from_cctbx(xs),
        params=refiner.params,
        target=out["target"],
        hkl=_miller_template(f_obs),
        n_probes=8,
        seed=0,
    )
    n = xs.scatterers().size()
    assert diag.d_site_frac.shape == (n, 3)
    assert float(diag.d_site_frac.sum()) != 0.0


@pytest.mark.parametrize("space_group", ["P1", "P21"])
def test_gauss_newton_blocks_site_match_finite_difference(space_group):
    xs = _structure(space_group, elements=("C", "N", "O"), n_repeat=1, seed=9)
    d_min = 2.0
    fc = xs.structure_factors(d_min=d_min, algorithm="direct").f_calc()
    hkl = np.array(list(fc.indices()))
    eng = _engine(xs, hkl, d_min, quality_factor=1000)
    f_obs = np.abs(eng.f_calc_numpy())
    obs = Observations.from_numpy(data=f_obs)
    tgt = LeastSquares(obs_type="F", scale_factor=1.0)
    with torch.no_grad():
        f = eng.f_calc(*eng.tensors())
    ev = tgt.evaluate(f, obs)
    blocks = eng.gauss_newton_blocks(ev.curv_radial, ev.curv_tangential)
    sites = eng.model.sites_frac.copy()
    h = 1e-4

    def grad_sites(s):
        params = list(eng.tensors(requires_grad=True))
        params[0] = torch.tensor(s, dtype=torch.float64, requires_grad=True)
        with torch.no_grad():
            ff = eng.f_calc(*params)
        e = tgt.evaluate(ff, obs)
        return eng.gradients(e.d_target_d_f_calc, params=tuple(params))["site_frac"]

    for j in range(sites.shape[0]):
        H_fd = np.zeros((3, 3))
        for a in range(3):
            sp, sm = sites.copy(), sites.copy()
            sp[j, a] += h
            sm[j, a] -= h
            H_fd[:, a] = (grad_sites(sp)[j] - grad_sites(sm)[j]) / (2 * h)
        assert _rel(H_fd, blocks["site_frac"][j]) < 5e-3

    # diagonal agrees with Hutchinson estimate
    hutch = eng.gauss_newton_diagonal(ev.curv_radial, ev.curv_tangential, n_probes=64, seed=0)
    block_diag = np.diagonal(blocks["site_frac"], axis1=1, axis2=2)
    assert np.corrcoef(block_diag.ravel(), hutch["site_frac"].ravel())[0, 1] > 0.95


def test_gauss_newton_blocks_op_round_trip():
    bridge = _bridge()
    xs = _structure("P1", elements=("C", "N", "O"), n_repeat=1, seed=10)
    fc = xs.structure_factors(d_min=2.0, algorithm="direct").f_calc()
    f_obs = fc.amplitudes()
    refiner = RemoteRefinementTarget(
        bridge, xs, f_obs, {"name": "ls", "obs_type": "F"}, params=SfEngineParams(d_min=2.0, quality_factor=1000)
    )
    curv = refiner.curvatures()
    n = xs.scatterers().size()
    assert curv.site_frac.shape == (n, 3, 3)
    assert curv.u_star.shape == (n, 6, 6)
    # round-trip pack/unpack
    from phridge.packing_scattering import unpack_sf_curvatures

    blob = curv.pack()
    again = unpack_sf_curvatures(blob, curv.meta)
    np.testing.assert_allclose(again.site_frac, curv.site_frac)


def test_lbfgs_with_and_without_diagonal_preconditioner():
    """Report iterations-to-convergence; diagonal should not hurt and usually helps."""
    import scitbx.lbfgs

    bridge = _bridge()
    xs = _structure("P21", n_repeat=2, seed=3)
    fc = xs.structure_factors(d_min=2.0, algorithm="direct").f_calc()
    f_obs = fc.amplitudes()

    def run(use_diag):
        xs2 = xs.deep_copy_scatterers()
        xs2.shake_sites_in_place(rms_difference=0.15)
        for sc in xs2.scatterers():
            sc.flags.set_grad_site(True)
            sc.flags.set_grad_occupancy(False)
            sc.flags.set_grad_u_iso(False)
            sc.flags.set_grad_u_aniso(False)
        refiner = RemoteRefinementTarget(
            bridge, xs2, f_obs, {"name": "ls", "obs_type": "F"}, params=SfEngineParams(d_min=2.0)
        )
        inv_diag = refiner.diagonal(xs2, method="blocks") if use_diag else None

        class Minimizer:
            def __init__(self):
                self.x = xs2.sites_cart().as_double()
                self.n_calls = 0
                self.diag_mode = "once" if use_diag else None
                scitbx.lbfgs.run(
                    target_evaluator=self,
                    termination_params=scitbx.lbfgs.termination_parameters(max_iterations=30),
                )

            def compute_functional_and_gradients(self):
                xs2.set_sites_cart(flex.vec3_double(self.x))
                self.n_calls += 1
                return refiner.target_and_gradients(xs2)

            def compute_functional_gradients_diag(self):
                t, g = self.compute_functional_and_gradients()
                return t, g, inv_diag

        m = Minimizer()
        t1, _ = refiner.target_and_gradients(xs2)
        return m.n_calls, t1, xs.rms_difference(xs2)

    n0, t0, rms0 = run(False)
    n1, t1, rms1 = run(True)
    assert t0 < 1e-2 and t1 < 1e-2
    assert rms0 < 0.1 and rms1 < 0.1
    # both converge; with diag should not need vastly more calls
    assert n1 <= n0 * 2
    print(f"LBFGS without diag: calls={n0} target={t0:.3e} rms={rms0:.4f}")
    print(f"LBFGS with diag:    calls={n1} target={t1:.3e} rms={rms1:.4f}")


def test_newton_cg_vs_lbfgs():
    bridge = _bridge()
    xs = _structure("P21", n_repeat=2, seed=3)
    fc = xs.structure_factors(d_min=2.0, algorithm="direct").f_calc()
    f_obs = fc.amplitudes()
    xs2 = xs.deep_copy_scatterers()
    xs2.shake_sites_in_place(rms_difference=0.15)
    for sc in xs2.scatterers():
        sc.flags.set_grad_site(True)
        sc.flags.set_grad_occupancy(False)
        sc.flags.set_grad_u_iso(False)
    refiner = RemoteRefinementTarget(
        bridge, xs2, f_obs, {"name": "ls", "obs_type": "F"}, params=SfEngineParams(d_min=2.0)
    )
    t0, _ = refiner.target_and_gradients(xs2)
    result = refiner.newton_cg(xs2, max_iterations=10, cg_max_iter=10, damping=1e-2)
    assert result["target"] < 0.2 * t0
    assert result["n_iterations"] <= 10
    # amplitude-only: origin along unique axis is free; check |F| fit
    fc_r = result["xray_structure"].structure_factors(d_min=2.0, algorithm="direct").f_calc()
    assert f_obs.r1_factor(fc_r.amplitudes()) < 1e-3

    xs3 = xs.deep_copy_scatterers()
    xs3.shake_sites_in_place(rms_difference=0.15)
    for sc in xs3.scatterers():
        sc.flags.set_grad_site(True)
        sc.flags.set_grad_occupancy(False)
        sc.flags.set_grad_u_iso(False)
    import scitbx.lbfgs

    refiner3 = RemoteRefinementTarget(
        bridge, xs3, f_obs, {"name": "ls", "obs_type": "F"}, params=SfEngineParams(d_min=2.0)
    )

    class M:
        def __init__(self):
            self.x = xs3.sites_cart().as_double()
            self.n_calls = 0
            scitbx.lbfgs.run(target_evaluator=self, termination_params=scitbx.lbfgs.termination_parameters(max_iterations=20))

        def compute_functional_and_gradients(self):
            xs3.set_sites_cart(flex.vec3_double(self.x))
            self.n_calls += 1
            return refiner3.target_and_gradients(xs3)

    m = M()
    t_lbfgs, _ = refiner3.target_and_gradients(xs3)
    assert t_lbfgs < 0.2 * t0
    print(f"Newton-CG: iters={result['n_iterations']} target={result['target']:.3e}")
    print(f"L-BFGS:    calls={m.n_calls} target={t_lbfgs:.3e}")


def test_gauss_newton_blocks_match_hvp_at_nonzero_residual():
    """Regression: the Tronrud sum/difference weights were swapped, visible only when the
    tangential curvature g'/|F| is nonzero (i.e. away from zero residual)."""
    bridge = _bridge()
    xs = _structure("P1", n_repeat=2, seed=5)
    fc = xs.structure_factors(d_min=2.5, algorithm="direct").f_calc()
    f_obs = fc.amplitudes()
    xs2 = xs.deep_copy_scatterers()
    xs2.shake_sites_in_place(rms_difference=0.2)
    for sc in xs2.scatterers():
        sc.flags.set_grad_site(True)
        sc.flags.set_grad_occupancy(False)
        sc.flags.set_grad_u_iso(False)
    refiner = RemoteRefinementTarget(
        bridge, xs2, f_obs, {"name": "ls", "obs_type": "F"}, params=SfEngineParams(d_min=2.5, quality_factor=1000)
    )
    from phridge.packing_scattering import PackedSfGradients
    from phridge.sfcalc.client import _miller_template, _packed_xray, psd_target

    out = refiner.compute(xs2)
    tgt = out["target"]
    assert np.asarray(tgt.curv_tangential).min() < 0  # the regime where the sign matters
    xray, table = _packed_xray(xs2, refiner.table)
    hkl = _miller_template(f_obs)
    n = xs2.scatterers().size()
    for target in (tgt, psd_target(tgt)):
        blocks = np.asarray(bridge.call("gauss_newton_blocks", xray=xray, table=table, params=refiner.params, target=target, hkl=hkl).site_frac)
        for i, a in ((0, 0), (1, 2), (n - 1, 1)):
            e = np.zeros((n, 3))
            e[i, a] = 1.0
            v = PackedSfGradients(d_site_frac=e, d_occupancy=np.zeros(n), d_u_iso=np.zeros(n), d_u_star=np.zeros((n, 6)), d_fp=np.zeros(n), d_fdp=np.zeros(n))
            hv = np.asarray(bridge.call("gauss_newton_hvp", xray=xray, table=table, params=refiner.params, target=target, hkl=hkl, v=v).d_site_frac)
            assert abs(hv[i, a] - blocks[i, a, a]) < 1e-2 * abs(hv[i, a])  # FFT-vs-direct accuracy; the bug gave 30-50%
            assert abs(hv[i, (a + 1) % 3] - blocks[i, a, (a + 1) % 3]) < 1e-2 * abs(hv[i, a])
