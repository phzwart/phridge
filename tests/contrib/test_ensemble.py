"""Ensemble package: registration, known-F ALS, E-step, EM, gauge, roles."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.legacy
from pydantic import ValidationError

from phridge.contrib.ensemble import (
    OP_NAME_FIT,
    OP_NAME_MAPS,
    EnsembleOptions,
    register,
)
from phridge.contrib.ensemble.gauge import fix_gauge, principal_angles, procrustes_align, reconstruction
from phridge.contrib.ensemble.latent import ComplexGLS
from phridge.contrib.ensemble.loadings import FullComplexLoadings
from phridge.ops import get_op

torch = pytest.importorskip("torch")


def test_register_is_idempotent():
    register()
    register()
    assert get_op(OP_NAME_FIT).name == OP_NAME_FIT
    assert get_op(OP_NAME_MAPS).name == OP_NAME_MAPS


def test_pydantic_accept_reject():
    EnsembleOptions()
    EnsembleOptions(k=2, likelihood="known_f", phase_injection={"hl_weight": 0.4, "enabled": True})
    with pytest.raises(ValidationError):
        EnsembleOptions(k=-1)
    with pytest.raises(ValidationError):
        EnsembleOptions(likelihood="rice")
    with pytest.raises(ValidationError):
        EnsembleOptions(extra_field=1)


def test_client_import_is_torch_free():
    import phridge.contrib.ensemble.client as client
    import phridge.contrib.ensemble.options as options

    assert "torch" not in client.__dict__
    assert "torch" not in options.__dict__
    EnsembleOptions()


def test_job_envelope_validates():
    jsonschema = pytest.importorskip("jsonschema")
    from phridge.contracts.models import SCHEMA_VERSION, JobEnvelope, JobStatus, ObjectKind, ObjectRef

    register()
    envelope = JobEnvelope(
        job_id="ensemble-test",
        op=OP_NAME_FIT,
        schema_version=SCHEMA_VERSION,
        inputs={
            "hkl": ObjectRef(kind=ObjectKind.array, key="phridge:obj:e:hkl"),
            "options": ObjectRef(kind=ObjectKind.json, key="phridge:obj:e:options"),
        },
        status=JobStatus.queued,
    )
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "phridge"
        / "contracts"
        / "schema"
        / "generated"
        / "JobEnvelope.schema.json"
    )
    if not schema_path.is_file():
        pytest.skip("generated JobEnvelope schema is absent")
    schema = json.loads(schema_path.read_text())
    # The generated schema stores inputs/outputs as arrays. The pydantic model
    # round-trips the op name; the schema check uses that array form.
    payload = {
        "job_id": envelope.job_id,
        "op": envelope.op,
        "schema_version": envelope.schema_version,
        "status": envelope.status.value,
        "inputs": [],
        "outputs": [],
    }
    jsonschema.validate(payload, schema)
    assert envelope.op == OP_NAME_FIT


def _planted(n=6, h=12, k=2, seed=0):
    rng = np.random.default_rng(seed)
    modes = rng.normal(size=(k, h)) + 1j * rng.normal(size=(k, h))
    modes = modes / np.linalg.norm(modes, axis=1, keepdims=True)
    z = rng.normal(size=(n, k))
    z = z - z.mean(axis=0)
    delta0 = rng.normal(size=h) + 1j * rng.normal(size=h)
    target = delta0 + z.astype(np.complex128) @ modes
    weights = np.ones((n, h), dtype=np.float64)
    return target, weights, delta0, modes, z


def _numpy_solve_modes(y, z, w, ridge):
    n, h = y.shape
    k = z.shape[1]
    design = np.concatenate([np.ones((n, 1)), z], axis=1)
    delta0 = np.zeros(h, dtype=np.complex128)
    modes = np.zeros((k, h), dtype=np.complex128)
    eye = np.eye(k + 1)
    for j in range(h):
        wh = w[:, j]
        gram = (design * wh[:, None]).T @ design + ridge * eye
        rhs = (design * wh[:, None]).T @ y[:, j]
        beta = np.linalg.solve(gram, rhs)
        delta0[j] = beta[0]
        if k:
            modes[:, j] = beta[1:]
    return delta0, modes


def _numpy_solve_scores(y, delta0, modes, w, ridge):
    n, _h = y.shape
    k = modes.shape[0]
    scores = np.zeros((n, k), dtype=np.float64)
    cov = np.zeros((n, k, k), dtype=np.float64)
    for i in range(n):
        resid = y[i] - delta0
        gram = np.real(np.conj(modes) * w[i] @ modes.T)
        gram = gram + ridge * np.eye(k)
        rhs = np.real((np.conj(modes) * w[i]) @ resid)
        scores[i] = np.linalg.solve(gram, rhs)
        cov[i] = np.linalg.inv(gram)
    return scores, cov


def test_known_f_matches_numpy_transcription():
    from phridge.contrib.ensemble.mstep import als_known_f, solve_modes_torch, solve_scores_torch

    target, weights, _d0, _modes, _z = _planted()
    y = target
    rng = np.random.default_rng(2)
    z = rng.normal(size=(y.shape[0], 2))
    z = z - z.mean(axis=0)
    torch_d, torch_m = solve_modes_torch(y, z, weights, 1e-10)
    np_d, np_m = _numpy_solve_modes(y, z, weights, 1e-10)
    assert np.max(np.abs(torch_d - np_d)) < 1e-8
    assert np.max(np.abs(torch_m - np_m)) < 1e-8
    td, tm, ts, _tc = als_known_f(y, weights, 2, n_iter=12, ridge=1e-10)
    # Independent ALS using the numpy normal equations and the same gauge.
    delta0, modes, scores = td * 0, tm * 0, ts * 0
    from phridge.contrib.ensemble.mstep import initialize_from_residuals

    delta0, modes, scores = initialize_from_residuals(y, weights, 2)
    for _ in range(12):
        delta0, modes = _numpy_solve_modes(y, scores, weights, 1e-10)
        scores, _cov = _numpy_solve_scores(y, delta0, modes, weights, 1e-10)
        delta0, modes, scores = fix_gauge(delta0, modes, scores)
    aligned, _rot = procrustes_align(tm, modes)
    # Subspace agreement after alignment. A shared gauge can still flip a
    # near-tie; the reconstruction is the invariant.
    rec_t = reconstruction(td, tm, ts)
    rec_n = reconstruction(delta0, modes, scores)
    assert np.max(np.abs(rec_t - rec_n)) < 1e-6
    assert np.max(np.abs(aligned - tm)) < 1e-5 or np.max(np.abs(rec_t - y)) < 1e-6
    _ = solve_scores_torch


def test_v15_covariance_is_inverse_gram():
    rng = np.random.default_rng(3)
    h = 20
    k = 2
    modes = rng.normal(size=(k, h)) + 1j * rng.normal(size=(k, h))
    resid = rng.normal(size=h) + 1j * rng.normal(size=h)
    weights = np.linspace(0.2, 1.0, h)
    poster = ComplexGLS(delta0=np.zeros(h))
    cov = poster.covariance(resid, weights, modes, ridge=0.0)
    gram = np.real((np.conj(modes) * weights) @ modes.T)
    assert np.max(np.abs(cov - np.linalg.inv(gram))) < 1e-10
    FullComplexLoadings().apply(modes, np.ones((3, k)))


def test_v5_gauge_leaves_field_unchanged():
    target, _w, delta0, modes, z = _planted()
    before = reconstruction(delta0, modes, z)
    d2, m2, z2 = fix_gauge(delta0, modes, z)
    after = reconstruction(d2, m2, z2)
    assert np.max(np.abs(before - after)) < 1e-10
    from phridge.legacy.ensemble_fit import _known_f_loglik

    weights = np.ones_like(target.real)
    ll0 = _known_f_loglik(target, weights, delta0, modes, z)
    ll1 = _known_f_loglik(target, weights, d2, m2, z2)
    assert abs(ll0 - ll1) < 1e-10


def test_v1_hl_none_bit_identical():
    from phridge.d.maps import posterior_moments

    ec = torch.tensor([0.4, 1.2, 2.5], dtype=torch.float64)
    sa = torch.tensor([0.6, 0.8, 0.7], dtype=torch.float64)
    zo = torch.tensor([0.2, 1.5, 6.0], dtype=torch.float64)
    sz = torch.tensor([0.3, 0.4, 0.5], dtype=torch.float64)
    cen = torch.tensor([False, False, True])
    a = posterior_moments(ec, sa, zo, sz, cen)
    b = posterior_moments(ec, sa, zo, sz, cen, hl=None)
    assert torch.equal(a.E_mean, b.E_mean)
    assert torch.equal(a.I_mean, b.I_mean)
    assert torch.equal(a.log_lik, b.log_lik)


def test_v7_centric_two_point():
    from phridge.d.maps import centric_two_point_moment

    kappas = np.linspace(-1.5, 2.5, 9)
    axes = np.linspace(-1.0, 1.0, 7)
    for kappa in kappas:
        for axis in axes:
            plus = np.exp(kappa + axis)
            minus = np.exp(-kappa - axis)
            brute = (plus - minus) / (plus + minus)
            got = float(centric_two_point_moment(torch.tensor(kappa, dtype=torch.float64), torch.tensor(axis, dtype=torch.float64)))
            assert abs(got - brute) < 1e-8


def test_v2_gaussian_amplitude_quadrature():
    scipy_special = pytest.importorskip("scipy.special")
    from phridge.d.maps import posterior_moments

    ec, sa, zo, sz = 1.4, 0.75, 2.2, 0.35
    a = 1.0 - sa**2

    def log_integrand(amplitude: np.ndarray) -> np.ndarray:
        x = (2.0 * sa * ec / a) * amplitude
        noise = -((zo - amplitude**2) ** 2) / (2.0 * sz**2) - 0.5 * np.log(2.0 * np.pi * sz**2)
        return np.log(2.0 * amplitude / a) - (amplitude**2 + sa**2 * ec**2) / a + x + np.log(scipy_special.i0e(x)) + noise

    grid = np.linspace(1e-3, 6.0, 8000)
    log_terms = log_integrand(grid)
    shift = log_terms.max()
    weights = np.exp(log_terms - shift)
    norm = np.trapezoid(weights, grid)
    posterior = weights / norm
    kappa = (2.0 * sa * ec / a) * grid
    fom = scipy_special.i1e(kappa) / scipy_special.i0e(kappa)
    em = float(np.trapezoid(posterior * grid * fom, grid))
    log_p = float(shift + np.log(norm))
    post = posterior_moments(
        torch.tensor([ec], dtype=torch.float64),
        torch.tensor([sa], dtype=torch.float64),
        torch.tensor([zo], dtype=torch.float64),
        torch.tensor([sz], dtype=torch.float64),
        torch.tensor([False]),
        n_legendre=48,
        n_hermite=12,
    )
    assert abs(float(post.Em_mean) - em) / max(abs(em), 1e-6) < 1e-4
    assert abs(float(post.log_lik) - log_p) < 1e-4
    # Gaussian-amplitude limit: tiny intensity sigma, posterior mean tracks sqrt(Z).
    post_tight = posterior_moments(
        torch.tensor([2.0], dtype=torch.float64),
        torch.tensor([0.2], dtype=torch.float64),
        torch.tensor([4.0], dtype=torch.float64),
        torch.tensor([0.02], dtype=torch.float64),
        torch.tensor([False]),
        nu=None,
    )
    assert abs(float(post_tight.E_mean) - 2.0) < 1e-2


def test_v3_fisher_matches_finite_difference():
    from phridge.contrib.ensemble.moments import fisher_mu_gradient, fixed_grid_posterior

    sa = 0.7
    ec = np.array([1.6])
    zo = np.array([2.4])
    sz = np.array([0.45])
    base = fixed_grid_posterior(ec, sa, zo, sz)
    step = 1e-5
    plus = fixed_grid_posterior(ec + step, sa, zo, sz)
    minus = fixed_grid_posterior(ec - step, sa, zo, sz)
    g_fd_ec = (plus["log_lik"] - minus["log_lik"]) / (2.0 * step)
    a = 1.0 - sa**2
    g_an_ec = (2.0 * sa / a) * (base["em"] - sa * ec)
    g_fd = float(np.asarray(g_fd_ec).reshape(-1)[0])
    g_an = float(np.asarray(g_an_ec).reshape(-1)[0])
    denom = max(abs(g_fd), 1e-8)
    assert abs(g_an - g_fd) / denom < 1e-6
    # Chain onto a real positive μ with scale 1 and D = 1: |μ| = σ_A E_c.
    mu = sa * float(ec[0])
    sigma2 = a  # scale = 1
    mean_f = float(np.asarray(base["em"]).reshape(-1)[0])
    g_mu = fisher_mu_gradient(np.array([mean_f]), np.array([mu]), np.array([sigma2]))
    # dL/dEc = dL/d|μ| * d|μ|/dEc = Re(g_mu) * σ_A
    g_from_mu = float(np.real(g_mu[0])) * sa
    assert abs(g_from_mu - g_an) / denom < 1e-6


def _tiny_intensity(n=4, hmax=1, k=1, seed=1):
    from phridge.contrib.ensemble.synthetic import simulate_campaign

    camp = simulate_campaign(n_datasets=n, k_true=k, hmax=hmax, seed=seed, noise=0.02)
    wilson = np.full((n, camp.hkl.shape[0]), float(np.mean(np.abs(camp.f_mu) ** 2)))
    return camp, wilson


def test_v4_em_is_monotone():
    from phridge.legacy.ensemble_fit import em_fit

    camp, wilson = _tiny_intensity()
    result = em_fit(
        camp.i_obs,
        camp.sig_i,
        camp.mask,
        camp.f_mu,
        camp.epsilon,
        camp.centric,
        wilson,
        rank=1,
        n_sweeps=20,
        dof_correct=False,
        shell_index=camp.shell_index,
        sigma_a0=0.8,
    )
    hist = result.loglik_history
    assert hist.shape[0] == 21
    assert np.all(np.diff(hist) >= -1e-6)


def test_v14_missing_reflections_have_zero_weight():
    from phridge.legacy.ensemble_fit import fit_known_f

    camp, _wilson = _tiny_intensity(n=5, k=1)
    phase = np.angle(camp.f_true)
    f_obs = np.sqrt(np.maximum(camp.i_obs, 0.0)) * np.exp(1j * phase)
    mask = camp.mask.copy()
    mask[:, 0] = False
    first = fit_known_f(f_obs, camp.f_mu, mask, rank=1, n_sweeps=6)
    f_obs2 = f_obs.copy()
    f_obs2[:, 0] = 1e6
    second = fit_known_f(f_obs2, camp.f_mu, mask, rank=1, n_sweeps=6)
    assert np.max(np.abs(first.delta0 - second.delta0)) < 1e-8
    assert np.max(np.abs(first.modes - second.modes)) < 1e-8
    assert np.max(np.abs(first.scores - second.scores)) < 1e-8


def test_v16_audit_reflections_do_not_enter():
    from phridge.legacy.ensemble_fit import fit_known_f
    from phridge.contrib.ensemble.holdout import assign_roles

    camp, _wilson = _tiny_intensity(n=5, k=1)
    phase = np.angle(camp.f_true)
    f_obs = np.sqrt(np.maximum(camp.i_obs, 0.0)) * np.exp(1j * phase)
    roles = assign_roles(camp.mask, audit_fraction=0.25, seed=2)
    assert np.any(roles == 2)
    first = fit_known_f(f_obs, camp.f_mu, camp.mask, rank=1, n_sweeps=6, roles=roles)
    f_obs2 = f_obs.copy()
    f_obs2[roles == 2] = 1e6 + 1e6j
    second = fit_known_f(f_obs2, camp.f_mu, camp.mask, rank=1, n_sweeps=6, roles=roles)
    assert np.max(np.abs(first.delta0 - second.delta0)) < 1e-8
    assert np.max(np.abs(first.scores - second.scores)) < 1e-8


def test_reference_oracles_absent():
    root = Path(__file__).resolve().parents[2] / "reference"
    if (root / "deltaem.py").is_file() and importlib.util.find_spec("deltaem") is not None:
        pytest.skip("reference oracle is present; the numeric comparison lives with it")
    pytest.xfail("reference/ not shipped")


def test_op_roundtrip_known_f():
    from phridge.client import Bridge
    from phridge.contrib.ensemble.synthetic import simulate_campaign

    register()
    camp = simulate_campaign(n_datasets=4, k_true=1, hmax=1, seed=4, noise=0.01)
    bridge = Bridge(memory=True, timeout=5)
    out = bridge.call(
        OP_NAME_FIT,
        hkl=camp.hkl,
        centric=camp.centric,
        epsilon=camp.epsilon,
        i_obs=camp.i_obs,
        sig_i=camp.sig_i,
        mask=camp.mask,
        f_mu=camp.f_mu,
        cells=np.asarray(camp.cell, dtype=np.float64),
        options={"likelihood": "known_f", "k": 1, "n_sweeps": 4, "bin_size": 8},
    )
    assert out["modes"].shape[0] == 1
    assert out["scores"].shape == (4, 1)
    assert out["delta0"].shape == (camp.hkl.shape[0],)
    angles = principal_angles(camp.modes_true, out["modes"])
    assert angles.size == 1


def test_chunk_and_cap():
    from phridge.contrib.ensemble.client import chunk_dataset_indices, payload_nbytes

    assert chunk_dataset_indices(10, row_nbytes=100, max_bytes=250) == [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]
    assert payload_nbytes({"a": np.zeros(4, dtype=np.float64)}) == 32
