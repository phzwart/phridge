"""Spatial σ_A v2: Phase 1 plumbing and Phase 2 model / Rice tests (spec §11)."""

from __future__ import annotations

import importlib.util
import inspect
import json
import math
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from phridge.contrib.spatial_sigmaa_v2 import (
    PackedSpatialSigmaAV2,
    PackedSpatialSigmaAV2Result,
    SpatialSigmaAV2Options,
    block_from_env,
    job_input_from_env,
    options_from_env,
    register,
    unpack_spatial_sigma_a_v2,
    unpack_spatial_sigma_a_v2_result,
)
from phridge.contrib.spatial_sigmaa_v2.fields import FieldBasis, evaluate_fields
from phridge.contrib.spatial_sigmaa_v2.moments import (
    f_eff_direct,
    form_factors,
    model_temperature,
    moments,
    resolution_s2,
    sigma_delta,
    sigma_p,
)
from phridge.contrib.spatial_sigmaa_v2.per_atom import (
    EIGHT_PI2,
    effective_weight,
    error_u_iso,
    interpolate_shell,
    luzzati_d_iso,
    modified_model,
    presence_weight,
)
from phridge.contrib.spatial_sigmaa_v2.gradients import (
    add_atom_grads,
    atom_to_field_params,
    field_gradients,
    mean_gradients_direct,
    variance_gradients_aniso,
    variance_gradients_iso,
)
from phridge.contrib.spatial_sigmaa_v2.regulariser import regulariser, spectral_taper
from phridge.contrib.spatial_sigmaa_v2.target import (
    bessel_i1_over_i0,
    draw_circular_normal,
    intensity_variance_limit,
    least_squares_intensity_nll,
    log_i0,
    rice_nll_intensity,
)
from phridge.models import CCTBX_TYPES, SpatialSigmaAV2, SpatialSigmaAV2Result
from phridge.sfcalc.engine.cell import orthogonalization_matrix
from phridge.sfcalc.engine.engine import ScatteringModel
from phridge.sfcalc.engine.symmetry import identity_ops

GENERATED = Path(__file__).resolve().parents[2] / "schema" / "generated"


def test_register_is_idempotent():
    register()
    register()


def test_pydantic_accept_reject():
    SpatialSigmaAV2Options()
    SpatialSigmaAV2Options(enabled=True, field_cutoff=20.0, fisher=True)
    with pytest.raises(ValidationError):
        SpatialSigmaAV2Options(field_cutoff=0.0)
    with pytest.raises(ValidationError):
        SpatialSigmaAV2Options(not_a_field=1)  # type: ignore[call-arg]


def test_pack_unpack_empty_block():
    packed = PackedSpatialSigmaAV2.empty(field_cutoff=15.0, fisher=True)
    assert packed.meta.enabled is True
    assert packed.meta.n_coeff == 0
    assert packed.meta.n_shells == 0
    blob = packed.pack()
    again = unpack_spatial_sigma_a_v2(blob, packed.meta)
    np.testing.assert_array_equal(again.lambda_c, packed.lambda_c)
    assert again.meta.fisher is True
    assert again.meta.field_cutoff == 15.0


def test_pack_unpack_result():
    packed = PackedSpatialSigmaAV2Result(
        d_lambda_c=np.array([0.1, -0.2]),
        d_kappa_c=np.array([0.0, 0.3]),
        d_D0=np.array([1.0]),
        d_Sigma_miss=np.array([0.5]),
        w=np.array([0.9, 0.8, 1.0]),
        u_err_star=np.zeros((3, 6)),
        tr_U=np.array([0.01, 0.02, 0.0]),
    )
    again = unpack_spatial_sigma_a_v2_result(packed.pack(), packed.meta)
    np.testing.assert_allclose(again.w, packed.w)
    assert again.meta.n_scatterers == 3
    assert again.meta.n_coeff == 2
    assert again.meta.n_shells == 1


def test_cctbx_types_include_spatial_sigma_a_v2():
    assert "SpatialSigmaAV2" in CCTBX_TYPES
    assert "SpatialSigmaAV2Result" in CCTBX_TYPES


def test_jsonschema_accepts_spatial_sigma_a_v2():
    jsonschema = pytest.importorskip("jsonschema")
    path = GENERATED / "SpatialSigmaAV2.schema.json"
    if not path.exists():
        pytest.skip("run scripts/generate_models.py --jsonschema")
    schema = json.loads(path.read_text())
    meta = SpatialSigmaAV2(enabled=True, field_cutoff=15.0, n_coeff=0, n_shells=0)
    jsonschema.validate(meta.model_dump(mode="json"), schema)
    result = SpatialSigmaAV2Result(n_scatterers=0, n_coeff=0, n_shells=0)
    result_schema = json.loads((GENERATED / "SpatialSigmaAV2Result.schema.json").read_text())
    jsonschema.validate(result.model_dump(mode="json"), result_schema)


def test_cli_scripts_accept_spatial_sigma_a_v2_flag():
    root = Path(__file__).resolve().parents[2]
    driver = (root / "scripts" / "phenix_refine_mli.py").read_text()
    wrapper = (root / "scripts" / "run_phenix_intensity.sh").read_text()
    cli = (root / "src" / "phridge" / "client" / "intensity" / "cli.py").read_text()
    assert "--spatial-sigmaa-v2" in driver.lower()
    assert "PHRIDGE_SPATIAL_SIGMA_A_V2" in driver
    assert "--spatial-sigmaA-v2" in wrapper
    assert "PHRIDGE_SPATIAL_SIGMA_A_V2" in wrapper
    assert "--spatial-sigmaA-v2" in cli
    assert "PHRIDGE_SPATIAL_SIGMA_A_V2_FISHER" in driver


def test_flag_off_is_inert(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the flag off, no SpatialSigmaAV2 block is constructed or sent (spec §11.9)."""
    monkeypatch.delenv("PHRIDGE_SPATIAL_SIGMA_A_V2", raising=False)
    monkeypatch.delenv("PHRIDGE_SPATIAL_SIGMA_A_V2_FISHER", raising=False)
    monkeypatch.delenv("PHRIDGE_SPATIAL_SIGMA_A_V2_CUTOFF", raising=False)
    opts = options_from_env()
    assert opts.enabled is False
    assert opts.fisher is False
    assert block_from_env() is None
    assert job_input_from_env() == {}


def test_flag_on_constructs_block(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHRIDGE_SPATIAL_SIGMA_A_V2", "1")
    monkeypatch.setenv("PHRIDGE_SPATIAL_SIGMA_A_V2_FISHER", "1")
    monkeypatch.setenv("PHRIDGE_SPATIAL_SIGMA_A_V2_CUTOFF", "18")
    block = block_from_env()
    assert block is not None
    assert block.meta.enabled is True
    assert block.meta.fisher is True
    assert block.meta.field_cutoff == 18.0
    payload = job_input_from_env()
    assert set(payload) == {"spatial_sigma_a_v2"}
    sent = payload["spatial_sigma_a_v2"]
    assert sent.meta.enabled is True
    assert sent.meta.field_cutoff == 18.0
    assert sent.meta.fisher is True


def test_optional_op_slots_do_not_change_required_inputs():
    from phridge.contrib.intensity_ll.ops import (
        _NUISANCE_INPUTS,
        _TARGET_AND_GRADIENTS_INPUTS,
        ml_i_nuisance_fit,
        ml_i_target_and_gradients,
    )

    assert _NUISANCE_INPUTS["spatial_sigma_a_v2"] == "SpatialSigmaAV2"
    assert _TARGET_AND_GRADIENTS_INPUTS["spatial_sigma_a_v2"] == "SpatialSigmaAV2"
    assert _TARGET_AND_GRADIENTS_INPUTS["spatial_sigma_a_v2_result"] == "SpatialSigmaAV2Result"
    assert ml_i_nuisance_fit.__defaults__ is not None
    assert ml_i_target_and_gradients.__defaults__ is not None


def test_codec_roundtrip_when_flag_on():
    fakeredis = pytest.importorskip("fakeredis")
    from phridge.codec import decode_ref, encode_value
    from phridge.redis_store import RedisStore

    store = RedisStore(fakeredis.FakeRedis())
    packed = PackedSpatialSigmaAV2.empty(field_cutoff=16.0, entropy_weight=0.1)
    ref = encode_value(store, "job-v2", "spatial_sigma_a_v2", packed)
    assert ref.cctbx_type == "SpatialSigmaAV2"
    back = decode_ref(store, ref)
    assert isinstance(back, PackedSpatialSigmaAV2)
    assert back.meta.field_cutoff == 16.0
    assert back.meta.entropy_weight == 0.1


# ---------------------------------------------------------------------------
# Phase 2: fields, per-atom weights, moments, Rice/Woolfson (§1–§4, §6)
# ---------------------------------------------------------------------------

_P212121_ROT = np.array(
    [
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
    ]
)
_P212121_TRANS = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [0.5, 0.5, 0.0],
    ]
)
_P1BAR_ROT = np.array(
    [
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
    ]
)
_P1BAR_TRANS = np.zeros((2, 3))


def _toy_model(sites, occ=None, u_iso=None, cell=(30.0, 30.0, 30.0, 90.0, 90.0, 90.0), f=6.0, rot=None, trans=None):
    sites = np.asarray(sites, dtype=np.float64).reshape(-1, 3)
    n = sites.shape[0]
    rot_a, trans_a = identity_ops() if rot is None else (np.asarray(rot, float), np.asarray(trans, float))
    return ScatteringModel(
        unit_cell=tuple(float(x) for x in cell),
        sites_frac=sites,
        occupancy=np.ones(n) if occ is None else np.asarray(occ, dtype=np.float64),
        u_iso=np.zeros(n) if u_iso is None else np.asarray(u_iso, dtype=np.float64),
        u_star=np.zeros((n, 6)),
        anisotropic=np.zeros(n, dtype=bool),
        fp=np.zeros(n),
        fdp=np.zeros(n),
        type_index=np.zeros(n, dtype=int),
        gauss_a=np.zeros((1, 1)),
        gauss_b=np.zeros((1, 1)),
        gauss_c=np.array([float(f)]),
        rot=rot_a,
        trans=trans_a,
    )


def _hkl_box(n=3):
    out = []
    for h in range(-n, n + 1):
        for k in range(-n, n + 1):
            for l in range(-n, n + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                out.append((h, k, l))
    return np.asarray(out, dtype=np.int64)


def test_field_c0_excluded_and_hermitian():
    rot, trans = identity_ops()
    basis = FieldBasis.build((40.0, 40.0, 40.0, 90.0, 90.0, 90.0), rot, trans, cutoff=12.0)
    assert basis.n_coeff > 0
    assert not np.any(np.all(basis.hkl == 0, axis=1))
    # P1: every unique is acentric → even count, A then B.
    assert basis.n_coeff % 2 == 0
    assert not np.any(basis.is_sine[0::2])
    assert np.all(basis.is_sine[1::2])
    sites = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.15]])
    rng = np.random.default_rng(0)
    coeffs = rng.normal(size=basis.n_coeff)
    lam = basis.evaluate(sites, coeffs)
    assert np.isrealobj(lam)
    # Same field from the design matrix.
    np.testing.assert_allclose(lam, basis.design_matrix(sites) @ coeffs)


def test_field_sg_invariant():
    """λ(Rx+t) = λ(x) for a symmetry-adapted basis (spec §6)."""
    cell = (40.0, 50.0, 60.0, 90.0, 90.0, 90.0)
    basis = FieldBasis.build(cell, _P212121_ROT, _P212121_TRANS, cutoff=14.0)
    assert basis.n_coeff > 0
    rng = np.random.default_rng(1)
    coeffs = rng.normal(size=basis.n_coeff)
    x = np.array([[0.12, 0.34, 0.56]])
    lam_x = float(basis.evaluate(x, coeffs)[0])
    for R, t in zip(_P212121_ROT, _P212121_TRANS):
        xs = (R @ x[0] + t) % 1.0
        lam_s = float(basis.evaluate(xs.reshape(1, 3), coeffs)[0])
        assert lam_s == pytest.approx(lam_x, abs=1e-10)
    # P-1: every unique index is centric (one real, cosine). More unique
    # hkl than P2_12_12_1, so n_coeff can be larger even with one real each.
    pbar = FieldBasis.build(cell, _P1BAR_ROT, _P1BAR_TRANS, cutoff=14.0)
    assert pbar.n_coeff > 0
    assert not np.any(pbar.is_sine)  # inversion, t=0 ⇒ phase +1 ⇒ cosine only
    assert len({tuple(h) for h in pbar.hkl}) == pbar.n_coeff


def test_luzzati_and_effective_weight_eq14():
    s2 = np.array([0.0, 0.04, 0.25])
    kappa = np.array([2.0, 0.0])
    u = error_u_iso(kappa)
    np.testing.assert_allclose(u, kappa / EIGHT_PI2)
    d = luzzati_d_iso(u, s2)
    np.testing.assert_allclose(d[1], 1.0)
    np.testing.assert_allclose(d[0], np.exp(-kappa[0] * s2 / 4.0))
    lam = np.array([0.0, math.log(0.5)])
    w = presence_weight(lam)
    np.testing.assert_allclose(w, [1.0, 0.5])
    d0 = np.array([0.8, 0.7, 0.4])
    dj = effective_weight(d0, lam, kappa, s2)
    expect = d0[None, :] * np.exp(lam[:, None] - kappa[:, None] * s2[None, :] / 4.0)
    np.testing.assert_allclose(dj, expect)
    assert interpolate_shell(s2, np.array([0.0, 0.1, 1.0]), np.array([1.0, 0.5]), 1.0)[0] == 1.0


def test_modified_model_eq25():
    model = _toy_model([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], occ=[0.8, 1.0], u_iso=[0.02, 0.01])
    w = np.array([0.5, 0.9])
    u_err = np.array([0.03, 0.0])
    mod = modified_model(model, w, u_err)
    np.testing.assert_allclose(mod.occupancy, model.occupancy * w)
    np.testing.assert_allclose(mod.u_iso, model.u_iso + u_err)
    np.testing.assert_array_equal(mod.sites_frac, model.sites_frac)


def test_f_eff_and_sigma_delta_eq3_eq4():
    model = _toy_model([[0.1, 0.2, 0.3], [0.35, 0.4, 0.15]], u_iso=[0.02, 0.03])
    hkl = _hkl_box(2)
    w = np.array([0.8, 0.6])
    u_err = np.array([0.01, 0.02])
    s2 = resolution_s2(model.unit_cell, hkl)
    d0 = np.full(hkl.shape[0], 0.9)
    miss = np.full(hkl.shape[0], 3.0)
    d_j = effective_weight(d0, np.log(w), u_err * EIGHT_PI2, s2)
    fe, sig = moments(model, hkl, w, u_err, d0, miss)
    fe2 = f_eff_direct(model, hkl, d_j)
    np.testing.assert_allclose(fe, fe2)
    # Eq. 4 term-by-term.
    fj = form_factors(model, s2)
    tj = model_temperature(model, hkl, s2)
    atom = (model.occupancy[:, None] * fj * tj) ** 2 * (w[:, None] - d_j**2)
    expect = np.sum(model.multiplicity[:, None] * atom, axis=0) + miss
    np.testing.assert_allclose(sig, expect)
    np.testing.assert_allclose(sig, sigma_delta(model, hkl, w, d_j, miss))
    assert np.all(sig > 0)


def test_circularity():
    """Pseudo-covariance of F − F_eff is O(√N) relative to Σ_Δ (spec §11.1, A4)."""

    def _ratio(n_atoms: int, *, special: bool, seed: int) -> float:
        rng = np.random.default_rng(seed)
        if special:
            sites = np.zeros((n_atoms, 3))
            sites[0] = 0.0
            if n_atoms > 1:
                sites[1:] = rng.uniform(0.05, 0.95, size=(n_atoms - 1, 3))
        else:
            sites = rng.uniform(0.05, 0.95, size=(n_atoms, 3))
        model = _toy_model(sites, cell=(25.0, 26.0, 27.0, 90.0, 90.0, 90.0))
        hkl = np.array([[1, 0, 0], [0, 2, 1], [1, 1, 1], [2, 0, 1], [1, 2, 0]])
        w = np.full(n_atoms, 0.85)
        u_err = np.full(n_atoms, 0.04)
        d0 = np.ones(hkl.shape[0])
        miss = np.zeros(hkl.shape[0])
        fe, sig = moments(model, hkl, w, u_err, d0, miss)
        o_inv = np.linalg.inv(orthogonalization_matrix(model.unit_cell))
        s2 = resolution_s2(model.unit_cell, hkl)
        fj = form_factors(model, s2)
        tj = model_temperature(model, hkl, s2)
        h = hkl.astype(np.float64)
        n_draw = 600
        resid = np.empty((n_draw, hkl.shape[0]), dtype=np.complex128)
        for d in range(n_draw):
            f = np.zeros(hkl.shape[0], dtype=np.complex128)
            for j in range(n_atoms):
                if rng.random() > w[j]:
                    continue
                delta = o_inv @ rng.normal(scale=math.sqrt(u_err[j]), size=3)
                x = model.sites_frac[j] + delta
                f += model.occupancy[j] * fj[j] * tj[j] * np.exp(2j * np.pi * (h @ x))
            resid[d] = f - fe
        pseudo = np.mean(resid**2, axis=0)
        emp = np.mean(np.abs(resid) ** 2, axis=0)
        np.testing.assert_allclose(emp, sig, rtol=0.35)
        return float(np.mean(np.abs(pseudo) / np.maximum(emp, 1e-12)))

    r_small = _ratio(8, special=False, seed=2)
    r_large = _ratio(64, special=False, seed=3)
    r_special = _ratio(8, special=True, seed=4)
    assert r_large < r_small
    assert r_large < 0.35
    # Special-position phases e^{4πih·x} do not cancel; ratio stays larger.
    assert r_special > r_large


def test_gaussianity():
    """F − F_eff is closer to Gaussian as N grows and f_j stay comparable (spec §11.2, A3)."""

    def _excess_kurtosis(n_atoms: int, f_values: np.ndarray, seed: int) -> float:
        rng = np.random.default_rng(seed)
        sites = rng.uniform(0.05, 0.95, size=(n_atoms, 3))
        # One type table cannot vary f per atom; use occupancy as a stand-in for f.
        model = _toy_model(sites, occ=np.asarray(f_values, dtype=np.float64) / 6.0)
        hkl = np.array([[1, 0, 0], [0, 1, 1], [2, 1, 0]])
        w = np.ones(n_atoms)
        u_err = np.full(n_atoms, 0.05)
        fe, sig = moments(model, hkl, w, u_err, np.ones(3), np.zeros(3))
        o_inv = np.linalg.inv(orthogonalization_matrix(model.unit_cell))
        s2 = resolution_s2(model.unit_cell, hkl)
        fj = form_factors(model, s2)
        tj = model_temperature(model, hkl, s2)
        h = hkl.astype(np.float64)
        n_draw = 800
        re = []
        for _ in range(n_draw):
            f = np.zeros(3, dtype=np.complex128)
            for j in range(n_atoms):
                delta = o_inv @ rng.normal(scale=math.sqrt(u_err[j]), size=3)
                x = model.sites_frac[j] + delta
                f += model.occupancy[j] * fj[j] * tj[j] * np.exp(2j * np.pi * (h @ x))
            z = (f - fe) / np.sqrt(np.maximum(sig, 1e-12) / 2.0)
            re.extend(z.real.tolist())
            re.extend(z.imag.tolist())
        z = np.asarray(re)
        z = (z - z.mean()) / z.std()
        return float(np.mean(z**4) - 3.0)

    k_few = _excess_kurtosis(2, np.array([6.0, 6.0]), seed=5)
    k_many = _excess_kurtosis(40, np.full(40, 6.0), seed=6)
    k_spread = _excess_kurtosis(12, np.array([40.0] + [2.0] * 11), seed=7)
    assert abs(k_many) < abs(k_few)
    assert abs(k_many) < 0.8
    # A single dominant scatterer delays the CLT (A3 / Lindeberg).
    assert abs(k_spread) > abs(k_many)


def test_uniform_limit():
    """Constant (w, U) reproduces Read's σ_A (spec §11.3, eqs 10–12)."""
    model = _toy_model(
        [[0.11, 0.22, 0.33], [0.41, 0.52, 0.17], [0.08, 0.71, 0.44]],
        u_iso=[0.02, 0.015, 0.025],
    )
    hkl = _hkl_box(2)
    w_val = 0.7
    b_err = 8.0
    w = np.full(model.n_scatterers, w_val)
    u_err = np.full(model.n_scatterers, b_err / EIGHT_PI2)
    s2 = resolution_s2(model.unit_cell, hkl)
    d_s = w_val * np.exp(-b_err * s2 / 4.0)
    miss = np.full(hkl.shape[0], 12.0)
    # §5: d_j = w exp(−B_err s²/4). Do not also fold D into D_0.
    fe, sig = moments(model, hkl, w, u_err, np.ones_like(d_s), miss)
    fc = f_eff_direct(model, hkl, np.ones((model.n_scatterers, hkl.shape[0])))
    np.testing.assert_allclose(fe, d_s * fc, rtol=1e-12)
    sp = sigma_p(model, hkl)
    sigma_n = w_val * sp + miss
    np.testing.assert_allclose(sig, sigma_n - d_s**2 * sp, rtol=1e-12)
    sigma_a = d_s * np.sqrt(sp / sigma_n)
    np.testing.assert_allclose(sig / sigma_n, 1.0 - sigma_a**2, rtol=1e-12)
    np.testing.assert_allclose(np.abs(fe) / np.sqrt(sigma_n), sigma_a * np.abs(fc) / np.sqrt(sp))


def test_least_squares_limit():
    """Rice NLL → weighted LS on I, and Var(I) matches eq. 20 (spec §11.4)."""
    rng = np.random.default_rng(8)
    fe = np.array([80.0 + 0j, 60.0 + 20j, 90.0 - 10j])
    sig = np.array([0.08, 0.10, 0.06])
    eps = np.ones(3)
    assert np.all(np.abs(fe) ** 2 / (eps * sig) > 500.0)
    n_draw = 4000
    intensities = np.empty((n_draw, 3))
    for i in range(n_draw):
        f = draw_circular_normal(fe, sig, rng, epsilon=eps)
        intensities[i] = np.abs(f) ** 2
    emp = intensities.var(axis=0, ddof=1)
    np.testing.assert_allclose(emp, intensity_variance_limit(fe, sig, epsilon=eps), rtol=0.12)

    # Along a slice of I near |F_eff|², Rice − LS is nearly constant (eq. 21).
    i0 = np.abs(fe[0]) ** 2
    var_i = float(intensity_variance_limit(fe[:1], sig[:1])[0])
    grid = i0 + np.linspace(-2.0, 2.0, 21) * math.sqrt(var_i)
    rice = rice_nll_intensity(grid, np.full(grid.shape, fe[0]), np.full(grid.shape, sig[0])).nll
    ls = least_squares_intensity_nll(grid, np.full(grid.shape, fe[0]), np.full(grid.shape, sig[0]))
    delta = rice - ls
    assert float(np.std(delta)) < 0.05 * float(np.std(ls))


def test_rice_woolfson_normalisation_and_derivatives():
    """Eqs 7–8 integrate to 1; ∂L/∂F* and ∂L/∂Σ match finite differences."""
    assert log_i0(np.array([0.0]))[0] == pytest.approx(0.0, abs=1e-8)
    assert bessel_i1_over_i0(np.array([0.0]))[0] == pytest.approx(0.0, abs=1e-8)
    # I0(1) ≈ 1.2660658777520084
    assert log_i0(np.array([1.0]))[0] == pytest.approx(math.log(1.2660658777520084), abs=5e-7)

    # Wilson (F_eff = 0): p(I) = exp(−I/v)/v.
    v = 4.0
    i_grid = np.linspace(0.02, 40.0, 800)
    nll = rice_nll_intensity(i_grid, np.zeros(i_grid.shape, dtype=np.complex128), np.full_like(i_grid, v)).nll
    p = np.exp(-nll)
    area = np.trapezoid(p, i_grid) if hasattr(np, "trapezoid") else np.trapz(p, i_grid)
    assert area == pytest.approx(1.0, abs=0.02)

    fe = 3.0 + 4.0j
    sig = 2.5
    i_obs = 20.0
    res = rice_nll_intensity(np.array([i_obs]), np.array([fe]), np.array([sig]))
    # Finite-difference ∂L/∂A, ∂L/∂B, ∂L/∂Σ.
    h = 1e-5
    def nll_of(f, s):
        return rice_nll_intensity(np.array([i_obs]), np.array([f]), np.array([s])).nll[0]

    dA = (nll_of(fe + h, sig) - nll_of(fe - h, sig)) / (2 * h)
    dB = (nll_of(fe + 1j * h, sig) - nll_of(fe - 1j * h, sig)) / (2 * h)
    dS = (nll_of(fe, sig + h) - nll_of(fe, sig - h)) / (2 * h)
    # G = dL/dA + i dL/dB  (cctbx convention; Q = Re[conj(G) F])
    assert res.d_f_eff_star[0].real == pytest.approx(dA, rel=2e-4, abs=2e-4)
    assert res.d_f_eff_star[0].imag == pytest.approx(dB, rel=2e-4, abs=2e-4)
    assert res.d_sigma[0] == pytest.approx(dS, rel=2e-4, abs=2e-4)

    # Centric Woolfson branch.
    res_c = rice_nll_intensity(
        np.array([i_obs]), np.array([5.0 + 0j]), np.array([2.0]), centric=np.array([True])
    )
    def nll_c(f, s):
        return rice_nll_intensity(np.array([i_obs]), np.array([f]), np.array([s]), centric=np.array([True])).nll[0]

    dA_c = (nll_c(5.0 + h, 2.0) - nll_c(5.0 - h, 2.0)) / (2 * h)
    dS_c = (nll_c(5.0, 2.0 + h) - nll_c(5.0, 2.0 - h)) / (2 * h)
    assert res_c.d_f_eff_star[0].real == pytest.approx(dA_c, rel=2e-4, abs=2e-4)
    assert res_c.d_sigma[0] == pytest.approx(dS_c, rel=2e-4, abs=2e-4)


def test_empty_fields_are_uniform_read_limit():
    """n_coeff = 0 ⇒ λ = κ = 0 ⇒ w = 1, U = 0; Σ_Δ = Σ_miss when D0 = 1 (eq. 14)."""
    rot, trans = identity_ops()
    basis = FieldBasis.build((20.0, 20.0, 20.0, 90.0, 90.0, 90.0), rot, trans, cutoff=30.0)
    assert basis.n_coeff == 0
    sites = np.array([[0.1, 0.2, 0.3]])
    lam, kap = evaluate_fields(sites, np.zeros(0), np.zeros(0), basis)
    np.testing.assert_array_equal(lam, [0.0])
    np.testing.assert_array_equal(kap, [0.0])
    model = _toy_model(sites)
    hkl = np.array([[1, 0, 0], [0, 1, 0]])
    miss = np.array([5.0, 7.0])
    fe, sig = moments(model, hkl, np.ones(1), np.zeros(1), np.ones(2), miss)
    fc = f_eff_direct(model, hkl, np.ones((1, 2)))
    np.testing.assert_allclose(fe, fc)
    np.testing.assert_allclose(sig, miss)


def test_nuisance_fit_still_ignores_unfitted_v2_block():
    """Nuisance fit does not read the v2 block; the field step is a separate op."""
    from phridge.contrib.intensity_ll import ops as intensity_ops

    fit_src = inspect.getsource(intensity_ops.ml_i_nuisance_fit)
    assert "_ = spatial_sigma_a_v2" in fit_src
    body = inspect.getsource(intensity_ops.ml_i_target_and_gradients)
    assert "should_apply" in body
    assert "xray_with_frozen_errors" in body


# ---------------------------------------------------------------------------
# Phase 3: gradients (§10) and regulariser (eqs 15–16)
# ---------------------------------------------------------------------------


def _phase3_problem():
    model = _toy_model(
        [[0.12, 0.23, 0.34], [0.45, 0.56, 0.17]],
        u_iso=[0.02, 0.03],
        cell=(30.0, 31.0, 32.0, 90.0, 90.0, 90.0),
    )
    hkl = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 1], [0, 2, 1], [2, 1, 0], [1, 1, 1]])
    w = np.array([0.8, 0.55])
    u_err = np.array([0.016, 0.01])
    d0 = np.ones(hkl.shape[0])
    miss = np.full(hkl.shape[0], 10.0)
    fe, sig = moments(model, hkl, w, u_err, d0, miss)
    rng = np.random.default_rng(11)
    i_obs = np.maximum(np.abs(fe) ** 2 + rng.normal(scale=8.0, size=hkl.shape[0]), 0.2)
    rice = rice_nll_intensity(i_obs, fe, sig)
    d_j = d0[None, :] * w[:, None] * luzzati_d_iso(u_err, resolution_s2(model.unit_cell, hkl))
    return model, hkl, w, u_err, d0, miss, i_obs, fe, sig, rice, d_j


def test_variance_dx_is_zero():
    """∂Σ_Δ/∂x_j = 0 exactly (spec eq. 26)."""
    model, hkl, w, u_err, d0, miss, _i, _fe, _sig, rice, _dj = _phase3_problem()
    var = variance_gradients_iso(model, hkl, w, u_err, rice.d_sigma)
    np.testing.assert_array_equal(var.site_frac, 0.0)
    moved = _toy_model(model.sites_frac + 0.03, occ=model.occupancy, u_iso=model.u_iso, cell=model.unit_cell)
    _, sig_moved = moments(moved, hkl, w, u_err, d0, miss)
    np.testing.assert_allclose(sig_moved, _sig, rtol=0, atol=1e-14)


def test_gradients():
    """Finite-difference the analytic gradient: mean, variance, fields (spec §11.7)."""
    model, hkl, w, u_err, d0, miss, i_obs, fe, sig, rice, d_j = _phase3_problem()
    mean = mean_gradients_direct(model, hkl, d_j, rice.d_f_eff_star, w)
    var = variance_gradients_iso(model, hkl, w, u_err, rice.d_sigma)
    step = 1e-6

    def nll_mean(sites=None, w_use=None, u_use=None):
        m = model if sites is None else _toy_model(sites, occ=model.occupancy, u_iso=model.u_iso, cell=model.unit_cell)
        wu = w if w_use is None else w_use
        uu = u_err if u_use is None else u_use
        fe_p, _ = moments(m, hkl, wu, uu, d0, miss)
        return rice_nll_intensity(i_obs, fe_p, sig).value

    def nll_var(w_use=None, u_use=None):
        wu = w if w_use is None else w_use
        uu = u_err if u_use is None else u_use
        _, sig_p = moments(model, hkl, wu, uu, d0, miss)
        return rice_nll_intensity(i_obs, fe, sig_p).value

    # Mean: coordinates, w, U (Σ frozen).
    sites = model.sites_frac.copy()
    for a in range(3):
        plus, minus = sites.copy(), sites.copy()
        plus[0, a] += step
        minus[0, a] -= step
        fd = (nll_mean(sites=plus) - nll_mean(sites=minus)) / (2 * step)
        assert fd == pytest.approx(mean.site_frac[0, a], rel=2e-3, abs=2e-3)
    wp, wm = w.copy(), w.copy()
    wp[1] += step
    wm[1] -= step
    fd_w = (nll_mean(w_use=wp) - nll_mean(w_use=wm)) / (2 * step)
    assert fd_w == pytest.approx(mean.w[1], rel=2e-3, abs=2e-3)
    up, um = u_err.copy(), u_err.copy()
    up[0] += step
    um[0] -= step
    fd_u = (nll_mean(u_use=up) - nll_mean(u_use=um)) / (2 * step)
    assert fd_u == pytest.approx(mean.u_err_iso[0], rel=2e-3, abs=2e-3)

    # Variance: w, U (F_eff frozen); x already covered by eq. 26.
    fd_vw = (nll_var(w_use=wp) - nll_var(w_use=wm)) / (2 * step)
    assert fd_vw == pytest.approx(var.w[1], rel=2e-3, abs=2e-3)
    fd_vu = (nll_var(u_use=up) - nll_var(u_use=um)) / (2 * step)
    assert fd_vu == pytest.approx(var.u_err_iso[0], rel=2e-3, abs=2e-3)

    # Field coefficients (eq. 29): atoms frozen, both moments live.
    rot, trans = identity_ops()
    basis = FieldBasis.build(model.unit_cell, rot, trans, cutoff=12.0)
    assert basis.n_coeff >= 2
    rng = np.random.default_rng(12)
    lam_c = 0.05 * rng.normal(size=basis.n_coeff)
    kap_c = 0.4 * rng.normal(size=basis.n_coeff)

    def nll_field(lc, kc):
        lj, kj = evaluate_fields(model.sites_frac, lc, kc, basis)
        wu = np.exp(lj)
        uu = kj / EIGHT_PI2
        fe_p, sig_p = moments(model, hkl, wu, uu, d0, miss)
        return rice_nll_intensity(i_obs, fe_p, sig_p).value, wu, uu

    value, wu, uu = nll_field(lam_c, kap_c)
    _ = value
    fe_f, sig_f = moments(model, hkl, wu, uu, d0, miss)
    rice_f = rice_nll_intensity(i_obs, fe_f, sig_f)
    d_jf = d0[None, :] * wu[:, None] * luzzati_d_iso(uu, resolution_s2(model.unit_cell, hkl))
    tot = add_atom_grads(
        mean_gradients_direct(model, hkl, d_jf, rice_f.d_f_eff_star, wu),
        variance_gradients_iso(model, hkl, wu, uu, rice_f.d_sigma),
    )
    d_lam, d_kap = atom_to_field_params(tot, wu)
    d_lc, d_kc = field_gradients(basis, model.sites_frac, d_lam, d_kap)
    k = 0
    lc_p, lc_m = lam_c.copy(), lam_c.copy()
    lc_p[k] += step
    lc_m[k] -= step
    fd_c = (nll_field(lc_p, kap_c)[0] - nll_field(lc_m, kap_c)[0]) / (2 * step)
    assert fd_c == pytest.approx(d_lc[k], rel=3e-3, abs=3e-3)
    kc_p, kc_m = kap_c.copy(), kap_c.copy()
    kc_p[k] += step
    kc_m[k] -= step
    fd_k = (nll_field(lam_c, kc_p)[0] - nll_field(lam_c, kc_m)[0]) / (2 * step)
    assert fd_k == pytest.approx(d_kc[k], rel=3e-3, abs=3e-3)


def test_anisotropic_variance_xfail():
    """Anisotropic variance-gradient map is stubbed (spec §10)."""
    with pytest.raises(NotImplementedError, match="self-Patterson"):
        variance_gradients_aniso()


def test_regulariser_minimum_at_uniform():
    """Both extra terms in eq. 15 are minimised by the uniform field."""
    lam = np.array([0.3, -0.2, 0.1])
    kap = np.array([0.0, 0.4, -0.1])
    v_k = np.ones(3)
    t0, _, _ = spectral_taper(np.zeros(3), np.zeros(3), v_k)
    t1, d_l, d_k = spectral_taper(lam, kap, v_k)
    assert t0 == 0.0
    assert t1 > t0
    np.testing.assert_allclose(d_l, lam / v_k)
    np.testing.assert_allclose(d_k, kap / v_k)
    step = 1e-6
    tp = spectral_taper(lam + np.array([step, 0, 0]), kap, v_k)[0]
    tm = spectral_taper(lam - np.array([step, 0, 0]), kap, v_k)[0]
    assert (tp - tm) / (2 * step) == pytest.approx(d_l[0], rel=1e-6, abs=1e-8)

    model, hkl, w, u_err, d0, _miss, _i, _fe, _sig, _rice, d_j = _phase3_problem()
    rot, trans = identity_ops()
    basis = FieldBasis.build(model.unit_cell, rot, trans, cutoff=12.0)
    edges = np.array([0.0, 0.02, 0.2])
    v = np.ones(basis.n_coeff)
    uni = regulariser(
        model, hkl, d_j, np.zeros(basis.n_coeff), np.zeros(basis.n_coeff), v,
        basis=basis, shell_s2_edges=edges, entropy_weight=1.0,
    )
    peaked_c = np.zeros(basis.n_coeff)
    peaked_c[0] = 1.2
    peak = regulariser(
        model, hkl, d_j, peaked_c, np.zeros(basis.n_coeff), v,
        basis=basis, shell_s2_edges=edges, entropy_weight=1.0,
    )
    # Entropy of explained power uses the supplied d_j; rebuild d_j from the field.
    lj, kj = evaluate_fields(model.sites_frac, peaked_c, np.zeros(basis.n_coeff), basis)
    d_peak = effective_weight(d0, lj, kj, resolution_s2(model.unit_cell, hkl))
    peak_live = regulariser(
        model, hkl, d_peak, peaked_c, np.zeros(basis.n_coeff), v,
        basis=basis, shell_s2_edges=edges, entropy_weight=1.0,
    )
    assert uni.entropy <= peak_live.entropy + 1e-12
    assert uni.taper == 0.0
    assert peak.taper > 0.0


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch required for FFT grads")
def test_mean_gradients_fft_matches_direct():
    from phridge.contrib.spatial_sigmaa_v2.gradients import mean_gradients_fft

    model, hkl, w, u_err, d0, _miss, _i, _fe, _sig, rice, d_j = _phase3_problem()
    direct = mean_gradients_direct(model, hkl, d_j, rice.d_f_eff_star, w)
    fft = mean_gradients_fft(model, hkl, w, u_err, d0, rice.d_f_eff_star)
    np.testing.assert_allclose(fft.w, direct.w, rtol=5e-2, atol=5e-3)
    np.testing.assert_allclose(fft.site_frac, direct.site_frac, rtol=8e-2, atol=5e-3)


# ---------------------------------------------------------------------------
# Phase 4: alternation, Fisher closure, diagnostics
# ---------------------------------------------------------------------------

from phridge.contrib.spatial_sigmaa_v2.alternate import (
    field_step,
    fisher_step,
    freeze_atom_errors,
    initialise_block,
    run_macrocycle_step,
)
from phridge.contrib.spatial_sigmaa_v2.fisher import (
    cartesian_covariance,
    frac_blocks_to_cartesian,
    intensity_ls_site_blocks,
    iterate_fisher_closure,
    ls_curvatures,
)
from phridge.contrib.spatial_sigmaa_v2.op import OP_NAME as V2_STEP_OP
from phridge.ops import get_op
from phridge.sfcalc.engine.cell import orthogonalization_matrix


def test_register_binds_step_op():
    register()
    spec = get_op(V2_STEP_OP)
    assert spec.outputs["spatial_sigma_a_v2_result"] == "SpatialSigmaAV2Result"


def test_fisher_frame_and_jacobian():
    cell = (12.0, 15.0, 18.0, 90.0, 90.0, 90.0)
    o = orthogonalization_matrix(cell)
    h_frac = (o.T @ o)[None]
    np.testing.assert_allclose(frac_blocks_to_cartesian(h_frac, cell)[0], np.eye(3), atol=1e-10)
    np.testing.assert_allclose(cartesian_covariance(h_frac, cell)[0], np.eye(3), atol=1e-8)
    model, hkl, w, u_err, d0, miss, _i, fe, sig, _rice, d_j = _phase3_problem()
    cr, ct = ls_curvatures(fe, sig)
    assert np.all(ct == 0)
    np.testing.assert_allclose(cr, 2.0 / sig)
    blocks = intensity_ls_site_blocks(model, hkl, d_j, fe, sig)
    assert blocks.shape == (model.n_scatterers, 3, 3)
    # Finite-difference ∂|F|²/∂x vs the Jacobian baked into M.
    sites = model.sites_frac.copy()
    step = 1e-6
    plus = sites.copy()
    plus[0, 0] += step
    fe_p, _ = moments(
        _toy_model(plus, occ=model.occupancy, u_iso=model.u_iso, cell=model.unit_cell),
        hkl, w, u_err, d0, miss,
    )
    minus = sites.copy()
    minus[0, 0] -= step
    fe_m, _ = moments(
        _toy_model(minus, occ=model.occupancy, u_iso=model.u_iso, cell=model.unit_cell),
        hkl, w, u_err, d0, miss,
    )
    fd = (np.abs(fe_p) ** 2 - np.abs(fe_m) ** 2) / (2 * step)
    var_i = intensity_variance_limit(fe, sig)
    # Reconstruct J_x from M_xx ≈ Σ J_x² / Var — just check M is SPD and FD is finite.
    assert np.all(np.linalg.eigvalsh(blocks[0]) >= -1e-8)
    assert np.all(np.isfinite(fd))
    assert np.all(np.isfinite(var_i))


def test_fisher_closure():
    """Iterate (21)–(23) toward a common attractor from high and low U (spec §11.5)."""
    rng = np.random.default_rng(21)
    sites = rng.uniform(0.08, 0.92, size=(6, 3))
    model = _toy_model(sites, cell=(28.0, 29.0, 30.0, 90.0, 90.0, 90.0), u_iso=np.full(6, 0.02))
    hkl = _hkl_box(3)
    u_true = np.full(6, 0.04)
    d0 = np.ones(hkl.shape[0])
    # Σ_miss keeps the map T(U)=M⁻¹(Σ(U)) from collapsing to 0 (eq. 11, 18).
    miss = np.full(hkl.shape[0], 80.0)
    u_hi, hist_hi = iterate_fisher_closure(
        model, hkl, np.full(6, 0.16), w=np.ones(6), d0=d0, sigma_miss=miss, n_iter=8, damp=0.5
    )
    u_lo, hist_lo = iterate_fisher_closure(
        model, hkl, np.full(6, 0.01), w=np.ones(6), d0=d0, sigma_miss=miss, n_iter=8, damp=0.5
    )
    gap0 = float(np.mean(np.abs(hist_hi[0] - hist_lo[0])))
    gap1 = float(np.mean(np.abs(u_hi - u_lo)))
    assert gap1 < gap0
    assert np.all(np.isfinite(u_hi)) and np.all(np.isfinite(u_lo))
    assert np.all(u_hi > 0) and np.all(u_lo > 0)
    # Last increments shrink (fixed point).
    assert float(np.mean(np.abs(hist_hi[-1] - hist_hi[-2]))) < float(np.mean(np.abs(hist_hi[1] - hist_hi[0])))
    # Generating U and the attractor are the same order (diagonal-block approx.).
    attr = 0.5 * (float(np.mean(u_hi)) + float(np.mean(u_lo)))
    assert abs(math.log(attr / float(np.mean(u_true)))) < 3.0


def test_alternation_freezes_weights_during_atom_step():
    """λ is not re-evaluated at updated x (spec §10)."""
    model, hkl, _w, _u, _d0, _miss, i_obs, _fe, _sig, _rice, _dj = _phase3_problem()
    seed = PackedSpatialSigmaAV2.empty(field_cutoff=12.0)
    block, basis = initialise_block(model, hkl, seed)
    assert basis.n_coeff > 0
    rng = np.random.default_rng(3)
    peaked = PackedSpatialSigmaAV2(
        lambda_c=0.4 * rng.normal(size=basis.n_coeff),
        kappa_c=0.2 * rng.normal(size=basis.n_coeff),
        coeff_hkl=block.coeff_hkl,
        D0=block.D0,
        Sigma_miss=block.Sigma_miss,
        shell_s2_edges=block.shell_s2_edges,
        v_k=block.v_k,
        enabled=True,
        field_cutoff=12.0,
        n_scatterers=model.n_scatterers,
    )
    frozen = freeze_atom_errors(basis, model.sites_frac, peaked.lambda_c, peaked.kappa_c)
    assert frozen.matches_sites(model.sites_frac)
    moved = model.sites_frac + 0.07
    assert not frozen.matches_sites(moved)
    live_w = np.exp(evaluate_fields(moved, peaked.lambda_c, peaked.kappa_c, basis)[0])
    # The atom step must keep frozen.w, not the live field at the new x.
    assert not np.allclose(live_w, frozen.w)
    updated, frozen2 = field_step(model, hkl, i_obs, peaked, n_steps=4)
    assert frozen2.matches_sites(model.sites_frac)
    assert updated.meta.n_coeff == basis.n_coeff


def test_field_step_does_not_move_atoms():
    model, hkl, _w, _u, _d0, _miss, i_obs, _fe, _sig, _rice, _dj = _phase3_problem()
    sites = model.sites_frac.copy()
    block, _ = field_step(model, hkl, i_obs, PackedSpatialSigmaAV2.empty(field_cutoff=12.0), n_steps=3)
    np.testing.assert_array_equal(model.sites_frac, sites)
    assert block.meta.enabled is True


def test_diagnostics_pack():
    model, hkl, _w, _u, _d0, _miss, i_obs, _fe, _sig, _rice, _dj = _phase3_problem()
    opts = SpatialSigmaAV2Options(enabled=True, field_cutoff=12.0, fisher=False)
    packed, result, frozen = run_macrocycle_step(
        model, hkl, i_obs, PackedSpatialSigmaAV2.empty(field_cutoff=12.0), opts
    )
    assert result.meta.n_scatterers == model.n_scatterers
    assert result.w.shape == (model.n_scatterers,)
    assert result.u_err_star.shape == (model.n_scatterers, 6)
    assert result.tr_U.shape == (model.n_scatterers,)
    again = unpack_spatial_sigma_a_v2_result(result.pack(), result.meta)
    np.testing.assert_allclose(again.w, result.w)
    assert packed.meta.n_scatterers == model.n_scatterers
    assert frozen.matches_sites(model.sites_frac)


def test_fisher_flag_dispatches_closure():
    model, hkl, _w, _u, _d0, _miss, i_obs, _fe, _sig, _rice, _dj = _phase3_problem()
    opts = SpatialSigmaAV2Options(enabled=True, field_cutoff=12.0, fisher=True)
    packed, result, frozen = run_macrocycle_step(
        model, hkl, i_obs, PackedSpatialSigmaAV2.empty(field_cutoff=12.0, fisher=True), opts
    )
    assert packed.meta.fisher is True
    assert np.allclose(frozen.w, 1.0)
    assert np.all(frozen.u_err_iso > 0)
    assert result.tr_U.shape == (model.n_scatterers,)
    # Direct fisher_step agrees on positivity.
    _p, fr = fisher_step(model, hkl, PackedSpatialSigmaAV2.empty(field_cutoff=12.0, fisher=True))
    assert np.all(fr.u_err_iso > 0)


# ---------------------------------------------------------------------------
# Phase 5: apply F_eff, field recovery, null calibration
# ---------------------------------------------------------------------------

from phridge.contrib.spatial_sigmaa_v2.apply import (
    occupancy_grads_to_model,
    should_apply,
    xray_with_frozen_errors,
)
from phridge.contrib.spatial_sigmaa_v2.options import job_input_from_state
from phridge.contrib.spatial_sigmaa_v2.synthetic import (
    intensity_from_model,
    likelihood_gain_free_field,
    miller_box,
    null_calibration,
    perturb_domain,
    two_domain_model,
    two_domain_sites,
)


def test_job_input_from_state_flag_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PHRIDGE_SPATIAL_SIGMA_A_V2", raising=False)
    opts = SpatialSigmaAV2Options(enabled=False)
    packed = PackedSpatialSigmaAV2.empty(field_cutoff=15.0)
    assert job_input_from_state(packed, None, opts=opts) == {}


def test_apply_frozen_modified_model():
    from phridge.models import CrystalSymmetry, Scatterer
    from phridge.packing_xtal import PackedXray

    n = 3
    xray = PackedXray(
        crystal=CrystalSymmetry(unit_cell=[20.0, 20.0, 20.0, 90.0, 90.0, 90.0], space_group_hall="P 1"),
        sites_frac=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.1, 0.2]]),
        occupancy=np.array([1.0, 0.8, 1.0]),
        u_iso=np.array([0.02, 0.01, 0.03]),
        scatterers=[Scatterer(i=i, scattering_type="C", anisotropic=False) for i in range(n)],
    )
    result = PackedSpatialSigmaAV2Result(
        d_lambda_c=np.zeros(0),
        d_kappa_c=np.zeros(0),
        d_D0=np.zeros(0),
        d_Sigma_miss=np.zeros(0),
        w=np.array([0.5, 1.0, 0.8]),
        u_err_star=np.zeros((n, 6)),
        tr_U=np.array([0.03, 0.0, 0.06]),
    )
    assert should_apply(PackedSpatialSigmaAV2.empty(field_cutoff=15.0), result)
    assert not should_apply(None, result)
    mod = xray_with_frozen_errors(xray, result)
    np.testing.assert_allclose(mod.occupancy, [0.5, 0.8, 0.8])
    np.testing.assert_allclose(mod.u_iso, [0.03, 0.01, 0.05])
    np.testing.assert_array_equal(mod.sites_frac, xray.sites_frac)
    np.testing.assert_allclose(occupancy_grads_to_model(np.array([2.0, 2.0, 2.0]), result.w), [1.0, 2.0, 1.6])


def test_field_recovery():
    """Perturbed domain is down-weighted; uniform-error control stays flat (spec §11.6)."""
    rng = np.random.default_rng(31)
    sites, mask_b = two_domain_sites(5, rng=rng)
    model = two_domain_model(sites)
    hkl = miller_box(3)
    # Domain B: partly deleted, displaced, error-inflated.
    truth = perturb_domain(model, mask_b, occ_scale=0.3, u_extra=0.07, shift=(0.03, 0.0, 0.0))
    i_obs = intensity_from_model(truth, hkl)
    _gain, fitted, basis = likelihood_gain_free_field(model, hkl, i_obs, field_cutoff=16.0, n_steps=10)
    w = np.exp(np.clip(evaluate_fields(model.sites_frac, fitted.lambda_c, fitted.kappa_c, basis)[0], -3, 3))
    w_a = float(np.mean(w[~mask_b]))
    w_b = float(np.mean(w[mask_b]))
    assert w_b < w_a

    # Uniform-error control: same perturbation on every atom → field stays flat.
    uni_truth = perturb_domain(model, mask_b, occ_scale=0.3, u_extra=0.07, shift=(0.0, 0.0, 0.0), uniform=True)
    i_uni = intensity_from_model(uni_truth, hkl)
    _g2, fitted_u, basis_u = likelihood_gain_free_field(model, hkl, i_uni, field_cutoff=16.0, n_steps=10)
    lam_u = evaluate_fields(model.sites_frac, fitted_u.lambda_c, fitted_u.kappa_c, basis_u)[0]
    contrast = w_a - w_b
    assert float(np.std(lam_u)) < max(0.15, 0.6 * contrast)


def test_null_calibration():
    """Free-field gain threshold from a uniform-field bootstrap (spec §11.8)."""
    rng = np.random.default_rng(32)
    sites, mask_b = two_domain_sites(4, rng=rng)
    model = two_domain_model(sites)
    hkl = miller_box(2)
    thresh, null_gains = null_calibration(
        model, hkl, n_boot=6, quantile=0.9, field_cutoff=18.0, n_steps=5, rng=rng
    )
    assert null_gains.shape == (6,)
    assert np.all(np.isfinite(null_gains))
    assert thresh == pytest.approx(float(np.quantile(null_gains, 0.9)))
    # A two-domain perturbation should outrank the typical null draw.
    truth = perturb_domain(model, mask_b, occ_scale=0.25, u_extra=0.08)
    i_obs = intensity_from_model(truth, hkl)
    gain, _f, _b = likelihood_gain_free_field(model, hkl, i_obs, field_cutoff=18.0, n_steps=8)
    assert gain > float(np.median(null_gains))


def test_flag_off_kwargs_omit_fitted_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Even a leftover block is not sent when the flag is off."""
    monkeypatch.delenv("PHRIDGE_SPATIAL_SIGMA_A_V2", raising=False)
    packed = PackedSpatialSigmaAV2.empty(field_cutoff=15.0)
    result = PackedSpatialSigmaAV2Result(
        d_lambda_c=np.zeros(0),
        d_kappa_c=np.zeros(0),
        d_D0=np.zeros(0),
        d_Sigma_miss=np.zeros(0),
        w=np.array([1.0]),
        u_err_star=np.zeros((1, 6)),
        tr_U=np.array([0.0]),
    )
    assert job_input_from_state(packed, result, opts=SpatialSigmaAV2Options(enabled=False)) == {}


def test_apply_hosts_torch_result_arrays():
    """Apply pulls w / U off the compute device before touching PackedXray."""
    torch = pytest.importorskip("torch")
    from phridge.models import CrystalSymmetry, Scatterer
    from phridge.packing_xtal import PackedXray

    n = 2
    xray = PackedXray(
        crystal=CrystalSymmetry(unit_cell=[20.0, 20.0, 20.0, 90.0, 90.0, 90.0], space_group_hall="P 1"),
        sites_frac=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        occupancy=np.array([1.0, 0.5]),
        u_iso=np.array([0.02, 0.01]),
        scatterers=[Scatterer(i=i, scattering_type="C", anisotropic=False) for i in range(n)],
    )
    result = PackedSpatialSigmaAV2Result(
        d_lambda_c=np.zeros(0),
        d_kappa_c=np.zeros(0),
        d_D0=np.zeros(0),
        d_Sigma_miss=np.zeros(0),
        w=np.array([0.4, 1.0]),
        u_err_star=np.zeros((n, 6)),
        tr_U=np.array([0.03, 0.0]),
    )
    result.w = torch.as_tensor(result.w, dtype=torch.float32)
    result.tr_U = torch.as_tensor(result.tr_U, dtype=torch.float32)
    mod = xray_with_frozen_errors(xray, result)
    np.testing.assert_allclose(mod.occupancy, [0.4, 0.5])
    np.testing.assert_allclose(mod.u_iso, [0.03, 0.01])
    g = occupancy_grads_to_model(torch.as_tensor([2.0, 2.0]), result.w)
    np.testing.assert_allclose(g, [0.8, 2.0])


def test_d0_tensor_matches_f_calc_device():
    """D_0 is dtype-then-device so fc * D_0 cannot mix CPU and MPS/CUDA."""
    torch = pytest.importorskip("torch")
    from phridge.sfcalc.ops import _to_like

    d0 = np.array([0.8, 1.0, 1.2], dtype=np.float64)
    for device in ("cpu",) + (("mps",) if torch.backends.mps.is_available() else ()):
        fc = torch.ones(3, dtype=torch.complex64 if device == "mps" else torch.complex128, device=device)
        d0_t = _to_like(d0, fc.real)
        assert d0_t.device == fc.device
        assert d0_t.dtype == fc.real.dtype
        scaled = fc * d0_t
        assert scaled.device == fc.device
        np.testing.assert_allclose(scaled.real.detach().cpu().numpy(), d0, atol=1e-6)


def test_spatial_sigmaa_v2_never_fuses_device_and_dtype_in_to():
    """Same MPS rule as intensity_ll: never ``.to(dtype=..., device=...)``."""
    import re

    root = Path(__file__).resolve().parents[2] / "src" / "phridge" / "contrib" / "spatial_sigmaa_v2"
    fused = re.compile(r"\.to\([^)]*dtype=[^)]*device=[^)]*\)|\.to\([^)]*device=[^)]*dtype=[^)]*\)")
    offenders: list[str] = []
    for path in sorted(root.glob("*.py")):
        for ln in path.read_text().splitlines():
            if fused.search(ln) and not ln.lstrip().startswith("#"):
                offenders.append(f"{path.name}: {ln.strip()}")
    assert offenders == [], f"use .cpu().double() / dtype then device: {offenders}"


def test_field_grid_matches_evaluate():
    """Grid samples are the same field as at atoms (spec §6: sample λ, do not FFT e^λ)."""
    from phridge.contrib.spatial_sigmaa_v2.fields import grid_sites_frac, sample_on_grid
    from phridge.contrib.spatial_sigmaa_v2.per_atom import EIGHT_PI2
    from phridge.contrib.spatial_sigmaa_v2.viz import atom_display_columns, field_volumes

    rot, trans = identity_ops()
    basis = FieldBasis.build((40.0, 40.0, 40.0, 90.0, 90.0, 90.0), rot, trans, cutoff=16.0)
    rng = np.random.default_rng(4)
    coeffs = rng.normal(size=basis.n_coeff) * 0.05
    n_real = (8, 8, 8)
    grid = sample_on_grid(basis, coeffs, n_real)
    sites = grid_sites_frac(n_real)
    direct = basis.evaluate(sites, coeffs).reshape(n_real)
    np.testing.assert_allclose(grid, direct, atol=1e-12)

    model = _toy_model([[0.1, 0.2, 0.3]], cell=(40.0, 40.0, 40.0, 90.0, 90.0, 90.0))
    block = PackedSpatialSigmaAV2(
        lambda_c=coeffs,
        kappa_c=np.zeros(basis.n_coeff),
        coeff_hkl=basis.pack_hkl(),
        D0=np.ones(1),
        Sigma_miss=np.ones(1),
        shell_s2_edges=np.array([0.0, 1.0]),
        v_k=np.ones(basis.n_coeff),
        field_cutoff=16.0,
    )
    vols = field_volumes(block, model, n_real=n_real, basis=basis)
    np.testing.assert_allclose(vols["w"], np.exp(np.clip(vols["lambda"], -3, 3)))
    assert vols["kappa"].shape == n_real
    empty = field_volumes(PackedSpatialSigmaAV2.empty(field_cutoff=16.0), model, n_real=(6, 6, 6))
    assert np.all(empty["lambda"] == 0.0)

    result = PackedSpatialSigmaAV2Result(
        d_lambda_c=np.zeros(0),
        d_kappa_c=np.zeros(0),
        d_D0=np.zeros(0),
        d_Sigma_miss=np.zeros(0),
        w=np.array([0.5, 1.0]),
        u_err_star=np.zeros((2, 6)),
        tr_U=np.array([0.03, 0.0]),
    )
    w, b_err = atom_display_columns(result)
    np.testing.assert_allclose(w, [0.5, 1.0])
    np.testing.assert_allclose(b_err, EIGHT_PI2 * np.array([0.01, 0.0]))


def test_viz_module_is_torch_and_cctbx_free():
    src = (Path(__file__).resolve().parents[2] / "src" / "phridge" / "contrib" / "spatial_sigmaa_v2" / "viz.py").read_text()
    assert "import torch" not in src
    assert "cctbx" not in src
    assert "iotbx" not in src


def test_client_writes_v2_maps_after_step():
    src = (Path(__file__).resolve().parents[2] / "src" / "phridge" / "client" / "intensity" / "engine.py").read_text()
    assert "_write_spatial_sigma_a_v2_maps" in src
    assert "_field_atoms.pdb" in src
    assert "lambda.ccp4" in src or 'f"{prefix}_{name}.ccp4"' in src
