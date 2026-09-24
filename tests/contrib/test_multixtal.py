"""phridge.contrib.multixtal — registration, Friedel identities, pipeline limits, influence."""

from __future__ import annotations

import math

import numpy as np
import pytest
from pydantic import ValidationError

from phridge.contrib.multixtal import MultixtalOptions, OP_NAME, register
from phridge.contrib.multixtal.friedel import (
    bijvoet_intensity_difference,
    bijvoet_intensity_sum,
    reconstruct_f,
    s_g_from_real_and_full,
    split_s_g,
)
from phridge.contrib.multixtal.influence import ClassicalInfluence, effective_n
from phridge.contrib.multixtal.selector import PopulationSelector, per_dataset_selector
from phridge.ops import get_op
from phridge.sfcalc.engine.engine import EngineParams, ScatteringModel, StructureFactorEngine
from phridge.sfcalc.engine.symmetry import identity_ops


class _Crystal:
    space_group_number = 1
    space_group_hall = "P 1"


def _tiny_model(*, fdp: float = 0.0, n: int = 1) -> ScatteringModel:
    rot, trans = identity_ops()
    rng = np.random.default_rng(1)
    sites = np.array([[0.12, 0.21, 0.33]], dtype=np.float64)
    if n > 1:
        extra = 0.3 + 0.4 * rng.random((n - 1, 3))
        sites = np.vstack([sites, extra])
    n = sites.shape[0]
    return ScatteringModel(
        unit_cell=(22.0, 20.0, 24.0, 90.0, 90.0, 90.0),
        sites_frac=sites,
        occupancy=np.ones(n),
        u_iso=np.full(n, 0.05),
        u_star=np.zeros((n, 6)),
        anisotropic=np.zeros(n, dtype=bool),
        fp=np.zeros(n),
        fdp=np.array([fdp] + [0.0] * (n - 1), dtype=np.float64),
        type_index=np.zeros(n, dtype=np.int64),
        gauss_a=np.array([[8.0, 0.0, 0.0, 0.0]]),
        gauss_b=np.array([[12.0, 0.0, 0.0, 0.0]]),
        gauss_c=np.array([1.5]),
        rot=rot,
        trans=trans,
    )


def test_register_is_idempotent():
    register()
    register()
    assert get_op(OP_NAME).name == OP_NAME


def test_pydantic_accept_reject():
    MultixtalOptions()
    MultixtalOptions(rank="auto", anomalous="on", n_spline_knots=8)
    MultixtalOptions(rank=2, anomalous=False)
    with pytest.raises(ValidationError):
        MultixtalOptions(strong_isig=0.0)
    with pytest.raises(ValidationError):
        MultixtalOptions(rank="maybe")
    with pytest.raises(ValidationError):
        MultixtalOptions(cell=[1.0, 2.0])


def test_friedel_convention():
    model = _tiny_model(fdp=2.5)
    plus = np.array([[1, 0, 0], [1, 1, 0], [2, 0, 1], [0, 2, 1], [1, 1, 1]], dtype=np.int64)
    hkl = np.vstack([plus, -plus])
    params = EngineParams(d_min=1.8, dtype="float64")
    eng = StructureFactorEngine(model, hkl, params, device="cpu")
    import torch

    sites, occ, u_iso, u_star, fp, fdp = eng.tensors()
    with torch.no_grad():
        f_full = eng.f_calc(sites, occ, u_iso, u_star, fp, fdp)
        f_real = eng.f_calc(sites, occ, u_iso, u_star, fp, torch.zeros_like(fdp))
    f_full_np = f_full.cpu().numpy()
    f_real_np = f_real.cpu().numpy()
    n = plus.shape[0]
    f_plus, f_minus = f_full_np[:n], f_full_np[n:]
    s, g = s_g_from_real_and_full(f_real_np[:n], f_plus)
    rec_plus, rec_minus = reconstruct_f(s, g)
    np.testing.assert_allclose(rec_plus, f_plus, atol=1e-8, rtol=1e-6)
    np.testing.assert_allclose(rec_minus, f_minus, atol=1e-8, rtol=1e-6)
    s2, g2 = split_s_g(f_plus, f_minus)
    np.testing.assert_allclose(s2, s, atol=1e-8, rtol=1e-6)
    np.testing.assert_allclose(g2, g, atol=1e-8, rtol=1e-6)
    i_plus = np.abs(f_plus) ** 2
    i_minus = np.abs(f_minus) ** 2
    np.testing.assert_allclose(i_plus - i_minus, bijvoet_intensity_difference(s, g), atol=1e-8, rtol=1e-6)
    np.testing.assert_allclose(i_plus + i_minus, bijvoet_intensity_sum(s, g), atol=1e-8, rtol=1e-6)
    rec_plus2, rec_minus2 = reconstruct_f(s2, g2)
    np.testing.assert_allclose(rec_plus2, f_plus, atol=1e-10)
    np.testing.assert_allclose(np.conj(rec_minus2), s2 - 1.0j * g2, atol=1e-10)


def _simulate_panel(
    n_data: int = 5,
    n_h: int = 40,
    rank: int = 1,
    seed: int = 0,
    outlier: bool = False,
    duplicate_first: bool = False,
) -> dict:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n_data, rank))
    if rank:
        z = (z - z.mean(axis=0)) / np.maximum(z.std(axis=0), 1e-6)
    l = rng.normal(size=(rank, n_h))
    mu = 0.1 * rng.normal(size=n_h)
    noise = 0.15 * rng.normal(size=(n_data, n_h))
    y = mu[None, :] + (z @ l if rank else 0.0) + noise
    if outlier:
        y[-1] += 3.0 * rng.normal(size=n_h)
    if duplicate_first:
        y = np.vstack([y, y[0]])
        z = np.vstack([z, z[0]]) if rank else z
        n_data += 1
    w = np.ones((n_data, n_h))
    return {"y": y, "w": w, "mu": mu, "z": z, "l": l, "n_data": n_data}


def test_influence_shares_sum_to_one():
    panel = _simulate_panel()
    shares = ClassicalInfluence().information_shares(panel["w"])
    np.testing.assert_allclose(shares.sum(), 1.0, atol=1e-12)
    assert math.isclose(effective_n(shares), float(panel["n_data"]), rel_tol=1e-6)


def test_influence_duplicate_dataset_doubles_share():
    base = _simulate_panel(n_data=4, seed=2)
    shares0 = ClassicalInfluence().information_shares(base["w"]).reshape(-1)
    dup = _simulate_panel(n_data=4, seed=2, duplicate_first=True)
    shares1 = ClassicalInfluence().information_shares(dup["w"]).reshape(-1)
    # First and last (duplicate) together ≈ 2 × original first share, after renormalization
    # of 5 equal-weight datasets: each has 1/5; original each had 1/4. Combined first+last = 2/5.
    assert shares1[0] + shares1[-1] == pytest.approx(2.0 / 5.0, abs=1e-9)


def test_influence_zero_weight_cooks_near_zero():
    from phridge.contrib.multixtal.factors import weighted_mean
    from phridge.contrib.multixtal.influence import analyze_influence

    panel = _simulate_panel(n_data=4, rank=0, seed=3)
    y, w = panel["y"], panel["w"]
    w[-1] = 0.0
    mu = weighted_mean(y, w)
    from phridge.contrib.multixtal.explain import explain_datasets

    expl = explain_datasets(y, w, mu, np.zeros((4, 0)), np.zeros((0, y.shape[1])), w)
    inf = analyze_influence(y, w, w, mu, np.zeros((4, 0)), np.zeros((0, y.shape[1])), expl.chi2_loo)
    assert inf[0].cooks[-1] == pytest.approx(0.0, abs=1e-8)


def test_selection_bias_guard():
    rng = np.random.default_rng(4)
    n_data, n_h = 10, 80
    z = np.linspace(-2.0, 2.0, n_data)
    l_true = np.zeros(n_h)
    l_true[20:50] = 1.0
    # Non-mode reflections are strong. Mode reflections have median I/σ just
    # above the population cut, so the shared set keeps them, but a per-dataset
    # I/σ cut drops the low-score datasets and biases the recovered loading.
    base = 6.6 + 0.05 * rng.normal(size=(n_data, n_h))
    base[:, ~l_true.astype(bool)] = 20.0
    i_bar = base + np.outer(z, 1.6 * l_true)
    sig = np.full_like(i_bar, 1.2)
    observed = np.ones_like(i_bar, dtype=bool)
    pop = PopulationSelector(strong_isig=5.0, strong_min_datasets=0.5).select(i_bar, sig, observed)
    wrong = per_dataset_selector(i_bar, sig, observed, strong_isig=5.0)
    assert pop.any() and wrong.any()

    def _loadings(mask) -> np.ndarray:
        hat = np.zeros(n_h)
        for h in range(n_h):
            sel = mask[:, h] if mask.ndim == 2 else (np.ones(n_data, dtype=bool) if mask[h] else np.zeros(n_data, dtype=bool))
            if int(sel.sum()) < 3:
                continue
            hat[h] = float(np.polyfit(z[sel], i_bar[sel, h], 1)[0])
        return hat

    hat_pop = _loadings(pop)
    hat_wrong = _loadings(wrong)
    mode = l_true > 0
    assert pop[mode].all()
    assert wrong[:, mode].any() and (~wrong[:, mode]).any()
    bias_wrong = abs(float(np.mean(hat_wrong[mode])) - 1.6)
    bias_pop = abs(float(np.mean(hat_pop[mode])) - 1.6)
    assert bias_wrong > bias_pop


def test_limits_n1_rank0_no_anomalous():
    pytest.importorskip("torch")
    from phridge.contrib.multixtal.options import MultixtalOptions
    from phridge.contrib.multixtal.pipeline import run_multixtal_core
    from phridge.contrib.multixtal.structure import f_calc_s_g

    model = _tiny_model(fdp=0.0, n=3)
    hkl = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [2, 0, 0], [0, 2, 0]], dtype=np.int64)
    params = EngineParams(d_min=2.0, dtype="float64")
    f, _s, _g = f_calc_s_g(model, hkl, params=params)
    i = np.abs(f) ** 2
    n_h = hkl.shape[0]
    i_plus = i[None, :].copy()
    i_minus = i[None, :].copy()
    sig = np.full((1, n_h), 0.15 * np.median(i) + 1.0)
    mask = np.ones((1, n_h), dtype=bool)
    f_mask = np.zeros((1, n_h), dtype=np.complex128)
    eps = np.ones(n_h)
    cen = np.zeros(n_h, dtype=bool)
    opts = MultixtalOptions(
        d_min=2.0,
        rank=0,
        anomalous=False,
        n_outer=1,
        max_lbfgs_iter=8,
        n_spline_knots=6,
        rank_perms=4,
        strong_isig=0.1,
        strong_min_datasets=1.0,
    )
    result = run_multixtal_core(
        model,
        hkl,
        np.array([model.unit_cell], dtype=np.float64),
        model.sites_frac[None, :, :],
        i_plus,
        i_minus,
        sig,
        sig,
        mask,
        mask,
        f_mask,
        eps,
        cen,
        _Crystal(),
        opts,
        engine_params=params,
        rng=np.random.default_rng(0),
    )
    assert result.factors.rank == 0
    assert result.y.shape[0] == 1
    assert result.anomalous is None
    assert result.nuisances[0].k_sol >= 0.0
    assert result.nuisances[0].sigma_a.shape == (n_h,)


def test_rank_zero_is_population_mean_only():
    from phridge.contrib.multixtal.factors import fit_factors

    panel = _simulate_panel(n_data=5, rank=1, seed=5)
    fit = fit_factors(panel["y"], panel["w"], rank=0, rank_perms=8, rng=np.random.default_rng(5))
    assert fit.rank == 0
    assert fit.scores.shape[1] == 0
    assert fit.loadings.shape[0] == 0
    np.testing.assert_allclose(fit.mu, panel["y"].mean(axis=0), atol=0.2)


def test_no_anomalous_intercept_near_zero():
    from phridge.contrib.multixtal.anomalous import BijvoetRegression

    rng = np.random.default_rng(6)
    n_data, n_h = 6, 30
    d_e2 = 0.05 * rng.normal(size=(n_data, n_h))
    z = rng.normal(size=(n_data, 1))
    w = np.ones((n_data, n_h))
    obs = np.ones((n_data, n_h), dtype=bool)
    intercept, slopes = BijvoetRegression().regress(d_e2, z, w, obs)
    assert float(np.mean(np.abs(intercept))) < 0.1


def test_spline_positive_and_partition():
    from phridge.contrib.multixtal.spline import bspline_design_numpy, eval_log_spline, open_uniform_knots

    x = np.linspace(0.0, 0.25, 40)
    knots = open_uniform_knots(0.0, 0.25, 6)
    design = bspline_design_numpy(x, knots)
    np.testing.assert_allclose(design.sum(axis=1), 1.0, atol=1e-6)
    y = eval_log_spline(x, knots, np.zeros(6))
    np.testing.assert_allclose(y, 1.0, atol=1e-6)
    assert np.all(y > 0)


def test_manifest_json(tmp_path):
    from phridge.contrib.multixtal.client import load_manifest
    from phridge.contrib.multixtal.options import DatasetManifest

    path = tmp_path / "datasets.json"
    path.write_text('{"datasets": [{"file": "a.mtz", "labels": "I(+),SIGI(+),I(-),SIGI(-)", "wavelength": 0.98}]}')
    man = load_manifest(path)
    assert len(man.datasets) == 1
    assert man.datasets[0].file == "a.mtz"
    DatasetManifest.from_mapping([{"file": "b.mtz"}])


def test_cli_parser():
    from phridge.contrib.multixtal.cli import build_parser, options_from_args

    args = build_parser().parse_args(
        ["--model", "m.pdb", "--data", "a.mtz", "b.mtz", "--out", "out", "--rank", "auto", "--anomalous", "off"]
    )
    opts = options_from_args(args)
    assert opts.rank == "auto"
    assert opts.anomalous is False


def test_op_wire_n1():
    pytest.importorskip("torch")
    from phridge.contrib.multixtal.op import multixtal_fit
    from phridge.contrib.multixtal.structure import f_calc_s_g
    from phridge.models import CrystalSymmetry, Scatterer, SfEngineParams, SymOp
    from phridge.packing_xtal import PackedXray
    from phridge.sfcalc.packing import PackedScatteringTable

    model = _tiny_model(fdp=0.0, n=2)
    hkl = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=np.int64)
    params = EngineParams(d_min=2.0, dtype="float64")
    f, _s, _g = f_calc_s_g(model, hkl, params=params)
    i = np.abs(f) ** 2
    n_h = hkl.shape[0]
    crystal = CrystalSymmetry(
        unit_cell=list(model.unit_cell),
        space_group_hall="P 1",
        space_group_number=1,
        symops=[SymOp(r=[1, 0, 0, 0, 1, 0, 0, 0, 1], t=[0, 0, 0])],
    )
    scatterers = [
        Scatterer(i=j, scattering_type="C", fdp=None, anisotropic=False)
        for j in range(model.n_scatterers)
    ]
    xray = PackedXray(
        crystal=crystal,
        sites_frac=model.sites_frac,
        occupancy=model.occupancy,
        u_iso=model.u_iso,
        scatterers=scatterers,
        u_star=model.u_star,
    )
    table = PackedScatteringTable(["C"], model.gauss_a, model.gauss_b, model.gauss_c)
    out = multixtal_fit(
        xray=xray,
        table=table,
        params=SfEngineParams(d_min=2.0, dtype="float64"),
        sites_frac=model.sites_frac[None, :, :],
        cells=np.array([model.unit_cell], dtype=np.float64),
        hkl=hkl,
        centric=np.zeros(n_h, dtype=bool),
        epsilon=np.ones(n_h),
        i_plus=i[None, :],
        sig_plus=np.full((1, n_h), 0.2 * np.median(i) + 1.0),
        i_minus=i[None, :],
        sig_minus=np.full((1, n_h), 0.2 * np.median(i) + 1.0),
        mask_plus=np.ones((1, n_h), dtype=bool),
        mask_minus=np.ones((1, n_h), dtype=bool),
        f_mask=np.zeros((1, n_h), dtype=np.complex128),
        options={"rank": 0, "anomalous": False, "n_outer": 1, "max_lbfgs_iter": 6, "n_spline_knots": 6, "strong_isig": 0.1, "rank_perms": 4},
    )
    assert out["y"].shape[0] == 1
    assert out["scores"].shape[1] == 0
    assert out["stats"]["n_datasets"] == 1


def test_client_import_is_torch_free():
    import phridge.contrib.multixtal.client as client
    import phridge.contrib.multixtal.options as options

    assert "torch" not in client.__dict__
    MultixtalOptions()
    assert options.MultixtalOptions is MultixtalOptions


def test_score_recovery_and_outlier_flag():
    pytest.importorskip("torch")
    from phridge.contrib.multixtal.options import MultixtalOptions
    from phridge.contrib.multixtal.pipeline import run_multixtal_core
    from phridge.contrib.multixtal.structure import f_calc_s_g

    model = _tiny_model(fdp=1.2, n=4)
    hkl = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [2, 0, 0], [0, 2, 0], [1, 1, 1], [2, 1, 0]],
        dtype=np.int64,
    )
    params = EngineParams(d_min=2.0, dtype="float64")
    f, _s, _g = f_calc_s_g(model, hkl, params=params)
    i0 = np.abs(f) ** 2
    n_h = hkl.shape[0]
    n_data = 6
    z_true = np.linspace(-1.4, 1.4, n_data)
    loading = np.sin(np.linspace(0.0, 3.0, n_h))
    rng = np.random.default_rng(8)
    i_plus = np.zeros((n_data, n_h))
    i_minus = np.zeros((n_data, n_h))
    for d in range(n_data):
        scale = 1.0 + 0.35 * z_true[d] * loading
        i_plus[d] = np.maximum(i0 * scale + rng.normal(scale=0.04 * i0, size=n_h), 0.1)
        i_minus[d] = np.maximum(i0 * scale + rng.normal(scale=0.04 * i0, size=n_h), 0.1)
    i_plus[-1] *= 2.4
    i_minus[-1] *= 2.4
    sig = 0.06 * np.maximum(i_plus, 1.0)
    mask = np.ones((n_data, n_h), dtype=bool)
    opts = MultixtalOptions(
        d_min=2.0,
        rank=1,
        anomalous=True,
        n_outer=1,
        max_lbfgs_iter=8,
        n_spline_knots=6,
        rank_perms=12,
        strong_isig=0.5,
        strong_min_datasets=0.5,
    )
    result = run_multixtal_core(
        model,
        hkl,
        np.tile(np.array(model.unit_cell, dtype=np.float64), (n_data, 1)),
        np.broadcast_to(model.sites_frac[None, :, :], (n_data, model.sites_frac.shape[0], 3)).copy(),
        i_plus,
        i_minus,
        sig,
        sig,
        mask,
        mask,
        np.zeros((n_data, n_h), dtype=np.complex128),
        np.ones(n_h),
        np.zeros(n_h, dtype=bool),
        _Crystal(),
        opts,
        engine_params=params,
        rng=np.random.default_rng(0),
    )
    assert result.factors.rank >= 1
    z_hat = result.factors.scores[:, 0]
    corr = abs(float(np.corrcoef(z_hat[:-1], z_true[:-1])[0, 1]))
    assert corr > 0.7
    assert result.explanation.chi2_in_sample[-1] < result.explanation.chi2_loo[-1] or bool(result.explanation.flagged[-1])


def test_synthetic_if_cctbx(tmp_path):
    pytest.importorskip("cctbx")
    from phridge.contrib.multixtal.synthetic import write_synthetic_benchmark

    truth = write_synthetic_benchmark(tmp_path, n_datasets=4, d_min=3.0, seed=1, include_outlier=True)
    assert truth.pdb_path.is_file()
    assert len(truth.mtz_paths) == 4
    assert truth.outlier_index == 3
    assert truth.occupancies.shape == (4,)
