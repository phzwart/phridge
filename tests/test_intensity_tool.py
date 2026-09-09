"""Unit and integration tests for phridge intensity tool."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

cctbx = pytest.importorskip("cctbx")
from cctbx import sgtbx
from cctbx.array_family import flex
from cctbx.development import random_structure
import iotbx.ccp4_map
import iotbx.pdb
from iotbx.reflection_file_reader import any_reflection_file

from phridge.client.intensity_tool import (
    IntensityModel,
    from_files,
    isotonic_non_increasing,
    main,
    run_intensity_pipeline,
    tv_denoise_1d,
)


def _make_synthetic_model(space_group="P21", n_atoms=15, d_min=2.5, seed=42):
    flex.set_random_seed(seed)
    np.random.seed(seed)
    xs = random_structure.xray_structure(
        space_group_info=sgtbx.space_group_info(space_group),
        elements=["C", "N", "O"] * (n_atoms // 3),
        volume_per_atom=50.0,
        random_u_iso=True,
    )
    fc = xs.structure_factors(d_min=d_min).f_calc()
    # Synthetic intensities with small noise
    iobs_data = flex.abs(fc.data()) ** 2
    # Add 5% Gaussian noise to intensities
    noise = np.random.normal(0.0, 0.05, size=fc.size()) * np.asarray(iobs_data)
    noisy_i = np.maximum(np.asarray(iobs_data) + noise, 0.1)
    sigmas = flex.double(np.maximum(0.05 * noisy_i, 0.05).tolist())

    i_obs = fc.customized_copy(data=flex.double(noisy_i.tolist()), sigmas=sigmas)
    i_obs.set_observation_type_xray_intensity()
    r_free = i_obs.generate_r_free_flags(fraction=0.1)

    pdb_str = xs.as_pdb_file()
    pdb_inp = iotbx.pdb.input(source_info=None, lines=pdb_str.split("\n"))
    hierarchy = pdb_inp.construct_hierarchy()

    return xs, i_obs, hierarchy, r_free


def test_intensity_model_init_and_binning():
    xs, i_obs, hierarchy, r_free = _make_synthetic_model()
    model = IntensityModel(xs, i_obs, hierarchy=hierarchy, r_free_flags=r_free, n_bins=5)

    assert model.i_obs.size() == i_obs.size()
    assert model.mean_i_per_refl is not None
    assert len(model.mean_i_per_refl) == i_obs.size()
    assert np.all(np.asarray(model.mean_i_per_refl) > 0.0)


def test_scale_factor_recovery():
    xs, i_obs, hierarchy, r_free = _make_synthetic_model()
    true_scale = 3.5
    # Scale observed intensities by true_scale^2
    scaled_data = flex.double((np.asarray(i_obs.data()) * (true_scale**2)).tolist())
    scaled_sigmas = flex.double((np.asarray(i_obs.sigmas()) * (true_scale**2)).tolist())
    i_obs_scaled = i_obs.customized_copy(data=scaled_data, sigmas=scaled_sigmas)

    model = IntensityModel(xs, i_obs_scaled, hierarchy=hierarchy, r_free_flags=r_free, n_bins=5)
    scales = model.refine_scale_and_solvent()

    # Should recover scale close to true_scale
    assert np.isclose(scales["k_total"], true_scale, rtol=0.15)
    assert np.isclose(model.k_total, true_scale, rtol=0.15)


def test_bulk_solvent_refinement():
    xs, i_obs, hierarchy, r_free = _make_synthetic_model()
    model = IntensityModel(
        xs,
        i_obs,
        hierarchy=hierarchy,
        r_free_flags=r_free,
        use_bulk_solvent=True,
        n_bins=5,
    )
    scales = model.refine_scale_and_solvent()

    assert model.f_mask is not None
    assert model.f_mask.size() == i_obs.size()
    assert scales["k_sol"] >= 0.0
    assert scales["b_sol"] >= 0.0


def test_tv_denoise_1d():
    # Erratic sequence
    y = np.array([0.9, 0.5, 0.85, 0.4, 0.8, 0.3])
    raw_tv = np.sum(np.abs(np.diff(y)))

    # TV regularized with different strengths
    denoised_light = tv_denoise_1d(y, lam=0.05)
    denoised_heavy = tv_denoise_1d(y, lam=0.5)

    tv_light = np.sum(np.abs(np.diff(denoised_light)))
    tv_heavy = np.sum(np.abs(np.diff(denoised_heavy)))

    assert tv_light < raw_tv
    assert tv_heavy < tv_light
    assert np.all(denoised_heavy >= 0.01)
    assert np.all(denoised_heavy <= 0.999)


def test_isotonic_non_increasing():
    y = np.array([0.5, 0.8, 0.6, 0.9, 0.4, 0.7])
    mono = isotonic_non_increasing(y)

    assert len(mono) == len(y)
    diffs = np.diff(mono)
    # Must be non-increasing everywhere
    assert np.all(diffs <= 1e-9)
    # Bounds preserved
    assert np.min(mono) >= np.min(y) - 1e-9
    assert np.max(mono) <= np.max(y) + 1e-9


def test_sigma_a_estimation():
    xs, i_obs, hierarchy, r_free = _make_synthetic_model()
    model = IntensityModel(xs, i_obs, hierarchy=hierarchy, r_free_flags=r_free, n_bins=5)

    # 1. Default (overlapping bins = 1)
    sa_bins = model.estimate_sigma_a()
    assert len(sa_bins) == 5
    for b_idx, sa_val in sa_bins.items():
        assert 0.01 <= sa_val <= 1.0
    assert model.sigma_a_per_refl is not None
    assert len(model.sigma_a_per_refl) == i_obs.size()

    # 2. Disjoint bins (overlap = 0)
    sa_disjoint = model.estimate_sigma_a(overlap=0)
    assert len(sa_disjoint) == 5
    for b_idx, sa_val in sa_disjoint.items():
        assert 0.01 <= sa_val <= 1.0

    # 3. TV regularized
    sa_tv = model.estimate_sigma_a(overlap=1, tv_lambda=0.1)
    assert len(sa_tv) == 5
    for b_idx, sa_val in sa_tv.items():
        assert 0.01 <= sa_val <= 1.0

    # 4. Monotonic enforcement
    sa_mono = model.estimate_sigma_a(overlap=1, enforce_monotonic=True)
    vals_mono = [sa_mono[i] for i in sorted(sa_mono.keys())]
    assert np.all(np.diff(vals_mono) <= 1e-9)


def test_nu_estimation():
    xs, i_obs, hierarchy, r_free = _make_synthetic_model()
    model = IntensityModel(xs, i_obs, hierarchy=hierarchy, r_free_flags=r_free, n_bins=5)

    # 1. Binned nu estimation
    nu_binned = model.estimate_nu(mode="binned", bounds=(2.05, 50.0))
    assert isinstance(nu_binned, dict)
    assert len(nu_binned) == 5
    for b_idx, nu_val in nu_binned.items():
        assert 2.05 <= nu_val <= 50.0
    assert model.nu_per_refl is not None
    assert len(model.nu_per_refl) == i_obs.size()
    assert model.nu is not None and 2.05 <= model.nu <= 50.0

    # 2. TV regularized binned nu
    nu_tv = model.estimate_nu(mode="binned", tv_lambda=0.05, bounds=(2.05, 50.0))
    assert len(nu_tv) == 5
    for b_idx, nu_val in nu_tv.items():
        assert 2.05 <= nu_val <= 50.0

    # 3. Global nu estimation
    nu_global = model.estimate_nu(mode="global", bounds=(2.05, 50.0))
    assert isinstance(nu_global, float)
    assert 2.05 <= nu_global <= 50.0
    assert model.nu == nu_global
    for b_idx, nu_val in model.nu_binned.items():
        assert nu_val == nu_global


def test_refine_sigma_a_and_nu():
    xs, i_obs, hierarchy, r_free = _make_synthetic_model()
    model = IntensityModel(xs, i_obs, hierarchy=hierarchy, r_free_flags=r_free, n_bins=5)

    sa_res, nu_res = model.refine_sigma_a_and_nu(max_cycles=2, nu_mode="binned")
    assert len(sa_res) == 5
    assert len(nu_res) == 5
    for b_idx, sa_val in sa_res.items():
        assert 0.01 <= sa_val <= 1.0
    for b_idx, nu_val in nu_res.items():
        assert 2.05 <= nu_val <= 50.0

    # Target & gradients evaluate correctly with refined nu
    target, grads = model.compute_target_and_gradients()
    assert np.isfinite(target)
    assert grads is not None
    g_cart = grads.d_target_d_site_cart()
    assert len(g_cart) == xs.scatterers().size()

    # Summary reports Student-t noise model
    s = model.summary()
    assert not np.isnan(s["mean_nu"])
    assert 2.05 <= s["mean_nu"] <= 50.0
    model.print_summary()


def test_target_and_gradients():
    xs, i_obs, hierarchy, r_free = _make_synthetic_model()
    model = IntensityModel(xs, i_obs, hierarchy=hierarchy, r_free_flags=r_free, n_bins=5)
    target, grads = model.compute_target_and_gradients()

    assert np.isfinite(target)
    assert grads is not None

    g_cart = grads.d_target_d_site_cart()
    assert len(g_cart) == xs.scatterers().size()

    g_frac = grads.d_target_d_site_frac()
    assert len(g_frac) == xs.scatterers().size()

    g_u_iso = grads.d_target_d_u_iso()
    assert len(g_u_iso) == xs.scatterers().size()

    packed = grads.packed()
    assert len(packed) > 0


def test_coordinate_refinement():
    xs, i_obs, hierarchy, r_free = _make_synthetic_model()
    model = IntensityModel(xs, i_obs, hierarchy=hierarchy, r_free_flags=r_free, n_bins=5)
    target0, _ = model.compute_target_and_gradients()

    # Perform a few small steps
    history = model.refine_coordinates(max_iterations=3, step_scale=0.0005)
    assert len(history) == 4
    # Coordinates should have updated
    sites = list(model.xray_structure.sites_frac())
    assert len(sites) == xs.scatterers().size()


def test_b_iso_refinement():
    xs, i_obs, hierarchy, r_free = _make_synthetic_model()
    model = IntensityModel(xs, i_obs, hierarchy=hierarchy, r_free_flags=r_free, n_bins=5)
    model.convert_to_isotropic()
    target0, _ = model.compute_target_and_gradients()

    b_orig = np.array([float(sc.u_iso * 8.0 * np.pi**2) for sc in model.xray_structure.scatterers()])
    sites_orig = list(model.xray_structure.sites_frac())

    # Perform a few B refinement steps
    history = model.refine_b_iso(max_iterations=3, step_scale=100.0)
    assert len(history) <= 4
    assert history[-1] <= history[0]

    # Verify sites were NOT modified
    sites_after = list(model.xray_structure.sites_frac())
    for s0, s1 in zip(sites_orig, sites_after):
        assert np.allclose(s0, s1, atol=1e-8)

    # Verify B values were modified
    b_after = np.array([float(sc.u_iso * 8.0 * np.pi**2) for sc in model.xray_structure.scatterers()])
    assert not np.allclose(b_orig, b_after)


def test_map_synthesis_and_writing(tmp_path: Path):
    xs, i_obs, hierarchy, r_free = _make_synthetic_model()
    model = IntensityModel(xs, i_obs, hierarchy=hierarchy, r_free_flags=r_free, n_bins=5)

    prefix = str(tmp_path / "test_out")
    files = model.write_maps(prefix=prefix, resolution_factor=0.33)

    # Check files exist
    assert os.path.exists(files["mtz"])
    assert os.path.exists(files["gradient"])
    assert os.path.exists(files["2fofc"])
    assert os.path.exists(files["fofc"])

    # Verify MTZ can be read and contains our columns
    reader = any_reflection_file(files["mtz"])
    arrays = reader.as_miller_arrays()
    labels = []
    for a in arrays:
        if a.info() and a.info().labels:
            labels.extend(a.info().labels)
    assert any("2FOFCWT" in l for l in labels)
    assert any("FOFCWT" in l for l in labels)
    assert any("FGRAD" in l for l in labels)

    # Verify CCP4 map can be read
    m_reader = iotbx.ccp4_map.map_reader(file_name=files["gradient"])
    assert m_reader.unit_cell() is not None
    assert m_reader.data.size() > 0


def test_pipeline_end_to_end_from_files(tmp_path: Path):
    xs, i_obs, hierarchy, r_free = _make_synthetic_model()

    pdb_file = tmp_path / "model.pdb"
    pdb_file.write_text(xs.as_pdb_file())

    mtz_file = tmp_path / "data.mtz"
    mtz_ds = i_obs.as_mtz_dataset(column_root_label="IOBS")
    mtz_ds.add_miller_array(r_free, column_root_label="FreeR_flag")
    mtz_ds.mtz_object().write(str(mtz_file))

    out_prefix = str(tmp_path / "run_out")
    res = run_intensity_pipeline(
        pdb_path=pdb_file,
        mtz_path=mtz_file,
        prefix=out_prefix,
        use_bulk_solvent=True,
        n_bins=4,
        max_iterations=1,
    )

    assert "model" in res
    assert "files" in res
    assert "summary" in res

    summary = res["summary"]
    assert summary["n_refl"] == i_obs.size()
    assert np.isfinite(summary["r_work"])
    assert np.isfinite(summary["r_int"])


def test_cli_entrypoint(tmp_path: Path):
    xs, i_obs, hierarchy, r_free = _make_synthetic_model()

    pdb_file = tmp_path / "cli_model.pdb"
    pdb_file.write_text(xs.as_pdb_file())

    mtz_file = tmp_path / "cli_data.mtz"
    mtz_ds = i_obs.as_mtz_dataset(column_root_label="I")
    mtz_ds.mtz_object().write(str(mtz_file))

    out_prefix = str(tmp_path / "cli_out")
    exit_code = main([
        str(pdb_file),
        str(mtz_file),
        "--prefix", out_prefix,
        "--n-bins", "4",
        "--estimate-nu",
        "--nu-mode", "binned",
        "--refine", "1",
    ])
    assert exit_code == 0
    assert os.path.exists(f"{out_prefix}_maps.mtz")
    assert os.path.exists(f"{out_prefix}_gradient.ccp4")
    assert os.path.exists(f"{out_prefix}_refined.pdb")


def test_omit_mode_delete(tmp_path: Path):
    from phridge.client.intensity_tool import parse_omit_selection, apply_omit
    assert parse_omit_selection("45-52") == "resseq 45:52"
    assert parse_omit_selection("45:52") == "resseq 45:52"
    assert parse_omit_selection("5") == "resseq 5"
    assert parse_omit_selection("chain A and resseq 1:3") == "chain A and resseq 1:3"

    xs, i_obs, hierarchy, r_free = _make_synthetic_model(n_atoms=15)
    pdb_file = tmp_path / "model.pdb"
    pdb_file.write_text(xs.as_pdb_file())
    mtz_file = tmp_path / "data.mtz"
    i_obs.as_mtz_dataset(column_root_label="IOBS").mtz_object().write(str(mtz_file))

    # Omit residue 1 with delete
    model = from_files(pdb_file, mtz_file, omit="1", omit_mode="delete")
    assert len(model.xray_structure.scatterers()) < 15
    assert len(list(model.hierarchy.atoms())) < 15


def test_omit_mode_zero_occ(tmp_path: Path):
    xs, i_obs, hierarchy, r_free = _make_synthetic_model(n_atoms=15)
    pdb_file = tmp_path / "model.pdb"
    pdb_file.write_text(xs.as_pdb_file())
    mtz_file = tmp_path / "data.mtz"
    i_obs.as_mtz_dataset(column_root_label="IOBS").mtz_object().write(str(mtz_file))

    # Omit residue 1 with zero_occ
    model = from_files(pdb_file, mtz_file, omit="1", omit_mode="zero_occ")
    assert len(model.xray_structure.scatterers()) == 15
    zero_occ = [sc for sc in model.xray_structure.scatterers() if sc.occupancy == 0.0]
    assert len(zero_occ) > 0


def test_viewer_server(tmp_path: Path):
    import urllib.request
    from phridge.client.viewer import launch_viewer, open_static_viewer

    pdb_file = tmp_path / "v_model.pdb"
    pdb_file.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N\n")
    map_file = tmp_path / "v_grad.ccp4"
    map_file.write_bytes(b"\x00" * 1024)

    # Test static zero-server viewer generation
    html_path = open_static_viewer(
        pdb_path=pdb_file,
        gradient_map=map_file,
        output_html=tmp_path / "test_viewer.html",
        open_browser=False,
    )
    assert html_path.is_file()
    html_content = html_path.read_text(encoding="utf-8")
    assert "Zero-Server 3D Map & Model Viewer" in html_content
    assert "v_model.pdb" in html_content
    assert "v_grad.ccp4" in html_content
    assert "ATOM      1  N   ALA A   1" in html_content  # PDB is embedded

    server, url = launch_viewer(
        pdb_path=pdb_file,
        gradient_map=map_file,
        open_browser=False,
        block=False,
    )
    try:
        with urllib.request.urlopen(url) as resp:
            html = resp.read().decode("utf-8")
            assert "phridge Molecular & Map Viewer" in html
            assert "v_model.pdb" in html

        with urllib.request.urlopen(url + "v_model.pdb") as resp:
            assert len(resp.read()) > 0
    finally:
        server.shutdown()
        server.server_close()


def test_compute_map_coefficients_from_gradient():
    xs, i_obs, hierarchy, r_free = _make_synthetic_model()
    model = IntensityModel(xs, i_obs, hierarchy=hierarchy, r_free_flags=r_free, n_bins=4)
    model.refine_scale_and_solvent()
    model.estimate_sigma_a()

    grad_c, f2_grad, f1_grad = model.compute_map_coefficients(weighted=True, from_gradient=True)
    grad_leg, f2_leg, f1_leg = model.compute_map_coefficients(weighted=True, from_gradient=False)

    # Gradient-derived fofc must equal grad_c
    assert np.allclose(np.asarray(f1_grad.data()), np.asarray(grad_c.data()))

    # Gradient-derived 2fofc must equal sa * fmod + 2 * grad_c
    sa_np = np.asarray(model.sigma_a_per_refl, dtype=np.float64)
    fmod_np = np.asarray(model.f_model.data(), dtype=np.complex128)
    expected_2f = sa_np * fmod_np + 2.0 * np.asarray(grad_c.data())
    assert np.allclose(np.asarray(f2_grad.data()), expected_2f)

    # Real-space correlation between gradient-derived and legacy 2Fo-Fc
    rho_grad = f2_grad.fft_map(resolution_factor=0.33).real_map_unpadded().as_numpy_array()
    rho_leg = f2_leg.fft_map(resolution_factor=0.33).real_map_unpadded().as_numpy_array()
    cc = np.corrcoef(rho_grad.ravel(), rho_leg.ravel())[0, 1]
    assert cc > 0.95


def test_amplitude_target_refinement(tmp_path: Path):
    xs, i_obs, hierarchy, r_free = _make_synthetic_model(n_atoms=20)
    model = IntensityModel(
        xs, i_obs, hierarchy=hierarchy, r_free_flags=r_free, n_bins=4,
        target_type="amplitude", use_bulk_solvent=False,
    )
    assert model.target_type == "amplitude"

    # Amplitude Miller array is lazily computed via French-Wilson
    f_obs = model.get_f_obs()
    assert f_obs.is_xray_amplitude_array()
    assert f_obs.size() == i_obs.size()

    # Scale refinement under amplitude target
    scales = model.refine_scale_and_solvent(max_iter=10)
    assert scales["k_total"] > 0

    # Sigma_A estimation under amplitude target
    sa = model.estimate_sigma_a()
    assert len(sa) == 4
    assert all(0.01 <= v <= 0.999 for v in sa.values())

    # Target & gradients
    val, grads = model.compute_target_and_gradients()
    assert np.isfinite(val)
    assert grads is not None

    # Coordinate step under amplitude target
    hist = model.refine_coordinates(max_iterations=1, step_scale=0.001)
    assert len(hist) == 2
    assert all(np.isfinite(x) for x in hist)

    # Map synthesis under amplitude target
    prefix = str(tmp_path / "amp_out")
    files = model.write_maps(prefix=prefix, resolution_factor=0.33)
    assert os.path.exists(files["mtz"])
    assert os.path.exists(files["2fofc"])


def test_cli_target_amplitude(tmp_path: Path):
    xs, i_obs, hierarchy, r_free = _make_synthetic_model()

    pdb_file = tmp_path / "cli_amp_model.pdb"
    pdb_file.write_text(xs.as_pdb_file())

    mtz_file = tmp_path / "cli_amp_data.mtz"
    mtz_ds = i_obs.as_mtz_dataset(column_root_label="IOBS")
    mtz_ds.add_miller_array(r_free, column_root_label="FreeR_flag")
    mtz_ds.mtz_object().write(str(mtz_file))

    out_prefix = str(tmp_path / "cli_amp_out")
    exit_code = main([
        str(pdb_file),
        str(mtz_file),
        "--prefix", out_prefix,
        "--n-bins", "4",
        "--target", "amplitude",
        "--refine", "1",
    ])
    assert exit_code == 0
    assert os.path.exists(f"{out_prefix}_maps.mtz")
    assert os.path.exists(f"{out_prefix}_refined.pdb")


def test_compute_all_map_coefficients_and_maps_export(tmp_path: Path):
    xs, i_obs, hierarchy, r_free = _make_synthetic_model(n_atoms=15, d_min=2.5)
    model = IntensityModel(xs, i_obs, hierarchy=hierarchy, r_free_flags=r_free, n_bins=4)
    model.refine_scale_and_solvent()
    model.estimate_sigma_a()

    all_maps = model.compute_all_map_coefficients(newton_damping=0.1)
    for key in (
        "difference",
        "model",
        "gradient",
        "newton",
        "gradient_weighted",
        "gradient_raw",
        "2fofc",
        "fofc",
        "fom",
        "f_post",
        "robust_weight",
        "curvature",
    ):
        assert key in all_maps
        assert all_maps[key].size() == i_obs.size()

    # Check export writes all map targets
    prefix = str(tmp_path / "all_maps_out")
    files = model.write_maps(prefix=prefix, resolution_factor=0.33, compute_all=True)

    assert "mtz" in files
    assert "newton" in files
    assert "diff_post" in files
    assert "model_post" in files
    assert os.path.exists(files["newton"])
    assert os.path.exists(files["diff_post"])
    assert os.path.exists(files["model_post"])

    # Verify MTZ has new columns
    reader = any_reflection_file(files["mtz"])
    col_labels = []
    for a in reader.as_miller_arrays():
        if a.info() and a.info().labels:
            col_labels.extend(a.info().labels)
    assert any("FNEWTON" in l for l in col_labels)
    assert any("FDIFF_POST" in l for l in col_labels)
    assert any("FMODEL_POST" in l for l in col_labels)
    assert any("FOM" in l for l in col_labels)
    assert any("F_POST" in l for l in col_labels)


def test_geometry_restraints_build_and_gradients():
    xs, i_obs, hierarchy, r_free = _make_synthetic_model(n_atoms=15, d_min=2.5)
    model = IntensityModel(xs, i_obs, hierarchy=hierarchy, r_free_flags=r_free, n_bins=4)
    model.convert_to_isotropic()

    e_geom, g_geom = model.compute_geometry_energy_and_gradients()
    assert isinstance(e_geom, float)
    assert g_geom.shape == (15, 3)


def test_hessian_preconditioners():
    xs, i_obs, hierarchy, r_free = _make_synthetic_model(n_atoms=15, d_min=2.5)
    model = IntensityModel(xs, i_obs, hierarchy=hierarchy, r_free_flags=r_free, n_bins=4)
    model.convert_to_isotropic()
    model.refine_scale_and_solvent()
    model.estimate_sigma_a()

    import torch
    from phridge.contrib.intensity_ll.target import IntensityLogLikelihood
    from phridge.sfcalc.targets.base import Observations

    fmod_t = torch.as_tensor(np.asarray(model.f_model.data(), dtype=np.complex128))
    obs = Observations.from_numpy(
        data=np.asarray(model.i_obs.data(), dtype=np.float64),
        sigmas=np.asarray(model.i_obs.sigmas(), dtype=np.float64),
        epsilon=np.asarray(model.i_obs.epsilons().data().as_double(), dtype=np.float64),
        centric=np.asarray(model.i_obs.centric_flags().data(), dtype=bool),
        alpha=np.asarray(model.sigma_a_per_refl, dtype=np.float64),
        beta=np.asarray(model.mean_i_per_refl, dtype=np.float64),
        r_free=np.asarray(model.r_free_flags.data(), dtype=bool),
    )
    tgt = IntensityLogLikelihood()
    ev = tgt.evaluate(fmod_t, obs, compute_curvature=True)

    p_sites, p_b = model.compute_hessian_preconditioners(ev, damping_factor=0.05)
    assert p_sites.shape == (15, 3)
    assert p_b.shape == (15,)
    assert np.all(p_sites > 0.0)
    assert np.all(p_b > 0.0)


def test_adam_refinement_with_hessian_and_geometry():
    xs, i_obs, hierarchy, r_free = _make_synthetic_model(n_atoms=15, d_min=2.5)
    model = IntensityModel(xs, i_obs, hierarchy=hierarchy, r_free_flags=r_free, n_bins=4, estimate_nu=True)
    model.convert_to_isotropic()
    model.refine_scale_and_solvent()
    model.estimate_sigma_a()

    sites_orig = list(model.xray_structure.sites_frac())
    b_orig = np.array([float(sc.u_iso * 8.0 * np.pi**2) for sc in model.xray_structure.scatterers()])
    k_orig = model.k_total

    history = model.refine_adam(
        max_iterations=3,
        lr_sites=0.005,
        lr_b=0.2,
        lr_scale=0.001,
        refine_scales=True,
        refine_b=True,
        refine_sites=True,
        refine_sigma_a=True,
        refine_nu=True,
        sigma_a_interval=1,
        verbose=False,
    )

    assert len(history["nll"]) == 3
    assert history["nll"][-1] <= history["nll"][0]

    # Verify sites were modified
    sites_after = list(model.xray_structure.sites_frac())
    any_diff = False
    for s0, s1 in zip(sites_orig, sites_after):
        if not np.allclose(s0, s1, atol=1e-6):
            any_diff = True
    assert any_diff

    # Verify B values were modified
    b_after = np.array([float(sc.u_iso * 8.0 * np.pi**2) for sc in model.xray_structure.scatterers()])
    assert not np.allclose(b_orig, b_after)


def test_cli_adam_refinement(tmp_path: Path):
    xs, i_obs, hierarchy, r_free = _make_synthetic_model(n_atoms=15, d_min=2.5)

    pdb_file = tmp_path / "adam_model.pdb"
    pdb_file.write_text(xs.as_pdb_file())

    mtz_file = tmp_path / "adam_data.mtz"
    mtz_ds = i_obs.as_mtz_dataset(column_root_label="IOBS")
    mtz_ds.add_miller_array(r_free, column_root_label="FreeR_flag")
    mtz_ds.mtz_object().write(str(mtz_file))

    out_prefix = str(tmp_path / "adam_out")
    exit_code = main([
        str(pdb_file),
        str(mtz_file),
        "--prefix", out_prefix,
        "--n-bins", "4",
        "--refine-mode", "adam",
        "--refine", "2",
        "--estimate-nu",
    ])
    assert exit_code == 0
    assert os.path.exists(f"{out_prefix}_maps.mtz")
    assert os.path.exists(f"{out_prefix}_refined.pdb")


def test_four_way_refinement_modes():
    """Verify that all four refinement modes execute properly on a synthetic model:
      1. Classic F-based likelihood (ml_f, unpreconditioned)
      2. Preconditioned F-based likelihood (ml_f, Gauss-Newton Hessian)
      3. Intensity-based likelihood (ml_i, unpreconditioned)
      4. Preconditioned intensity likelihood (ml_i, Gauss-Newton Hessian)
    """
    xs, i_obs, hierarchy, r_free = _make_synthetic_model(n_atoms=15, d_min=2.5)

    # 1. Classic F-based likelihood (unpreconditioned)
    m_f = IntensityModel(xs, i_obs, hierarchy=hierarchy, r_free_flags=r_free, n_bins=4, target_type="amplitude")
    m_f.convert_to_isotropic()
    m_f.refine_scale_and_solvent()
    m_f.estimate_sigma_a()
    hist_f = m_f.refine_adam(max_iterations=2, use_preconditioner=False, verbose=False)
    assert len(hist_f["nll"]) == 2
    assert np.isfinite(hist_f["nll"][-1])

    # 2. Preconditioned F-based likelihood
    m_f_prec = IntensityModel(xs, i_obs, hierarchy=hierarchy, r_free_flags=r_free, n_bins=4, target_type="amplitude")
    m_f_prec.convert_to_isotropic()
    m_f_prec.refine_scale_and_solvent()
    m_f_prec.estimate_sigma_a()
    hist_f_prec = m_f_prec.refine_adam(max_iterations=2, use_preconditioner=True, verbose=False)
    assert len(hist_f_prec["nll"]) == 2
    assert np.isfinite(hist_f_prec["nll"][-1])

    # 3. Intensity-based likelihood (unpreconditioned)
    m_i_unprec = IntensityModel(xs, i_obs, hierarchy=hierarchy, r_free_flags=r_free, n_bins=4, target_type="intensity", estimate_nu=True)
    m_i_unprec.convert_to_isotropic()
    m_i_unprec.refine_scale_and_solvent()
    m_i_unprec.estimate_sigma_a()
    hist_i_unprec = m_i_unprec.refine_adam(max_iterations=2, use_preconditioner=False, verbose=False)
    assert len(hist_i_unprec["nll"]) == 2
    assert np.isfinite(hist_i_unprec["nll"][-1])

    # 4. Preconditioned intensity likelihood
    m_i_prec = IntensityModel(xs, i_obs, hierarchy=hierarchy, r_free_flags=r_free, n_bins=4, target_type="intensity", estimate_nu=True)
    m_i_prec.convert_to_isotropic()
    m_i_prec.refine_scale_and_solvent()
    m_i_prec.estimate_sigma_a()
    hist_i_prec = m_i_prec.refine_adam(max_iterations=2, use_preconditioner=True, verbose=False)
    assert len(hist_i_prec["nll"]) == 2
    assert np.isfinite(hist_i_prec["nll"][-1])


def test_hessian_xray_weighting_and_intact_geometry():
    """Verify that leaving geometry scale intact and reweighting X-ray via Hessian ratios works."""
    import torch
    from phridge.sfcalc.targets.maximum_likelihood import MaximumLikelihoodAmplitude
    from phridge.sfcalc.targets.base import Observations

    xs, i_obs, hierarchy, r_free = _make_synthetic_model(n_atoms=15, d_min=2.5)
    model = IntensityModel(xs, i_obs, hierarchy=hierarchy, r_free_flags=r_free, n_bins=4, estimate_nu=False)
    model.convert_to_isotropic()
    model.refine_scale_and_solvent()
    model.estimate_sigma_a()

    # 1. Test estimate_geometry_curvature fallback on model without restraints
    h_geom = model.estimate_geometry_curvature(n_shakes=2)
    assert h_geom > 0

    # 2. Test compute_hessian_preconditioners with w_xray and h_geom
    tgt = MaximumLikelihoodAmplitude(scale_factor=1.0)
    fo = model.get_f_obs()
    obs = Observations.from_numpy(
        device=model.device,
        dtype=model.float_dtype,
        data=np.asarray(fo.data(), dtype=np.float64),
        sigmas=np.asarray(fo.sigmas(), dtype=np.float64),
        epsilon=np.asarray(model.i_obs.epsilons().data().as_double(), dtype=np.float64),
        centric=np.asarray(model.i_obs.centric_flags().data(), dtype=bool),
        alpha=np.asarray(model.sigma_a_per_refl, dtype=np.float64),
        beta=np.asarray(model.mean_i_per_refl, dtype=np.float64) * (1.0 - np.asarray(model.sigma_a_per_refl, dtype=np.float64)**2),
        r_free=np.asarray(model.r_free_flags.data(), dtype=bool),
    )
    fmod_t = torch.as_tensor(np.asarray(model.f_model.data(), dtype=np.complex128), dtype=model.complex_dtype, device=model.torch_device)
    ev = tgt.evaluate(fmod_t, obs, compute_curvature=True)

    p_sites, p_B = model.compute_hessian_preconditioners(ev, damping_factor=0.05, w_xray=1e6, h_geom=5000.0)
    assert p_sites.shape == (15, 3)
    assert np.all(p_sites > 0)
    assert np.all(np.isfinite(p_sites))
    assert p_B.shape == (15,)
    assert np.all(p_B > 0)

    # 3. Test refine_adam with xray_weight_mode='hessian'
    history = model.refine_adam(
        max_iterations=3,
        xray_weight_mode="hessian",
        xray_scale=1.0,
        use_preconditioner=True,
        refine_scales=True,
        refine_b=True,
        refine_sites=True,
        refine_sigma_a=False,
        refine_nu=False,
        verbose=False,
    )
    assert len(history["nll"]) == 3
    assert len(history["w_xray"]) == 3
    assert np.isfinite(history["nll"][-1])


def test_lbfgs_refinement():
    """Verify that L-BFGS refinement executes properly on synthetic data."""
    xs, i_obs, hierarchy, r_free = _make_synthetic_model(n_atoms=15, d_min=2.5)
    model = IntensityModel(xs, i_obs, hierarchy=hierarchy, r_free_flags=r_free, n_bins=4)
    model.convert_to_isotropic()
    model.refine_scale_and_solvent()
    model.estimate_sigma_a()

    hist = model.refine_lbfgs(
        macrocycles=2,
        max_iterations_per_cycle=5,
        regularize_geometry=False,
        xray_weight_mode="hessian",
        xray_scale=1.0,
        refine_scales=True,
        refine_b=True,
        refine_sites=True,
        refine_sigma_a=False,
        refine_nu=False,
        polish_geometry=False,
        verbose=False,
    )
    assert len(hist["nll"]) >= 2
    assert np.isfinite(hist["nll"][-1])
    assert len(hist["r_work"]) >= 2
    assert len(hist["r_free"]) >= 2


def test_cli_lbfgs_refinement(tmp_path: Path):
    """Verify CLI execution with --refine-mode lbfgs."""
    xs, i_obs, hierarchy, r_free = _make_synthetic_model(n_atoms=15, d_min=2.5)

    pdb_file = tmp_path / "lbfgs_model.pdb"
    pdb_file.write_text(xs.as_pdb_file())

    mtz_file = tmp_path / "lbfgs_data.mtz"
    mtz_ds = i_obs.as_mtz_dataset(column_root_label="IOBS")
    mtz_ds.add_miller_array(r_free, column_root_label="FreeR_flag")
    mtz_ds.mtz_object().write(str(mtz_file))

    out_prefix = str(tmp_path / "lbfgs_out")
    exit_code = main([
        str(pdb_file),
        str(mtz_file),
        "--prefix", out_prefix,
        "--n-bins", "4",
        "--refine-mode", "lbfgs",
        "--macrocycles", "1",
        "--lbfgs-max-iter", "3",
        "--no-regularize-geometry",
        "--no-polish-geometry",
    ])
    assert exit_code == 0
    assert os.path.exists(f"{out_prefix}_maps.mtz")
    assert os.path.exists(f"{out_prefix}_refined.pdb")


def test_co_refine_sigma_a_and_global_nu():
    """Verify co-refinement of sigma_A and a single global nu across the full dataset."""
    xs, i_obs, hierarchy, r_free = _make_synthetic_model(n_atoms=15, d_min=2.5)
    model = IntensityModel(
        xs,
        i_obs,
        hierarchy=hierarchy,
        r_free_flags=r_free,
        n_bins=4,
        nu=7.0,
        estimate_nu=True,
        nu_mode="global",
    )
    assert model.nu == 7.0
    assert model.nu_mode == "global"
    assert model.nu_per_refl is not None
    assert len(model.nu_per_refl) == i_obs.size()

    # Refine scales & solvent first
    model.refine_scale_and_solvent()

    # Co-refine sigma_A and global nu
    sa_dict, nu_val = model.refine_sigma_a_and_nu(max_cycles=2, nu_mode="global")
    assert isinstance(nu_val, float)
    assert 3.0 <= nu_val <= 30.0
    assert len(sa_dict) == 4
    for sa in sa_dict.values():
        assert 0.01 <= sa <= 0.999
    assert model.nu == nu_val
    assert np.allclose(list(model.nu_per_refl), nu_val)


def test_shake_b_iso_reset_fractional():
    """Verify B-factor reset with fractional shake (e.g. +/- 25%)."""
    xs, i_obs, hierarchy, r_free = _make_synthetic_model(n_atoms=20, d_min=2.5)
    model = IntensityModel(xs, i_obs, hierarchy=hierarchy, r_free_flags=r_free, n_bins=4)
    model.convert_to_isotropic()

    b_orig = np.array([float(sc.u_iso * 8.0 * np.pi**2) for sc in model.xray_structure.scatterers()])
    b_mean = float(np.mean(b_orig))

    # 1. Reset to mean B with +/- 25% uniform fractional shake
    rng = np.random.default_rng(123)
    b_shaken = model.shake_b_iso(fraction=0.25, reset=True, rng=rng)

    assert len(b_shaken) == len(b_orig)
    # Every shaken B should be within [0.75 * b_mean, 1.25 * b_mean]
    assert np.all(b_shaken >= 0.75 * b_mean - 1e-6)
    assert np.all(b_shaken <= 1.25 * b_mean + 1e-6)
    assert np.isclose(np.mean(b_shaken), b_mean, atol=2.0)

    # 2. Shake individual B-factors without resetting to mean
    rng2 = np.random.default_rng(456)
    b_shaken_ind = model.shake_b_iso(fraction=0.25, reset=False, rng=rng2)
    # Check that individual ratios are within [0.75, 1.25]
    ratios = b_shaken_ind / b_shaken
    assert np.all(ratios >= 0.75 - 1e-6)
    assert np.all(ratios <= 1.25 + 1e-6)


def test_combined_shake_and_recover_sites_and_b():
    """Verify combined coordinate shake and B-factor reset with fractional shake."""
    xs, i_obs, hierarchy, r_free = _make_synthetic_model(n_atoms=20, d_min=2.5)
    model = IntensityModel(xs, i_obs, hierarchy=hierarchy, r_free_flags=r_free, n_bins=4)
    model.convert_to_isotropic()

    sites_orig = np.asarray(model.xray_structure.sites_cart(), dtype=np.float64).copy()
    b_orig = np.array([float(sc.u_iso * 8.0 * np.pi**2) for sc in model.xray_structure.scatterers()])

    # Combined shake: 0.10 Å coordinate RMSD + B-factor reset with +/- 25% shake
    res = model.shake(rmsd=0.10, b_fraction=0.25, reset_b=True, rng=np.random.default_rng(42))

    sites_shaken = np.asarray(model.xray_structure.sites_cart(), dtype=np.float64)
    b_shaken = np.array([float(sc.u_iso * 8.0 * np.pi**2) for sc in model.xray_structure.scatterers()])

    actual_rmsd = float(np.sqrt(np.mean(np.sum((sites_shaken - sites_orig)**2, axis=1))))
    assert np.isclose(actual_rmsd, 0.10, atol=0.01)

    b_mean = float(np.mean(b_orig))
    assert np.all(b_shaken >= 0.75 * b_mean - 1e-6)
    assert np.all(b_shaken <= 1.25 * b_mean + 1e-6)


def test_preconditioned_b_factor_refinement_recovery():
    """Verify that preconditioned Newton steps for B-factors actively reduce NLL and recover B-factors."""
    xs, i_obs, hierarchy, r_free = _make_synthetic_model(n_atoms=15, d_min=2.5)
    model = IntensityModel(xs, i_obs, hierarchy=hierarchy, r_free_flags=r_free, n_bins=4, estimate_nu=False)
    model.convert_to_isotropic()
    b_true = np.array([float(sc.u_iso * 8.0 * np.pi**2) for sc in model.xray_structure.scatterers()])

    # Shake B-factors with +/- 25% noise
    model.shake_b_iso(fraction=0.25, reset=True, rng=np.random.default_rng(42))
    b_start = np.array([float(sc.u_iso * 8.0 * np.pi**2) for sc in model.xray_structure.scatterers()])
    rmsd_start = float(np.sqrt(np.mean((b_start - b_true)**2)))

    # Refine with preconditioned B steps
    model.refine_scale_and_solvent()
    model.estimate_sigma_a()
    hist = model.refine_b_iso(max_iterations=3, use_preconditioner=True, verbose=False)

    b_final = np.array([float(sc.u_iso * 8.0 * np.pi**2) for sc in model.xray_structure.scatterers()])
    rmsd_final = float(np.sqrt(np.mean((b_final - b_true)**2)))

    # Verify that NLL decreased and B-factor RMSD improved
    assert hist[-1] < hist[0]
    assert rmsd_final < rmsd_start
