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

