"""Tests for IntensityFModel and the CCTBX-like intensity engine interface."""

from __future__ import annotations

import numpy as np
import pytest

from cctbx import crystal, miller, sgtbx, xray
from cctbx.array_family import flex
from cctbx.development import random_structure

from phridge.client.intensity import (
    IntensityElectronDensityMap,
    IntensityFModel,
    IntensityGradients,
    IntensityLikelihoodEngine,
)


def _make_test_data(d_min: float = 2.5, seed: int = 42):
    np.random.seed(seed)
    flex.set_random_seed(seed)
    xs = random_structure.xray_structure(
        space_group_info=sgtbx.space_group_info("P212121"),
        elements=["C", "N", "O"] * 4,
        volume_per_atom=40,
        random_u_iso=True,
    )
    for sc in xs.scatterers():
        sc.flags.set_grad_site(True)
        sc.flags.set_grad_u_iso(True)

    fc = xs.structure_factors(d_min=d_min).f_calc()
    i_true = fc.intensities()
    noise = flex.double(np.random.normal(0, 0.2, size=i_true.size()))
    raw_data = i_true.data() + noise
    raw_data.set_selected(raw_data < 0.01, 0.01)
    i_obs = i_true.customized_copy(
        data=raw_data,
        sigmas=flex.double(i_true.size(), 0.5),
    ).set_observation_type_xray_intensity()
    r_free = i_obs.generate_r_free_flags(fraction=0.10)
    return xs, i_obs, r_free, fc


def test_intensity_fmodel_init_and_properties():
    xs, i_obs, r_free, _ = _make_test_data()

    # Standard initialization
    fmodel = IntensityFModel(i_obs, xs, r_free, memory=True)
    assert fmodel.i_obs().size() == i_obs.size()
    assert fmodel.r_free_flags().size() == r_free.size()
    assert fmodel.scale_k1() == 1.0
    assert fmodel.k_total() == 1.0
    assert fmodel.unit_cell().parameters() == xs.unit_cell().parameters()
    assert fmodel.space_group() == xs.space_group()

    # Test alias IntensityLikelihoodEngine
    engine = IntensityLikelihoodEngine(i_obs, xs, r_free, memory=True)
    assert isinstance(engine, IntensityFModel)

    # Test reversed argument order (xs, i_obs)
    fmodel_rev = IntensityFModel(xs, i_obs, r_free, memory=True)
    assert fmodel_rev.i_obs().size() == i_obs.size()

    # Test amplitude-only construction is rejected (no F² reconstruction)
    from phridge.client.intensity.engine import IntensityDataError

    f_obs_amp = i_obs.as_amplitude_array()
    with pytest.raises(IntensityDataError):
        IntensityFModel(f_obs=f_obs_amp, xray_structure=xs, r_free_flags=r_free, memory=True)


def test_f_calc_and_f_model():
    xs, i_obs, r_free, fc_expected = _make_test_data()
    fmodel = IntensityFModel(i_obs, xs, r_free, scale_factor=1.25, memory=True)

    fc = fmodel.f_calc()
    assert fc.is_complex_array()
    assert fc.size() == i_obs.size()

    fmod = fmodel.f_model()
    assert fmod.size() == fc.size()
    # f_model includes k_isotropic · k_anisotropic (seeded from scale_factor at init)
    np.testing.assert_allclose(
        np.abs(np.asarray(fmod.data())),
        1.25 * np.abs(np.asarray(fc.data())),
        rtol=1e-5,
    )


def test_target_and_gradients_unpreconditioned_and_preconditioned():
    xs, i_obs, r_free, _ = _make_test_data()
    fmodel = IntensityFModel(i_obs, xs, r_free, memory=True)

    # 1. Unpreconditioned
    val_unprec, grads_unprec = fmodel.target_and_gradients(precondition=False)
    assert isinstance(val_unprec, float)
    assert val_unprec > 0.0
    assert isinstance(grads_unprec, flex.double)
    assert isinstance(grads_unprec, IntensityGradients)
    assert not grads_unprec.preconditioned
    assert grads_unprec.curvatures is None

    # Component accessors
    site_cart = grads_unprec.d_target_d_site_cart()
    assert site_cart.size() == len(xs.scatterers())
    assert isinstance(site_cart, flex.vec3_double)

    site_frac = grads_unprec.d_target_d_site_frac()
    assert site_frac.size() == len(xs.scatterers())
    assert isinstance(site_frac, flex.vec3_double)

    u_iso = grads_unprec.d_target_d_u_iso()
    assert u_iso.size() == len(xs.scatterers())
    assert isinstance(u_iso, flex.double)

    # 2. Preconditioned
    val_prec, grads_prec = fmodel.target_and_gradients(precondition=True, damping=0.05)
    assert pytest.approx(val_prec, rel=1e-5) == val_unprec
    assert grads_prec.preconditioned
    assert grads_prec.curvatures is not None

    # Preconditioned gradients should differ in scale due to curvature weighting
    assert not np.allclose(
        np.asarray(grads_prec.d_target_d_site_cart().as_double()),
        np.asarray(grads_unprec.d_target_d_site_cart().as_double()),
    )
    assert not np.allclose(
        np.asarray(list(grads_prec.d_target_d_u_iso())),
        np.asarray(list(grads_unprec.d_target_d_u_iso())),
    )
    assert not np.allclose(
        np.asarray(list(grads_prec.d_target_d_occupancy())),
        np.asarray(list(grads_unprec.d_target_d_occupancy())),
    )


def test_phenix_gradients_wrt_atomic_respects_precondition(monkeypatch):
    """PHRIDGE_PRECONDITION routes Phenix atomic grads through packed GN preconditioning."""
    xs, i_obs, r_free, _ = _make_test_data()
    for sc in xs.scatterers():
        sc.flags.set_grad_site(True)
        sc.flags.set_grad_u_iso(True)
        sc.flags.set_grad_occupancy(True)
    fmodel = IntensityFModel(i_obs, xs, r_free, memory=True)

    _, g_ref_plain = fmodel.target_and_gradients(precondition=False)
    packed_plain = np.asarray(list(g_ref_plain.packed()))

    monkeypatch.setenv("PHRIDGE_PRECONDITION", "1")
    res_pre = fmodel.target_functor()(compute_gradients=True)
    g_pre = np.asarray(list(res_pre.gradients_wrt_atomic_parameters()))
    assert res_pre._atomic_gradients is not None
    assert res_pre._atomic_gradients.preconditioned is True
    assert np.allclose(g_pre, np.asarray(list(res_pre._atomic_gradients.packed())))
    assert not np.allclose(g_pre, packed_plain)


def test_target_w_and_target_t():
    xs, i_obs, r_free, _ = _make_test_data()
    fmodel = IntensityFModel(i_obs, xs, r_free, memory=True)

    tw = fmodel.target_w()
    tt = fmodel.target_t()
    assert isinstance(tw, float)
    assert isinstance(tt, float)
    assert np.isfinite(tw)
    assert np.isfinite(tt)

    # one_time_gradients_wrt_atomic_parameters
    grads = fmodel.one_time_gradients_wrt_atomic_parameters()
    assert isinstance(grads, IntensityGradients)
    assert grads.d_target_d_site_cart().size() == len(xs.scatterers())


def test_electron_density_map_module():
    xs, i_obs, r_free, _ = _make_test_data()
    fmodel = IntensityFModel(i_obs, xs, r_free, memory=True)

    emap = fmodel.electron_density_map(newton_damping=0.1, include_free=True)
    assert isinstance(emap, IntensityElectronDensityMap)

    # 2mFo-DFc map coefficients
    c_2fofc = emap.map_coefficients("2mFo-DFc")
    assert c_2fofc.is_complex_array()
    assert c_2fofc.size() == i_obs.size()

    # mFo-DFc map coefficients
    c_fofc = emap.map_coefficients("mFo-DFc")
    assert c_fofc.is_complex_array()
    assert c_fofc.size() == i_obs.size()

    # Phenix / mmtbx standard naming conventions (e.g. for ordered_solvent)
    c_mfobs = emap.map_coefficients("mFobs-DFmodel")
    assert c_mfobs.is_complex_array()
    assert c_mfobs.size() == i_obs.size()

    c_2mfobs = emap.map_coefficients("2mFobs-DFmodel")
    assert c_2mfobs.is_complex_array()
    assert c_2mfobs.size() == i_obs.size()

    c_fmodel = emap.map_coefficients("Fmodel")
    assert c_fmodel.is_complex_array()
    assert c_fmodel.size() == i_obs.size()

    c_anom = emap.map_coefficients("anomalous")
    assert c_anom is None

    # Difference map should have smaller amplitude than 2mFo-DFc
    assert flex.mean(c_fofc.amplitudes().data()) < flex.mean(c_2fofc.amplitudes().data())

    # Map scaling should be consistent with f_calc (in electrons)
    fc_mean = flex.mean(fmodel.f_calc().amplitudes().data())
    m_2fofc_mean = flex.mean(c_2fofc.amplitudes().data())
    assert abs(m_2fofc_mean - fc_mean) / fc_mean < 0.5

    # Test fill_missing
    c_filled = emap.map_coefficients("2mFo-DFc", fill_missing=True)
    assert c_filled.is_complex_array()
    assert c_filled.size() >= i_obs.size()

    # Test k_isotropic and k_anisotropic accessors
    assert hasattr(fmodel, "k_isotropic")
    assert hasattr(fmodel, "k_anisotropic")
    k_iso = fmodel.k_isotropic()
    k_aniso = fmodel.k_anisotropic()
    assert k_iso.size() == i_obs.size()
    assert k_aniso.size() == i_obs.size()

    # Gradient and Newton map coefficients
    c_grad = emap.map_coefficients("gradient")
    c_newt = emap.map_coefficients("newton")
    assert c_grad.is_complex_array()
    assert c_newt.is_complex_array()

    # Figures of merit and posterior mode/mean amplitudes
    assert emap.fom.size() == i_obs.size()
    assert emap.f_mode.size() == i_obs.size()
    assert emap.f_post.size() == i_obs.size()
    assert flex.min(emap.f_mode.data()) >= 0.0

    # Real space maps
    two_real = emap.two_fofc_map(resolution_factor=0.25)
    fofc_real = emap.fofc_map(resolution_factor=0.25)
    assert len(two_real.focus()) == 3
    assert len(fofc_real.focus()) == 3

    # CCTBX fft_map object compatibility (used in phenix.refinement.xyz_real_space)
    fft_obj = emap.fft_map(resolution_factor=0.25, map_type="2mFo-DFc", use_all_data=False)
    assert hasattr(fft_obj, "apply_sigma_scaling")
    assert hasattr(fft_obj, "real_map_unpadded")
    fft_obj.apply_sigma_scaling()
    map_data = fft_obj.real_map_unpadded()
    assert len(map_data.focus()) == 3

    # Direct map methods on fmodel
    two_direct = fmodel.two_fofc_map()
    fofc_direct = fmodel.fofc_map()
    assert two_direct.focus() == two_real.focus()
    assert fofc_direct.focus() == fofc_real.focus()


def test_inferred_r_values_from_posterior_mode():
    xs, i_obs, r_free, _ = _make_test_data()
    fmodel = IntensityFModel(i_obs, xs, r_free, memory=True)

    rv = fmodel.inferred_r_values()
    assert "r_work" in rv
    assert "r_free" in rv
    assert "r_all" in rv
    assert "cc_work" in rv
    assert "cc_free" in rv

    # Posterior mean estimate <F>
    assert "r_post_work" in rv
    assert "r_post_free" in rv
    assert "r_post_all" in rv
    assert "cc_post_work" in rv
    assert "cc_post_free" in rv

    # Posterior mode estimate F_mode
    assert "r_mode_work" in rv
    assert "r_mode_free" in rv
    assert "r_mode_all" in rv
    assert "cc_mode_work" in rv
    assert "cc_mode_free" in rv

    # Direct intensities
    assert "r_intensity_work" in rv
    assert "r_intensity_free" in rv
    assert "r_intensity_all" in rv
    assert "cc_intensity_work" in rv
    assert "cc_intensity_free" in rv

    rw = fmodel.r_work()
    rf = fmodel.r_free()
    ra = fmodel.r_all()
    assert 0.0 <= rw <= 1.0
    assert 0.0 <= rf <= 1.0
    assert 0.0 <= ra <= 1.0

    rw_post = fmodel.r_post_work()
    rf_post = fmodel.r_post_free()
    assert 0.0 <= rw_post <= 1.0
    assert 0.0 <= rf_post <= 1.0

    rw_mode = fmodel.r_mode_work()
    rf_mode = fmodel.r_mode_free()
    assert 0.0 <= rw_mode <= 1.0
    assert 0.0 <= rf_mode <= 1.0

    rw_i = fmodel.r_intensity_work()
    rf_i = fmodel.r_intensity_free()
    assert 0.0 <= rw_i <= 2.0
    assert 0.0 <= rf_i <= 2.0

    # Amplitudes in electron units
    assert fmodel.f_post is not None
    assert fmodel.f_post.size() == i_obs.size()
    assert fmodel.f_mode is not None
    assert fmodel.f_mode.size() == i_obs.size()

    # Formatted string
    r_str = fmodel.r_factors(as_string=True)
    assert "r_work=" in r_str
    assert "r_free=" in r_str

    # Group args object
    r_obj = fmodel.r_factors(as_string=False)
    assert hasattr(r_obj, "r_work")
    assert hasattr(r_obj, "r_free")

    # IntensityFModelInfo display
    info = fmodel.info()
    assert hasattr(info, "r_post_work")
    assert hasattr(info, "r_mode_work")
    assert hasattr(info, "r_intensity_work")
    import io
    buf = io.StringIO()
    info.show_rfactors_targets_scales_overall(out=buf)
    out_str = buf.getvalue()
    assert "Direct Intensity" in out_str
    # The S family replaced the integrated R names, and the shrunken posterior
    # mean / mode statistics are no longer part of the default banner.
    assert "S_post" in out_str
    assert "S_prior" in out_str
    assert "Posterior Mean <F>" not in out_str
    assert "Posterior Mode" not in out_str


def test_model_and_scale_updates():
    xs, i_obs, r_free, _ = _make_test_data()
    fmodel = IntensityFModel(i_obs, xs, r_free, memory=True)

    # Update atomic structure
    xs_new = xs.deep_copy_scatterers()
    sites = xs_new.sites_cart()
    sites[0] = (sites[0][0] + 0.1, sites[0][1], sites[0][2])
    xs_new.set_sites_cart(sites)

    fmodel.update_xray_structure(xs_new)
    assert fmodel._f_calc is None  # Cache cleared
    fc = fmodel.f_calc()
    assert fc.size() == i_obs.size()

    # Scale update via worker nuisance fit
    init_scale = fmodel.scale_factor
    res = fmodel.update_all_scales(fit_scale=True, fit_nu=False)
    assert res is fmodel
    assert fmodel.sigma_a is not None
    assert fmodel.sigma_wilson is not None
    assert fmodel.scale_factor > 0.0

    # Deep copy
    fmodel_copy = fmodel.deep_copy()
    assert fmodel_copy.scale_factor == fmodel.scale_factor
    assert fmodel_copy.i_obs().size() == fmodel.i_obs().size()


def test_lbfgs_minimization_with_intensity_fmodel():
    from scitbx import lbfgs

    xs, i_obs, r_free, _ = _make_test_data()

    # Shake coordinates slightly
    xs_shaken = xs.deep_copy_scatterers()
    for sc in xs_shaken.scatterers():
        sc.flags.set_grad_site(True)
        sc.flags.set_grad_u_iso(False)
    shift = flex.vec3_double(np.random.normal(0, 0.08, size=(len(xs.scatterers()), 3)))
    xs_shaken.set_sites_cart(xs.sites_cart() + shift)

    fmodel = IntensityFModel(i_obs, xs_shaken, r_free, memory=True)
    initial_target = fmodel.target_w()

    class LbfgsAdapter:
        def __init__(self, fmod, precondition=True):
            self.fmod = fmod
            self.precondition = precondition
            self.x = flex.double(fmod.xray_structure.sites_cart().as_double())

        def compute_functional_and_gradients(self):
            sc = self.fmod.xray_structure
            sc.set_sites_cart(flex.vec3_double(self.x))
            self.fmod.update_xray_structure(sc)
            f, g = self.fmod.target_and_gradients(precondition=self.precondition)
            return f, g.d_target_d_site_cart().as_double()

    adapter = LbfgsAdapter(fmodel, precondition=True)
    pr = lbfgs.core_parameters(maxfev=15)
    lbfgs.run(target_evaluator=adapter, core_params=pr)

    final_target = fmodel.target_w()
    assert final_target < initial_target


def test_student_t_nu_target_and_maps():
    xs, i_obs, r_free, _ = _make_test_data()
    fmodel_t = IntensityFModel(i_obs, xs, r_free, nu=5.0, memory=True)
    assert fmodel_t.nu == 5.0
    assert fmodel_t.target_spec["nu"] == 5.0

    val, grads = fmodel_t.target_and_gradients(precondition=True)
    assert val > 0.0

    emap = fmodel_t.electron_density_map()
    rw = emap.robust_weight
    assert rw.size() == i_obs.size()
    # Robust weights for Student-t should be strictly positive
    assert flex.min(rw.data()) > 0.0


def test_target_functor_compatibility():
    xs, i_obs, r_free, fc = _make_test_data()
    fmodel = IntensityFModel(i_obs, xs, r_free, memory=True)

    tf = fmodel.target_functor()
    assert tf.f_obs().size() == i_obs.size()

    res = tf(fc, compute_gradients=True)
    assert res.target() is not None
    assert res.derivatives() is not None
    assert res.derivatives().size() == fc.size()


def test_client_clean_import_no_torch():
    import subprocess
    import sys

    code = (
        "import sys\n"
        "from phridge.client.intensity import IntensityFModel, IntensityElectronDensityMap\n"
        "assert 'torch' not in sys.modules, f'torch was unexpectedly imported: {list(sys.modules.keys())}'\n"
        "print('SUCCESS_NO_TORCH')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert "SUCCESS_NO_TORCH" in result.stdout


def test_twinning_hard_error():
    """Verify that requesting twinning with mli_quad raises a hard error."""
    import pytest
    from phridge.client.intensity.engine import IntensityTwinningError

    xs, i_obs, r_free, _ = _make_test_data()

    # 1. twin=True passed in constructor
    with pytest.raises(IntensityTwinningError) as exc_info:
        IntensityFModel(i_obs, xs, r_free, twin=True, memory=True)
    assert issubclass(IntensityTwinningError, RuntimeError)
    assert "Twinning is not supported" in str(exc_info.value)

    # 2. twin_law passed in constructor
    with pytest.raises(IntensityTwinningError):
        IntensityFModel(i_obs, xs, r_free, twin_law="-h,-k,l", memory=True)

    # 3. is_twin=True passed in constructor
    with pytest.raises(IntensityTwinningError):
        IntensityFModel(i_obs, xs, r_free, is_twin=True, memory=True)

    # 4. twin=True via kwargs
    with pytest.raises(IntensityTwinningError):
        IntensityFModel(i_obs, xs, r_free, **{"twin": True}, memory=True)

    # 5. Setting twin property after construction
    fmodel = IntensityFModel(i_obs, xs, r_free, memory=True)
    assert not fmodel.is_twin_fmodel_manager()
    with pytest.raises(IntensityTwinningError):
        fmodel.twin = True

    # 6. Setting twin_law property after construction
    with pytest.raises(IntensityTwinningError):
        fmodel.twin_law = "-h,-k,l"


def test_synthetic_scale_and_b_study():
    """Verify scale estimation and R-values on synthetic data with known scale and B-factor."""
    xs, _, _, fc = _make_test_data()
    r_free = fc.generate_r_free_flags(fraction=0.10)
    fc_amp = np.abs(np.asarray(fc.data()))
    s_sq = np.asarray(1.0 / (4.0 * flex.pow2(fc.d_spacings().data())))

    # 1. Pure scale factor k=3.5, delta_B=0
    k_true = 3.5
    i_synth = (k_true**2) * (fc_amp**2)
    sig_synth = np.maximum(i_synth * 0.02, 0.01)
    i_obs_scaled = fc.customized_copy(data=flex.double(i_synth), sigmas=flex.double(sig_synth))

    fm = IntensityFModel(i_obs=i_obs_scaled, xray_structure=xs, r_free_flags=r_free, memory=True)
    fm.update_all_scales(fit_scale=True, fit_nu=False)
    assert abs(fm.scale_factor - k_true) / k_true < 0.05
    assert fm.r_work() < 0.02  # R-work < 2% on exact model

    # 2. Scale factor k=2.0 with delta_B=15.0
    k_true = 2.0
    delta_B = 15.0
    i_synth_b = (k_true**2) * (fc_amp**2) * np.exp(-2.0 * delta_B * s_sq)
    sig_synth_b = np.maximum(i_synth_b * 0.02, 0.01)
    i_obs_b = fc.customized_copy(data=flex.double(i_synth_b), sigmas=flex.double(sig_synth_b))

    fm_b = IntensityFModel(i_obs=i_obs_b, xray_structure=xs, r_free_flags=r_free, memory=True)
    fm_b.update_all_scales(fit_scale=True, fit_nu=False)
    # Scale is recovered by anisotropic/overall scaling
    assert abs(fm_b.scale_factor - k_true) / k_true < 0.35
    assert fm_b.r_work() < 0.05

    # When atomic B-factors are updated to true B, scale is recovered and R drops
    xs_true_b = xs.deep_copy_scatterers()
    for sc in xs_true_b.scatterers():
        sc.u_iso = sc.u_iso + delta_B / (8.0 * np.pi**2)
    fm_b.update_xray_structure(xs_true_b)
    fm_b.update_all_scales(fit_scale=True, fit_nu=False)
    assert abs(fm_b.scale_factor - k_true) / k_true < 0.05
    assert fm_b.r_work() < 0.02


def test_residual_scale_folded_after_apply_back_trace():
    """Phenix apply_back_trace=True leaves overall k in residual scale_k1; fold into k_iso."""
    xs, _, _, fc = _make_test_data()
    r_free = fc.generate_r_free_flags(fraction=0.10)
    k_true = 0.35
    fc_amp = np.abs(np.asarray(fc.data()))
    i_synth = (k_true**2) * (fc_amp**2)
    sig_synth = np.maximum(i_synth * 0.02, 0.01)
    i_obs = fc.customized_copy(data=flex.double(i_synth), sigmas=flex.double(sig_synth))

    fm = IntensityFModel(i_obs=i_obs, xray_structure=xs, r_free_flags=r_free, memory=True)
    # Mimic Phenix BSS with apply_back_trace (no worker / nuisance fit needed).
    from mmtbx.bulk_solvent import f_model_all_scales

    fm.__dict__["xray_structure"] = fm._xray_structure
    fm.__dict__["twin"] = False
    fm.__dict__["twin_law"] = None
    fm.__dict__["target_name"] = "mli_quad"
    f_model_all_scales.run(
        fmodel=fm,
        apply_back_trace=True,
        remove_outliers=False,
        fast=True,
        params=None,
        refine_hd_scattering=True,
        log=None,
    )
    # Before fold, residual LS vs f_model should recover k_true (scale not in f_model).
    fm._scale_fitted = True
    residual_before = fm.scale_k1()
    assert abs(residual_before - k_true) / k_true < 0.05

    fm._fold_residual_scale_into_k_isotropic()
    fm.scale_factor = float(np.mean(np.asarray(fm.k_isotropic()) * np.asarray(fm.k_anisotropic())))

    assert abs(fm.scale_k1() - 1.0) < 1e-4
    assert abs(fm.scale_factor - k_true) / k_true < 0.05
    fo = flex.sqrt(flex.abs(fm.i_obs().data()))
    fm_abs = flex.abs(fm.f_model().data())
    ls = float(flex.sum(fo * fm_abs) / flex.sum(fm_abs * fm_abs))
    assert abs(ls - 1.0) < 1e-4


def test_bulk_solvent_mask_and_scaling():
    """Verify that bulk solvent mask (Fmask) is computed, scaled, and incorporated into F_model."""
    from pathlib import Path
    import iotbx.pdb
    import iotbx.mtz

    pdb_path = Path("examples/6czg/6czg.pdb")
    mtz_path = Path("examples/6czg/6czg.mtz")
    if not pdb_path.exists() or not mtz_path.exists():
        pytest.skip("6czg example files not found")

    pdb_in = iotbx.pdb.input(str(pdb_path))
    xrs = pdb_in.xray_structure_simple()
    mtz_in = iotbx.mtz.object(str(mtz_path))
    i_obs = None
    r_free = None
    for ma in mtz_in.as_miller_arrays():
        if ma.is_xray_intensity_array():
            i_obs = ma
        if ma.is_integer_array() and "free" in ma.info().labels[0].lower():
            r_free = ma

    assert i_obs is not None and r_free is not None
    fm = IntensityFModel(
        i_obs=i_obs,
        xray_structure=xrs,
        r_free_flags=r_free.array(data=(r_free.data() == 1)),
        memory=True,
    )

    # Verify solvent mask exists
    assert len(fm.f_masks()) >= 1
    assert fm.f_mask() is not None
    assert fm.f_mask().size() == i_obs.size()

    # Before bulk solvent scaling, k_mask is zero
    assert np.all(np.asarray(fm.k_masks()[0]) == 0.0)

    # Run bulk solvent & scale update
    fm.update_all_scales(fast=True)

    # Solvent parameters should be fitted
    ksol, bsol = fm.k_sol_b_sol()
    assert ksol is not None and ksol > 0.1
    assert bsol is not None and bsol > 10.0
    assert fm.k_sol is not None and fm.k_sol > 0.1
    assert fm.b_sol is not None and fm.b_sol > 10.0

    # F_bulk should now be non-zero
    fb = fm.f_bulk()
    assert fb is not None
    assert np.mean(np.abs(np.asarray(fb.data()))) > 5.0

    # F_model should include bulk solvent and drop R-work below 25% (down from 31% without solvent)
    rw = fm.r_work()
    rf = fm.r_free()
    assert rw < 0.25, f"Expected R_work < 25% with bulk solvent, got {rw*100:.2f}%"
    assert rf < 0.30, f"Expected R_free < 30% with bulk solvent, got {rf*100:.2f}%"


def test_preconditioners_and_nu_toggles(monkeypatch):
    """Verify programmatic and environment variable toggles for preconditioner and nu refinement."""
    xs, i_obs, r_free, _ = _make_test_data()

    # 1. Nu toggle programmatic
    # Default is None (Gaussian)
    fm_gauss = IntensityFModel(i_obs, xs, r_free, memory=True)
    assert fm_gauss.nu is None
    assert fm_gauss.target_spec.get("nu") is None

    # Explicit nu=6.0 (Student-t)
    fm_t = IntensityFModel(i_obs, xs, r_free, nu=6.0, memory=True)
    assert fm_t.nu == 6.0
    assert fm_t.target_spec["nu"] == 6.0

    # Nu via environment variable PHRIDGE_NU
    monkeypatch.setenv("PHRIDGE_NU", "8.5")
    fm_env_nu = IntensityFModel(i_obs, xs, r_free, memory=True)
    assert fm_env_nu.nu == 8.5
    assert fm_env_nu.target_spec["nu"] == 8.5
    monkeypatch.delenv("PHRIDGE_NU", raising=False)

    # 2. Preconditioner toggle programmatic
    _, g_unprec = fm_gauss.target_and_gradients(precondition=False)
    assert g_unprec.preconditioned is False

    _, g_prec = fm_gauss.target_and_gradients(precondition=True)
    assert g_prec.preconditioned is True
    assert g_prec.curvatures is not None

    # Preconditioner via environment variable PHRIDGE_PRECONDITION
    monkeypatch.setenv("PHRIDGE_PRECONDITION", "1")
    _, g_env = fm_gauss.target_and_gradients()
    assert g_env.preconditioned is True
    monkeypatch.delenv("PHRIDGE_PRECONDITION", raising=False)

    # 3. Fit nu toggle programmatic & via env var
    fm_fit = IntensityFModel(i_obs, xs, r_free, nu=5.0, memory=True)
    fm_fit.update_all_scales(fit_nu=True, nu_bounds=(3.0, 15.0), bulk_solvent_and_scaling=False)
    assert fm_fit.nu is not None

    monkeypatch.setenv("PHRIDGE_FIT_NU", "1")
    fm_env_fit = IntensityFModel(i_obs, xs, r_free, nu=5.0, memory=True)
    fm_env_fit.update_all_scales(bulk_solvent_and_scaling=False)
    assert fm_env_fit.nu is not None
    monkeypatch.delenv("PHRIDGE_FIT_NU", raising=False)





