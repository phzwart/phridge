"""Tests for phenix.refine mli_quad intensity likelihood integration hooks."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

import numpy as np
import pytest

from cctbx import crystal, miller, sgtbx, xray
from cctbx.array_family import flex
from cctbx.development import random_structure
from libtbx import group_args

import mmtbx.f_model
import mmtbx.maps
import mmtbx.refinement.targets
import mmtbx.utils

from phridge.client.intensity.engine import (
    IntensityElectronDensityMap,
    IntensityFModel,
    IntensityFModelInfo,
    IntensityTargetFunctor,
    IntensityTargetResult,
    IntensityDataError,
)
from phridge.client.intensity.phenix_hook import (
    disable_intensity_in_phenix,
    enable_intensity_in_phenix,
    is_intensity_enabled,
    patch_phil_master_params,
)


@pytest.fixture(autouse=True)
def cleanup_hooks():
    """Ensure hooks are cleanly disabled after each test."""
    yield
    disable_intensity_in_phenix()


def _make_test_data(d_min: float = 2.5, seed: int = 123):
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


def test_target_registration():
    """Test target name registration for mli_quad and mli alias."""
    assert not is_intensity_enabled()
    enable_intensity_in_phenix()
    assert is_intensity_enabled()

    tn = mmtbx.refinement.targets.target_names
    assert "mli_quad" in tn
    assert "mli" in tn

    attr = tn["mli_quad"]
    assert attr.family == "ml"
    assert attr.specialization == "i"
    assert not attr.requires_experimental_phases()


def test_phil_parsing_mli_quad():
    """Test PHIL choice validation recognizes mli_quad."""
    import iotbx.phil

    enable_intensity_in_phenix()

    # Create master scope mimicking phenix.refinement
    phil_str = """
    refinement {
      main {
        target = *auto ml mlhl ml_sad ls mli
          .type = choice
      }
    }
    """
    master = iotbx.phil.parse(phil_str)

    # Patch the test scope
    nodes = master.get_without_substitution("refinement.main.target")
    words = [w.value for w in nodes[0].words]
    if "mli_quad" not in words:
        nodes[0].words.append(iotbx.phil.tokenizer.word("mli_quad"))

    fetched = master.fetch(iotbx.phil.parse("refinement.main.target = mli_quad")).extract()
    assert fetched.refinement.main.target == "mli_quad"


def test_fmodel_manager2_creates_intensity_fmodel():
    """Test mmtbx.utils.fmodel_manager2 creates an IntensityFModel when target_name='mli_quad'."""
    enable_intensity_in_phenix()
    xs, i_obs, r_free, _ = _make_test_data()

    fmodel = mmtbx.utils.fmodel_manager2(
        f_obs=i_obs,
        r_free_flags=r_free,
        xray_structure=xs,
        target_name="mli_quad",
    )

    assert isinstance(fmodel, IntensityFModel)
    assert fmodel.target_name == "mli_quad"
    assert not fmodel.is_twin_fmodel_manager()
    assert fmodel.i_obs().size() == i_obs.size()


def test_fmodel_manager2_rejects_twinning():
    """Test that mmtbx.utils.fmodel_manager2 raises IntensityTwinningError if twin/twin_law is provided."""
    enable_intensity_in_phenix()
    xs, i_obs, r_free, _ = _make_test_data()

    from phridge.client.intensity.engine import IntensityTwinningError

    # 1. twin_law provided to fmodel_manager2
    with pytest.raises(IntensityTwinningError) as exc_info:
        mmtbx.utils.fmodel_manager2(
            f_obs=i_obs,
            r_free_flags=r_free,
            xray_structure=xs,
            target_name="mli_quad",
            twin_law="-h,-k,l",
        )
    assert "Twinning is not supported" in str(exc_info.value)

    # 2. twin=True provided to fmodel_manager2
    with pytest.raises(IntensityTwinningError):
        mmtbx.utils.fmodel_manager2(
            f_obs=i_obs,
            r_free_flags=r_free,
            xray_structure=xs,
            target_name="mli_quad",
            twin=True,
        )

    # 3. twin=True provided to fmodel_manager
    with pytest.raises(IntensityTwinningError):
        mmtbx.utils.fmodel_manager(
            f_obs=i_obs,
            r_free_flags=r_free,
            xray_structure=xs,
            target_name="mli_quad",
            twin=True,
        )


def test_target_functor_dispatch_and_evaluation():
    """Test mmtbx.refinement.targets.target_functor produces IntensityTargetFunctor."""
    enable_intensity_in_phenix()
    xs, i_obs, r_free, _ = _make_test_data()

    fmodel = IntensityFModel(i_obs, xs, r_free, memory=True)
    tf = mmtbx.refinement.targets.target_functor(fmodel)

    assert isinstance(tf, IntensityTargetFunctor)
    assert tf.target_function_is_invariant_under_allowed_origin_shifts()

    tres = tf(compute_gradients=True)
    assert isinstance(tres, IntensityTargetResult)
    assert tres.target_work() > 0.0
    assert tres.target_test() is not None
    assert tres.gradients_work().size() > 0


def test_fmodels_target_and_gradients_macro_cycle():
    """Test full mmtbx.fmodels macro cycle target and gradients calculation."""
    enable_intensity_in_phenix()
    xs, i_obs, r_free, _ = _make_test_data()

    fmodel = mmtbx.utils.fmodel_manager2(
        f_obs=i_obs,
        r_free_flags=r_free,
        xray_structure=xs,
        target_name="mli_quad",
    )

    fmodels = mmtbx.fmodels(
        fmodel_xray=fmodel,
        xray_scattering_dict=xs.scattering_type_registry().as_type_gaussian_dict(),
    )

    tg = fmodels.target_and_gradients(compute_gradients=True)
    assert tg.target_work_xray > 0.0
    assert len(tg.gradient_xray) == xs.n_parameters()
    assert np.all(np.isfinite(tg.gradient_xray))


def test_electron_density_map_and_compute_map_coefficients():
    """Test generation of 2mFo-DFc and mFo-DFc map coefficients via mmtbx.maps."""
    enable_intensity_in_phenix()
    xs, i_obs, r_free, _ = _make_test_data()

    fmodel = IntensityFModel(i_obs, xs, r_free, memory=True)
    edm = fmodel.electron_density_map()

    assert isinstance(edm, IntensityElectronDensityMap)
    assert edm.model is not None
    assert edm.difference is not None
    assert edm.gradient is not None
    assert edm.newton is not None

    mcp1 = group_args(
        map_type="2mFo-DFc",
        format=["mtz"],
        isotropize=False,
        fill_missing_f_obs=False,
        sharpening=False,
        acentrics_scale=2.0,
        centrics_pre_scale=1.0,
        exclude_free_r_reflections=False,
        mtz_label_amplitudes="2FOFCWT",
        mtz_label_phases="PH2FOFCWT",
    )
    mcp2 = group_args(
        map_type="mFo-DFc",
        format=["mtz"],
        isotropize=False,
        fill_missing_f_obs=False,
        sharpening=False,
        acentrics_scale=1.0,
        centrics_pre_scale=1.0,
        exclude_free_r_reflections=False,
        mtz_label_amplitudes="FOFCWT",
        mtz_label_phases="PHFOFCWT",
    )

    cmo = mmtbx.maps.compute_map_coefficients(fmodel=fmodel, params=[mcp1, mcp2])
    assert len(cmo.map_coeffs) == 2
    assert cmo.mtz_dataset is not None


def test_inferred_and_intensity_r_factors_and_info():
    """Test posterior mode and direct intensity R-factors and info formatting."""
    enable_intensity_in_phenix()
    xs, i_obs, r_free, _ = _make_test_data()

    fmodel = IntensityFModel(i_obs, xs, r_free, memory=True)

    rw = fmodel.r_work()
    rf = fmodel.r_free()
    ra = fmodel.r_all()
    assert 0.0 < rw < 2.0
    assert 0.0 < rf < 2.0
    assert 0.0 < ra < 2.0

    ri_w = fmodel.r_intensity_work()
    ri_f = fmodel.r_intensity_free()
    ri_a = fmodel.r_intensity_all()
    assert 0.0 < ri_w < 2.0
    assert 0.0 < ri_f < 2.0
    assert 0.0 < ri_a < 2.0

    info = fmodel.info()
    assert isinstance(info, IntensityFModelInfo)
    assert info.r_work == rw
    assert info.r_intensity_work == ri_w

    import io

    buf = io.StringIO()
    info.show_rfactors_targets_scales_overall(out=buf)
    out_str = buf.getvalue()
    assert "mli_quad" in out_str
    assert "Direct Intensity R:" in out_str
    assert "S_post" in out_str
    assert "Posterior Mode" not in out_str


def test_clean_import_no_torch():
    """Verify that importing phenix_hook does not eagerly import torch on the client."""
    code = """
import sys
from phridge.client.intensity.phenix_hook import enable_intensity_in_phenix, disable_intensity_in_phenix
assert 'torch' not in sys.modules, f'torch was imported: {sys.modules.get("torch")}'
print('OK')
"""
    cmd = [sys.executable, "-c", code]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "OK" in res.stdout


def test_french_wilson_and_f2_disabled():
    """mli_quad hooks block French–Wilson and refuse F² reconstruction."""
    from cctbx import french_wilson
    from libtbx.utils import null_out

    xs, i_obs, r_free, fc = _make_test_data()
    # Inject negatives
    data = list(i_obs.data())
    data[0] = -5.0
    data[1] = -2.0
    i_neg = i_obs.customized_copy(data=flex.double(data)).set_observation_type_xray_intensity()

    enable_intensity_in_phenix()
    try:
        with pytest.raises(IntensityDataError):
            french_wilson.french_wilson_scale(miller_array=i_neg, params=None, log=null_out())

        fm = IntensityFModel(i_obs=i_neg, xray_structure=xs, r_free_flags=r_free, memory=True)
        assert int((np.asarray(fm.i_obs().data()) < 0).sum()) == 2

        # extract_xtal_data path builds scaffold without FW / F²
        from iotbx import extract_xtal_data, reflection_file_reader, reflection_file_utils
        from pathlib import Path
        import tempfile

        mtz_path = Path(tempfile.mkdtemp()) / "neg.mtz"
        i_neg.as_mtz_dataset(column_root_label="I").mtz_object().write(str(mtz_path))
        rfs = reflection_file_utils.reflection_file_server(
            reflection_files=[reflection_file_reader.any_reflection_file(str(mtz_path))],
            crystal_symmetry=xs.crystal_symmetry(),
        )
        params = extract_xtal_data.data_and_flags_master_params().extract()
        assert params.french_wilson_scale is True  # default before run
        ex = extract_xtal_data.run(
            reflection_file_server=rfs,
            parameters=params,
            working_point_group=xs.space_group().build_derived_point_group(),
            keep_going=True,
        )
        assert params.french_wilson_scale is False  # forced by patch
        res = ex.result()
        assert res.i_obs is not None
        n_neg = int((np.asarray(res.i_obs.data()) < 0).sum())
        assert n_neg >= 2
        assert res.f_obs.size() >= res.i_obs.size() - 5  # no FW mass-rejection
    finally:
        disable_intensity_in_phenix()


def test_weight_metric_helpers_and_patches():
    """NLL weight-metric helpers + scorer/ADP patches install cleanly."""
    from phridge.client.intensity.phenix_hook import (
        _weight_metric_is_nll,
    )

    old = os.environ.get("PHRIDGE_WEIGHT_METRIC")
    try:
        os.environ["PHRIDGE_WEIGHT_METRIC"] = "nll"
        assert _weight_metric_is_nll() is True
        os.environ["PHRIDGE_WEIGHT_METRIC"] = "rfree"
        assert _weight_metric_is_nll() is False
    finally:
        if old is None:
            os.environ.pop("PHRIDGE_WEIGHT_METRIC", None)
        else:
            os.environ["PHRIDGE_WEIGHT_METRIC"] = old

    enable_intensity_in_phenix()

    import phenix.refinement.xyz_reciprocal_space as xyz_rs
    import mmtbx.refinement.adp_refinement as adp_ref

    assert xyz_rs.xyz_refinement_scorer._select_best.__name__ == "patched_select_best"
    assert adp_ref.refine_adp.show.__name__ == "patched_adp_show"

    disable_intensity_in_phenix()
    assert xyz_rs.xyz_refinement_scorer._select_best.__name__ == "_select_best"


def test_phenix_python_execution_if_available():
    """If phenix.python binary is available, test hook enablement inside real Phenix env."""
    phenix_py = "/Users/phzwart/Applications/phenix-2.0-5936/bin/phenix.python"
    if not os.path.isfile(phenix_py) or not os.access(phenix_py, os.X_OK):
        pytest.skip("phenix.python not available on this system")

    code = """
import sys
import phridge
from phridge.client.intensity.phenix_hook import enable_intensity_in_phenix, disable_intensity_in_phenix, is_intensity_enabled
import mmtbx.refinement.targets
import phenix.refinement
import iotbx.phil

assert not is_intensity_enabled()
enable_intensity_in_phenix()
assert is_intensity_enabled()

assert 'mli_quad' in mmtbx.refinement.targets.target_names
assert 'mli' in mmtbx.refinement.targets.target_names

mp = phenix.refinement.master_params()
fetched = mp.fetch(iotbx.phil.parse('refinement.main.target=mli_quad')).extract()
assert fetched.refinement.main.target == 'mli_quad'

disable_intensity_in_phenix()
assert not is_intensity_enabled()
print('PHENIX_PYTHON_OK')
"""
    cmd = [phenix_py, "-c", code]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "PHENIX_PYTHON_OK" in res.stdout
