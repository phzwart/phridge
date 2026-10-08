"""The ``ml_i_maps`` worker op still lives in legacy. Importing it registers the op."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.legacy

torch = pytest.importorskip("torch")

import phridge.legacy.intensity_ll_ops  # noqa: E402, F401

from phridge.api import Bridge  # noqa: E402
from phridge.contracts.models import CrystalSymmetry, ObservationType  # noqa: E402
from phridge.contracts.packing import PackedMiller  # noqa: E402
from phridge.contrib.intensity_ll import MAPS_OP_NAME  # noqa: E402
from phridge.d.maps import IntensityMapOptions, intensity_map_coefficients  # noqa: E402
from phridge.d.rules.amplitude import Observations  # noqa: E402
from phridge.d.rules.mli_quad.target import IntensityLogLikelihood  # noqa: E402
from phridge.ops import get_implementation, get_op  # noqa: E402


def _synthetic(n: int = 200, seed: int = 5, sA: float = 0.85, sigma: float = 100.0) -> tuple:
    rng = np.random.default_rng(seed)
    ft = np.sqrt(sigma / 2) * (rng.normal(size=n) + 1j * rng.normal(size=n))
    fc = sA * ft + np.sqrt(sigma * (1 - sA**2) / 2) * (rng.normal(size=n) + 1j * rng.normal(size=n))
    sig = np.exp(rng.uniform(np.log(0.5), np.log(150), n))
    io = np.abs(ft) ** 2 + sig * rng.normal(size=n)
    obs = Observations.from_numpy(
        data=io,
        sigmas=sig,
        epsilon=np.ones(n),
        centric=np.zeros(n, bool),
        alpha=np.full(n, sA),
        beta=np.full(n, sigma),
    )
    return fc, io, sig, obs


def test_register_adds_maps_op() -> None:
    spec = get_op(MAPS_OP_NAME)
    assert spec.outputs["difference"] == "MillerArray"
    assert spec.inputs["maps"] == "json"
    assert get_implementation(MAPS_OP_NAME) is not None


def test_ml_i_maps_op_round_trip_memory_bridge() -> None:
    fc, io, sig, obs = _synthetic(n=200, seed=5)
    n = len(fc)
    rng = np.random.default_rng(0)
    hkl = rng.integers(-8, 9, size=(n, 3))
    crystal = CrystalSymmetry(unit_cell=[20.0, 30.0, 40.0, 90.0, 90.0, 90.0], space_group_hall="P 1")
    f_obs = PackedMiller(crystal=crystal, hkl=hkl, data=io, sigmas=sig, observation_type=ObservationType.intensity)
    f_calc = PackedMiller(crystal=crystal, hkl=hkl, data=fc, observation_type=ObservationType.complex)
    bridge = Bridge(memory=True, timeout=5)
    out = bridge.call(
        MAPS_OP_NAME,
        f_calc=f_calc,
        f_obs=f_obs,
        target={"name": "ml_i", "nu": 6.0},
        alpha=np.full(n, 0.85),
        beta=np.full(n, 100.0),
        epsilon=np.ones(n),
        centric=np.zeros(n, bool),
        maps={"newton_damping": 0.2},
    )
    assert set(out) >= {"difference", "model", "gradient", "newton", "fom", "robust_weight", "stats", "d_loglik_d_nu"}
    diff = np.asarray(out["difference"].data)
    assert diff.shape == (n,) and np.iscomplexobj(diff)
    assert out["stats"]["nu"] == 6.0 and out["stats"]["n_ok"] == n
    tgt = IntensityLogLikelihood(nu=6.0)
    ms = intensity_map_coefficients(tgt, torch.as_tensor(fc), obs, IntensityMapOptions(newton_damping=0.2))
    np.testing.assert_allclose(diff, ms.difference, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(out["robust_weight"]), ms.robust_weight, rtol=1e-12)
    with pytest.raises(Exception, match="ml_i"):
        bridge.call(MAPS_OP_NAME, f_calc=f_calc, f_obs=f_obs, target={"name": "ml_f"})


def test_remote_intensity_maps_cctbx_round_trip() -> None:
    cctbx = pytest.importorskip("cctbx")  # noqa: F841
    from cctbx import sgtbx
    from cctbx.array_family import flex
    from cctbx.development import random_structure

    from phridge.contrib.intensity_ll.client import RemoteIntensityMaps

    flex.set_random_seed(0)
    xs = random_structure.xray_structure(
        space_group_info=sgtbx.space_group_info("P212121"),
        elements=["C", "N", "O", "S"] * 4,
        volume_per_atom=50,
        random_u_iso=True,
    )
    fc = xs.structure_factors(d_min=2.5, algorithm="direct").f_calc()
    xs2 = xs.deep_copy_scatterers()
    xs2.shake_sites_in_place(rms_difference=0.3)
    fc2 = xs2.structure_factors(d_min=2.5, algorithm="direct").f_calc()
    i_true = fc.amplitudes().data() ** 2
    n = i_true.size()
    rng = np.random.default_rng(0)
    sig = flex.double(0.05 * np.asarray(i_true) + 0.5)
    i_obs = fc.customized_copy(data=i_true + sig * flex.double(rng.normal(size=n)), sigmas=sig)
    i_obs.set_observation_type_xray_intensity()
    beta = flex.double(np.full(n, float(flex.mean(i_true))))
    alpha = flex.double(np.full(n, 0.9))
    r_free = flex.bool(rng.random(n) < 0.1)

    maps = RemoteIntensityMaps(
        Bridge(memory=True, timeout=10),
        i_obs,
        {"name": "ml_i", "nu": 8.0},
        alpha=alpha,
        beta=beta,
        r_free_flags=r_free,
        maps={"include_free": False},
    )
    res = maps(fc2)
    assert res.difference.is_complex_array() and res.difference.size() == n
    assert res.difference.indices().all_eq(i_obs.indices())
    assert res.model.size() == n and res.newton.size() == n and res.gradient.size() == n
    assert res.fom.size() == n and 0.0 <= flex.min(res.fom) and flex.max(res.fom) <= 1.0 + 1e-12
    assert res.robust_weight.size() == n and flex.min(res.robust_weight) > 0
    assert res.d_loglik_d_nu_total is not None
    free = np.asarray(r_free)
    diff = np.asarray(res.difference.data())
    assert np.all(diff[free] == 0) and np.all(diff[~free] != 0)
    assert res.stats["nu"] == 8.0
    ph = np.asarray(fc2.data()) / np.abs(np.asarray(fc2.data()))
    assert np.allclose(np.imag(diff[~free] * np.conj(ph[~free])), 0.0, atol=1e-9 * np.abs(diff).max())
