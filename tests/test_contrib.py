"""phridge.contrib.intensity_ll and target entry-point bootstrap."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from phridge.contrib.intensity_ll import register
from phridge.contrib.intensity_ll.target import (
    IntensityLogLikelihood,
    IntensityLogLikelihoodOptions,
)
from phridge.ops import bootstrap_plugins
from phridge.sfcalc.targets import build_target, list_targets


def test_register_adds_ml_i():
    register()
    assert "ml_i" in list_targets()
    tgt = build_target({"name": "ml_i", "sigma_a": 0.8, "sigma_wilson": 1.0})
    assert isinstance(tgt, IntensityLogLikelihood)
    assert tgt.sigma_a == 0.8
    assert tgt.sigma_wilson == 1.0


def test_options_pydantic_accept_reject():
    opts = IntensityLogLikelihoodOptions.model_validate(
        {"sigma_a": 0.7, "sigma_wilson": 2.0, "use_sigmas": False, "n_legendre": 16}
    )
    assert opts.sigma_a == 0.7
    assert opts.sigma_wilson == 2.0
    assert opts.use_sigmas is False
    assert opts.n_legendre == 16
    with pytest.raises(ValidationError):
        IntensityLogLikelihoodOptions.model_validate({"sigma_a": 1.5})
    with pytest.raises(ValidationError):
        IntensityLogLikelihoodOptions.model_validate({"sigma": -1.0})
    with pytest.raises(ValidationError):
        IntensityLogLikelihoodOptions.model_validate({"sigma_a": 0.5, "extra_field": True})
    with pytest.raises(ValidationError):
        build_target({"name": "ml_i", "sigma": 0.0})


def test_ml_i_evaluate_smoke():
    pytest.importorskip("torch")
    register()
    from phridge.sfcalc.targets import Observations

    rng = np.random.default_rng(0)
    n = 8
    f_calc = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex128)
    i_obs = (np.abs(f_calc) ** 2).astype(np.float64)
    obs = Observations.from_numpy(
        data=i_obs,
        sigmas=np.full(n, 0.5),
        epsilon=np.ones(n),
        centric=np.zeros(n, dtype=bool),
        alpha=np.full(n, 0.85),
        beta=np.ones(n),
    )
    import torch

    tgt = IntensityLogLikelihood()
    ev = tgt.evaluate(torch.as_tensor(f_calc), obs, compute_curvature=True)
    assert np.isfinite(ev.value)
    assert ev.value > 0.0
    assert ev.d_target_d_f_calc.shape == (n,)
    assert np.isfinite(ev.d_target_d_f_calc).all()
    assert ev.curv_radial is not None


def test_ml_i_requires_sigma_a_or_alpha():
    pytest.importorskip("torch")
    register()
    from phridge.sfcalc.targets import Observations
    import torch

    obs = Observations.from_numpy(data=np.ones(2), sigmas=np.ones(2), beta=np.ones(2))
    tgt = IntensityLogLikelihood()
    with pytest.raises(ValueError, match="sigma_A|alpha"):
        tgt.per_reflection(torch.ones(2, dtype=torch.complex128), obs)


def test_shims_still_import():
    from phridge.client import xtal_engine as xe
    from phridge.packing_scattering import PackedTargetResult
    from phridge.worker import targets as wt
    from phridge.worker import xtal as wx

    assert hasattr(xe, "StructureFactorServer")
    assert hasattr(wt, "register_target")
    assert hasattr(wx, "StructureFactorEngine")
    assert PackedTargetResult is not None


def test_bootstrap_loads_targets_group(monkeypatch):
    """Simulate a phridge.targets entry point without requiring reinstall."""
    calls: list[str] = []

    class _EP:
        name = "intensity_ll"

        def load(self):
            def _reg():
                calls.append(self.name)
                register()

            return _reg

    class _EPs:
        def select(self, group):
            if group == "phridge.targets":
                return [_EP()]
            return []

    monkeypatch.setattr(
        "importlib.metadata.entry_points",
        lambda: _EPs(),
    )
    # Ensure ml_i may already be registered; bootstrap should still invoke EP.
    out = bootstrap_plugins(load_eps=True, groups=("phridge.targets",), preload=())
    assert "intensity_ll" in out["entry_points"]
    assert out["entry_points_by_group"]["phridge.targets"] == ["intensity_ll"]
    assert calls == ["intensity_ll"]
    assert "ml_i" in list_targets()
