"""Engine side of the NLL binned k_mask fit: what gets installed, and when it does not."""

from __future__ import annotations

import types

import numpy as np
import pytest

engine = pytest.importorskip("phridge.client.intensity.engine")

_N = 40


class _Arr:
    def __init__(self, values):
        self._v = np.asarray(values)

    def data(self):
        return self

    def as_double(self):
        return self._v.astype(np.float64)

    def __array__(self, dtype=None, copy=None):
        return self._v if dtype is None else self._v.astype(dtype)


class _IObs:
    def size(self):
        return _N

    def d_spacings(self):
        return _Arr(np.linspace(20.0, 2.0, _N))

    def epsilons(self):
        return _Arr(np.ones(_N))

    def centric_flags(self):
        return _Arr(np.zeros(_N, dtype=bool))


class _Bridge:
    def __init__(self, raw):
        self.raw, self.calls = raw, []

    def call(self, name, **kw):
        self.calls.append((name, kw))
        return self.raw


def _probe(monkeypatch, raw):
    monkeypatch.setattr(
        engine, "flex", types.SimpleNamespace(double=np.asarray, bool=np.asarray), raising=False
    )
    probe = engine.IntensityFModel.__new__(engine.IntensityFModel)
    probe.__dict__.update(
        arrays=object(),
        _i_obs=_IObs(),
        _r_free_flags=None,
        sigma_wilson=np.ones(_N),
        nu_per_refl=None,
        bridge=_Bridge(raw),
    )
    for name in ("_f_model", "_last_maps", "_r_values", "_f_post", "_f_mode", "_last_gradients"):
        setattr(probe, name, object())
    installed, folds = [], []
    probe.update_core = lambda **kw: installed.append(kw)
    probe._fold_residual_scale_into_k_isotropic = lambda: folds.append(True) or 1.0
    probe.f_masks = lambda: [_Arr(np.ones(_N, dtype=np.complex128))]
    probe.k_masks = lambda: [np.full(_N, 0.3)]
    probe._model_scale_array = lambda: np.full(_N, 2.0)
    probe.k_sol_b_sol = lambda: (0.30, 40.0)
    probe.f_calc = lambda: "f_calc"
    return probe, installed, folds


def _raw(accepted):
    return {
        "k_sol": 0.41,
        "b_sol": 62.0,
        "k_mask": np.full(_N, 0.2),
        "accepted": accepted,
        "stats": {
            "n_bins": 12,
            "k_ls": [0.40, 0.15, 0.02],
            "k_new": [0.42, 0.18, 0.01],
            "nll_fit_ref": 1.0,
            "nll_fit_new": 0.99 if accepted else 1.0,
            "nll_rest_ref": 1.1,
            "nll_rest_new": 1.095,
            "lbfgs_failures": [],
        },
    }


def test_an_accepted_fit_is_installed_and_the_caches_dropped(monkeypatch):
    probe, installed, folds = _probe(monkeypatch, _raw(True))
    detail = probe._fit_bulk_solvent_nll()
    assert len(installed) == 1
    (k_list,) = installed[0].values()
    assert len(k_list) == 1
    np.testing.assert_allclose(k_list[0], 0.2)
    assert (probe.k_sol, probe.b_sol) == (0.41, 62.0)
    for name in ("_f_model", "_last_maps", "_r_values", "_f_post", "_f_mode", "_last_gradients"):
        assert getattr(probe, name) is None, name
    assert folds == [True]
    assert "k_mask bins=12" in detail
    assert "0.40/0.15/0.02 -> 0.42/0.18/0.01" in detail
    assert "equiv k_sol/B_sol 0.300/40.0 -> 0.410/62.0" in detail
    assert "work -0.01000" in detail and "free -0.00500" in detail and "installed" in detail
    name, kw = probe.bridge.calls[0]
    assert name == engine.BULK_SOLVENT_OP_NAME
    np.testing.assert_allclose(kw["ss"], 1.0 / (4.0 * np.linspace(20.0, 2.0, _N) ** 2))
    assert np.all(kw["fit_mask"])


def test_a_declined_fit_leaves_the_least_squares_k_mask(monkeypatch):
    probe, installed, folds = _probe(monkeypatch, _raw(False))
    detail = probe._fit_bulk_solvent_nll()
    assert installed == [] and folds == []
    assert "kept LS k_mask" in detail


def test_the_env_switch_turns_it_off(monkeypatch):
    monkeypatch.setenv("PHRIDGE_BULK_SOLVENT_NLL", "0")
    probe, installed, _ = _probe(monkeypatch, _raw(True))
    assert probe._fit_bulk_solvent_nll() is None
    assert installed == [] and probe.bridge.calls == []
