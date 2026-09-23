"""The target op must evaluate the same F_model the client builds everywhere else.

The client's ``f_model_scaled_with_k1()`` is ``k1 * k(h) * (F_calc + F_bulk)`` with a
per-reflection ``k(h)`` from mmtbx and a least-squares residual ``k1``. The nuisance fit,
the maps, the checkpoints and the journal all see that F. A target that applied a scalar
mean of ``k(h)`` instead scored σ_A and β against a differently shaped F_model.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_ml_i_omit_windows import _synthetic_structure  # noqa: E402

from phridge.contrib.intensity_ll.ops import ml_i_target_and_gradients  # noqa: E402
from phridge.contrib.intensity_ll.target import IntensityLogLikelihood  # noqa: E402
from phridge.sfcalc.ops import _observations  # noqa: E402


def _run(s, **kw):
    return ml_i_target_and_gradients(
        xray=s["xray"],
        table=s["table"],
        f_obs=s["f_obs"],
        params=s["params"],
        target={"name": "ml_i"},
        alpha=s["sigma_a"],
        beta=s["sigma_wilson"],
        epsilon=s["epsilon"],
        **kw,
    )


def _direct(s, f_model):
    obs = _observations(s["f_obs"], None, None, s["sigma_a"], s["sigma_wilson"], s["epsilon"], None)
    fm = torch.as_tensor(f_model, dtype=torch.complex128)
    return float(IntensityLogLikelihood().evaluate(fm, obs, compute_curvature=False).value)


def _value(out):
    return float(out["target"].meta.value)


def test_constant_k_model_matches_the_scalar_scale():
    s = _synthetic_structure(seed=3)
    n = len(s["hkl"])
    a = _run(s, scale_factor=1.7)
    b = _run(s, scale_factor=1.0, k_model=np.full(n, 1.7))
    assert _value(a) == pytest.approx(_value(b), rel=1e-12)
    np.testing.assert_allclose(
        a["gradients"].d_site_frac, b["gradients"].d_site_frac, rtol=1e-10, atol=1e-14
    )


def test_resolution_dependent_k_model_is_what_the_target_scores():
    s = _synthetic_structure(seed=4)
    k = 1.3 * np.exp(-2.0 * s["dstar2"])  # a k_isotropic with a real resolution slope
    out = _run(s, k_model=k)
    assert _value(out) == pytest.approx(_direct(s, k * s["f_calc"]), rel=1e-9)
    # and it is not the scalar mean the op used to apply
    assert abs(_value(out) - _direct(s, np.mean(k) * s["f_calc"])) > 1e-3


def test_residual_k1_is_the_clients_least_squares_scale():
    s = _synthetic_structure(seed=5)
    k = 0.4 * np.exp(-1.0 * s["dstar2"])
    fm = k * s["f_calc"]
    fo = np.sqrt(np.abs(np.asarray(s["f_obs"].data)))
    k1 = float(np.sum(fo * np.abs(fm)) / np.sum(np.abs(fm) ** 2))
    assert abs(k1 - 1.0) > 0.1  # the test means nothing if k1 is already 1
    out = _run(s, k_model=k, residual_k1=True)
    assert _value(out) == pytest.approx(_direct(s, k1 * fm), rel=1e-9)


def test_k_model_of_the_wrong_length_is_refused():
    s = _synthetic_structure(seed=6)
    with pytest.raises(ValueError, match="k_model"):
        _run(s, k_model=np.ones(3))
