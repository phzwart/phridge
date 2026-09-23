"""After mmtbx's f_model_all_scales.run the engine must not keep pre-scaling state.

That class copies the fmodel's ``__dict__``, runs ``mmtbx.f_model.manager`` methods on
the copy and copies the dict back, so the engine's own caches come back stale and, with
``apply_back_trace``, ``xray_structure`` is a new object the engine never adopted.
"""

from __future__ import annotations

import io

import pytest

engine = pytest.importorskip("phridge.client.intensity.engine")


def _probe(xrs):
    probe = engine.IntensityFModel.__new__(engine.IntensityFModel)
    probe._xray_structure = xrs
    probe.__dict__["xray_structure"] = xrs
    for name in ("_f_model", "_f_calc", "_last_maps", "_r_values", "_f_post", "_f_mode", "_last_gradients"):
        setattr(probe, name, object())
    calls = []
    probe.update_xray_structure = lambda x, update_f_calc=False: calls.append((x, update_f_calc))
    return probe, calls


def test_stale_caches_are_dropped_even_when_the_model_is_unchanged():
    xrs = object()
    probe, calls = _probe(xrs)
    probe._resync_after_mmtbx_scaling(log=io.StringIO())
    for name in ("_f_model", "_f_calc", "_last_maps", "_r_values", "_f_post", "_f_mode", "_last_gradients"):
        assert getattr(probe, name) is None, name
    assert calls == []


def test_a_back_traced_structure_is_adopted_and_f_calc_recomputed():
    old, new = object(), object()
    probe, calls = _probe(old)
    probe.__dict__["xray_structure"] = new  # what apply_back_trace leaves behind
    buf = io.StringIO()
    probe._resync_after_mmtbx_scaling(log=buf)
    assert calls == [(new, True)]
    assert probe._f_model is None
    assert "overall B moved into the atoms" in buf.getvalue()
