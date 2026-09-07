"""External / third-party ops via register_op and --preload."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from phridge.client import Bridge
from phridge.ops import (
    get_implementation,
    get_op,
    list_ops,
    preload_modules,
    register_op,
)
from phridge.redis_store import RedisStore
from phridge.worker.runner import process_envelope


def _store():
    fakeredis = pytest.importorskip("fakeredis")
    return RedisStore(fakeredis.FakeRedis())


def _install_plugin_module(name: str = "phridge_test_plugin") -> types.ModuleType:
    """Synthetic third-party module that registers an op at import time."""

    def add_one(array, shift=1.0):
        return array + float(shift)

    mod = types.ModuleType(name)

    def _register():
        register_op(
            "add_one",
            add_one,
            inputs={"array": "array", "shift": "json"},
            outputs={"array": "array"},
        )

    mod.register = _register  # type: ignore[attr-defined]
    mod.add_one = add_one  # type: ignore[attr-defined]
    # Import-time registration (preload style)
    _register()
    sys.modules[name] = mod
    return mod


def test_register_op_catalog_and_impl():
    def double(array, scale=2.0):
        return array * float(scale)

    register_op(
        "test_double",
        double,
        inputs={"array": "array", "scale": "json"},
        outputs={"array": "array"},
    )
    assert get_op("test_double").inputs["array"] == "array"
    assert get_implementation("test_double") is double
    assert get_implementation("test_double", runtime="torch") is double
    with pytest.raises(RuntimeError, match="cannot execute on cctbx worker"):
        get_implementation("test_double", runtime="cctbx")
    assert "test_double" in {s.name for s in list_ops()}


def test_preload_module_enables_bridge_call():
    _install_plugin_module("phridge_test_plugin_preload")
    # Simulate worker/client discovering the package
    loaded = preload_modules(["phridge_test_plugin_preload"])
    assert loaded == ["phridge_test_plugin_preload"]

    store = _store()
    bridge = Bridge(store=store, timeout=2)
    job_id = bridge.submit("add_one", array=np.arange(3, dtype=np.float64), shift=5.0)
    envelope = store.get_envelope(job_id)
    assert envelope is not None
    process_envelope(store, envelope, device="cpu")
    result = bridge.result(job_id, timeout=1)
    np.testing.assert_allclose(result, [5, 6, 7])


def test_memory_bridge_with_external_op():
    def negate(array):
        return -array

    register_op(
        "test_negate",
        negate,
        inputs={"array": "array"},
        outputs={"array": "array"},
    )
    bridge = Bridge(memory=True, timeout=2)
    out = bridge.call("test_negate", array=np.array([1.0, -2.0]))
    np.testing.assert_allclose(out, [-1.0, 2.0])


def test_register_op_requires_slots_for_string_name():
    with pytest.raises(TypeError, match="inputs"):
        register_op("missing_slots", lambda x: x)
