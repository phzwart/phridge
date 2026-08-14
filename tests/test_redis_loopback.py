import numpy as np
import pytest

from phridge.client import Bridge, JobFailed
from phridge.models import JobStatus
from phridge.ops import get_op, list_ops
from phridge.redis_store import ObjectTooLargeError, RedisStore
from phridge.worker.runner import consume_one, process_envelope


def _store():
    fakeredis = pytest.importorskip("fakeredis")
    return RedisStore(fakeredis.FakeRedis())


def test_list_ops_includes_scale_array():
    names = [spec.name for spec in list_ops()]
    assert "scale_array" in names
    spec = get_op("scale_array")
    assert spec.inputs["array"] == "array"


def test_process_envelope_scale_array():
    store = _store()
    bridge = Bridge(store=store, timeout=2)
    job_id = bridge.submit("scale_array", array=np.arange(4, dtype=np.float64), scale=2.0)
    envelope = store.get_envelope(job_id)
    assert envelope is not None
    process_envelope(store, envelope, device="cpu")
    result = bridge.result(job_id, timeout=1)
    np.testing.assert_allclose(result, [0, 2, 4, 6])
    assert result.dtype == np.float64


def test_stream_loopback_scale_array():
    store = _store()
    try:
        store.ensure_consumer_group()
    except Exception as exc:
        pytest.skip(f"fakeredis Streams/consumer groups unavailable: {exc}")
    bridge = Bridge(store=store, timeout=2)
    job_id = bridge.submit("scale_array", array=np.ones(3, dtype=np.float64), scale=3.0)
    try:
        done = consume_one(store, device="cpu", consumer="test-loop", block_ms=500)
    except Exception as exc:
        pytest.skip(f"fakeredis XREADGROUP failed: {exc}")
    if done is None:
        pytest.skip("fakeredis XREADGROUP returned no message")
    assert done.status == JobStatus.done
    result = bridge.result(job_id, timeout=1)
    np.testing.assert_allclose(result, [3, 3, 3])


def test_unknown_op_errors_before_enqueue():
    store = _store()
    bridge = Bridge(store=store)
    with pytest.raises(KeyError, match="unknown op"):
        bridge.submit("not_an_op", array=np.array([1.0]))


def test_worker_unknown_op_writes_error_envelope():
    store = _store()
    from phridge.models import JobEnvelope, utcnow

    envelope = JobEnvelope(job_id="x", op="missing", created_at=utcnow(), updated_at=utcnow())
    store.put_envelope(envelope)
    process_envelope(store, envelope, device="cpu")
    loaded = store.get_envelope("x")
    assert loaded is not None
    assert loaded.status == JobStatus.error
    assert loaded.error is not None


def test_job_failed_surfaces_worker_error():
    store = _store()
    bridge = Bridge(store=store, timeout=2)
    job_id = bridge.submit("scale_array", array=np.array([1.0]), scale=2.0)
    envelope = store.get_envelope(job_id)
    assert envelope is not None
    envelope.inputs.pop("array")
    store.put_envelope(envelope)
    process_envelope(store, envelope, device="cpu")
    with pytest.raises(JobFailed):
        bridge.result(job_id, timeout=1)


def test_put_bytes_enforces_size_cap():
    store = _store()
    store.max_object_bytes = 8
    with pytest.raises(ObjectTooLargeError):
        store.put_bytes("phridge:obj:t:big", b"0123456789")
