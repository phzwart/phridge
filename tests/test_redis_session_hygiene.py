"""Tests for Redis stream hygiene, consumer group offsets, and session LRU/TTL management."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from phridge.cctbx_worker.sessions import (
    clear_sessions,
    clean_expired_sessions,
    drop_session,
    get_session,
    set_session_policy,
    stash_restraints_manager,
)
from phridge.memory_redis import MemoryRedis
from phridge.models import JobEnvelope, WorkerRuntime
from phridge.protocol import Protocol
from phridge.redis_store import RedisStore, jobs_stream, worker_group


def test_protocol_stream_truncation():
    """Verify that protocol.enqueue respects stream_maxlen."""
    store = RedisStore.memory()
    protocol = Protocol(store, stream_maxlen=3)

    stream = jobs_stream(WorkerRuntime.torch)
    for i in range(7):
        envelope = JobEnvelope(
            job_id=f"job-{i}",
            op="scale_array",
            inputs={},
        )
        protocol.enqueue(envelope)

    # In-memory redis should hold at most maxlen=3 entries in the stream
    assert store.client.xlen(stream) == 3


def test_redis_store_delete_job():
    """Verify delete_job purges job envelope and ready notification."""
    store = RedisStore.memory()
    envelope = JobEnvelope(
        job_id="job-delete-test",
        op="scale_array",
        inputs={},
    )
    store.put_envelope(envelope)
    store.signal_ready(envelope.job_id)

    assert store.get_envelope("job-delete-test") is not None
    store.delete_job("job-delete-test")
    assert store.get_envelope("job-delete-test") is None


def test_ensure_consumer_group_offset_dollar():
    """Verify consumer group creation uses id='$' by default."""
    mock_client = MagicMock()
    store = RedisStore(mock_client)
    store.ensure_consumer_group(WorkerRuntime.torch)

    stream = jobs_stream(WorkerRuntime.torch)
    group = worker_group(WorkerRuntime.torch)
    mock_client.xgroup_create.assert_called_once_with(stream, group, id="$", mkstream=True)


def test_cctbx_sessions_lru_eviction():
    """Verify LRU session eviction when exceeding max_sessions."""
    clear_sessions()
    try:
        set_session_policy(max_sessions=3, ttl_seconds=3600.0)
        h1 = stash_restraints_manager("manager_1", n_sites=10)
        h2 = stash_restraints_manager("manager_2", n_sites=20)
        h3 = stash_restraints_manager("manager_3", n_sites=30)

        # Access h1 so h2 becomes the LRU
        assert get_session(h1)["grm"] == "manager_1"

        # Stash h4, which should evict h2
        h4 = stash_restraints_manager("manager_4", n_sites=40)

        # h2 should be gone
        with pytest.raises(KeyError, match="unknown restraints_handle"):
            get_session(h2)

        # h1, h3, h4 should still be present
        assert get_session(h1)["grm"] == "manager_1"
        assert get_session(h3)["grm"] == "manager_3"
        assert get_session(h4)["grm"] == "manager_4"
    finally:
        set_session_policy(max_sessions=256, ttl_seconds=3600.0)
        clear_sessions()


def test_cctbx_sessions_ttl_expiration():
    """Verify sessions expire after TTL elapses."""
    clear_sessions()
    try:
        set_session_policy(max_sessions=10, ttl_seconds=0.05)
        h1 = stash_restraints_manager("manager_exp", n_sites=5)
        assert get_session(h1)["grm"] == "manager_exp"

        time.sleep(0.06)
        with pytest.raises(KeyError, match="has expired"):
            get_session(h1)
    finally:
        set_session_policy(max_sessions=256, ttl_seconds=3600.0)
        clear_sessions()


def test_cctbx_sessions_manual_clean_and_drop():
    """Verify clean_expired_sessions and drop_session."""
    clear_sessions()
    try:
        set_session_policy(max_sessions=10, ttl_seconds=0.05)
        h1 = stash_restraints_manager("manager_a", n_sites=1)
        h2 = stash_restraints_manager("manager_b", n_sites=2)

        drop_session(h1)
        with pytest.raises(KeyError, match="unknown restraints_handle"):
            get_session(h1)

        time.sleep(0.06)
        evicted = clean_expired_sessions()
        assert evicted >= 1
        with pytest.raises(KeyError, match="unknown restraints_handle"):
            get_session(h2)
    finally:
        set_session_policy(max_sessions=256, ttl_seconds=3600.0)
        clear_sessions()
