"""Enqueue jobs on a Redis Stream and wait via BLPOP / envelope poll."""

from __future__ import annotations

import time
from typing import Optional

from phridge.models import JobEnvelope, JobStatus
from phridge.ops import get_op
from phridge.redis_store import RedisStore, jobs_stream


class JobTimeoutError(TimeoutError):
    """Timed out waiting for a job to finish."""


class Protocol:
    def __init__(self, store: RedisStore) -> None:
        self.store = store

    def enqueue(self, envelope: JobEnvelope) -> str:
        self.store.put_envelope(envelope)
        stream = jobs_stream(get_op(envelope.op).runtime)
        self.store.client.xadd(stream, {"job_id": envelope.job_id})
        return envelope.job_id

    def wait(self, job_id: str, timeout: float) -> JobEnvelope:
        """Block until status is done or error.

        Uses BLPOP on the per-job ready list. If the list is empty after
        the wait, falls back to reading the envelope (covers a missed
        push or a job that finished just before we blocked).
        """
        deadline = time.monotonic() + timeout
        ready_key = self.store.ready_key(job_id)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            block = max(1, min(int(remaining), 5))
            self.store.client.blpop(ready_key, timeout=block)
            envelope = self.store.get_envelope(job_id)
            if envelope is not None and envelope.status in (JobStatus.done, JobStatus.error):
                return envelope
        envelope = self.store.get_envelope(job_id)
        if envelope is not None and envelope.status in (JobStatus.done, JobStatus.error):
            return envelope
        raise JobTimeoutError(f"job {job_id} did not finish within {timeout}s")

    def status(self, job_id: str) -> Optional[JobStatus]:
        envelope = self.store.get_envelope(job_id)
        return None if envelope is None else envelope.status
