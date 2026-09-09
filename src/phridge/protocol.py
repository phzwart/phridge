"""Enqueue jobs on a Redis Stream and wait via BLPOP / envelope poll."""

from __future__ import annotations

import os
import sys
import time
from typing import Optional

import redis

from phridge.models import JobEnvelope, JobStatus
from phridge.ops import get_op
from phridge.redis_store import RedisStore, jobs_stream


class JobTimeoutError(TimeoutError):
    """Timed out waiting for a job to finish."""


class Protocol:
    def __init__(self, store: RedisStore, stream_maxlen: int = 10000) -> None:
        self.store = store
        self.stream_maxlen = stream_maxlen

    def enqueue(self, envelope: JobEnvelope) -> str:
        self.store.put_envelope(envelope)
        stream = jobs_stream(get_op(envelope.op).runtime)
        self.store.client.xadd(
            stream,
            {"job_id": envelope.job_id},
            maxlen=self.stream_maxlen,
            approximate=True,
        )
        return envelope.job_id

    def wait(self, job_id: str, timeout: float) -> JobEnvelope:
        """Block until status is done or error.

        Uses BLPOP on the per-job ready list. If the list is empty after
        the wait, falls back to reading the envelope (covers a missed
        push or a job that finished just before we blocked).

        Emits a heartbeat every ``PHRIDGE_HEARTBEAT_INTERVAL`` seconds
        (default 30) so long Redis waits are visible; socket timeouts on
        the Redis client prevent indefinite hangs on dead connections.
        """
        deadline = time.monotonic() + timeout
        ready_key = self.store.ready_key(job_id)
        t0 = time.monotonic()
        last_beat = t0
        try:
            beat_every = float(os.environ.get("PHRIDGE_HEARTBEAT_INTERVAL", "30") or "30")
        except ValueError:
            beat_every = 30.0
        hb_env = os.environ.get("PHRIDGE_HEARTBEAT", "1").strip().lower()
        do_hb = hb_env in ("", "1", "true", "yes", "on")

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            block = max(1, min(int(remaining), 5))
            try:
                self.store.client.blpop(ready_key, timeout=block)
            except (redis.exceptions.TimeoutError, TimeoutError):
                pass
            except (redis.exceptions.ConnectionError, OSError) as exc:
                # Broken socket: fail fast instead of spinning forever.
                raise JobTimeoutError(
                    f"job {job_id}: Redis connection error while waiting ({exc})"
                ) from exc
            envelope = self.store.get_envelope(job_id)
            if envelope is not None and envelope.status in (JobStatus.done, JobStatus.error):
                return envelope

            now = time.monotonic()
            if do_hb and (now - last_beat) >= beat_every:
                last_beat = now
                try:
                    status = envelope.status.value if envelope is not None else "missing"
                except Exception:
                    status = "?"
                msg = (
                    f">>> [mli_quad heartbeat] waiting on Redis job {job_id[:8]}… "
                    f"status={status} elapsed={now - t0:.0f}s"
                )
                try:
                    print(msg, file=sys.stdout)
                    sys.stdout.flush()
                except Exception:
                    pass

        envelope = self.store.get_envelope(job_id)
        if envelope is not None and envelope.status in (JobStatus.done, JobStatus.error):
            return envelope
        raise JobTimeoutError(f"job {job_id} did not finish within {timeout}s")

    def status(self, job_id: str) -> Optional[JobStatus]:
        envelope = self.store.get_envelope(job_id)
        return None if envelope is None else envelope.status
