"""Phenix / torch-facing Bridge: convert, enqueue, wait, convert back."""

from __future__ import annotations

import uuid
from typing import Any, Optional

from phridge.client.convert import from_canonical, is_cctbx_object, to_canonical
from phridge.codec import decode_ref, encode_value
from phridge.models import JobEnvelope, JobStatus, WorkerRuntime, utcnow
from phridge.ops import get_op
from phridge.protocol import Protocol
from phridge.redis_store import (
    DEFAULT_MAX_OBJECT_BYTES,
    DEFAULT_TTL_SECONDS,
    RedisStore,
)


class JobFailed(RuntimeError):
    def __init__(self, job_id: str, error_type: str, message: str, traceback: Optional[str] = None):
        super().__init__(f"job {job_id} failed: {error_type}: {message}")
        self.job_id = job_id
        self.error_type = error_type
        self.message = message
        self.traceback = traceback


class Bridge:
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        *,
        store: Optional[RedisStore] = None,
        memory: bool = False,
        device: str = "cpu",
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        max_object_bytes: int = DEFAULT_MAX_OBJECT_BYTES,
        timeout: float = 3600.0,
    ) -> None:
        if memory and store is not None:
            raise ValueError("pass memory=True or store=, not both")
        if memory:
            self.store = RedisStore.memory(
                ttl_seconds=ttl_seconds,
                max_object_bytes=max_object_bytes,
            )
        else:
            self.store = store or RedisStore.from_url(
                redis_url,
                ttl_seconds=ttl_seconds,
                max_object_bytes=max_object_bytes,
            )
        self.protocol = Protocol(self.store)
        self.timeout = timeout
        self.memory = memory
        self.device = device
        self._prefer_cctbx: dict[str, bool] = {}

    def call(self, op: str, timeout: Optional[float] = None, **kwargs: Any) -> Any:
        job_id = self.submit(op, **kwargs)
        return self.result(job_id, timeout=timeout)

    def submit(self, op: str, **kwargs: Any) -> str:
        spec = get_op(op)
        job_id = str(uuid.uuid4())
        self._prefer_cctbx[job_id] = any(is_cctbx_object(v) for v in kwargs.values())
        inputs = {}
        for name, value in kwargs.items():
            inputs[name] = encode_value(self.store, job_id, name, to_canonical(value))
        envelope = JobEnvelope(
            job_id=job_id,
            op=spec.name,
            inputs=inputs,
            status=JobStatus.queued,
            created_at=utcnow(),
            updated_at=utcnow(),
        )
        self.protocol.enqueue(envelope)
        if self.memory:
            self._process_memory(envelope.job_id, WorkerRuntime(spec.runtime))
        return job_id

    def _process_memory(self, job_id: str, runtime: WorkerRuntime) -> None:
        loaded = self.store.get_envelope(job_id)
        if loaded is None:
            raise RuntimeError(f"memory store lost envelope {job_id}")
        if runtime == WorkerRuntime.cctbx:
            from phridge.cctbx_worker.runner import process_envelope

            process_envelope(self.store, loaded)
        else:
            from phridge.worker.runner import process_envelope

            process_envelope(self.store, loaded, device=self.device)

    def result(self, job_id: str, timeout: Optional[float] = None) -> Any:
        envelope = self.protocol.wait(job_id, timeout if timeout is not None else self.timeout)
        if envelope.status == JobStatus.error:
            err = envelope.error
            raise JobFailed(
                job_id,
                err.type if err else "Error",
                err.message if err else "unknown error",
                err.traceback if err else None,
            )
        prefer_cctbx = self._prefer_cctbx.pop(job_id, False)
        decoded = {
            name: from_canonical(decode_ref(self.store, ref), prefer_cctbx=prefer_cctbx)
            for name, ref in envelope.outputs.items()
        }
        if len(decoded) == 1:
            return next(iter(decoded.values()))
        return decoded
