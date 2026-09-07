"""Redis object store: envelopes and raw blobs with TTL and size cap."""

from __future__ import annotations

from typing import Any, Optional, Union

try:
    import redis
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "phridge requires the redis package (pip install redis). "
        "There is no automatic FakeRedis substitute when redis is missing. "
        "Use Bridge(memory=True) for an explicit in-process store "
        "(still requires the redis package)."
    ) from exc

from phridge.memory_redis import MemoryRedis
from phridge.models import JobEnvelope, WorkerRuntime

DEFAULT_TTL_SECONDS = 3600
DEFAULT_MAX_OBJECT_BYTES = 64 * 1024 * 1024

JOB_KEY = "phridge:job:{job_id}"
OBJ_KEY = "phridge:obj:{job_id}:{name}"
READY_KEY = "phridge:job:{job_id}:ready"

# Per-runtime job streams so torch and cctbx workers never steal each other's jobs.
# Legacy single stream was ``phridge:jobs``; torch now uses ``phridge:jobs:torch``.
JOBS_STREAM_TORCH = "phridge:jobs:torch"
JOBS_STREAM_CCTBX = "phridge:jobs:cctbx"
WORKER_GROUP_TORCH = "phridge-workers:torch"
WORKER_GROUP_CCTBX = "phridge-workers:cctbx"

# Backward-compatible aliases (torch runtime).
JOBS_STREAM = JOBS_STREAM_TORCH
WORKER_GROUP = WORKER_GROUP_TORCH


def normalize_runtime(runtime: Union[str, WorkerRuntime, None] = None) -> WorkerRuntime:
    if runtime is None:
        return WorkerRuntime.torch
    if isinstance(runtime, WorkerRuntime):
        return runtime
    return WorkerRuntime(str(runtime))


def jobs_stream(runtime: Union[str, WorkerRuntime, None] = None) -> str:
    rt = normalize_runtime(runtime)
    if rt == WorkerRuntime.cctbx:
        return JOBS_STREAM_CCTBX
    return JOBS_STREAM_TORCH


def worker_group(runtime: Union[str, WorkerRuntime, None] = None) -> str:
    rt = normalize_runtime(runtime)
    if rt == WorkerRuntime.cctbx:
        return WORKER_GROUP_CCTBX
    return WORKER_GROUP_TORCH


class ObjectTooLargeError(ValueError):
    """Raised when a blob exceeds max_object_bytes."""


class RedisStore:
    def __init__(
        self,
        client: Any,
        *,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        max_object_bytes: int = DEFAULT_MAX_OBJECT_BYTES,
    ) -> None:
        self.client = client
        self.ttl_seconds = ttl_seconds
        self.max_object_bytes = max_object_bytes

    @classmethod
    def from_url(
        cls,
        url: str,
        **kwargs,
    ) -> "RedisStore":
        return cls(redis.Redis.from_url(url, decode_responses=False), **kwargs)

    @classmethod
    def memory(cls, **kwargs) -> "RedisStore":
        """In-process store; no redis-server. Same key layout as Redis."""
        return cls(MemoryRedis(), **kwargs)

    def job_key(self, job_id: str) -> str:
        return JOB_KEY.format(job_id=job_id)

    def obj_key(self, job_id: str, name: str) -> str:
        return OBJ_KEY.format(job_id=job_id, name=name)

    def ready_key(self, job_id: str) -> str:
        return READY_KEY.format(job_id=job_id)

    def put_bytes(self, key: str, data: bytes) -> None:
        if len(data) > self.max_object_bytes:
            raise ObjectTooLargeError(
                f"{key} is {len(data)} bytes; max_object_bytes="
                f"{self.max_object_bytes}. Redis is a poor store for large "
                "maps; raise the cap or use a later blob backend."
            )
        self.client.set(key, data, ex=self.ttl_seconds)

    def get_bytes(self, key: str) -> bytes:
        data = self.client.get(key)
        if data is None:
            raise KeyError(key)
        if isinstance(data, str):
            return data.encode("utf-8")
        return data

    def put_envelope(self, envelope: JobEnvelope) -> None:
        key = self.job_key(envelope.job_id)
        payload = envelope.model_dump_json().encode("utf-8")
        self.client.set(key, payload, ex=self.ttl_seconds)

    def get_envelope(self, job_id: str) -> Optional[JobEnvelope]:
        data = self.client.get(self.job_key(job_id))
        if data is None:
            return None
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        return JobEnvelope.model_validate_json(data)

    def signal_ready(self, job_id: str) -> None:
        key = self.ready_key(job_id)
        self.client.rpush(key, b"1")
        self.client.expire(key, self.ttl_seconds)

    def delete_job(self, job_id: str) -> None:
        """Explicitly remove envelope and ready notification keys for a job."""
        keys = [self.job_key(job_id), self.ready_key(job_id)]
        try:
            self.client.delete(*keys)
        except Exception:
            pass

    def ensure_consumer_group(
        self,
        runtime: Union[str, WorkerRuntime, None] = None,
        id: str = "$",
    ) -> None:
        stream = jobs_stream(runtime)
        group = worker_group(runtime)
        try:
            self.client.xgroup_create(stream, group, id=id, mkstream=True)
        except redis.ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise
