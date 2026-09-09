"""CCTBX stream consumer: XREADGROUP, run cctbx ops, signal ready. No torch."""

from __future__ import annotations

import argparse
import inspect
import os
import socket
import traceback
from typing import Any, Optional

import redis

from phridge.client.convert import from_canonical, to_canonical
from phridge.codec import decode_ref, encode_value
from phridge.models import JobEnvelope, JobError, JobStatus, WorkerRuntime, utcnow
from phridge.ops import _parse_preload_arg, bootstrap_plugins, get_implementation, get_op
from phridge.redis_store import RedisStore, jobs_stream, worker_group


class UnknownOpError(KeyError):
    pass


def process_envelope(store: RedisStore, envelope: JobEnvelope) -> JobEnvelope:
    envelope.status = JobStatus.running
    envelope.updated_at = utcnow()
    store.put_envelope(envelope)
    try:
        try:
            spec = get_op(envelope.op)
        except KeyError as exc:
            raise UnknownOpError(str(exc)) from exc
        if WorkerRuntime(spec.runtime) != WorkerRuntime.cctbx:
            raise UnknownOpError(
                f"{envelope.op} is runtime={spec.runtime.value}; this is the cctbx worker"
            )
        impl = get_implementation(envelope.op, runtime=WorkerRuntime.cctbx)
        if impl is None:
            raise UnknownOpError(f"no worker implementation for {envelope.op}")
        native: dict[str, Any] = {}
        for name, ref in envelope.inputs.items():
            native[name] = from_canonical(decode_ref(store, ref), prefer_cctbx=True)
        kwargs = _bind_kwargs(impl, native)
        result = impl(**kwargs)
        if not isinstance(result, dict):
            out_name = next(iter(spec.outputs))
            result = {out_name: result}
        outputs = {}
        for name, value in result.items():
            if name not in spec.outputs:
                continue
            outputs[name] = encode_value(
                store, envelope.job_id, f"out:{name}", to_canonical(value)
            )
        envelope.outputs = outputs
        envelope.status = JobStatus.done
        envelope.error = None
    except Exception as exc:
        envelope.status = JobStatus.error
        envelope.error = JobError(
            type=type(exc).__name__,
            message=str(exc),
            traceback=traceback.format_exc(),
        )
    envelope.updated_at = utcnow()
    store.put_envelope(envelope)
    store.signal_ready(envelope.job_id)
    return envelope


def _bind_kwargs(impl: Any, native: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(impl)
    kwargs = {}
    for name, param in signature.parameters.items():
        if name in native:
            kwargs[name] = native[name]
        elif param.default is inspect.Parameter.empty:
            raise TypeError(f"op missing required input {name!r}")
    return kwargs


def consume_one(
    store: RedisStore,
    *,
    consumer: str = "test",
    block_ms: int = 1000,
) -> Optional[JobEnvelope]:
    runtime = WorkerRuntime.cctbx
    stream = jobs_stream(runtime)
    group = worker_group(runtime)
    store.ensure_consumer_group(runtime)
    messages = store.client.xreadgroup(
        group,
        consumer,
        {stream: ">"},
        count=1,
        block=block_ms,
    )
    if not messages:
        return None
    result = None
    for _stream, entries in messages:
        for message_id, fields in entries:
            job_id = _field(fields, "job_id")
            envelope = store.get_envelope(job_id)
            if envelope is not None:
                result = process_envelope(store, envelope)
            store.client.xack(stream, group, message_id)
    return result


def consume_forever(
    store: RedisStore,
    *,
    consumer: Optional[str] = None,
    block_ms: int = 5000,
) -> None:
    runtime = WorkerRuntime.cctbx
    stream = jobs_stream(runtime)
    group = worker_group(runtime)
    store.ensure_consumer_group(runtime)
    consumer = consumer or socket.gethostname()
    while True:
        try:
            messages = store.client.xreadgroup(
                group,
                consumer,
                {stream: ">"},
                count=1,
                block=block_ms,
            )
        except (redis.exceptions.TimeoutError, TimeoutError):
            continue
        if not messages:
            continue
        for _stream, entries in messages:
            for message_id, fields in entries:
                job_id = _field(fields, "job_id")
                envelope = store.get_envelope(job_id)
                if envelope is None:
                    store.client.xack(stream, group, message_id)
                    continue
                process_envelope(store, envelope)
                store.client.xack(stream, group, message_id)


def _field(fields: dict, key: str) -> str:
    raw = fields.get(key) or fields.get(key.encode("utf-8"))
    if raw is None:
        raise KeyError(key)
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    return str(raw)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="phridge CCTBX worker (no torch)")
    parser.add_argument(
        "--redis-url",
        default=os.environ.get("PHRIDGE_REDIS_URL", "redis://localhost:6379/0"),
    )
    parser.add_argument("--consumer", default=os.environ.get("PHRIDGE_CONSUMER"))
    parser.add_argument(
        "--max-object-bytes",
        type=int,
        default=int(os.environ.get("PHRIDGE_MAX_OBJECT_BYTES", 64 * 1024 * 1024)),
    )
    parser.add_argument(
        "--preload",
        action="append",
        default=[],
        help=(
            "Import module(s) that call phridge.ops.register_op / "
            "register_target before serving (repeatable; comma-separated ok). "
            "Also reads PHRIDGE_PRELOAD."
        ),
    )
    parser.add_argument(
        "--no-entry-points",
        action="store_true",
        help="Skip loading phridge.ops / phridge.targets entry-point groups",
    )
    args = parser.parse_args(argv)
    try:
        import cctbx  # noqa: F401
        import mmtbx  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "cctbx worker requires cctbx and mmtbx "
            "(conda: cctbx-base + chem_data). "
            f"Import failed: {exc}"
        ) from exc

    env_preload = os.environ.get("PHRIDGE_PRELOAD", "")
    bootstrap_plugins(
        preload=_parse_preload_arg([*args.preload, env_preload]),
        load_eps=not args.no_entry_points,
    )
    store = RedisStore.from_url(args.redis_url, max_object_bytes=args.max_object_bytes)
    consume_forever(store, consumer=args.consumer)
