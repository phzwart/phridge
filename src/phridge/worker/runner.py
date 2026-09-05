"""Stream consumer: XREADGROUP, run registered ops, signal ready."""

from __future__ import annotations

import argparse
import inspect
import os
import socket
import traceback
from typing import Any, Optional

from phridge.codec import decode_ref, encode_value
from phridge.models import JobEnvelope, JobError, JobStatus, utcnow
from phridge.ops import _parse_preload_arg, bootstrap_plugins, get_implementation, get_op
from phridge.redis_store import JOBS_STREAM, WORKER_GROUP, RedisStore
from phridge.worker.convert import from_torch, resolve_device, to_torch


class UnknownOpError(KeyError):
    pass


def process_envelope(
    store: RedisStore,
    envelope: JobEnvelope,
    *,
    device: str = "cpu",
) -> JobEnvelope:
    envelope.status = JobStatus.running
    envelope.updated_at = utcnow()
    store.put_envelope(envelope)
    try:
        try:
            spec = get_op(envelope.op)
        except KeyError as exc:
            raise UnknownOpError(str(exc)) from exc
        impl = get_implementation(envelope.op)
        if impl is None:
            raise UnknownOpError(f"no worker implementation for {envelope.op}")
        native: dict[str, Any] = {}
        compute_dtype = _compute_dtype(impl)
        for name, ref in envelope.inputs.items():
            native[name] = to_torch(decode_ref(store, ref), device, compute_dtype)
        kwargs = _bind_kwargs(impl, native)
        result = impl(**kwargs)
        if not isinstance(result, dict):
            out_name = next(iter(spec.outputs))
            result = {out_name: result}
        outputs = {}
        for name, value in result.items():
            outputs[name] = encode_value(store, envelope.job_id, f"out:{name}", from_torch(value))
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


def _compute_dtype(impl: Any) -> Any:
    """Ops may declare ``impl.compute_dtype = "float64"`` to keep full precision."""
    name = getattr(impl, "compute_dtype", None)
    if name is None:
        return None
    try:
        import torch
    except ImportError:
        return None
    return {"float64": torch.float64, "float32": torch.float32}[name]


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
    device: str = "cpu",
    consumer: str = "test",
    block_ms: int = 1000,
) -> Optional[JobEnvelope]:
    store.ensure_consumer_group()
    _set_op_device(device)
    messages = store.client.xreadgroup(
        WORKER_GROUP,
        consumer,
        {JOBS_STREAM: ">"},
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
                result = process_envelope(store, envelope, device=device)
            store.client.xack(JOBS_STREAM, WORKER_GROUP, message_id)
    return result


def consume_forever(
    store: RedisStore,
    *,
    device: str = "auto",
    consumer: Optional[str] = None,
    block_ms: int = 5000,
) -> None:
    store.ensure_consumer_group()
    device = resolve_device(device)
    _set_op_device(device)
    consumer = consumer or socket.gethostname()
    while True:
        messages = store.client.xreadgroup(
            WORKER_GROUP,
            consumer,
            {JOBS_STREAM: ">"},
            count=1,
            block=block_ms,
        )
        if not messages:
            continue
        for _stream, entries in messages:
            for message_id, fields in entries:
                job_id = _field(fields, "job_id")
                envelope = store.get_envelope(job_id)
                if envelope is None:
                    store.client.xack(JOBS_STREAM, WORKER_GROUP, message_id)
                    continue
                process_envelope(store, envelope, device=device)
                store.client.xack(JOBS_STREAM, WORKER_GROUP, message_id)


def _set_op_device(device: str) -> None:
    from phridge.worker.ops import xtal_ops

    xtal_ops.set_device(device)


def _field(fields: dict, key: str) -> str:
    raw = fields.get(key) or fields.get(key.encode("utf-8"))
    if raw is None:
        raise KeyError(key)
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    return str(raw)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="phridge PyTorch worker")
    parser.add_argument(
        "--redis-url",
        default=os.environ.get("PHRIDGE_REDIS_URL", "redis://localhost:6379/0"),
    )
    parser.add_argument("--device", default=os.environ.get("PHRIDGE_DEVICE", "auto"))
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
            "Import module(s) that call phridge.ops.register_op before serving "
            "(repeatable; comma-separated ok). Also reads PHRIDGE_PRELOAD."
        ),
    )
    parser.add_argument(
        "--no-entry-points",
        action="store_true",
        help="Skip loading the phridge.ops setuptools entry-point group",
    )
    args = parser.parse_args(argv)
    env_preload = os.environ.get("PHRIDGE_PRELOAD", "")
    bootstrap_plugins(
        preload=_parse_preload_arg([*args.preload, env_preload]),
        load_eps=not args.no_entry_points,
    )
    store = RedisStore.from_url(args.redis_url, max_object_bytes=args.max_object_bytes)
    consume_forever(store, device=args.device, consumer=args.consumer)
