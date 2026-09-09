"""Heartbeat / liveness prints for long mli_quad operations.

Default **on**, beat every **30 s**. Fast calls stay quiet until the first
interval elapses (no start/done spam on every target_eval).

Env:
  PHRIDGE_HEARTBEAT=0            disable
  PHRIDGE_HEARTBEAT_INTERVAL=30  seconds between beats (default 30)
  PHRIDGE_HEARTBEAT_EVERY=25     short progress line every N target evals
"""

from __future__ import annotations

import os
import sys
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator, Optional, TextIO


def heartbeat_enabled(default: bool = True) -> bool:
    raw = os.environ.get("PHRIDGE_HEARTBEAT")
    if raw is None or not str(raw).strip():
        return bool(default)
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def heartbeat_interval(default: float = 30.0) -> float:
    raw = os.environ.get("PHRIDGE_HEARTBEAT_INTERVAL")
    if raw is None or not str(raw).strip():
        return float(default)
    try:
        val = float(str(raw).strip())
        return val if val > 0 else float(default)
    except ValueError:
        return float(default)


def heartbeat_every_n_evals(default: int = 25) -> int:
    raw = os.environ.get("PHRIDGE_HEARTBEAT_EVERY")
    if raw is None or not str(raw).strip():
        return int(default)
    try:
        n = int(str(raw).strip())
        return n if n >= 1 else int(default)
    except ValueError:
        return int(default)


def _emit(msg: str, streams: Optional[list[TextIO]] = None) -> None:
    outs = streams or [sys.stdout]
    for out in outs:
        try:
            print(msg, file=out)
            if hasattr(out, "flush"):
                out.flush()
        except Exception:
            pass


@contextmanager
def mli_heartbeat(
    label: str,
    *,
    interval: Optional[float] = None,
    enabled: Optional[bool] = None,
    log: Any = None,
    announce: bool = False,
) -> Iterator[None]:
    """Print periodic ``still working`` lines while the body runs.

    First beat fires after ``interval`` seconds (default 30). With
    ``announce=False`` (default) there is no start/done line unless at least
    one beat fired — keeps LBFGS target evals quiet.
    """
    if enabled is None:
        enabled = heartbeat_enabled()
    if not enabled:
        yield
        return

    period = float(interval) if interval is not None else heartbeat_interval()
    stop = threading.Event()
    t0 = time.monotonic()
    beats = {"n": 0}
    streams: list[TextIO] = [sys.stdout]
    if log is not None and hasattr(log, "write") and log not in streams:
        streams.append(log)

    def _loop() -> None:
        while not stop.wait(period):
            beats["n"] += 1
            elapsed = time.monotonic() - t0
            _emit(
                f">>> [mli_quad heartbeat] still working: {label} "
                f"(elapsed {elapsed:.0f}s, beat {beats['n']})",
                streams,
            )

    thread = threading.Thread(
        target=_loop,
        name=f"mli-heartbeat-{label[:32]}",
        daemon=True,
    )
    thread.start()
    try:
        if announce:
            _emit(f">>> [mli_quad] start: {label}", streams)
        yield
    finally:
        stop.set()
        thread.join(timeout=1.0)
        elapsed = time.monotonic() - t0
        if announce or beats["n"] > 0:
            _emit(f">>> [mli_quad] done: {label} ({elapsed:.1f}s)", streams)


def maybe_progress_eval(
    eval_idx: int,
    *,
    target_work: float,
    elapsed_ms: float,
    compute_gradients: bool,
    log: Any = None,
) -> None:
    """Short progress line every N target evaluations (default 25)."""
    if not heartbeat_enabled():
        return
    every = heartbeat_every_n_evals()
    if eval_idx <= 0 or (eval_idx % every) != 0:
        return
    grad = "grad" if compute_gradients else "noll"
    extras = [log] if log is not None and hasattr(log, "write") else []
    _emit(
        f">>> [mli_quad progress] eval #{eval_idx}  target_work={target_work:.6f}  "
        f"last={elapsed_ms:.0f}ms  ({grad})",
        [sys.stdout] + extras,
    )
