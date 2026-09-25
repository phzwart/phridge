"""Process-local live structure-factor engines on the torch worker.

Handles are opaque UUIDs from ``sf_bind`` / ``nufft_sf_bind``. They are valid
only in the same worker process (sticky consumer). LRU + TTL cap memory.
"""

from __future__ import annotations

import collections
import threading
import time
import uuid
from typing import Any, Optional

DEFAULT_MAX_SESSIONS: int = 32
DEFAULT_SESSION_TTL_SECONDS: float = 3600.0

_LOCK = threading.Lock()
_SESSIONS: collections.OrderedDict[str, dict[str, Any]] = collections.OrderedDict()
_MAX_SESSIONS: int = DEFAULT_MAX_SESSIONS
_SESSION_TTL_SECONDS: float = DEFAULT_SESSION_TTL_SECONDS


def set_session_policy(
    max_sessions: Optional[int] = None,
    ttl_seconds: Optional[float] = None,
) -> None:
    global _MAX_SESSIONS, _SESSION_TTL_SECONDS
    with _LOCK:
        if max_sessions is not None:
            _MAX_SESSIONS = max(1, int(max_sessions))
        if ttl_seconds is not None:
            _SESSION_TTL_SECONDS = max(0.0, float(ttl_seconds))
        _evict_expired_locked()
        _evict_lru_locked()


def _evict_expired_locked(now: Optional[float] = None) -> int:
    if _SESSION_TTL_SECONDS <= 0:
        return 0
    if now is None:
        now = time.monotonic()
    expired = [
        k
        for k, v in _SESSIONS.items()
        if (now - v.get("last_accessed", now)) > _SESSION_TTL_SECONDS
    ]
    for k in expired:
        _SESSIONS.pop(k, None)
    return len(expired)


def _evict_lru_locked() -> int:
    evicted = 0
    while len(_SESSIONS) > _MAX_SESSIONS:
        _SESSIONS.popitem(last=False)
        evicted += 1
    return evicted


def stash_engine(engine: Any, hkl: Any) -> str:
    """Store a live SF engine; return an opaque handle."""
    handle = str(uuid.uuid4())
    now = time.monotonic()
    entry = {
        "engine": engine,
        "hkl": hkl,
        "created_at": now,
        "last_accessed": now,
    }
    with _LOCK:
        _evict_expired_locked(now)
        _SESSIONS[handle] = entry
        _evict_lru_locked()
    return handle


def get_engine_session(handle: str) -> dict[str, Any]:
    now = time.monotonic()
    with _LOCK:
        if handle not in _SESSIONS:
            raise KeyError(
                f"unknown sf_handle {handle!r} (process-local; bind again on this worker)"
            )
        entry = _SESSIONS[handle]
        if (
            _SESSION_TTL_SECONDS > 0
            and (now - entry.get("last_accessed", now)) > _SESSION_TTL_SECONDS
        ):
            _SESSIONS.pop(handle, None)
            raise KeyError(f"sf_handle {handle!r} has expired (TTL: {_SESSION_TTL_SECONDS}s)")
        entry["last_accessed"] = now
        _SESSIONS.move_to_end(handle)
        return entry


def drop_engine(handle: str) -> None:
    with _LOCK:
        _SESSIONS.pop(handle, None)


def clear_engines() -> None:
    with _LOCK:
        _SESSIONS.clear()
