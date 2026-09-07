"""Process-local live geometry-restraints sessions for the CCTBX worker.

Handles are opaque UUIDs returned by ``build_geometry_restraints`` and
consumed by ``geometry_restraints_energy_grad``. They are valid only in the
same CCTBX worker process (sticky single worker in production).

Includes LRU eviction and TTL expiry to prevent unbounded memory growth.
"""

from __future__ import annotations

import collections
import threading
import time
from typing import Any, Optional
import uuid

DEFAULT_MAX_SESSIONS: int = 256
DEFAULT_SESSION_TTL_SECONDS: float = 3600.0

_LOCK = threading.Lock()
_SESSIONS: collections.OrderedDict[str, dict[str, Any]] = collections.OrderedDict()
_MAX_SESSIONS: int = DEFAULT_MAX_SESSIONS
_SESSION_TTL_SECONDS: float = DEFAULT_SESSION_TTL_SECONDS


def set_session_policy(
    max_sessions: Optional[int] = None,
    ttl_seconds: Optional[float] = None,
) -> None:
    """Configure maximum number of in-memory sessions and TTL in seconds."""
    global _MAX_SESSIONS, _SESSION_TTL_SECONDS
    with _LOCK:
        if max_sessions is not None:
            _MAX_SESSIONS = max(1, int(max_sessions))
        if ttl_seconds is not None:
            _SESSION_TTL_SECONDS = max(0.0, float(ttl_seconds))
        _evict_expired_locked()
        _evict_lru_locked()


def _evict_expired_locked(now: Optional[float] = None) -> int:
    """Evict sessions that have exceeded TTL. Must be called while holding _LOCK."""
    if _SESSION_TTL_SECONDS <= 0:
        return 0
    if now is None:
        now = time.monotonic()
    expired = [
        k for k, v in _SESSIONS.items()
        if (now - v.get("last_accessed", now)) > _SESSION_TTL_SECONDS
    ]
    for k in expired:
        _SESSIONS.pop(k, None)
    return len(expired)


def _evict_lru_locked() -> int:
    """Evict least recently used sessions if exceeding _MAX_SESSIONS. Must be called while holding _LOCK."""
    evicted = 0
    while len(_SESSIONS) > _MAX_SESSIONS:
        _SESSIONS.popitem(last=False)
        evicted += 1
    return evicted


def stash_restraints_manager(grm: Any, *, n_sites: int) -> str:
    """Store a live ``geometry_restraints.manager``; return opaque handle."""
    handle = str(uuid.uuid4())
    now = time.monotonic()
    entry = {
        "grm": grm,
        "n_sites": int(n_sites),
        "created_at": now,
        "last_accessed": now,
    }
    with _LOCK:
        _evict_expired_locked(now)
        _SESSIONS[handle] = entry
        _evict_lru_locked()
    return handle


def get_session(handle: str) -> dict[str, Any]:
    """Retrieve session by handle, updating its LRU access timestamp."""
    now = time.monotonic()
    with _LOCK:
        if handle not in _SESSIONS:
            raise KeyError(
                f"unknown restraints_handle {handle!r} "
                "(process-local; rebuild on this CCTBX worker)"
            )
        entry = _SESSIONS[handle]
        if (
            _SESSION_TTL_SECONDS > 0
            and (now - entry.get("last_accessed", now)) > _SESSION_TTL_SECONDS
        ):
            _SESSIONS.pop(handle, None)
            raise KeyError(
                f"restraints_handle {handle!r} has expired "
                f"(TTL: {_SESSION_TTL_SECONDS}s)"
            )
        entry["last_accessed"] = now
        _SESSIONS.move_to_end(handle)
        return entry


def drop_session(handle: str) -> None:
    """Explicitly drop a session handle."""
    with _LOCK:
        _SESSIONS.pop(handle, None)


def clear_sessions() -> None:
    """Clear all active sessions and reset state."""
    with _LOCK:
        _SESSIONS.clear()


def clean_expired_sessions(ttl_seconds: Optional[float] = None) -> int:
    """Manually trigger eviction of expired sessions. Returns count of evicted sessions."""
    global _SESSION_TTL_SECONDS
    now = time.monotonic()
    with _LOCK:
        if ttl_seconds is not None:
            old_ttl = _SESSION_TTL_SECONDS
            try:
                _SESSION_TTL_SECONDS = max(0.0, float(ttl_seconds))
                return _evict_expired_locked(now)
            finally:
                _SESSION_TTL_SECONDS = old_ttl
        return _evict_expired_locked(now)
