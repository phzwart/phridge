"""Process-local live geometry-restraints sessions for the CCTBX worker.

Handles are opaque UUIDs returned by ``build_geometry_restraints`` and
consumed by ``geometry_restraints_energy_grad``. They are valid only in the
same CCTBX worker process (sticky single worker in production).
"""

from __future__ import annotations

import threading
import uuid
from typing import Any

_LOCK = threading.Lock()
_SESSIONS: dict[str, dict[str, Any]] = {}


def stash_restraints_manager(grm: Any, *, n_sites: int) -> str:
    """Store a live ``geometry_restraints.manager``; return opaque handle."""
    handle = str(uuid.uuid4())
    with _LOCK:
        _SESSIONS[handle] = {"grm": grm, "n_sites": int(n_sites)}
    return handle


def get_session(handle: str) -> dict[str, Any]:
    with _LOCK:
        try:
            return _SESSIONS[handle]
        except KeyError as exc:
            raise KeyError(
                f"unknown restraints_handle {handle!r} "
                "(process-local; rebuild on this CCTBX worker)"
            ) from exc


def drop_session(handle: str) -> None:
    with _LOCK:
        _SESSIONS.pop(handle, None)


def clear_sessions() -> None:
    with _LOCK:
        _SESSIONS.clear()
