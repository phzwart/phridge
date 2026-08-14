"""Worker implementations keyed by op name from phridge.ops."""

from __future__ import annotations

from typing import Any, Callable

from phridge.worker.ops.scale_array import scale_array

IMPLEMENTATIONS: dict[str, Callable[..., Any]] = {
    "scale_array": scale_array,
}
