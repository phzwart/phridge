"""Dummy op: multiply an array by a scalar. Proves the Redis loop."""

from __future__ import annotations

from typing import Any


def scale_array(array: Any, scale: float = 1.0) -> Any:
    return array * float(scale)
