"""Write an MTZ. cctbx stays in the shim."""

from __future__ import annotations

from typing import Any

from phridge.shims.cctbx.miller import write_mtz_file, write_observation_mtz

__all__ = ["write_mtz", "write_observation_mtz", "write_mtz_file"]


def write_mtz(path: str, packed: Any) -> None:
    """Write a packed reflection file."""
    write_mtz_file(path, packed)
