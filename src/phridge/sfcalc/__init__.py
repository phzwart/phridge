"""Structure-factor core: FFT engine, targets, SF ops, and Phenix-facing client.

Torch boundary:
- ``phridge.sfcalc.client`` / ``phridge.sfcalc.packing`` — no torch (Phenix-safe).
- ``phridge.sfcalc.engine`` / ``targets`` / ``ops`` — torch; worker-side only.
"""

from __future__ import annotations

from typing import Any

__all__ = ["StructureFactorServer"]


def __getattr__(name: str) -> Any:
    if name == "StructureFactorServer":
        from phridge.sfcalc.client import StructureFactorServer

        return StructureFactorServer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
