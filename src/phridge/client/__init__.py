from __future__ import annotations

from typing import Any

from phridge.client.api import Bridge, JobFailed
from phridge.client.geometry import RemoteGeometry, RemoteRestraintBuilder

__all__ = [
    "Bridge",
    "JobFailed",
    "RemoteGeometry",
    "RemoteRestraintBuilder",
    "StructureFactorServer",
]


def __getattr__(name: str) -> Any:
    if name == "StructureFactorServer":
        from phridge.sfcalc.client import StructureFactorServer

        return StructureFactorServer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
