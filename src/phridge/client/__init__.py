from __future__ import annotations

from typing import Any

from phridge.client.api import Bridge, JobFailed
from phridge.client.geometry import RemoteGeometry, RemoteRestraintBuilder

__all__ = [
    "Bridge",
    "IntensityElectronDensityMap",
    "IntensityFModel",
    "IntensityFModelInfo",
    "IntensityFmodel",
    "IntensityGradients",
    "IntensityLikelihoodEngine",
    "IntensityTargetFunctor",
    "IntensityTargetResult",
    "JobFailed",
    "RemoteGeometry",
    "RemoteRestraintBuilder",
    "StructureFactorServer",
    "disable_intensity_in_phenix",
    "enable_intensity_in_phenix",
    "is_intensity_enabled",
]


def __getattr__(name: str) -> Any:
    if name == "StructureFactorServer":
        from phridge.sfcalc.client import StructureFactorServer

        return StructureFactorServer
    if name in (
        "enable_intensity_in_phenix",
        "disable_intensity_in_phenix",
        "is_intensity_enabled",
    ):
        from phridge.client.intensity.phenix_hook import (
            disable_intensity_in_phenix,
            enable_intensity_in_phenix,
            is_intensity_enabled,
        )

        return {
            "enable_intensity_in_phenix": enable_intensity_in_phenix,
            "disable_intensity_in_phenix": disable_intensity_in_phenix,
            "is_intensity_enabled": is_intensity_enabled,
        }[name]
    if name in (
        "IntensityFModel",
        "IntensityFModelInfo",
        "IntensityFmodel",
        "IntensityLikelihoodEngine",
        "IntensityElectronDensityMap",
        "IntensityGradients",
        "IntensityTargetFunctor",
        "IntensityTargetResult",
        "IntensityTwinningError",
    ):
        from phridge.client.intensity.engine import (
            IntensityElectronDensityMap,
            IntensityFModel,
            IntensityFModelInfo,
            IntensityFmodel,
            IntensityGradients,
            IntensityLikelihoodEngine,
            IntensityTargetFunctor,
            IntensityTargetResult,
            IntensityTwinningError,
        )

        return {
            "IntensityFModel": IntensityFModel,
            "IntensityFModelInfo": IntensityFModelInfo,
            "IntensityFmodel": IntensityFmodel,
            "IntensityLikelihoodEngine": IntensityLikelihoodEngine,
            "IntensityElectronDensityMap": IntensityElectronDensityMap,
            "IntensityGradients": IntensityGradients,
            "IntensityTargetFunctor": IntensityTargetFunctor,
            "IntensityTargetResult": IntensityTargetResult,
            "IntensityTwinningError": IntensityTwinningError,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
