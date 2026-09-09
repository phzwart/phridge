"""Intensity-based refinement, likelihood modeling, and map synthesis."""

from __future__ import annotations

from typing import Any

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
    IntensityDataError,
    register_mli_targets,
)
from phridge.client.intensity.phenix_hook import (
    disable_intensity_in_phenix,
    enable_intensity_in_phenix,
    is_intensity_enabled,
)

__all__ = [
    "IntensityElectronDensityMap",
    "IntensityFModel",
    "IntensityFModelInfo",
    "IntensityFmodel",
    "IntensityGradients",
    "IntensityLikelihoodEngine",
    "IntensityModel",
    "IntensityTargetFunctor",
    "IntensityTargetResult",
    "IntensityTwinningError",
    "IntensityDataError",
    "register_mli_targets",
    "IntensityMapMixin",
    "IntensityRefineMixin",
    "disable_intensity_in_phenix",
    "enable_intensity_in_phenix",
    "is_intensity_enabled",
    "IntensityMapMixin",
    "IntensityRefineMixin",
    "apply_omit",
    "from_files",
    "golden_section_search_torch",
    "isotonic_non_increasing",
    "main",
    "parse_omit_selection",
    "run_intensity_pipeline",
    "tv_denoise_1d",
]


def __getattr__(name: str) -> Any:
    if name in ("apply_omit", "from_files", "main", "parse_omit_selection", "run_intensity_pipeline"):
        from phridge.client.intensity import cli

        return getattr(cli, name)
    if name == "IntensityModel":
        from phridge.client.intensity.model import IntensityModel

        return IntensityModel
    if name == "IntensityMapMixin":
        from phridge.client.intensity.maps import IntensityMapMixin

        return IntensityMapMixin
    if name == "IntensityRefineMixin":
        from phridge.client.intensity.refine import IntensityRefineMixin

        return IntensityRefineMixin
    if name in ("_fom_rice_woolfson", "golden_section_search_torch", "isotonic_non_increasing", "tv_denoise_1d"):
        from phridge.client.intensity import math_utils

        return getattr(math_utils, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
