"""Intensity-based refinement, likelihood modeling, and map synthesis."""

from __future__ import annotations

from typing import Any

__all__ = [
    "IntensityDataError",
    "IntensityElectronDensityMap",
    "IntensityFModel",
    "IntensityFModelInfo",
    "IntensityFmodel",
    "IntensityGradients",
    "IntensityLikelihoodEngine",
    "IntensityMapMixin",
    "IntensityModel",
    "IntensityRefineMixin",
    "IntensityTargetFunctor",
    "IntensityTargetResult",
    "IntensityTwinningError",
    "apply_omit",
    "disable_intensity_in_phenix",
    "enable_intensity_in_phenix",
    "from_files",
    "golden_section_search_torch",
    "is_intensity_enabled",
    "isotonic_non_increasing",
    "main",
    "parse_omit_selection",
    "register_mli_targets",
    "run_intensity_pipeline",
    "tv_denoise_1d",
]


_ENGINE = {
    "IntensityDataError",
    "IntensityElectronDensityMap",
    "IntensityFModel",
    "IntensityFModelInfo",
    "IntensityFmodel",
    "IntensityGradients",
    "IntensityLikelihoodEngine",
    "IntensityTargetFunctor",
    "IntensityTargetResult",
    "IntensityTwinningError",
    "register_mli_targets",
}

_PHENIX = {
    "disable_intensity_in_phenix",
    "enable_intensity_in_phenix",
    "is_intensity_enabled",
}


def __getattr__(name: str) -> Any:
    if name in _ENGINE:
        from phridge.legacy import intensity_engine as engine

        return getattr(engine, name)
    if name in _PHENIX:
        from phridge.legacy import intensity_phenix_hook as hook

        return getattr(hook, name)
    if name in ("apply_omit", "from_files", "main", "parse_omit_selection", "run_intensity_pipeline"):
        from phridge.legacy import intensity_cli as cli

        return getattr(cli, name)
    if name == "IntensityModel":
        from phridge.tools.intensity.model import IntensityModel

        return IntensityModel
    if name == "IntensityMapMixin":
        from phridge.d.map_coefficients import IntensityMapMixin

        return IntensityMapMixin
    if name == "IntensityRefineMixin":
        from phridge.legacy.intensity_refine import IntensityRefineMixin

        return IntensityRefineMixin
    if name in ("_fom_rice_woolfson", "golden_section_search_torch", "isotonic_non_increasing", "tv_denoise_1d"):
        from phridge.tools.intensity import math_utils

        return getattr(math_utils, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
