"""Backward-compatibility shim. The live package does not import legacy."""

from __future__ import annotations

from phridge.d.map_coefficients import IntensityMapMixin
from phridge.legacy import intensity_cli as _cli
from phridge.legacy.intensity_cli import apply_omit, from_files, main, parse_omit_selection, run_intensity_pipeline
from phridge.legacy.intensity_refine import IntensityRefineMixin
from phridge.tools.intensity.math_utils import golden_section_search_torch, isotonic_non_increasing, tv_denoise_1d
from phridge.tools.intensity.model import IntensityModel as _LiveModel


class IntensityModel(IntensityRefineMixin, _LiveModel):
    """Live model plus the legacy coordinate and B-factor steps."""


_cli.IntensityModel = IntensityModel

__all__ = [
    "IntensityMapMixin",
    "IntensityModel",
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

if __name__ == "__main__":
    raise SystemExit(main())
