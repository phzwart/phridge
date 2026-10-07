"""Backward-compatibility shim for phridge.tools.intensity_tool.

All functionality has been modularized into ``phridge.tools.intensity.*``:
  - ``phridge.tools.intensity.model.IntensityModel``
  - ``phridge.tools.intensity.cli``: ``from_files``, ``run_intensity_pipeline``, ``main``
  - ``phridge.d.map_coefficients.IntensityMapMixin``
  - ``phridge.tools.intensity.refine.IntensityRefineMixin``
  - ``phridge.tools.intensity.math_utils``: ``tv_denoise_1d``, ``isotonic_non_increasing``
"""

from __future__ import annotations

from phridge.tools.intensity import (
    IntensityMapMixin,
    IntensityModel,
    IntensityRefineMixin,
    apply_omit,
    from_files,
    golden_section_search_torch,
    isotonic_non_increasing,
    main,
    parse_omit_selection,
    run_intensity_pipeline,
    tv_denoise_1d,
)

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
