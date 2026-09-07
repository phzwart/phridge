"""Backward-compatibility shim for phridge.client.intensity_tool.

All functionality has been modularized into ``phridge.client.intensity.*``:
  - ``phridge.client.intensity.model.IntensityModel``
  - ``phridge.client.intensity.cli``: ``from_files``, ``run_intensity_pipeline``, ``main``
  - ``phridge.client.intensity.maps.IntensityMapMixin``
  - ``phridge.client.intensity.refine.IntensityRefineMixin``
  - ``phridge.client.intensity.math_utils``: ``tv_denoise_1d``, ``isotonic_non_increasing``
"""

from __future__ import annotations

from phridge.client.intensity import (
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
