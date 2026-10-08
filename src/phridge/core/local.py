"""Marker bound by the amplitude example. This is not a French–Wilson fitter."""

from __future__ import annotations

from typing import Any


def french_wilson(observed: Any = None) -> Any:
    """Return the observations unchanged. Calibration is intentionally absent."""
    return observed
