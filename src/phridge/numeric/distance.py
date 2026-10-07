"""Euclidean distance transform."""

from __future__ import annotations

from typing import Any


def distance_transform_edt(image: Any, sampling: tuple[float, float, float]) -> Any:
    from scipy.ndimage import distance_transform_edt as edt

    return edt(image, sampling=sampling)
