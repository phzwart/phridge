"""Gaussian intensity negative log-likelihood target (``ml_i``).

Entry point: ``phridge.targets`` → ``phridge.contrib.intensity_ll:register``.
"""

from __future__ import annotations

from phridge.contrib.intensity_ll import target as _target  # noqa: F401
from phridge.contrib.intensity_ll.target import IntensityLogLikelihood, IntensityLogLikelihoodOptions

__all__ = [
    "IntensityLogLikelihood",
    "IntensityLogLikelihoodOptions",
    "register",
]


def register() -> None:
    """Entry-point hook; importing this package already registers ``ml_i``."""
    # Re-import ensures the decorator ran even if another importer skipped it.
    from phridge.contrib.intensity_ll import target as _  # noqa: F401
