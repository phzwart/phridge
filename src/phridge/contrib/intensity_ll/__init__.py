"""Crystallographic intensity likelihood target (``ml_i``).

Entry point: ``phridge.targets`` → ``phridge.contrib.intensity_ll:register``.
"""

from __future__ import annotations

from phridge.contrib.intensity_ll import target as _target  # noqa: F401
from phridge.contrib.intensity_ll.mli import (
    log_likelihood_normal,
    log_likelihood_t,
    normalize,
)
from phridge.contrib.intensity_ll.target import IntensityLogLikelihood, IntensityLogLikelihoodOptions

__all__ = [
    "IntensityLogLikelihood",
    "IntensityLogLikelihoodOptions",
    "log_likelihood_normal",
    "log_likelihood_t",
    "normalize",
    "register",
]


def register() -> None:
    """Entry-point hook; importing this package already registers ``ml_i``."""
    from phridge.contrib.intensity_ll import target as _  # noqa: F401
