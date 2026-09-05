"""Crystallographic intensity likelihood target (``ml_i``).

Entry point: ``phridge.targets`` → ``phridge.contrib.intensity_ll:register``.
"""

from __future__ import annotations

from typing import Any

from phridge.contrib.intensity_ll import target as _target  # noqa: F401
from phridge.contrib.intensity_ll.ops import OP_NAME as MAPS_OP_NAME
from phridge.contrib.intensity_ll.ops import register_ops
from phridge.contrib.intensity_ll.target import IntensityLogLikelihood, IntensityLogLikelihoodOptions

__all__ = [
    "IntensityLogLikelihood",
    "IntensityLogLikelihoodOptions",
    "MAPS_OP_NAME",
    "log_likelihood_normal",
    "log_likelihood_t",
    "normalize",
    "register",
]


_LAZY_MLI = {"log_likelihood_normal", "log_likelihood_t", "normalize"}


def __getattr__(name: str) -> Any:
    """Lazy re-exports of the torch-based quadrature so cctbx clients can import this package torch-free."""
    if name in _LAZY_MLI:
        from phridge.contrib.intensity_ll import mli

        return getattr(mli, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def register() -> None:
    """Entry-point hook; importing this package already registers ``ml_i`` and the ``ml_i_maps`` op."""
    from phridge.contrib.intensity_ll import target as _  # noqa: F401

    register_ops()
