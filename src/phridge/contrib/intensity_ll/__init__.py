"""Crystallographic intensity likelihood target (``ml_i``).

Entry point: ``phridge.targets`` → ``phridge.contrib.intensity_ll:register``.
"""

from __future__ import annotations

from typing import Any

from phridge.contrib.intensity_ll import target as _target  # noqa: F401
from phridge.contrib.intensity_ll.ops import (
    NUISANCE_FIT_OP_NAME,
    OP_NAME as MAPS_OP_NAME,
    TARGET_AND_GRADIENTS_OP_NAME,
    register_ops,
)
from phridge.contrib.intensity_ll.omit_windows import OMIT_OP_NAME
from phridge.contrib.intensity_ll.surrogate_op import SURROGATE_FIT_OP_NAME
from phridge.contrib.intensity_ll.target import IntensityLogLikelihood, IntensityLogLikelihoodOptions

__all__ = [
    "IntensityLogLikelihood",
    "IntensityLogLikelihoodOptions",
    "MAPS_OP_NAME",
    "NUISANCE_FIT_OP_NAME",
    "OMIT_OP_NAME",
    "SURROGATE_FIT_OP_NAME",
    "TARGET_AND_GRADIENTS_OP_NAME",
    "log_likelihood_normal",
    "log_likelihood_t",
    "normalize",
    "posterior_mode_E",
    "register",
]


_LAZY_MLI = {"log_likelihood_normal", "log_likelihood_t", "normalize", "posterior_mode_E"}


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
