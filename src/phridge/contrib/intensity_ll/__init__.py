"""Crystallographic intensity likelihood target (``ml_i``).

Entry point: ``phridge.plugins`` → ``phridge.d.rules.mli_quad:register``.
"""

from __future__ import annotations

from typing import Any

from phridge.d.rules.mli_quad.target import IntensityLogLikelihood, IntensityLogLikelihoodOptions
from phridge.tools.intensity.omit_windows import OMIT_OP_NAME
from phridge.tools.intensity.surrogate_op import SURROGATE_FIT_OP_NAME

MAPS_OP_NAME = "ml_i_maps"
NUISANCE_FIT_OP_NAME = "ml_i_nuisance_fit"
TARGET_AND_GRADIENTS_OP_NAME = "ml_i_target_and_gradients"

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
        from phridge.d.rules import mli_quad as mli

        return getattr(mli, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def register() -> None:
    """Register ``ml_i`` plus the solvent and surrogate ops that live outside legacy."""
    from phridge.d.rules.mli_quad import target as _  # noqa: F401
    from phridge.ops import register_op
    from phridge.q.scales.solvent import (
        _BULK_SOLVENT_INPUTS,
        _BULK_SOLVENT_OUTPUTS,
        BULK_SOLVENT_OP_NAME,
        ml_i_bulk_solvent_fit,
    )
    from phridge.tools.intensity.surrogate_op import (
        _SURROGATE_INPUTS,
        _SURROGATE_OUTPUTS,
        ml_i_surrogate_fit,
    )

    register_op(
        BULK_SOLVENT_OP_NAME,
        ml_i_bulk_solvent_fit,
        inputs=dict(_BULK_SOLVENT_INPUTS),
        outputs=dict(_BULK_SOLVENT_OUTPUTS),
    )
    register_op(
        SURROGATE_FIT_OP_NAME,
        ml_i_surrogate_fit,
        inputs=dict(_SURROGATE_INPUTS),
        outputs=dict(_SURROGATE_OUTPUTS),
    )
