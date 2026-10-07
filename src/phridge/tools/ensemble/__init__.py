"""Shared reciprocal-space perturbation modes against the intensity likelihood.

The client side (options, ``prepare_ensemble_job``, the CLI) does not import
torch. Worker implementations live in ``ops`` and load torch lazily.
"""

from __future__ import annotations

from phridge.tools.ensemble.options import (
    BootstrapOptions,
    DatasetRoles,
    EnsembleOptions,
    HoldoutOptions,
    PhaseInjectionOptions,
    ProjectionOptions,
)
from phridge.tools.ensemble.ops import OP_NAME_FIT, OP_NAME_MAPS, OP_NAME_SCORES, register_ops

__all__ = [
    "BootstrapOptions",
    "DatasetRoles",
    "EnsembleOptions",
    "HoldoutOptions",
    "OP_NAME_FIT",
    "OP_NAME_MAPS",
    "OP_NAME_SCORES",
    "PhaseInjectionOptions",
    "ProjectionOptions",
    "register",
    "register_ops",
]


def register() -> None:
    """Entry-point hook. Safe to call twice."""
    register_ops()
