"""Compatibility shim — prefer ``phridge.sfcalc.targets``."""

from phridge.sfcalc.targets import (  # noqa: F401
    LeastSquares,
    MaximumLikelihoodAmplitude,
    Observations,
    Target,
    TargetEval,
    build_target,
    list_targets,
    register_target,
)

__all__ = [
    "Observations",
    "Target",
    "TargetEval",
    "LeastSquares",
    "MaximumLikelihoodAmplitude",
    "build_target",
    "list_targets",
    "register_target",
]
