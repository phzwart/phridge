"""Reciprocal-space refinement targets Q = sum_h w_h g(obs_h, F_h) in torch.

A target only has to define its per-reflection value; autograd supplies
d_target_d_f_calc (cctbx convention dQ/dA + i dQ/dB) and, for
amplitude-only targets, the radial / tangential curvatures.
"""

from phridge.d.rules.amplitude.base import (
    Observations,
    Target,
    TargetEval,
    build_target,
    list_targets,
    register_target,
)
from phridge.d.rules.amplitude.least_squares import LeastSquares
from phridge.d.rules.amplitude.maximum_likelihood import MaximumLikelihoodAmplitude

__all__ = [
    "Observations",
    "Target",
    "TargetEval",
    "LeastSquares",
    "MaximumLikelihoodAmplitude",
    "build_target",
    "list_targets",
    "register",
    "register_target",
]


def register() -> None:
    """Importing this package already registers the ls and ml_f targets."""
    return None
