"""Parameter optimizers. Import walls: torch, numpy, pydantic, phridge."""

from __future__ import annotations

from phridge.optim.base import Budget, Optimizer, OptimizerRefused, StepResult
from phridge.optim.gauss_newton import GaussNewton
from phridge.optim.grid import GridSearch
from phridge.optim.lbfgs import LBFGS
from phridge.optim.manifests import MANIFESTS, register
from phridge.optim.newton_cg import NewtonCG

__all__ = [
    "Budget",
    "GaussNewton",
    "GridSearch",
    "LBFGS",
    "MANIFESTS",
    "NewtonCG",
    "Optimizer",
    "OptimizerRefused",
    "StepResult",
    "register",
]
