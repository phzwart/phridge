"""Gauss-Newton. The Fourier operator's ``gauss_newton_blocks`` are the block-diagonal preconditioner."""

from __future__ import annotations

from typing import Any

from phridge.optim.base import Budget, StepResult, require_capabilities
from phridge.optim._search import backtrack


class GaussNewton:
    """Damped step along ``-g / diag(blocks)``. ``blocks`` come from the objective preconditioner."""

    requires = ("grad", "hvp", "preconditioner")

    def __init__(self, operator: Any = None) -> None:
        self.operator = operator

    def step(self, objective: Any, x0: Any, bounds: Any, budget: Budget) -> StepResult:
        """One damped Gauss-Newton step. ``bounds`` stay in the registry transforms."""
        _ = bounds
        _ = budget
        require_capabilities(self, objective)
        vector = x0.detach().clone()
        value0 = objective.value(vector)
        grad = objective.grad(vector).detach().reshape(-1)
        diag = _diagonal(self.operator, objective, vector, grad.numel())
        direction = (-grad / diag).reshape_as(vector)
        objective.hvp(vector, direction)
        updated, _ = backtrack(objective, vector, direction, value0)
        return StepResult(updated.detach(), (float(value0.detach()),), 2)


def _diagonal(operator: Any, objective: Any, vector: Any, length: int) -> Any:
    """Prefer ``operator.gauss_newton_blocks``; otherwise the objective preconditioner."""
    if operator is not None and hasattr(operator, "gauss_newton_blocks"):
        blocks = operator.gauss_newton_blocks(vector)
        flat = blocks.detach().reshape(-1)
        if flat.numel() == length:
            return flat.abs().clamp_min(1e-8)
    diag = objective.preconditioner().detach().reshape(-1).abs().clamp_min(1e-8)
    if diag.numel() == 1:
        return diag.expand(length)
    return diag
