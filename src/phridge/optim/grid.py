"""One-block grid search. ``nu_screen`` uses this; the criterion is supplied by the caller."""

from __future__ import annotations

from typing import Any, Callable, Optional

from phridge.optim.base import Budget, StepResult, as_float, require_capabilities


class GridSearch:
    """Search one scalar across a block. Every element of the block takes that scalar."""

    requires: tuple[str, ...] = ()

    def __init__(self, criterion: Optional[Callable[[Any, Any], Any]] = None, points: int = 11) -> None:
        self.criterion = criterion
        self.points = points

    def step(self, objective: Any, x0: Any, bounds: Any, budget: Budget) -> StepResult:
        """Pick the grid node with the smallest criterion. Default criterion is ``objective.value``."""
        import torch

        require_capabilities(self, objective)
        count = budget.iterations if budget.iterations is not None else self.points
        lo, hi = _span(bounds, x0)
        nodes = torch.linspace(lo, hi, int(count), dtype=x0.dtype, device=x0.device)
        best = x0.detach().clone()
        best_score = float("inf")
        trace: list[float] = []
        for node in nodes:
            trial = torch.full_like(x0, float(node))
            score = self.criterion(objective, trial) if self.criterion is not None else objective.value(trial)
            value = as_float(score)
            trace.append(value)
            if value < best_score:
                best_score = value
                best = trial.detach().clone()
        return StepResult(best, tuple(trace), len(trace))


def _span(bounds: Any, x0: Any) -> tuple[float, float]:
    if bounds is None:
        return float(x0.detach().reshape(-1)[0]) - 1.0, float(x0.detach().reshape(-1)[0]) + 1.0
    lo, hi = bounds
    return float(lo), float(hi)
