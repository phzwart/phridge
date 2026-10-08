"""Shared backtracking line search. Kept out of the optimizer methods so they stay short."""

from __future__ import annotations

from typing import Any


def backtrack(objective: Any, x: Any, direction: Any, value0: Any) -> tuple[Any, Any]:
    """Halve the step until the value drops, or return the smallest trial."""
    step = 1.0
    base = float(value0.detach()) if hasattr(value0, "detach") else float(value0)
    best = x
    best_value = value0
    for _ in range(8):
        trial = (x + step * direction).detach()
        trial_value = objective.value(trial)
        if float(trial_value.detach()) < base:
            return trial, trial_value
        step *= 0.5
        best = trial
        best_value = trial_value
    return best, best_value
