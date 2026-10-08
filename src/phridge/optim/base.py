"""Optimizer slot. Bounds live in the registry transforms, not in the step."""

from __future__ import annotations

from typing import Any, NamedTuple, Optional, Protocol

from pydantic import BaseModel, ConfigDict


class Budget(BaseModel):
    """How long one ``step`` may run."""

    model_config = ConfigDict(extra="forbid")

    iterations: Optional[int] = None
    seconds: Optional[float] = None


class StepResult(NamedTuple):
    """The vector after one step, the values seen, and how many times the objective was called."""

    x1: Any
    trace: tuple[float, ...]
    n_evals: int


class OptimizerRefused(Exception):
    """The objective does not provide a capability this optimizer requires."""

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


class Optimizer(Protocol):
    """One algorithm. ``requires`` names capabilities the objective must provide."""

    requires: tuple[str, ...]

    def step(self, objective: Any, x0: Any, bounds: Any, budget: Budget) -> StepResult:
        """Advance ``x0``. ``bounds`` are already encoded by the registry transforms."""


def require_capabilities(optimizer: Optimizer, objective: Any) -> None:
    """Refuse when ``objective.provides`` lacks anything in ``optimizer.requires``."""
    have = set(getattr(objective, "provides", ()))
    missing = [name for name in optimizer.requires if name not in have]
    if missing:
        raise OptimizerRefused(type(optimizer).__name__, "missing " + ", ".join(missing))


def as_float(value: Any) -> float:
    """A python float from a tensor or a number."""
    if hasattr(value, "detach"):
        return float(value.detach().reshape(-1)[0]) if value.numel() != 1 else float(value.detach())
    return float(value)
