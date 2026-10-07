"""Derivative-free outer loops."""

from __future__ import annotations

from typing import Any, Callable


def minimize(fun: Callable[..., float], x0: Any, method: str, options: dict[str, Any]) -> Any:
    from scipy.optimize import minimize as scipy_minimize

    return scipy_minimize(fun, x0, method=method, options=options)
