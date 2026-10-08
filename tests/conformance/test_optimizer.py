"""Optimizer conformance suite."""

from __future__ import annotations

from phridge.testing.conformance.optimizer import (
    test_decreases_quadratic,
    test_refuses_missing_capability,
    test_respects_bounds,
)

__all__ = [
    "test_decreases_quadratic",
    "test_refuses_missing_capability",
    "test_respects_bounds",
]
