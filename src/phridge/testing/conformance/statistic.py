"""Statistic conformance."""

from __future__ import annotations

import inspect

from phridge.report.base import Statistic


def test_compute_takes_masks() -> None:
    parameters = inspect.signature(Statistic.compute).parameters
    assert "masks" in parameters
    assert "labels" not in parameters
