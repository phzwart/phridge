"""Collect the statistic conformance suite."""

from __future__ import annotations

from phridge.testing.conformance import statistic as suite


def test_compute_takes_masks() -> None:
    suite.test_compute_takes_masks()
