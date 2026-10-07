"""Collect the scoring-rule conformance suite."""

from __future__ import annotations

from phridge.testing.conformance import scoring_rule as suite


def test_needs_subset_of_predictive() -> None:
    suite.test_needs_subset_of_predictive()


def test_score_equals_masked_sum() -> None:
    suite.test_score_equals_masked_sum()


def test_complex_step_d_location() -> None:
    suite.test_complex_step_d_location()
