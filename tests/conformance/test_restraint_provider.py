"""Collect the restraint-provider conformance suite."""

from __future__ import annotations

from phridge.testing.conformance import restraint_provider as suite


def test_gradient_matches_finite_difference() -> None:
    suite.test_gradient_matches_finite_difference()


def test_energy_nonnegative() -> None:
    suite.test_energy_nonnegative()


def test_covers_declared() -> None:
    suite.test_covers_declared()
