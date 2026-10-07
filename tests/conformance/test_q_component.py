"""Collect the Q-component conformance suite."""

from __future__ import annotations

from phridge.testing.conformance import q_component as suite


def test_produces_declared_fields() -> None:
    suite.test_produces_declared_fields()


def test_adjoint_shapes_match() -> None:
    suite.test_adjoint_shapes_match()
