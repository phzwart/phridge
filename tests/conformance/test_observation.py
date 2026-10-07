"""Collect the observation conformance suite."""

from __future__ import annotations

from phridge.testing.conformance import observation as suite


def test_identity_round_trip() -> None:
    suite.test_identity_round_trip()


def test_adjoint_shapes() -> None:
    suite.test_adjoint_shapes()


def test_record_indexing_declared() -> None:
    suite.test_record_indexing_declared()
