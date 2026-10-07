"""Observation-operator conformance."""

from __future__ import annotations

import torch

from phridge.transforms.observation.identity import INDEX_FIELDS, IdentityObservation


def test_identity_round_trip() -> None:
    record = {"hkl": torch.tensor([[1, 0, 0], [0, 1, 0]]), "values": torch.ones(2)}
    operator = IdentityObservation()
    assert operator.forward(operator.forward(record))["hkl"].shape == record["hkl"].shape
    assert torch.equal(operator.forward(record)["values"], record["values"])


def test_adjoint_shapes() -> None:
    cotangent = {"values": torch.ones(4, dtype=torch.float64)}
    back = IdentityObservation().adjoint(cotangent)
    assert back["values"].shape == cotangent["values"].shape


def test_record_indexing_declared() -> None:
    assert IdentityObservation.index_fields == INDEX_FIELDS
    assert "hkl" in INDEX_FIELDS
