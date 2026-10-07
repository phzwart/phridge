"""Q-component conformance."""

from __future__ import annotations

import torch

from phridge.contracts.tensors import Predictive
from phridge.q.manifests import MANIFESTS, component_for


def test_produces_declared_fields() -> None:
    location = torch.arange(8, dtype=torch.float64)
    for name, manifest in MANIFESTS.items():
        out = component_for(name).forward(Predictive({"location": location}))
        assert set(manifest.produces) <= set(out.fields)
        for field in manifest.produces:
            assert out.fields[field].shape == location.shape


def test_adjoint_shapes_match() -> None:
    location = torch.arange(8, dtype=torch.float64)
    for name, manifest in MANIFESTS.items():
        component = component_for(name)
        cotangent = Predictive({field: torch.ones_like(location) for field in manifest.produces})
        back = component.adjoint(cotangent)
        assert back.fields["location"].shape == location.shape
