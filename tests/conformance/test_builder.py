"""Builders propose without mutating, and apply/revert is bitwise."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from phridge.contracts.manifest import Space
from phridge.core.controller import Controller
from phridge.core.graph import Graph, Node
from phridge.core.registry import Block, Registry
from phridge.transforms.build.base import RChange
from phridge.transforms.build.rotamer import RotamerBuilder
from phridge.transforms.build.water import WaterBuilder


def _ready(sites: torch.Tensor) -> tuple[Registry, Controller]:
    registry = Registry()
    registry.register(
        Block(
            id="sites",
            component="model",
            space=Space.R,
            shape=tuple(sites.shape),
            bounds=None,
            transform="identity",
            frozen=False,
            value_version=0,
            structural_version=0,
            value=sites.clone(),
        )
    )
    graph = Graph(registry)
    graph.add(Node(name="sites", kind="block"))
    controller = Controller(registry, graph)
    controller.configure()
    controller.calibrate()
    return registry, controller


def _map() -> torch.Tensor:
    values = torch.zeros((6, 6, 6), dtype=torch.float64)
    values[1, 1, 1] = 8
    values[4, 4, 4] = 9
    return values


def test_propose_does_not_mutate() -> None:
    difference = _map()
    original = difference.clone()
    sites = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
    change = WaterBuilder().propose({"difference": difference}, {"sites": sites, "threshold": 1.0, "min_distance": 0.05}, None)
    assert torch.equal(difference, original)
    assert change.add is not None
    RotamerBuilder().propose({}, {"residue": "val"}, None)


def test_idempotent_on_unchanged_map() -> None:
    difference = _map()
    stats = {"threshold": 1.0, "min_distance": 0.0}
    first = WaterBuilder().propose({"difference": difference}, stats, None)
    second = WaterBuilder().propose({"difference": difference}, stats, None)
    assert torch.equal(first.add.sites_frac, second.add.sites_frac)


def test_apply_then_revert_bitwise() -> None:
    sites = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float64)
    registry, controller = _ready(sites)
    change = WaterBuilder().propose({"difference": _map()}, {"threshold": 1.0, "min_distance": 0.0}, None)
    original = registry.block("sites").value.clone()
    controller.propose()
    controller.apply(change)
    assert registry.block("sites").shape[0] > original.shape[0]
    controller.propose()
    controller.apply(RChange(None, torch.zeros(0, dtype=torch.bool), {"sites": original}))
    assert torch.equal(registry.block("sites").value, original)
