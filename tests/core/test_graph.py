"""Arithmetic graph: recompute the stale chain and leave the other branch cached."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from phridge.contracts.manifest import PluginManifest, Slot, Space
from phridge.core.graph import Graph, Node
from phridge.core.moveset import Entry, MoveSet
from phridge.core.registry import Block, Registry


def _block(name: str, value: float) -> Block:
    return Block(
        id=name,
        component="toy",
        space=Space.R,
        shape=(),
        bounds=None,
        transform="identity",
        frozen=False,
        value_version=0,
        structural_version=0,
        value=torch.tensor(value, dtype=torch.float64),
    )


def _graph() -> tuple[Registry, Graph]:
    registry = Registry()
    registry.register(_block("x", 2.0))
    registry.register(_block("y", 3.0))
    graph = Graph(registry)
    graph.add(Node(name="x", kind="block"))
    graph.add(Node(name="y", kind="block"))
    graph.add(Node(name="sum", kind="derived", reads=["x", "y"], compute=lambda inputs: inputs["x"] + inputs["y"]))
    graph.add(Node(name="prod", kind="derived", reads=["x", "y"], compute=lambda inputs: inputs["x"] * inputs["y"]))
    graph.add(
        Node(name="total", kind="derived", reads=["sum", "prod"], compute=lambda inputs: inputs["sum"] + inputs["prod"])
    )
    graph.add(Node(name="other", kind="derived", reads=["y"], compute=lambda inputs: inputs["y"] * 10))
    return registry, graph


def test_recompute_count_matches_the_stale_chain() -> None:
    registry, graph = _graph()
    assert graph.cost("total") == ["sum", "prod", "total"]
    assert graph.get("total").item() == 11.0
    assert len(graph.receipts) == 3
    assert graph.cost("total") == []
    graph.get("total")
    assert len(graph.receipts) == 3
    graph.get("other")
    assert graph.nodes["other"].cached.item() == 30.0
    other_version = graph.nodes["other"].value_version
    registry.set_value("x", torch.tensor(5.0, dtype=torch.float64))
    assert graph.cost("total") == ["sum", "prod", "total"]
    assert "other" not in graph.cost("other")
    before = len(graph.receipts)
    assert graph.get("total").item() == 23.0
    assert len(graph.receipts) - before == 3
    assert graph.nodes["other"].value_version == other_version
    assert graph.nodes["other"].cached.item() == 30.0


def test_structural_bump_clears_downstream_shape_cache_only() -> None:
    _registry, graph = _graph()
    for name in ("x", "sum", "prod", "total", "other"):
        graph.nodes[name].shape_cache["kept"] = name
    graph.structural_bump("x")
    assert graph.nodes["x"].shape_cache["kept"] == "x"
    assert graph.nodes["other"].shape_cache["kept"] == "other"
    assert graph.nodes["sum"].shape_cache == {}
    assert graph.nodes["prod"].shape_cache == {}
    assert graph.nodes["total"].shape_cache == {}


def test_rebuild_recomputes_everything() -> None:
    _registry, graph = _graph()
    graph.get("total")
    graph.get("other")
    assert len(graph.receipts) == 4
    graph.rebuild()
    assert graph.cost("total") == ["sum", "prod", "total"]
    assert graph.cost("other") == ["other"]
    graph.get("total")
    graph.get("other")
    assert len(graph.receipts) == 8


def test_readers_affected_and_from_manifests() -> None:
    registry, graph = _graph()
    assert graph.readers("x") == {"sum", "prod"}
    assert graph.readers("sum") == {"total"}
    assert graph.affected(["x"]) == {"sum", "prod", "total"}
    assert graph.affected(MoveSet(name="y", entries=[Entry(block="y")])) == {"sum", "prod", "total", "other"}
    built = Graph.from_manifests(
        [
            PluginManifest(name="sum", slot=Slot.scoring_rule, space=Space.D, reads=["x", "y"], produces=["score"]),
            PluginManifest(name="other", slot=Slot.statistic, space=Space.D, reads=["y"], produces=["s_post"]),
        ],
        registry,
    )
    assert built.readers("x") == {"sum"}
    assert built.readers("y") == {"sum", "other"}
    assert built.nodes["x"].kind == "block"
    assert built.nodes["sum"].produces == ("score",)
