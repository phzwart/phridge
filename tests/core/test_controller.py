"""Controller guards, receipt replay, and tensor-free observations."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pydantic import BaseModel

from phridge.contracts.manifest import Space
from phridge.core.controller import Controller, ControllerRefused, Observation
from phridge.core.graph import Graph, Node
from phridge.core.moveset import Entry, MoveSet
from phridge.core.registry import Block, Registry


def _registry(value: float = 1.0) -> tuple[Registry, Graph]:
    registry = Registry()
    registry.register(
        Block(
            id="a",
            component="toy",
            space=Space.R,
            shape=(2,),
            bounds=None,
            transform="identity",
            frozen=False,
            value_version=0,
            structural_version=0,
            value=torch.tensor([value, value], dtype=torch.float64),
        )
    )
    graph = Graph(registry)
    graph.add(Node(name="a", kind="block"))
    return registry, graph


def _no_tensors(value: object) -> None:
    if isinstance(value, torch.Tensor):
        raise AssertionError("observation holds a tensor")
    if isinstance(value, BaseModel):
        for field in type(value).model_fields:
            _no_tensors(getattr(value, field))
        return
    if isinstance(value, dict):
        for item in value.values():
            _no_tensors(item)
    if isinstance(value, (list, tuple)):
        for item in value:
            _no_tensors(item)


def test_guards_and_replay() -> None:
    registry, graph = _registry()
    initial = registry.block("a").value.clone()
    controller = Controller(registry, graph)

    controller.configure()
    controller.calibrate()
    controller.step(lambda reg: reg.set_value("a", reg.block("a").value + 1))
    controller.evaluate()
    controller.accept()
    report = controller.report()
    _no_tensors(report)
    assert isinstance(report, Observation)
    with pytest.raises(ControllerRefused, match="accepted step"):
        controller.report()
    with pytest.raises(ControllerRefused, match="continue_"):
        controller.step()
    controller.continue_()
    controller.step(lambda reg: reg.set_value("a", reg.block("a").value + 1))
    with pytest.raises(ControllerRefused, match="Step/Evaluate"):
        controller.reconfigure()
    controller.evaluate()
    controller.accept()
    with pytest.raises(ControllerRefused, match="Ready"):
        held = Controller(registry, graph)
        held.configure()
        held.propose()
    controller.propose()
    controller.apply("a", (3,), torch.tensor([9.0, 8.0, 7.0], dtype=torch.float64))
    assert registry.block("a").structural_version >= 1
    assert controller.receipts[-1].action == "apply"

    final = registry.block("a").value.clone()
    registry.reshape("a", (2,), initial)
    controller.replay(registry)
    assert torch.equal(registry.block("a").value, final)


def test_step_refuses_a_frozen_move_set_and_a_bad_audit() -> None:
    registry, graph = _registry()
    registry.freeze("a")
    controller = Controller(registry, graph, move_set=MoveSet(name="held", entries=[Entry(block="a")]))
    controller.configure()
    controller.calibrate()
    with pytest.raises(ControllerRefused, match="frozen"):
        controller.step()

    registry, graph = _registry()
    graph.add(Node(name="partition.audit", kind="block"))
    graph.add(Node(name="fit", kind="derived", reads=["partition.audit"]))
    controller = Controller(registry, graph)
    controller.configure()
    controller.calibrate()
    with pytest.raises(ControllerRefused, match="partition.audit"):
        controller.step()
