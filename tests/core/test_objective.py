"""Toy quadratic objective: analytic gradient, masked entry, and the hvp gate."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from phridge.contracts.manifest import Space
from phridge.core.graph import Graph, Node
from phridge.core.moveset import Entry, MoveSet, resolve
from phridge.core.objective import Objective, ObjectiveRefused
from phridge.core.registry import Block, Registry


def _block(name: str) -> Block:
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
        value=torch.tensor(0.0, dtype=torch.float64),
    )


def _objective(mask: torch.Tensor | None = None, *, restraint_hvp: bool = True) -> tuple[Objective, Registry]:
    registry = Registry()
    registry.register(_block("a"))
    registry.register(_block("b"))
    graph = Graph(registry)
    graph.add(Node(name="a", kind="block"))
    graph.add(Node(name="b", kind="block"))
    graph.add(
        Node(
            name="score",
            kind="derived",
            reads=["a", "b"],
            compute=lambda inputs: inputs["a"] ** 2 + inputs["b"] ** 2,
            provides=("hvp",),
        )
    )
    restraints: list[str] = []
    weights: list[float] = []
    if not restraint_hvp:
        graph.add(Node(name="pull", kind="derived", reads=["a"], compute=lambda inputs: inputs["a"], provides=()))
        restraints = ["pull"]
        weights = [1.0]
    resolved = resolve(MoveSet(name="free", entries=[Entry(block="a"), Entry(block="b")]), registry, lambda _name: None)
    return Objective(graph, registry, resolved, "score", restraints, weights, mask=mask), registry


def test_gradient_matches_the_analytic_quadratic() -> None:
    objective, _registry = _objective()
    x = torch.tensor([3.0, 4.0], dtype=torch.float64)
    assert objective.value(x).item() == 25.0
    torch.testing.assert_close(objective.grad(x), torch.tensor([6.0, 8.0], dtype=torch.float64))
    torch.testing.assert_close(
        objective.hvp(x, torch.ones(2, dtype=torch.float64)), torch.tensor([2.0, 2.0], dtype=torch.float64)
    )


def test_masked_entry_has_zero_gradient() -> None:
    mask = torch.tensor([True, False])
    objective, _registry = _objective(mask)
    gradient = objective.grad(torch.tensor([3.0, 4.0], dtype=torch.float64))
    torch.testing.assert_close(gradient, torch.tensor([6.0, 0.0], dtype=torch.float64))


def test_hvp_is_refused_when_a_node_lacks_it() -> None:
    objective, _registry = _objective(restraint_hvp=False)
    with pytest.raises(ObjectiveRefused, match="hvp") as caught:
        objective.hvp(torch.tensor([3.0, 4.0], dtype=torch.float64), torch.ones(2, dtype=torch.float64))
    assert caught.value.path == "pull"


def test_generator_adjoint_replaces_the_packed_gradient() -> None:
    registry = Registry()
    registry.register(
        Block(
            id="g",
            component="toy",
            space=Space.R,
            shape=(2,),
            bounds=None,
            transform="identity",
            frozen=False,
            value_version=0,
            structural_version=0,
            value=torch.zeros(2, dtype=torch.float64),
        )
    )
    graph = Graph(registry)
    graph.add(Node(name="g", kind="block"))
    graph.add(
        Node(
            name="coords",
            kind="derived",
            reads=["g"],
            compute=lambda inputs: inputs["g"].detach().requires_grad_(True),
            produces=("atoms",),
            adjoint=lambda atom_grad: atom_grad * 4,
        )
    )
    graph.add(
        Node(
            name="score",
            kind="derived",
            reads=["coords"],
            compute=lambda inputs: (inputs["coords"] ** 2).sum(),
            provides=("hvp",),
        )
    )
    resolved = resolve(MoveSet(name="gen", entries=[Entry(block="g")]), registry, lambda _name: None)
    objective = Objective(graph, registry, resolved, "score", [], [], mask=None)
    gradient = objective.grad(torch.tensor([1.0, 2.0], dtype=torch.float64))
    torch.testing.assert_close(gradient, torch.tensor([8.0, 16.0], dtype=torch.float64))
