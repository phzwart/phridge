"""Optimizer slot: a quadratic decreases, bounds hold, a missing capability is refused."""

from __future__ import annotations

from typing import Any

import torch

from phridge.contracts.manifest import Space
from phridge.core.graph import Graph, Node
from phridge.core.moveset import Entry, MoveSet, resolve
from phridge.core.objective import Objective
from phridge.core.registry import Block, Registry
from phridge.optim.base import Budget, OptimizerRefused
from phridge.optim.gauss_newton import GaussNewton
from phridge.optim.lbfgs import LBFGS
from phridge.optim.newton_cg import NewtonCG


class _Quadratic:
    """f(x) = ||x||^2. Hessian is 2 I."""

    provides = ("grad", "hvp", "preconditioner")

    def value(self, x: Any) -> Any:
        return (x * x).sum()

    def grad(self, x: Any) -> Any:
        return 2 * x

    def hvp(self, x: Any, v: Any) -> Any:
        _ = x
        return 2 * v

    def preconditioner(self) -> Any:
        return torch.tensor(2.0, dtype=torch.float64)


def test_decreases_quadratic() -> None:
    objective = _Quadratic()
    x0 = torch.tensor([3.0, -4.0], dtype=torch.float64)
    start = float(objective.value(x0))
    budget = Budget(iterations=6)
    for optimizer in (LBFGS(), NewtonCG(), GaussNewton()):
        result = optimizer.step(objective, x0, None, budget)
        assert float(objective.value(result.x1)) < start


def test_respects_bounds() -> None:
    registry = Registry()
    registry.register(
        Block(
            id="p",
            component="toy",
            space=Space.R,
            shape=(2,),
            bounds=(torch.tensor(0.0), torch.tensor(1.0)),
            transform="logit",
            frozen=False,
            value_version=0,
            structural_version=0,
            value=torch.tensor([0.2, 0.8], dtype=torch.float64),
        )
    )
    graph = Graph(registry)
    graph.add(Node(name="p", kind="block"))
    graph.add(
        Node(
            name="score",
            kind="derived",
            reads=["p"],
            compute=lambda inputs: ((inputs["p"] - 0.9) ** 2).sum(),
            provides=("hvp",),
        )
    )
    resolved = resolve(MoveSet(name="p", entries=[Entry(block="p")]), registry, lambda _name: None)
    objective = Objective(graph, registry, resolved, "score", [], [])
    objective.provides = ("grad",)
    packed = registry.pack(resolved)
    result = LBFGS().step(objective, packed.vector, packed.bounds, Budget(iterations=4))
    registry.unpack(result.x1, resolved)
    values = registry.block("p").value
    assert bool((values > 0).all()) and bool((values < 1).all())


def test_refuses_missing_capability() -> None:
    class _GradOnly:
        provides = ("grad",)

        def value(self, x: Any) -> Any:
            return (x * x).sum()

        def grad(self, x: Any) -> Any:
            return 2 * x

    objective = _GradOnly()
    x0 = torch.ones(2, dtype=torch.float64)
    budget = Budget(iterations=2)
    for optimizer in (NewtonCG(), GaussNewton(), LBFGS()):
        if set(optimizer.requires) <= set(objective.provides):
            continue
        try:
            optimizer.step(objective, x0, None, budget)
        except OptimizerRefused:
            continue
        raise AssertionError(type(optimizer).__name__)
