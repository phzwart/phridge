"""Objective over a resolved move set: value, gradient, and an optional Hessian-vector product."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from phridge.core.graph import Graph
from phridge.core.moveset import ResolvedEntry
from phridge.core.registry import Registry


class ObjectiveRefused(Exception):
    """``hvp`` was asked for along a path that does not provide it."""

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


class Objective:
    """Pull a score and restraints, then mask and scale the packed gradient."""

    def __init__(
        self,
        graph: Graph,
        registry: Registry,
        resolved: Sequence[ResolvedEntry],
        score_node: str,
        restraint_nodes: Sequence[str],
        weights: Sequence[float],
        mask: Any = None,
    ) -> None:
        if len(restraint_nodes) != len(weights):
            raise ValueError("weights and restraint nodes differ in length")
        self.graph = graph
        self.registry = registry
        self.resolved = list(resolved)
        self.score_node = score_node
        self.restraint_nodes = list(restraint_nodes)
        self.weights = [float(weight) for weight in weights]
        self.mask = mask
        self._scales = registry.pack(self.resolved).scales.detach().clone()

    def value(self, x: Any) -> Any:
        self.registry.unpack(x, self.resolved)
        return self._pull()

    def grad(self, x: Any) -> Any:
        """Gradient of the packed vector. A generator ``adjoint`` replaces autograd through atoms."""
        import torch

        leaf = x.detach().clone().requires_grad_(True)
        self.registry.unpack(leaf, self.resolved)
        total = self._pull()
        _retain(self.graph, self._path())
        total.backward()
        if leaf.grad is None:
            packed = torch.zeros_like(leaf)
        else:
            packed = leaf.grad.detach().clone()
        return self._mask_and_scale(_apply_adjoints(self.graph, self.registry, self.resolved, self._path(), packed))

    def hvp(self, x: Any, v: Any) -> Any:
        """Hessian-vector product. Refused unless every derived node on the path provides ``hvp``."""
        import torch

        self._require_hvp()
        leaf = x.detach().clone().requires_grad_(True)
        self.registry.unpack(leaf, self.resolved)
        total = self._pull()
        first = torch.autograd.grad(total, leaf, create_graph=True)[0]
        product = torch.dot(first.reshape(-1), v.reshape(-1).to(dtype=first.dtype))
        second = torch.autograd.grad(product, leaf)[0]
        return self._mask_and_scale(second.detach())

    def preconditioner(self) -> Any:
        """Diagonal scale the optimizer can use as a simple preconditioner."""
        return self._scales.clone()

    def _pull(self) -> Any:
        total = self.graph.get(self.score_node)
        for node, weight in zip(self.restraint_nodes, self.weights):
            total = total + weight * self.graph.get(node)
        return total

    def _path(self) -> list[str]:
        seen: list[str] = []
        for name in (self.score_node, *self.restraint_nodes):
            for item in _ancestors(self.graph, name):
                if item not in seen:
                    seen.append(item)
        return seen

    def _require_hvp(self) -> None:
        for name in self._path():
            node = self.graph.nodes[name]
            if node.kind == "derived" and "hvp" not in node.provides:
                raise ObjectiveRefused(name, "missing hvp")

    def _mask_and_scale(self, gradient: Any) -> Any:
        import torch

        scaled = gradient * self._scales.to(dtype=gradient.dtype, device=gradient.device)
        if self.mask is None:
            return scaled
        return torch.where(self.mask.reshape(-1), scaled, torch.zeros_like(scaled))


def _ancestors(graph: Graph, name: str) -> list[str]:
    """Derived nodes upstream of ``name``, including it, when nothing is stale."""
    order: list[str] = []
    seen: set[str] = set()

    def walk(current: str) -> None:
        if current in seen or current not in graph.nodes:
            return
        seen.add(current)
        node = graph.nodes[current]
        for read in node.reads:
            walk(read)
        if node.kind == "derived":
            order.append(current)

    walk(name)
    return order


def _retain(graph: Graph, path: Sequence[str]) -> None:
    for name in path:
        node = graph.nodes.get(name)
        if node is None or node.cached is None or not getattr(node.cached, "requires_grad", False):
            continue
        if not node.cached.is_leaf:
            node.cached.retain_grad()


def _apply_adjoints(
    graph: Graph,
    registry: Registry,
    resolved: Sequence[ResolvedEntry],
    path: Sequence[str],
    gradient: Any,
) -> Any:
    slices = registry.pack(resolved).slices
    for name in path:
        node = graph.nodes[name]
        if node.adjoint is None or "atoms" not in node.produces or node.cached is None:
            continue
        atom_grad = getattr(node.cached, "grad", None)
        if atom_grad is None:
            continue
        mapped = node.adjoint(atom_grad).reshape(-1)
        for read in node.reads:
            slot = slices.get(read)
            if slot is None or mapped.numel() != (slot.stop - slot.start):
                continue
            gradient[slot] = mapped.to(dtype=gradient.dtype)
            break
    return gradient
