"""Pull-based dependency graph. A node recomputes only when a version says it must."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Optional, Sequence

from phridge.contracts.manifest import PluginManifest
from phridge.core.moveset import MoveSet
from phridge.core.receipts import Receipt, ReceiptLog
from phridge.core.registry import Registry

Inputs = Mapping[str, Any]
Compute = Callable[[Inputs], Any]


@dataclass
class Node:
    """A registry block or a derived value with a compute function."""

    name: str
    kind: Literal["block", "derived"]
    reads: list[str] = field(default_factory=list)
    compute: Optional[Compute] = None
    value_version: int = 0
    structural_version: int = 0
    last_input_versions: dict[str, tuple[int, int]] = field(default_factory=dict)
    cached: Any = None
    shape_cache: dict[str, Any] = field(default_factory=dict)
    seen_epoch: int = -1
    produces: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()
    partitions_consulted: tuple[str, ...] = ()
    adjoint: Optional[Callable[[Any], Any]] = None


class Graph:
    """Versions on blocks and on the registry epoch decide what ``get`` recomputes."""

    def __init__(self, registry: Registry) -> None:
        self.registry = registry
        self.nodes: dict[str, Node] = {}
        self.receipts = ReceiptLog()

    def add(self, node: Node) -> None:
        if node.name in self.nodes:
            raise ValueError(f"duplicate node {node.name!r}")
        self.nodes[node.name] = node

    @classmethod
    def from_manifests(cls, manifests: Sequence[PluginManifest], registry: Registry) -> Graph:
        """Edges are the union of each manifest's ``reads``."""
        graph = cls(registry)
        for manifest in manifests:
            graph.add(
                Node(
                    name=manifest.name,
                    kind="derived",
                    reads=list(manifest.reads),
                    produces=tuple(manifest.produces),
                    provides=tuple(manifest.provides),
                )
            )
        for manifest in manifests:
            for read in manifest.reads:
                graph._ensure_block(read)
        return graph

    def get(self, name: str) -> Any:
        """Return ``name``, recomputing only the stale chain. One receipt per recompute."""
        node = self.nodes[name]
        if node.kind == "block":
            return self._load_block(name)
        for item in self.cost(name):
            self._recompute(item)
        return self.nodes[name].cached

    def stale(self, name: str) -> list[str]:
        """Derived nodes in the upstream chain whose versions are out of date."""
        return [item for item in self._topo(name) if self._is_stale(item)]

    def cost(self, name: str) -> list[str]:
        """The chain ``get`` would recompute, dependencies first."""
        return self.stale(name)

    def readers(self, name: str) -> set[str]:
        """Nodes that list ``name`` in ``reads``."""
        return {node.name for node in self.nodes.values() if name in node.reads}

    def affected(self, blocks_or_moveset: MoveSet | Sequence[str] | str) -> set[str]:
        """Transitive readers of the given blocks or move-set entries."""
        seeds = _seeds(blocks_or_moveset)
        seen: set[str] = set()
        frontier = list(seeds)
        while frontier:
            current = frontier.pop()
            for reader in sorted(self.readers(current)):
                if reader in seen:
                    continue
                seen.add(reader)
                frontier.append(reader)
        return seen

    def structural_bump(self, name: str) -> None:
        """Drop ``shape_cache`` on downstream nodes only."""
        for downstream in self.affected(name):
            self.nodes[downstream].shape_cache.clear()

    def rebuild(self) -> None:
        """Bump the registry epoch so the next ``get`` recomputes every derived node."""
        self.registry.bump_epoch()

    def _ensure_block(self, name: str) -> None:
        if name in self.nodes:
            return
        try:
            self.registry.block(name)
        except KeyError:
            return
        self.add(Node(name=name, kind="block"))

    def _topo(self, name: str) -> list[str]:
        order: list[str] = []
        seen: set[str] = set()

        def walk(current: str) -> None:
            if current in seen:
                return
            seen.add(current)
            node = self.nodes[current]
            for read in node.reads:
                self._ensure_block(read)
                if read in self.nodes:
                    walk(read)
            if node.kind == "derived":
                order.append(current)

        walk(name)
        return order

    def _is_stale(self, name: str) -> bool:
        node = self.nodes[name]
        if node.kind != "derived":
            return False
        if node.cached is None or node.seen_epoch != self.registry.epoch():
            return True
        for read in node.reads:
            if self._version(read) != node.last_input_versions.get(read):
                return True
            if read in self.nodes and self._is_stale(read):
                return True
        return False

    def _version(self, name: str) -> tuple[int, int]:
        self._ensure_block(name)
        node = self.nodes.get(name)
        if node is not None and node.kind == "block":
            block = self.registry.block(name)
            return (block.value_version, block.structural_version)
        if node is None:
            return (0, 0)
        return (node.value_version, node.structural_version)

    def _load_block(self, name: str) -> Any:
        self._ensure_block(name)
        block = self.registry.block(name)
        node = self.nodes[name]
        node.cached = block.value
        node.value_version = block.value_version
        node.structural_version = block.structural_version
        return block.value

    def _recompute(self, name: str) -> None:
        import time

        node = self.nodes[name]
        if node.compute is None:
            raise RuntimeError(f"{name} has no compute")
        inputs: dict[str, Any] = {}
        versions: dict[str, tuple[int, int]] = {}
        for read in node.reads:
            if read in self.nodes and self.nodes[read].kind == "block":
                inputs[read] = self._load_block(read)
            else:
                inputs[read] = self.nodes[read].cached
            versions[read] = self._version(read)
        started = time.perf_counter()
        node.cached = node.compute(inputs)
        elapsed = time.perf_counter() - started
        node.value_version += 1
        node.last_input_versions = versions
        node.seen_epoch = self.registry.epoch()
        self.receipts.append(
            Receipt(
                action="recompute",
                node=name,
                inputs_versions=dict(versions),
                output_version=(node.value_version, node.structural_version),
                wall_time=elapsed,
                partitions_consulted=node.partitions_consulted,
            )
        )


def _seeds(blocks_or_moveset: MoveSet | Sequence[str] | str) -> set[str]:
    if isinstance(blocks_or_moveset, MoveSet):
        return {entry.block for entry in blocks_or_moveset.entries}
    if isinstance(blocks_or_moveset, str):
        return {blocks_or_moveset}
    return set(blocks_or_moveset)
