"""Move sets: which parameters may move, and the refusals that keep that honest."""

from __future__ import annotations

from typing import Any, Callable, Mapping, NamedTuple, Optional

from pydantic import BaseModel, ConfigDict

from phridge.core.registry import Registry


class View(BaseModel):
    """How an entry looks at a block: a selection string, indices, or a predicate."""

    model_config = ConfigDict(extra="forbid")

    block: str
    selection: Optional[str] = None
    index: Optional[list[int]] = None
    predicate: Optional[str] = None


class Entry(BaseModel):
    """One block in a move set, with optional view, bounds, scale, and freeze."""

    model_config = ConfigDict(extra="forbid")

    block: str
    view: Optional[View] = None
    bounds: Optional[tuple[float, float]] = None
    scale: float = 1.0
    frozen: bool = False


class MoveSet(BaseModel):
    """A named set of entries the optimizer is allowed to touch."""

    model_config = ConfigDict(extra="forbid")

    name: str
    entries: list[Entry]
    coupled: list[list[str]] = []
    ready_epoch: Optional[int] = None


class ResolvedEntry(NamedTuple):
    """An entry after its view has been turned into a mask. ``pack`` reads this."""

    block_id: str
    view_mask: Any
    scale: float
    bounds: Optional[tuple[float, float]]


class MoveSetRefused(Exception):
    """``validate`` rejected one entry."""

    def __init__(self, entry: str, reason: str) -> None:
        self.entry = entry
        self.reason = reason
        super().__init__(f"{entry}: {reason}")


def resolve(ms: MoveSet, registry: Registry, selector: Callable[[str], Any]) -> list[ResolvedEntry]:
    """Turn each view into a mask. ``selection`` and ``predicate`` go to ``selector``."""
    resolved: list[ResolvedEntry] = []
    for entry in ms.entries:
        block = registry.block(entry.block)
        resolved.append(
            ResolvedEntry(
                block_id=entry.block,
                view_mask=_mask(entry, block.value, selector),
                scale=entry.scale,
                bounds=entry.bounds,
            )
        )
    return resolved


def validate(
    ms: MoveSet,
    registry: Registry,
    graph_capabilities: Mapping[str, set[str]],
    *,
    ready_epoch: Optional[int] = None,
    generator_outputs: Optional[Mapping[str, set[str]]] = None,
) -> None:
    """Refuse a move set that is frozen, stale, overlapping a generator, or missing an adjoint."""
    ready = ms.ready_epoch if ready_epoch is None else ready_epoch
    outputs = {name: set(produced) for name, produced in (generator_outputs or {}).items()}
    chosen = {entry.block for entry in ms.entries}
    for entry in ms.entries:
        block = registry.block(entry.block)
        if entry.frozen or block.frozen:
            raise MoveSetRefused(entry.block, "frozen")
        if ready is not None and block.structural_epoch > ready:
            raise MoveSetRefused(entry.block, "structural epoch is newer than ready_epoch")
        produced = outputs.get(entry.block, set())
        overlap = produced & chosen
        if overlap:
            raise MoveSetRefused(entry.block, "generator overlaps its outputs")
        if entry.block in outputs and "adjoint" not in graph_capabilities.get(entry.block, set()):
            raise MoveSetRefused(entry.block, "missing adjoint")


def _mask(entry: Entry, value: Any, selector: Callable[[str], Any]) -> Any:
    import torch

    view = entry.view
    if view is None:
        return None
    mask = torch.ones(value.shape, dtype=torch.bool, device=value.device)
    used = False
    if view.index is not None:
        flat = torch.zeros(value.numel(), dtype=torch.bool, device=value.device)
        flat[list(view.index)] = True
        mask = mask & flat.reshape(value.shape)
        used = True
    if view.predicate is not None:
        mask = mask & selector(view.predicate).to(device=value.device).reshape(value.shape)
        used = True
    if view.selection is not None:
        mask = mask & selector(view.selection).to(device=value.device).reshape(value.shape)
        used = True
    if not used:
        return None
    return mask
