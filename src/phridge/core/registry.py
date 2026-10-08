"""Parameter blocks and the pack/unpack boundary optimizers see."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, Optional

from phridge.contracts.manifest import Space

TransformName = Literal["identity", "log", "logit"]


class PackEntry(NamedTuple):
    """One block, optionally restricted to a boolean view, as packed into a vector."""

    block_id: str
    view_mask: Any = None


class Packed(NamedTuple):
    """Unconstrained vector plus the bookkeeping ``unpack`` needs."""

    vector: Any
    slices: dict[str, slice]
    bounds: dict[str, Optional[tuple[Any, Any]]]
    transform_ops: dict[str, str]
    scales: Any


@dataclass
class Block:
    """One named tensor owned by a component."""

    id: str
    component: str
    space: Space
    shape: tuple[int, ...]
    bounds: Optional[tuple[Any, Any]]
    transform: TransformName
    frozen: bool
    value_version: int
    structural_version: int
    value: Any
    structural_epoch: int = 0


class Registry:
    """Blocks addressed by id, with value and structural versions."""

    def __init__(self) -> None:
        self._blocks: dict[str, Block] = {}
        self._epoch = 0

    def register(self, block: Block) -> None:
        """Refuse a second block with the same id."""
        if block.id in self._blocks:
            raise ValueError(f"duplicate block id {block.id!r}")
        if tuple(block.value.shape) != tuple(block.shape):
            raise ValueError(f"{block.id} value shape {tuple(block.value.shape)} != {block.shape}")
        self._blocks[block.id] = block

    def block(self, id: str) -> Block:
        try:
            return self._blocks[id]
        except KeyError as exc:
            raise KeyError(id) from exc

    def set_value(self, id: str, value: Any) -> None:
        """Replace a block's tensor. The shape must match; ``value_version`` bumps."""
        block = self.block(id)
        if tuple(value.shape) != tuple(block.shape):
            raise ValueError(f"{id} shape {tuple(value.shape)} != {block.shape}")
        block.value = value
        block.value_version += 1

    def reshape(self, id: str, new_shape: tuple[int, ...], new_value: Any) -> None:
        """Replace shape and value. Bumps this block's structural version only."""
        block = self.block(id)
        shape = tuple(new_shape)
        if tuple(new_value.shape) != shape:
            raise ValueError(f"{id} new value shape {tuple(new_value.shape)} != {shape}")
        block.shape = shape
        block.value = new_value
        block.structural_version += 1
        block.structural_epoch = self._epoch
        block.value_version += 1

    def freeze(self, id: str) -> None:
        self.block(id).frozen = True

    def unfreeze(self, id: str) -> None:
        self.block(id).frozen = False

    def epoch(self) -> int:
        return self._epoch

    def bump_epoch(self) -> None:
        self._epoch += 1

    def pack(self, entries: list[PackEntry]) -> Packed:
        """Free blocks become one unconstrained vector. Frozen blocks are left out."""
        import torch

        pieces: list[Any] = []
        scale_pieces: list[Any] = []
        slices: dict[str, slice] = {}
        bounds: dict[str, Optional[tuple[Any, Any]]] = {}
        transform_ops: dict[str, str] = {}
        cursor = 0
        for entry in entries:
            block = self.block(entry.block_id)
            if block.frozen:
                continue
            flat = _view(block.value, entry.view_mask)
            transformed = _forward(block.transform, flat, block.bounds)
            n = int(transformed.numel())
            slices[block.id] = slice(cursor, cursor + n)
            cursor += n
            pieces.append(transformed.reshape(-1))
            scale = float(getattr(entry, "scale", 1.0))
            scale_pieces.append(torch.full((n,), scale, dtype=transformed.dtype, device=transformed.device))
            bounds[block.id] = block.bounds
            transform_ops[block.id] = block.transform
        if pieces:
            vector = torch.cat(pieces)
            scales = torch.cat(scale_pieces)
        else:
            vector = torch.empty(0, dtype=torch.float64)
            scales = torch.empty(0, dtype=torch.float64)
        return Packed(vector, slices, bounds, transform_ops, scales)

    def unpack(self, packed_vector: Any, entries: list[PackEntry]) -> None:
        """Write the inverse transform back. Only touched blocks bump ``value_version``."""
        layout = _layout(self, entries)
        for entry in entries:
            block = self.block(entry.block_id)
            if block.id not in layout:
                continue
            sl = layout[block.id]
            chunk = packed_vector[sl]
            raw = _inverse(block.transform, chunk, block.bounds)
            if entry.view_mask is None:
                block.value = raw.reshape(block.shape).to(dtype=block.value.dtype)
            else:
                updated = block.value.clone()
                updated.reshape(-1)[entry.view_mask.reshape(-1)] = raw.to(dtype=updated.dtype)
                block.value = updated
            block.value_version += 1


def _layout(registry: Registry, entries: list[PackEntry]) -> dict[str, slice]:
    cursor = 0
    out: dict[str, slice] = {}
    for entry in entries:
        block = registry.block(entry.block_id)
        if block.frozen:
            continue
        n = int(_view(block.value, entry.view_mask).numel())
        out[block.id] = slice(cursor, cursor + n)
        cursor += n
    return out


def _view(value: Any, mask: Any) -> Any:
    flat = value.reshape(-1)
    if mask is None:
        return flat
    return flat[mask.reshape(-1)]


def _forward(transform: str, flat: Any, bounds: Optional[tuple[Any, Any]]) -> Any:
    import torch

    if transform == "identity":
        return flat
    if transform == "log":
        return torch.log(flat)
    if transform == "logit":
        lo, hi = _bounds(flat, bounds)
        z = (flat - lo) / (hi - lo)
        return torch.log(z / (1.0 - z))
    raise ValueError(f"unknown transform {transform!r}")


def _inverse(transform: str, flat: Any, bounds: Optional[tuple[Any, Any]]) -> Any:
    import torch

    if transform == "identity":
        return flat
    if transform == "log":
        return torch.exp(flat)
    if transform == "logit":
        lo, hi = _bounds(flat, bounds)
        return lo + (hi - lo) * torch.sigmoid(flat)
    raise ValueError(f"unknown transform {transform!r}")


def _bounds(flat: Any, bounds: Optional[tuple[Any, Any]]) -> tuple[Any, Any]:
    import torch

    if bounds is None:
        return torch.zeros((), dtype=flat.dtype, device=flat.device), torch.ones((), dtype=flat.dtype, device=flat.device)
    return bounds
