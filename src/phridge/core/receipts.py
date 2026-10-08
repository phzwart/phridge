"""Append-only record of graph recomputes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Receipt:
    """One recompute. The log only grows."""

    action: str
    node: str
    inputs_versions: dict[str, tuple[int, int]]
    output_version: tuple[int, int]
    wall_time: float
    partitions_consulted: tuple[str, ...] = ()


class ReceiptLog:
    """Append-only list of receipts."""

    def __init__(self) -> None:
        self._items: list[Receipt] = []

    def append(self, receipt: Receipt) -> None:
        self._items.append(receipt)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> object:
        return iter(self._items)

    def __getitem__(self, index: int) -> Receipt:
        return self._items[index]

    def last(self) -> Optional[Receipt]:
        if not self._items:
            return None
        return self._items[-1]
