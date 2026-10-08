"""Builders propose an ``RChange``. They never apply it."""

from __future__ import annotations

from typing import Any, Optional, Protocol

from phridge.contracts.tensors import RObject


class RChange(tuple):
    """A structural proposal. ``add`` is new atoms, ``remove`` a mask, ``reinit`` replacement tensors."""

    add: Optional[RObject]
    remove: Any
    reinit: dict[str, Any]

    def __new__(cls, add: Optional[RObject], remove: Any, reinit: dict[str, Any]) -> RChange:
        return tuple.__new__(cls, (add, remove, dict(reinit)))

    @property
    def add(self) -> Optional[RObject]:
        return self[0]

    @property
    def remove(self) -> Any:
        return self[1]

    @property
    def reinit(self) -> dict[str, Any]:
        return self[2]


class Builder(Protocol):
    """Propose a change from maps and a selection. Applying it is the controller's job."""

    def propose(self, maps: dict[str, Any], stats: dict[str, Any], selection: Any) -> RChange:
        """Return an ``RChange`` and leave ``maps`` untouched."""
