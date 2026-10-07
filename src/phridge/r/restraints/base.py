"""Restraint provider protocol."""

from __future__ import annotations

from typing import Any, Protocol

from phridge.contracts.tensors import RObject


class RestraintProvider(Protocol):
    def energy(self, r: RObject) -> tuple[Any, RObject]:
        """Scalar energy and its gradient in the same record layout."""

    def hvp(self, r: RObject, tangents: RObject) -> RObject:
        """Optional Hessian-vector product."""

    def block_diag(self, r: RObject) -> Any:
        """Optional per-atom blocks."""

    def covers(self) -> set[str]:
        """Parameter kinds this provider differentiates."""
