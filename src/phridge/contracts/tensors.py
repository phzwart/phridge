"""Tensor records that cross the sandbox boundary. Never a cctbx object."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from torch import Tensor
else:
    Tensor = Any


def _agree(tensors: list[Any]) -> None:
    axes = [int(tensor.shape[0]) for tensor in tensors if tensor is not None]
    if len(set(axes)) > 1:
        raise ValueError(f"leading axes disagree: {axes}")


class RObject(NamedTuple):
    """Named tensors describing an atom list. Never an xray_structure."""

    sites_frac: Tensor
    u_iso: Tensor
    u_star: Tensor
    occupancy: Tensor
    fp: Tensor
    fdp: Tensor
    scatterer_type: Tensor
    uij_defined: Tensor

    def validate(self) -> None:
        _agree(list(self))


class DensityObject(NamedTuple):
    """Continuum component on a grid."""

    values: Tensor
    origin: Tensor
    spacing: Tensor
    cell: Tensor

    def validate(self) -> None:
        _agree(list(self))


class ComplexArray(NamedTuple):
    """Complex coefficients over a Miller table the sandbox owns."""

    hkl: Tensor
    values: Tensor

    def validate(self) -> None:
        _agree([self.hkl, self.values])


class Predictive(NamedTuple):
    """Named predictive fields. ``require`` names any field that is absent."""

    fields: dict[str, Tensor]

    def has(self, name: str) -> bool:
        return name in self.fields

    def require(self, *names: str) -> None:
        missing = [name for name in names if name not in self.fields]
        if missing:
            raise KeyError(", ".join(missing))

    def validate(self) -> None:
        _agree(list(self.fields.values()))


class Observed(NamedTuple):
    """Indexed observations, their data, and boolean partitions."""

    index: dict[str, Tensor]
    data: dict[str, Tensor]
    partitions: dict[str, Tensor]

    def validate(self) -> None:
        _agree([*self.index.values(), *self.data.values(), *self.partitions.values()])
