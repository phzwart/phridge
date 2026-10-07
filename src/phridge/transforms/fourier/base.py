"""Fourier operator protocol. Tensors in, tensors out."""

from __future__ import annotations

from typing import Any, Protocol

from phridge.contracts.tensors import ComplexArray, RObject


class FourierOperator(Protocol):
    def forward(self, r: RObject, hkl: Any) -> ComplexArray:
        """F(h) from an atom record."""

    def adjoint(self, r: RObject, hkl: Any, d_values: Any) -> RObject:
        """Gradient of a scalar loss w.r.t. the atom record (VJP layout)."""

    def jvp(self, tangents: Any, params: Any = None) -> Any:
        """Optional directional derivative."""

    def gauss_newton_hvp(self, tangents: Any, curv_radial: Any, curv_tangential: Any, params: Any = None) -> Any:
        """Optional Gauss-Newton Hessian-vector product."""

    def gauss_newton_diagonal(self, curv_radial: Any, curv_tangential: Any, n_probes: int = 8, seed: int = 0, params: Any = None) -> Any:
        """Optional Gauss-Newton diagonal estimate."""

    def gauss_newton_blocks(self, curv_radial: Any, curv_tangential: Any, params: Any = None) -> Any:
        """Optional per-atom Gauss-Newton blocks."""


def robject_from_gradients(record: RObject, gradients: dict[str, Any]) -> RObject:
    """Pack an engine gradient dict into the same record layout."""
    import torch

    def _like(name: str, like: Any) -> Any:
        return torch.as_tensor(gradients[name], dtype=like.dtype, device=like.device)

    return RObject(
        sites_frac=_like("site_frac", record.sites_frac),
        u_iso=_like("u_iso", record.u_iso),
        u_star=_like("u_star", record.u_star),
        occupancy=_like("occupancy", record.occupancy),
        fp=_like("fp", record.fp),
        fdp=_like("fdp", record.fdp),
        scatterer_type=record.scatterer_type,
        uij_defined=record.uij_defined,
    )
