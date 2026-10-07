"""Torch geometry and ADP priors as restraint providers."""

from __future__ import annotations

from typing import Any

from phridge.contracts.manifest import PluginManifest, Slot, Space, Tier
from phridge.contracts.tensors import RObject
from phridge.ops import register_manifest
from phridge.r.restraints.torch_geometry.energy import _RestraintTables

GEOMETRY_MANIFEST = PluginManifest(
    name="torch_geometry",
    slot=Slot.restraint_provider,
    space=Space.R,
    tier=Tier.hybrid,
    dtype="float64",
    produces=["sites"],
)

ADP_MANIFEST = PluginManifest(
    name="adp_prior",
    slot=Slot.restraint_provider,
    space=Space.R,
    tier=Tier.hybrid,
    dtype="float64",
    produces=["u_iso", "u_star"],
)


def _zeros_like(record: RObject, sites: Any) -> RObject:
    import torch

    return RObject(
        sites_frac=sites,
        u_iso=torch.zeros_like(record.u_iso),
        u_star=torch.zeros_like(record.u_star),
        occupancy=torch.zeros_like(record.occupancy),
        fp=torch.zeros_like(record.fp),
        fdp=torch.zeros_like(record.fdp),
        scatterer_type=record.scatterer_type,
        uij_defined=record.uij_defined,
    )


class TorchGeometry:
    """Bond/angle/dihedral energy. ``sites_frac`` carries Cartesian coordinates."""

    def __init__(self, restraints: Any) -> None:
        self.restraints = restraints

    def covers(self) -> set[str]:
        return set(GEOMETRY_MANIFEST.produces)

    def energy(self, r: RObject) -> tuple[Any, RObject]:
        xyz = r.sites_frac.detach().clone().requires_grad_(True)
        tables = _RestraintTables.from_packed(self.restraints, device=str(xyz.device), dtype=xyz.dtype)
        value = tables.energy(xyz)
        (grad,) = __import__("torch").autograd.grad(value, xyz)
        return value.detach(), _zeros_like(r, grad.detach())


class AdpPriorProvider:
    def covers(self) -> set[str]:
        return set(ADP_MANIFEST.produces)


def register() -> None:
    register_manifest(GEOMETRY_MANIFEST, impl=TorchGeometry)
    register_manifest(ADP_MANIFEST, impl=AdpPriorProvider)
