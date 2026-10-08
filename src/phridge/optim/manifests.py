"""Optimizer manifests. ``needs`` is the capability list ``step`` refuses without."""

from __future__ import annotations

from phridge.contracts.manifest import PluginManifest, Slot, Space
from phridge.ops import register_manifest
from phridge.optim.gauss_newton import GaussNewton
from phridge.optim.grid import GridSearch
from phridge.optim.lbfgs import LBFGS
from phridge.optim.newton_cg import NewtonCG

MANIFESTS: dict[str, PluginManifest] = {
    "lbfgs": PluginManifest(name="lbfgs", slot=Slot.optimizer, space=Space.R, needs=["grad"]),
    "newton_cg": PluginManifest(
        name="newton_cg", slot=Slot.optimizer, space=Space.R, needs=["grad", "hvp", "preconditioner"]
    ),
    "gauss_newton": PluginManifest(
        name="gauss_newton", slot=Slot.optimizer, space=Space.R, needs=["grad", "hvp", "preconditioner"]
    ),
    "grid": PluginManifest(name="grid", slot=Slot.optimizer, space=Space.R, needs=[]),
}

_IMPLS = {"lbfgs": LBFGS, "newton_cg": NewtonCG, "gauss_newton": GaussNewton, "grid": GridSearch}


def register() -> None:
    """Register the four optimizer manifests. Safe to call twice."""
    for name, manifest in MANIFESTS.items():
        register_manifest(manifest, impl=_IMPLS[name])
