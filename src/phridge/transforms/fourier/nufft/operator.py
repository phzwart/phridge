"""NUFFT engine as a FourierOperator. No new mathematics."""

from __future__ import annotations

from typing import Any

from phridge.contracts.manifest import PluginManifest, Slot, Space, Tier
from phridge.contracts.tensors import ComplexArray, RObject
from phridge.ops import register_manifest
from phridge.transforms.fourier.base import robject_from_gradients
from phridge.transforms.fourier.nufft.nufft_engine import NufftStructureFactorEngine

MANIFEST = PluginManifest(
    name="nufft",
    slot=Slot.fourier,
    space=Space.transform,
    from_space=Space.R,
    to_space=Space.Q,
    tier=Tier.native,
    dtype="float64",
    provides=["adjoint"],
)


class NufftFourier:
    """Calls :class:`NufftStructureFactorEngine` with the tensors it already takes."""

    def __init__(self, engine: NufftStructureFactorEngine) -> None:
        self.engine = engine

    def forward(self, r: RObject, hkl: Any) -> ComplexArray:
        values = self.engine.f_calc(r.sites_frac, r.occupancy, r.u_iso, r.u_star, r.fp, r.fdp)
        return ComplexArray(hkl=hkl, values=values)

    def adjoint(self, r: RObject, hkl: Any, d_values: Any) -> RObject:
        import numpy as np

        gradient = self.engine.gradients(np.asarray(d_values.detach().cpu().numpy(), dtype=np.complex128))
        return robject_from_gradients(r, gradient)


def register() -> None:
    register_manifest(MANIFEST, impl=NufftFourier)
