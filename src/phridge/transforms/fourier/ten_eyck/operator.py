"""Ten Eyck engine as a FourierOperator. No new mathematics."""

from __future__ import annotations

from typing import Any

from phridge.contracts.manifest import PluginManifest, Slot, Space, Tier
from phridge.contracts.tensors import ComplexArray, RObject
from phridge.ops import register_manifest
from phridge.transforms.fourier.base import robject_from_gradients
from phridge.transforms.fourier.ten_eyck.engine import StructureFactorEngine

MANIFEST = PluginManifest(
    name="ten_eyck",
    slot=Slot.fourier,
    space=Space.transform,
    from_space=Space.R,
    to_space=Space.Q,
    tier=Tier.native,
    dtype="float64",
    provides=["adjoint", "jvp", "hvp", "block_diag"],
)


class TenEyckFourier:
    """Calls :class:`StructureFactorEngine` with the tensors it already takes."""

    def __init__(self, engine: StructureFactorEngine) -> None:
        self.engine = engine

    def forward(self, r: RObject, hkl: Any) -> ComplexArray:
        values = self.engine.f_calc(r.sites_frac, r.occupancy, r.u_iso, r.u_star, r.fp, r.fdp)
        return ComplexArray(hkl=hkl, values=values)

    def adjoint(self, r: RObject, hkl: Any, d_values: Any) -> RObject:
        import numpy as np

        gradient = self.engine.gradients(np.asarray(d_values.detach().cpu().numpy(), dtype=np.complex128))
        return robject_from_gradients(r, gradient)

    def jvp(self, tangents: Any, params: Any = None) -> Any:
        return self.engine.jvp(tangents, params=params)

    def gauss_newton_hvp(self, tangents: Any, curv_radial: Any, curv_tangential: Any, params: Any = None) -> Any:
        return self.engine.gauss_newton_hvp(tangents, curv_radial, curv_tangential, params=params)

    def gauss_newton_diagonal(self, curv_radial: Any, curv_tangential: Any, n_probes: int = 8, seed: int = 0, params: Any = None) -> Any:
        return self.engine.gauss_newton_diagonal(curv_radial, curv_tangential, n_probes=n_probes, seed=seed, params=params)

    def gauss_newton_blocks(self, curv_radial: Any, curv_tangential: Any, params: Any = None) -> Any:
        return self.engine.gauss_newton_blocks(curv_radial, curv_tangential, params=params)


def register() -> None:
    register_manifest(MANIFEST, impl=TenEyckFourier)
