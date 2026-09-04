"""Geometry restraint converters and Phenix-facing RemoteGeometry manager."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from phridge.client.convert_geometry import model_geometry, restraints_from_cctbx, restraints_to_proxies
from phridge.client.convert_xtal import hierarchy_from_cctbx, sites_cart_from_cctbx
from phridge.packing_geometry import PackedRestraints
from phridge.packing_xtal import PackedCartesian, PackedHierarchy

__all__ = [
    "RemoteGeometry",
    "model_geometry",
    "restraints_from_cctbx",
    "restraints_to_proxies",
]


class RemoteGeometry:
    """Phenix-facing geometry minimizer. Packs cctbx → Redis; worker runs torch optim.

    From Phenix this looks like a local call that blocks until the phridge
    worker finishes torch LBFGS / Adam / SGD on packed restraints::

        geo = RemoteGeometry(bridge, hierarchy, restraints_manager)
        hierarchy_out = geo.minimize(max_iterations=100, optimizer="lbfgs")
    """

    def __init__(
        self,
        bridge: Any,
        hierarchy: Any,
        restraints: Any,
        *,
        selection: Optional[Any] = None,
    ) -> None:
        if selection is not None:
            raise NotImplementedError("RemoteGeometry selection is not supported in v1")
        self.bridge = bridge
        self._packed_hier = _as_packed_hierarchy(hierarchy)
        self._packed_restr = _as_packed_restraints(restraints, n_sites=self._packed_hier.meta.n_atoms)
        self._header = model_geometry(hierarchy=self._packed_hier, restraints=self._packed_restr)
        self.last_target: Optional[dict[str, Any]] = None

    @property
    def n_sites(self) -> int:
        return int(self._packed_restr.n_sites)

    def energy(self, sites_cart: Optional[Any] = None) -> float:
        """Remote energy evaluation (no minimization steps)."""
        out = self._call(sites_cart=sites_cart, max_iterations=0, optimizer="lbfgs")
        return float(out["target"]["before"])

    def minimize(
        self,
        *,
        max_iterations: int = 100,
        optimizer: str = "lbfgs",
        lr: Optional[float] = None,
        sites_cart: Optional[Any] = None,
        update_hierarchy: bool = True,
    ) -> Any:
        """Run remote torch minimization; return a cctbx hierarchy (or packed sites).

        Blocks on ``bridge.call("geometry_minimize", ...)`` until the worker
        reports done. ``optimizer`` is ``"lbfgs"``, ``"adam"``, or ``"sgd"``.
        """
        out = self._call(
            sites_cart=sites_cart,
            max_iterations=max_iterations,
            optimizer=optimizer,
            lr=lr,
        )
        sites = out["sites"]
        if isinstance(sites, PackedCartesian):
            xyz = sites.xyz
        else:
            xyz = np.asarray(list(sites), dtype=np.float64)
        self._packed_hier = PackedHierarchy(
            xyz=np.asarray(xyz, dtype=np.float64),
            occupancy=self._packed_hier.occupancy,
            b_iso=self._packed_hier.b_iso,
            atoms=self._packed_hier.meta.atoms,
            crystal=self._packed_hier.meta.crystal,
            frame=self._packed_hier.meta.frame,
            u_cart=self._packed_hier.u_cart,
            uij_defined=self._packed_hier.uij_defined,
        )
        if not update_hierarchy:
            return sites
        from phridge.client.convert_xtal import hierarchy_to_cctbx

        return hierarchy_to_cctbx(self._packed_hier)

    def _call(
        self,
        *,
        sites_cart: Optional[Any],
        max_iterations: int,
        optimizer: str,
        lr: Optional[float] = None,
    ) -> dict[str, Any]:
        sites = _sites_for_call(sites_cart, self._packed_hier)
        params: dict[str, Any] = {
            "max_iterations": int(max_iterations),
            "optimizer": str(optimizer),
        }
        if lr is not None:
            params["lr"] = float(lr)
        result = self.bridge.call(
            "geometry_minimize",
            sites=sites,
            restraints=self._packed_restr,
            params=params,
        )
        if not isinstance(result, dict) or "sites" not in result or "target" not in result:
            raise RuntimeError("geometry_minimize did not return sites and target")
        self.last_target = dict(result["target"])
        return result


def _as_packed_hierarchy(hierarchy: Any) -> PackedHierarchy:
    if isinstance(hierarchy, PackedHierarchy):
        return hierarchy
    return hierarchy_from_cctbx(hierarchy)


def _as_packed_restraints(restraints: Any, *, n_sites: int) -> PackedRestraints:
    if isinstance(restraints, PackedRestraints):
        if restraints.n_sites != n_sites:
            raise ValueError(f"restraints n_sites={restraints.n_sites} != hierarchy n_atoms={n_sites}")
        return restraints
    return restraints_from_cctbx(restraints, n_sites=n_sites)


def _sites_for_call(sites_cart: Optional[Any], packed_hier: PackedHierarchy) -> PackedCartesian:
    if sites_cart is None:
        return PackedCartesian(packed_hier.xyz, crystal=packed_hier.meta.crystal)
    if isinstance(sites_cart, PackedCartesian):
        return sites_cart
    if isinstance(sites_cart, np.ndarray):
        return PackedCartesian(sites_cart, crystal=packed_hier.meta.crystal)
    return sites_cart_from_cctbx(sites_cart, packed_hier.meta.crystal)
