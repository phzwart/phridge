"""Geometry restraint converters and Phenix / torch-facing helpers."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from phridge.client.convert_geometry import model_geometry, restraints_from_cctbx, restraints_to_proxies
from phridge.client.convert_xtal import hierarchy_from_cctbx, sites_cart_from_cctbx
from phridge.packing_geometry import PackedRestraints
from phridge.packing_xtal import PackedCartesian, PackedHierarchy

__all__ = [
    "RemoteGeometry",
    "RemoteRestraintBuilder",
    "model_geometry",
    "restraints_from_cctbx",
    "restraints_to_proxies",
]


class RemoteRestraintBuilder:
    """Torch-facing helper: build packed restraints via the CCTBX worker.

    The calling process need not import cctbx. Jobs go to the ``cctbx``
    Redis stream (``phridge-worker --runtime cctbx`` / ``phridge-cctbx-worker``)::

        builder = RemoteRestraintBuilder(bridge)
        out = builder.build(pdb_string=pdb)
        geo = RemoteGeometry(bridge, out["hierarchy"], out["restraints"])
    """

    def __init__(self, bridge: Any) -> None:
        self.bridge = bridge

    def build(
        self,
        *,
        pdb_string: Optional[str] = None,
        hierarchy: Any = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """Return ``{"hierarchy", "restraints"}`` as packed types (no cctbx)."""
        params: dict[str, Any] = dict(extra)
        if pdb_string is not None:
            params["pdb_string"] = pdb_string
        kwargs: dict[str, Any] = {}
        if params:
            kwargs["params"] = params
        if hierarchy is not None:
            kwargs["hierarchy"] = hierarchy
        if not kwargs:
            raise TypeError("build() needs pdb_string and/or hierarchy")
        out = self.bridge.call("build_geometry_restraints", **kwargs)
        if not isinstance(out, dict) or "hierarchy" not in out or "restraints" not in out:
            raise RuntimeError("build_geometry_restraints did not return hierarchy and restraints")
        return out


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
        self.last_target = None  # type: Optional[dict[str, Any]]

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
        lr_min: Optional[float] = None,
        schedule: Optional[str] = None,
        momentum: float = 0.0,
        sites_cart: Optional[Any] = None,
        update_hierarchy: bool = True,
    ) -> Any:
        """Run remote torch minimization; return a cctbx hierarchy (or packed sites).

        Blocks on ``bridge.call("geometry_minimize", ...)`` until the worker
        reports done. ``optimizer`` is ``"lbfgs"``, ``"adam"``, ``"adamw"``, or ``"sgd"``.
        First-order methods accept ``schedule`` (``none`` / ``cosine`` / ``triangular``),
        ``lr_min``, and SGD ``momentum``.

        With packed hierarchy inputs (torch driver), pass ``update_hierarchy=False``
        to get :class:`PackedCartesian` sites without importing cctbx.
        """
        out = self._call(
            sites_cart=sites_cart,
            max_iterations=max_iterations,
            optimizer=optimizer,
            lr=lr,
            lr_min=lr_min,
            schedule=schedule,
            momentum=momentum,
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
        from phridge.client.convert import has_cctbx
        from phridge.client.convert_xtal import hierarchy_to_cctbx

        if not has_cctbx():
            return sites
        return hierarchy_to_cctbx(self._packed_hier)

    def _call(
        self,
        *,
        sites_cart: Optional[Any],
        max_iterations: int,
        optimizer: str,
        lr: Optional[float] = None,
        lr_min: Optional[float] = None,
        schedule: Optional[str] = None,
        momentum: float = 0.0,
    ) -> dict[str, Any]:
        sites = _sites_for_call(sites_cart, self._packed_hier)
        params: dict[str, Any] = {
            "max_iterations": int(max_iterations),
            "optimizer": str(optimizer),
            "momentum": float(momentum),
        }
        if lr is not None:
            params["lr"] = float(lr)
        if lr_min is not None:
            params["lr_min"] = float(lr_min)
        if schedule is not None:
            params["schedule"] = str(schedule)
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
