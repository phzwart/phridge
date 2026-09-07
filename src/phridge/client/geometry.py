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
        # out["restraints_handle"] keeps the live GRM on the CCTBX worker
        geo = RemoteGeometry(
            bridge, out["hierarchy"], out["restraints"],
            restraints_handle=out["restraints_handle"],
        )
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
        """Return packed hierarchy/restraints plus ``restraints_handle`` (CCTBX)."""
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
    """Phenix / torch-facing geometry helper over phridge.

    Torch minimize (``geometry_minimize``) uses packed restraints on the torch
    worker. CCTBX energy+gradients (``energy_and_gradients``) use a live GRM
    kept on the CCTBX worker behind ``restraints_handle``.
    """

    def __init__(
        self,
        bridge: Any,
        hierarchy: Any,
        restraints: Any,
        *,
        selection: Optional[Any] = None,
        restraints_handle: Optional[str] = None,
    ) -> None:
        if selection is not None:
            raise NotImplementedError("RemoteGeometry selection is not supported in v1")
        self.bridge = bridge
        self._packed_hier = _as_packed_hierarchy(hierarchy)
        self._packed_restr = _as_packed_restraints(restraints, n_sites=self._packed_hier.meta.n_atoms)
        self._header = model_geometry(hierarchy=self._packed_hier, restraints=self._packed_restr)
        self.restraints_handle = restraints_handle
        self.last_target: Optional[dict[str, Any]] = None
        self.last_gn_solve_stats: Optional[dict[str, Any]] = None

    @classmethod
    def from_build(cls, bridge: Any, built: dict[str, Any]) -> "RemoteGeometry":
        """Construct from ``RemoteRestraintBuilder.build`` output (includes handle)."""
        handle = built.get("restraints_handle")
        if handle is not None:
            handle = str(handle)
        return cls(
            bridge,
            built["hierarchy"],
            built["restraints"],
            restraints_handle=handle,
        )

    @property
    def n_sites(self) -> int:
        return int(self._packed_restr.n_sites)

    def energy(self, sites_cart: Optional[Any] = None) -> float:
        """Remote energy evaluation (no minimization steps).

        Prefer ``energy_and_gradients`` when a ``restraints_handle`` is available
        (CCTBX ``energies_sites``). Otherwise falls back to torch packed energy
        via ``geometry_minimize`` with ``max_iterations=0``.
        """
        if self.restraints_handle is not None:
            e, _ = self.energy_and_gradients(sites_cart=sites_cart)
            return e
        out = self._call(sites_cart=sites_cart, max_iterations=0, optimizer="lbfgs")
        return float(out["target"]["before"])

    def energy_and_gradients(
        self,
        sites_cart: Optional[Any] = None,
    ) -> tuple[float, np.ndarray]:
        """CCTBX restraint energy and ∂E/∂x via phridge (``geometry_restraints_energy_grad``).

        Requires ``restraints_handle`` from ``RemoteRestraintBuilder.build``.
        Every call packs sites through Bridge onto the CCTBX stream.
        """
        if self.restraints_handle is None:
            raise RuntimeError(
                "energy_and_gradients requires restraints_handle "
                "(use RemoteRestraintBuilder.build / RemoteGeometry.from_build)"
            )
        sites = _sites_for_call(sites_cart, self._packed_hier)
        result = self.bridge.call(
            "geometry_restraints_energy_grad",
            sites=sites,
            params={"restraints_handle": self.restraints_handle},
        )
        if not isinstance(result, dict) or "energy" not in result or "sites_grad" not in result:
            raise RuntimeError("geometry_restraints_energy_grad did not return energy and sites_grad")
        energy = float(result["energy"])
        grad = np.asarray(result["sites_grad"], dtype=np.float64)
        if grad.ndim != 2 or grad.shape[1] != 3:
            raise ValueError(f"sites_grad must have shape (N, 3), got {grad.shape}")
        stats = result.get("stats")
        self.last_target = {
            "energy": energy,
            "source": "cctbx",
            "stats": dict(stats) if isinstance(stats, dict) else None,
        }
        return energy, grad

    def minimize(
        self,
        *,
        max_iterations: int = 100,
        optimizer: str = "lbfgs",
        lr: Optional[float] = None,
        lr_min: Optional[float] = None,
        schedule: Optional[str] = None,
        momentum: float = 0.0,
        preconditioner: Optional[str] = None,
        precond_refresh: int = 0,
        sites_cart: Optional[Any] = None,
        update_hierarchy: bool = True,
    ) -> Any:
        """Run remote torch minimization; return a cctbx hierarchy (or packed sites).

        Blocks on ``bridge.call("geometry_minimize", ...)`` until the worker
        reports done. ``optimizer`` is ``"lbfgs"``, ``"adam"``, ``"adamw"``, ``"sgd"`` or
        ``"gauss_newton"`` (Levenberg-Marquardt with a sparse direct solve of the
        Gauss-Newton system; ``lr`` is then the initial relative damping, default 1e-3).
        First-order methods accept ``schedule`` (``none`` / ``cosine`` / ``triangular``),
        ``lr_min``, and SGD ``momentum``. ``preconditioner="diagonal"`` runs L-BFGS /
        first-order methods in Jacobi-scaled coordinates (weak for pure geometry — the
        stiffness is off-diagonal — see docs/geometry_curvature.md); ``precond_refresh``
        recomputes the scaling every that many iterations.

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
            preconditioner=preconditioner,
            precond_refresh=precond_refresh,
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

    # -- second-derivative information (torch packed restraints) -----------------------
    def curvature(self, sites_cart: Optional[Any] = None, *, sparse: bool = True) -> dict[str, Any]:
        """Gauss-Newton curvature of the torch restraint energy at ``sites_cart``.

        Returns ``diagonal`` (N, 3), ``blocks`` (N, 3, 3), ``gradient`` (N, 3), ``energy``,
        ``stats`` and, with ``sparse=True``, COO triplets ``rows`` / ``cols`` / ``vals`` of
        2 J^T W J over the 3N cartesian coordinates.
        """
        sites = _sites_for_call(sites_cart, self._packed_hier)
        out = self.bridge.call("geometry_curvature", sites=sites, restraints=self._packed_restr, params={"sparse": bool(sparse)})
        return {k: (np.asarray(v) if k not in ("energy", "stats") else v) for k, v in out.items()}

    def hvp(self, v: Any, sites_cart: Optional[Any] = None, *, hessian: str = "gn") -> np.ndarray:
        """Hessian-vector product of the restraint energy (``hessian`` = "gn" or "full"), cartesian (N, 3)."""
        sites = _sites_for_call(sites_cart, self._packed_hier)
        out = self.bridge.call(
            "geometry_hvp",
            sites=sites,
            restraints=self._packed_restr,
            v=np.asarray(v, dtype=np.float64),
            params={"hessian": str(hessian)},
        )
        return np.asarray(out, dtype=np.float64).reshape(-1, 3)

    def gn_solve(
        self,
        rhs: Any,
        sites_cart: Optional[Any] = None,
        *,
        extra_diag: Optional[Any] = None,
        weight: float = 1.0,
        damping: float = 1e-3,
        method: str = "sparse",
        groups: Optional[list[Any]] = None,
        blocks: Optional[list[str]] = None,
        block: Optional[str] = None,
        adp_params: Optional[Any] = None,
    ) -> np.ndarray:
        """Solve (weight 2 J^T W J + diag(extra_diag) + mu I) p = rhs on the worker.

        ``method="sparse"`` factorises the full sparse Gauss-Newton matrix; ``"tridiagonal"``
        keeps only residue blocks and consecutive-residue couplings (``groups`` = atom-index
        lists in chain order, e.g. from :meth:`residue_groups`). With ``rhs = -grad`` and no
        ``extra_diag`` this is one Levenberg-Marquardt step of the geometry term; with
        ``extra_diag`` = x-ray Gauss-Newton diagonal (cartesian) it is the preconditioner used
        by :class:`phridge.client.joint.JointSiteRefinement`.
        ``blocks``: ["sites"] (default), ["adp"], or ["sites", "adp"] for joint solve.
        """
        sites = _sites_for_call(sites_cart, self._packed_hier)
        params: dict[str, Any] = {"weight": float(weight), "damping": float(damping), "method": str(method)}
        if groups is not None:
            params["groups"] = [[int(i) for i in g] for g in groups]
        if blocks is not None:
            params["blocks"] = list(blocks)
        if block is not None:
            params["block"] = str(block)
        if adp_params is not None:
            params["adp_params"] = np.asarray(adp_params, dtype=np.float64).tolist()
        kwargs: dict[str, Any] = dict(
            sites=sites,
            restraints=self._packed_restr,
            rhs=np.asarray(rhs, dtype=np.float64),
            params=params,
        )
        if extra_diag is not None:
            kwargs["extra_diag"] = np.asarray(extra_diag, dtype=np.float64)
        out = self.bridge.call("geometry_gn_solve", **kwargs)
        self.last_gn_solve_stats = dict(out["stats"])
        return np.asarray(out["solution"], dtype=np.float64)

    def adp_eval(
        self,
        adp_params: Any,
        sites_cart: Optional[Any] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Evaluate ADP prior energy, gradient, GN diagonal, and sparse COO triplets."""
        sites = _sites_for_call(sites_cart, self._packed_hier)
        return self.bridge.call(
            "adp_prior_eval",
            sites=sites,
            adp_params=np.asarray(adp_params, dtype=np.float64),
            restraints=self._packed_restr,
            params=dict(params or {}),
        )

    def adp_hvp(
        self,
        adp_params: Any,
        v: Any,
        sites_cart: Optional[Any] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> np.ndarray:
        """Evaluate ADP prior HVP."""
        sites = _sites_for_call(sites_cart, self._packed_hier)
        out = self.bridge.call(
            "adp_prior_hvp",
            sites=sites,
            adp_params=np.asarray(adp_params, dtype=np.float64),
            restraints=self._packed_restr,
            v=np.asarray(v, dtype=np.float64),
            params=dict(params or {}),
        )
        if isinstance(out, dict):
            return np.asarray(out["hv"], dtype=np.float64)
        return np.asarray(out, dtype=np.float64)

    def residue_groups(self) -> list[list[int]]:
        """Atom indices grouped by (chain, resseq, icode) in hierarchy order — blocks for the tridiagonal solve."""
        groups: dict[tuple[Any, ...], list[int]] = {}
        for atom in self._packed_hier.meta.atoms:
            key = (getattr(atom, "chain_id", None), getattr(atom, "resseq", None), getattr(atom, "icode", None))
            groups.setdefault(key, []).append(int(atom.i))
        return list(groups.values())

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
        preconditioner: Optional[str] = None,
        precond_refresh: int = 0,
    ) -> dict[str, Any]:
        sites = _sites_for_call(sites_cart, self._packed_hier)
        params: dict[str, Any] = {
            "max_iterations": int(max_iterations),
            "optimizer": str(optimizer),
            "momentum": float(momentum),
            "precond_refresh": int(precond_refresh),
        }
        if preconditioner is not None:
            params["preconditioner"] = str(preconditioner)
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
