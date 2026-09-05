"""CCTBX-side geometry ops (restraints build + energy/gradients)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from phridge.cctbx_worker.sessions import get_session, stash_restraints_manager
from phridge.client.convert_geometry import model_geometry, restraints_from_cctbx
from phridge.client.convert_xtal import hierarchy_from_cctbx, hierarchy_to_cctbx
from phridge.packing_xtal import PackedCartesian, PackedHierarchy


def build_geometry_restraints(
    params: Optional[dict[str, Any]] = None,
    hierarchy: Any = None,
) -> dict[str, Any]:
    """Build packed geometry restraints from a PDB string and/or hierarchy.

    Also stashes the live ``geometry_restraints.manager`` in this CCTBX worker
    process and returns ``restraints_handle`` for later
    ``geometry_restraints_energy_grad`` calls.
    """
    pdb_string = _pdb_string(params, hierarchy)
    if not pdb_string or not str(pdb_string).strip():
        raise TypeError("build_geometry_restraints needs params.pdb_string or hierarchy")

    hierarchy_cctbx, grm = _process_model(str(pdb_string))
    packed_hier = hierarchy_from_cctbx(hierarchy_cctbx)
    packed_restr = restraints_from_cctbx(grm, n_sites=packed_hier.meta.n_atoms)
    model_geometry(hierarchy=packed_hier, restraints=packed_restr)
    handle = stash_restraints_manager(grm, n_sites=packed_hier.meta.n_atoms)
    return {
        "hierarchy": packed_hier,
        "restraints": packed_restr,
        "restraints_handle": handle,
    }


def geometry_restraints_energy_grad(
    sites: Any,
    params: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Evaluate CCTBX restraint energy and Cartesian site gradients.

    Requires ``params.restraints_handle`` from a prior ``build_geometry_restraints``
    on **this** CCTBX worker process. ``sites`` may be :class:`PackedCartesian`,
    a numpy ``(N,3)`` array, or ``flex.vec3_double`` (cctbx decode path).
    """
    if not isinstance(params, dict):
        raise TypeError("params must be a dict with restraints_handle")
    handle = params.get("restraints_handle")
    if not handle:
        raise TypeError("params.restraints_handle is required")

    session = get_session(str(handle))
    grm = session["grm"]
    n_sites = int(session["n_sites"])

    from scitbx.array_family import flex

    if isinstance(sites, PackedCartesian):
        xyz = sites.xyz
        if hasattr(xyz, "detach"):
            xyz = xyz.detach().cpu().numpy()
        xyz = np.asarray(xyz, dtype=np.float64)
        sites_cart = flex.vec3_double(xyz)
    elif hasattr(sites, "size") and hasattr(sites, "__iter__") and not isinstance(sites, np.ndarray):
        # flex.vec3_double from prefer_cctbx decode
        sites_cart = sites
        xyz = np.asarray(list(sites_cart), dtype=np.float64).reshape(-1, 3)
    else:
        xyz = np.asarray(sites, dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("sites must have shape (N, 3)")
        sites_cart = flex.vec3_double(xyz)

    if xyz.shape[0] != n_sites:
        raise ValueError(f"n_sites mismatch: sites={xyz.shape[0]} session={n_sites}")

    energies = grm.energies_sites(sites_cart=sites_cart, compute_gradients=True)
    grads = energies.gradients
    if grads is None:
        raise RuntimeError("energies_sites returned no gradients")
    sites_grad = np.asarray(list(grads), dtype=np.float64).reshape(n_sites, 3)
    return {
        "energy": float(energies.target),
        "sites_grad": sites_grad,
        "stats": _energies_stats(energies),
    }


def _energies_stats(energies: Any) -> dict[str, Any]:
    """JSON-safe CCTBX ``energies_sites`` breakdown (residuals + RMS deviations)."""

    def _residual(name: str, n_attr: str, sum_attr: str) -> dict[str, Any]:
        n = getattr(energies, n_attr, None)
        s = getattr(energies, sum_attr, 0.0) or 0.0
        return {
            "name": name,
            "n": int(n) if n is not None else 0,
            "residual_sum": float(s),
        }

    terms = [
        _residual("bond", "n_bond_proxies", "bond_residual_sum"),
        _residual("nonbonded", "n_nonbonded_proxies", "nonbonded_residual_sum"),
        _residual("angle", "n_angle_proxies", "angle_residual_sum"),
        _residual("dihedral", "n_dihedral_proxies", "dihedral_residual_sum"),
        _residual("chirality", "n_chirality_proxies", "chirality_residual_sum"),
        _residual("planarity", "n_planarity_proxies", "planarity_residual_sum"),
        _residual("parallelity", "n_parallelity_proxies", "parallelity_residual_sum"),
        _residual("bond_similarity", "n_bond_similarity_proxies", "bond_similarity_residual_sum"),
    ]

    def _dev4(fn: Any) -> dict[str, float] | None:
        if not callable(fn):
            return None
        try:
            out = fn()
        except Exception:
            return None
        if out is None:
            return None
        if len(out) >= 4:
            b_min, b_max, b_rms, n = out[:4]
            return {
                "min": float(b_min),
                "max": float(b_max),
                "rms": float(b_rms),
                "n": int(n),
            }
        if len(out) == 3:
            b_min, b_max, b_rms = out
            return {"min": float(b_min), "max": float(b_max), "rms": float(b_rms)}
        return None

    deviations = {
        "bond_A": _dev4(getattr(energies, "bond_deviations", None)),
        "angle_deg": _dev4(getattr(energies, "angle_deviations", None)),
        "dihedral_deg": _dev4(getattr(energies, "dihedral_deviations", None)),
        "chirality": _dev4(getattr(energies, "chirality_deviations", None)),
        "planarity_A": _dev4(getattr(energies, "planarity_deviations", None)),
        "nonbonded_A": _dev4(getattr(energies, "nonbonded_deviations", None)),
    }
    show_lines: list[str] = []
    try:
        import io

        buf = io.StringIO()
        energies.show(f=buf)
        show_lines = [ln.rstrip() for ln in buf.getvalue().splitlines() if ln.strip()]
    except Exception:
        show_lines = []

    return {
        "target": float(energies.target),
        "residual_sum": float(getattr(energies, "residual_sum", energies.target) or 0.0),
        "number_of_restraints": int(getattr(energies, "number_of_restraints", 0) or 0),
        "terms": terms,
        "deviations": {k: v for k, v in deviations.items() if v is not None},
        "show": show_lines,
    }


def _pdb_string(params: Optional[dict[str, Any]], hierarchy: Any) -> Optional[str]:
    if isinstance(params, dict):
        for key in ("pdb_string", "model_input", "pdb"):
            value = params.get(key)
            if value is not None:
                return str(value)
    if hierarchy is None:
        return None
    if isinstance(hierarchy, PackedHierarchy):
        hierarchy = hierarchy_to_cctbx(hierarchy)
    as_pdb = getattr(hierarchy, "as_pdb_string", None)
    if callable(as_pdb):
        return as_pdb()
    raise TypeError(f"cannot derive PDB text from hierarchy type {type(hierarchy)!r}")


def _process_model(pdb_string: str) -> tuple[Any, Any]:
    import iotbx.pdb
    import mmtbx.model
    from libtbx.utils import null_out

    model = mmtbx.model.manager(
        model_input=iotbx.pdb.input(source_info=None, lines=pdb_string.split("\n")),
        log=null_out(),
    )
    model.process(make_restraints=True)
    hierarchy = model.get_hierarchy()
    restraints_manager = model.get_restraints_manager()
    if restraints_manager is None:
        raise RuntimeError("mmtbx did not build a restraints manager (chem_data / mon_lib?)")
    grm = restraints_manager.geometry
    return hierarchy, grm
