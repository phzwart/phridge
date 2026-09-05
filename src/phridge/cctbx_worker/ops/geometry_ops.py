"""CCTBX-side geometry ops (restraints build via mmtbx)."""

from __future__ import annotations

from typing import Any, Optional

from phridge.client.convert_geometry import model_geometry, restraints_from_cctbx
from phridge.client.convert_xtal import hierarchy_from_cctbx, hierarchy_to_cctbx
from phridge.packing_xtal import PackedHierarchy


def build_geometry_restraints(
    params: Optional[dict[str, Any]] = None,
    hierarchy: Any = None,
) -> dict[str, Any]:
    """Build packed geometry restraints from a PDB string and/or hierarchy.

    Parameters
    ----------
    params:
        JSON dict. Prefer ``pdb_string`` (or ``model_input``) so a torch driver
        needs no cctbx. Optional keys are passed through for future flags.
    hierarchy:
        cctbx ``pdb.hierarchy.root`` or :class:`PackedHierarchy`. Used when
        ``pdb_string`` is omitted (converted to PDB text for mmtbx).

    Returns
    -------
    dict with packed ``hierarchy`` and ``restraints`` (and ``model_geometry``).
    """
    pdb_string = _pdb_string(params, hierarchy)
    if not pdb_string or not str(pdb_string).strip():
        raise TypeError("build_geometry_restraints needs params.pdb_string or hierarchy")

    hierarchy_cctbx, grm = _process_model(str(pdb_string))
    packed_hier = hierarchy_from_cctbx(hierarchy_cctbx)
    packed_restr = restraints_from_cctbx(grm, n_sites=packed_hier.meta.n_atoms)
    # Validate i_seq correspondence before packing crosses Redis.
    model_geometry(hierarchy=packed_hier, restraints=packed_restr)
    return {"hierarchy": packed_hier, "restraints": packed_restr}


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
