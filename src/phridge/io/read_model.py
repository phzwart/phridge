"""Read a coordinate file into an ``RObject``. cctbx stays in the shim."""

from __future__ import annotations

from phridge.contracts.tensors import RObject
from phridge.shims.cctbx.xray_structure import to_robject, xray_structure_from_pdb


def read_model(path: str) -> RObject:
    """PDB or mmCIF path to scatterer tensors."""
    return to_robject(xray_structure_from_pdb(path))
