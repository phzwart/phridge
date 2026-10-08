"""Write a coordinate model. cctbx stays in the shim."""

from __future__ import annotations

from typing import Any

from phridge.shims.cctbx.xray_structure import write_pdb_file


def write_model(path: str, structure: Any) -> None:
    """Write PDB. The shim also writes the sibling mmCIF."""
    write_pdb_file(path, structure)
