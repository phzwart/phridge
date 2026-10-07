"""Scattering-table adapter."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from phridge.sfcalc.packing import PackedScatteringTable


from phridge.contracts.manifest import PluginManifest, Slot, Space, Tier

MANIFEST = PluginManifest(
    name='cctbx_scattering',
    slot=Slot.io_adapter,
    space=Space.D,
    tier=Tier.shim,
)

def scattering_table_from_cctbx(structure: Any, table: Optional[str] = None) -> Any:
    """Gaussian form factors for every scattering type in an xray.structure.

    Uses the structure's existing scattering_type_registry when ``table``
    is None (falling back to wk1995 if none was set up yet).
    """
    from phridge.sfcalc.packing import PackedScatteringTable

    if table is not None:
        structure.scattering_type_registry(table=table)
    try:
        reg = structure.scattering_type_registry()
    except Exception:  # registry not set up yet
        table = table or "wk1995"
        structure.scattering_type_registry(table=table)
        reg = structure.scattering_type_registry()
    labels = sorted(reg.type_index_pairs_as_dict().keys())
    k = max(reg.gaussian(t).n_terms() for t in labels) if labels else 0
    gauss_a = np.zeros((len(labels), k), dtype=np.float64)
    gauss_b = np.zeros((len(labels), k), dtype=np.float64)
    gauss_c = np.zeros(len(labels), dtype=np.float64)
    for i, label in enumerate(labels):
        g = reg.gaussian(label)
        n = g.n_terms()
        gauss_a[i, :n] = list(g.array_of_a())
        gauss_b[i, :n] = list(g.array_of_b())
        gauss_c[i] = g.c()
    return PackedScatteringTable(labels, gauss_a, gauss_b, gauss_c, table=str(table or "cctbx"))
