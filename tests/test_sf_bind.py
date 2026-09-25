"""Kept structure-factor engine: bind once, reuse handle."""

from __future__ import annotations

import numpy as np
import pytest

from phridge.client import Bridge
from phridge.models import CrystalSymmetry, Scatterer, SfEngineParams, SymOp
from phridge.packing import PackedMiller
from phridge.packing_xtal import PackedXray
from phridge.sfcalc.ops import sf_bind, sf_calc
from phridge.sfcalc.packing import PackedScatteringTable
from phridge.sfcalc.sessions import clear_engines


def _packed_carbon(n_atoms: int = 6):
    rng = np.random.default_rng(1)
    crystal = CrystalSymmetry(
        unit_cell=[20.0, 22.0, 24.0, 90.0, 90.0, 90.0],
        space_group_hall=" P 1",
        space_group_number=1,
        symops=[SymOp(r=[1, 0, 0, 0, 1, 0, 0, 0, 1], t=[0, 0, 0])],
    )
    xray = PackedXray(
        crystal=crystal,
        sites_frac=rng.random((n_atoms, 3)),
        occupancy=np.ones(n_atoms),
        u_iso=np.full(n_atoms, 0.04),
        scatterers=[Scatterer(i=i, scattering_type="C") for i in range(n_atoms)],
    )
    table = PackedScatteringTable(
        labels=["C"],
        gauss_a=np.array([[2.31, 1.02, 1.5886, 0.865]], dtype=np.float64),
        gauss_b=np.array([[20.8439, 10.2075, 0.5687, 51.6512]], dtype=np.float64),
        gauss_c=np.array([0.2156], dtype=np.float64),
    )
    hkl = np.array([(1, 0, 0), (0, 1, 0), (1, 1, 0), (2, 0, 1)], dtype=np.int32)
    miller = PackedMiller(crystal=crystal, hkl=hkl, data=np.zeros(len(hkl), dtype=np.complex128))
    params = SfEngineParams(d_min=2.0, quality_factor=100.0, stamp_backend="torch")
    return xray, table, miller, params


def test_sf_bind_reuse_matches_stateless():
    pytest.importorskip("torch")
    clear_engines()
    xray, table, miller, params = _packed_carbon()
    once = sf_calc(xray, table, miller, params)
    handle = sf_bind(xray, table, miller, params)
    kept = sf_calc(handle=handle)
    assert np.allclose(once.data, kept.data)


def test_sf_bind_unknown_handle():
    with pytest.raises(KeyError, match="unknown sf_handle"):
        sf_calc(handle="not-a-real-handle")


def test_sf_bind_via_memory_bridge():
    pytest.importorskip("torch")
    clear_engines()
    xray, table, miller, params = _packed_carbon()
    bridge = Bridge(memory=True, timeout=30)
    handle = str(bridge.call("sf_bind", xray=xray, table=table, hkl=miller, params=params))
    a = bridge.call("sf_calc", handle=handle, hkl=miller)
    b = bridge.call("sf_calc", handle=handle, hkl=miller)
    assert np.allclose(np.asarray(a.data), np.asarray(b.data))
