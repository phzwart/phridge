"""Model and reflection files round-trip through the shims."""

from __future__ import annotations

import numpy as np

from phridge.io.read_data import read_data
from phridge.io.read_model import read_model
from phridge.io.write_model import write_model
from phridge.shims.cctbx.miller import write_observation_mtz
from phridge.shims.cctbx.xray_structure import random_xray_structure


def test_model_and_mtz_round_trip(tmp_path) -> None:
    structure = random_xray_structure("P 1", ["C", "N", "O", "C"], seed=1)
    pdb = tmp_path / "model.pdb"
    write_model(str(pdb), structure)
    assert pdb.with_suffix(".cif").is_file()
    record = read_model(str(pdb))
    assert len(record.sites_frac) == 4
    assert np.isfinite(record.sites_frac).all()

    cell = tuple(float(value) for value in structure.unit_cell().parameters())
    symbol = str(structure.space_group_info().type().lookup_symbol())
    hkl = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=np.int32)
    values = np.array([10.0, 8.0, 6.0, 4.0])
    sigmas = np.ones(4)
    flags = np.array([0.0, 1.0, 0.0, 1.0])
    mtz = tmp_path / "data.mtz"
    write_observation_mtz(str(mtz), cell, symbol, hkl, values, sigmas, flags, label="F")
    observed = read_data(str(mtz), "amplitude")
    assert observed.data["amplitude"].shape == (4,)
    assert observed.index["hkl"].shape == (4, 3)
    assert int(observed.partitions["free"].sum()) == 2
