"""Two cycles of the stock policy on a small crystal, for both example configs."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from phridge.api import Refinement
from phridge.policies.stock_macrocycle import StockMacrocycle
from phridge.shims.cctbx.miller import write_observation_mtz
from phridge.shims.cctbx.xray_structure import random_xray_structure, write_pdb_file

_ROOT = Path(__file__).resolve().parents[2]


def _inputs(tmp_path: Path, kind: str) -> tuple[Path, Path]:
    structure = random_xray_structure("P 1", ["C", "N", "O", "C"], seed=3)
    pdb = tmp_path / "lysozyme.pdb"
    write_pdb_file(str(pdb), structure)
    cell = tuple(float(value) for value in structure.unit_cell().parameters())
    symbol = str(structure.space_group_info().type().lookup_symbol())
    hkl = np_hkl()
    values = torch.tensor([12.0, 9.0, 7.0, 5.0, 4.0, 3.0], dtype=torch.float64).numpy()
    flags = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]).numpy()
    mtz = tmp_path / f"{kind}.mtz"
    write_observation_mtz(
        str(mtz),
        cell,
        symbol,
        hkl,
        values,
        torch.ones(6).numpy(),
        flags,
        label="I" if kind == "intensity" else "F",
        intensity=kind == "intensity",
    )
    return pdb, mtz


def np_hkl():
    import numpy as np

    return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=np.int32)


def _config(tmp_path: Path, example: str, pdb: Path, mtz: Path, kind: str) -> Path:
    text = (_ROOT / "examples" / "configs" / example).read_text(encoding="utf-8")
    text = text.replace("observed:\n", f"observed:\n  pdb: {pdb}\n  mtz: {mtz}\n  kind: {kind}\n", 1)
    path = tmp_path / example
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.slow
def test_minimal_loop(tmp_path: Path) -> None:
    cases = (("amplitude_ml.yaml", "amplitude"), ("intensity_mli.yaml", "intensity"))
    for example, kind in cases:
        pdb, mtz = _inputs(tmp_path, kind)
        config = _config(tmp_path, example, pdb, mtz, kind)
        job = Refinement.from_config(str(config))
        job.run(StockMacrocycle(cycles=2))
        out = tmp_path / kind
        job.finalize(str(out))
        print(f"{kind} R_free per cycle: {job.r_free_trace}")
        assert len(job.r_free_trace) == 2
        assert (out / "model.pdb").is_file()
        assert (out / "model.cif").is_file()
        assert (out / "data.mtz").is_file()
        final = job.registry.block("sites").value.clone()
        job.registry.set_value("sites", torch.zeros_like(final))
        job.controller.replay(job.registry)
        assert torch.equal(job.registry.block("sites").value, final)
