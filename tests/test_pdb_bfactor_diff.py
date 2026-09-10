"""Tests for scripts/pdb_bfactor_diff.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("iotbx")


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "pdb_bfactor_diff.py"
    name = "pdb_bfactor_diff_test"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_pdb(path: Path, b_values: list[float]) -> None:
    lines = [
        "CRYST1   10.000   10.000   10.000  90.00  90.00  90.00 P 1",
    ]
    for i, b in enumerate(b_values, start=1):
        lines.append(
            f"ATOM  {i:5d}  CA  ALA A{i:4d}       {float(i):7.3f}   0.000   0.000  1.00{b:6.2f}           C"
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def test_bfactor_diff_writes_delta(tmp_path):
    mod = _load_module()
    a = tmp_path / "a.pdb"
    b = tmp_path / "b.pdb"
    out = tmp_path / "diff.pdb"
    _write_pdb(a, [20.0, 30.0])
    _write_pdb(b, [18.0, 35.0])

    assert mod.run([str(a), str(b), "-o", str(out), "--quiet"]) == 0

    from iotbx import pdb

    atoms = pdb.input(str(out)).construct_hierarchy().atoms()
    bs = [float(atom.b) for atom in atoms]
    assert bs == pytest.approx([2.0, -5.0])


def test_bfactor_diff_swap(tmp_path):
    mod = _load_module()
    a = tmp_path / "a.pdb"
    b = tmp_path / "b.pdb"
    out = tmp_path / "diff.pdb"
    _write_pdb(a, [20.0])
    _write_pdb(b, [18.0])

    mod.run([str(a), str(b), "-o", str(out), "--swap", "--quiet"])

    from iotbx import pdb

    atom = pdb.input(str(out)).construct_hierarchy().atoms()[0]
    assert float(atom.b) == pytest.approx(-2.0)
