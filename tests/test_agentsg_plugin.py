"""External agentsg plugin via register_op (skips if agentsg not installed)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

agentsg = pytest.importorskip("agentsg")
from agentsg.cell import niggli_reduce  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

import plugins.agentsg_niggli  # noqa: E402,F401

from phridge.client import Bridge  # noqa: E402


INPUT = (9.0, 5.0, 7.0, 80.0, 100.0, 95.0)


def test_agentsg_niggli_through_memory_bridge():
    direct, cob = niggli_reduce(*INPUT)
    bridge = Bridge(memory=True, timeout=5)
    out = bridge.call(
        "agentsg_niggli_reduce",
        cell={
            "a": INPUT[0],
            "b": INPUT[1],
            "c": INPUT[2],
            "alpha": INPUT[3],
            "beta": INPUT[4],
            "gamma": INPUT[5],
        },
    )
    assert out["source"] == "agentsg"
    assert out["unit_cell"] == pytest.approx(list(direct), abs=1e-9)
    assert out["change_of_basis"] == [[int(v) for v in row] for row in cob]


def test_agentsg_plugin_accepts_six_list():
    from plugins.agentsg_niggli import agentsg_niggli_reduce

    out = agentsg_niggli_reduce(list(INPUT))
    direct, _ = niggli_reduce(*INPUT)
    assert out["unit_cell"] == pytest.approx(list(direct), abs=1e-9)


def test_niggli_cell_cctbx_wrapper():
    cctbx = pytest.importorskip("cctbx")
    from cctbx import uctbx
    from plugins.agentsg_niggli import niggli_cell

    direct, _ = niggli_reduce(*INPUT)
    bridge = Bridge(memory=True, timeout=5)
    uc = uctbx.unit_cell(INPUT)
    uc_red = niggli_cell(uc, bridge=bridge)
    assert isinstance(uc_red, uctbx.unit_cell)
    assert list(uc_red.parameters()) == pytest.approx(list(direct), abs=1e-9)
    uc_red2, cb_op = niggli_cell(uc, bridge=bridge, return_change_of_basis=True)
    assert list(uc_red2.parameters()) == pytest.approx(list(direct), abs=1e-9)
    assert hasattr(cb_op, "as_xyz")
