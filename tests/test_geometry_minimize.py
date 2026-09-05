"""RemoteGeometry: cctbx pack → torch minimize on worker → cctbx hierarchy back."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from phridge.client import Bridge, RemoteGeometry  # noqa: E402
from phridge.packing_geometry import PackedRestraints  # noqa: E402
from phridge.packing_xtal import PackedCartesian, PackedHierarchy  # noqa: E402
from phridge.models import Atom, CoordinateFrame  # noqa: E402
from phridge.worker.geometry.energy import energy_and_sites  # noqa: E402


def _bridge():
    return Bridge(memory=True, timeout=2)


def _triangle_restraints() -> PackedRestraints:
    return PackedRestraints(
        n_sites=3,
        bond_i_seqs=np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int32),
        bond_distance_ideal=np.array([1.5, 1.5, 1.5]),
        bond_weight=np.array([100.0, 100.0, 100.0]),
        bond_slack=np.array([0.0, 0.0, 0.0]),
        angle_i_seqs=np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]], dtype=np.int32),
        angle_ideal=np.array([60.0, 60.0, 60.0]),
        angle_weight=np.array([1.0, 1.0, 1.0]),
    )


def _triangle_hierarchy() -> PackedHierarchy:
    xyz = np.array([(0.0, 0.0, 0.0), (1.8, 0.0, 0.0), (0.9, 1.2, 0.0)])
    atoms = [
        Atom(i=i, name=f" C{i} ", element="C", chain_id="A", resseq=1, resname="TRI") for i in range(3)
    ]
    return PackedHierarchy(
        xyz=xyz,
        occupancy=np.ones(3),
        b_iso=np.full(3, 20.0),
        atoms=atoms,
        frame=CoordinateFrame.cartesian,
    )


def test_torch_energy_triangle_collapses():
    restr = _triangle_restraints()
    sites = np.array([(0.0, 0.0, 0.0), (1.8, 0.0, 0.0), (0.9, 1.2, 0.0)])
    out, target = energy_and_sites(sites, restr, max_iterations=50, optimizer="lbfgs")
    assert target["before"] > 10.0
    assert target["after"] < 1e-3 * target["before"]
    d01 = np.linalg.norm(out[0] - out[1])
    assert abs(d01 - 1.5) < 1e-3
    assert target["n_calls"] >= target["n_steps"] > 0
    assert target["n_calls"] > target["n_steps"]  # Wolfe line search
    assert target["optimizer_state_mb"] > 0.0
    assert "rss_before_mb" in target and "rss_after_mb" in target
    assert "rss_delta_mb" in target and "cuda_peak_mb" in target


@pytest.mark.parametrize("optimizer", ["lbfgs", "adam", "adamw", "sgd"])
def test_torch_optimizers_reduce_energy(optimizer):
    restr = _triangle_restraints()
    sites = np.array([(0.0, 0.0, 0.0), (1.8, 0.0, 0.0), (0.9, 1.2, 0.0)])
    if optimizer == "lbfgs":
        max_iter, lr = 50, None
    elif optimizer in ("adam", "adamw"):
        max_iter, lr = 800, 5e-2
    else:
        max_iter, lr = 2000, 1e-5
    _, target = energy_and_sites(sites, restr, max_iterations=max_iter, optimizer=optimizer, lr=lr)
    assert np.isfinite(target["after"])
    assert target["after"] < target["before"]
    assert target["optimizer"] == optimizer
    assert int(target["n_calls"]) >= int(target["n_steps"]) > 0
    if optimizer == "lbfgs":
        assert int(target["n_calls"]) > int(target["n_steps"])
    else:
        assert int(target["n_calls"]) == int(target["n_steps"])
    # SGD with momentum=0 keeps no state tensors; others always allocate.
    if optimizer == "sgd":
        assert float(target["optimizer_state_mb"]) >= 0.0
    else:
        assert float(target["optimizer_state_mb"]) > 0.0
    assert float(target["rss_before_mb"]) > 0.0
    assert float(target["cuda_peak_mb"]) >= 0.0


def test_optimizer_memory_lbfgs_larger_than_sgd():
    """LBFGS history buffers should exceed SGD's single velocity buffer."""
    restr = _triangle_restraints()
    sites = np.array([(0.0, 0.0, 0.0), (1.8, 0.0, 0.0), (0.9, 1.2, 0.0)])
    _, lbfgs = energy_and_sites(sites, restr, max_iterations=30, optimizer="lbfgs")
    _, sgd = energy_and_sites(
        sites, restr, max_iterations=30, optimizer="sgd", lr=1e-4, momentum=0.9
    )
    assert lbfgs["optimizer_state_mb"] > sgd["optimizer_state_mb"] > 0.0
    _, adam = energy_and_sites(sites, restr, max_iterations=30, optimizer="adam", lr=5e-2)
    # Adam stores m+v; LBFGS history_size=20 grows past that on this tiny problem
    # once enough iterations fill the history — just require both reported.
    assert adam["optimizer_state_mb"] > 0.0


def test_remote_geometry_minimize_triangle():
    bridge = _bridge()
    hier = _triangle_hierarchy()
    restr = _triangle_restraints()
    geo = RemoteGeometry(bridge, hier, restr)
    e0 = geo.energy()
    assert e0 > 10.0
    out = geo.minimize(max_iterations=50, optimizer="lbfgs")
    assert geo.last_target is not None
    assert geo.last_target["after"] < 0.2 * geo.last_target["before"]
    # packed path returns iotbx hierarchy when prefer_cctbx false → hierarchy_to_cctbx
    assert out.atoms().size() == 3
    xyz = np.array([atom.xyz for atom in out.atoms()])
    assert abs(np.linalg.norm(xyz[0] - xyz[1]) - 1.5) < 1e-2


def test_geometry_minimize_op_round_trip():
    bridge = _bridge()
    sites = PackedCartesian(np.array([(0.0, 0.0, 0.0), (1.8, 0.0, 0.0), (0.9, 1.2, 0.0)]))
    restr = _triangle_restraints()
    out = bridge.call(
        "geometry_minimize",
        sites=sites,
        restraints=restr,
        params={"max_iterations": 50, "optimizer": "lbfgs"},
    )
    assert out["target"]["after"] < 0.2 * out["target"]["before"]
    assert float(out["target"]["optimizer_state_mb"]) > 0.0
    assert "rss_delta_mb" in out["target"]
    assert "cuda_peak_mb" in out["target"]


def test_remote_geometry_peptide_when_chem_data_available():
    cctbx = pytest.importorskip("cctbx")
    from mmtbx.monomer_library import server as mon_server

    if mon_server.find_mon_lib_file(relative_path_components=["list", "mon_lib_list.cif"]) is None:
        pytest.skip("chem_data not installed")

    from libtbx.utils import null_out
    import iotbx.pdb
    import mmtbx.model
    from scitbx.array_family import flex

    pdb = """\
CRYST1   21.937    4.866   23.477  90.00 107.08  90.00 P 1 21 1      2
ATOM      1  N   GLY A   1      -9.009   4.612   6.102  1.00 16.77           N
ATOM      2  CA  GLY A   1      -9.052   4.207   4.651  1.00 16.57           C
ATOM      3  C   GLY A   1      -8.015   3.140   4.419  1.00 16.16           C
ATOM      4  O   GLY A   1      -7.523   2.521   5.381  1.00 16.78           O
ATOM      5  N   ASN A   2      -7.656   2.923   3.155  1.00 15.02           N
ATOM      6  CA  ASN A   2      -6.522   2.038   2.831  1.00 14.10           C
ATOM      7  C   ASN A   2      -5.241   2.537   3.427  1.00 13.13           C
ATOM      8  O   ASN A   2      -4.978   3.742   3.426  1.00 11.91           O
ATOM      9  CB  ASN A   2      -6.346   1.881   1.341  1.00 15.38           C
END
"""
    model = mmtbx.model.manager(
        model_input=iotbx.pdb.input(source_info=None, lines=pdb.split("\n")),
        log=null_out(),
    )
    model.process(make_restraints=True)
    hier = model.get_hierarchy()
    grm = model.get_restraints_manager().geometry
    sites = hier.atoms().extract_xyz()
    sites = sites + flex.vec3_double([(0.25, -0.15, 0.1)] * sites.size())
    hier.atoms().set_xyz(sites)

    bridge = _bridge()
    geo = RemoteGeometry(bridge, hier, grm)
    out = geo.minimize(max_iterations=100, optimizer="lbfgs")
    assert out.atoms().size() == hier.atoms().size()
    assert geo.last_target["after"] < 0.5 * geo.last_target["before"]
