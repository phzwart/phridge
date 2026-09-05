"""Tests for OpSpec.runtime routing and CCTBX build_geometry_restraints."""

from __future__ import annotations

import numpy as np
import pytest

from phridge.client import Bridge
from phridge.models import WorkerRuntime
from phridge.ops import get_op, list_ops
from phridge.redis_store import (
    JOBS_STREAM_CCTBX,
    JOBS_STREAM_TORCH,
    RedisStore,
    jobs_stream,
    worker_group,
)


def test_builtin_runtimes():
    assert get_op("scale_array").runtime == WorkerRuntime.torch
    assert get_op("geometry_minimize").runtime == WorkerRuntime.torch
    assert get_op("build_geometry_restraints").runtime == WorkerRuntime.cctbx
    assert get_op("geometry_restraints_energy_grad").runtime == WorkerRuntime.cctbx
    names = {s.name for s in list_ops()}
    assert "build_geometry_restraints" in names
    assert "geometry_restraints_energy_grad" in names


def test_jobs_stream_by_runtime():
    assert jobs_stream(WorkerRuntime.torch) == JOBS_STREAM_TORCH
    assert jobs_stream("cctbx") == JOBS_STREAM_CCTBX
    assert worker_group("torch").endswith(":torch")
    assert worker_group("cctbx").endswith(":cctbx")


def test_enqueue_routes_to_runtime_stream():
    fakeredis = pytest.importorskip("fakeredis")
    store = RedisStore(fakeredis.FakeRedis())
    bridge = Bridge(store=store, timeout=2)
    job_id = bridge.submit("scale_array", array=np.arange(2, dtype=np.float64), scale=1.0)
    # Torch stream should have the message; cctbx stream should not.
    torch_len = store.client.xlen(JOBS_STREAM_TORCH)
    cctbx_len = store.client.xlen(JOBS_STREAM_CCTBX)
    assert torch_len >= 1
    assert cctbx_len == 0
    assert store.get_envelope(job_id) is not None


def test_enqueue_cctbx_op_to_cctbx_stream():
    fakeredis = pytest.importorskip("fakeredis")
    store = RedisStore(fakeredis.FakeRedis())
    bridge = Bridge(store=store, timeout=2)
    # Do not process — just check routing. Envelope stays queued.
    job_id = bridge.submit(
        "build_geometry_restraints",
        params={"pdb_string": "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n"},
    )
    assert store.client.xlen(JOBS_STREAM_CCTBX) >= 1
    assert store.client.xlen(JOBS_STREAM_TORCH) == 0
    env = store.get_envelope(job_id)
    assert env is not None
    assert env.op == "build_geometry_restraints"


def _chem_data_available() -> bool:
    try:
        from mmtbx.monomer_library import server as mon_server
    except ImportError:
        return False
    path = mon_server.find_mon_lib_file(relative_path_components=["list", "mon_lib_list.cif"])
    return path is not None


def _helix_pdb(n_res: int = 5) -> str:
    from mmtbx.secondary_structure.build import ss_idealization as ssb

    helix = ssb.secondary_structure_from_sequence(ssb.alpha_helix_str, "A" * n_res)
    return (
        "CRYST1  100.000  100.000  100.000  90.00  90.00  90.00 P 1           1\n"
        + helix.as_pdb_string()
    )


@pytest.mark.skipif(not _chem_data_available(), reason="chem_data / mon_lib not installed")
def test_build_geometry_restraints_memory():
    pytest.importorskip("cctbx")
    pytest.importorskip("mmtbx")
    from phridge.client import RemoteRestraintBuilder
    from phridge.packing_geometry import PackedRestraints
    from phridge.packing_xtal import PackedHierarchy

    bridge = Bridge(memory=True, timeout=30)
    out = RemoteRestraintBuilder(bridge).build(pdb_string=_helix_pdb(5))
    assert isinstance(out["hierarchy"], PackedHierarchy)
    assert isinstance(out["restraints"], PackedRestraints)
    assert out["hierarchy"].meta.n_atoms == out["restraints"].n_sites
    assert out["restraints"].n_sites > 0
    assert out["restraints"].bond_i_seqs.size > 0
    assert out.get("restraints_handle")


@pytest.mark.skipif(not _chem_data_available(), reason="chem_data / mon_lib not installed")
def test_geometry_restraints_energy_grad_via_handle():
    pytest.importorskip("cctbx")
    pytest.importorskip("mmtbx")
    from phridge.client import RemoteGeometry, RemoteRestraintBuilder

    bridge = Bridge(memory=True, timeout=30)
    built = RemoteRestraintBuilder(bridge).build(pdb_string=_helix_pdb(5))
    geo = RemoteGeometry.from_build(bridge, built)
    xyz = np.asarray(built["hierarchy"].xyz, dtype=np.float64)
    e, g = geo.energy_and_gradients(xyz)
    assert np.isfinite(e)
    assert g.shape == xyz.shape
    stats = geo.last_target.get("stats")
    assert isinstance(stats, dict)
    assert "terms" in stats and "deviations" in stats
    assert any(t.get("name") == "bond" for t in stats["terms"])
    # Distorted sites should raise energy vs ideal.
    xyz_d = xyz + np.random.default_rng(1).normal(0.0, 0.3, size=xyz.shape)
    e_d, g_d = geo.energy_and_gradients(xyz_d)
    assert e_d > e
    assert np.linalg.norm(g_d) > 0
    assert float(geo.last_target["stats"]["target"]) == pytest.approx(e_d)


def test_unknown_restraints_handle_errors():
    pytest.importorskip("cctbx")
    from phridge.cctbx_worker.ops.geometry_ops import geometry_restraints_energy_grad
    from phridge.packing_xtal import PackedCartesian

    sites = PackedCartesian(np.zeros((3, 3), dtype=np.float64))
    with pytest.raises(KeyError, match="unknown restraints_handle"):
        geometry_restraints_energy_grad(sites, params={"restraints_handle": "no-such-handle"})


@pytest.mark.skipif(not _chem_data_available(), reason="chem_data / mon_lib not installed")
def test_torch_driver_build_then_minimize():
    pytest.importorskip("cctbx")
    pytest.importorskip("mmtbx")
    pytest.importorskip("torch")
    from phridge.client import RemoteGeometry, RemoteRestraintBuilder
    from phridge.packing_xtal import PackedCartesian

    bridge = Bridge(memory=True, timeout=60)
    built = RemoteRestraintBuilder(bridge).build(pdb_string=_helix_pdb(5))
    # Distort sites slightly so minimize has work to do.
    xyz = np.asarray(built["hierarchy"].xyz, dtype=np.float64)
    xyz = xyz + np.random.default_rng(0).normal(0.0, 0.2, size=xyz.shape)
    geo = RemoteGeometry(bridge, built["hierarchy"], built["restraints"])
    before = geo.energy(sites_cart=xyz)
    sites_out = geo.minimize(
        max_iterations=30,
        optimizer="lbfgs",
        sites_cart=xyz,
        update_hierarchy=False,
    )
    assert isinstance(sites_out, PackedCartesian)
    after = float(geo.last_target["after"])
    assert after < before
