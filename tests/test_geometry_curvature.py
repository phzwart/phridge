"""Geometry second derivatives: GN blocks / HVP / sparse factorisation, gauss_newton optimizer, ops."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from phridge.client import Bridge  # noqa: E402
from phridge.packing_geometry import PackedRestraints  # noqa: E402
from phridge.packing_xtal import PackedCartesian  # noqa: E402
from phridge.worker.geometry.curvature import (  # noqa: E402
    GaussNewtonPreconditioner,
    RestraintCurvature,
    gn_sparse_coo,
    jacobi_scale,
)
from phridge.worker.geometry.energy import _RestraintTables, energy_and_sites  # noqa: E402


def _ring(n=12, seed=0):
    rng = np.random.default_rng(seed)
    xyz = rng.normal(size=(n, 3)) * 2
    bonds = np.array([[i, (i + 1) % n] for i in range(n)] + [[0, 5], [3, 9]], dtype=np.int32)
    angles = np.array([[i, (i + 1) % n, (i + 2) % n] for i in range(n)], dtype=np.int32)
    dih = np.array([[i, (i + 1) % n, (i + 2) % n, (i + 3) % n] for i in range(n)], dtype=np.int32)
    pr = PackedRestraints(
        n_sites=n,
        bond_i_seqs=bonds, bond_distance_ideal=np.full(len(bonds), 1.5), bond_weight=np.full(len(bonds), 2500.0),
        bond_slack=np.r_[np.zeros(len(bonds) - 2), 0.1, 0.1],
        angle_i_seqs=angles, angle_ideal=np.full(n, 110.0), angle_weight=np.full(n, 0.1),
        dihedral_i_seqs=dih, dihedral_angle_ideal=np.full(n, 60.0), dihedral_weight=np.full(n, 3.0),
        dihedral_periodicity=np.array([1, 2, 3] * (n // 3), dtype=np.int32),
    )
    return xyz, pr


def chain(n, seed=0, shake=0.3):
    """Ideal-geometry polymer chain (bond 1.5 A, angle 110 deg, random dihedrals) with restraints at ideal values."""
    rng = np.random.default_rng(seed)
    xyz = np.zeros((n, 3))
    xyz[1] = [1.5, 0, 0]
    for i in range(2, n):
        b1 = xyz[i - 1] - xyz[i - 2]
        b1 /= np.linalg.norm(b1)
        r = rng.normal(size=3)
        r -= r.dot(b1) * b1
        r /= np.linalg.norm(r)
        th = np.deg2rad(180 - 110)
        xyz[i] = xyz[i - 1] + 1.5 * (np.cos(th) * b1 + np.sin(th) * r)
    bonds = np.array([[i, i + 1] for i in range(n - 1)], dtype=np.int32)
    angles = np.array([[i, i + 1, i + 2] for i in range(n - 2)], dtype=np.int32)
    dih = np.array([[i, i + 1, i + 2, i + 3] for i in range(n - 3)], dtype=np.int32)

    def ang(a, b, c):
        v1, v2 = a - b, c - b
        return np.degrees(np.arccos(v1.dot(v2) / np.linalg.norm(v1) / np.linalg.norm(v2)))

    def dihed(p0, p1, p2, p3):
        b1, b2, b3 = p1 - p0, p2 - p1, p3 - p2
        n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
        n1 /= np.linalg.norm(n1)
        n2 /= np.linalg.norm(n2)
        m1 = np.cross(n1, b2 / np.linalg.norm(b2))
        return np.degrees(np.arctan2(m1.dot(n2), n1.dot(n2)))

    pr = PackedRestraints(
        n_sites=n,
        bond_i_seqs=bonds, bond_distance_ideal=np.full(n - 1, 1.5), bond_weight=np.full(n - 1, 1 / 0.02**2),
        bond_slack=np.zeros(n - 1),
        angle_i_seqs=angles, angle_ideal=np.array([ang(*xyz[a]) for a in angles]), angle_weight=np.full(n - 2, 1 / 3.0**2),
        dihedral_i_seqs=dih, dihedral_angle_ideal=np.array([dihed(*xyz[d]) for d in dih]), dihedral_weight=np.full(n - 3, 1.0),
        dihedral_periodicity=np.ones(n - 3, dtype=np.int32),
    )
    start = xyz + shake * rng.normal(size=xyz.shape)
    return xyz, start, pr


def _curv(pr):
    return RestraintCurvature(_RestraintTables.from_packed(pr, device="cpu", dtype=torch.float64))


# ------------------------------------------------------------------ exactness
def test_residual_energy_matches_tables_energy():
    xyz, pr = _ring()
    t = _RestraintTables.from_packed(pr, device="cpu", dtype=torch.float64)
    c = RestraintCurvature(t)
    x = torch.as_tensor(xyz)
    assert abs(float(t.energy(x)) - float(c.energy_from_residuals(x))) < 1e-9 * float(t.energy(x))


def test_gn_blocks_sparse_and_hvps_match_dense_autograd():
    xyz, pr = _ring()
    c = _curv(pr)
    x = torch.as_tensor(xyz)
    n = xyz.shape[0]
    J = torch.autograd.functional.jacobian(lambda z: c.residuals(z)[0], x).reshape(-1, 3 * n)
    w = c.residuals(x)[1]
    H_gn = (2 * J.T @ (w[:, None] * J)).numpy()
    B = c.gn_blocks(x).numpy()
    for i in range(n):
        np.testing.assert_allclose(B[i], H_gn[3 * i : 3 * i + 3, 3 * i : 3 * i + 3], atol=1e-9 * np.abs(H_gn).max())
    np.testing.assert_allclose(c.gn_diagonal(x).numpy().reshape(-1), np.diag(H_gn), atol=1e-9 * np.abs(H_gn).max())
    rows, cols, vals, n3 = gn_sparse_coo(c, x)
    import scipy.sparse as sp

    H_sp = sp.csr_matrix((vals, (rows, cols)), shape=(n3, n3)).toarray()
    np.testing.assert_allclose(H_sp, H_gn, atol=1e-9 * np.abs(H_gn).max())
    v = torch.as_tensor(np.random.default_rng(1).normal(size=(n, 3)))
    np.testing.assert_allclose(c.gn_hvp(x, v).numpy().reshape(-1), H_gn.dot(v.numpy().reshape(-1)), rtol=1e-9, atol=1e-8)
    H_full = torch.autograd.functional.hessian(lambda z: c.energy_from_residuals(z), x).reshape(3 * n, 3 * n).numpy()
    np.testing.assert_allclose(c.full_hvp(x, v).numpy().reshape(-1), H_full.dot(v.numpy().reshape(-1)), rtol=1e-9, atol=1e-8)
    assert np.linalg.eigvalsh(H_gn).min() > -1e-8 * np.abs(H_gn).max()  # GN is PSD
    assert np.linalg.eigvalsh(H_full).min() < 0  # full Hessian is not, away from the minimum


def test_gn_preconditioner_solves_assembled_system():
    xyz, pr = _ring()
    c = _curv(pr)
    x = torch.as_tensor(xyz)
    n3 = 3 * xyz.shape[0]
    extra = np.linspace(1.0, 2.0, n3)
    M = GaussNewtonPreconditioner(c, x, extra_diag=extra, damping=0.0, weight=0.5)
    r = np.random.default_rng(2).normal(size=n3)
    p = M.solve(r)
    np.testing.assert_allclose(M.matvec(p), r, atol=1e-9 * np.abs(r).max())
    s = jacobi_scale(c.gn_diagonal(x))
    assert bool((s > 0).all())


# ------------------------------------------------------------------ optimizers
def test_gauss_newton_optimizer_converges_where_lbfgs_crawls():
    xyz, start, pr = chain(120)
    out_gn, st_gn = energy_and_sites(start, pr, max_iterations=40, optimizer="gauss_newton")
    out_lb, st_lb = energy_and_sites(start, pr, max_iterations=100, optimizer="lbfgs")
    assert st_gn["after"] < 1e-5 * st_gn["before"]
    assert st_gn["after"] < 0.1 * st_lb["after"]
    assert st_gn["n_steps"] <= 40 and "gn_final_damping" in st_gn
    # bonds restored to 1.5 A
    d = np.linalg.norm(out_gn[:-1] - out_gn[1:], axis=1)
    assert np.abs(d - 1.5).max() < 1e-3


def test_diagonal_preconditioner_option_runs_for_all_optimizers():
    xyz, start, pr = chain(40, shake=0.1)
    for opt, iters in (("lbfgs", 30), ("adam", 30), ("sgd", 10)):
        out, st = energy_and_sites(start, pr, max_iterations=iters, optimizer=opt, preconditioner="diagonal", precond_refresh=10)
        assert st["preconditioner"] == "diagonal" and st["after"] < st["before"]
        assert "curvature" in st and st["curvature"]["condition_estimate"] > 1
        assert np.isfinite(out).all()
    with pytest.raises(ValueError, match="preconditioner"):
        energy_and_sites(start, pr, max_iterations=1, preconditioner="bogus")


# ------------------------------------------------------------------ ops
def test_geometry_curvature_ops_round_trip():
    xyz, pr = _ring()
    bridge = Bridge(memory=True, timeout=10)
    sites = PackedCartesian(xyz)
    out = bridge.call("geometry_curvature", sites=sites, restraints=pr, params={"sparse": True})
    c = _curv(pr)
    x = torch.as_tensor(xyz)
    np.testing.assert_allclose(np.asarray(out["blocks"]), c.gn_blocks(x).numpy(), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(np.asarray(out["diagonal"]), c.gn_diagonal(x).numpy(), rtol=1e-10, atol=1e-10)
    xr = x.clone().requires_grad_(True)
    (g,) = torch.autograd.grad(c.energy_from_residuals(xr), xr)
    np.testing.assert_allclose(np.asarray(out["gradient"]), g.numpy(), rtol=1e-10, atol=1e-10)
    assert out["stats"]["nnz_coo"] == len(np.asarray(out["vals"]))
    v = np.random.default_rng(3).normal(size=xyz.shape)
    hv = bridge.call("geometry_hvp", sites=sites, restraints=pr, v=v, params={"hessian": "gn"})
    np.testing.assert_allclose(np.asarray(hv), c.gn_hvp(x, torch.as_tensor(v)).numpy(), rtol=1e-10, atol=1e-10)
    hv_full = bridge.call("geometry_hvp", sites=sites, restraints=pr, v=v, params={"hessian": "full"})
    assert not np.allclose(np.asarray(hv_full), np.asarray(hv))
    sol = bridge.call("geometry_gn_solve", sites=sites, restraints=pr, rhs=v, params={"damping": 1e-2, "weight": 2.0})
    M = GaussNewtonPreconditioner(c, x, damping=1e-2, weight=2.0)
    np.testing.assert_allclose(np.asarray(sol["solution"]), M.solve(v.reshape(-1)).reshape(v.shape), rtol=1e-9, atol=1e-12)
    assert sol["stats"]["nnz"] == M.nnz
    mini = bridge.call(
        "geometry_minimize", sites=sites, restraints=pr, params={"optimizer": "gauss_newton", "max_iterations": 30}
    )
    assert mini["target"]["after"] < 1e-3 * mini["target"]["before"]  # ring restraints are inconsistent: nonzero minimum
    assert mini["target"]["optimizer"] == "gauss_newton"


def test_remote_geometry_curvature_api():
    from phridge.client import RemoteGeometry
    from phridge.models import Atom, CoordinateFrame
    from phridge.packing_xtal import PackedHierarchy

    xyz, pr = _ring()
    n = xyz.shape[0]
    hier = PackedHierarchy(
        xyz=xyz, occupancy=np.ones(n), b_iso=np.full(n, 20.0),
        atoms=[Atom(i=i, name=" C  ", element="C", chain_id="A", resseq=i, resname="RNG") for i in range(n)],
        frame=CoordinateFrame.cartesian,
    )
    geo = RemoteGeometry(Bridge(memory=True, timeout=10), hier, pr)
    cur = geo.curvature()
    assert cur["blocks"].shape == (n, 3, 3) and cur["gradient"].shape == (n, 3)
    v = np.ones((n, 3))
    assert geo.hvp(v).shape == (n, 3)
    p = geo.gn_solve(-cur["gradient"], damping=1e-3)
    assert p.shape == (n, 3) and geo.last_gn_solve_stats["nnz"] > 0
    sites = geo.minimize(max_iterations=30, optimizer="gauss_newton", update_hierarchy=False)
    assert geo.last_target["after"] < 1e-3 * geo.last_target["before"]
    assert np.asarray(sites.xyz).shape == (n, 3)


def test_block_tridiagonal_exact_on_chain_and_op():
    from phridge.worker.geometry.curvature import BlockTridiagonalPreconditioner

    n = 60
    xyz, start, pr = chain(n)
    c = _curv(pr)
    x = torch.as_tensor(start)
    groups = [np.arange(s, min(s + 5, n)) for s in range(0, n, 5)]
    extra = np.repeat(np.linspace(1, 5, n), 3)
    tri = BlockTridiagonalPreconditioner(c, x, groups, extra_diag=extra, damping=1e-3)
    full = GaussNewtonPreconditioner(c, x, extra_diag=extra, damping=1e-3)
    r = np.random.default_rng(4).normal(size=3 * n)
    np.testing.assert_allclose(tri.solve(r), full.solve(r), rtol=1e-9, atol=1e-12)  # chain restraints never skip a group
    with pytest.raises(ValueError, match="partition"):
        BlockTridiagonalPreconditioner(c, x, groups[:-1])
    bridge = Bridge(memory=True, timeout=10)
    out = bridge.call(
        "geometry_gn_solve", sites=PackedCartesian(start), restraints=pr, rhs=r.reshape(n, 3), extra_diag=extra.reshape(n, 3),
        params={"method": "tridiagonal", "groups": [g.tolist() for g in groups], "damping": 1e-3},
    )
    np.testing.assert_allclose(np.asarray(out["solution"]).reshape(-1), tri.solve(r), rtol=1e-12)
    assert out["stats"]["method"] == "tridiagonal" and out["stats"]["n_groups"] == len(groups)


def test_geometry_gn_solve_adp_and_joint_fallback():
    """Verify geometry_gn_solve works when adp_params is omitted (testing math.log fallback)."""
    from phridge.client.adp_restraints import ADPPriorOptions, build_adp_restraints
    from phridge.worker.geometry.adp import ADPPriorTables

    n = 10
    xyz = np.random.randn(n, 3) * 2.0
    bonds = np.array([[i, i + 1] for i in range(n - 1)], dtype=np.int32)
    angles = np.array([[i, i + 1, i + 2] for i in range(n - 2)], dtype=np.int32)
    is_aniso = np.array([False, True] * 5, dtype=bool)

    pr = PackedRestraints(
        n_sites=n,
        bond_i_seqs=bonds,
        bond_distance_ideal=np.full(len(bonds), 1.5),
        bond_weight=np.ones(len(bonds)),
        bond_slack=np.zeros(len(bonds)),
        angle_i_seqs=angles,
        angle_ideal=np.full(len(angles), 110.0),
        angle_weight=np.ones(len(angles)),
    )

    opts = ADPPriorOptions(
        tau_1_2=0.15,
        tau_1_3=0.20,
        tau_sphere=0.35,
        sphere_radius=4.5,
        wilson_b=25.0,
        nu=4.0,
        level_weight=0.5,
    )
    pr_adp = build_adp_restraints(xyz, pr, options=opts, is_aniso=is_aniso)
    tables = ADPPriorTables.from_packed(pr_adp)
    n_adp = tables.n_params
    packed_sites = PackedCartesian(xyz=xyz)

    bridge = Bridge(memory=True)

    # 1. ADP block solve without adp_params
    rhs_adp = np.random.randn(n_adp)
    out_adp = bridge.call(
        "geometry_gn_solve",
        rhs=rhs_adp,
        sites=packed_sites,
        restraints=pr_adp,
        params={"blocks": ["adp"], "weight": 1.0, "damping": 1e-3, "method": "sparse"},
    )
    assert len(out_adp["solution"]) == n_adp

    # 2. Joint solve without adp_params
    rhs_joint = np.random.randn(3 * n + n_adp)
    out_joint = bridge.call(
        "geometry_gn_solve",
        rhs=rhs_joint,
        sites=packed_sites,
        restraints=pr_adp,
        params={"blocks": ["sites", "adp"], "weight": 1.0, "damping": 1e-3, "method": "sparse"},
    )
    assert len(out_joint["solution"]) == len(rhs_joint)
