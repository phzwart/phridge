"""Joint x-ray + geometry Newton-CG with the factorised geometry Gauss-Newton preconditioner."""

from __future__ import annotations

import numpy as np
import pytest

cctbx = pytest.importorskip("cctbx")
from cctbx import sgtbx  # noqa: E402
from cctbx.array_family import flex  # noqa: E402
from cctbx.development import random_structure  # noqa: E402

torch = pytest.importorskip("torch")

from phridge.client import Bridge  # noqa: E402
from phridge.client.geometry import RemoteGeometry  # noqa: E402
from phridge.client.joint import JointSiteRefinement  # noqa: E402
from phridge.models import Atom, CoordinateFrame, SfEngineParams  # noqa: E402
from phridge.packing_geometry import PackedRestraints  # noqa: E402
from phridge.packing_xtal import PackedHierarchy  # noqa: E402
from phridge.sfcalc.client import RemoteRefinementTarget  # noqa: E402


def _setup(seed=0):
    import random

    random.seed(seed)
    flex.set_random_seed(seed)
    xs = random_structure.xray_structure(
        space_group_info=sgtbx.space_group_info("P21"), elements=["C", "N", "O"] * 12, volume_per_atom=40, random_u_iso=True
    )
    for sc in xs.scatterers():
        sc.flags.set_grad_site(True)
    n = xs.scatterers().size()
    X = np.asarray(xs.sites_cart(), dtype=np.float64)
    from scipy.spatial import cKDTree

    bonds = set((i, i + 1) for i in range(n - 1))
    _, idx = cKDTree(X).query(X, k=3)
    for i in range(n):
        for j in idx[i, 1:]:
            bonds.add(tuple(sorted((i, int(j)))))
    bonds = np.array(sorted(bonds), dtype=np.int32)
    d0 = np.linalg.norm(X[bonds[:, 0]] - X[bonds[:, 1]], axis=1)
    angles = np.array([[i, i + 1, i + 2] for i in range(n - 2)], dtype=np.int32)

    def ang(a, b, c):
        v1, v2 = a - b, c - b
        return np.degrees(np.arccos(np.clip(v1.dot(v2) / np.linalg.norm(v1) / np.linalg.norm(v2), -1, 1)))

    pr = PackedRestraints(
        n_sites=n,
        bond_i_seqs=bonds, bond_distance_ideal=d0, bond_weight=np.full(len(bonds), 1 / 0.02**2), bond_slack=np.zeros(len(bonds)),
        angle_i_seqs=angles, angle_ideal=np.array([ang(*X[a]) for a in angles]), angle_weight=np.full(len(angles), 1 / 3.0**2),
    )
    hier = PackedHierarchy(
        xyz=X, occupancy=np.ones(n), b_iso=np.full(n, 20.0),
        atoms=[Atom(i=i, name=" C  ", element="C", chain_id="A", resseq=i, resname="X") for i in range(n)],
        frame=CoordinateFrame.cartesian,
    )
    bridge = Bridge(memory=True, timeout=60)
    geo = RemoteGeometry(bridge, hier, pr)
    f_obs = xs.structure_factors(d_min=2.5, algorithm="direct").f_calc().amplitudes()
    xs2 = xs.deep_copy_scatterers()
    xs2.shake_sites_in_place(rms_difference=0.25)
    refiner = RemoteRefinementTarget(
        bridge, xs2, f_obs, {"name": "ls", "obs_type": "F"}, params=SfEngineParams(d_min=2.5, quality_factor=100)
    )
    return xs, xs2, refiner, geo


def test_joint_newton_cg_geometry_preconditioner():
    xs, xs2, refiner, geo = _setup()
    j0 = JointSiteRefinement(refiner, geo, 1.0)
    q, gx, _, _ = j0._xray(xs2)
    e, ge = j0._geom(np.asarray(xs2.sites_cart()))
    w = 20.0 * float(np.linalg.norm(gx) / np.linalg.norm(ge))  # geometry-dominated weight
    rms0 = xs.rms_difference(xs2)
    t0 = JointSiteRefinement(refiner, geo, w).total(xs2)

    pre = JointSiteRefinement(refiner, geo, w).newton_cg(xs2, max_iterations=6, cg_max_iter=25, precondition=True)
    plain = JointSiteRefinement(refiner, geo, w).newton_cg(xs2, max_iterations=6, cg_max_iter=25, precondition=False)
    rms_pre = xs.rms_difference(pre["xray_structure"])
    rms_plain = xs.rms_difference(plain["xray_structure"])
    assert pre["final_target"] < 1e-3 * t0 and plain["final_target"] < 1e-2 * t0
    assert rms_pre < 0.05 * rms0
    assert rms_pre <= rms_plain
    assert pre["n_hvp"] < plain["n_hvp"]  # the factorised geometry GN cuts CG work
    assert all(h["accepted"] for h in pre["history"])


def test_joint_refinement_sites_and_adp_blocks():
    xs, xs2, refiner, geo = _setup()
    for sc in xs2.scatterers():
        sc.flags.set_grad_u_iso(True)
    j_joint = JointSiteRefinement(
        refiner,
        geo,
        weight=1.0,
        weight_adp=1.0,
        refine=("sites", "adp"),
        hessian_geom="diagonal",
    )
    t0 = j_joint.total(xs2)
    assert not np.isnan(t0)
    assert not np.isinf(t0)
    out = j_joint.newton_cg(xs2, max_iterations=3, cg_max_iter=10, precondition=True)
    assert out["final_target"] <= t0
    assert not np.isnan(out["final_target"])

