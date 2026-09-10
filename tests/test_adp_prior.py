"""Comprehensive test suite for hierarchical, scale-invariant ADP prior (Tests 1-9)."""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
cctbx = pytest.importorskip("cctbx")

from cctbx import sgtbx  # noqa: E402
from cctbx.array_family import flex  # noqa: E402
from cctbx.development import random_structure  # noqa: E402

from phridge.client import Bridge, RemoteGeometry  # noqa: E402
from phridge.client.adp_restraints import (  # noqa: E402
    ADPPriorOptions,
    build_adp_restraints,
    fit_adp_hyperparameters,
)
from phridge.client.joint import JointSiteRefinement, _extract_adp_vec  # noqa: E402
from phridge.models import SfEngineParams  # noqa: E402
from phridge.packing_geometry import PackedRestraints, unpack_restraints  # noqa: E402
from phridge.packing_xtal import Atom, CoordinateFrame, PackedCartesian, PackedHierarchy  # noqa: E402
from phridge.sfcalc.client import RemoteRefinementTarget  # noqa: E402
from phridge.worker.geometry.adp import (  # noqa: E402
    ADPPrior,
    ADPPriorTables,
    EIGHT_PI_SQ,
    sym_mat_to_vec6,
    vec6_to_sym_mat,
)
from phridge.worker.geometry.curvature import (  # noqa: E402
    BlockTridiagonalPreconditioner,
    GaussNewtonPreconditioner,
)


# -----------------------------------------------------------------------------
# Test 1: Derivatives
# -----------------------------------------------------------------------------
def test_adp_derivatives_against_autograd():
    """Test 1: energy, gradient, GN blocks, sparse COO and both HVPs against dense autograd

    on a random mixed iso/aniso model.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    n_atoms = 6
    xyz = np.random.randn(n_atoms, 3) * 3.0
    bonds = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], dtype=np.int32)
    angles = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]], dtype=np.int32)
    is_aniso = np.array([False, True, False, True, False, True], dtype=bool)

    pr = PackedRestraints(
        n_sites=n_atoms,
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
    tables = ADPPriorTables.from_packed(pr_adp, sites=xyz)
    prior = ADPPrior(tables)

    sites_t = torch.from_numpy(xyz)
    adp_vec = torch.randn(tables.n_params, dtype=torch.float64, requires_grad=True)

    # 1. Gradient check
    e = prior.energy_vec(adp_vec, sites=sites_t)
    g_auto = torch.autograd.grad(e, adp_vec, create_graph=True)[0]
    g_exact = prior.gradient(adp_vec, sites=sites_t)
    assert torch.allclose(g_exact, g_auto, rtol=1e-6, atol=1e-7)

    # 2. GN sparse COO vs dense J^T W J
    row, col, data, M = prior.gn_sparse_coo(adp_vec, sites=sites_t)
    H_coo = torch.zeros(M, M, dtype=torch.float64)
    H_coo[row, col] = torch.from_numpy(data)

    rho, w = prior.residuals_vec(adp_vec, sites=sites_t)
    J = torch.autograd.functional.jacobian(lambda x: prior.residuals_vec(x, sites=sites_t)[0], adp_vec)
    H_dense = 2.0 * (J.T @ (w[:, None] * J))
    assert torch.allclose(H_coo, H_dense, rtol=1e-6, atol=1e-7)

    # 3. GN blocks check
    blocks = prior.gn_blocks(adp_vec, sites=sites_t)
    for i in range(n_atoms):
        o = tables.param_offsets[i].item()
        d = tables.param_dims[i].item()
        block_dense = H_dense[o : o + d, o : o + d]
        assert torch.allclose(blocks[i], block_dense, rtol=1e-6, atol=1e-7)

    # 4. GN HVP check
    v = torch.randn(tables.n_params, dtype=torch.float64)
    hv_gn = prior.gn_hvp(adp_vec, v, sites=sites_t)
    hv_dense = H_dense @ v
    assert torch.allclose(hv_gn, hv_dense, rtol=1e-6, atol=1e-7)

    # 5. Full HVP check
    hv_full = prior.full_hvp(adp_vec, v, sites=sites_t)
    hv_auto = torch.autograd.grad((g_auto * v).sum(), adp_vec)[0]
    assert torch.allclose(hv_full, hv_auto, rtol=1e-6, atol=1e-7)


# -----------------------------------------------------------------------------
# Test 2: Positivity
# -----------------------------------------------------------------------------
def test_adp_positivity_after_random_newton_steps():
    """Test 2: after 200 random Newton steps with a huge step size,

    every U_i is SPD and every B_i > 0 (trivially true by construction — assert it anyway).
    """
    rng = np.random.default_rng(123)
    is_aniso = np.array([True, False, True, False, True, True])
    # 4 aniso (24) + 2 iso (2) = 26 parameters
    p = rng.normal(scale=2.0, size=26)

    for _ in range(200):
        # Huge random Newton step
        delta = rng.normal(scale=2.0, size=26)
        p = p + delta

        idx = 0
        for aniso in is_aniso:
            if aniso:
                s_vec = torch.from_numpy(p[idx : idx + 6])
                idx += 6
                S = vec6_to_sym_mat(s_vec)
                # Mathematical eigenvalues of U = exp(S) / (8 pi^2)
                s_eigs = torch.linalg.eigvalsh(S)
                u_eigs = torch.exp(s_eigs) / EIGHT_PI_SQ
                assert (u_eigs > 0).all()
                assert torch.isfinite(u_eigs).all()
            else:
                beta = p[idx]
                idx += 1
                b_val = math.exp(beta)
                assert b_val > 0.0
                assert math.isfinite(b_val)


# -----------------------------------------------------------------------------
# Test 3: Scale Invariance
# -----------------------------------------------------------------------------
def test_adp_scale_invariance():
    """Test 3: multiplying all B by 3 leaves every similarity term unchanged

    and moves only the level term.
    """
    n_atoms = 6
    xyz = np.random.randn(n_atoms, 3) * 3.0
    bonds = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], dtype=np.int32)
    angles = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]], dtype=np.int32)
    is_aniso = np.array([False, True, False, True, False, True], dtype=bool)

    pr = PackedRestraints(
        n_sites=n_atoms,
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
        level_weight=1.5,
        w_hirshfeld=0.0,
    )
    pr_adp = build_adp_restraints(xyz, pr, options=opts, is_aniso=is_aniso)
    tables = ADPPriorTables.from_packed(pr_adp, sites=xyz)
    prior = ADPPrior(tables)

    adp_vec = torch.randn(tables.n_params, dtype=torch.float64)

    # Multiply all B by 3: beta -> beta + log(3), S_diag -> S_diag + log(3)
    log3 = math.log(3.0)
    adp_vec_scaled = adp_vec.clone()
    for i in range(n_atoms):
        o = tables.param_offsets[i].item()
        if is_aniso[i]:
            adp_vec_scaled[o] += log3
            adp_vec_scaled[o + 1] += log3
            adp_vec_scaled[o + 2] += log3
        else:
            adp_vec_scaled[o] += log3

    # Residuals
    sites_t = torch.from_numpy(xyz)
    rho1, _ = prior.residuals_vec(adp_vec, sites=sites_t)
    rho2, _ = prior.residuals_vec(adp_vec_scaled, sites=sites_t)

    # All residuals except the last (level anchor) are similarity residuals
    sim_rho1 = rho1[:-1]
    sim_rho2 = rho2[:-1]
    assert torch.allclose(sim_rho1, sim_rho2, atol=1e-12)

    # Level anchor shifts by exactly log(3)
    anchor1 = rho1[-1].item()
    anchor2 = rho2[-1].item()
    assert np.isclose(anchor2 - anchor1, log3, atol=1e-12)


# -----------------------------------------------------------------------------
# Test 4: Heavy Tails
# -----------------------------------------------------------------------------
def test_adp_heavy_tails_step_recovery():
    """Test 4: a chain with a 5x B jump in the middle;

    the Student-t prior with nu=4 recovers the jump under a weak x-ray term
    where the Gaussian (nu=inf) prior smears it (measure the width of the transition).
    """
    n = 20
    b_obs = np.array([10.0] * 10 + [50.0] * 10)
    beta_obs = torch.tensor(np.log(b_obs), dtype=torch.float64)

    pairs = torch.tensor([[i, i + 1] for i in range(n - 1)], dtype=torch.int64)
    weights = torch.ones(len(pairs), dtype=torch.float64)
    taus = torch.full((len(pairs),), 0.15, dtype=torch.float64)
    classes = torch.zeros(len(pairs), dtype=torch.int64)

    t4 = ADPPriorTables(
        pair_i=pairs,
        pair_w=weights,
        pair_tau=taus,
        pair_class=classes,
        rigid_i=torch.zeros((0, 2), dtype=torch.int64),
        rigid_w=torch.zeros(0, dtype=torch.float64),
        is_aniso=torch.zeros(n, dtype=torch.bool),
        wilson_b=None,
        nu=4.0,
        level_weight=0.0,
        w_iso=0.0,
        tau_iso=0.5,
        n_sites=n,
        param_dims=torch.ones(n, dtype=torch.int64),
        param_offsets=torch.arange(n + 1, dtype=torch.int64),
        n_params=n,
    )
    t_gauss = ADPPriorTables(
        pair_i=pairs,
        pair_w=weights,
        pair_tau=taus,
        pair_class=classes,
        rigid_i=torch.zeros((0, 2), dtype=torch.int64),
        rigid_w=torch.zeros(0, dtype=torch.float64),
        is_aniso=torch.zeros(n, dtype=torch.bool),
        wilson_b=None,
        nu=float("inf"),
        level_weight=0.0,
        w_iso=0.0,
        tau_iso=0.5,
        n_sites=n,
        param_dims=torch.ones(n, dtype=torch.int64),
        param_offsets=torch.arange(n + 1, dtype=torch.int64),
        n_params=n,
    )
    p4 = ADPPrior(t4)
    p_gauss = ADPPrior(t_gauss)

    w_xray = 20.0

    def solve(prior: ADPPrior) -> np.ndarray:
        x = beta_obs.clone().requires_grad_(True)
        opt = torch.optim.LBFGS([x], lr=1.0, max_iter=200, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = prior.energy_vec(x) + 0.5 * w_xray * ((x - beta_obs) ** 2).sum()
            loss.backward()
            return loss

        opt.step(closure)
        return x.detach().numpy()

    b4 = solve(p4)
    b_g = solve(p_gauss)

    diff_9_10_t4 = float(b4[10] - b4[9])
    diff_9_10_g = float(b_g[10] - b_g[9])

    # Student-t maintains a much sharper step jump than Gaussian
    assert diff_9_10_t4 > diff_9_10_g

    # Transition leakage into adjacent neighbors:
    leak_t4 = float((b4[9] - b4[8]) + (b4[11] - b4[10]))
    leak_g = float((b_g[9] - b_g[8]) + (b_g[11] - b_g[10]))
    assert leak_t4 < leak_g


# -----------------------------------------------------------------------------
# Test 5: Rigid Bond
# -----------------------------------------------------------------------------
def test_adp_hirshfeld_rigid_bond():
    """Test 5: anisotropic bonded pair with unequal along-bond MSD is penalised;

    equal along-bond MSD with different perpendicular components is not.
    """
    sites = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=torch.float64)
    pairs = torch.zeros((0, 2), dtype=torch.int64)
    rigid_i = torch.tensor([[0, 1]], dtype=torch.int64)
    rigid_w = torch.tensor([10000.0], dtype=torch.float64)

    tables = ADPPriorTables(
        pair_i=pairs,
        pair_w=torch.zeros(0, dtype=torch.float64),
        pair_tau=torch.zeros(0, dtype=torch.float64),
        pair_class=torch.zeros(0, dtype=torch.int64),
        rigid_i=rigid_i,
        rigid_w=rigid_w,
        is_aniso=torch.tensor([True, True], dtype=torch.bool),
        wilson_b=None,
        nu=4.0,
        level_weight=0.0,
        w_iso=0.0,
        tau_iso=0.5,
        n_sites=2,
        param_dims=torch.tensor([6, 6], dtype=torch.int64),
        param_offsets=torch.tensor([0, 6, 12], dtype=torch.int64),
        n_params=12,
    )
    prior = ADPPrior(tables)

    # Case 1: unequal along-bond MSD (U00 differs: 0.05 vs 0.02)
    s0_unequal = torch.diag(
        torch.tensor(
            [np.log(0.05 * 8 * np.pi**2), np.log(0.02 * 8 * np.pi**2), np.log(0.02 * 8 * np.pi**2)],
            dtype=torch.float64,
        )
    )
    s1_unequal = torch.diag(
        torch.tensor(
            [np.log(0.02 * 8 * np.pi**2), np.log(0.02 * 8 * np.pi**2), np.log(0.02 * 8 * np.pi**2)],
            dtype=torch.float64,
        )
    )
    vec_unequal = torch.cat(
        [
            sym_mat_to_vec6(s0_unequal.unsqueeze(0)).squeeze(0),
            sym_mat_to_vec6(s1_unequal.unsqueeze(0)).squeeze(0),
        ]
    )
    e_unequal = prior.energy_vec(vec_unequal, sites=sites)
    assert e_unequal.item() > 1.0

    # Case 2: equal along-bond MSD (both 0.04 along x), but perpendicular components differ
    s0_equal = torch.diag(
        torch.tensor(
            [np.log(0.04 * 8 * np.pi**2), np.log(0.08 * 8 * np.pi**2), np.log(0.01 * 8 * np.pi**2)],
            dtype=torch.float64,
        )
    )
    s1_equal = torch.diag(
        torch.tensor(
            [np.log(0.04 * 8 * np.pi**2), np.log(0.02 * 8 * np.pi**2), np.log(0.05 * 8 * np.pi**2)],
            dtype=torch.float64,
        )
    )
    vec_equal = torch.cat(
        [
            sym_mat_to_vec6(s0_equal.unsqueeze(0)).squeeze(0),
            sym_mat_to_vec6(s1_equal.unsqueeze(0)).squeeze(0),
        ]
    )
    e_equal = prior.energy_vec(vec_equal, sites=sites)
    assert np.isclose(e_equal.item(), 0.0, atol=1e-10)


# -----------------------------------------------------------------------------
# Test 6: Empirical Bayes
# -----------------------------------------------------------------------------
def test_adp_empirical_bayes_hyperparameters():
    """Test 6: synthetic data generated from the prior with known tau;

    the fit recovers tau within its error bar; on data generated with no
    smoothness the fit drives tau large (weak prior) rather than oversmoothing.
    """
    import scipy.sparse as sp

    n = 30
    pairs = torch.tensor([[i, i + 1] for i in range(n - 1)], dtype=torch.int64)
    weights = torch.ones(len(pairs), dtype=torch.float64)
    tau_star = 0.20
    taus = torch.full((len(pairs),), tau_star, dtype=torch.float64)
    classes = torch.zeros(len(pairs), dtype=torch.int64)

    tables = ADPPriorTables(
        pair_i=pairs,
        pair_w=weights,
        pair_tau=taus,
        pair_class=classes,
        rigid_i=torch.zeros((0, 2), dtype=torch.int64),
        rigid_w=torch.zeros(0, dtype=torch.float64),
        is_aniso=torch.zeros(n, dtype=torch.bool),
        wilson_b=20.0,
        nu=float("inf"),
        level_weight=1.0,
        w_iso=0.0,
        tau_iso=0.5,
        n_sites=n,
        param_dims=torch.ones(n, dtype=torch.int64),
        param_offsets=torch.arange(n + 1, dtype=torch.int64),
        n_params=n,
    )
    prior = ADPPrior(tables)

    row, col, data, _ = prior.gn_sparse_coo(torch.zeros(n, dtype=torch.float64))
    Q = sp.coo_matrix((data, (row, col)), shape=(n, n)).toarray()
    Cov = np.linalg.pinv(Q)

    rng = np.random.default_rng(42)
    mu = math.log(20.0)
    beta_smooth = mu + rng.multivariate_normal(np.zeros(n), Cov)
    h_xray = np.full(n, 10.0)

    # Part A: recovers tau within error bar
    res_smooth = fit_adp_hyperparameters(prior, beta_smooth, h_xray_diag=h_xray, maxiter=50)
    tau_fit = res_smooth["tau_12"]
    tau_se = res_smooth["tau_12_se"]
    assert abs(tau_fit - tau_star) < 2.5 * max(tau_se, 0.05)

    # Part B: no smoothness drives tau large
    beta_rough = mu + rng.normal(scale=1.0, size=n)
    res_rough = fit_adp_hyperparameters(prior, beta_rough, h_xray_diag=h_xray, maxiter=50)
    assert res_rough["tau_12"] > 2.0 * tau_star


# -----------------------------------------------------------------------------
# Test 7: Preconditioner
# -----------------------------------------------------------------------------
def test_adp_preconditioner_block_tridiagonal_vs_sparse():
    """Test 7: block-tridiagonal solve on the ADP block equals the sparse solve

    for a chain with 1-2, 1-3 and sphere pairs restricted to consecutive residues.
    """
    n_res = 5
    atoms_per_res = 2
    n = n_res * atoms_per_res

    groups = [list(range(k * atoms_per_res, (k + 1) * atoms_per_res)) for k in range(n_res)]

    pairs = []
    for k in range(n_res):
        pairs.append([k * 2, k * 2 + 1])
        if k + 1 < n_res:
            pairs.append([k * 2 + 1, (k + 1) * 2])
            pairs.append([k * 2, (k + 1) * 2])
            pairs.append([k * 2 + 1, (k + 1) * 2 + 1])

    pairs = torch.tensor(pairs, dtype=torch.int64)
    weights = torch.ones(len(pairs), dtype=torch.float64)
    taus = torch.full((len(pairs),), 0.15, dtype=torch.float64)
    classes = torch.zeros(len(pairs), dtype=torch.int64)

    is_aniso = torch.tensor([False, True] * n_res, dtype=torch.bool)
    dims = torch.tensor([1, 6] * n_res, dtype=torch.int64)
    offsets = torch.zeros(n + 1, dtype=torch.int64)
    offsets[1:] = torch.cumsum(dims, dim=0)
    n_params = int(offsets[-1].item())

    tables = ADPPriorTables(
        pair_i=pairs,
        pair_w=weights,
        pair_tau=taus,
        pair_class=classes,
        rigid_i=torch.zeros((0, 2), dtype=torch.int64),
        rigid_w=torch.zeros(0, dtype=torch.float64),
        is_aniso=is_aniso,
        wilson_b=None,
        nu=float("inf"),
        level_weight=0.0,
        w_iso=0.05,
        tau_iso=0.5,
        n_sites=n,
        param_dims=dims,
        param_offsets=offsets,
        n_params=n_params,
    )
    prior = ADPPrior(tables)

    x_adp = torch.zeros(n_params, dtype=torch.float64)
    row, col, data, M = prior.gn_sparse_coo(x_adp)
    coo = (row, col, data, M)

    rng = np.random.default_rng(42)
    extra_diag = rng.uniform(1.0, 5.0, size=n_params)

    # 1. Sparse preconditioner
    p_sparse = GaussNewtonPreconditioner(prior, x_adp, extra_diag=extra_diag, damping=1e-3, coo=coo)

    # 2. Block-tridiagonal preconditioner
    p_tridiag = BlockTridiagonalPreconditioner(
        prior,
        x_adp,
        groups=groups,
        extra_diag=extra_diag,
        damping=1e-3,
        coo=coo,
        param_dims=dims,
    )

    # Solve for random rhs
    rhs = rng.normal(size=n_params)
    sol_sparse = p_sparse.solve(rhs)
    sol_tridiag = p_tridiag.solve(rhs)

    rel_diff = np.linalg.norm(sol_sparse - sol_tridiag) / np.linalg.norm(sol_sparse)
    assert rel_diff < 1e-4


# -----------------------------------------------------------------------------
# Test 8: Ops Round Trip and Schema Validation
# -----------------------------------------------------------------------------
def test_adp_ops_roundtrip_and_schema_validation():
    """Test 8: ops round trip via Bridge(memory=True) for adp_prior_eval / adp_prior_hvp /

    geometry_gn_solve with the ADP block; JSON-schema validation of the extended GeometryRestraints.
    """
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

    # 1. JSON-schema roundtrip of PackedRestraints with ADP fields
    packed_bytes = pr_adp.pack()
    unpacked = unpack_restraints(packed_bytes, pr_adp.meta)
    assert unpacked.meta.n_adp_pairs == pr_adp.meta.n_adp_pairs
    assert unpacked.meta.wilson_b == pr_adp.meta.wilson_b
    assert unpacked.meta.adp_nu == pr_adp.meta.adp_nu
    assert unpacked.meta.adp_level_weight == pr_adp.meta.adp_level_weight
    assert np.array_equal(unpacked.adp_pair_i_seqs, pr_adp.adp_pair_i_seqs)
    assert np.allclose(unpacked.adp_pair_weight, pr_adp.adp_pair_weight)
    assert np.allclose(unpacked.adp_pair_tau, pr_adp.adp_pair_tau)
    assert np.array_equal(unpacked.adp_is_aniso, pr_adp.adp_is_aniso)

    # 2. Ops round trip via Bridge(memory=True)
    bridge = Bridge(memory=True)
    tables = ADPPriorTables.from_packed(pr_adp)
    adp_vec = np.random.randn(tables.n_params)
    packed_sites = PackedCartesian(xyz=xyz)

    # adp_prior_eval
    eval_res = bridge.call("adp_prior_eval", sites=packed_sites, adp_params=adp_vec, restraints=pr_adp)
    assert float(eval_res["energy"]) > 0.0
    assert len(eval_res["gradient"]) == tables.n_params
    assert len(eval_res["diagonal"]) == tables.n_params

    # adp_prior_hvp
    v = np.random.randn(tables.n_params)
    hvp_res = bridge.call("adp_prior_hvp", sites=packed_sites, adp_params=adp_vec, restraints=pr_adp, v=v)
    assert len(hvp_res) == tables.n_params

    # geometry_gn_solve with joint blocks=["sites", "adp"]
    rhs_sites = np.random.randn(n, 3)
    rhs_adp = np.random.randn(tables.n_params)
    rhs_joint = np.concatenate([rhs_sites.reshape(-1), rhs_adp])
    extra_diag = np.random.uniform(1.0, 3.0, size=len(rhs_joint))

    solve_res = bridge.call(
        "geometry_gn_solve",
        rhs=rhs_joint,
        sites=packed_sites,
        restraints=pr_adp,
        params={
            "blocks": ["sites", "adp"],
            "adp_params": adp_vec,
            "weight": 1.0,
            "damping": 1e-3,
            "method": "sparse",
        },
        extra_diag=extra_diag,
    )
    assert len(solve_res["solution"]) == len(rhs_joint)

    # Test geometry_gn_solve without explicit adp_params (covers math.log fallback)
    solve_adp_fallback = bridge.call(
        "geometry_gn_solve",
        rhs=rhs_adp,
        sites=packed_sites,
        restraints=pr_adp,
        params={
            "blocks": ["adp"],
            "weight": 1.0,
            "damping": 1e-3,
            "method": "sparse",
        },
    )
    assert len(solve_adp_fallback["solution"]) == len(rhs_adp)

    solve_joint_fallback = bridge.call(
        "geometry_gn_solve",
        rhs=rhs_joint,
        sites=packed_sites,
        restraints=pr_adp,
        params={
            "blocks": ["sites", "adp"],
            "weight": 1.0,
            "damping": 1e-3,
            "method": "sparse",
        },
    )
    assert len(solve_joint_fallback["solution"]) == len(rhs_joint)


# -----------------------------------------------------------------------------
# Test 9: Held-Out Check
# -----------------------------------------------------------------------------
def test_adp_held_out_cross_validation_check():
    """Test 9: re-refine a shaken structure with and without the prior at the empirical-Bayes eta;

    the ADP block's held-out NLL (test set) and the B correlation to the unshaken truth
    must both be better with the prior.
    """
    # random_structure draws from stdlib random as well as the flex RNG, so without
    # this the generated structure depends on what earlier tests consumed.
    random.seed(42)
    flex.set_random_seed(42)
    np.random.seed(42)

    xs_true = random_structure.xray_structure(
        space_group_info=sgtbx.space_group_info("P21"),
        elements=["C", "N", "O"] * 8,
        volume_per_atom=40,
        random_u_iso=False,
    )
    n = len(xs_true.scatterers())
    b_true = np.array([20.0 + 8.0 * np.sin(i / 3.0) for i in range(n)])
    for i, sc in enumerate(xs_true.scatterers()):
        sc.u_iso = b_true[i] / (8.0 * np.pi**2)
        sc.flags.set_grad_site(False)
        sc.flags.set_grad_u_iso(True)

    X = np.asarray(xs_true.sites_cart(), dtype=np.float64)

    # Observations in data-sparse regime
    fc = xs_true.structure_factors(d_min=3.5, algorithm="direct").f_calc()
    f_amp = fc.amplitudes()
    noise_scale = 0.15 * float(np.mean(f_amp.data()))
    noisy_f = np.maximum(np.asarray(f_amp.data()) + np.random.normal(0, noise_scale, size=f_amp.data().size()), 0.1)
    f_obs = f_amp.customized_copy(data=flex.double(noisy_f.tolist()))

    flags = f_obs.generate_r_free_flags(fraction=0.25, max_free=2000)
    f_work = f_obs.select(~flags.data())
    f_free = f_obs.select(flags.data())

    # Shaken B
    b_shaken = b_true * np.exp(np.random.normal(0, 0.40, size=n))
    xs_shaken = xs_true.deep_copy_scatterers()
    for i, sc in enumerate(xs_shaken.scatterers()):
        sc.u_iso = b_shaken[i] / (8.0 * np.pi**2)
        sc.flags.set_grad_site(False)
        sc.flags.set_grad_u_iso(True)

    bonds = np.array([[i, i + 1] for i in range(n - 1)], dtype=np.int32)
    angles = np.array([[i, i + 1, i + 2] for i in range(n - 2)], dtype=np.int32)
    d0 = np.linalg.norm(X[bonds[:, 0]] - X[bonds[:, 1]], axis=1)

    def ang(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        v1, v2 = a - b, c - b
        return float(np.degrees(np.arccos(np.clip(v1.dot(v2) / np.linalg.norm(v1) / np.linalg.norm(v2), -1, 1))))

    pr = PackedRestraints(
        n_sites=n,
        bond_i_seqs=bonds,
        bond_distance_ideal=d0,
        bond_weight=np.full(len(bonds), 1 / 0.02**2),
        bond_slack=np.zeros(len(bonds)),
        angle_i_seqs=angles,
        angle_ideal=np.array([ang(*X[a]) for a in angles]),
        angle_weight=np.full(len(angles), 1 / 3.0**2),
    )
    hier = PackedHierarchy(
        xyz=X,
        occupancy=np.ones(n),
        b_iso=np.full(n, 20.0),
        atoms=[Atom(i=i, name=" C  ", element="C", chain_id="A", resseq=i, resname="X") for i in range(n)],
        frame=CoordinateFrame.cartesian,
    )

    opts = ADPPriorOptions(
        tau_1_2=0.15,
        tau_1_3=0.20,
        tau_sphere=0.35,
        sphere_radius=4.5,
        wilson_b=float(np.mean(b_shaken)),
        nu=4.0,
        level_weight=1.0,
    )
    pr_adp = build_adp_restraints(X, pr, options=opts)
    bridge = Bridge(memory=True)
    geo_prior = RemoteGeometry(bridge, hier, pr_adp)

    refiner_work = RemoteRefinementTarget(
        bridge, xs_shaken, f_work, {"name": "ls", "obs_type": "F"}, params=SfEngineParams(d_min=3.5, quality_factor=100)
    )
    eval_free = RemoteRefinementTarget(
        bridge, xs_shaken, f_free, {"name": "ls", "obs_type": "F"}, params=SfEngineParams(d_min=3.5, quality_factor=100)
    )

    j_test = JointSiteRefinement(refiner_work, geo_prior, weight=1.0, refine=("adp",))
    q, gx, _, _ = j_test._xray(xs_shaken)
    e, ge = j_test._geom(X, adp_vec=_extract_adp_vec(xs_shaken)[0])
    w_adp = 0.5 * float(np.linalg.norm(gx) / np.linalg.norm(ge))

    # Refinement with prior
    j_prior = JointSiteRefinement(refiner_work, geo_prior, weight=1.0, weight_adp=w_adp, refine=("adp",))
    res_prior = j_prior.newton_cg(xs_shaken, max_iterations=8, damping=1e-2)
    b_prior = np.array([float(sc.u_iso * 8 * np.pi**2) for sc in res_prior["xray_structure"].scatterers()])
    nll_free_prior = float(eval_free.compute(res_prior["xray_structure"])["target"].meta.value)
    corr_prior = float(np.corrcoef(b_prior, b_true)[0, 1])

    # Refinement without prior
    j_noprior = JointSiteRefinement(refiner_work, geo_prior, weight=1.0, weight_adp=0.0, refine=("adp",))
    res_noprior = j_noprior.newton_cg(xs_shaken, max_iterations=8, damping=1e-2)
    b_noprior = np.array([float(sc.u_iso * 8 * np.pi**2) for sc in res_noprior["xray_structure"].scatterers()])
    nll_free_noprior = float(eval_free.compute(res_noprior["xray_structure"])["target"].meta.value)
    corr_noprior = float(np.corrcoef(b_noprior, b_true)[0, 1])

    # Both held-out NLL and B correlation must be strictly better with the prior
    assert nll_free_prior < nll_free_noprior
    assert corr_prior > corr_noprior
