"""NUFFT structure-factor engine: backend conventions, accuracy, and grads.

Backend tests run without cctbx. CCTBX comparisons skip when cctbx is absent.
The whole module skips if ``pytorch_finufft`` is not installed.
"""

from __future__ import annotations

import math
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pytorch_finufft")

from phridge.sfcalc.engine.nufft_backend import nufft_type1, nufft_type2  # noqa: E402


def _fft_mode_indices(n: int) -> np.ndarray:
    """Miller indices on an FFT-order (modeord=1) axis of length ``n``."""
    return np.array([i if i < (n + 1) // 2 else i - n for i in range(n)], dtype=np.int64)


def _brute_type1(x_frac: np.ndarray, weights: np.ndarray, n_modes: tuple[int, int, int]) -> np.ndarray:
    """Σ_j w_tj exp(+2πi h·x_j) on the FFT-order mode grid."""
    n1, n2, n3 = n_modes
    h1 = _fft_mode_indices(n1)
    h2 = _fft_mode_indices(n2)
    h3 = _fft_mode_indices(n3)
    hh, kk, ll = np.meshgrid(h1, h2, h3, indexing="ij")
    hkl = np.stack([hh.ravel(), kk.ravel(), ll.ravel()], axis=1).astype(np.float64)
    with np.errstate(all="ignore"):
        phase = np.exp(2j * math.pi * (hkl @ x_frac.T))  # (N_modes, M)
        if weights.ndim == 1:
            grid = phase @ weights
            return grid.reshape(n1, n2, n3)
        return (weights @ phase.T).reshape(weights.shape[0], n1, n2, n3)


def test_backend_conventions_type1_type2_adjoint():
    """10 random points: type-1/type-2 vs brute force; type-2 is the adjoint of type-1."""
    rng = np.random.default_rng(0)
    m = 10
    n_modes = (7, 8, 6)
    x_frac = rng.random((m, 3))
    weights = rng.normal(size=m) + 1j * rng.normal(size=m)
    weights = weights.astype(np.complex128)

    xt = torch.as_tensor(x_frac, dtype=torch.float64)
    wt = torch.as_tensor(weights, dtype=torch.complex128)
    grid = nufft_type1(xt, wt, n_modes, eps=1e-12)
    ref = _brute_type1(x_frac, weights, n_modes)
    err = np.max(np.abs(grid.detach().cpu().numpy() - ref))
    assert err < 1e-10, f"type1 vs brute max abs err {err}"

    y = rng.normal(size=n_modes) + 1j * rng.normal(size=n_modes)
    y = y.astype(np.complex128)
    yt = torch.as_tensor(y, dtype=torch.complex128)
    adj = nufft_type2(xt, yt, eps=1e-12)

    h1 = _fft_mode_indices(n_modes[0])
    h2 = _fft_mode_indices(n_modes[1])
    h3 = _fft_mode_indices(n_modes[2])
    hh, kk, ll = np.meshgrid(h1, h2, h3, indexing="ij")
    hkl = np.stack([hh.ravel(), kk.ravel(), ll.ravel()], axis=1).astype(np.float64)
    with np.errstate(all="ignore"):
        phase = np.exp(-2j * math.pi * (x_frac @ hkl.T))  # (M, N_modes)  isign=-1
        ref_adj = phase @ y.ravel()
    err2 = np.max(np.abs(adj.detach().cpu().numpy() - ref_adj))
    assert err2 < 1e-10, f"type2 vs brute max abs err {err2}"

    lhs = np.vdot(y.ravel(), grid.detach().cpu().numpy().ravel())
    rhs = np.vdot(adj.detach().cpu().numpy(), weights)
    rel = abs(lhs - rhs) / max(abs(lhs), 1e-300)
    assert rel < 1e-10, f"adjoint <y,Ax> vs <AHy,x> rel {rel} lhs={lhs} rhs={rhs}"


# ----------------------------------------------------------------- engine helpers


def _r_factor(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.abs(a - b).sum() / max(np.abs(b).sum(), 1e-300))


def _structure(space_group="P21", elements=("C", "N", "O", "S"), n_repeat=3, aniso=False, anomalous=False, seed=0):
    pytest.importorskip("cctbx")
    import random

    from cctbx import sgtbx
    from cctbx.array_family import flex
    from cctbx.development import random_structure

    random.seed(seed)
    flex.set_random_seed(seed)
    xs = random_structure.xray_structure(
        space_group_info=sgtbx.space_group_info(space_group),
        elements=list(elements) * n_repeat,
        volume_per_atom=50,
        random_u_iso=True,
        random_occupancy=True,
        use_u_aniso=aniso,
    )
    for sc in xs.scatterers():
        if anomalous:
            sc.fp = 0.3
            sc.fdp = 0.7
        sc.flags.set_grad_site(True)
        sc.flags.set_grad_occupancy(True)
        sc.flags.set_grad_u_iso(sc.flags.use_u_iso())
        sc.flags.set_grad_u_aniso(sc.flags.use_u_aniso())
        sc.flags.set_grad_fp(anomalous)
        sc.flags.set_grad_fdp(anomalous)
    return xs


def _set_uniform_u_iso(xs, u_iso: float = 0.04):
    for sc in xs.scatterers():
        sc.u_iso = float(u_iso)
        sc.flags.set_use_u_iso(True)
        sc.flags.set_use_u_aniso(False)


def _nufft_engine(xs, hkl, d_min, **kwargs):
    from phridge.client.convert_xtal import scattering_table_from_cctbx, xray_from_cctbx
    from phridge.sfcalc.engine.nufft_engine import NufftEngineParams, NufftStructureFactorEngine
    from phridge.sfcalc.ops import scattering_model

    model = scattering_model(xray_from_cctbx(xs), scattering_table_from_cctbx(xs))
    params = NufftEngineParams(d_min=d_min, **kwargs)
    return NufftStructureFactorEngine(model, hkl, params)


def _direct_f_calc(model, hkl, atom_indices=None) -> np.ndarray:
    """Exact expanded-atom summation for a subset of ASU atoms."""
    from phridge.sfcalc.engine.cell import reciprocal_cartesian, sym6_to_mat
    from phridge.sfcalc.engine.nufft_engine import TWO_PI2, _form_factors

    idx = np.arange(model.n_scatterers) if atom_indices is None else np.asarray(atom_indices, dtype=np.int64)
    hkl = np.asarray(hkl, dtype=np.float64).reshape(-1, 3)
    dstar = reciprocal_cartesian(model.unit_cell, hkl)
    dstar2 = np.sum(dstar**2, axis=1)
    stol2 = dstar2 / 4.0
    ff_type = _form_factors(model.gauss_a, model.gauss_b, model.gauss_c, stol2)
    rot = np.asarray(model.rot, dtype=np.float64)
    trans = np.asarray(model.trans, dtype=np.float64)
    o = None
    from phridge.sfcalc.engine.cell import orthogonalization_matrix

    o = orthogonalization_matrix(model.unit_cell)
    f = np.zeros(hkl.shape[0], dtype=np.complex128)
    n_sym = rot.shape[0]
    for j in idx:
        ga = ff_type[int(model.type_index[j])]
        fj = ga + float(model.fp[j]) + 1j * float(model.fdp[j])
        w = float(model.occupancy[j]) * float(model.multiplicity[j]) / float(n_sym)
        u_iso = float(model.u_iso[j])
        aniso = bool(model.anisotropic[j])
        if aniso:
            u_mat = sym6_to_mat(np.asarray(model.u_star[j : j + 1], dtype=np.float64))[0]
        for s in range(n_sym):
            x = rot[s] @ model.sites_frac[j] + trans[s]
            if aniso:
                u_sym = rot[s] @ u_mat @ rot[s].T
                u_cart = o @ u_sym @ o.T
                dw = np.exp(-TWO_PI2 * np.einsum("hi,ij,hj->h", dstar, u_cart, dstar))
            else:
                dw = np.exp(-TWO_PI2 * u_iso * dstar2)
            with np.errstate(all="ignore"):
                phase = np.exp(2j * math.pi * (hkl @ x))
            f += w * fj * dw * phase
    return f


def _toy_carbon_p1(n_atoms: int = 8, u_iso: float = 0.04, seed: int = 0, random_u: bool = False):
    """Minimal all-carbon P1 model (no cctbx)."""
    from phridge.sfcalc.engine.engine import ScatteringModel
    from phridge.sfcalc.engine.symmetry import identity_ops

    rng = np.random.default_rng(seed)
    rot, trans = identity_ops()
    u = np.full(n_atoms, u_iso) if not random_u else rng.uniform(0.01, 0.12, size=n_atoms)
    # eltbx-like carbon Gaussians (IT1992-ish)
    gauss_a = np.array([[2.31, 1.02, 1.5886, 0.865]], dtype=np.float64)
    gauss_b = np.array([[20.8439, 10.2075, 0.5687, 51.6512]], dtype=np.float64)
    gauss_c = np.array([0.2156], dtype=np.float64)
    return ScatteringModel(
        unit_cell=(12.0, 13.0, 14.0, 90.0, 90.0, 90.0),
        sites_frac=rng.random((n_atoms, 3)),
        occupancy=np.ones(n_atoms),
        u_iso=u,
        u_star=np.zeros((n_atoms, 6)),
        anisotropic=np.zeros(n_atoms, dtype=bool),
        fp=np.zeros(n_atoms),
        fdp=np.zeros(n_atoms),
        type_index=np.zeros(n_atoms, dtype=np.int64),
        gauss_a=gauss_a,
        gauss_b=gauss_b,
        gauss_c=gauss_c,
        rot=rot,
        trans=trans,
        multiplicity=np.ones(n_atoms, dtype=np.int64),
    )


def _miller_sphere(d_min: float, unit_cell, max_index: int = 8) -> np.ndarray:
    from phridge.sfcalc.engine.cell import reciprocal_cartesian

    hs = []
    for h in range(-max_index, max_index + 1):
        for k in range(-max_index, max_index + 1):
            for l in range(-max_index, max_index + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                hs.append((h, k, l))
    hkl = np.array(hs, dtype=np.int64)
    dstar2 = np.sum(reciprocal_cartesian(unit_cell, hkl) ** 2, axis=1)
    return hkl[dstar2 <= (1.0 / d_min) ** 2 + 1e-12]


def test_identical_atoms_n0_vs_direct_sum():
    """Isolates spreader/grid/gather/f_t without cctbx."""
    from phridge.sfcalc.engine.nufft_engine import NufftEngineParams, NufftStructureFactorEngine

    model = _toy_carbon_p1(n_atoms=8, u_iso=0.04, seed=1)
    d_min = 2.0
    eps = 1e-6
    hkl = _miller_sphere(d_min, model.unit_cell)
    eng = NufftStructureFactorEngine(model, hkl, NufftEngineParams(d_min=d_min, n_max=0, tau=1.0, eps=eps))
    assert len(eng.plan.groups) == 1
    mine = eng.f_calc_numpy()
    ref = _direct_f_calc(model, hkl)
    r = _r_factor(mine, ref)
    assert r <= 10.0 * eps, f"R(F)={r} vs 10*eps={10*eps}  n_refl={len(hkl)}"


def test_identical_atoms_n0_p1():
    pytest.importorskip("cctbx")
    from cctbx import xray  # noqa: F401

    xs = _structure("P1", elements=("C",), n_repeat=8, seed=1)
    _set_uniform_u_iso(xs, 0.04)
    d_min = 2.0
    eps = 1e-6
    fc = xs.structure_factors(d_min=d_min, algorithm="direct").f_calc()
    hkl = np.array(list(fc.indices()))
    eng = _nufft_engine(xs, hkl, d_min, n_max=0, tau=1.0, eps=eps)
    assert len(eng.plan.groups) == 1
    mine = eng.f_calc_numpy()
    ref = np.array(fc.data())
    r = _r_factor(mine, ref)
    assert r <= 10.0 * eps, f"R(F)={r} vs 10*eps={10*eps}"


@pytest.mark.parametrize("space_group", ["P21", "P212121", "C2", "R3:H"])
def test_identical_atoms_n0_symmetry(space_group):
    pytest.importorskip("cctbx")
    xs = _structure(space_group, elements=("C",), n_repeat=6, seed=2)
    _set_uniform_u_iso(xs, 0.05)
    d_min = 2.0
    eps = 1e-6
    fc = xs.structure_factors(d_min=d_min, algorithm="direct").f_calc()
    hkl = np.array(list(fc.indices()))
    eng = _nufft_engine(xs, hkl, d_min, n_max=0, tau=1.0, eps=eps)
    mine = eng.f_calc_numpy()
    ref = np.array(fc.data())
    r = _r_factor(mine, ref)
    assert r <= 10.0 * eps, f"{space_group} R(F)={r} vs 10*eps={10*eps}"


def test_plan_bound_is_honest_direct():
    """Per-group relative error vs direct sum of that group's atoms (no cctbx)."""
    from phridge.sfcalc.engine.nufft_engine import NufftEngineParams, NufftStructureFactorEngine

    model = _toy_carbon_p1(n_atoms=16, u_iso=0.04, seed=4, random_u=True)
    d_min = 2.0
    tau = 1e-4
    hkl = _miller_sphere(d_min, model.unit_cell)
    eng = NufftStructureFactorEngine(model, hkl, NufftEngineParams(d_min=d_min, n_max=2, tau=tau, eps=1e-8))
    params = eng.tensors()
    for gi, g in enumerate(eng.plan.groups):
        with torch.no_grad():
            f_g = eng.f_calc(*params, group_index=gi).cpu().numpy()
        ref = _direct_f_calc(eng.model, hkl, atom_indices=g.atom_indices)
        r = _r_factor(f_g, ref)
        assert r <= tau, f"group {gi} n={len(g.atom_indices)} R={r} > tau={tau} bound={g.lambda_max}"


def test_plan_bound_is_honest():
    pytest.importorskip("cctbx")
    xs = _structure("P21", elements=("C", "N", "O", "S"), n_repeat=3, seed=3)
    d_min = 2.0
    tau = 1e-4
    fc = xs.structure_factors(d_min=d_min, algorithm="direct").f_calc()
    hkl = np.array(list(fc.indices()))
    eng = _nufft_engine(xs, hkl, d_min, n_max=2, tau=tau, eps=1e-8)
    params = eng.tensors()
    for gi, g in enumerate(eng.plan.groups):
        with torch.no_grad():
            f_g = eng.f_calc(*params, group_index=gi).cpu().numpy()
        ref = _direct_f_calc(eng.model, hkl, atom_indices=g.atom_indices)
        r = _r_factor(f_g, ref)
        assert r <= tau, f"group {gi} type={g.type_index} n={len(g.atom_indices)} R={r} > tau={tau}"


def _toy_mixed_iso(n_repeat: int = 3, seed: int = 0):
    """C/N/O/S isotropic atoms with random U."""
    from phridge.sfcalc.engine.engine import ScatteringModel
    from phridge.sfcalc.engine.symmetry import identity_ops

    rng = np.random.default_rng(seed)
    n_types = 4
    n = n_types * n_repeat
    rot, trans = identity_ops()
    gauss_a = np.array(
        [
            [2.31, 1.02, 1.5886, 0.865],
            [12.2126, 3.1322, 2.0125, 1.1663],
            [3.0485, 2.2868, 1.5463, 0.867],
            [6.9053, 5.2034, 1.4379, 1.5863],
        ],
        dtype=np.float64,
    )
    gauss_b = np.array(
        [
            [20.8439, 10.2075, 0.5687, 51.6512],
            [0.0057, 9.8933, 28.9975, 0.5826],
            [13.2771, 5.7011, 0.3239, 32.9089],
            [1.4679, 22.2151, 0.2536, 56.172],
        ],
        dtype=np.float64,
    )
    gauss_c = np.array([0.2156, 0.0, 0.2508, 0.8669], dtype=np.float64)
    return ScatteringModel(
        unit_cell=(18.0, 16.0, 20.0, 90.0, 90.0, 90.0),
        sites_frac=rng.random((n, 3)),
        occupancy=rng.uniform(0.7, 1.0, size=n),
        u_iso=rng.uniform(0.01, 0.15, size=n),
        u_star=np.zeros((n, 6)),
        anisotropic=np.zeros(n, dtype=bool),
        fp=np.zeros(n),
        fdp=np.zeros(n),
        type_index=np.repeat(np.arange(n_types), n_repeat),
        gauss_a=gauss_a,
        gauss_b=gauss_b,
        gauss_c=gauss_c,
        rot=rot,
        trans=trans,
        multiplicity=np.ones(n, dtype=np.int64),
    )


def _toy_aniso_anom(n_atoms: int = 10, seed: int = 5, se_like: bool = False):
    """Mixed iso/aniso with anomalous scattering."""
    from phridge.sfcalc.engine.cell import orthogonalization_matrix
    from phridge.sfcalc.engine.engine import ScatteringModel
    from phridge.sfcalc.engine.symmetry import identity_ops

    rng = np.random.default_rng(seed)
    rot, trans = identity_ops()
    cell = (15.0, 14.0, 16.0, 90.0, 90.0, 90.0)
    o = orthogonalization_matrix(cell)
    o_inv = np.linalg.inv(o)
    aniso = np.zeros(n_atoms, dtype=bool)
    aniso[n_atoms // 2 :] = True
    u_iso = rng.uniform(0.02, 0.08, size=n_atoms)
    u_star = np.zeros((n_atoms, 6))
    for j in np.where(aniso)[0]:
        diag = rng.uniform(0.02, 0.08, size=3)
        u_cart = np.diag(diag)
        u_s = o_inv @ u_cart @ o_inv.T
        u_star[j] = [u_s[0, 0], u_s[1, 1], u_s[2, 2], u_s[0, 1], u_s[0, 2], u_s[1, 2]]
    n_types = 2 if se_like else 1
    type_index = np.zeros(n_atoms, dtype=np.int64)
    gauss_a = np.array([[2.31, 1.02, 1.5886, 0.865]], dtype=np.float64)
    gauss_b = np.array([[20.8439, 10.2075, 0.5687, 51.6512]], dtype=np.float64)
    gauss_c = np.array([0.2156], dtype=np.float64)
    fp = np.full(n_atoms, 0.3)
    fdp = np.full(n_atoms, 0.7)
    if se_like:
        type_index[-1] = 1
        gauss_a = np.vstack([gauss_a, np.array([[17.0, 5.0, 3.0, 2.0]])])
        gauss_b = np.vstack([gauss_b, np.array([[2.0, 0.5, 12.0, 30.0]])])
        gauss_c = np.append(gauss_c, 1.0)
        fp[-1] = -4.0
        fdp[-1] = 3.0
    return ScatteringModel(
        unit_cell=cell,
        sites_frac=rng.random((n_atoms, 3)),
        occupancy=np.ones(n_atoms),
        u_iso=u_iso,
        u_star=u_star,
        anisotropic=aniso,
        fp=fp,
        fdp=fdp,
        type_index=type_index,
        gauss_a=gauss_a,
        gauss_b=gauss_b,
        gauss_c=gauss_c,
        rot=rot,
        trans=trans,
        multiplicity=np.ones(n_atoms, dtype=np.int64),
    )


@pytest.mark.parametrize("d_min", [2.0, 1.5, 1.0])
def test_mixed_iso_nmax_vs_direct(d_min):
    from phridge.sfcalc.engine.nufft_engine import NufftEngineParams, NufftStructureFactorEngine

    model = _toy_mixed_iso(n_repeat=3, seed=6)
    hmax = int(np.ceil(max(model.unit_cell[:3]) / d_min) + 1)
    hkl = _miller_sphere(d_min, model.unit_cell, max_index=hmax)
    tau = 1e-4
    eng = NufftStructureFactorEngine(model, hkl, NufftEngineParams(d_min=d_min, n_max=2, tau=tau, eps=1e-8))
    ref = _direct_f_calc(model, hkl)
    r2 = _r_factor(eng.f_calc_numpy(), ref)
    assert r2 <= 3e-4, f"d_min={d_min} n_max=2 R(F)={r2}"
    plan = eng.plan
    rs = []
    for n in (0, 1, 2, 3):
        eng.apply_plan(plan.with_order(n))
        rs.append(_r_factor(eng.f_calc_numpy(), ref))
    for a, b, n in zip(rs, rs[1:], (0, 1, 2)):
        assert b <= a + 1e-12, f"d_min={d_min} R not monotone at n={n}->{n+1}: {rs}"


def test_mixed_iso_cctbx():
    pytest.importorskip("cctbx")
    xs = _structure("P21", elements=("C", "N", "O", "S"), n_repeat=3, seed=3)
    for d_min in (2.0, 1.5, 1.0):
        fc = xs.structure_factors(d_min=d_min, algorithm="direct").f_calc()
        hkl = np.array(list(fc.indices()))
        ref = np.array(fc.data())
        eng = _nufft_engine(xs, hkl, d_min, n_max=2, tau=1e-4, eps=1e-8)
        r = _r_factor(eng.f_calc_numpy(), ref)
        assert r <= 3e-4, f"d_min={d_min} R(F)={r}"
        plan = eng.plan
        rs = []
        for n in (0, 1, 2, 3):
            eng.apply_plan(plan.with_order(n))
            rs.append(_r_factor(eng.f_calc_numpy(), ref))
        for a, b in zip(rs, rs[1:]):
            assert b <= a + 1e-12, f"cctbx monotone failed {rs}"


def test_aniso_anomalous_vs_direct():
    from phridge.sfcalc.engine.nufft_engine import NufftEngineParams, NufftStructureFactorEngine

    d_min = 1.5
    for se_like in (False, True):
        model = _toy_aniso_anom(n_atoms=10, seed=7, se_like=se_like)
        hkl = _miller_sphere(d_min, model.unit_cell, max_index=12)
        eng = NufftStructureFactorEngine(model, hkl, NufftEngineParams(d_min=d_min, n_max=2, tau=1e-4, eps=1e-8))
        r = _r_factor(eng.f_calc_numpy(), _direct_f_calc(model, hkl))
        assert r <= 3e-4, f"se_like={se_like} R(F)={r}"


def test_aniso_anomalous_cctbx():
    pytest.importorskip("cctbx")
    xs = _structure("P21", elements=("C", "N", "O", "S"), n_repeat=3, aniso=True, anomalous=True, seed=8)
    d_min = 1.5
    fc = xs.structure_factors(d_min=d_min, algorithm="direct").f_calc()
    hkl = np.array(list(fc.indices()))
    eng = _nufft_engine(xs, hkl, d_min, n_max=2, tau=1e-4, eps=1e-8)
    r = _r_factor(eng.f_calc_numpy(), np.array(fc.data()))
    assert r <= 3e-4, f"aniso+anom R(F)={r}"
    xs2 = _structure("P1", elements=("C", "C", "Se"), n_repeat=2, aniso=True, anomalous=True, seed=9)
    for sc in xs2.scatterers():
        if sc.scattering_type == "Se":
            sc.fdp = 3.0
    fc2 = xs2.structure_factors(d_min=d_min, algorithm="direct").f_calc()
    hkl2 = np.array(list(fc2.indices()))
    eng2 = _nufft_engine(xs2, hkl2, d_min, n_max=2, tau=1e-4, eps=1e-8)
    r2 = _r_factor(eng2.f_calc_numpy(), np.array(fc2.data()))
    assert r2 <= 3e-4, f"Se-like R(F)={r2}"


def test_chunking_invariance():
    from phridge.sfcalc.engine.nufft_engine import NufftEngineParams, NufftStructureFactorEngine

    model = _toy_mixed_iso(n_repeat=3, seed=10)
    d_min = 2.0
    hkl = _miller_sphere(d_min, model.unit_cell)
    vals = []
    for t_chunk in (1, 4, 1000):
        eng = NufftStructureFactorEngine(
            model, hkl, NufftEngineParams(d_min=d_min, n_max=2, tau=1e-4, eps=1e-8, t_chunk=t_chunk)
        )
        vals.append(eng.f_calc_numpy())
    for other in vals[1:]:
        assert np.allclose(vals[0], other, rtol=0, atol=1e-12), "chunking changed F"


def test_cross_engine_vs_stamp():
    from phridge.sfcalc.engine.engine import EngineParams, StructureFactorEngine
    from phridge.sfcalc.engine.nufft_engine import NufftEngineParams, NufftStructureFactorEngine

    # Identical atoms: NUFFT is exact at n_max=0. Stamp at quality_factor=1000 on the
    # default 1/3 grid is only ~8e-4 vs direct; a slightly finer grid keeps the
    # comparison inside the 5e-4 bound without changing quality_factor.
    model = _toy_carbon_p1(n_atoms=8, u_iso=0.04, seed=1)
    d_min = 2.0
    hkl = _miller_sphere(d_min, model.unit_cell)
    nufft = NufftStructureFactorEngine(model, hkl, NufftEngineParams(d_min=d_min, n_max=0, tau=1.0, eps=1e-6))
    stamp = StructureFactorEngine(
        model, hkl, EngineParams(d_min=d_min, quality_factor=1000, grid_resolution_factor=0.2, stamp_backend="torch")
    )
    r = _r_factor(nufft.f_calc_numpy(), stamp.f_calc_numpy())
    assert r <= 5e-4, f"cross-engine R(F)={r}"


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-300 or nb < 1e-300:
        return 1.0 if na < 1e-300 and nb < 1e-300 else 0.0
    return float(np.dot(a, b) / (na * nb))


def _rel_norm(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def test_gradients_match_finite_difference():
    """Autograd gradients vs central FD of Q = Σ Re[conj(G) F] (truncated model)."""
    from phridge.sfcalc.engine.nufft_engine import NufftEngineParams, NufftStructureFactorEngine

    model = _toy_mixed_iso(n_repeat=2, seed=12)
    d_min = 2.0
    hkl = _miller_sphere(d_min, model.unit_cell)
    eng = NufftStructureFactorEngine(model, hkl, NufftEngineParams(d_min=d_min, n_max=2, tau=1e-5, eps=1e-8))
    rng = np.random.default_rng(0)
    dtdf = rng.normal(size=len(hkl)) + 1j * rng.normal(size=len(hkl))
    grads = eng.gradients(dtdf)

    def q_of(params_np):
        tensors = [torch.as_tensor(x, dtype=torch.float64) for x in params_np]
        with torch.no_grad():
            f = eng.f_calc(*tensors).cpu().numpy()
        return float(np.sum(np.real(np.conj(dtdf) * f)))

    names = ("site_frac", "occupancy", "u_iso", "u_star", "fp", "fdp")
    base = [np.asarray(x, dtype=np.float64).copy() for x in (model.sites_frac, model.occupancy, model.u_iso, model.u_star, model.fp, model.fdp)]
    h = 1e-6
    fd = []
    for i, arr in enumerate(base):
        gfd = np.zeros_like(arr)
        it = np.nditer(arr, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            plus = [a.copy() for a in base]
            minus = [a.copy() for a in base]
            plus[i][idx] += h
            minus[i][idx] -= h
            gfd[idx] = (q_of(plus) - q_of(minus)) / (2 * h)
            it.iternext()
        fd.append(gfd)
        if names[i] == "u_star":
            continue  # all-iso model, u_star unused
        assert _cosine(grads[names[i]], gfd) >= 0.99999, f"{names[i]} cosine {_cosine(grads[names[i]], gfd)}"
        assert _rel_norm(grads[names[i]], gfd) <= 1e-3, f"{names[i]} rel {_rel_norm(grads[names[i]], gfd)}"


def test_gradients_match_cctbx_direct():
    pytest.importorskip("cctbx")
    from cctbx import xray
    from cctbx.array_family import flex

    for aniso in (False, True):
        xs = _structure("P21", elements=("C", "N", "O", "S"), n_repeat=3, aniso=aniso, anomalous=True, seed=0)
        d_min = 1.6
        fc = xs.structure_factors(d_min=d_min, algorithm="direct").f_calc()
        hkl = np.array(list(fc.indices()))
        rng = np.random.default_rng(0)
        dtdf = rng.normal(size=len(hkl)) + 1j * rng.normal(size=len(hkl))
        ref = xray.structure_factors.gradients_direct(
            xray_structure=xs,
            u_iso_refinable_params=None,
            miller_set=fc,
            d_target_d_f_calc=flex.complex_double(dtdf),
            n_parameters=0,
        )._results
        mine = _nufft_engine(xs, hkl, d_min, n_max=2, tau=1e-5, eps=1e-8).gradients(dtdf)
        pairs = [
            ("site_frac", ref.d_target_d_site_frac()),
            ("occupancy", ref.d_target_d_occupancy()),
            ("fp", ref.d_target_d_fp()),
            ("fdp", ref.d_target_d_fdp()),
        ]
        if aniso:
            mask = np.array([sc.flags.use_u_aniso() for sc in xs.scatterers()])
            pairs.append(("u_star", np.asarray(ref.d_target_d_u_star())[mask]))
            mine_u = mine["u_star"][mask]
        else:
            mask = np.array([sc.flags.use_u_iso() for sc in xs.scatterers()])
            pairs.append(("u_iso", np.asarray(ref.d_target_d_u_iso())[mask]))
            mine["u_iso"] = mine["u_iso"][mask]
        for name, r in pairs:
            m = mine_u if name == "u_star" else mine[name]
            assert _cosine(m, r) >= 0.99999, f"{name} aniso={aniso} cosine {_cosine(m, r)}"
            assert _rel_norm(m, r) <= 1e-3, f"{name} aniso={aniso} rel {_rel_norm(m, r)}"


def test_jvp_finite_difference():
    from phridge.sfcalc.engine.nufft_engine import NufftEngineParams, NufftStructureFactorEngine

    model = _toy_mixed_iso(n_repeat=2, seed=13)
    d_min = 2.0
    hkl = _miller_sphere(d_min, model.unit_cell)
    eng = NufftStructureFactorEngine(model, hkl, NufftEngineParams(d_min=d_min, n_max=2, tau=1e-5, eps=1e-8))
    rng = np.random.default_rng(2)
    params = eng.tensors()
    tangents = []
    for i, p in enumerate(params):
        t = rng.normal(size=tuple(p.shape)).astype(np.float64)
        if i == 3:  # u_star unused for iso
            t[:] = 0.0
        tangents.append(t)
    _f, df = eng.jvp(tangents)
    eps = 1e-6
    plus = [p.detach() + eps * torch.as_tensor(t, dtype=p.dtype) for p, t in zip(params, tangents)]
    minus = [p.detach() - eps * torch.as_tensor(t, dtype=p.dtype) for p, t in zip(params, tangents)]
    with torch.no_grad():
        fd = ((eng.f_calc(*plus) - eng.f_calc(*minus)) / (2 * eps)).cpu().numpy()
    df_np = df.cpu().numpy()
    denom = max(np.linalg.norm(fd), 1e-300)
    rel = float(np.linalg.norm(df_np - fd) / denom)
    assert rel < 1e-4, f"jvp vs FD rel {rel}"
