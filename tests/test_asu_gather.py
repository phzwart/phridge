"""ASU stamp + agentsg algebraic F matches expand-to-P1."""

from __future__ import annotations

import numpy as np
import pytest

from phridge.sfcalc.engine.engine import EngineParams, ScatteringModel, StructureFactorEngine
from phridge.sfcalc.engine.sg_gather import agentsg_available, resolve_space_group


def _p212121(n_atoms: int = 12):
    rng = np.random.default_rng(3)
    rot = np.array(
        [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
            [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
            [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
        ],
        dtype=np.float64,
    )
    trans = np.array([[0, 0, 0], [0.5, 0, 0.5], [0, 0.5, 0.5], [0.5, 0.5, 0]], dtype=np.float64)
    n = int(n_atoms)
    return ScatteringModel(
        unit_cell=(40.0, 42.0, 44.0, 90.0, 90.0, 90.0),
        sites_frac=rng.random((n, 3)),
        occupancy=np.ones(n),
        u_iso=rng.uniform(0.02, 0.08, size=n),
        u_star=np.zeros((n, 6)),
        anisotropic=np.zeros(n, dtype=bool),
        fp=np.zeros(n),
        fdp=np.zeros(n),
        type_index=np.zeros(n, dtype=np.int64),
        gauss_a=np.array([[2.31, 1.02, 1.5886, 0.865]], dtype=np.float64),
        gauss_b=np.array([[20.8439, 10.2075, 0.5687, 51.6512]], dtype=np.float64),
        gauss_c=np.array([0.2156], dtype=np.float64),
        rot=rot,
        trans=trans,
        multiplicity=np.full(n, 4, dtype=np.int64),
        space_group_hall="P 2ac 2ab",
    )


def _hkl(model, d_min: float = 3.0) -> np.ndarray:
    from phridge.sfcalc.engine.cell import reciprocal_cartesian

    a, b, c = (float(x) for x in model.unit_cell[:3])
    hs = [
        (h, k, l)
        for h in range(-int(a / d_min) - 1, int(a / d_min) + 2)
        for k in range(-int(b / d_min) - 1, int(b / d_min) + 2)
        for l in range(-int(c / d_min) - 1, int(c / d_min) + 2)
        if (h, k, l) != (0, 0, 0)
    ]
    hkl = np.array(hs, dtype=np.int64)
    dstar2 = np.sum(reciprocal_cartesian(model.unit_cell, hkl) ** 2, axis=1)
    return hkl[dstar2 <= (1.0 / d_min) ** 2 + 1e-12]


@pytest.mark.parametrize("backend", ["torch", "numba", "cpp"])
def test_asu_f_matches_p1_expand(backend):
    pytest.importorskip("torch")
    if not agentsg_available():
        pytest.skip("agentsg required")
    if backend == "numba":
        from phridge.sfcalc.engine.stamp_numba import numba_available

        if not numba_available():
            pytest.skip("numba required")
    if backend == "cpp":
        from phridge.sfcalc.engine.stamp_cpp import cpp_available

        if not cpp_available():
            pytest.skip("C++ stamp required")
    model = _p212121()
    d_min = 3.0
    hkl = _hkl(model, d_min)
    asu = StructureFactorEngine(
        model, hkl, EngineParams(d_min=d_min, quality_factor=100.0, stamp_backend=backend)
    )
    exp = StructureFactorEngine(
        model,
        hkl,
        EngineParams(d_min=d_min, quality_factor=100.0, stamp_backend=backend, p1_expand=True),
    )
    assert asu._asu_stamp
    assert not exp._asu_stamp
    f_a = asu.f_calc_numpy()
    f_e = exp.f_calc_numpy()
    den = float(np.sum(np.abs(f_e)))
    r = float(np.sum(np.abs(f_a - f_e)) / max(den, 1e-300))
    # Grid splat of Wx+w is not bit-equal to the reciprocal sum; amplitudes must match.
    r_abs = float(np.sum(np.abs(np.abs(f_a) - np.abs(f_e))) / max(den, 1e-300))
    assert r_abs < 2e-2, f"{backend} R(|F|) asu vs expand = {r_abs}"
    assert 0.5 < float(np.median(np.abs(f_a)) / max(np.median(np.abs(f_e)), 1e-300)) < 2.0
    assert r < 5e-2, f"{backend} R(F) asu vs expand = {r}"


def _ops_to_rt(ops):
    rot = np.array([[[float(x) for x in row] for row in op.W.rows] for op in ops], dtype=np.float64)
    trans = np.array([[float(x) for x in op.w.v] for op in ops], dtype=np.float64)
    return rot, trans


def _r3h(n_atoms: int = 10):
    from agentsg import space_group

    rot, trans = _ops_to_rt(space_group(146).operations())
    rng = np.random.default_rng(4)
    n = int(n_atoms)
    return ScatteringModel(
        unit_cell=(48.0, 48.0, 54.0, 90.0, 90.0, 120.0),
        sites_frac=rng.random((n, 3)),
        occupancy=np.ones(n),
        u_iso=rng.uniform(0.02, 0.08, size=n),
        u_star=np.zeros((n, 6)),
        anisotropic=np.zeros(n, dtype=bool),
        fp=np.zeros(n),
        fdp=np.zeros(n),
        type_index=np.zeros(n, dtype=np.int64),
        gauss_a=np.array([[2.31, 1.02, 1.5886, 0.865]], dtype=np.float64),
        gauss_b=np.array([[20.8439, 10.2075, 0.5687, 51.6512]], dtype=np.float64),
        gauss_c=np.array([0.2156], dtype=np.float64),
        rot=rot,
        trans=trans,
        multiplicity=np.full(n, 9, dtype=np.int64),
        space_group_hall=" R 3",
    )


def test_resolve_r3h_aliases():
    if not agentsg_available():
        pytest.skip("agentsg required")
    for key in (146, "R 3", " R 3", "R 3 :H", "R3:H", "r3h"):
        sg = resolve_space_group(key)
        assert sg.number == 146
        assert sg.order() == 9


@pytest.mark.parametrize("backend", ["torch", "cpp"])
def test_r3h_asu_matches_p1_expand(backend):
    pytest.importorskip("torch")
    if not agentsg_available():
        pytest.skip("agentsg required")
    if backend == "cpp":
        from phridge.sfcalc.engine.stamp_cpp import cpp_available

        if not cpp_available():
            pytest.skip("C++ stamp required")
    model = _r3h()
    d_min = 3.0
    hkl = _hkl(model, d_min)
    asu = StructureFactorEngine(
        model, hkl, EngineParams(d_min=d_min, quality_factor=100.0, stamp_backend=backend)
    )
    exp = StructureFactorEngine(
        model,
        hkl,
        EngineParams(d_min=d_min, quality_factor=100.0, stamp_backend=backend, p1_expand=True),
    )
    assert asu._asu_stamp and asu.mate_index.shape[1] == 9
    f_a = asu.f_calc_numpy()
    f_e = exp.f_calc_numpy()
    den = float(np.sum(np.abs(f_e)))
    r = float(np.sum(np.abs(f_a - f_e)) / max(den, 1e-300))
    assert r < 1e-8, f"{backend} R3:H R(F) asu vs expand = {r}"


def _assert_agarwal_matches_autograd(model, d_min: float = 3.0):
    hkl = _hkl(model, d_min)
    eng = StructureFactorEngine(
        model, hkl, EngineParams(d_min=d_min, quality_factor=100.0, stamp_backend="torch")
    )
    assert eng._asu_stamp
    rng = np.random.default_rng(1)
    dtdf = rng.normal(size=len(hkl)) + 1j * rng.normal(size=len(hkl))
    agarwal = eng.gradients(dtdf)
    tape = eng._gradients_autograd(dtdf)
    for key in ("site_frac", "occupancy", "u_iso"):
        a = np.asarray(agarwal[key]).ravel()
        b = np.asarray(tape[key]).ravel()
        rel = float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))
        assert rel < 1e-7, f"{key} rel={rel}"


def test_asu_agarwal_matches_autograd():
    pytest.importorskip("torch")
    if not agentsg_available():
        pytest.skip("agentsg required")
    _assert_agarwal_matches_autograd(_p212121(8))


def test_r3h_agarwal_matches_autograd():
    pytest.importorskip("torch")
    if not agentsg_available():
        pytest.skip("agentsg required")
    _assert_agarwal_matches_autograd(_r3h(8))
