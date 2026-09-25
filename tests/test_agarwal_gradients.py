"""Hand-coded Agarwal VJP matches the old autograd tape on the FFT model."""

from __future__ import annotations

import numpy as np
import pytest

from phridge.sfcalc.engine.engine import EngineParams, ScatteringModel, StructureFactorEngine
from phridge.sfcalc.engine.symmetry import identity_ops


def _toy(n_atoms: int = 8):
    rng = np.random.default_rng(2)
    rot, trans = identity_ops()
    n = int(n_atoms)
    return ScatteringModel(
        unit_cell=(32.0, 34.0, 36.0, 90.0, 90.0, 90.0),
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
        multiplicity=np.ones(n, dtype=np.int64),
    )


def _hkl(model, d_min: float = 2.5, max_index: int = 6) -> np.ndarray:
    from phridge.sfcalc.engine.cell import reciprocal_cartesian

    a, b, c = (float(x) for x in model.unit_cell[:3])
    hm = min(max_index, int(np.ceil(a / d_min) + 1))
    km = min(max_index, int(np.ceil(b / d_min) + 1))
    lm = min(max_index, int(np.ceil(c / d_min) + 1))
    hs = [
        (h, k, l)
        for h in range(-hm, hm + 1)
        for k in range(-km, km + 1)
        for l in range(-lm, lm + 1)
        if (h, k, l) != (0, 0, 0)
    ]
    hkl = np.array(hs, dtype=np.int64)
    dstar2 = np.sum(reciprocal_cartesian(model.unit_cell, hkl) ** 2, axis=1)
    return hkl[dstar2 <= (1.0 / d_min) ** 2 + 1e-12]


@pytest.mark.parametrize("backend", ["torch", "numba", "cpp", "cuda"])
def test_agarwal_matches_autograd(backend):
    pytest.importorskip("torch")
    device = "cpu"
    if backend == "numba":
        from phridge.sfcalc.engine.stamp_numba import numba_available

        if not numba_available():
            pytest.skip("numba required")
    if backend == "cpp":
        from phridge.sfcalc.engine.stamp_cpp import cpp_available

        if not cpp_available():
            pytest.skip("C++ stamp extension required")
    if backend == "cuda":
        from phridge.sfcalc.engine.stamp_cuda import cuda_available

        if not cuda_available():
            pytest.skip("CUDA stamp extension required")
        device = "cuda"
    model = _toy()
    d_min = 2.5
    hkl = _hkl(model, d_min)
    eng = StructureFactorEngine(
        model, hkl, EngineParams(d_min=d_min, quality_factor=100.0, stamp_backend=backend), device=device
    )
    rng = np.random.default_rng(0)
    dtdf = rng.normal(size=len(hkl)) + 1j * rng.normal(size=len(hkl))
    agarwal = eng.gradients(dtdf)
    tape = eng._gradients_autograd(dtdf)
    for key in ("site_frac", "occupancy", "u_iso", "u_star", "fp", "fdp"):
        a = np.asarray(agarwal[key]).ravel()
        b = np.asarray(tape[key]).ravel()
        if np.linalg.norm(b) < 1e-12 and np.linalg.norm(a) < 1e-12:
            continue
        rel = float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))
        assert rel < 1e-8, f"{backend} {key} rel={rel}"


def test_cpp_f_matches_numba():
    pytest.importorskip("torch")
    from phridge.sfcalc.engine.stamp_cpp import cpp_available
    from phridge.sfcalc.engine.stamp_numba import numba_available

    if not cpp_available():
        pytest.skip("C++ stamp extension required")
    if not numba_available():
        pytest.skip("numba required")
    model = _toy()
    d_min = 2.5
    hkl = _hkl(model, d_min)
    numba = StructureFactorEngine(model, hkl, EngineParams(d_min=d_min, quality_factor=100.0, stamp_backend="numba"))
    cpp = StructureFactorEngine(model, hkl, EngineParams(d_min=d_min, quality_factor=100.0, stamp_backend="cpp"))
    f_n = numba.f_calc_numpy()
    f_c = cpp.f_calc_numpy()
    den = float(np.sum(np.abs(f_n)))
    r = float(np.sum(np.abs(f_c - f_n)) / max(den, 1e-300))
    assert r < 5e-4, f"R(F) cpp vs numba = {r}"


def test_rfft_f_matches_complex_fftn():
    torch = pytest.importorskip("torch")
    model = _toy()
    d_min = 2.5
    hkl = _hkl(model, d_min)
    eng = StructureFactorEngine(
        model, hkl, EngineParams(d_min=d_min, quality_factor=100.0, stamp_backend="torch")
    )
    f_r = eng.f_calc_numpy()
    with torch.no_grad():
        rho = eng.density(*eng.tensors())
        assert not rho.is_complex()
        ft = torch.fft.fftn(rho.to(eng.cdtype)).reshape(-1)
        f_c = eng._f_from_fft(ft).detach().cpu().numpy()
    den = float(np.sum(np.abs(f_c)))
    r = float(np.sum(np.abs(f_r - f_c)) / max(den, 1e-300))
    assert r < 1e-9, f"R(F) rfft vs fftn = {r}"


def test_cuda_f_matches_cpp():
    pytest.importorskip("torch")
    from phridge.sfcalc.engine.stamp_cpp import cpp_available
    from phridge.sfcalc.engine.stamp_cuda import cuda_available

    if not cpp_available():
        pytest.skip("C++ stamp extension required")
    if not cuda_available():
        pytest.skip("CUDA stamp extension required")
    model = _toy()
    d_min = 2.5
    hkl = _hkl(model, d_min)
    cpp = StructureFactorEngine(model, hkl, EngineParams(d_min=d_min, quality_factor=100.0, stamp_backend="cpp"))
    gpu = StructureFactorEngine(
        model, hkl, EngineParams(d_min=d_min, quality_factor=100.0, stamp_backend="cuda"), device="cuda"
    )
    f_c = cpp.f_calc_numpy()
    f_g = gpu.f_calc_numpy()
    den = float(np.sum(np.abs(f_c)))
    r = float(np.sum(np.abs(f_g - f_c)) / max(den, 1e-300))
    assert r < 5e-4, f"R(F) cuda vs cpp = {r}"


def test_auto_mps_uses_cpp():
    torch = pytest.importorskip("torch")
    from phridge.sfcalc.engine.stamp_cpp import cpp_available

    if not cpp_available():
        pytest.skip("C++ stamp extension required")
    if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
        pytest.skip("MPS required")
    model = _toy()
    d_min = 2.5
    hkl = _hkl(model, d_min)
    mps = StructureFactorEngine(model, hkl, EngineParams(d_min=d_min, quality_factor=100.0), device="mps")
    cpu = StructureFactorEngine(
        model, hkl, EngineParams(d_min=d_min, quality_factor=100.0, stamp_backend="cpp")
    )
    assert mps._stamp_kind() == "cpp"
    f_m = mps.f_calc_numpy()
    f_c = cpu.f_calc_numpy()
    den = float(np.sum(np.abs(f_c)))
    r = float(np.sum(np.abs(f_m - f_c)) / max(den, 1e-300))
    assert r < 5e-4, f"R(F) mps-auto vs cpu-cpp = {r}"


@pytest.mark.parametrize("dtype,tol", [("float32", 2e-3), ("float16", 3e-2)])
def test_cpp_lowp_f_matches_f64(dtype, tol):
    pytest.importorskip("torch")
    from phridge.sfcalc.engine.stamp_cpp import cpp_available

    if not cpp_available():
        pytest.skip("C++ stamp extension required")
    model = _toy()
    d_min = 2.5
    hkl = _hkl(model, d_min)
    ref = StructureFactorEngine(model, hkl, EngineParams(d_min=d_min, quality_factor=100.0, stamp_backend="cpp"))
    low = StructureFactorEngine(
        model, hkl, EngineParams(d_min=d_min, quality_factor=100.0, stamp_backend="cpp", dtype=dtype)
    )
    f_r = ref.f_calc_numpy()
    f_l = low.f_calc_numpy()
    den = float(np.sum(np.abs(f_r)))
    r = float(np.sum(np.abs(f_l - f_r)) / max(den, 1e-300))
    assert r < tol, f"R(F) cpp {dtype} vs float64 = {r}"


def test_auto_cpu_stays_numba():
    pytest.importorskip("torch")
    from phridge.sfcalc.engine.stamp_numba import numba_available

    if not numba_available():
        pytest.skip("numba required")
    model = _toy()
    d_min = 2.5
    hkl = _hkl(model, d_min)
    eng = StructureFactorEngine(model, hkl, EngineParams(d_min=d_min, quality_factor=100.0))
    assert eng._stamp_kind() == "numba"


def test_fast_cpu_uses_cpp():
    pytest.importorskip("torch")
    from phridge.sfcalc.engine.stamp_cpp import cpp_available

    if not cpp_available():
        pytest.skip("C++ stamp required")
    model = _toy()
    d_min = 2.5
    hkl = _hkl(model, d_min)
    eng = StructureFactorEngine(model, hkl, EngineParams(d_min=d_min, stamp_backend="fast"))
    assert eng._stamp_kind() == "cpp"
