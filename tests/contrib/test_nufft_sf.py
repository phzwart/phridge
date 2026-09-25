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
