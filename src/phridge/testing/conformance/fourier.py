"""Fourier conformance: direct-sum reference, adjoint VJP, declared dtype."""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from phridge.contracts.tensors import RObject

_COS_MIN = math.cos(math.radians(0.3))
_MAG_TOL = 5e-4
_RF_TOL = 2e-3


def _toy(n_atoms: int, seed: int = 0) -> Any:
    from phridge.transforms.fourier.ten_eyck.engine import ScatteringModel
    from phridge.transforms.fourier.ten_eyck.symmetry import identity_ops

    rng = np.random.default_rng(seed)
    rot, trans = identity_ops()
    gauss_a = np.array([[2.31, 1.02, 1.5886, 0.865]], dtype=np.float64)
    gauss_b = np.array([[20.8439, 10.2075, 0.5687, 51.6512]], dtype=np.float64)
    gauss_c = np.array([0.2156], dtype=np.float64)
    return ScatteringModel(
        unit_cell=(12.0, 13.0, 14.0, 90.0, 90.0, 90.0),
        sites_frac=rng.random((n_atoms, 3)),
        occupancy=np.ones(n_atoms),
        u_iso=np.full(n_atoms, 0.04),
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


def _hkl(unit_cell: tuple, d_min: float = 4.0, max_index: int = 2) -> np.ndarray:
    from phridge.transforms.fourier.ten_eyck.cell import reciprocal_cartesian

    rows = []
    for h in range(-max_index, max_index + 1):
        for k in range(-max_index, max_index + 1):
            for l in range(-max_index, max_index + 1):
                if h == k == l == 0:
                    continue
                rows.append((h, k, l))
    hkl = np.asarray(rows, dtype=np.float64)
    dstar = reciprocal_cartesian(unit_cell, hkl)
    keep = np.sqrt(np.sum(dstar**2, axis=1)) <= 1.0 / d_min
    return hkl[keep]


def _record(model: Any) -> RObject:
    import torch

    n = model.n_scatterers
    return RObject(
        sites_frac=torch.tensor(np.asarray(model.sites_frac), dtype=torch.float64),
        u_iso=torch.tensor(np.asarray(model.u_iso), dtype=torch.float64),
        u_star=torch.tensor(np.asarray(model.u_star), dtype=torch.float64),
        occupancy=torch.tensor(np.asarray(model.occupancy), dtype=torch.float64),
        fp=torch.tensor(np.asarray(model.fp), dtype=torch.float64),
        fdp=torch.tensor(np.asarray(model.fdp), dtype=torch.float64),
        scatterer_type=torch.tensor(np.asarray(model.type_index), dtype=torch.int64),
        uij_defined=torch.zeros(n, dtype=torch.bool),
    )


def _hkl_tensor(hkl: np.ndarray) -> Any:
    import torch

    return torch.tensor(hkl, dtype=torch.int64)


def _rfactor(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).sum() / max(np.abs(b).sum(), 1e-300))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float64).reshape(-1)
    bv = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.dot(av, bv) / max(np.linalg.norm(av) * np.linalg.norm(bv), 1e-300))


def test_forward_matches_direct(build_operator: Callable[[Any, np.ndarray], Any]) -> None:
    """R(F) and site-gradient direction against the direct sum."""
    import torch

    from phridge.transforms.fourier.direct import DirectFourier

    model = _toy(8, seed=1)
    hkl = _hkl(model.unit_cell, d_min=2.0, max_index=6)
    record = _record(model)
    hkl_t = _hkl_tensor(hkl)
    operator = build_operator(model, hkl)
    direct = DirectFourier(model)
    got = operator.forward(record, hkl_t).values.detach().cpu().numpy()
    ref = direct.forward(record, hkl_t).values.detach().cpu().numpy()
    assert _rfactor(got, ref) < _RF_TOL
    g = torch.randn(hkl.shape[0], dtype=torch.complex128)
    g_op = operator.adjoint(record, hkl_t, g).sites_frac.detach().cpu().numpy()
    g_ref = direct.adjoint(record, hkl_t, g).sites_frac.detach().cpu().numpy()
    assert _cosine(g_op, g_ref) >= _COS_MIN
    ratio = np.linalg.norm(g_op) / max(np.linalg.norm(g_ref), 1e-300)
    assert abs(ratio - 1.0) < _MAG_TOL


def test_adjoint_is_vjp(build_operator: Callable[[Any, np.ndarray], Any]) -> None:
    """Adjoint matches a central difference of forward on a 20-atom P1 case."""
    import torch

    model = _toy(20, seed=2)
    hkl = _hkl(model.unit_cell, d_min=5.0, max_index=1)
    record = _record(model)
    hkl_t = _hkl_tensor(hkl)
    operator = build_operator(model, hkl)
    g = torch.randn(hkl.shape[0], dtype=torch.complex128)
    analytic = operator.adjoint(record, hkl_t, g).sites_frac.detach()
    eps = 1e-6
    numeric = torch.zeros_like(analytic)
    sites = record.sites_frac.detach()
    for i in range(min(4, sites.shape[0])):
        for k in range(3):
            plus = sites.clone()
            minus = sites.clone()
            plus[i, k] += eps
            minus[i, k] -= eps
            fp = operator.forward(record._replace(sites_frac=plus), hkl_t).values
            fm = operator.forward(record._replace(sites_frac=minus), hkl_t).values
            numeric[i, k] = ((fp - fm) * g.conj()).real.sum() / (2.0 * eps)
    assert _cosine(analytic[:4].numpy(), numeric[:4].numpy()) >= _COS_MIN
    ratio = np.linalg.norm(analytic[:4].numpy()) / max(np.linalg.norm(numeric[:4].numpy()), 1e-300)
    assert abs(ratio - 1.0) < 5e-2


def test_dtype_declared(manifest: Any) -> None:
    assert manifest.dtype in ("float32", "float64")
