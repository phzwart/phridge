"""Register ten_eyck and nufft against the Fourier conformance suite."""

from __future__ import annotations

import numpy as np
import pytest

from phridge.testing.conformance import fourier as fourier_suite
from phridge.transforms.fourier.direct import MANIFEST as DIRECT_MANIFEST
from phridge.transforms.fourier.nufft.operator import MANIFEST as NUFFT_MANIFEST
from phridge.transforms.fourier.ten_eyck.engine import EngineParams, StructureFactorEngine
from phridge.transforms.fourier.ten_eyck.operator import MANIFEST as TEN_EYCK_MANIFEST
from phridge.transforms.fourier.ten_eyck.operator import TenEyckFourier


def _ten_eyck(model, hkl: np.ndarray) -> TenEyckFourier:
    params = EngineParams(d_min=2.0, quality_factor=1000, grid_resolution_factor=0.2, stamp_backend="torch", dtype="float64")
    engine = StructureFactorEngine(model, hkl, params, device="cpu")
    return TenEyckFourier(engine)


def _nufft(model, hkl: np.ndarray):
    from phridge.sfcalc.backends.nufft_backend import nufft_available
    from phridge.transforms.fourier.nufft.nufft_engine import NufftEngineParams, NufftStructureFactorEngine
    from phridge.transforms.fourier.nufft.operator import NufftFourier

    if not nufft_available():
        pytest.skip("nufft backend is not installed")
    engine = NufftStructureFactorEngine(model, hkl, NufftEngineParams(d_min=2.0), device="cpu")
    return NufftFourier(engine)


def test_ten_eyck_forward_matches_direct() -> None:
    fourier_suite.test_forward_matches_direct(_ten_eyck)


def test_ten_eyck_adjoint_is_vjp() -> None:
    fourier_suite.test_adjoint_is_vjp(_ten_eyck)


def test_ten_eyck_dtype_declared() -> None:
    fourier_suite.test_dtype_declared(TEN_EYCK_MANIFEST)


def test_nufft_forward_matches_direct() -> None:
    fourier_suite.test_forward_matches_direct(_nufft)


def test_nufft_adjoint_is_vjp() -> None:
    fourier_suite.test_adjoint_is_vjp(_nufft)


def test_nufft_dtype_declared() -> None:
    fourier_suite.test_dtype_declared(NUFFT_MANIFEST)


def test_direct_dtype_declared() -> None:
    fourier_suite.test_dtype_declared(DIRECT_MANIFEST)
