"""Torch structure-factor engine: differentiable FFT-based F_calc and gradients.

Never imports cctbx. Everything cctbx would normally supply (symmetry
operators, form-factor Gaussians) arrives as arrays from the client.
"""

from __future__ import annotations

from phridge.transforms.fourier.ten_eyck.engine import EngineParams, ScatteringModel, StructureFactorEngine
from phridge.transforms.fourier.ten_eyck.operator import register as register_fourier

__all__ = [
    "EngineParams",
    "ScatteringModel",
    "StructureFactorEngine",
    "register",
]


def register() -> None:
    register_fourier()
