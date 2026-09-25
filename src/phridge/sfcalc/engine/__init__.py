"""Torch structure-factor engine: differentiable FFT-based F_calc and gradients.

Never imports cctbx. Everything cctbx would normally supply (symmetry
operators, form-factor Gaussians) arrives as arrays from the client.
"""

from phridge.sfcalc.engine.engine import EngineParams, ScatteringModel, StructureFactorEngine
from phridge.sfcalc.engine.nufft_engine import GroupPlan, NufftEngineParams, NufftStructureFactorEngine

__all__ = [
    "EngineParams",
    "GroupPlan",
    "NufftEngineParams",
    "NufftStructureFactorEngine",
    "ScatteringModel",
    "StructureFactorEngine",
]
