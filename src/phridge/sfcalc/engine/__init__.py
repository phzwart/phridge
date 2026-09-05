"""Torch structure-factor engine: differentiable FFT-based F_calc and gradients.

Never imports cctbx. Everything cctbx would normally supply (symmetry
operators, form-factor Gaussians) arrives as arrays from the client.
"""

from phridge.sfcalc.engine.engine import EngineParams, ScatteringModel, StructureFactorEngine

__all__ = ["EngineParams", "ScatteringModel", "StructureFactorEngine"]
