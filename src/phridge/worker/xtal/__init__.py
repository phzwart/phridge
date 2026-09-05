"""Compatibility shim — prefer ``phridge.sfcalc.engine``."""

from phridge.sfcalc.engine import EngineParams, ScatteringModel, StructureFactorEngine

__all__ = ["EngineParams", "ScatteringModel", "StructureFactorEngine"]
