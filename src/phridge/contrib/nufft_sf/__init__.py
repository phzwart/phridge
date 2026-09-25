"""NUFFT structure-factor engine (contrib ops + options).

Torch-free at import. The engine lives in ``phridge.sfcalc.engine.nufft_engine``.
Entry point: ``phridge.ops`` → ``phridge.contrib.nufft_sf:register``.
"""

from __future__ import annotations

from phridge.contrib.nufft_sf.op import CALC_OP, GRAD_OP, register_ops
from phridge.contrib.nufft_sf.options import NufftEngineOptions

__all__ = ["CALC_OP", "GRAD_OP", "NufftEngineOptions", "register"]


def register() -> None:
    """Entry-point hook; safe if called twice."""
    register_ops()
