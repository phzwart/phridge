"""NUFFT structure-factor engine.

Torch-free at import of the ops. Entry point: ``phridge.ops`` →
``phridge.transforms.fourier.nufft:register``.
"""

from __future__ import annotations

from phridge.transforms.fourier.nufft.op import BIND_OP, CALC_OP, GRAD_OP, register_ops
from phridge.transforms.fourier.nufft.operator import register as register_manifest_hook
from phridge.transforms.fourier.nufft.options import NufftEngineOptions

__all__ = ["BIND_OP", "CALC_OP", "GRAD_OP", "NufftEngineOptions", "register"]


def register() -> None:
    """Entry-point hook; safe if called twice."""
    register_ops()
    register_manifest_hook()


register()
