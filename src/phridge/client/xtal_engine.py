"""Compatibility shim — prefer ``phridge.sfcalc.client``."""

from phridge.sfcalc.client import *  # noqa: F403
from phridge.sfcalc.client import (  # noqa: F401
    StructureFactorServer,
    _miller_template,
    _optional_array,
    _pack_block_diagonal,
    _pack_diagonal_like_gradients,
)
