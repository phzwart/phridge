"""Compatibility shim — prefer ``phridge.sfcalc.ops``."""

from phridge.sfcalc.ops import *  # noqa: F403
from phridge.sfcalc.ops import _DEVICE, set_device  # noqa: F401
