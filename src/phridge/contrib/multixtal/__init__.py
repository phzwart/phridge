"""Multi-dataset covariance model (phase 1: strong data, plug-in estimates).

cctbx clients stay torch-free: import options, client, cli, and ``register()``.
The numerical core is loaded only by the worker op.

``bulk_solvent_op`` / ``free_beta`` / ``maps.intensity_map_coefficients`` are
**not** called: they do not implement the phase-1 formulae (see the module
docstrings of ``nuisance``, ``spline``, and ``maps_out``).
"""

from __future__ import annotations

from phridge.contrib.multixtal.options import DatasetManifest, DatasetSpec, MultixtalOptions
from phridge.contrib.multixtal.op import OP_NAME, register_ops

__all__ = [
    "DatasetManifest",
    "DatasetSpec",
    "MultixtalOptions",
    "OP_NAME",
    "register",
    "register_ops",
]


def register() -> None:
    """Entry-point hook; idempotent."""
    register_ops()
