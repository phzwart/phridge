"""Worker op ``spatial_sigma_a_v2_step``: field or Fisher half-step.

Torch-free. Registered on the torch worker stream so Redis jobs land with
the other intensity ops; the body is numpy.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from phridge.ops import register_op
from phridge.packing import PackedMiller
from phridge.packing_xtal import PackedXray
from phridge.sfcalc.ops import scattering_model

OP_NAME = "spatial_sigma_a_v2_step"

_INPUTS = {
    "xray": "XrayStructure",
    "table": "ScatteringTable",
    "f_obs": "MillerArray",
    "spatial_sigma_a_v2": "SpatialSigmaAV2",
    "epsilon": "array",
    "centric": "array",
}
_OUTPUTS = {
    "spatial_sigma_a_v2": "SpatialSigmaAV2",
    "spatial_sigma_a_v2_result": "SpatialSigmaAV2Result",
}


def spatial_sigma_a_v2_step(
    xray: PackedXray,
    table: Any,
    f_obs: PackedMiller,
    spatial_sigma_a_v2: Any,
    epsilon: Optional[Any] = None,
    centric: Optional[Any] = None,
) -> dict[str, Any]:
    """Run one alternation half-step. No-op payload is rejected by the caller."""
    from phridge.contrib.spatial_sigmaa_v2.alternate import run_macrocycle_step
    from phridge.contrib.spatial_sigmaa_v2.options import SpatialSigmaAV2Options
    from phridge.contrib.spatial_sigmaa_v2.packing import PackedSpatialSigmaAV2

    if spatial_sigma_a_v2 is None:
        raise ValueError("spatial_sigma_a_v2_step requires a SpatialSigmaAV2 block")
    block = spatial_sigma_a_v2
    if not isinstance(block, PackedSpatialSigmaAV2):
        raise TypeError("spatial_sigma_a_v2 must be PackedSpatialSigmaAV2")
    model = scattering_model(xray, table)
    hkl = np.asarray(f_obs.hkl, dtype=np.int64).reshape(-1, 3)
    intensity = np.abs(np.asarray(f_obs.data, dtype=np.float64).reshape(-1))
    eps = None if epsilon is None else np.asarray(epsilon, dtype=np.float64).reshape(-1)
    cen = None if centric is None else np.asarray(centric, dtype=bool).reshape(-1)
    opts = SpatialSigmaAV2Options(
        enabled=True,
        field_cutoff=block.meta.field_cutoff,
        entropy_weight=block.meta.entropy_weight,
        spectral_taper_scale=block.meta.spectral_taper_scale,
        fisher=bool(block.meta.fisher),
    )
    packed, result, _frozen = run_macrocycle_step(
        model, hkl, intensity, block, opts, epsilon=eps, centric=cen
    )
    return {"spatial_sigma_a_v2": packed, "spatial_sigma_a_v2_result": result}


def register_ops() -> None:
    register_op(OP_NAME, spatial_sigma_a_v2_step, inputs=dict(_INPUTS), outputs=dict(_OUTPUTS))
