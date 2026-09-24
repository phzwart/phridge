"""Env / CLI options for spatial σ_A v2. Torch-free."""

from __future__ import annotations

import os
from typing import Any, Optional

from pydantic import BaseModel, Field

from phridge.contrib.spatial_sigmaa_v2.packing import PackedSpatialSigmaAV2

DEFAULT_FIELD_CUTOFF = 15.0
DEFAULT_ENTROPY_WEIGHT = 0.0
DEFAULT_SPECTRAL_TAPER = 1.0

_ENV_ENABLED = "PHRIDGE_SPATIAL_SIGMA_A_V2"
_ENV_FISHER = "PHRIDGE_SPATIAL_SIGMA_A_V2_FISHER"
_ENV_CUTOFF = "PHRIDGE_SPATIAL_SIGMA_A_V2_CUTOFF"
_ENV_ENTROPY = "PHRIDGE_SPATIAL_SIGMA_A_V2_ENTROPY"
_ENV_TAPER = "PHRIDGE_SPATIAL_SIGMA_A_V2_TAPER"


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return float(default)
    try:
        value = float(raw)
    except ValueError:
        return float(default)
    if value != value or value < 0.0:
        return float(default)
    return value


class SpatialSigmaAV2Options(BaseModel):
    """JSON / env options. ``enabled=False`` means do not construct a wire block."""

    model_config = {"extra": "forbid"}

    enabled: bool = False
    field_cutoff: float = Field(default=DEFAULT_FIELD_CUTOFF, gt=0.0)
    entropy_weight: float = Field(default=DEFAULT_ENTROPY_WEIGHT, ge=0.0)
    spectral_taper_scale: float = Field(default=DEFAULT_SPECTRAL_TAPER, gt=0.0)
    fisher: bool = False


def options_from_env() -> SpatialSigmaAV2Options:
    """Read ``PHRIDGE_SPATIAL_SIGMA_A_V2*``. Default is off."""
    return SpatialSigmaAV2Options(
        enabled=_flag(_ENV_ENABLED, "0"),
        field_cutoff=_float(_ENV_CUTOFF, DEFAULT_FIELD_CUTOFF),
        entropy_weight=_float(_ENV_ENTROPY, DEFAULT_ENTROPY_WEIGHT),
        spectral_taper_scale=_float(_ENV_TAPER, DEFAULT_SPECTRAL_TAPER),
        fisher=_flag(_ENV_FISHER, "0"),
    )


def block_from_options(opts: SpatialSigmaAV2Options) -> Optional[PackedSpatialSigmaAV2]:
    """Empty (unfitted) block when enabled; ``None`` when off so nothing is packed."""
    if not opts.enabled:
        return None
    return PackedSpatialSigmaAV2.empty(
        field_cutoff=opts.field_cutoff,
        entropy_weight=opts.entropy_weight,
        spectral_taper_scale=opts.spectral_taper_scale,
        fisher=opts.fisher,
    )


def block_from_env() -> Optional[PackedSpatialSigmaAV2]:
    return block_from_options(options_from_env())


def job_input_from_env() -> dict[str, Any]:
    """Kwargs to merge into ``Bridge.call``. Empty when the flag is off."""
    block = block_from_env()
    if block is None:
        return {}
    return {"spatial_sigma_a_v2": block}


def job_input_from_state(
    block: Optional[PackedSpatialSigmaAV2],
    result: Any = None,
    *,
    opts: Optional[SpatialSigmaAV2Options] = None,
) -> dict[str, Any]:
    """Fitted block + result when the flag is on. Empty when off (spec §11.9)."""
    if opts is None:
        opts = options_from_env()
    if not opts.enabled:
        return {}
    out: dict[str, Any] = {}
    if block is not None:
        out["spatial_sigma_a_v2"] = block
    elif (fallback := block_from_options(opts)) is not None:
        out["spatial_sigma_a_v2"] = fallback
    if result is not None:
        out["spatial_sigma_a_v2_result"] = result
    return out
