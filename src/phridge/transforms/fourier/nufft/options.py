"""Pydantic options for the NUFFT structure-factor engine. Torch-free."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class NufftEngineOptions(BaseModel):
    """Wire options for ``nufft_sf_calc`` / ``nufft_sf_gradients``."""

    model_config = {"extra": "forbid"}

    engine: Literal["nufft"] = "nufft"
    d_min: float = Field(gt=0.0)
    tau: float = Field(default=1e-4, gt=0.0)
    n_max: int = Field(default=2, ge=0)
    eps: float = Field(default=1e-6, gt=0.0, le=1e-2)
    dtype: Literal["float64", "float32"] = "float64"
    t_chunk: int = Field(default=16, ge=1)
    symmetry: Literal["expand"] = "expand"
