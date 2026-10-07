"""Plugin manifests. Own strict base so this module does not import models."""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

_TRANSFORM_SLOTS = frozenset({"fourier", "observation", "builder"})


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Slot(str, Enum):
    r_component = "r_component"
    restraint_provider = "restraint_provider"
    q_component = "q_component"
    fourier = "fourier"
    observation = "observation"
    builder = "builder"
    scoring_rule = "scoring_rule"
    optimizer = "optimizer"
    policy = "policy"
    statistic = "statistic"
    io_adapter = "io_adapter"


class Space(str, Enum):
    R = "R"
    Q = "Q"
    D = "D"
    transform = "transform"


class Tier(str, Enum):
    shim = "shim"
    hybrid = "hybrid"
    native = "native"


class BlockDecl(_Strict):
    id: str
    shape: list[int | str] = Field(default_factory=list)
    bounds: Optional[tuple[float, float]] = None
    transform: Literal["identity", "log", "logit"] = "identity"
    frozen: bool = False


class PluginManifest(_Strict):
    name: str
    slot: Slot
    space: Space
    from_space: Optional[Space] = None
    to_space: Optional[Space] = None
    reads: list[str] = Field(default_factory=list)
    produces: list[str] = Field(default_factory=list)
    blocks: list[BlockDecl] = Field(default_factory=list)
    needs: list[str] = Field(default_factory=list)
    provides: list[str] = Field(default_factory=list)
    tier: Tier = Tier.native
    dtype: Literal["float32", "float64"] = "float64"
    device: Literal["cpu", "cuda", "mps", "any"] = "any"
    version: str = "0"
    fit_on: Optional[str] = None

    @model_validator(mode="after")
    def _check_spaces(self) -> PluginManifest:
        is_transform = self.space == Space.transform
        if is_transform and (self.from_space is None or self.to_space is None):
            raise ValueError("from_space and to_space are required when space is transform")
        if not is_transform and (self.from_space is not None or self.to_space is not None):
            raise ValueError("from_space and to_space are only allowed when space is transform")
        if is_transform and self.slot.value not in _TRANSFORM_SLOTS:
            raise ValueError("space transform is only allowed for fourier, observation, and builder")
        return self
