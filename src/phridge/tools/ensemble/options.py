"""Pydantic options for the ensemble perturbation-mode fit. Torch-free."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class ProjectionOptions(BaseModel):
    """Real-space constraints. Off by default; they change the objective."""

    model_config = {"extra": "forbid"}

    enabled: bool = False
    support: bool = False
    support_dilate: int = Field(default=0, ge=0, le=8)
    smooth_sigma: float = Field(default=0.0, ge=0.0)
    zero_f000: bool = True
    positivity: bool = False
    grid: list[int] = Field(default_factory=lambda: [16, 16, 16])

    @field_validator("grid")
    @classmethod
    def _grid_ok(cls, value: list[int]) -> list[int]:
        if len(value) != 3 or any(int(n) < 4 for n in value):
            raise ValueError("grid must be three integers >= 4")
        return [int(n) for n in value]


class HoldoutOptions(BaseModel):
    """Crystal folds, reflection roles, half-split, and the K scan."""

    model_config = {"extra": "forbid"}

    enabled: bool = False
    n_folds: int = Field(default=5, ge=2, le=32)
    k_values: list[int] = Field(default_factory=list)
    half_split: bool = False
    seed: int = 0
    audit_fraction: float = Field(default=0.0, ge=0.0, lt=1.0)
    calibration_fraction: float = Field(default=0.0, ge=0.0, lt=1.0)

    @field_validator("k_values")
    @classmethod
    def _k_ok(cls, value: list[int]) -> list[int]:
        out = [int(k) for k in value]
        if any(k < 0 for k in out):
            raise ValueError("k_values must be non-negative")
        return out


class PhaseInjectionOptions(BaseModel):
    """Optional Hendrickson–Lattman phase information.

    ``hl_weight`` scales the HL vector. Likelihoods at different weights are
    not comparable.
    """

    model_config = {"extra": "forbid"}

    enabled: bool = False
    hl_weight: float = Field(default=0.0, ge=0.0, le=1.0)


class BootstrapOptions(BaseModel):
    """Stratified bootstrap over datasets. Modes are aligned before the spread is reported."""

    model_config = {"extra": "forbid"}

    n_bootstrap: int = Field(default=0, ge=0, le=200)
    seed: int = 0
    strata: Optional[list[str]] = None


class DatasetRoles(BaseModel):
    """How reflection roles are drawn when the caller does not pass an array.

    0 work, 1 calibration, 2 audit, -1 absent. Free flags, when supplied on
    the job, are audit.
    """

    model_config = {"extra": "forbid"}

    audit_fraction: float = Field(default=0.0, ge=0.0, lt=1.0)
    calibration_fraction: float = Field(default=0.0, ge=0.0, lt=1.0)
    seed: int = 0


class EnsembleOptions(BaseModel):
    """JSON options for ``ensemble_fit``. Every run parameter lives here."""

    model_config = {"extra": "forbid"}

    k: int = Field(default=1, ge=0, le=32)
    n_sweeps: int = Field(default=20, ge=1, le=200)
    likelihood: Literal["known_f", "intensity"] = "intensity"
    ridge: float = Field(default=1e-8, ge=0.0)
    dof_correct: bool = True
    d_min: float = Field(default=2.0, gt=0.0)
    labels: str = "I(+),SIGI(+),I(-),SIGI(-)"
    placement: Literal["fractional", "cartesian"] = "fractional"
    space_group: Optional[str] = None
    cell: Optional[list[float]] = None
    bin_size: int = Field(default=50, ge=1)
    nu: Optional[float] = Field(default=None, gt=1.0)
    sigma_a0: float = Field(default=0.9, gt=0.0, lt=1.0)
    fit_nuisance: bool = False
    direct: bool = False
    max_lbfgs_iter: int = Field(default=8, ge=1, le=40)
    projection: ProjectionOptions = Field(default_factory=ProjectionOptions)
    holdout: HoldoutOptions = Field(default_factory=HoldoutOptions)
    phase_injection: PhaseInjectionOptions = Field(default_factory=PhaseInjectionOptions)
    bootstrap: BootstrapOptions = Field(default_factory=BootstrapOptions)
    roles: DatasetRoles = Field(default_factory=DatasetRoles)

    @field_validator("cell")
    @classmethod
    def _cell_ok(cls, value: Optional[list[float]]) -> Optional[list[float]]:
        if value is None:
            return None
        if len(value) != 6:
            raise ValueError("cell must be six unit-cell parameters")
        return [float(x) for x in value]
