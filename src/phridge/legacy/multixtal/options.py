"""Pydantic options and dataset manifest for multixtal phase 1. Torch-free."""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


Placement = Literal["fractional", "cartesian"]
RankChoice = Union[int, Literal["auto"]]


DoseModelName = Literal["linear", "exp", "learned"]


class DatasetSpec(BaseModel):
    """One diffraction dataset in a YAML/JSON manifest."""

    model_config = {"extra": "forbid"}

    file: str
    labels: Optional[str] = None
    wavelength: Optional[float] = Field(default=None, gt=0.0)
    notes: Optional[str] = None
    name: Optional[str] = None
    unmerged: Optional[str] = None
    dose_table: Optional[str] = None
    dose_rate: Optional[float] = Field(default=None, gt=0.0)


class DatasetManifest(BaseModel):
    """``--manifest datasets.yaml`` listing files, labels, wavelength, notes."""

    model_config = {"extra": "forbid"}

    datasets: list[DatasetSpec] = Field(min_length=1)

    @classmethod
    def from_mapping(cls, data: Any) -> DatasetManifest:
        if isinstance(data, DatasetManifest):
            return data
        if isinstance(data, list):
            return cls(datasets=[DatasetSpec.model_validate(item) for item in data])
        if isinstance(data, dict):
            if "datasets" in data:
                return cls.model_validate(data)
            return cls(datasets=[DatasetSpec.model_validate(data)])
        raise TypeError("manifest must be a mapping or a list of dataset specs")


class MultixtalOptions(BaseModel):
    """JSON / CLI options for ``multixtal_fit``. Every run parameter lives here."""

    model_config = {"extra": "forbid"}

    d_min: float = Field(default=2.0, gt=0.0)
    strong_isig: float = Field(default=5.0, gt=0.0)
    strong_min_datasets: float = Field(default=0.5, gt=0.0, le=1.0)
    e_calc_floor: Optional[float] = Field(default=None, ge=0.0)
    rank: Union[int, str] = Field(default="auto")
    rank_perms: int = Field(default=50, ge=1)
    anomalous: bool = True
    placement: Placement = "fractional"
    n_spline_knots: int = Field(default=8, ge=4, le=16)
    n_outer: int = Field(default=2, ge=1)
    ridge: float = Field(default=1e-6, ge=0.0)
    max_lbfgs_iter: int = Field(default=40, ge=1)
    anom_sites: Optional[list[int]] = None
    labels: str = "I(+),SIGI(+),I(-),SIGI(-)"
    space_group: Optional[str] = None
    cell: Optional[list[float]] = None
    flag_z: float = Field(default=3.0, gt=0.0)
    dose_model: DoseModelName = "linear"
    dose_bins: int = Field(default=10, ge=2, le=64)
    dose_rate: float = Field(default=0.05, gt=0.0)
    absorption_order: int = Field(default=4, ge=0, le=8)
    damage_rank: Union[int, str] = Field(default="auto")
    smoothness_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    n_exp_modes: int = Field(default=1, ge=1, le=8)
    frame_reject_k: float = Field(default=3.0, gt=0.0)
    confound_corr: float = Field(default=0.85, ge=0.0, le=1.0)

    @field_validator("rank", "damage_rank")
    @classmethod
    def _rank_ok(cls, value: Union[int, str]) -> Union[int, str]:
        if isinstance(value, str):
            text = value.strip().lower()
            if text != "auto":
                raise ValueError("rank must be 'auto' or a non-negative int")
            return "auto"
        if int(value) < 0:
            raise ValueError("rank must be >= 0")
        return int(value)

    @field_validator("cell")
    @classmethod
    def _cell_ok(cls, value: Optional[list[float]]) -> Optional[list[float]]:
        if value is None:
            return None
        if len(value) != 6:
            raise ValueError("cell must be six unit-cell parameters")
        return [float(x) for x in value]

    @field_validator("anomalous", mode="before")
    @classmethod
    def _anom_flag(cls, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        raise ValueError("anomalous must be on/off")
