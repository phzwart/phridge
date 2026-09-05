"""Pydantic models matching schema/phridge.yaml and schema/cctbx.yaml.

Regenerate after schema edits with ``python scripts/generate_models.py``
(or edit this file to stay aligned). Generated output is committed so a
Phenix client does not need LinkML at runtime.

schema_version on envelopes is this module's SCHEMA_VERSION (the LinkML
``version`` field).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = 1


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ObservationType(str, Enum):
    fobs = "fobs"
    iobs = "iobs"
    amplitude = "amplitude"
    intensity = "intensity"
    complex = "complex"
    hl = "hl"
    other = "other"


class AnomalousLayout(str, Enum):
    asu = "asu"
    both_hemispheres = "both_hemispheres"


class MtzColumnType(str, Enum):
    F = "F"
    J = "J"
    D = "D"
    Q = "Q"
    G = "G"
    L = "L"
    K = "K"
    M = "M"
    P = "P"
    W = "W"
    A = "A"
    B = "B"
    C = "C"
    I = "I"
    R = "R"
    other = "other"


class CoordinateFrame(str, Enum):
    cartesian = "cartesian"
    fractional = "fractional"


class MapSpace(str, Enum):
    asu = "asu"
    p1 = "p1"


class ExperimentType(str, Enum):
    xray = "xray"
    cryo_em = "cryo_em"
    neutron = "neutron"
    other = "other"


class ObjectKind(str, Enum):
    array = "array"
    json = "json"
    blob = "blob"
    cctbx = "cctbx"


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    done = "done"
    error = "error"


class ByteOrder(str, Enum):
    little = "little"
    big = "big"


class Compression(str, Enum):
    none = "none"
    gzip = "gzip"


class WorkerRuntime(str, Enum):
    torch = "torch"
    cctbx = "cctbx"


class SymOp(_Strict):
    """One symmetry operator, x' = r @ x + t (fractional). r row-major 3x3."""

    r: list[float]
    t: list[float]

    @field_validator("r")
    @classmethod
    def _nine(cls, value: list[float]) -> list[float]:
        if len(value) != 9:
            raise ValueError("r must have 9 values")
        return [float(x) for x in value]

    @field_validator("t")
    @classmethod
    def _three(cls, value: list[float]) -> list[float]:
        if len(value) != 3:
            raise ValueError("t must have 3 values")
        return [float(x) for x in value]


class CrystalSymmetry(_Strict):
    """Canonical cctbx.crystal.symmetry. JSON only; no binary blob.

    ``symops`` (optional) is the full operator list so a worker without
    sgtbx can expand to P1.
    """

    unit_cell: list[float]
    space_group_hall: str
    space_group_number: Optional[int] = None
    symops: Optional[list[SymOp]] = None

    @field_validator("unit_cell")
    @classmethod
    def _six_cell(cls, value: list[float]) -> list[float]:
        if len(value) != 6:
            raise ValueError("unit_cell must be [a, b, c, alpha, beta, gamma]")
        return [float(x) for x in value]


class MillerArray(_Strict):
    """MillerArray metadata. Buffers are a packed npz at ObjectRef.key."""

    label: Optional[str] = None
    crystal: CrystalSymmetry
    anomalous: bool
    observation_type: ObservationType
    anomalous_layout: AnomalousLayout
    n_refl: int
    has_sigmas: bool
    index_dtype: str = "int32"
    data_dtype: str = "float64"


class HendricksonLattman(_Strict):
    """HL coefficients. npz: hkl, A, B, C, D."""

    label: Optional[str] = None
    crystal: CrystalSymmetry
    anomalous: bool
    anomalous_layout: AnomalousLayout
    n_refl: int
    index_dtype: str = "int32"


class ReflectionColumn(_Strict):
    label: str
    mtz_type: MtzColumnType
    mtz_type_raw: Optional[str] = None
    observation_type: ObservationType
    anomalous: bool
    has_sigmas: bool
    npz_name: str
    data_dtype: str = "float64"


class ReflectionFile(_Strict):
    """iotbx.mtz.object analog. npz: hkl plus named columns."""

    crystal: CrystalSymmetry
    wavelength: Optional[float] = None
    history: list[str] = Field(default_factory=list)
    n_refl: int
    index_dtype: str = "int32"
    anomalous_layout: AnomalousLayout
    columns: list[ReflectionColumn]


class CrystalGridding(_Strict):
    crystal: CrystalSymmetry
    n_real: list[int]
    resolution_factor: Optional[float] = None
    d_min: Optional[float] = None
    space: MapSpace = MapSpace.p1

    @field_validator("n_real")
    @classmethod
    def _xyz_grid(cls, value: list[int]) -> list[int]:
        if len(value) != 3:
            raise ValueError("n_real must have length 3")
        return [int(x) for x in value]


class RealMap(_Strict):
    """Real-map metadata. Buffer is npz array 'data' at ObjectRef.key."""

    label: Optional[str] = None
    crystal: CrystalSymmetry
    origin: list[int]
    n_real: list[int]
    space: MapSpace = MapSpace.p1
    dtype: str = "float64"

    @field_validator("origin", "n_real")
    @classmethod
    def _xyz(cls, value: list[int]) -> list[int]:
        if len(value) != 3:
            raise ValueError("origin and n_real must have length 3")
        return [int(x) for x in value]


class ComplexMap(_Strict):
    """Complex grid metadata. npz array 'data' complex128."""

    label: Optional[str] = None
    crystal: CrystalSymmetry
    origin: list[int]
    n_real: list[int]
    space: MapSpace = MapSpace.p1
    dtype: str = "complex128"

    @field_validator("origin", "n_real")
    @classmethod
    def _xyz(cls, value: list[int]) -> list[int]:
        if len(value) != 3:
            raise ValueError("origin and n_real must have length 3")
        return [int(x) for x in value]


class MapCoefficients(_Strict):
    label: Optional[str] = None
    miller: MillerArray


class EmMap(_Strict):
    label: Optional[str] = None
    crystal: CrystalSymmetry
    origin: list[int]
    n_real: list[int]
    space: MapSpace = MapSpace.p1
    dtype: str = "float64"
    experiment_type: ExperimentType = ExperimentType.cryo_em
    wrapping: bool = False
    is_mask: bool = False
    pixel_sizes: Optional[list[float]] = None
    origin_cart: Optional[list[float]] = None
    resolution: Optional[float] = None

    @field_validator("origin", "n_real")
    @classmethod
    def _xyz(cls, value: list[int]) -> list[int]:
        if len(value) != 3:
            raise ValueError("origin and n_real must have length 3")
        return [int(x) for x in value]


class CartesianSites(_Strict):
    crystal: Optional[CrystalSymmetry] = None
    n_sites: int
    dtype: str = "float64"


class FractionalSites(_Strict):
    crystal: CrystalSymmetry
    n_sites: int
    dtype: str = "float64"


class Atom(_Strict):
    i: int
    name: str
    element: str
    model_id: Optional[str] = None
    chain_id: str
    resseq: int
    icode: Optional[str] = None
    resname: str
    altloc: Optional[str] = None
    hetero: bool = False


class Hierarchy(_Strict):
    crystal: Optional[CrystalSymmetry] = None
    n_atoms: int
    frame: CoordinateFrame = CoordinateFrame.cartesian
    has_uij: bool = False
    atoms: list[Atom]


class Scatterer(_Strict):
    i: int
    scattering_type: str
    fp: Optional[float] = None
    fdp: Optional[float] = None
    anisotropic: bool = False
    use_u_iso: bool = True


class XrayStructure(_Strict):
    crystal: CrystalSymmetry
    n_scatterers: int
    scatterers: list[Scatterer]


class GeometryRestraints(_Strict):
    crystal: Optional[CrystalSymmetry] = None
    n_sites: int
    n_bonds: int = 0
    n_bond_asu: int = 0
    n_angles: int = 0
    n_dihedrals: int = 0
    n_chiralities: int = 0
    n_planarities: int = 0
    n_parallelities: int = 0
    n_nonbonded: int = 0
    n_reference_coords: int = 0
    n_bond_similarities: int = 0
    bond_asu_rt_mx: list[str] = Field(default_factory=list)


class ScatteringTable(_Strict):
    """Gaussian form factors per scattering type. npz: gauss_a, gauss_b, gauss_c."""

    table: str
    labels: list[str]
    n_terms: int


class SfEngineParams(_Strict):
    """FFT structure-factor engine controls. JSON only."""

    d_min: float
    grid_resolution_factor: float = 1.0 / 3.0
    quality_factor: float = 100.0
    wing_cutoff: float = 1e-4
    u_extra: Optional[float] = None
    n_real: Optional[list[int]] = None
    dtype: str = "float64"


class SfGradients(_Strict):
    """d target / d scatterer parameters. npz: d_site_frac, d_occupancy, d_u_iso, d_u_star, d_fp, d_fdp."""

    n_scatterers: int
    target: Optional[float] = None


class SfCurvatures(_Strict):
    """Per-atom Gauss-Newton blocks. npz: site_frac, occupancy, u_iso, u_star, fp, fdp."""

    n_scatterers: int


class TargetResult(_Strict):
    """Target evaluation. npz: per_reflection, d_target_d_f_calc, optional curv_radial, curv_tangential."""

    name: str
    value: float
    value_test: Optional[float] = None
    n_refl: int
    scale_factor: Optional[float] = None
    has_curvature: bool


class ModelGeometry(_Strict):
    n_sites: int
    has_hierarchy: bool = False
    has_xray: bool = False
    has_restraints: bool = False
    i_seq_identity: bool = True


CCTBX_TYPES = {
    "CrystalSymmetry": CrystalSymmetry,
    "CrystalGridding": CrystalGridding,
    "MillerArray": MillerArray,
    "HendricksonLattman": HendricksonLattman,
    "ReflectionFile": ReflectionFile,
    "RealMap": RealMap,
    "ComplexMap": ComplexMap,
    "MapCoefficients": MapCoefficients,
    "EmMap": EmMap,
    "CartesianSites": CartesianSites,
    "FractionalSites": FractionalSites,
    "Hierarchy": Hierarchy,
    "XrayStructure": XrayStructure,
    "GeometryRestraints": GeometryRestraints,
    "ModelGeometry": ModelGeometry,
    "ScatteringTable": ScatteringTable,
    "SfEngineParams": SfEngineParams,
    "SfGradients": SfGradients,
    "SfCurvatures": SfCurvatures,
    "TargetResult": TargetResult,
}


class ArrayMeta(_Strict):
    dtype: str
    shape: list[int]
    order: str = "C"


class BlobMeta(_Strict):
    content_type: str


class ObjectRef(_Strict):
    kind: ObjectKind
    key: Optional[str] = None
    dtype: Optional[str] = None
    shape: Optional[list[int]] = None
    byte_order: ByteOrder = ByteOrder.little
    compression: Compression = Compression.none
    cctbx_type: Optional[str] = None
    meta: Optional[dict[str, Any]] = None

    @field_validator("cctbx_type")
    @classmethod
    def _known_cctbx(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and value not in CCTBX_TYPES:
            raise ValueError(f"unknown cctbx_type: {value}")
        return value


class JobError(_Strict):
    type: str
    message: str
    traceback: Optional[str] = None


class JobEnvelope(_Strict):
    job_id: str
    op: str
    schema_version: int = SCHEMA_VERSION
    inputs: dict[str, ObjectRef] = Field(default_factory=dict)
    outputs: dict[str, ObjectRef] = Field(default_factory=dict)
    status: JobStatus = JobStatus.queued
    error: Optional[JobError] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @field_validator("schema_version")
    @classmethod
    def _version_one(cls, value: int) -> int:
        if value != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
        return value


class SlotBinding(_Strict):
    name: str
    type: str


class OpSpec(_Strict):
    name: str
    schema_version: int = SCHEMA_VERSION
    runtime: WorkerRuntime = WorkerRuntime.torch
    inputs: dict[str, str] = Field(default_factory=dict)
    outputs: dict[str, str] = Field(default_factory=dict)
