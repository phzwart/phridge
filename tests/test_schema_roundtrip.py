from datetime import timezone

import pytest
from pydantic import ValidationError

from phridge.models import (
    SCHEMA_VERSION,
    CCTBX_TYPES,
    AnomalousLayout,
    Atom,
    CartesianSites,
    ComplexMap,
    CoordinateFrame,
    CrystalGridding,
    CrystalSymmetry,
    FractionalSites,
    HendricksonLattman,
    Hierarchy,
    JobEnvelope,
    JobStatus,
    MapCoefficients,
    MapSpace,
    MillerArray,
    MtzColumnType,
    ObjectKind,
    ObjectRef,
    ObservationType,
    RealMap,
    ReflectionColumn,
    ReflectionFile,
    Scatterer,
    XrayStructure,
)


def test_schema_version_is_one():
    assert SCHEMA_VERSION == 1


def test_crystal_symmetry_valid():
    cs = CrystalSymmetry(
        unit_cell=[78.1, 78.1, 37.0, 90, 90, 90],
        space_group_hall=" P 4nw 2abw",
        space_group_number=96,
    )
    assert cs.unit_cell[2] == 37.0
    roundtrip = CrystalSymmetry.model_validate(cs.model_dump())
    assert roundtrip.space_group_number == 96


def test_crystal_symmetry_rejects_bad_cell():
    with pytest.raises(ValidationError):
        CrystalSymmetry(unit_cell=[1, 2, 3], space_group_hall="P 1")


def test_miller_array_valid():
    cs = CrystalSymmetry(unit_cell=[10, 10, 10, 90, 90, 90], space_group_hall="P 1")
    meta = MillerArray(
        crystal=cs,
        anomalous=False,
        observation_type=ObservationType.fobs,
        anomalous_layout=AnomalousLayout.asu,
        n_refl=4,
        has_sigmas=True,
    )
    assert meta.index_dtype == "int32"
    assert meta.data_dtype == "float64"


def test_real_map_valid():
    cs = CrystalSymmetry(unit_cell=[10, 10, 10, 90, 90, 90], space_group_hall="P 1")
    meta = RealMap(crystal=cs, origin=[0, 0, 0], n_real=[8, 8, 8], space=MapSpace.p1)
    assert meta.dtype == "float64"
    with pytest.raises(ValidationError):
        RealMap(crystal=cs, origin=[0, 0], n_real=[8, 8, 8])


def test_reflection_file_and_hl():
    cs = CrystalSymmetry(unit_cell=[10, 10, 10, 90, 90, 90], space_group_hall="P 1")
    col = ReflectionColumn(
        label="FOBS",
        mtz_type=MtzColumnType.F,
        observation_type=ObservationType.fobs,
        anomalous=False,
        has_sigmas=True,
        npz_name="FOBS",
        data_dtype="float64",
    )
    rf = ReflectionFile(
        crystal=cs,
        n_refl=10,
        anomalous_layout=AnomalousLayout.asu,
        columns=[col],
        wavelength=0.979,
    )
    assert rf.columns[0].label == "FOBS"
    hl = HendricksonLattman(
        crystal=cs,
        anomalous=False,
        anomalous_layout=AnomalousLayout.asu,
        n_refl=10,
    )
    assert hl.index_dtype == "int32"


def test_maps_and_coefficients():
    cs = CrystalSymmetry(unit_cell=[10, 10, 10, 90, 90, 90], space_group_hall="P 1")
    grid = CrystalGridding(crystal=cs, n_real=[16, 16, 16], space=MapSpace.p1, d_min=2.0)
    cmap = ComplexMap(crystal=cs, origin=[0, 0, 0], n_real=[16, 16, 16], space=MapSpace.p1)
    miller = MillerArray(
        label="Fcalc",
        crystal=cs,
        anomalous=False,
        observation_type=ObservationType.complex,
        anomalous_layout=AnomalousLayout.asu,
        n_refl=8,
        has_sigmas=False,
        data_dtype="complex128",
    )
    coef = MapCoefficients(label="Fcalc", miller=miller)
    assert grid.n_real[0] == 16
    assert cmap.dtype == "complex128"
    assert coef.miller.observation_type is ObservationType.complex


def test_coordinates_hierarchy_and_xray():
    cs = CrystalSymmetry(unit_cell=[10, 10, 10, 90, 90, 90], space_group_hall="P 1")
    sites = CartesianSites(crystal=cs, n_sites=1)
    frac = FractionalSites(crystal=cs, n_sites=1)
    atom = Atom(i=0, name="CA", element="C", chain_id="A", resseq=10, resname="ALA", hetero=False)
    hier = Hierarchy(crystal=cs, n_atoms=1, frame=CoordinateFrame.cartesian, atoms=[atom])
    sc = Scatterer(i=0, scattering_type="C", anisotropic=False, use_u_iso=True)
    xrs = XrayStructure(crystal=cs, n_scatterers=1, scatterers=[sc])
    assert sites.n_sites == frac.n_sites == 1
    assert hier.atoms[0].name == "CA"
    assert xrs.scatterers[0].scattering_type == "C"


def test_job_envelope_roundtrip():
    envelope = JobEnvelope(
        job_id="abc",
        op="scale_array",
        schema_version=SCHEMA_VERSION,
        inputs={
            "array": ObjectRef(
                kind=ObjectKind.array,
                key="phridge:obj:abc:array",
                dtype="float64",
                shape=[3],
            )
        },
        status=JobStatus.queued,
    )
    loaded = JobEnvelope.model_validate_json(envelope.model_dump_json())
    assert loaded.inputs["array"].kind == ObjectKind.array
    assert loaded.created_at is None or loaded.created_at.tzinfo in (None, timezone.utc)


def test_object_ref_unknown_cctbx_type():
    with pytest.raises(ValidationError):
        ObjectRef(kind=ObjectKind.cctbx, cctbx_type="NotAType", meta={})


def test_cctbx_types_registry():
    for name in (
        "MillerArray",
        "ReflectionFile",
        "HendricksonLattman",
        "RealMap",
        "ComplexMap",
        "MapCoefficients",
        "CartesianSites",
        "FractionalSites",
        "Hierarchy",
        "XrayStructure",
        "GeometryRestraints",
        "ModelGeometry",
        "EmMap",
    ):
        assert name in CCTBX_TYPES
