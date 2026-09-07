"""JSON Schema validation of LinkML metadata (generated artifacts)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

jsonschema = pytest.importorskip("jsonschema")

from phridge.models import (
    CrystalSymmetry,
    JobEnvelope,
    JobStatus,
    ModelGeometry,
    ObjectKind,
    ObjectRef,
)

GENERATED = Path(__file__).resolve().parents[1] / "schema" / "generated"


def _schema(name: str) -> dict:
    path = GENERATED / f"{name}.schema.json"
    if not path.exists():
        pytest.skip(f"run scripts/generate_models.py --jsonschema ({path.name} missing)")
    return json.loads(path.read_text())


def test_jsonschema_accepts_crystal_symmetry():
    cs = CrystalSymmetry(
        unit_cell=[78.1, 78.1, 37.0, 90, 90, 90],
        space_group_hall=" P 4nw 2abw",
        space_group_number=96,
    )
    jsonschema.validate(cs.model_dump(mode="json"), _schema("CrystalSymmetry"))


def test_jsonschema_rejects_short_unit_cell():
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            {"unit_cell": [1.0, 2.0, 3.0], "space_group_hall": "P 1"},
            _schema("CrystalSymmetry"),
        )


def test_jsonschema_cctbx_objectref_requires_type():
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"kind": "cctbx"}, _schema("ObjectRef"))


def test_jsonschema_accepts_valid_object_ref():
    ref = ObjectRef(
        kind=ObjectKind.array,
        key="phridge:obj:123:test",
        dtype="float64",
        shape=[10, 3],
    )
    jsonschema.validate(ref.model_dump(mode="json"), _schema("ObjectRef"))


def test_jsonschema_accepts_model_geometry():
    mg = ModelGeometry(
        n_sites=100,
        has_hierarchy=True,
        has_xray=True,
        has_restraints=True,
        i_seq_identity=True,
    )
    jsonschema.validate(mg.model_dump(mode="json"), _schema("ModelGeometry"))


def test_jsonschema_rejects_model_geometry_missing_required():
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            {"n_sites": 100},  # Missing required boolean flags
            _schema("ModelGeometry"),
        )


def test_jsonschema_accepts_job_envelope():
    env = {
        "job_id": "test-job-123",
        "op": "scale_array",
        "schema_version": 1,
        "status": "queued",
        "inputs": [],
        "outputs": [],
    }
    jsonschema.validate(env, _schema("JobEnvelope"))
