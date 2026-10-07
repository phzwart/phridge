"""PluginManifest validation, registry, and generated JSON Schema."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from phridge.contracts.manifest import PluginManifest, Slot, Space
from phridge.ops import get_manifest, manifests_for_slot, register_manifest

GENERATED = (
    Path(__file__).resolve().parents[2] / "src" / "phridge" / "contracts" / "schema" / "generated"
)


def test_transform_manifest_rules() -> None:
    PluginManifest(
        name="fourier_rq",
        slot=Slot.fourier,
        space=Space.transform,
        from_space=Space.R,
        to_space=Space.Q,
    )
    with pytest.raises(ValidationError):
        PluginManifest(
            name="fourier_open",
            slot=Slot.fourier,
            space=Space.transform,
            to_space=Space.Q,
        )
    with pytest.raises(ValidationError):
        PluginManifest(
            name="score_as_transform",
            slot=Slot.scoring_rule,
            space=Space.transform,
            from_space=Space.R,
            to_space=Space.Q,
        )


def test_register_manifest_round_trip() -> None:
    manifest = PluginManifest(name="b2_round_trip", slot=Slot.io_adapter, space=Space.R)
    register_manifest(manifest)
    try:
        assert get_manifest("b2_round_trip") == manifest
        assert manifest in manifests_for_slot(Slot.io_adapter)
        assert manifest not in manifests_for_slot(Slot.fourier)
    finally:
        from phridge.ops import _MANIFESTS

        _MANIFESTS.pop("b2_round_trip", None)


def test_jsonschema_accepts_manifest() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((GENERATED / "PluginManifest.schema.json").read_text())
    jsonschema.validate(
        {
            "name": "fourier_rq",
            "slot": "fourier",
            "space": "transform",
            "from_space": "R",
            "to_space": "Q",
        },
        schema,
    )
