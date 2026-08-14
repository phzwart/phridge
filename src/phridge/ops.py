"""Shared op catalog. Installed with both client and worker extras."""

from __future__ import annotations

import json
import sys
from typing import Iterable

from phridge.models import SCHEMA_VERSION, OpSpec

_SPECS: dict[str, OpSpec] = {}


def register(spec: OpSpec) -> OpSpec:
    _SPECS[spec.name] = spec
    return spec


def get_op(name: str) -> OpSpec:
    try:
        return _SPECS[name]
    except KeyError as exc:
        raise KeyError(f"unknown op: {name}") from exc


def list_ops() -> list[OpSpec]:
    return [spec for _, spec in sorted(_SPECS.items())]


def list_ops_json() -> list[dict]:
    return [spec.model_dump(mode="json") for spec in list_ops()]


register(
    OpSpec(
        name="scale_array",
        schema_version=SCHEMA_VERSION,
        inputs={"array": "array", "scale": "json"},
        outputs={"array": "array"},
    )
)


def main(argv: Iterable[str] | None = None) -> None:
    _ = argv
    json.dump(list_ops_json(), sys.stdout, indent=2)
    sys.stdout.write("\n")
