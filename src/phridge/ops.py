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

_XRAY_INPUTS = {"xray": "XrayStructure", "table": "ScatteringTable", "params": "SfEngineParams"}
_TARGET_INPUTS = {
    "f_obs": "MillerArray",
    "target": "json",
    "weights": "array",
    "r_free": "array",
    "alpha": "array",
    "beta": "array",
    "epsilon": "array",
    "centric": "array",
    "compute_curvature": "json",
}

register(
    OpSpec(
        name="sf_calc",
        schema_version=SCHEMA_VERSION,
        inputs={**_XRAY_INPUTS, "hkl": "MillerArray"},
        outputs={"f_calc": "MillerArray"},
    )
)

register(
    OpSpec(
        name="sf_gradients",
        schema_version=SCHEMA_VERSION,
        inputs={**_XRAY_INPUTS, "d_target_d_f_calc": "MillerArray"},
        outputs={"gradients": "SfGradients"},
    )
)

register(
    OpSpec(
        name="target_eval",
        schema_version=SCHEMA_VERSION,
        inputs={"f_calc": "MillerArray", **_TARGET_INPUTS},
        outputs={"target": "TargetResult"},
    )
)

register(
    OpSpec(
        name="refine_gradients",
        schema_version=SCHEMA_VERSION,
        inputs={**_XRAY_INPUTS, **_TARGET_INPUTS},
        outputs={"f_calc": "MillerArray", "target": "TargetResult", "gradients": "SfGradients"},
    )
)

register(
    OpSpec(
        name="gauss_newton_hvp",
        schema_version=SCHEMA_VERSION,
        inputs={**_XRAY_INPUTS, "target": "TargetResult", "hkl": "MillerArray", "v": "SfGradients"},
        outputs={"hv": "SfGradients"},
    )
)

register(
    OpSpec(
        name="geometry_minimize",
        schema_version=SCHEMA_VERSION,
        inputs={"sites": "CartesianSites", "restraints": "GeometryRestraints", "params": "json"},
        outputs={"sites": "CartesianSites", "target": "json"},
    )
)


def main(argv: Iterable[str] | None = None) -> None:
    _ = argv
    json.dump(list_ops_json(), sys.stdout, indent=2)
    sys.stdout.write("\n")
