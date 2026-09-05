"""Shared op catalog. Installed with both client and worker extras.

Third-party packages register ops with :func:`register_op` (catalog + optional
impl). Workers discover them via ``--preload`` / ``PHRIDGE_PRELOAD`` or the
``phridge.ops`` setuptools entry-point group. See ``docs/extending.md``.
"""

from __future__ import annotations

import importlib
import json
import sys
from typing import Any, Callable, Iterable, Optional, Union

from phridge.models import SCHEMA_VERSION, OpSpec

_SPECS: dict[str, OpSpec] = {}
# External / late-bound implementations (avoids importing worker.ops on the client).
_IMPLS: dict[str, Callable[..., Any]] = {}


def register(spec: OpSpec) -> OpSpec:
    """Register an :class:`OpSpec` in the shared catalog (client + worker)."""
    _SPECS[spec.name] = spec
    return spec


def register_impl(name: str, impl: Callable[..., Any]) -> Callable[..., Any]:
    """Bind a worker callable to an already-registered op name."""
    if name not in _SPECS:
        raise KeyError(f"unknown op: {name} (register OpSpec first)")
    _IMPLS[name] = impl
    return impl


def register_op(
    name: Union[str, OpSpec],
    impl: Optional[Callable[..., Any]] = None,
    *,
    inputs: Optional[dict[str, str]] = None,
    outputs: Optional[dict[str, str]] = None,
    schema_version: int = SCHEMA_VERSION,
) -> OpSpec:
    """Register an op for clients and (optionally) workers.

    Parameters
    ----------
    name:
        Op name, or a fully built :class:`OpSpec`.
    impl:
        Worker callable. Omit on a client-only process if the worker loads the
        same module via ``--preload``. When provided, the impl is stored here
        without importing ``phridge.worker`` (so Phenix clients stay torch-free).
    inputs / outputs:
        Required when ``name`` is a string. Values are type tags
        (``array``, ``json``, or a LinkML class name such as ``MillerArray``).
    """
    if isinstance(name, OpSpec):
        spec = register(name)
    else:
        if inputs is None or outputs is None:
            raise TypeError("register_op(str, ...) requires inputs= and outputs=")
        spec = register(
            OpSpec(
                name=str(name),
                schema_version=schema_version,
                inputs=dict(inputs),
                outputs=dict(outputs),
            )
        )
    if impl is not None:
        _IMPLS[spec.name] = impl
    return spec


def get_op(name: str) -> OpSpec:
    try:
        return _SPECS[name]
    except KeyError as exc:
        raise KeyError(f"unknown op: {name}") from exc


def get_implementation(name: str) -> Optional[Callable[..., Any]]:
    """Return the worker callable for ``name``, if any.

    Looks in the late-bound map first, then the built-in
    ``phridge.worker.ops.IMPLEMENTATIONS`` table (imported lazily).
    """
    if name in _IMPLS:
        return _IMPLS[name]
    try:
        from phridge.worker.ops import IMPLEMENTATIONS
    except ImportError:
        return None
    return IMPLEMENTATIONS.get(name)


def list_ops() -> list[OpSpec]:
    return [spec for _, spec in sorted(_SPECS.items())]


def list_ops_json() -> list[dict]:
    return [spec.model_dump(mode="json") for spec in list_ops()]


def preload_modules(modules: Iterable[str]) -> list[str]:
    """Import dotted module paths for side-effect registration.

    Each module should call :func:`register_op` (or :func:`register` +
    :func:`register_impl`) at import time.
    """
    loaded: list[str] = []
    for raw in modules:
        name = str(raw).strip()
        if not name:
            continue
        importlib.import_module(name)
        loaded.append(name)
    return loaded


def load_entry_points(group: str = "phridge.ops") -> list[str]:
    """Load setuptools / hatch entry points in ``group``.

    Each entry point must resolve to a zero-argument callable that registers
    ops (typically via :func:`register_op`). Returns the entry-point names
    that were invoked.
    """
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover
        return []

    loaded: list[str] = []
    eps = entry_points()
    selected = eps.select(group=group) if hasattr(eps, "select") else eps.get(group, [])
    for ep in selected:
        obj = ep.load()
        if not callable(obj):
            raise TypeError(
                f"entry point {ep.name!r} in group {group!r} must be a callable, "
                f"got {type(obj).__name__}"
            )
        obj()
        loaded.append(ep.name)
    return loaded


def bootstrap_plugins(
    *,
    preload: Optional[Iterable[str]] = None,
    load_eps: bool = True,
    group: str = "phridge.ops",
) -> dict[str, list[str]]:
    """Load entry points then explicit preload modules (preload wins on clash)."""
    eps = load_entry_points(group=group) if load_eps else []
    mods = preload_modules(preload or ())
    return {"entry_points": eps, "preload": mods}


def _parse_preload_arg(values: Optional[Iterable[str]]) -> list[str]:
    """Flatten ``--preload a,b --preload c`` and ``PHRIDGE_PRELOAD`` style lists."""
    out: list[str] = []
    for value in values or ():
        for part in str(value).split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out


# --- built-in catalog -------------------------------------------------------

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


def main(argv: Optional[Iterable[str]] = None) -> None:
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Dump registered OpSpec JSON")
    parser.add_argument(
        "--preload",
        action="append",
        default=[],
        help="Import module(s) before dumping (repeatable; comma-separated ok)",
    )
    parser.add_argument(
        "--no-entry-points",
        action="store_true",
        help="Skip loading the phridge.ops entry-point group",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    env_preload = os.environ.get("PHRIDGE_PRELOAD", "")
    bootstrap_plugins(
        preload=_parse_preload_arg([*args.preload, env_preload]),
        load_eps=not args.no_entry_points,
    )
    json.dump(list_ops_json(), sys.stdout, indent=2)
    sys.stdout.write("\n")
