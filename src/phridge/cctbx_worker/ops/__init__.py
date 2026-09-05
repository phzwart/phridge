"""CCTBX worker implementations keyed by op name."""

from __future__ import annotations

from typing import Any, Callable

from phridge.cctbx_worker.ops.geometry_ops import (
    build_geometry_restraints,
    geometry_restraints_energy_grad,
)

IMPLEMENTATIONS: dict[str, Callable[..., Any]] = {
    "build_geometry_restraints": build_geometry_restraints,
    "geometry_restraints_energy_grad": geometry_restraints_energy_grad,
}
