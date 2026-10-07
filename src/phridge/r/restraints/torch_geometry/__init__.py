"""Torch geometry minimization from packed restraint tables (no cctbx)."""

from __future__ import annotations

from phridge.r.restraints.torch_geometry.energy import energy_and_sites


def register() -> None:
    from phridge.r.restraints.torch_geometry.provider import register as _register

    _register()


__all__ = ["energy_and_sites", "register"]
