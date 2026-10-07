"""Continuum and atomic components of R."""

from __future__ import annotations


def register() -> None:
    from phridge.r.components.mask_manifest import register as _register

    _register()
