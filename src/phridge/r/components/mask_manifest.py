"""Mask component manifest."""

from __future__ import annotations

from phridge.contracts.manifest import PluginManifest, Slot, Space, Tier
from phridge.ops import register_manifest

MANIFEST = PluginManifest(
    name="distance_mask",
    slot=Slot.r_component,
    space=Space.R,
    tier=Tier.native,
    produces=["mask_density"],
)


def register() -> None:
    register_manifest(MANIFEST)
