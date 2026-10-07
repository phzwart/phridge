"""Reciprocal-space perturbation modes."""

from __future__ import annotations

from phridge.contracts.manifest import BlockDecl, PluginManifest, Slot, Space, Tier
from phridge.ops import register_manifest

MANIFEST = PluginManifest(
    name="perturbation_modes",
    slot=Slot.r_component,
    space=Space.R,
    tier=Tier.native,
    blocks=[
        BlockDecl(id="loadings", shape=["n_refl", "rank"]),
        BlockDecl(id="amplitudes", shape=["n_dataset", "rank"]),
    ],
)


def register() -> None:
    register_manifest(MANIFEST)
