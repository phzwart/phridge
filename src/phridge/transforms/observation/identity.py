"""Pass-through observation. Q records are already indexed like D."""

from __future__ import annotations

from typing import Any

from phridge.contracts.manifest import PluginManifest, Slot, Space, Tier
from phridge.ops import register_manifest

INDEX_FIELDS = ("hkl",)

MANIFEST = PluginManifest(
    name="identity",
    slot=Slot.observation,
    space=Space.transform,
    from_space=Space.Q,
    to_space=Space.D,
    tier=Tier.native,
    reads=["hkl"],
    produces=["hkl"],
)


class IdentityObservation:
    index_fields = INDEX_FIELDS

    def forward(self, record: Any) -> Any:
        return record

    def adjoint(self, cotangent: Any) -> Any:
        return cotangent


def register() -> None:
    register_manifest(MANIFEST, impl=IdentityObservation)
