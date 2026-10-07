"""Scoring-rule manifests. Names match the target registry except ``ml_i``."""

from __future__ import annotations

from phridge.contracts.manifest import PluginManifest, Slot, Space
from phridge.ops import register_manifest

_SCORING: dict[str, PluginManifest] = {
    "ls": PluginManifest(name="ls", slot=Slot.scoring_rule, space=Space.D, produces=["score"]),
    "ml_f": PluginManifest(name="ml_f", slot=Slot.scoring_rule, space=Space.D, produces=["score"]),
    "ml_i": PluginManifest(
        name="mli_quad",
        slot=Slot.scoring_rule,
        space=Space.D,
        needs=["location", "sigma_a", "beta", "sigma_w", "nu"],
        produces=["score"],
    ),
}


def scoring_manifest(target_name: str) -> PluginManifest | None:
    return _SCORING.get(target_name)


def register() -> None:
    for manifest in _SCORING.values():
        register_manifest(manifest)
