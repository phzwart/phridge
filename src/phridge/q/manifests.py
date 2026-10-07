"""Q-component manifests."""

from __future__ import annotations

from phridge.contracts.manifest import PluginManifest, Slot, Space
from phridge.ops import register_manifest
from phridge.q.base import FieldScale

MANIFESTS: dict[str, PluginManifest] = {
    "wilson_aniso": PluginManifest(
        name="wilson_aniso", slot=Slot.q_component, space=Space.Q, produces=["sigma_w"]
    ),
    "sigma_a_beta": PluginManifest(
        name="sigma_a_beta", slot=Slot.q_component, space=Space.Q, produces=["sigma_a", "beta"]
    ),
    "local_sigma_a": PluginManifest(
        name="local_sigma_a", slot=Slot.q_component, space=Space.Q, produces=["sigma_a"]
    ),
    "spatial_sigma_a": PluginManifest(
        name="spatial_sigma_a", slot=Slot.q_component, space=Space.Q, produces=["sigma_a"]
    ),
    "solvent": PluginManifest(
        name="solvent", slot=Slot.q_component, space=Space.Q, produces=["k_mask"]
    ),
}


def component_for(name: str) -> FieldScale:
    manifest = MANIFESTS[name]
    return FieldScale(1.0, list(manifest.produces))


def register() -> None:
    for name, manifest in MANIFESTS.items():
        register_manifest(manifest, impl=component_for(name))
    from phridge.policies.calibrate_intensity import register as register_calibration

    register_calibration()
