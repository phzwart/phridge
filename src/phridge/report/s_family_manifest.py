"""S-family statistic manifest."""

from __future__ import annotations

from phridge.contracts.manifest import PluginManifest, Slot, Space
from phridge.ops import register_manifest
from phridge.report.base import Statistic

MANIFEST = PluginManifest(
    name="s_family",
    slot=Slot.statistic,
    space=Space.D,
    produces=["s_post", "s_prior"],
)


class SFamilyStatistic(Statistic):
    def compute(self, pred, obs, masks):
        return super().compute(pred, obs, masks)


def register() -> None:
    register_manifest(MANIFEST, impl=SFamilyStatistic)
