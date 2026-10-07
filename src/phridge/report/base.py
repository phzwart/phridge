"""Report statistics. ``compute`` receives boolean masks, not raw labels."""

from __future__ import annotations

from typing import Any


class Statistic:
    def compute(self, pred: Any, obs: Any, masks: dict[str, Any]) -> dict[str, Any]:
        """Summarize ``pred`` and ``obs`` on each boolean mask."""
        return {name: int(mask.sum()) for name, mask in masks.items()}
