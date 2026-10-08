"""The four optimizers register and a one-block grid picks the best node."""

from __future__ import annotations

import torch

from phridge.contracts.manifest import Slot
from phridge.optim import MANIFESTS, Budget, GridSearch, register
from phridge.ops import get_manifest


def test_optimizer_manifests_register() -> None:
    register()
    register()
    for name, manifest in MANIFESTS.items():
        stored = get_manifest(name)
        assert stored.slot == Slot.optimizer
        assert list(stored.needs) == list(manifest.needs)


def test_grid_picks_the_lowest_node() -> None:
    class _Bowl:
        provides: tuple[str, ...] = ()

        def value(self, x: torch.Tensor) -> torch.Tensor:
            return ((x - 0.3) ** 2).sum()

    result = GridSearch(points=9).step(
        _Bowl(),
        torch.tensor([0.0, 0.0], dtype=torch.float64),
        (0.0, 1.0),
        Budget(iterations=9),
    )
    assert abs(float(result.x1[0]) - 0.25) < 0.2
    assert result.n_evals == 9
