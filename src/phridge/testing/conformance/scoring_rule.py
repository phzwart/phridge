"""Scoring-rule conformance."""

from __future__ import annotations

import torch

from phridge.contracts.tensors import Predictive
from phridge.d.rules.amplitude.base import Observations, Target
from phridge.d.rules.base import ScoringRule
from phridge.d.rules.manifests import scoring_manifest


class _HolomorphicShift(Target):
    """``sum (F - y)``; analytic, so a complex step hits ``d_location``."""

    name = "holomorphic_shift"
    amplitude_only = False

    def per_reflection(self, f_calc, obs: Observations):
        return (f_calc - obs.data.to(f_calc.dtype)).real


def test_needs_subset_of_predictive() -> None:
    manifest = scoring_manifest("ml_i")
    assert manifest is not None
    present = {"location", "sigma_a", "beta", "sigma_w", "nu", "extra"}
    assert set(manifest.needs) <= present
    Predictive({name: torch.zeros(2) for name in present}).require(*manifest.needs)


def test_score_equals_masked_sum() -> None:
    n = 50
    y = torch.linspace(0.2, 2.0, n, dtype=torch.float64)
    f = torch.complex(y + 0.1, torch.full((n,), 0.05, dtype=torch.float64))
    obs = Observations.from_numpy(data=y.numpy())
    mask = torch.tensor([i % 5 != 0 for i in range(n)], dtype=torch.float64)
    rule = ScoringRule(_HolomorphicShift())
    value, _ = rule.score(f, obs, mask)
    expected = float(((f.real - y) * mask).sum())
    assert abs(value - expected) < 1e-8


def test_complex_step_d_location() -> None:
    n = 50
    y = torch.linspace(0.2, 2.0, n, dtype=torch.float64)
    f = torch.complex(y + 0.1, torch.full((n,), 0.05, dtype=torch.float64))
    obs = Observations.from_numpy(data=y.numpy())
    mask = torch.ones(n, dtype=torch.float64)
    rule = ScoringRule(_HolomorphicShift())
    _, grad = rule.score(f, obs, mask)
    eps = 1e-20
    real = f.real.detach().numpy()
    y_np = y.numpy()
    mask_np = mask.numpy()

    def _at(shifted_real):
        return complex(((shifted_real - y_np) * mask_np).sum())

    for i in range(n):
        stepped = real.astype(complex).copy()
        stepped[i] = stepped[i] + 1j * eps
        numeric = _at(stepped).imag / eps
        assert abs(numeric - float(grad[i].real)) < 1e-6
