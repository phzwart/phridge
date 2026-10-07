"""Scoring-rule wrapper. Reads ``Target.evaluate`` and does not change its formula."""

from __future__ import annotations

from typing import Any

from phridge.d.rules.amplitude.base import Observations, Target


class ScoringRule:
    def __init__(self, target: Target) -> None:
        self.target = target

    def score(self, f_calc: Any, obs: Observations, mask: Any) -> tuple[float, Any]:
        import torch

        f = f_calc.detach().clone().requires_grad_(True)
        aligned = obs.to_like(f)
        self.target.prepare(f.detach(), aligned)
        per = self.target.per_reflection(f, aligned)
        weight = torch.as_tensor(mask, device=per.device, dtype=per.dtype)
        total = (per * weight).sum()
        (grad,) = torch.autograd.grad(total, f)
        self.target.evaluate(f_calc.detach(), aligned, compute_curvature=False)
        return float(total.detach().item()), grad.detach()
