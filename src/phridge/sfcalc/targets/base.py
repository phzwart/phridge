"""Target base class, observation container, registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np


@dataclass
class Observations:
    """Per-reflection experimental data as torch tensors (same order as f_calc).

    ``data`` is |F_obs| or I_obs depending on the target. Optional arrays
    are None when absent. ``r_free`` True marks test reflections; targets
    are evaluated on the work set (cctbx ``target_work``).
    """

    data: Any
    sigmas: Optional[Any] = None
    weights: Optional[Any] = None
    r_free: Optional[Any] = None
    epsilon: Optional[Any] = None
    centric: Optional[Any] = None
    alpha: Optional[Any] = None
    beta: Optional[Any] = None

    @classmethod
    def from_numpy(cls, device: str = "cpu", dtype: Any = None, **arrays: Optional[np.ndarray]) -> "Observations":
        import torch

        dtype = dtype or torch.float64
        out: dict[str, Any] = {}
        for name, arr in arrays.items():
            if arr is None:
                out[name] = None
                continue
            arr = np.asarray(arr)
            if name in ("r_free", "centric"):
                out[name] = torch.as_tensor(arr.astype(bool), device=device)
            else:
                out[name] = torch.as_tensor(arr.astype(np.float64), dtype=dtype, device=device)
        return cls(**out)

    @property
    def work(self):
        """Boolean mask of work reflections."""
        import torch

        if self.r_free is None:
            return torch.ones_like(self.data, dtype=torch.bool)
        return ~self.r_free

    @property
    def n_work(self) -> int:
        return int(self.work.sum().item())


@dataclass
class TargetEval:
    value: float
    per_reflection: np.ndarray  # (N,) target per reflection, all reflections
    d_target_d_f_calc: np.ndarray  # (N,) complex, zero on free reflections
    curv_radial: Optional[np.ndarray] = None  # d2 g / d|F|^2 (amplitude targets)
    curv_tangential: Optional[np.ndarray] = None  # (dg/d|F|) / |F|
    value_test: Optional[float] = None
    scale_factor: Optional[float] = None


class Target:
    """Base: subclasses implement ``per_reflection``.

    ``per_reflection(f_calc, obs)`` returns t_h for every reflection.
    ``value`` = mean over work reflections by default (cctbx ML convention);
    override ``reduce`` for other normalizations.
    ``amplitude_only`` = True means t_h depends on F_h only through |F_h|,
    which lets ``evaluate`` return per-reflection curvatures.
    """

    name: str = "target"
    amplitude_only: bool = True

    def __init__(self, **options: Any) -> None:
        self.options = options

    # -- to override ---------------------------------------------------------
    def per_reflection(self, f_calc, obs: Observations):
        raise NotImplementedError

    def reduce(self, t, obs: Observations):
        work = obs.work
        n = work.sum().clamp(min=1)
        return (t * work).sum() / n

    def prepare(self, f_calc, obs: Observations) -> None:
        """Hook for data-dependent constants (e.g. LS scale). Called once per evaluate."""

    # -- generic machinery ---------------------------------------------------
    def value(self, f_calc, obs: Observations):
        return self.reduce(self.per_reflection(f_calc, obs), obs)

    def evaluate(self, f_calc, obs: Observations, compute_curvature: bool = True) -> TargetEval:
        import torch

        f = f_calc.detach().clone().requires_grad_(True)
        self.prepare(f.detach(), obs)
        t = self.per_reflection(f, obs)
        q = self.reduce(t, obs)
        (g,) = torch.autograd.grad(q, f)
        result = TargetEval(
            value=float(q.item()),
            per_reflection=t.detach().cpu().numpy().astype(np.float64),
            d_target_d_f_calc=g.detach().cpu().numpy().astype(np.complex128),
            scale_factor=getattr(self, "scale_factor_", None),
        )
        if obs.r_free is not None and bool(obs.r_free.any()):
            n_test = obs.r_free.sum()
            result.value_test = float(((t.detach() * obs.r_free).sum() / n_test).item())
        if compute_curvature and self.amplitude_only:
            result.curv_radial, result.curv_tangential = self._amplitude_curvature(f_calc.detach(), obs)
        return result

    def _amplitude_curvature(self, f_calc, obs: Observations):
        """For separable amplitude targets: g''(|F|) and g'(|F|)/|F| per reflection."""
        import torch

        amp = f_calc.abs().clone().requires_grad_(True)
        phase = f_calc / f_calc.abs().clamp(min=1e-300)
        f = amp * phase
        t = self.per_reflection(f, obs)
        q = self.reduce(t, obs)
        (g1,) = torch.autograd.grad(q, amp, create_graph=True)
        (g2,) = torch.autograd.grad(g1.sum(), amp)
        radial = g2.detach()
        tangential = (g1.detach() / amp.detach().clamp(min=1e-300))
        return (
            radial.cpu().numpy().astype(np.float64),
            tangential.cpu().numpy().astype(np.float64),
        )


_REGISTRY: dict[str, Callable[..., Target]] = {}


def register_target(name: str):
    def deco(cls):
        _REGISTRY[name] = cls
        cls.name = name
        return cls

    return deco


def build_target(spec: dict) -> Target:
    """spec = {"name": "ml_f", ...options}."""
    spec = dict(spec)
    name = spec.pop("name")
    try:
        cls = _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown target {name!r}; known: {sorted(_REGISTRY)}") from exc
    return cls(**spec)


def list_targets() -> list[str]:
    return sorted(_REGISTRY)
