"""Torch geometry-restraint energies from packed tables (no cctbx).

v1: bonds, angles, dihedrals. Chirality / planarity / nonbonded / ASU ignored.
Minimizer is always a ``torch.optim`` algorithm: ``lbfgs``, ``adam``, ``adamw``, or ``sgd``.

First-order methods support optional LR schedules (``cosine``, ``triangular``)
and SGD momentum.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from phridge.packing_geometry import PackedRestraints

_OPTIMIZERS = ("lbfgs", "adam", "adamw", "sgd")
_SCHEDULES = ("none", "cosine", "triangular")


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _rss_mb() -> float:
    """Current resident set size in MiB (Linux VmRSS; fallback via resource)."""
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0  # kB → MiB
    except OSError:
        pass
    import resource

    # Linux: ru_maxrss is kB; macOS: bytes. Prefer current RSS path above.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _optimizer_state_nbytes(opt: Any) -> int:
    import torch

    total = 0
    for state in opt.state.values():
        if not isinstance(state, dict):
            continue
        for value in state.values():
            if torch.is_tensor(value):
                total += int(value.numel()) * int(value.element_size())
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if torch.is_tensor(item):
                        total += int(item.numel()) * int(item.element_size())
    return total


def _normalize_schedule(schedule: Optional[str]) -> str:
    if schedule is None or schedule == "":
        return "none"
    name = str(schedule).lower().strip()
    if name not in _SCHEDULES:
        raise ValueError(f"schedule must be one of {_SCHEDULES}, got {schedule!r}")
    return name


def energy_and_sites(
    sites_cart: Any,
    restraints: PackedRestraints,
    *,
    max_iterations: int = 100,
    optimizer: str = "lbfgs",
    lr: Optional[float] = None,
    lr_min: Optional[float] = None,
    schedule: Optional[str] = None,
    momentum: float = 0.0,
    device: str = "cpu",
    dtype: Optional[Any] = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Minimize sites under packed restraints with a torch optimizer.

    First-order kwargs:

    - ``schedule``: ``none`` | ``cosine`` | ``triangular``
    - ``lr_min``: floor LR for schedules (default ``lr * 0.01``)
    - ``momentum``: SGD momentum (ignored for Adam / AdamW / LBFGS)

    Stats include ``n_steps``, ``n_calls`` (closure / loss evals), ``schedule``,
    ``momentum``, ``lr``, ``lr_min``, plus memory:

    - ``rss_before_mb`` / ``rss_after_mb`` / ``rss_delta_mb``: process VmRSS
    - ``optimizer_state_mb``: torch optimizer state tensors after the run
    - ``cuda_peak_mb``: ``torch.cuda.max_memory_allocated`` delta (0 on CPU)
    """
    import gc

    import torch

    name = optimizer.lower().strip()
    if name not in _OPTIMIZERS:
        raise ValueError(f"optimizer must be one of {_OPTIMIZERS}, got {optimizer!r}")
    sched = _normalize_schedule(schedule)

    if dtype is None:
        dtype = torch.float64
    xyz0 = torch.as_tensor(_as_numpy(sites_cart), dtype=dtype, device=device)
    if xyz0.ndim != 2 or xyz0.shape[1] != 3:
        raise ValueError("sites_cart must have shape (N, 3)")
    if int(xyz0.shape[0]) != restraints.n_sites:
        raise ValueError(f"n_sites mismatch: sites={xyz0.shape[0]} restraints={restraints.n_sites}")

    tables = _RestraintTables.from_packed(restraints, device=device, dtype=dtype)
    xyz = xyz0.clone().requires_grad_(True)
    before = float(tables.energy(xyz).detach().cpu())

    base_stats: dict[str, Any] = {
        "before": before,
        "optimizer": name,
        "schedule": sched,
        "momentum": float(momentum) if name == "sgd" else 0.0,
    }

    empty_mem = {
        "rss_before_mb": 0.0,
        "rss_after_mb": 0.0,
        "rss_delta_mb": 0.0,
        "optimizer_state_mb": 0.0,
        "cuda_peak_mb": 0.0,
    }

    if max_iterations <= 0:
        return _as_numpy(xyz0), {
            **base_stats,
            **empty_mem,
            "after": before,
            "n_iter": 0.0,
            "n_steps": 0.0,
            "n_calls": 0.0,
            "lr": float(lr) if lr is not None else None,
            "lr_min": float(lr_min) if lr_min is not None else None,
        }

    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        cuda_before = float(torch.cuda.memory_allocated()) / (1024.0 ** 2)
    else:
        cuda_before = 0.0
    rss_before = _rss_mb()

    n_calls = 0
    n_steps = 0
    used_lr: Optional[float] = None
    used_lr_min: Optional[float] = None
    opt: Any = None

    if name == "lbfgs":
        step_lr = 1.0 if lr is None else float(lr)
        used_lr = step_lr
        opt = torch.optim.LBFGS(
            [xyz],
            lr=step_lr,
            max_iter=max_iterations,
            line_search_fn="strong_wolfe",
            history_size=20,
            tolerance_grad=1e-9,
            tolerance_change=1e-11,
        )

        def closure():
            nonlocal n_calls
            opt.zero_grad(set_to_none=True)
            loss = tables.energy(xyz)
            loss.backward()
            n_calls += 1
            return loss

        opt.step(closure)
        state = opt.state[xyz]
        n_steps = int(state.get("n_iter", 0))
    else:
        defaults = {"adam": 1e-2, "adamw": 1e-2, "sgd": 1e-3}
        step_lr = defaults[name] if lr is None else float(lr)
        floor = (step_lr * 0.01) if lr_min is None else float(lr_min)
        used_lr, used_lr_min = step_lr, floor
        if name == "adam":
            opt = torch.optim.Adam([xyz], lr=step_lr)
        elif name == "adamw":
            opt = torch.optim.AdamW([xyz], lr=step_lr)
        else:
            opt = torch.optim.SGD([xyz], lr=step_lr, momentum=float(momentum))

        scheduler = None
        if sched == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(max_iterations, 1), eta_min=floor
            )
        elif sched == "triangular":
            period = max(max_iterations // 2, 1)
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                opt,
                base_lr=floor,
                max_lr=step_lr,
                step_size_up=period,
                step_size_down=max(max_iterations - period, 1),
                mode="triangular",
                cycle_momentum=False,
            )

        for _ in range(max_iterations):
            opt.zero_grad(set_to_none=True)
            loss = tables.energy(xyz)
            loss.backward()
            n_calls += 1
            torch.nn.utils.clip_grad_norm_([xyz], max_norm=1.0)
            opt.step()
            n_steps += 1
            if scheduler is not None:
                scheduler.step()

    after = float(tables.energy(xyz).detach().cpu())
    opt_state_mb = _optimizer_state_nbytes(opt) / (1024.0 ** 2) if opt is not None else 0.0
    rss_after = _rss_mb()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
        cuda_peak = float(torch.cuda.max_memory_allocated()) / (1024.0 ** 2) - cuda_before
    else:
        cuda_peak = 0.0

    return _as_numpy(xyz.detach()), {
        **base_stats,
        "after": after,
        "n_iter": float(n_calls),
        "n_steps": float(n_steps),
        "n_calls": float(n_calls),
        "lr": used_lr,
        "lr_min": used_lr_min,
        "rss_before_mb": float(rss_before),
        "rss_after_mb": float(rss_after),
        "rss_delta_mb": float(rss_after - rss_before),
        "optimizer_state_mb": float(opt_state_mb),
        "cuda_peak_mb": float(max(cuda_peak, 0.0)),
    }


class _RestraintTables:
    def __init__(
        self,
        *,
        bond_i: Any,
        bond_d0: Any,
        bond_w: Any,
        bond_slack: Any,
        angle_i: Any,
        angle_d0: Any,
        angle_w: Any,
        dih_i: Any,
        dih_d0: Any,
        dih_w: Any,
        dih_period: Any,
    ) -> None:
        self.bond_i = bond_i
        self.bond_d0 = bond_d0
        self.bond_w = bond_w
        self.bond_slack = bond_slack
        self.angle_i = angle_i
        self.angle_d0 = angle_d0
        self.angle_w = angle_w
        self.dih_i = dih_i
        self.dih_d0 = dih_d0
        self.dih_w = dih_w
        self.dih_period = dih_period

    @classmethod
    def from_packed(cls, packed: PackedRestraints, *, device: str, dtype: Any) -> "_RestraintTables":
        import torch

        def t_int(a: np.ndarray) -> Any:
            return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.long, device=device)

        def t_f(a: np.ndarray) -> Any:
            return torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=device)

        slack = packed.bond_slack
        if slack.size == 0 and packed.bond_i_seqs.shape[0]:
            slack = np.zeros(packed.bond_i_seqs.shape[0], dtype=np.float64)
        period = packed.dihedral_periodicity
        if period.size == 0 and packed.dihedral_i_seqs.shape[0]:
            period = np.ones(packed.dihedral_i_seqs.shape[0], dtype=np.int32)

        return cls(
            bond_i=t_int(packed.bond_i_seqs),
            bond_d0=t_f(packed.bond_distance_ideal),
            bond_w=t_f(packed.bond_weight),
            bond_slack=t_f(slack),
            angle_i=t_int(packed.angle_i_seqs),
            angle_d0=t_f(packed.angle_ideal),
            angle_w=t_f(packed.angle_weight),
            dih_i=t_int(packed.dihedral_i_seqs),
            dih_d0=t_f(packed.dihedral_angle_ideal),
            dih_w=t_f(packed.dihedral_weight),
            dih_period=t_int(period),
        )

    def energy(self, xyz: Any) -> Any:
        import torch

        e = xyz.new_zeros(())
        if self.bond_i.numel():
            a = xyz[self.bond_i[:, 0]]
            b = xyz[self.bond_i[:, 1]]
            d = torch.linalg.norm(a - b, dim=-1)
            delta = torch.abs(d - self.bond_d0) - self.bond_slack
            delta = torch.clamp(delta, min=0.0)
            e = e + (self.bond_w * delta * delta).sum()
        if self.angle_i.numel():
            i, j, k = self.angle_i[:, 0], self.angle_i[:, 1], self.angle_i[:, 2]
            v1 = xyz[i] - xyz[j]
            v2 = xyz[k] - xyz[j]
            n1 = torch.linalg.norm(v1, dim=-1).clamp_min(1e-12)
            n2 = torch.linalg.norm(v2, dim=-1).clamp_min(1e-12)
            cos = (v1 * v2).sum(-1) / (n1 * n2)
            cos = cos.clamp(-1.0 + 1e-8, 1.0 - 1e-8)
            theta = torch.rad2deg(torch.acos(cos))
            delta = theta - self.angle_d0
            e = e + (self.angle_w * delta * delta).sum()
        if self.dih_i.numel():
            i, j, k, l = self.dih_i[:, 0], self.dih_i[:, 1], self.dih_i[:, 2], self.dih_i[:, 3]
            b1 = xyz[j] - xyz[i]
            b2 = xyz[k] - xyz[j]
            b3 = xyz[l] - xyz[k]
            n1 = torch.cross(b1, b2, dim=-1)
            n2 = torch.cross(b2, b3, dim=-1)
            n1n = torch.linalg.norm(n1, dim=-1).clamp_min(1e-12)
            n2n = torch.linalg.norm(n2, dim=-1).clamp_min(1e-12)
            n1u = n1 / n1n.unsqueeze(-1)
            n2u = n2 / n2n.unsqueeze(-1)
            b2u = b2 / torch.linalg.norm(b2, dim=-1).clamp_min(1e-12).unsqueeze(-1)
            m1 = torch.cross(n1u, b2u, dim=-1)
            x = (n1u * n2u).sum(-1)
            y = (m1 * n2u).sum(-1)
            phi = torch.atan2(y, x)
            phi0 = torch.deg2rad(self.dih_d0)
            n = self.dih_period.to(dtype=phi.dtype).clamp_min(1.0)
            e = e + (self.dih_w * (1.0 - torch.cos(n * (phi - phi0)))).sum()
        return e
