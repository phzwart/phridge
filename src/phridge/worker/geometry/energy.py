"""Torch geometry-restraint energies from packed tables (no cctbx).

v1: bonds, angles, dihedrals. Chirality / planarity / nonbonded / ASU ignored.
Minimizer is always a ``torch.optim`` algorithm: ``lbfgs``, ``adam``, or ``sgd``.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from phridge.packing_geometry import PackedRestraints

_OPTIMIZERS = ("lbfgs", "adam", "sgd")


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def energy_and_sites(
    sites_cart: Any,
    restraints: PackedRestraints,
    *,
    max_iterations: int = 100,
    optimizer: str = "lbfgs",
    lr: Optional[float] = None,
    device: str = "cpu",
    dtype: Optional[Any] = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Minimize sites under packed restraints with a torch optimizer.

    Returns ``(sites_out, {"before", "after", "n_iter", "optimizer"})``.
    ``max_iterations=0`` evaluates energy only (no steps).
    """
    import torch

    name = optimizer.lower().strip()
    if name not in _OPTIMIZERS:
        raise ValueError(f"optimizer must be one of {_OPTIMIZERS}, got {optimizer!r}")

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

    if max_iterations <= 0:
        return _as_numpy(xyz0), {
            "before": before,
            "after": before,
            "n_iter": 0.0,
            "optimizer": name,
        }

    n_iter = 0
    if name == "lbfgs":
        step_lr = 1.0 if lr is None else float(lr)
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
            nonlocal n_iter
            opt.zero_grad(set_to_none=True)
            loss = tables.energy(xyz)
            loss.backward()
            n_iter += 1
            return loss

        opt.step(closure)
    else:
        step_lr = (1e-2 if name == "adam" else 1e-4) if lr is None else float(lr)
        if name == "adam":
            opt = torch.optim.Adam([xyz], lr=step_lr)
        else:
            opt = torch.optim.SGD([xyz], lr=step_lr, momentum=0.0)
        for _ in range(max_iterations):
            opt.zero_grad(set_to_none=True)
            loss = tables.energy(xyz)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([xyz], max_norm=1.0)
            opt.step()
            n_iter += 1

    after = float(tables.energy(xyz).detach().cpu())
    return _as_numpy(xyz.detach()), {
        "before": before,
        "after": after,
        "n_iter": float(n_iter),
        "optimizer": name,
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
