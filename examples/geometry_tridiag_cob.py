"""Residue block-tridiagonal change of basis, then stock Adam / L-BFGS.

The geometry test in docs/geometry_curvature.md measured CG on one damped
Gauss-Newton system. This runs the same chain as a minimisation: factor the
residue block-tridiagonal curvature once (block Cholesky M = G G^T), optimise
z in x = x_ref + G^{-T} z, and compare with plain coordinates, Jacobi scaling,
and Levenberg-Marquardt.

Usage: python examples/geometry_tridiag_cob.py [n_atoms]
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch

from phridge.packing_geometry import PackedRestraints
from phridge.worker.geometry.curvature import (
    BlockTridiagonalPreconditioner,
    RestraintCurvature,
    gn_sparse_coo,
)
from phridge.worker.geometry.energy import _RestraintTables, energy_and_sites


def chain(n: int, seed: int = 0, shake: float = 0.3, n_long: int = 5):
    """Ideal polymer plus the documented restraint set.

    Bond σ = 0.02 Å, angle σ = 3°, dihedral weight 1, and ``n_long`` long-range
    bonds whose ideal lengths are the ideal-geometry distances (so the built
    chain is still a minimum). Long-range pairs skip residue groups, which is
    the part the block-tridiagonal factor drops.
    """
    rng = np.random.default_rng(seed)
    xyz = np.zeros((n, 3))
    xyz[1] = [1.5, 0, 0]
    for i in range(2, n):
        b1 = xyz[i - 1] - xyz[i - 2]
        b1 /= np.linalg.norm(b1)
        r = rng.normal(size=3)
        r -= r.dot(b1) * b1
        r /= np.linalg.norm(r)
        th = np.deg2rad(180 - 110)
        xyz[i] = xyz[i - 1] + 1.5 * (np.cos(th) * b1 + np.sin(th) * r)
    bonds = np.array([[i, i + 1] for i in range(n - 1)], dtype=np.int32)
    angles = np.array([[i, i + 1, i + 2] for i in range(n - 2)], dtype=np.int32)
    dih = np.array([[i, i + 1, i + 2, i + 3] for i in range(n - 3)], dtype=np.int32)

    def ang(a, b, c):
        v1, v2 = a - b, c - b
        return np.degrees(np.arccos(np.clip(v1.dot(v2) / np.linalg.norm(v1) / np.linalg.norm(v2), -1, 1)))

    def dihed(p0, p1, p2, p3):
        b1, b2, b3 = p1 - p0, p2 - p1, p3 - p2
        n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
        n1 /= np.linalg.norm(n1)
        n2 /= np.linalg.norm(n2)
        m1 = np.cross(n1, b2 / np.linalg.norm(b2))
        return np.degrees(np.arctan2(m1.dot(n2), n1.dot(n2)))

    bond_w = np.full(n - 1, 1 / 0.02**2)
    if n_long:
        anchors = np.linspace(0, n // 2 - 1, n_long).astype(np.int32)
        extra = np.array([[int(a), int(a) + n // 2] for a in anchors], dtype=np.int32)
        extra_d = np.linalg.norm(xyz[extra[:, 0]] - xyz[extra[:, 1]], axis=1)
        bonds = np.concatenate([bonds, extra], axis=0)
        bond_w = np.concatenate([bond_w, np.full(n_long, 1 / 0.02**2)])
        bond_d = np.concatenate([np.full(n - 1, 1.5), extra_d])
    else:
        bond_d = np.full(n - 1, 1.5)
    pr = PackedRestraints(
        n_sites=n,
        bond_i_seqs=bonds,
        bond_distance_ideal=bond_d,
        bond_weight=bond_w,
        bond_slack=np.zeros(len(bonds)),
        angle_i_seqs=angles,
        angle_ideal=np.array([ang(*xyz[a]) for a in angles]),
        angle_weight=np.full(n - 2, 1 / 3.0**2),
        dihedral_i_seqs=dih,
        dihedral_angle_ideal=np.array([dihed(*xyz[d]) for d in dih]),
        dihedral_weight=np.full(n - 3, 1.0),
        dihedral_periodicity=np.ones(n - 3, dtype=np.int32),
    )
    start = xyz + shake * rng.normal(size=xyz.shape)
    return xyz, start, pr


def _wrap_deg(delta: np.ndarray) -> np.ndarray:
    return (delta + 180.0) % 360.0 - 180.0


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else float("nan")


def _maxabs(x: np.ndarray) -> float:
    return float(np.max(np.abs(x))) if x.size else float("nan")


def _pair(dev: dict[str, float], name: str) -> str:
    return f"{dev[name + '_rms']:.3g}/{dev[name + '_max']:.3g}"


def restraint_deviations(xyz: np.ndarray, pr: PackedRestraints) -> dict[str, float]:
    """Coordinate-free restraint residuals: covalent bonds (Å), angles and dihedrals (degrees)."""
    xyz = np.asarray(xyz, dtype=np.float64)
    bi = np.asarray(pr.bond_i_seqs)
    dist = np.linalg.norm(xyz[bi[:, 0]] - xyz[bi[:, 1]], axis=1)
    bond = dist - np.asarray(pr.bond_distance_ideal, dtype=np.float64)
    covalent = np.isclose(np.asarray(pr.bond_distance_ideal, dtype=np.float64), 1.5)
    bond_c = bond[covalent] if np.any(covalent) else bond
    bond_l = bond[~covalent]

    ai = np.asarray(pr.angle_i_seqs)
    v1 = xyz[ai[:, 0]] - xyz[ai[:, 1]]
    v2 = xyz[ai[:, 2]] - xyz[ai[:, 1]]
    cos = np.sum(v1 * v2, axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))
    theta = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    angle = theta - np.asarray(pr.angle_ideal, dtype=np.float64)

    di = np.asarray(pr.dihedral_i_seqs)
    b1 = xyz[di[:, 1]] - xyz[di[:, 0]]
    b2 = xyz[di[:, 2]] - xyz[di[:, 1]]
    b3 = xyz[di[:, 3]] - xyz[di[:, 2]]
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1u = n1 / np.maximum(np.linalg.norm(n1, axis=1, keepdims=True), 1e-12)
    n2u = n2 / np.maximum(np.linalg.norm(n2, axis=1, keepdims=True), 1e-12)
    b2u = b2 / np.maximum(np.linalg.norm(b2, axis=1, keepdims=True), 1e-12)
    m1 = np.cross(n1u, b2u)
    phi = np.degrees(np.arctan2(np.sum(m1 * n2u, axis=1), np.sum(n1u * n2u, axis=1)))
    dihedral = _wrap_deg(phi - np.asarray(pr.dihedral_angle_ideal, dtype=np.float64))
    return {
        "bond_rms": _rms(bond_c),
        "bond_max": _maxabs(bond_c),
        "long_bond_rms": _rms(bond_l),
        "angle_rms": _rms(angle),
        "angle_max": _maxabs(angle),
        "dihedral_rms": _rms(dihedral),
        "dihedral_max": _maxabs(dihedral),
    }


def residue_groups(n: int, size: int = 5) -> list[np.ndarray]:
    return [np.arange(s, min(s + size, n)) for s in range(0, n, size)]


class BlockChol:
    """Block Cholesky of the damped residue-block-tridiagonal GN matrix.

    M = G G^T with G block-lower bidiagonal. ``displace(z)`` applies G^{-T},
    so x = x_ref + displace(z) whitens that metric: dx^T M dx = ||z||^2.
    """

    def __init__(self, curv: RestraintCurvature, xyz: torch.Tensor, groups: list[np.ndarray], damping: float = 1e-3):
        import scipy.sparse as sp

        rows, cols, vals, n3 = gn_sparse_coo(curv, xyz.detach())
        H = sp.csr_matrix((vals, (rows, cols)), shape=(n3, n3))
        d = np.asarray(H.diagonal())
        pos = d[d > 0]
        ref = float(np.median(pos)) if pos.size else 1.0
        self.mu = float(damping) * ref
        self.n3 = n3
        self.groups = [np.asarray(g, dtype=np.int64).reshape(-1) for g in groups]
        self.coord = [(3 * g[:, None] + np.arange(3)).reshape(-1) for g in self.groups]
        D = [H[c][:, c].toarray() + self.mu * np.eye(c.size) for c in self.coord]
        E = [H[self.coord[i + 1]][:, self.coord[i]].toarray() for i in range(len(self.coord) - 1)]

        R: list[np.ndarray] = []
        Gsub: list[np.ndarray] = []  # G_{i, i-1}, shape (n_i, n_{i-1}); empty at i = 0
        jitter_used = 0.0
        S = D[0]
        for i in range(len(self.coord)):
            if i:
                # G_sub = E @ R_prev^{-T}
                X = np.linalg.solve(R[-1], E[i - 1].T)
                G = X.T
                S = D[i] - G @ G.T
                Gsub.append(G)
            R.append(_chol_spd(S))
            # record the largest jitter _chol_spd had to add
            jitter_used = max(jitter_used, float(np.linalg.norm(R[-1] @ R[-1].T - S)))
        self.jitter = jitter_used
        self.R = [torch.as_tensor(r, dtype=torch.float64) for r in R]
        self.Gsub = [torch.as_tensor(g, dtype=torch.float64) for g in Gsub]
        self.coord_t = [torch.as_tensor(c, dtype=torch.long) for c in self.coord]

    def displace(self, z: torch.Tensor) -> torch.Tensor:
        """Apply G^{-T} to a flat vector. Differentiable in ``z``."""
        n_b = len(self.coord_t)
        v_blocks: list[torch.Tensor | None] = [None] * n_b
        for i in range(n_b - 1, -1, -1):
            rhs = z[self.coord_t[i]]
            if i + 1 < n_b:
                rhs = rhs - self.Gsub[i].T @ v_blocks[i + 1]
            # R is lower, so R^T is upper.
            v_blocks[i] = torch.linalg.solve_triangular(self.R[i].T, rhs.unsqueeze(-1), upper=True).squeeze(-1)
        out = z.new_zeros(self.n3)
        for i in range(n_b):
            out = out.index_copy(0, self.coord_t[i], v_blocks[i])
        return out


def _chol_spd(S: np.ndarray) -> np.ndarray:
    jitter = 0.0
    scale = float(np.trace(S) / max(S.shape[0], 1))
    eye = np.eye(S.shape[0])
    for _ in range(10):
        try:
            return np.linalg.cholesky(S + jitter * eye)
        except np.linalg.LinAlgError:
            jitter = 1e-14 * abs(scale) if jitter == 0.0 else jitter * 100.0
    raise np.linalg.LinAlgError("block Schur complement is not SPD")


def _rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def _aligned_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """RMSD after removing rigid motion of ``a`` onto ``b``."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ac = a - a.mean(axis=0)
    bc = b - b.mean(axis=0)
    u, _, vt = np.linalg.svd(ac.T @ bc)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u = u.copy()
        u[:, -1] *= -1
        rot = u @ vt
    return _rmsd(ac @ rot, bc)


def run_cob(
    name: str,
    start: np.ndarray,
    ideal: np.ndarray,
    tables: _RestraintTables,
    curv: RestraintCurvature,
    groups: list[np.ndarray],
    *,
    optimizer: str,
    max_iterations: int,
    lr: float,
    refresh: int,
    clip: float | None,
    damping: float,
) -> dict:
    """Minimise in the block-tridiagonal basis. ``refresh`` rebuilds G every that many steps."""
    x_np = np.asarray(start, dtype=np.float64)
    hist: list[float] = []
    n_calls = 0
    n_refresh = 0
    t0 = time.perf_counter()

    def energy_of(x_t: torch.Tensor) -> torch.Tensor:
        nonlocal n_calls
        n_calls += 1
        return tables.energy(x_t)

    step_done = 0
    while step_done < max_iterations:
        chunk = max_iterations - step_done
        if refresh > 0:
            chunk = min(chunk, refresh)
        factor = BlockChol(curv, torch.as_tensor(x_np), groups, damping=damping)
        anchor = torch.as_tensor(x_np, dtype=torch.float64)
        z = torch.zeros(factor.n3, dtype=torch.float64, requires_grad=True)
        if optimizer == "lbfgs":
            opt = torch.optim.LBFGS(
                [z],
                lr=lr,
                max_iter=chunk,
                line_search_fn="strong_wolfe",
                history_size=20,
                tolerance_grad=1e-12,
                tolerance_change=1e-14,
            )

            def closure():
                opt.zero_grad(set_to_none=True)
                x = anchor + factor.displace(z).view(-1, 3)
                loss = energy_of(x)
                loss.backward()
                hist.append(float(loss.detach()))
                return loss

            opt.step(closure)
            done = int(opt.state[z].get("n_iter", 0))
            step_done += done
            if done == 0:
                break
        else:
            opt = torch.optim.Adam([z], lr=lr) if optimizer == "adam" else torch.optim.SGD([z], lr=lr)
            done = 0
            for _ in range(chunk):
                opt.zero_grad(set_to_none=True)
                x = anchor + factor.displace(z).view(-1, 3)
                loss = energy_of(x)
                loss.backward()
                if clip is not None:
                    torch.nn.utils.clip_grad_norm_([z], max_norm=clip)
                hist.append(float(loss.detach()))
                opt.step()
                done += 1
                step_done += 1
        with torch.no_grad():
            x_np = (anchor + factor.displace(z).view(-1, 3)).detach().numpy()
        n_refresh += 1
        # A short chunk means the line search stalled in this basis. Rebuilding
        # is the point of a refresh; a frozen basis has nothing new to try.
        if optimizer == "lbfgs" and done < chunk and refresh <= 0:
            break
    final = float(tables.energy(torch.as_tensor(x_np)))
    return {
        "name": name,
        "optimizer": optimizer,
        "E": final,
        "rmsd": _rmsd(x_np, ideal),
        "rmsd_aligned": _aligned_rmsd(x_np, ideal),
        "xyz": x_np,
        "steps": step_done,
        "evals": n_calls,
        "refreshes": n_refresh,
        "seconds": time.perf_counter() - t0,
        "hist": hist,
        "mu": factor.mu,
        "jitter": factor.jitter,
    }


def _worst_group(xyz: np.ndarray, pr: PackedRestraints, groups: list[np.ndarray]) -> int:
    """Residue group touched by the dihedral furthest from its target."""
    di = np.asarray(pr.dihedral_i_seqs)
    xyz = np.asarray(xyz, dtype=np.float64)
    b1 = xyz[di[:, 1]] - xyz[di[:, 0]]
    b2 = xyz[di[:, 2]] - xyz[di[:, 1]]
    b3 = xyz[di[:, 3]] - xyz[di[:, 2]]
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1u = n1 / np.maximum(np.linalg.norm(n1, axis=1, keepdims=True), 1e-12)
    n2u = n2 / np.maximum(np.linalg.norm(n2, axis=1, keepdims=True), 1e-12)
    b2u = b2 / np.maximum(np.linalg.norm(b2, axis=1, keepdims=True), 1e-12)
    m1 = np.cross(n1u, b2u)
    phi = np.degrees(np.arctan2(np.sum(m1 * n2u, axis=1), np.sum(n1u * n2u, axis=1)))
    delta = np.abs(_wrap_deg(phi - np.asarray(pr.dihedral_angle_ideal, dtype=np.float64)))
    owner = np.full(xyz.shape[0], -1, dtype=np.int32)
    for gi, g in enumerate(groups):
        owner[np.asarray(g, dtype=np.int64)] = gi
    touched = owner[di]
    score = np.zeros(len(groups))
    for atoms, err in zip(touched, delta):
        for gi in atoms:
            if gi >= 0:
                score[gi] = max(score[gi], float(err))
    return int(np.argmax(score))


def _local_kick(
    xyz: np.ndarray,
    tables: _RestraintTables,
    curv: RestraintCurvature,
    groups: list[np.ndarray],
    pr: PackedRestraints,
    *,
    damping: float,
    eta: float,
    n_trials: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Smallest-energy kick of one residue in the block-tridiagonal metric.

    Only that group's whitened coordinates move, by length ``eta``. Stiff
    directions inside the group stay small in Cartesian space; soft ones,
    including the bad torsion, can move.
    """
    factor = BlockChol(curv, torch.as_tensor(xyz), groups, damping=damping)
    gi = _worst_group(xyz, pr, groups)
    coord = factor.coord[gi]
    base = torch.as_tensor(np.asarray(xyz, dtype=np.float64).reshape(-1))
    best_x = np.asarray(xyz, dtype=np.float64)
    best_e = float("inf")
    for _ in range(n_trials):
        direction = rng.normal(size=coord.size)
        direction /= max(np.linalg.norm(direction), 1e-12)
        z = torch.zeros(factor.n3, dtype=torch.float64)
        z[torch.as_tensor(coord, dtype=torch.long)] = torch.as_tensor(eta * direction, dtype=torch.float64)
        with torch.no_grad():
            trial = (base + factor.displace(z)).view(-1, 3)
        e_try = float(tables.energy(trial))
        if np.isfinite(e_try) and e_try < best_e:
            best_e = e_try
            best_x = trial.numpy()
    return best_x, best_e


def run_mstep_restarts(
    name: str,
    start: np.ndarray,
    ideal: np.ndarray,
    tables: _RestraintTables,
    curv: RestraintCurvature,
    groups: list[np.ndarray],
    pr: PackedRestraints,
    *,
    optimizer: str,
    m_steps: int,
    n_cycles: int,
    lr: float,
    clip: float | None,
    damping: float,
    perturb: bool,
    eta: float = 0.5,
) -> dict:
    """M optimizer steps per cycle, then a local perturbation restart when a cycle stalls."""
    x = np.asarray(start, dtype=np.float64).copy()
    best_x = x.copy()
    best_e = float(tables.energy(torch.as_tensor(x)))
    rng = np.random.default_rng(1)
    steps = 0
    evals = 0
    kicks = 0
    t0 = time.perf_counter()
    mu = None
    for _cycle in range(n_cycles):
        e_before = float(tables.energy(torch.as_tensor(x)))
        chunk = run_cob(
            name,
            x,
            ideal,
            tables,
            curv,
            groups,
            optimizer=optimizer,
            max_iterations=m_steps,
            lr=lr,
            refresh=0,
            clip=clip,
            damping=damping,
        )
        x = np.asarray(chunk["xyz"], dtype=np.float64)
        steps += int(chunk["steps"])
        evals += int(chunk["evals"])
        mu = chunk["mu"]
        e_after = float(chunk["E"])
        if e_after < best_e:
            best_e = e_after
            best_x = x.copy()
        dev_after = restraint_deviations(x, pr)
        # Bonds can be tight while a few torsions are still degrees off. That is
        # the stall the kick is for; a relative energy drop hides it.
        torsion_stuck = dev_after["bond_rms"] < 0.01 and dev_after["dihedral_max"] > 1.0 and (
            e_before - e_after
        ) < max(0.01, 0.05 * max(e_before, 1e-8))
        stalled = torsion_stuck or int(chunk["steps"]) == 0
        if not (perturb and stalled):
            if int(chunk["steps"]) == 0:
                break
            continue
        kicked, e_kick = _local_kick(
            x, tables, curv, groups, pr, damping=damping, eta=eta, n_trials=6, rng=rng
        )
        kicks += 1
        evals += 6
        if e_kick < best_e:
            best_e = e_kick
            best_x = kicked.copy()
        # Restart from the kick when it is not a disaster, so the next M steps
        # can cross a torsion barrier. Otherwise stay at the current point.
        if np.isfinite(e_kick) and e_kick < e_after + max(5.0, 0.25 * max(e_after, 1.0)):
            x = kicked
        elif int(chunk["steps"]) == 0:
            break
    return {
        "name": f"{name} kicks={kicks}",
        "optimizer": optimizer,
        "E": best_e,
        "rmsd": _rmsd(best_x, ideal),
        "rmsd_aligned": _aligned_rmsd(best_x, ideal),
        "xyz": best_x,
        "steps": steps,
        "evals": evals,
        "refreshes": kicks,
        "seconds": time.perf_counter() - t0,
        "hist": [],
        "mu": mu,
        "jitter": None,
    }


def run_baseline(name, start, ideal, pr, **kw) -> dict:
    t0 = time.perf_counter()
    out, st = energy_and_sites(start, pr, **kw)
    return {
        "name": name,
        "optimizer": kw.get("optimizer"),
        "E": st["after"],
        "rmsd": _rmsd(out, ideal),
        "rmsd_aligned": _aligned_rmsd(out, ideal),
        "xyz": out,
        "steps": int(st["n_steps"]),
        "evals": int(st["n_calls"]),
        "refreshes": int(st.get("n_precond_refresh", 0)),
        "seconds": time.perf_counter() - t0,
        "hist": [],
        "mu": None,
        "jitter": None,
        "before": st["before"],
    }


def run_tridiag_newton(
    start: np.ndarray,
    ideal: np.ndarray,
    tables: _RestraintTables,
    curv: RestraintCurvature,
    groups: list[np.ndarray],
    *,
    max_iterations: int,
    damping: float,
) -> dict:
    """Rebuild the block-tridiagonal factor every step and line-search along M^{-1} (-g)."""
    x = np.asarray(start, dtype=np.float64).copy()
    n_calls = 0
    steps = 0
    t0 = time.perf_counter()
    e_cur = float(tables.energy(torch.as_tensor(x)))
    n_calls += 1
    mu = None
    for _ in range(max_iterations):
        xr = torch.as_tensor(x).requires_grad_(True)
        e_t = tables.energy(xr)
        (g,) = torch.autograd.grad(e_t, xr)
        n_calls += 1
        g_np = g.detach().numpy()
        tri = BlockTridiagonalPreconditioner(curv, xr.detach(), groups, damping=damping)
        mu = tri.mu
        p = tri.solve(-g_np.reshape(-1)).reshape(-1, 3)
        slope = float(np.sum(g_np * p))
        if slope >= 0 or not np.isfinite(slope):
            break
        alpha = 1.0
        accepted = False
        e_new = e_cur
        for _ls in range(24):
            trial = x + alpha * p
            e_try = float(tables.energy(torch.as_tensor(trial)))
            n_calls += 1
            if np.isfinite(e_try) and e_try <= e_cur + 1e-4 * alpha * slope:
                x = trial
                e_new = e_try
                accepted = True
                break
            alpha *= 0.5
        steps += 1
        if not accepted:
            break
        e_cur = e_new
        if e_cur < 1e-18 or abs(slope) < 1e-18:
            break
    return {
        "name": "tridiag Newton line search",
        "optimizer": "newton",
        "E": e_cur,
        "rmsd": _rmsd(x, ideal),
        "rmsd_aligned": _aligned_rmsd(x, ideal),
        "xyz": x,
        "steps": steps,
        "evals": n_calls,
        "refreshes": steps,
        "seconds": time.perf_counter() - t0,
        "hist": [],
        "mu": mu,
        "jitter": None,
    }


def newton_reference(start, tables, curv, groups, damping: float) -> tuple[float, float]:
    """One damped block-tridiagonal Newton step, for the COB wiring check."""
    x = torch.as_tensor(start, dtype=torch.float64)
    xr = x.clone().requires_grad_(True)
    e = tables.energy(xr)
    (g,) = torch.autograd.grad(e, xr)
    tri = BlockTridiagonalPreconditioner(curv, x, groups, damping=damping)
    p = tri.solve(-g.detach().numpy().reshape(-1)).reshape(-1, 3)
    x_new = x.numpy() + p
    e_new = float(tables.energy(torch.as_tensor(x_new)))
    # The same step through G^{-T} G^{-1}.
    factor = BlockChol(curv, x, groups, damping=damping)
    # Forward substitution G w = g, then displace(w) = G^{-T} w = M^{-1} g.
    gflat = g.detach().reshape(-1)
    w_blocks = []
    for i, c in enumerate(factor.coord_t):
        rhs = gflat[c]
        if i:
            rhs = rhs - factor.Gsub[i - 1] @ w_blocks[-1]
        w_blocks.append(torch.linalg.solve_triangular(factor.R[i], rhs.unsqueeze(-1), upper=False).squeeze(-1))
    w = gflat.new_zeros(factor.n3)
    for i, c in enumerate(factor.coord_t):
        w = w.index_copy(0, c, w_blocks[i])
    dx = factor.displace(w).detach().numpy().reshape(-1, 3)
    e_cob = float(tables.energy(torch.as_tensor(x.numpy() - dx)))
    return e_new, e_cob


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    n_long = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    lbfgs_iters = 80
    adam_iters = 150
    damping = 1e-3
    ideal, start, pr = chain(n, n_long=n_long)
    groups = residue_groups(n, 5)
    tables = _RestraintTables.from_packed(pr, device="cpu", dtype=torch.float64)
    curv = RestraintCurvature(tables)
    e0 = float(tables.energy(torch.as_tensor(start)))
    print(
        f"n={n}  long-range bonds={n_long}  groups={len(groups)} of 5  "
        f"E0={e0:.6g}  rmsd0={_rmsd(start, ideal):.4f} A"
    )

    e_newton, e_cob = newton_reference(start, tables, curv, groups, damping)
    print(f"one tridiagonal Newton step: E={e_newton:.6g}   same step via COB: E={e_cob:.6g}")

    rows = []
    rows.append(run_baseline("gauss-newton", start, ideal, pr, max_iterations=40, optimizer="gauss_newton"))
    rows.append(run_baseline("lbfgs", start, ideal, pr, max_iterations=lbfgs_iters, optimizer="lbfgs"))
    rows.append(
        run_baseline(
            "lbfgs jacobi once",
            start,
            ideal,
            pr,
            max_iterations=lbfgs_iters,
            optimizer="lbfgs",
            preconditioner="diagonal",
            precond_refresh=0,
        )
    )
    rows.append(
        run_baseline(
            "lbfgs jacobi refresh/20",
            start,
            ideal,
            pr,
            max_iterations=lbfgs_iters,
            optimizer="lbfgs",
            preconditioner="diagonal",
            precond_refresh=20,
        )
    )
    rows.append(
        run_cob(
            "lbfgs tridiag once",
            start,
            ideal,
            tables,
            curv,
            groups,
            optimizer="lbfgs",
            max_iterations=lbfgs_iters,
            lr=1.0,
            refresh=0,
            clip=None,
            damping=damping,
        )
    )
    rows.append(
        run_cob(
            "lbfgs tridiag refresh/20",
            start,
            ideal,
            tables,
            curv,
            groups,
            optimizer="lbfgs",
            max_iterations=lbfgs_iters,
            lr=1.0,
            refresh=20,
            clip=None,
            damping=damping,
        )
    )
    rows.append(
        run_cob(
            "lbfgs tridiag refresh/1",
            start,
            ideal,
            tables,
            curv,
            groups,
            optimizer="lbfgs",
            max_iterations=40,
            lr=1.0,
            refresh=1,
            clip=None,
            damping=damping,
        )
    )
    rows.append(
        run_tridiag_newton(
            start, ideal, tables, curv, groups, max_iterations=120, damping=damping
        )
    )
    for perturb, eta in ((False, 0.5), (True, 0.5), (True, 1.5)):
        tag = "no kick" if not perturb else f"kick η={eta:g}"
        rows.append(
            run_mstep_restarts(
                f"lbfgs M=10 {tag}",
                start,
                ideal,
                tables,
                curv,
                groups,
                pr,
                optimizer="lbfgs",
                m_steps=10,
                n_cycles=16,
                lr=1.0,
                clip=None,
                damping=damping,
                perturb=perturb,
                eta=eta,
            )
        )
    for perturb, eta in ((False, 0.5), (True, 1.5)):
        tag = "no kick" if not perturb else f"kick η={eta:g}"
        rows.append(
            run_mstep_restarts(
                f"adam M=20 {tag}",
                start,
                ideal,
                tables,
                curv,
                groups,
                pr,
                optimizer="adam",
                m_steps=20,
                n_cycles=10,
                lr=1.0,
                clip=None,
                damping=damping,
                perturb=perturb,
                eta=eta,
            )
        )
    for label, lr, clip in (("adam", 1e-2, 1.0),):
        rows.append(
            run_baseline(
                f"{label} lr={lr:g}",
                start,
                ideal,
                pr,
                max_iterations=adam_iters,
                optimizer="adam",
                lr=lr,
            )
        )
        rows.append(
            run_baseline(
                f"{label} jacobi lr={lr:g}",
                start,
                ideal,
                pr,
                max_iterations=adam_iters,
                optimizer="adam",
                lr=lr,
                preconditioner="diagonal",
                precond_refresh=20,
            )
        )
    for label, lr, clip, refresh in (
        ("adam tridiag", 1e-2, 1.0, 0),
        ("adam tridiag", 1e-2, 1.0, 20),
        ("adam tridiag noclipped", 1.0, None, 0),
        ("adam tridiag noclipped", 1.0, None, 20),
    ):
        rows.append(
            run_cob(
                f"{label} lr={lr:g} refresh={refresh or 'once'}",
                start,
                ideal,
                tables,
                curv,
                groups,
                optimizer="adam",
                max_iterations=adam_iters,
                lr=lr,
                refresh=refresh,
                clip=clip,
                damping=damping,
            )
        )

    start_dev = restraint_deviations(start, pr)
    print()
    header = (
        f"{'method':<36} {'steps':>5} {'E':>12} {'rmsd':>7} "
        f"{'bond':>11} {'angle':>11} {'dihedral':>11}"
    )
    print(header)
    print(
        f"{'start':<36} {0:5d} {e0:12.4g} {_aligned_rmsd(start, ideal):7.3f} "
        f"{_pair(start_dev, 'bond'):>11} {_pair(start_dev, 'angle'):>11} {_pair(start_dev, 'dihedral'):>11}"
    )
    for r in rows:
        dev = restraint_deviations(r["xyz"], pr)
        print(
            f"{r['name']:<36} {r['steps']:5d} {r['E']:12.4g} {r['rmsd_aligned']:7.3f} "
            f"{_pair(dev, 'bond'):>11} {_pair(dev, 'angle'):>11} {_pair(dev, 'dihedral'):>11}"
        )
    if n_long:
        print("\nbond/angle/dihedral columns are rms/max. bond is covalent (1.5 A); long-range bond rms:")
        for r in rows:
            dev = restraint_deviations(r["xyz"], pr)
            print(f"  {r['name']:<36} {dev['long_bond_rms']:.4f} A")
    else:
        print("\nbond/angle/dihedral columns are rms/max. angles and dihedrals are in degrees, bonds in A.")
    cob = next(r for r in rows if r["name"].startswith("lbfgs tridiag"))
    print(f"tridiag damping mu={cob['mu']:.6g}  chol residual ||RR^T - S||={cob['jitter']:.3g}")


def _grad_z(factor: BlockChol, anchor: torch.Tensor, tables: _RestraintTables) -> tuple[float, np.ndarray]:
    """Energy and ∂E/∂z at z = 0. In exact whitened coordinates this gradient is the Newton step."""
    z = torch.zeros(factor.n3, dtype=torch.float64, requires_grad=True)
    loss = tables.energy(anchor + factor.displace(z).view(-1, 3))
    (g,) = torch.autograd.grad(loss, z)
    return float(loss.detach()), g.detach().numpy()


def _model_ratio(
    factor: BlockChol,
    anchor: torch.Tensor,
    tables: _RestraintTables,
    g_z: np.ndarray,
    e0: float,
    direction: np.ndarray,
    radius: float,
) -> tuple[float, float, float]:
    """Actual energy drop over the quadratic model's drop for a step of length ``radius``."""
    z_np = radius * direction
    pred = -(float(g_z @ z_np) + 0.5 * radius * radius)
    z = torch.as_tensor(z_np, dtype=torch.float64)
    with torch.no_grad():
        e1 = float(tables.energy(anchor + factor.displace(z).view(-1, 3)))
    ared = e0 - e1
    rho = ared / pred if abs(pred) > 1e-18 else float("nan")
    return ared, pred, rho


def _step_geometry(xyz0: np.ndarray, xyz1: np.ndarray, pr: PackedRestraints) -> tuple[float, float]:
    """Cartesian rms displacement, and the largest rotation of a covalent bond direction."""
    dxyz = np.asarray(xyz1, dtype=np.float64) - np.asarray(xyz0, dtype=np.float64)
    rms = float(np.sqrt(np.mean(np.sum(dxyz * dxyz, axis=1))))
    bi = np.asarray(pr.bond_i_seqs)
    ideal = np.asarray(pr.bond_distance_ideal, dtype=np.float64)
    cov = np.isclose(ideal, 1.5)
    bi = bi[cov]
    a0 = xyz0[bi[:, 0]] - xyz0[bi[:, 1]]
    a1 = xyz1[bi[:, 0]] - xyz1[bi[:, 1]]
    a0 = a0 / np.maximum(np.linalg.norm(a0, axis=1, keepdims=True), 1e-12)
    a1 = a1 / np.maximum(np.linalg.norm(a1, axis=1, keepdims=True), 1e-12)
    cos = np.clip(np.sum(a0 * a1, axis=1), -1.0, 1.0)
    turn = float(np.max(np.degrees(np.arccos(cos))))
    return rms, turn


def _curvature_fd(
    factor: BlockChol,
    anchor: torch.Tensor,
    tables: _RestraintTables,
    direction: np.ndarray,
    e0: float,
    h: float = 1e-3,
) -> float:
    """True second derivative of E along a unit whitened direction. The model says 1."""
    vals = []
    for sign in (-1.0, 1.0):
        z = torch.as_tensor(sign * h * direction, dtype=torch.float64)
        with torch.no_grad():
            vals.append(float(tables.energy(anchor + factor.displace(z).view(-1, 3))))
    return (vals[0] + vals[1] - 2.0 * e0) / (h * h)


def _largest_radius(rows: list[tuple[float, float]], threshold: float) -> float:
    """Largest radius whose model ratio still exceeds ``threshold``, scanning outward."""
    good = 0.0
    for radius, rho in rows:
        if not np.isfinite(rho) or rho < threshold:
            break
        good = radius
    return good


def trust_report(n: int = 300, n_long: int = 5, damping: float = 1e-3) -> None:
    """Trust radius of the block-tridiagonal change of basis along a minimisation."""
    ideal, start, pr = chain(n, n_long=n_long)
    groups = residue_groups(n, 5)
    tables = _RestraintTables.from_packed(pr, device="cpu", dtype=torch.float64)
    curv = RestraintCurvature(tables)
    # Points from the schedule that actually descended: M=10 L-BFGS with a fresh factor.
    snapshots: list[tuple[str, np.ndarray]] = [("start", np.asarray(start, dtype=np.float64))]
    x = np.asarray(start, dtype=np.float64)
    for cycle in range(8):
        chunk = run_cob(
            "trust",
            x,
            ideal,
            tables,
            curv,
            groups,
            optimizer="lbfgs",
            max_iterations=10,
            lr=1.0,
            refresh=0,
            clip=None,
            damping=damping,
        )
        x = np.asarray(chunk["xyz"], dtype=np.float64)
        snapshots.append((f"after {(cycle + 1) * 10} LBFGS", x))

    rng = np.random.default_rng(0)
    print(
        f"{'stage':<22} {'E':>10} {'||g||':>8} {'φ′′':>6} "
        f"{'r75 N':>8} {'r75 rnd':>8} {'A@N':>7} {'deg':>6} {'||z||10':>8} {'ρ(z10)':>8}"
    )
    for label, xyz in snapshots:
        anchor = torch.as_tensor(xyz, dtype=torch.float64)
        factor = BlockChol(curv, anchor, groups, damping=damping)
        e0, g_z = _grad_z(factor, anchor, tables)
        gnorm = float(np.linalg.norm(g_z))
        if gnorm < 1e-14:
            print(f"{label:<22} {e0:10.4g}  gradient vanished")
            continue
        newton = -g_z / gnorm
        fd = _curvature_fd(factor, anchor, tables, newton, e0)
        # Out to twice the model Newton step. The quadratic model's decrease
        # stays positive only for r < 2 ||g_z|| along this direction.
        radii = np.geomspace(1e-3, max(2.0 * gnorm, 1.0), 28)
        series = []
        for radius in radii:
            _ared, _pred, rho = _model_ratio(factor, anchor, tables, g_z, e0, newton, float(radius))
            series.append((float(radius), rho))
        r75 = _largest_radius(series, 0.75)
        rms, turn = 0.0, 0.0
        if r75 > 0:
            z = torch.as_tensor(r75 * newton, dtype=torch.float64)
            with torch.no_grad():
                xyz1 = (anchor + factor.displace(z).view(-1, 3)).numpy()
            rms, turn = _step_geometry(xyz, xyz1, pr)
        random_r = []
        for _ in range(4):
            direction = rng.normal(size=g_z.shape)
            direction /= np.linalg.norm(direction)
            if float(g_z @ direction) > 0:
                direction = -direction
            row = []
            for radius in radii:
                _a, _p, rho = _model_ratio(factor, anchor, tables, g_z, e0, direction, float(radius))
                row.append((float(radius), rho))
            random_r.append(_largest_radius(row, 0.75))
        r75_rnd = float(np.median(random_r))
        # Net whitened displacement of one 10-step L-BFGS block from this point.
        z_var = torch.zeros(factor.n3, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([z_var], lr=1.0, max_iter=10, line_search_fn="strong_wolfe", history_size=10)

        def closure():
            opt.zero_grad(set_to_none=True)
            loss = tables.energy(anchor + factor.displace(z_var).view(-1, 3))
            loss.backward()
            return loss

        opt.step(closure)
        z_net_vec = z_var.detach().numpy()
        z_net = float(np.linalg.norm(z_net_vec))
        if z_net > 1e-12:
            _a, _p, rho_net = _model_ratio(
                factor, anchor, tables, g_z, e0, z_net_vec / z_net, z_net
            )
        else:
            rho_net = float("nan")
        print(
            f"{label:<22} {e0:10.4g} {gnorm:8.3g} {fd:6.2f} "
            f"{r75:8.3g} {r75_rnd:8.3g} {rms:7.4f} {turn:6.2f} {z_net:8.3g} {rho_net:8.2f}"
        )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "trust":
        trust_report(int(sys.argv[2]) if len(sys.argv) > 2 else 300)
    else:
        main()
