"""Second-derivative information for the torch geometry term (preconditioners).

The restraint energy is a weighted sum of squares,

    E(x) = sum_r w_r rho_r(x)^2

with residuals rho_r for bonds (|d| - d0 beyond slack), angles (theta - theta0, degrees)
and dihedrals (sqrt(2) sin(n (phi - phi0) / 2), so that w rho^2 = w (1 - cos n(phi - phi0))).
The Gauss-Newton Hessian

    H_GN = 2 J^T W J,   J_ri = d rho_r / d x_i

is positive semi-definite and, because each residual touches at most four atoms, its
diagonal and its per-atom 3x3 blocks are exact and cheap: evaluate every residual on
*gathered* coordinates P_r (k_r atoms x 3), so autograd of sum_r rho_r with respect to P
returns every per-residual gradient at once (the gathered Jacobian is block diagonal),
then scatter-add 2 w_r g g^T onto the atoms. The full Hessian (adds sum_r 2 w_r rho_r
nabla^2 rho_r, indefinite away from the minimum) is available as a Hessian-vector
product by double backward.

Everything here consumes ``_RestraintTables`` so energies agree with ``energy.py``
to round-off; ``energy_from_residuals`` is asserted equal in the tests.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional

import numpy as np

from phridge.worker.geometry.energy import _RestraintTables

Tensor = Any


# ---------------------------------------------------------------- residual kernels on gathered coordinates
def _bond_rho(P: Tensor, d0: Tensor, slack: Tensor) -> Tensor:
    import torch

    d = torch.linalg.norm(P[:, 0] - P[:, 1], dim=-1)
    dev = d - d0
    mag = torch.clamp(dev.abs() - slack, min=0.0)
    return torch.sign(dev) * mag


def _angle_rho(P: Tensor, theta0: Tensor) -> Tensor:
    import torch

    v1 = P[:, 0] - P[:, 1]
    v2 = P[:, 2] - P[:, 1]
    n1 = torch.linalg.norm(v1, dim=-1).clamp_min(1e-12)
    n2 = torch.linalg.norm(v2, dim=-1).clamp_min(1e-12)
    cos = ((v1 * v2).sum(-1) / (n1 * n2)).clamp(-1.0 + 1e-8, 1.0 - 1e-8)
    return torch.rad2deg(torch.acos(cos)) - theta0


def _dihedral_rho(P: Tensor, phi0_deg: Tensor, period: Tensor) -> Tensor:
    import torch

    b1 = P[:, 1] - P[:, 0]
    b2 = P[:, 2] - P[:, 1]
    b3 = P[:, 3] - P[:, 2]
    n1 = torch.cross(b1, b2, dim=-1)
    n2 = torch.cross(b2, b3, dim=-1)
    n1u = n1 / torch.linalg.norm(n1, dim=-1).clamp_min(1e-12).unsqueeze(-1)
    n2u = n2 / torch.linalg.norm(n2, dim=-1).clamp_min(1e-12).unsqueeze(-1)
    b2u = b2 / torch.linalg.norm(b2, dim=-1).clamp_min(1e-12).unsqueeze(-1)
    m1 = torch.cross(n1u, b2u, dim=-1)
    phi = torch.atan2((m1 * n2u).sum(-1), (n1u * n2u).sum(-1))
    n = period.to(dtype=phi.dtype).clamp_min(1.0)
    return math.sqrt(2.0) * torch.sin(0.5 * n * (phi - torch.deg2rad(phi0_deg)))


class RestraintCurvature:
    """Gauss-Newton / full second derivatives of the packed restraint energy."""

    def __init__(self, tables: _RestraintTables) -> None:
        self.t = tables

    # -- residual groups: (index (R,k), rho(P) -> (R,), weights (R,))
    def _groups(self) -> list[tuple[Tensor, Callable[[Tensor], Tensor], Tensor]]:
        t = self.t
        groups: list[tuple[Tensor, Callable[[Tensor], Tensor], Tensor]] = []
        if t.bond_i.numel():
            groups.append((t.bond_i, lambda P: _bond_rho(P, t.bond_d0, t.bond_slack), t.bond_w))
        if t.angle_i.numel():
            groups.append((t.angle_i, lambda P: _angle_rho(P, t.angle_d0), t.angle_w))
        if t.dih_i.numel():
            groups.append((t.dih_i, lambda P: _dihedral_rho(P, t.dih_d0, t.dih_period), t.dih_w))
        return groups

    def residuals(self, xyz: Tensor) -> tuple[Tensor, Tensor]:
        """Concatenated (rho, w) with E = sum w rho^2 (differentiable in xyz)."""
        import torch

        rhos, ws = [], []
        for idx, fn, w in self._groups():
            rhos.append(fn(xyz[idx]))
            ws.append(w)
        if not rhos:
            z = xyz.new_zeros((0,))
            return z, z
        return torch.cat(rhos), torch.cat(ws)

    def energy_from_residuals(self, xyz: Tensor) -> Tensor:
        rho, w = self.residuals(xyz)
        return (w * rho * rho).sum()

    # -- exact Gauss-Newton diagonal and per-atom blocks
    def gn_blocks(self, xyz: Tensor) -> Tensor:
        """Per-atom 3x3 blocks of 2 J^T W J, shape (N, 3, 3)."""
        import torch

        n = xyz.shape[0]
        blocks = xyz.new_zeros((n, 3, 3))
        x = xyz.detach()
        for idx, fn, w in self._groups():
            P = x[idx].clone().requires_grad_(True)  # (R, k, 3)
            rho = fn(P)
            (g,) = torch.autograd.grad(rho.sum(), P)  # exact per-residual gradients: block-diagonal Jacobian
            gw = g * (2.0 * w)[:, None, None]
            outer = torch.einsum("rka,rkb->rkab", gw, g)  # (R, k, 3, 3)
            blocks.index_add_(0, idx.reshape(-1), outer.reshape(-1, 3, 3))
        return blocks

    def gn_diagonal(self, xyz: Tensor) -> Tensor:
        """diag(2 J^T W J) as (N, 3)."""
        import torch

        return torch.diagonal(self.gn_blocks(xyz), dim1=1, dim2=2).contiguous()

    # -- Hessian-vector products
    def gn_hvp(self, xyz: Tensor, v: Tensor) -> Tensor:
        """(2 J^T W J) v by double vjp (no functorch)."""
        import torch

        x = xyz.detach().clone().requires_grad_(True)
        rho, w = self.residuals(x)
        if rho.numel() == 0:
            return torch.zeros_like(xyz)
        u = torch.zeros_like(rho, requires_grad=True)
        (vjp,) = torch.autograd.grad(rho, x, grad_outputs=u, create_graph=True)  # J^T u (linear in u)
        (Jv,) = torch.autograd.grad((vjp * v).sum(), u, create_graph=True)  # d/du <J^T u, v> = J v
        (hv,) = torch.autograd.grad(rho, x, grad_outputs=2.0 * w * Jv)
        return hv

    def full_hvp(self, xyz: Tensor, v: Tensor) -> Tensor:
        """Exact Hessian-vector product of E (includes the residual term)."""
        import torch

        x = xyz.detach().clone().requires_grad_(True)
        e = self.energy_from_residuals(x)
        (g,) = torch.autograd.grad(e, x, create_graph=True)
        (hv,) = torch.autograd.grad((g * v).sum(), x)
        return hv

    def hvp(self, xyz: Tensor, v: Tensor, hessian: str = "gn") -> Tensor:
        if hessian == "gn":
            return self.gn_hvp(xyz, v)
        if hessian == "full":
            return self.full_hvp(xyz, v)
        raise ValueError("hessian must be 'gn' or 'full'")


# ---------------------------------------------------------------- preconditioner helpers
def jacobi_scale(diag: Tensor, floor_fraction: float = 1e-3) -> Tensor:
    """sqrt of a floored GN diagonal: coordinates y = s * x have unit GN curvature.

    Entries below ``floor_fraction`` x median(positive) (unrestrained atoms, flat bond
    slack) are floored so the change of variables stays invertible.
    """
    import torch

    pos = diag[diag > 0]
    ref = pos.median() if pos.numel() else torch.tensor(1.0, dtype=diag.dtype, device=diag.device)
    return torch.sqrt(torch.clamp(diag, min=float(floor_fraction) * ref))


def curvature_summary(diag: Tensor) -> dict[str, float]:
    pos = diag[diag > 0]
    if pos.numel() == 0:
        return {"diag_min": 0.0, "diag_median": 0.0, "diag_max": 0.0, "condition_estimate": 1.0}
    return {
        "diag_min": float(pos.min()),
        "diag_median": float(pos.median()),
        "diag_max": float(pos.max()),
        "condition_estimate": float(pos.max() / pos.min()),
    }


def as_numpy(t: Optional[Tensor]) -> Optional[np.ndarray]:
    if t is None:
        return None
    return t.detach().cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------- sparse Gauss-Newton matrix
def gn_sparse_coo(curv: "RestraintCurvature", xyz: Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """COO triplets (rows, cols, vals, n3) of 2 J^T W J over cartesian coordinates (3N x 3N).

    Duplicate (row, col) entries are summed by the consumer (scipy does this on
    construction). Each residual contributes a dense (3k x 3k) block, k <= 4.
    """
    import torch

    x = xyz.detach()
    rows, cols, vals = [], [], []
    ar3 = torch.arange(3, device=x.device)
    for idx, fn, w in curv._groups():
        P = x[idx].clone().requires_grad_(True)
        rho = fn(P)
        (g,) = torch.autograd.grad(rho.sum(), P)  # (R, k, 3)
        R, k, _ = g.shape
        gw = (2.0 * w)[:, None, None] * g
        blk = torch.einsum("rka,rlb->rkalb", gw, g)  # (R, k, 3, k, 3)
        ai = 3 * idx[:, :, None] + ar3[None, None, :]  # (R, k, 3) coordinate indices
        rr = ai[:, :, :, None, None].expand(R, k, 3, k, 3)
        cc = ai[:, None, None, :, :].expand(R, k, 3, k, 3)
        rows.append(rr.reshape(-1).cpu().numpy())
        cols.append(cc.reshape(-1).cpu().numpy())
        vals.append(blk.reshape(-1).cpu().numpy().astype(np.float64))
    n3 = 3 * int(x.shape[0])
    if not rows:
        e = np.zeros(0, dtype=np.int64)
        return e, e, np.zeros(0), n3
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals), n3


class GaussNewtonPreconditioner:
    """Factorised M = 2 J^T W J + diag(extra) + mu * ref * I over parameters.

    ``solve(r)`` applies M^{-1}. With ``extra`` = the x-ray GN diagonal (per coordinate)
    this is the preconditioner for conjugate gradients on the joint x-ray + geometry
    Gauss-Newton system; with ``extra`` = 0 it is one Levenberg-Marquardt step of the
    geometry term. Uses scipy's sparse LU (SuperLU); ``sksparse.cholmod`` when importable.
    """

    def __init__(
        self,
        curv: Any,
        xyz: Optional[Tensor] = None,
        *,
        extra_diag: Optional[np.ndarray] = None,
        damping: float = 1e-3,
        weight: float = 1.0,
        coo: Optional[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = None,
    ) -> None:
        import scipy.sparse as sp

        if coo is not None:
            rows, cols, vals, n3 = coo
        elif hasattr(curv, "gn_sparse_coo"):
            rows, cols, vals, n3 = curv.gn_sparse_coo(xyz)
        else:
            rows, cols, vals, n3 = gn_sparse_coo(curv, xyz)

        H = sp.csr_matrix((float(weight) * vals, (rows, cols)), shape=(n3, n3))
        d = H.diagonal()
        if extra_diag is not None:
            ex = np.asarray(extra_diag, dtype=np.float64).reshape(-1)
            if ex.shape[0] != n3:
                raise ValueError(f"extra_diag must have {n3} entries, got {ex.shape[0]}")
            H = H + sp.diags(ex)
            d = d + ex
        pos = d[d > 0]
        self.ref = float(np.median(pos)) if pos.size else 1.0
        self.n3 = n3
        self.nnz = int(H.nnz)
        self.mu = float(damping) * self.ref
        self._H = H
        self._lu: Any = None
        self._chol: Any = None
        self._factor = self._factorise(H + sp.diags(np.full(n3, self.mu)))

    def _factorise(self, A: Any) -> Callable[[np.ndarray], np.ndarray]:
        try:  # pragma: no cover - optional dependency
            from sksparse.cholmod import cholesky

            f = cholesky(A.tocsc())
            self._chol = f
            return lambda b: np.asarray(f(b))
        except Exception:
            import scipy.sparse.linalg as spl

            lu = spl.splu(A.tocsc())
            self._lu = lu
            return lambda b: lu.solve(b)

    def logdet(self) -> float:
        """Log determinant log det(M) via the sparse Cholesky / LU factorisation."""
        if self._chol is not None:
            return float(self._chol.logdet())
        if self._lu is not None:
            diag_U = self._lu.U.diagonal()
            return float(np.sum(np.log(np.abs(diag_U))))
        raise RuntimeError("No factorisation available to compute logdet")

    def matvec(self, v: np.ndarray) -> np.ndarray:
        """(M - mu I) v, i.e. the undamped assembled matrix."""
        return self._H @ np.asarray(v, dtype=np.float64).reshape(-1)

    def solve(self, r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=np.float64)
        return self._factor(r.reshape(-1)).reshape(r.shape)


# ---------------------------------------------------------------- block-tridiagonal (residue) preconditioner
class BlockTridiagonalPreconditioner:
    """Residue-block-tridiagonal approximation of M = w 2 J^T W J + diag(extra) + mu I.

    ``groups`` lists atom indices per block in chain order (e.g. residues). The diagonal
    blocks and the off-diagonal blocks between *consecutive* groups are kept, everything
    else is dropped; the result is factorised by block LDL^T (Thomas algorithm), O(n).
    For a linear chain whose restraints only ever bridge consecutive groups (peptide bond,
    angles, omega, planarity across one boundary) this is exact; disulfides, H-bond /
    base-pair restraints, NCS and non-bonded contacts are the dropped terms.
    """

    def __init__(
        self,
        curv: Any,
        xyz: Optional[Tensor] = None,
        groups: Optional[list[np.ndarray]] = None,
        *,
        extra_diag: Optional[np.ndarray] = None,
        damping: float = 1e-3,
        weight: float = 1.0,
        param_dims: Optional[Any] = None,
        coo: Optional[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = None,
    ) -> None:
        import scipy.sparse as sp

        if groups is None:
            raise ValueError("groups (list of atom-index lists in chain order) is required")

        if coo is not None:
            rows, cols, vals, n3 = coo
        elif hasattr(curv, "gn_sparse_coo"):
            rows, cols, vals, n3 = curv.gn_sparse_coo(xyz)
        else:
            rows, cols, vals, n3 = gn_sparse_coo(curv, xyz)

        H = sp.csr_matrix((float(weight) * vals, (rows, cols)), shape=(n3, n3))
        d = H.diagonal()
        if extra_diag is not None:
            ex = np.asarray(extra_diag, dtype=np.float64).reshape(-1)
            H = H + sp.diags(ex)
            d = d + ex
        pos = d[d > 0]
        self.ref = float(np.median(pos)) if pos.size else 1.0
        self.mu = float(damping) * self.ref
        self.n3 = n3
        self.groups = [np.asarray(g, dtype=np.int64).reshape(-1) for g in groups]

        if param_dims is None:
            self.coord = [(3 * g[:, None] + np.arange(3)).reshape(-1) for g in self.groups]
        elif isinstance(param_dims, (int, np.integer)):
            d_param = int(param_dims)
            self.coord = [(d_param * g[:, None] + np.arange(d_param)).reshape(-1) for g in self.groups]
        else:
            p_dims = np.asarray(param_dims, dtype=np.int64).reshape(-1)
            offsets = np.zeros(len(p_dims) + 1, dtype=np.int64)
            np.cumsum(p_dims, out=offsets[1:])
            self.coord = [
                np.concatenate([np.arange(offsets[int(idx)], offsets[int(idx) + 1]) for idx in g]) if g.size > 0 else np.zeros(0, dtype=np.int64)
                for g in self.groups
            ]

        covered = np.concatenate(self.coord) if self.coord else np.zeros(0, dtype=np.int64)
        if covered.size != n3 or np.unique(covered).size != n3:
            raise ValueError(f"groups must partition all {n3} parameters exactly once, got {covered.size} (unique {np.unique(covered).size})")
        n_b = len(self.coord)
        D = [H[c][:, c].toarray() + self.mu * np.eye(c.size) for c in self.coord]
        E = [H[self.coord[i + 1]][:, self.coord[i]].toarray() for i in range(n_b - 1)]  # E_i = block(i+1, i)
        # block LDL^T: S_0 = D_0, S_i = D_i - E_{i-1} S_{i-1}^{-1} E_{i-1}^T
        self._S_inv: list[np.ndarray] = []
        self._E = E
        S_prev_inv = None
        for i in range(n_b):
            S = D[i] if i == 0 else D[i] - E[i - 1].dot(S_prev_inv).dot(E[i - 1].T)
            S_prev_inv = np.linalg.inv(S)
            self._S_inv.append(S_prev_inv)
        self.nnz_kept = int(sum(b.size for b in D) + 2 * sum(b.size for b in E))
        self.nnz_total = int(H.nnz)
        self.nnz_kept = int(sum(b.size for b in D) + 2 * sum(b.size for b in E))
        self.nnz_total = int(H.nnz)

    def solve(self, r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=np.float64)
        shape = r.shape
        b = r.reshape(-1)
        n_b = len(self.coord)
        y = [None] * n_b
        # forward: y_i = S_i^{-1} (b_i - E_{i-1} y_{i-1})
        for i in range(n_b):
            rhs = b[self.coord[i]] if i == 0 else b[self.coord[i]] - self._E[i - 1].dot(y[i - 1])
            y[i] = self._S_inv[i].dot(rhs)
        # backward: x_i = y_i - S_i^{-1} E_i^T x_{i+1}
        x = np.zeros_like(b)
        x_next = None
        for i in range(n_b - 1, -1, -1):
            xi = y[i] if i == n_b - 1 else y[i] - self._S_inv[i].dot(self._E[i].T.dot(x_next))
            x[self.coord[i]] = xi
            x_next = xi
        return x.reshape(shape)
