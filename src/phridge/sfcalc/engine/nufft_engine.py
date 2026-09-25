"""Differentiable NUFFT structure-factor engine.

Groups atoms by scattering type and U-shell, Taylor-expands residual ADPs,
and evaluates each scalar-weight sum as a type-1 NUFFT. See docs/nufft_engine.md.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from typing import Any, Optional, Sequence

import numpy as np

from phridge.sfcalc.engine.cell import (
    orthogonalization_matrix,
    reciprocal_cartesian,
    sym6_to_mat,
    u_star_to_cart,
)
from phridge.sfcalc.engine.engine import ScatteringModel, _fft_friendly
from phridge.sfcalc.engine.nufft_backend import nufft_available, nufft_type1

TWO_PI2 = 2.0 * math.pi * math.pi
log = logging.getLogger(__name__)


@dataclass
class NufftEngineParams:
    d_min: float
    tau: float = 1e-4
    n_max: int = 2
    eps: float = 1e-6
    dtype: str = "float64"
    symmetry: str = "expand"
    t_chunk: int = 16


@dataclass
class GroupSpec:
    """One (scattering type, U-shell) group of ASU atoms."""

    atom_indices: np.ndarray
    U_g: float
    type_index: int
    order: int
    needs_anom: bool
    has_aniso: bool
    lambda_max: float


@dataclass
class TransformRow:
    """One NUFFT: scalar weights on a subset of expanded atoms, times ``multiplier(h)``."""

    group_index: int
    asu_indices: np.ndarray
    kind: str  # "ff" | "anom"
    order_n: int
    multiplier: Any  # torch tensor (N_refl,)
    iso_power: Optional[int] = None
    voigt_k: Optional[int] = None
    voigt_kl: Optional[tuple[int, int]] = None
    expanded_index: Any = None  # torch int64 (M_group,)


@dataclass
class GroupPlan:
    """Cached partition of ASU atoms into NUFFT groups.

    Built from the model at construction and reused across ``f_calc`` calls.
    ``U_g`` is a constant; gradients flow through ``ΔU_j``, not through ``U_g``.
    Call :meth:`NufftStructureFactorEngine.replan` after large ADP changes.
    """

    groups: list[GroupSpec]
    tau: float
    n_max: int
    d_min: float
    n_transforms: int
    predicted_rel_error: float

    @classmethod
    def from_model(
        cls,
        model: ScatteringModel,
        d_min: float,
        tau: float = 1e-4,
        n_max: int = 2,
        fp: Optional[np.ndarray] = None,
        fdp: Optional[np.ndarray] = None,
    ) -> GroupPlan:
        n = model.n_scatterers
        fp_arr = np.asarray(model.fp if fp is None else fp, dtype=np.float64).reshape(n)
        fdp_arr = np.asarray(model.fdp if fdp is None else fdp, dtype=np.float64).reshape(n)
        u_iso = np.asarray(model.u_iso, dtype=np.float64).reshape(n)
        aniso = np.asarray(model.anisotropic, dtype=bool).reshape(n)
        type_index = np.asarray(model.type_index, dtype=np.int64).reshape(n)
        u_cart = np.zeros((n, 3, 3), dtype=np.float64)
        if np.any(aniso):
            u_cart[aniso] = u_star_to_cart(model.unit_cell, np.asarray(model.u_star, dtype=np.float64)[aniso])
        u_eq = np.where(aniso, np.trace(u_cart, axis1=1, axis2=2) / 3.0, u_iso)
        s_max = 1.0 / float(d_min)
        groups: list[GroupSpec] = []
        worst = 0.0
        for t in np.unique(type_index):
            members = np.where(type_index == t)[0]
            order = members[np.argsort(u_eq[members], kind="stable")]
            start = 0
            while start < order.size:
                end = start + 1
                while end <= order.size:
                    cand = order[start:end]
                    ok, U_g, lam, n_use = _shell_stats(cand, u_eq, u_cart, aniso, n_max, tau, s_max)
                    if not ok:
                        end -= 1
                        break
                    end += 1
                if end <= start:
                    end = start + 1
                shell = order[start:end]
                ok, U_g, lam, n_use = _shell_stats(shell, u_eq, u_cart, aniso, n_max, tau, s_max)
                has_aniso = bool(np.any(aniso[shell]))
                needs_anom = bool(np.any(np.abs(fp_arr[shell]) > 0) or np.any(np.abs(fdp_arr[shell]) > 0))
                groups.append(
                    GroupSpec(
                        atom_indices=np.asarray(shell, dtype=np.int64),
                        U_g=float(U_g),
                        type_index=int(t),
                        order=int(n_use),
                        needs_anom=needs_anom,
                        has_aniso=has_aniso,
                        lambda_max=float(lam),
                    )
                )
                worst = max(worst, _truncation_bound(lam, n_use, s_max))
                start = end
        plan = cls(
            groups=groups,
            tau=float(tau),
            n_max=int(n_max),
            d_min=float(d_min),
            n_transforms=_count_transforms(groups),
            predicted_rel_error=float(worst),
        )
        log.info(
            "NUFFT GroupPlan: %d groups, %d transforms, predicted worst-case rel err %.3e (tau=%.3e, n_max=%d, d_min=%.3f)",
            len(groups),
            plan.n_transforms,
            plan.predicted_rel_error,
            plan.tau,
            plan.n_max,
            plan.d_min,
        )
        return plan

    def with_order(self, n: int) -> GroupPlan:
        """Same shells, Taylor order ``n`` (aniso groups still cap at 2)."""
        groups = []
        s_max = 1.0 / self.d_min
        worst = 0.0
        for g in self.groups:
            n_use = min(int(n), 2) if g.has_aniso else int(n)
            groups.append(replace(g, order=n_use))
            worst = max(worst, _truncation_bound(g.lambda_max, n_use, s_max))
        return GroupPlan(
            groups=groups,
            tau=self.tau,
            n_max=int(n),
            d_min=self.d_min,
            n_transforms=_count_transforms(groups),
            predicted_rel_error=float(worst),
        )


def _allowed_delta(n: int, tau: float, s_max: float) -> float:
    return (math.factorial(n + 1) * tau) ** (1.0 / (n + 1)) / (TWO_PI2 * s_max * s_max)


def _truncation_bound(lambda_max: float, n: int, s_max: float) -> float:
    x = TWO_PI2 * s_max * s_max * float(lambda_max)
    return float(x ** (n + 1) / math.factorial(n + 1))


def _shell_stats(
    members: np.ndarray,
    u_eq: np.ndarray,
    u_cart: np.ndarray,
    aniso: np.ndarray,
    n_max: int,
    tau: float,
    s_max: float,
) -> tuple[bool, float, float, int]:
    U_g = float(np.median(u_eq[members]))
    has_aniso = bool(np.any(aniso[members]))
    n_use = min(int(n_max), 2) if has_aniso else int(n_max)
    lam = float(np.max(np.abs(u_eq[members] - U_g))) if members.size else 0.0
    if has_aniso:
        for j in members[aniso[members]]:
            evals = np.linalg.eigvalsh(u_cart[j] - U_g * np.eye(3))
            lam = max(lam, float(np.max(np.abs(evals))))
    delta = _allowed_delta(n_use, tau, s_max)
    return lam <= delta + 1e-15, U_g, lam, n_use


def _count_transforms(groups: Sequence[GroupSpec]) -> int:
    t = 0
    for g in groups:
        if g.has_aniso:
            n_terms = 1
            if g.order >= 1:
                n_terms += 6
            if g.order >= 2:
                n_terms += 21
        else:
            n_terms = g.order + 1
        t += n_terms * (2 if g.needs_anom else 1)
    return t


def nufft_mode_shape(hkl: np.ndarray) -> tuple[int, int, int]:
    """Smallest FFT-friendly ``n`` with ``n ≥ 2 max|h| + 1`` on each axis."""
    h = np.asarray(hkl, dtype=np.int64).reshape(-1, 3)
    return tuple(_fft_friendly(int(2 * int(np.max(np.abs(h[:, i]))) + 1)) for i in range(3))


class NufftStructureFactorEngine:
    """Torch NUFFT engine bound to one model layout and one hkl list."""

    def __init__(
        self,
        model: ScatteringModel,
        hkl: np.ndarray,
        params: NufftEngineParams,
        device: str = "cpu",
        plan: Optional[GroupPlan] = None,
    ) -> None:
        import torch

        if params.symmetry != "expand":
            raise NotImplementedError(
                "NufftStructureFactorEngine phase-2 ASU symmetry is not enabled; "
                "use symmetry='expand'"
            )
        if not nufft_available():
            raise RuntimeError(
                "pytorch-finufft is required for NufftStructureFactorEngine; "
                "install with pip install 'phridge[nufft]'"
            )
        self.torch = torch
        self.model = model
        self.params = params
        self.device = device
        if device.startswith("mps"):
            self.dtype = torch.float32
            self.cdtype = torch.complex64
        else:
            self.dtype = torch.float64 if params.dtype == "float64" else torch.float32
            self.cdtype = torch.complex128 if params.dtype == "float64" else torch.complex64
        self.hkl = np.asarray(hkl, dtype=np.int64).reshape(-1, 3)
        self.n_modes = nufft_mode_shape(self.hkl)
        n = np.array(self.n_modes, dtype=np.int64)
        if np.any(np.abs(self.hkl) * 2 >= n[None, :]):
            raise ValueError("hkl outside NUFFT mode grid")
        self.gather_h = torch.as_tensor(np.mod(self.hkl[:, 0], n[0]), dtype=torch.int64, device=device)
        self.gather_k = torch.as_tensor(np.mod(self.hkl[:, 1], n[1]), dtype=torch.int64, device=device)
        self.gather_l = torch.as_tensor(np.mod(self.hkl[:, 2], n[2]), dtype=torch.int64, device=device)

        cell = model.unit_cell
        o = orthogonalization_matrix(cell)
        self.o_mat = torch.as_tensor(o, dtype=self.dtype, device=device)
        dstar = reciprocal_cartesian(cell, self.hkl)
        self._dstar = torch.as_tensor(dstar, dtype=self.dtype, device=device)
        self._dstar2 = (self._dstar**2).sum(-1)
        stol2 = (self._dstar2 / 4.0).detach().cpu().numpy()
        self._ff_type = torch.as_tensor(
            _form_factors(model.gauss_a, model.gauss_b, model.gauss_c, stol2),
            dtype=self.dtype,
            device=device,
        )
        self.rot = torch.as_tensor(model.rot, dtype=self.dtype, device=device)
        self.trans = torch.as_tensor(model.trans, dtype=self.dtype, device=device)
        self.weight_wo_occ = torch.as_tensor(
            model.multiplicity / float(model.n_sym), dtype=self.dtype, device=device
        )
        self.aniso = torch.as_tensor(model.anisotropic.astype(bool), device=device)
        self.plan = plan if plan is not None else GroupPlan.from_model(
            model, params.d_min, tau=params.tau, n_max=params.n_max
        )
        self.rows: list[TransformRow] = []
        self._build_rows()

    def _build_rows(self) -> None:
        torch = self.torch
        dstar = self._dstar
        dstar2 = self._dstar2
        m_voigt = torch.stack(
            [
                dstar[:, 0] ** 2,
                dstar[:, 1] ** 2,
                dstar[:, 2] ** 2,
                2.0 * dstar[:, 0] * dstar[:, 1],
                2.0 * dstar[:, 0] * dstar[:, 2],
                2.0 * dstar[:, 1] * dstar[:, 2],
            ],
            dim=1,
        )
        n_sym = int(self.rot.shape[0])
        rows: list[TransformRow] = []
        for gi, g in enumerate(self.plan.groups):
            dw = torch.exp(-TWO_PI2 * float(g.U_g) * dstar2)
            ft = self._ff_type[int(g.type_index)]
            base_ff = (ft * dw).to(self.cdtype)
            base_anom = dw.to(self.cdtype)
            asu = np.asarray(g.atom_indices, dtype=np.int64)
            exp_idx = (asu[:, None] * n_sym + np.arange(n_sym, dtype=np.int64)[None, :]).reshape(-1)
            exp_t = torch.as_tensor(exp_idx, dtype=torch.int64, device=self.device)

            def _add(kind: str, mult, order_n: int, iso_power=None, voigt_k=None, voigt_kl=None, _gi=gi, _asu=asu, _exp=exp_t) -> None:
                rows.append(
                    TransformRow(
                        group_index=_gi,
                        asu_indices=_asu,
                        kind=kind,
                        order_n=order_n,
                        multiplier=mult,
                        iso_power=iso_power,
                        voigt_k=voigt_k,
                        voigt_kl=voigt_kl,
                        expanded_index=_exp,
                    )
                )

            kinds = (("ff", base_ff),)
            if g.needs_anom:
                kinds = (("ff", base_ff), ("anom", base_anom))
            if g.has_aniso:
                for kind, base in kinds:
                    _add(kind, base, 0)
                if g.order >= 1:
                    for k in range(6):
                        mk = (-TWO_PI2) * m_voigt[:, k]
                        for kind, base in kinds:
                            _add(kind, base * mk.to(self.cdtype), 1, voigt_k=k)
                if g.order >= 2:
                    for k in range(6):
                        for ell in range(k, 6):
                            c = 2.0 if k != ell else 1.0
                            mk = (TWO_PI2**2 / 2.0) * c * m_voigt[:, k] * m_voigt[:, ell]
                            for kind, base in kinds:
                                _add(kind, base * mk.to(self.cdtype), 2, voigt_kl=(k, ell))
            else:
                for n in range(g.order + 1):
                    coeff = ((-TWO_PI2) ** n) / float(math.factorial(n))
                    mk = coeff * (dstar2**n)
                    for kind, base in kinds:
                        _add(kind, base * mk.to(self.cdtype), n, iso_power=n)
        self.rows = rows

    def apply_plan(self, plan: GroupPlan) -> None:
        self.plan = plan
        self._build_rows()

    def replan(self, u_iso: Any = None, u_star: Any = None, fp: Any = None, fdp: Any = None) -> GroupPlan:
        """Rebuild the group plan from current (or supplied) ADPs."""
        m = self.model
        if u_iso is not None:
            m = replace(m, u_iso=np.asarray(u_iso, dtype=np.float64).reshape(-1))
        if u_star is not None:
            m = replace(m, u_star=np.asarray(u_star, dtype=np.float64).reshape(-1, 6))
        fp_np = None if fp is None else np.asarray(fp, dtype=np.float64)
        fdp_np = None if fdp is None else np.asarray(fdp, dtype=np.float64)
        self.plan = GroupPlan.from_model(m, self.params.d_min, tau=self.params.tau, n_max=self.params.n_max, fp=fp_np, fdp=fdp_np)
        self._build_rows()
        return self.plan

    def expand(self, sites_frac, u_star):
        """Symmetry-expand sites (N,3)->(N*S,3) and u_star (N,6)->(N*S,3,3)."""
        torch = self.torch
        x = torch.einsum("sij,nj->nsi", self.rot, sites_frac) + self.trans[None]
        u = sym6_to_mat(u_star)
        u_exp = torch.einsum("sij,njk,slk->nsil", self.rot, u, self.rot)
        return x.reshape(-1, 3), u_exp.reshape(-1, 3, 3)

    def _expanded_u_and_weights(self, sites_frac, occupancy, u_iso, u_star, fp, fdp):
        torch = self.torch
        n_sym = int(self.rot.shape[0])
        x_exp, ustar_exp = self.expand(sites_frac, u_star)
        rep = lambda t: t.repeat_interleave(n_sym, dim=0)  # noqa: E731
        w = rep(occupancy * self.weight_wo_occ)
        fp_e = rep(fp)
        fdp_e = rep(fdp)
        u_iso_e = rep(u_iso)
        aniso_e = self.aniso.repeat_interleave(n_sym)
        u_cart_aniso = self.o_mat @ ustar_exp @ self.o_mat.T
        eye = torch.eye(3, dtype=self.dtype, device=self.device)
        u_cart = torch.where(aniso_e[:, None, None], u_cart_aniso, u_iso_e[:, None, None] * eye)
        u_g = torch.zeros(sites_frac.shape[0], dtype=self.dtype, device=self.device)
        for g in self.plan.groups:
            u_g[torch.as_tensor(g.atom_indices, dtype=torch.int64, device=self.device)] = float(g.U_g)
        u_g_e = u_g.repeat_interleave(n_sym)
        delta = u_cart - u_g_e[:, None, None] * eye
        voigt = torch.stack(
            [delta[:, 0, 0], delta[:, 1, 1], delta[:, 2, 2], delta[:, 0, 1], delta[:, 0, 2], delta[:, 1, 2]],
            dim=1,
        )
        delta_iso = (delta[:, 0, 0] + delta[:, 1, 1] + delta[:, 2, 2]) / 3.0
        return x_exp, w, fp_e, fdp_e, voigt, delta_iso

    def _assemble_weights(self, rows: Sequence[TransformRow], w, fp, fdp, voigt, delta_iso):
        torch = self.torch
        t = len(rows)
        m = int(w.shape[0])
        W = torch.zeros(t, m, dtype=self.cdtype, device=self.device)
        for i, row in enumerate(rows):
            idx = row.expanded_index
            ww = w[idx].to(self.cdtype)
            if row.kind == "anom":
                ww = ww * torch.complex(fp[idx].to(self.dtype), fdp[idx].to(self.dtype))
            if row.iso_power is not None and row.iso_power > 0:
                ww = ww * (delta_iso[idx] ** row.iso_power).to(self.cdtype)
            elif row.voigt_k is not None:
                ww = ww * voigt[idx, row.voigt_k].to(self.cdtype)
            elif row.voigt_kl is not None:
                k, ell = row.voigt_kl
                ww = ww * (voigt[idx, k] * voigt[idx, ell]).to(self.cdtype)
            W[i].index_put_((idx,), ww)
        return W

    def f_calc(self, sites_frac, occupancy, u_iso, u_star, fp, fdp, group_index: Optional[int] = None):
        """Structure factors at self.hkl, complex tensor (N_refl,)."""
        torch = self.torch
        x_exp, w, fp_e, fdp_e, voigt, delta_iso = self._expanded_u_and_weights(
            sites_frac, occupancy, u_iso, u_star, fp, fdp
        )
        if group_index is None:
            rows = self.rows
        else:
            rows = [r for r in self.rows if r.group_index == int(group_index)]
        n_refl = int(self.hkl.shape[0])
        acc = torch.zeros(n_refl, dtype=self.cdtype, device=self.device)
        if not rows:
            return acc
        chunk = max(1, int(self.params.t_chunk))
        for start in range(0, len(rows), chunk):
            sl = rows[start : start + chunk]
            W = self._assemble_weights(sl, w, fp_e, fdp_e, voigt, delta_iso)
            grid = nufft_type1(x_exp, W, self.n_modes, self.params.eps)
            gathered = grid[:, self.gather_h, self.gather_k, self.gather_l]
            mult = torch.stack([r.multiplier for r in sl], dim=0)
            acc = acc + (gathered * mult).sum(0)
            del grid
        return acc

    def tensors(self, requires_grad: bool = False):
        """Model parameters as torch tensors (sites_frac, occupancy, u_iso, u_star, fp, fdp)."""
        torch = self.torch
        m = self.model
        out = []
        for arr in (m.sites_frac, m.occupancy, m.u_iso, m.u_star, m.fp, m.fdp):
            t = torch.as_tensor(np.asarray(arr, dtype=np.float64), dtype=self.dtype, device=self.device).clone()
            t.requires_grad_(requires_grad)
            out.append(t)
        return tuple(out)

    def f_calc_numpy(self) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            return self.f_calc(*self.tensors()).cpu().numpy().astype(np.complex128)

    def gradients(self, d_target_d_f_calc: np.ndarray, params=None):
        """dQ/d(params) for Q with given per-reflection complex gradient.

        Convention (cctbx d_target_d_f_calc): G_h = dQ/dA_h + i dQ/dB_h, so
        dQ/dp = sum_h Re[conj(G_h) dF_h/dp].
        ``u_iso`` / ``u_star`` derivatives are of the truncated (current-plan) model.
        """
        torch = self.torch
        params = self.tensors(requires_grad=True) if params is None else params
        g = torch.as_tensor(np.asarray(d_target_d_f_calc, dtype=np.complex128), dtype=self.cdtype, device=self.device)
        f = self.f_calc(*params)
        q = (f * g.conj()).real.sum()
        grads = torch.autograd.grad(q, params, allow_unused=True)
        names = ("site_frac", "occupancy", "u_iso", "u_star", "fp", "fdp")
        return {
            k: (torch.zeros_like(p) if gr is None else gr).detach().cpu().numpy().astype(np.float64)
            for k, p, gr in zip(names, params, grads)
        }

    def jvp(self, tangents, params=None):
        """Directional derivative (dF/dp) . v as a complex tensor (N_refl,)."""
        torch = self.torch
        params = self.tensors(requires_grad=True) if params is None else params
        tangents = tuple(torch.as_tensor(np.asarray(t, dtype=np.float64), dtype=self.dtype, device=self.device) for t in tangents)
        f = self.f_calc(*params)
        u = torch.zeros_like(f, requires_grad=True)
        q = (f * u.conj()).real.sum()
        grads = torch.autograd.grad(q, params, create_graph=True, allow_unused=True)
        s = sum((g * v).sum() for g, v in zip(grads, tangents) if g is not None)
        (df,) = torch.autograd.grad(s, u)
        return f.detach(), df.detach()

    def gauss_newton_hvp(self, tangents, curv_radial, curv_tangential, params=None):
        """Gauss-Newton Hessian-vector product (J^T H_F J) v."""
        torch = self.torch
        f, df = self.jvp(tangents, params=params)
        phase = f / f.abs().clamp(min=1e-300)
        c = df * phase.conj()
        cr = torch.as_tensor(np.asarray(curv_radial, dtype=np.float64), dtype=self.dtype, device=self.device)
        ct = torch.as_tensor(np.asarray(curv_tangential, dtype=np.float64), dtype=self.dtype, device=self.device)
        h_df = phase * torch.complex(cr * c.real, ct * c.imag)
        return self.gradients(h_df.cpu().numpy(), params=self.tensors(requires_grad=True))

    def gauss_newton_diagonal(
        self,
        curv_radial,
        curv_tangential,
        n_probes: int = 8,
        seed: int = 0,
        params=None,
    ):
        """Hutchinson estimate of diag(J^T H_F J) with Rademacher probes."""
        torch = self.torch
        if params is None:
            leaf_params = None
            shapes = [tuple(p.shape) for p in self.tensors(requires_grad=False)]
        else:
            leaf_params = tuple(
                p if getattr(p, "requires_grad", False) else p.detach().clone().requires_grad_(True)
                for p in params
            )
            shapes = [tuple(p.shape) for p in leaf_params]
        rng = np.random.default_rng(int(seed))
        names = ("site_frac", "occupancy", "u_iso", "u_star", "fp", "fdp")
        aniso = np.asarray(self.model.anisotropic, dtype=bool)
        acc = [np.zeros(s, dtype=np.float64) for s in shapes]
        m = max(1, int(n_probes))
        for _ in range(m):
            tangents = []
            for i, shape in enumerate(shapes):
                z = rng.choice(np.array([-1.0, 1.0]), size=shape).astype(np.float64)
                if i == 2:
                    z[aniso] = 0.0
                elif i == 3:
                    z[~aniso] = 0.0
                tangents.append(z)
            hv = self.gauss_newton_hvp(tangents, curv_radial, curv_tangential, params=leaf_params)
            for i, name in enumerate(names):
                acc[i] += tangents[i] * hv[name]
        out = {name: acc[i] / float(m) for i, name in enumerate(names)}
        out["u_iso"][aniso] = 0.0
        out["u_star"][~aniso] = 0.0
        return out

    def gauss_newton_blocks(self, curv_radial, curv_tangential, params=None):
        raise NotImplementedError("analytic gauss_newton_blocks are not implemented for the NUFFT engine")


def _form_factors(gauss_a: np.ndarray, gauss_b: np.ndarray, gauss_c: np.ndarray, stol2: np.ndarray) -> np.ndarray:
    """eltbx ``f(stol²) = Σ_k a_k exp(-b_k stol²) + c``, shape ``(n_types, N_refl)``."""
    a = np.asarray(gauss_a, dtype=np.float64)
    b = np.asarray(gauss_b, dtype=np.float64)
    c = np.asarray(gauss_c, dtype=np.float64)
    s = np.asarray(stol2, dtype=np.float64).reshape(-1)
    terms = np.sum(a[:, :, None] * np.exp(-b[:, :, None] * s[None, None, :]), axis=1)
    return c[:, None] + terms
