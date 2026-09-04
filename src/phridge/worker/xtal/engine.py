"""Differentiable FFT structure-factor engine (Ten Eyck sampling + torch.fft).

Forward model, per reflection h:

    F_h = sum_atoms sum_ops w_j (f_j(s) + fp_j + i fdp_j) exp(-2 pi^2 h^T U*_j h) exp(2 pi i h.(R x_j + t))

evaluated by sampling each (symmetry-expanded) atom as a sum of real-space
Gaussians on a grid, FFT, and gathering at h. An extra isotropic ``u_extra``
is added to every atom before sampling and divided out in reciprocal space
(cctbx ``u_base``), which also makes the constant form-factor term and
fp/fdp samplable.

Everything from the density builder to the gather is a torch function of
the scatterer parameters, so ``torch.autograd`` / ``torch.func`` supply
d F / d params, vector-Jacobian products (the Agarwal gradient-map trick)
and Gauss-Newton Hessian-vector products.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from phridge.worker.xtal.cell import (
    cell_volume,
    orthogonalization_matrix,
    reciprocal_cartesian,
    sym6_to_mat,
    u_base,
)
from phridge.worker.xtal.symmetry import identity_ops, site_multiplicities

TWO_PI2 = 2.0 * math.pi * math.pi
EIGHT_PI2 = 8.0 * math.pi * math.pi


@dataclass
class ScatteringModel:
    """Everything the engine needs about the model, as numpy arrays.

    Gaussian form factors follow eltbx: f(stol^2) = sum_k a_k exp(-b_k stol^2) + c.
    ``type_index[j]`` selects the row of ``gauss_a/gauss_b/gauss_c`` for atom j.
    Symmetry operators: x' = rot[s] @ x + trans[s], full list incl. centering.
    """

    unit_cell: tuple
    sites_frac: np.ndarray  # (N, 3)
    occupancy: np.ndarray  # (N,)
    u_iso: np.ndarray  # (N,)
    u_star: np.ndarray  # (N, 6)
    anisotropic: np.ndarray  # (N,) bool
    fp: np.ndarray  # (N,)
    fdp: np.ndarray  # (N,)
    type_index: np.ndarray  # (N,) int
    gauss_a: np.ndarray  # (T, K)
    gauss_b: np.ndarray  # (T, K)
    gauss_c: np.ndarray  # (T,)
    rot: np.ndarray = field(default_factory=lambda: identity_ops()[0])
    trans: np.ndarray = field(default_factory=lambda: identity_ops()[1])
    multiplicity: Optional[np.ndarray] = None  # (N,) int; computed if None

    def __post_init__(self) -> None:
        n = self.sites_frac.shape[0]
        for name in ("occupancy", "u_iso", "anisotropic", "fp", "fdp", "type_index"):
            if getattr(self, name).shape[0] != n:
                raise ValueError(f"{name} length mismatch")
        if self.u_star.shape != (n, 6):
            raise ValueError("u_star must be (N, 6)")
        if self.multiplicity is None:
            self.multiplicity = site_multiplicities(self.sites_frac, self.rot, self.trans, self.unit_cell)

    @property
    def n_scatterers(self) -> int:
        return int(self.sites_frac.shape[0])

    @property
    def n_sym(self) -> int:
        return int(self.rot.shape[0])


@dataclass
class EngineParams:
    d_min: float
    grid_resolution_factor: float = 1.0 / 3.0
    quality_factor: float = 100.0
    wing_cutoff: float = 1e-4
    u_extra: Optional[float] = None  # override cctbx u_base
    n_real: Optional[tuple] = None  # override gridding
    max_chunk_points: int = 8_000_000  # atoms x box points per chunk
    dtype: str = "float64"
    cpu_numpy_fft: bool = True  # route CPU FFTs through numpy (torch/MKL threading bug workaround)
    n_radius_buckets: int = 4  # group expanded atoms by cutoff radius to shrink sampling boxes


def _fft_friendly(n: int) -> int:
    """Smallest m >= n whose prime factors are all in {2,3,5}."""
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


def default_gridding(unit_cell, d_min: float, factor: float) -> tuple:
    a, b, c = (float(x) for x in unit_cell[:3])
    spacing = d_min * factor
    return tuple(_fft_friendly(int(math.ceil(length / spacing))) for length in (a, b, c))


def _iso_gauss(r2, sigma2):
    """Normalized spherical Gaussian: r2 (C,P), sigma2 (C,)."""
    import torch

    norm = (2.0 * math.pi * sigma2) ** (-1.5)
    return norm[:, None] * torch.exp(-0.5 * r2 / sigma2[:, None])


def _inv_det_3x3(m):
    """Batched closed-form inverse and determinant of symmetric (C,3,3)."""
    import torch

    a, b, c = m[:, 0, 0], m[:, 0, 1], m[:, 0, 2]
    d, e = m[:, 1, 1], m[:, 1, 2]
    f = m[:, 2, 2]
    c00 = d * f - e * e
    c01 = c * e - b * f
    c02 = b * e - c * d
    c11 = a * f - c * c
    c12 = b * c - a * e
    c22 = a * d - b * b
    det = a * c00 + b * c01 + c * c02
    inv = torch.stack(
        [
            torch.stack([c00, c01, c02], dim=1),
            torch.stack([c01, c11, c12], dim=1),
            torch.stack([c02, c12, c22], dim=1),
        ],
        dim=1,
    ) / det[:, None, None]
    return inv, det


def _np_fftn(t):
    import torch

    out = np.fft.fftn(t.detach().cpu().numpy())
    return torch.from_numpy(np.ascontiguousarray(out)).to(t.dtype)


def _np_ifftn_unnormalized(t):
    import torch

    out = np.fft.ifftn(t.detach().cpu().numpy(), norm="forward")
    return torch.from_numpy(np.ascontiguousarray(out)).to(t.dtype)


def _make_numpy_fftn():
    """fftn on CPU through numpy (pocketfft), as a pair of autograd Functions.

    torch.fft on CPU with intra-op threads > 1 was observed to return
    garbage intermittently (torch 2.14 / MKL) once other parallel ops had
    run in the process. numpy's FFT is single-threaded and reliable; the
    cost is negligible next to the density sampling. Only used for CPU
    tensors; CUDA goes through cuFFT. Forward and unnormalized inverse are
    each other's adjoints, so backward-of-backward works (needed for the
    Gauss-Newton Hessian-vector product).
    """
    import torch

    class NumpyFFTN(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inp):
            return _np_fftn(inp)

        @staticmethod
        def backward(ctx, grad):
            return NumpyIFFTN.apply(grad)

    class NumpyIFFTN(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inp):
            return _np_ifftn_unnormalized(inp)

        @staticmethod
        def backward(ctx, grad):
            return NumpyFFTN.apply(grad)

    return NumpyFFTN


_NUMPY_FFTN = None


def _fftn(x, cpu_numpy: bool):
    import torch

    global _NUMPY_FFTN
    if cpu_numpy and not x.is_cuda:
        if _NUMPY_FFTN is None:
            _NUMPY_FFTN = _make_numpy_fftn()
        return _NUMPY_FFTN.apply(x)
    return torch.fft.fftn(x)


class StructureFactorEngine:
    """Torch engine bound to one model layout, one hkl list, one grid."""

    def __init__(
        self,
        model: ScatteringModel,
        hkl: np.ndarray,
        params: EngineParams,
        device: str = "cpu",
    ) -> None:
        import torch

        self.torch = torch
        self.model = model
        self.params = params
        self.device = device
        self.dtype = torch.float64 if params.dtype == "float64" else torch.float32
        self.cdtype = torch.complex128 if params.dtype == "float64" else torch.complex64
        self.hkl = np.asarray(hkl, dtype=np.int64).reshape(-1, 3)

        cell = model.unit_cell
        self.n_real = tuple(int(x) for x in (params.n_real or default_gridding(cell, params.d_min, params.grid_resolution_factor)))
        self.volume = cell_volume(cell)
        o = orthogonalization_matrix(cell)
        self.o_mat = torch.as_tensor(o, dtype=self.dtype, device=device)
        self.u_extra = params.u_extra if params.u_extra is not None else u_base(params.d_min, params.grid_resolution_factor, params.quality_factor)

        # symmetry
        self.rot = torch.as_tensor(model.rot, dtype=self.dtype, device=device)  # (S,3,3)
        self.trans = torch.as_tensor(model.trans, dtype=self.dtype, device=device)  # (S,3)
        self.weight_wo_occ = torch.as_tensor(model.multiplicity / float(model.n_sym), dtype=self.dtype, device=device)

        # form factors per atom, (N, K)
        ti = model.type_index
        self.gauss_a = torch.as_tensor(model.gauss_a[ti], dtype=self.dtype, device=device)
        self.gauss_b = torch.as_tensor(model.gauss_b[ti], dtype=self.dtype, device=device)
        self.gauss_c = torch.as_tensor(model.gauss_c[ti], dtype=self.dtype, device=device)
        self.aniso = torch.as_tensor(model.anisotropic.astype(bool), device=device)

        # reciprocal-space bookkeeping
        n = np.array(self.n_real)
        if np.any(np.abs(self.hkl) * 2 >= n[None, :]):
            raise ValueError("hkl outside FFT grid; increase gridding or lower d_min")
        neg = (-self.hkl) % n[None, :]
        flat = (neg[:, 0] * n[1] + neg[:, 1]) * n[2] + neg[:, 2]
        self.gather_index = torch.as_tensor(flat, dtype=torch.int64, device=device)
        dstar2 = np.sum(reciprocal_cartesian(cell, self.hkl) ** 2, axis=1)
        self.u_extra_correction = torch.as_tensor(np.exp(TWO_PI2 * self.u_extra * dstar2), dtype=self.dtype, device=device)

        self._build_box(o)

    # ------------------------------------------------------------------ setup
    def _build_box(self, o: np.ndarray) -> None:
        """Per-expanded-atom cutoff radii and radius buckets with their own offset boxes."""
        torch = self.torch
        m = self.model
        n = np.array(self.n_real, dtype=np.float64)
        # widest variance per atom: largest eigenvalue of U_cart (+ widest form-factor term + u_extra)
        u_cart = np.einsum("ij,njk,lk->nil", o, sym6_to_mat(m.u_star), o)
        lam = np.linalg.eigvalsh(u_cart).max(axis=1)
        u_atom = np.where(m.anisotropic, lam, m.u_iso)
        b_max = m.gauss_b[m.type_index].max(axis=1)
        sigma2 = u_atom + b_max / EIGHT_PI2 + self.u_extra
        r_cut = np.sqrt(2.0 * math.log(1.0 / self.params.wing_cutoff) * sigma2)
        r_cut_exp = np.repeat(r_cut, m.n_sym)  # symmetry does not change U eigenvalues
        self.r_cut = torch.as_tensor(r_cut, dtype=self.dtype, device=self.device)
        self.r_cut_exp = torch.as_tensor(r_cut_exp, dtype=self.dtype, device=self.device)
        o_inv = np.linalg.inv(o)
        frac_scale = np.linalg.norm(o_inv, axis=1) * n  # cartesian radius -> grid units per axis

        # bucket by box half-width (largest axis) so small atoms use small boxes
        half_all = np.ceil(r_cut_exp[:, None] * frac_scale[None, :]).astype(int)  # (NS,3)
        key = half_all.max(axis=1)
        n_buckets = int(min(self.params.n_radius_buckets, max(1, len(np.unique(key)))))
        edges = np.quantile(key, np.linspace(0, 1, n_buckets + 1)[1:]) if n_buckets > 1 else [key.max()]
        edges = np.unique(np.ceil(edges).astype(int))
        self.buckets = []
        self.bucket_offsets = []
        assigned = np.zeros(len(key), dtype=bool)
        for edge in edges:
            members = np.where((key <= edge) & ~assigned)[0]
            assigned[members] = True
            if members.size == 0:
                continue
            half = half_all[members].max(axis=0)
            ranges = [np.arange(-h, h + 1) for h in half]
            off = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1).reshape(-1, 3)
            self.buckets.append(torch.as_tensor(members, dtype=torch.int64, device=self.device))
            self.bucket_offsets.append(torch.as_tensor(off, dtype=torch.int64, device=self.device))
        self.box_half = half_all.max(axis=0)
        self.offsets = self.bucket_offsets[-1]

    # ---------------------------------------------------------------- forward
    def expand(self, sites_frac, u_star):
        """Symmetry-expand sites (N,3)->(N*S,3) and u_star (N,6)->(N*S,3,3)."""
        torch = self.torch
        x = torch.einsum("sij,nj->nsi", self.rot, sites_frac) + self.trans[None]
        u = sym6_to_mat(u_star)
        u_exp = torch.einsum("sij,njk,slk->nsil", self.rot, u, self.rot)
        return x.reshape(-1, 3), u_exp.reshape(-1, 3, 3)

    def density(self, sites_frac, occupancy, u_iso, u_star, fp, fdp):
        """Sample the symmetry-expanded model onto the grid. Returns (complex) grid.

        Isotropic atoms take a spherical fast path (one |r|^2 per point);
        anisotropic atoms go through the full 3x3 covariance. Expanded atoms
        are bucketed by cutoff radius so small atoms do not pay for the
        largest atom's box.
        """
        torch = self.torch
        n_atoms, s = sites_frac.shape[0], self.rot.shape[0]
        x_exp, ustar_exp = self.expand(sites_frac, u_star)  # (NS,3), (NS,3,3)
        atom = torch.arange(n_atoms, device=self.device).repeat_interleave(s)
        rep = lambda t: t.repeat_interleave(s, dim=0)  # noqa: E731  (deterministic backward)

        u_cart_aniso = self.o_mat @ ustar_exp @ self.o_mat.T  # (NS,3,3)
        u_iso_exp = rep(u_iso)
        weight = rep(occupancy * self.weight_wo_occ)  # (NS,)
        fp_exp, fdp_exp = rep(fp), rep(fdp)
        a_all, b_all, c_all = self.gauss_a[atom], self.gauss_b[atom], self.gauss_c[atom]

        total = int(np.prod(self.n_real))
        grid_re = torch.zeros(total, dtype=self.dtype, device=self.device)
        has_imag = bool(torch.any(fdp != 0))
        grid_im = torch.zeros(total, dtype=self.dtype, device=self.device) if has_imag else None

        for bucket, offsets in zip(self.buckets, self.bucket_offsets):
            members = bucket  # indices into expanded atoms (int64 tensor)
            if members.numel() == 0:
                continue
            for aniso_flag in (False, True):
                sel = members[self.aniso[atom[members]] == aniso_flag]
                if sel.numel() == 0:
                    continue
                p = offsets.shape[0]
                chunk = max(1, self.params.max_chunk_points // p)
                for start in range(0, sel.numel(), chunk):
                    idx = sel[start : start + chunk]
                    r_cart, flat, inside = self._points(x_exp[idx], offsets, self.r_cut_exp[idx])
                    if aniso_flag:
                        base, terms = self._aniso_terms(r_cart, u_cart_aniso[idx], a_all[idx], b_all[idx])
                    else:
                        base, terms = self._iso_terms(r_cart, u_iso_exp[idx], a_all[idx], b_all[idx])
                    w = weight[idx]
                    rho_re = (terms + (c_all[idx] + fp_exp[idx])[:, None] * base) * w[:, None] * inside
                    grid_re = grid_re.index_add(0, flat.reshape(-1), rho_re.reshape(-1))
                    if has_imag:
                        rho_im = fdp_exp[idx][:, None] * base * w[:, None] * inside
                        grid_im = grid_im.index_add(0, flat.reshape(-1), rho_im.reshape(-1))

        grid = grid_re if grid_im is None else torch.complex(grid_re, grid_im)
        return grid.reshape(self.n_real)

    def _points(self, xf, offsets, r_cut):
        """Grid points around atoms xf (C,3): cartesian offsets, flat indices, cutoff mask."""
        torch = self.torch
        n_grid = torch.as_tensor(self.n_real, dtype=self.dtype, device=self.device)
        n_int = torch.as_tensor(self.n_real, dtype=torch.int64, device=self.device)
        g0 = torch.round(xf.detach() * n_grid).to(torch.int64)  # nearest grid point (constant)
        g = g0[:, None, :] + offsets[None, :, :]  # (C,P,3)
        r_frac = g.to(self.dtype) / n_grid - xf[:, None, :]
        r_cart = r_frac @ self.o_mat.T  # (C,P,3); K=3 GEMM, benign
        inside = (r_cart.detach() ** 2).sum(-1) <= r_cut[:, None] ** 2
        flat = ((g[..., 0] % n_int[0]) * n_int[1] + (g[..., 1] % n_int[1])) * n_int[2] + (g[..., 2] % n_int[2])
        return r_cart, flat, inside

    def _iso_terms(self, r_cart, u_iso, a, b):
        """Spherical Gaussians. Returns (base, sum_k a_k G_k) with base = G(u_iso + u_extra)."""
        torch = self.torch
        r2 = (r_cart**2).sum(-1)  # (C,P)
        sigma2_base = u_iso + self.u_extra  # (C,)
        base = _iso_gauss(r2, sigma2_base)
        terms = torch.zeros_like(r2)
        for k in range(a.shape[1]):
            terms = terms + a[:, k][:, None] * _iso_gauss(r2, sigma2_base + b[:, k] / EIGHT_PI2)
        return base, terms

    def _aniso_terms(self, r_cart, u_cart, a, b):
        torch = self.torch
        base = self._gauss(r_cart, u_cart, self.u_extra)
        terms = torch.zeros_like(base)
        for k in range(a.shape[1]):
            terms = terms + a[:, k][:, None] * self._gauss(r_cart, u_cart, b[:, k] / EIGHT_PI2 + self.u_extra)
        return base, terms

    def _gauss(self, r_cart, u_cart, extra_iso):
        """Normalized 3D Gaussian with covariance (u_cart + extra_iso I). r_cart: (C,P,3)."""
        torch = self.torch
        eye = torch.eye(3, dtype=self.dtype, device=self.device)
        cov = u_cart + extra_iso[:, None, None] * eye if torch.is_tensor(extra_iso) and extra_iso.ndim == 1 else u_cart + extra_iso * eye
        prec, det = _inv_det_3x3(cov)  # closed form: no LAPACK calls inside the hot loop
        # quadratic form r^T prec r written out (no batched GEMM: its backward
        # reduces over the P axis through MKL, which misbehaves with threads on
        # some CPU builds; elementwise products are exact and deterministic)
        x, y, z = r_cart[..., 0], r_cart[..., 1], r_cart[..., 2]
        p = lambda i, j: prec[:, i, j][:, None]  # noqa: E731
        q = p(0, 0) * x * x + p(1, 1) * y * y + p(2, 2) * z * z + 2.0 * (p(0, 1) * x * y + p(0, 2) * x * z + p(1, 2) * y * z)
        norm = (2.0 * math.pi) ** (-1.5) * det.rsqrt()
        return norm[:, None] * torch.exp(-0.5 * q)

    def f_calc(self, sites_frac, occupancy, u_iso, u_star, fp, fdp):
        """Structure factors at self.hkl, complex tensor (N_refl,)."""
        torch = self.torch
        rho = self.density(sites_frac, occupancy, u_iso, u_star, fp, fdp)
        if not torch.is_complex(rho):
            rho = rho.to(self.cdtype)
        ft = _fftn(rho, self.params.cpu_numpy_fft).reshape(-1)
        scale = self.volume / float(np.prod(self.n_real))
        return ft[self.gather_index] * (scale * self.u_extra_correction)

    # ---------------------------------------------------------------- helpers
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
        Returns dict of numpy arrays keyed site_frac, occupancy, u_iso, u_star, fp, fdp.
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
        """Directional derivative (dF/dp) . v as a complex tensor (N_refl,).

        Uses the double-vjp identity jvp(v) = d/du [ v . vjp(u) ] so that no
        functorch transform is needed (the CPU FFT is a custom Function).
        """
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
        """Gauss-Newton Hessian-vector product (J^T H_F J) v.

        ``tangents``: tuple of arrays shaped like (sites_frac, occupancy,
        u_iso, u_star, fp, fdp). ``curv_radial`` = d2g/d|F|^2 and
        ``curv_tangential`` = (dg/d|F|)/|F| per reflection (TargetEval).
        Returns dict of numpy arrays like ``gradients``.
        """
        torch = self.torch
        f, df = self.jvp(tangents, params=params)
        phase = f / f.abs().clamp(min=1e-300)
        c = df * phase.conj()  # radial (real) and tangential (imag) components of dF
        cr = torch.as_tensor(np.asarray(curv_radial, dtype=np.float64), dtype=self.dtype, device=self.device)
        ct = torch.as_tensor(np.asarray(curv_tangential, dtype=np.float64), dtype=self.dtype, device=self.device)
        h_df = phase * torch.complex(cr * c.real, ct * c.imag)  # H_h dF_h in the (A,B) plane
        return self.gradients(h_df.cpu().numpy(), params=self.tensors(requires_grad=True))
