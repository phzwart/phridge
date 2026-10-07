"""Differentiable FFT structure-factor engine (Ten Eyck sampling + torch.fft).

Forward model, per reflection h:

    F_h = sum_atoms sum_ops w_j (f_j(s) + fp_j + i fdp_j) exp(-2 pi^2 h^T U*_j h) exp(2 pi i h.(R x_j + t))

evaluated by sampling each (symmetry-expanded) atom as a sum of real-space
Gaussians on a grid, FFT, and gathering at h. An extra isotropic ``u_extra``
is added to every atom before sampling and divided out in reciprocal space
(cctbx ``u_base``), which also makes the constant form-factor term and
fp/fdp samplable.

First-order ``gradients`` use a hand-coded Agarwal step (scatter
``dQ/dF`` onto the FFT grid, IFFT, stamp VJP). ``jvp`` / Gauss-Newton
still go through autograd so the graph can be differentiated again.
"""

from __future__ import annotations

import math

import numpy as np

from phridge.transforms.fourier.ten_eyck.cell import (
    cell_volume,
    orthogonalization_matrix,
    reciprocal_cartesian,
    sym6_to_mat,
    u_base,
)
from phridge.transforms.fourier.ten_eyck.gauss_newton import (
    _gn_adp_row,
    _gn_block_from_df,
    _gn_scalar_from_df,
    _u_star_gradient_transform,
)
from phridge.transforms.fourier.ten_eyck.grids import (
    EIGHT_PI2,
    TWO_PI2,
    EngineParams,
    ScatteringModel,
    _fft_friendly,
    _fftn,
    _full_flat_to_rfft,
    _scatter_add_complex,
    _stamp_chunk,
    default_gridding,
)
from phridge.transforms.fourier.ten_eyck.symmetry import site_multiplicities


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
        name = (params.dtype or "float64").lower().replace("fp", "float")
        if name in ("16", "half", "float16"):
            self.dtype = torch.float16
            self.cdtype = getattr(torch, "complex32", torch.complex64)
        elif name in ("32", "float32"):
            self.dtype = torch.float32
            self.cdtype = torch.complex64
        else:
            self.dtype = torch.float64
            self.cdtype = torch.complex128
        if device.startswith("mps") and self.dtype == torch.float64:
            self.dtype = torch.float32
            self.cdtype = torch.complex64
        self.hkl = np.asarray(hkl, dtype=np.int64).reshape(-1, 3)

        cell = model.unit_cell
        self.n_real = tuple(int(x) for x in (params.n_real or default_gridding(cell, params.d_min, params.grid_resolution_factor)))
        self.n_real_f = torch.as_tensor(self.n_real, dtype=self.dtype, device=device)
        self.n_real_i = torch.as_tensor(self.n_real, dtype=torch.int64, device=device)
        self.volume = cell_volume(cell)
        if params.compile_stamp:
            try:
                self._stamp_chunk_fn = torch.compile(_stamp_chunk, fullgraph=False)
            except Exception:
                self._stamp_chunk_fn = _stamp_chunk
        else:
            self._stamp_chunk_fn = _stamp_chunk
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

        self._asu_stamp = False
        self.mate_index = None
        self.mate_phase = None
        if (not params.p1_expand) and model.n_sym > 1:
            try:
                from phridge.sfcalc.backends.sg_gather import agentsg_available, mate_tables

                if agentsg_available():
                    idx, ph = mate_tables(
                        self.hkl,
                        model.rot,
                        model.trans,
                        self.n_real,
                        hall=model.space_group_hall,
                    )
                    self.mate_index = torch.as_tensor(idx, dtype=torch.int64, device=device)
                    self.mate_phase = torch.as_tensor(ph, dtype=self.cdtype, device=device)
                    self._asu_stamp = True
            except Exception:
                self._asu_stamp = False

        # reciprocal-space bookkeeping
        n = np.array(self.n_real)
        if np.any(np.abs(self.hkl) * 2 >= n[None, :]):
            raise ValueError("hkl outside FFT grid; increase gridding or lower d_min")
        neg = (-self.hkl) % n[None, :]
        flat = (neg[:, 0] * n[1] + neg[:, 1]) * n[2] + neg[:, 2]
        self.gather_index = torch.as_tensor(flat, dtype=torch.int64, device=device)
        n0, n1, n2 = (int(v) for v in self.n_real)
        self.n_rfft = (n0, n1, n2 // 2 + 1)
        r_idx, r_conj, r_uniq = _full_flat_to_rfft(flat, n0, n1, n2)
        self.rfft_index = torch.as_tensor(r_idx, dtype=torch.int64, device=device)
        self.rfft_conj = torch.as_tensor(r_conj, dtype=torch.bool, device=device)
        self.rfft_unique = torch.as_tensor(r_uniq, dtype=torch.bool, device=device)
        if self._asu_stamp:
            m_idx, m_conj, m_uniq = _full_flat_to_rfft(
                self.mate_index.detach().cpu().numpy(), n0, n1, n2
            )
            self.rfft_mate_index = torch.as_tensor(m_idx, dtype=torch.int64, device=device)
            self.rfft_mate_conj = torch.as_tensor(m_conj, dtype=torch.bool, device=device)
            self.rfft_mate_unique = torch.as_tensor(m_uniq, dtype=torch.bool, device=device)
        else:
            self.rfft_mate_index = None
            self.rfft_mate_conj = None
            self.rfft_mate_unique = None
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
        n_stamp = 1 if self._asu_stamp else m.n_sym
        r_cut_exp = np.repeat(r_cut, n_stamp)
        self.r_cut = torch.as_tensor(r_cut, dtype=self.dtype, device=self.device)
        self.r_cut_exp = torch.as_tensor(r_cut_exp, dtype=self.dtype, device=self.device)
        o_inv = np.linalg.inv(o)
        frac_scale = np.linalg.norm(o_inv, axis=1) * n  # cartesian radius -> grid units per axis
        self.frac_scale = frac_scale

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
        """Symmetry-expand sites (N,3)->(N*S,3) and u_star (N,6)->(N*S,3,3).

        ASU stamp skips expansion: the FFT gather applies the group.
        """
        torch = self.torch
        if self._asu_stamp:
            return sites_frac, sym6_to_mat(u_star)
        x = torch.einsum("sij,nj->nsi", self.rot, sites_frac) + self.trans[None]
        u = sym6_to_mat(u_star)
        u_exp = torch.einsum("sij,njk,slk->nsil", self.rot, u, self.rot)
        return x.reshape(-1, 3), u_exp.reshape(-1, 3, 3)

    def _stamp_kind(self) -> str:
        """Resolved stamp implementation: ``torch``, ``triton``, ``numba``, ``cpp``, or ``cuda``."""
        backend = (self.params.stamp_backend or "auto").lower()
        device = str(self.device)
        if backend == "torch":
            return "torch"
        if backend == "triton":
            from phridge.sfcalc.backends.stamp_triton import triton_available

            if triton_available() and device.startswith("cuda"):
                return "triton"
            raise RuntimeError("stamp_backend='triton' requires CUDA + triton")
        if backend == "numba":
            from phridge.sfcalc.backends.stamp_numba import numba_available

            if numba_available() and not device.startswith(("cuda", "mps")):
                return "numba"
            raise RuntimeError("stamp_backend='numba' requires CPU + numba")
        if backend == "cpp":
            from phridge.transforms.fourier.ten_eyck.stamp_cpp import cpp_available

            if cpp_available():
                return "cpp"
            raise RuntimeError("stamp_backend='cpp' requires a C++ compiler")
        if backend == "cuda":
            from phridge.transforms.fourier.ten_eyck.stamp_cuda import cuda_available

            if cuda_available() and device.startswith("cuda"):
                return "cuda"
            raise RuntimeError("stamp_backend='cuda' requires NVIDIA CUDA")
        if backend == "fast":
            if device.startswith("cuda"):
                from phridge.transforms.fourier.ten_eyck.stamp_cuda import cuda_available

                if cuda_available():
                    return "cuda"
            from phridge.transforms.fourier.ten_eyck.stamp_cpp import cpp_available

            if cpp_available():
                return "cpp"
            backend = "auto"
        if backend == "auto":
            if device.startswith("cuda"):
                from phridge.transforms.fourier.ten_eyck.stamp_cuda import cuda_available
                from phridge.sfcalc.backends.stamp_triton import triton_available

                if cuda_available():
                    return "cuda"
                if triton_available():
                    return "triton"
            elif device.startswith("mps"):
                from phridge.transforms.fourier.ten_eyck.stamp_cpp import cpp_available

                if cpp_available():
                    return "cpp"
            else:
                from phridge.sfcalc.backends.stamp_numba import numba_available

                if numba_available():
                    return "numba"
            return "torch"
        raise ValueError(f"unknown stamp_backend {self.params.stamp_backend!r}")

    def _want_triton(self) -> bool:
        return self._stamp_kind() == "triton"

    def _expanded_stamp_inputs(self, sites_frac, occupancy, u_iso, u_star, fp, fdp):
        torch = self.torch
        n_atoms = sites_frac.shape[0]
        s = 1 if self._asu_stamp else self.rot.shape[0]
        x_exp, ustar_exp = self.expand(sites_frac, u_star)
        atom = torch.arange(n_atoms, device=self.device).repeat_interleave(s)
        rep = (lambda t: t) if self._asu_stamp else (lambda t: t.repeat_interleave(s, dim=0))
        return {
            "x_exp": x_exp,
            "u_cart_aniso": self.o_mat @ ustar_exp @ self.o_mat.T,
            "u_iso_exp": rep(u_iso),
            "weight": rep(occupancy * self.weight_wo_occ),
            "fp_exp": rep(fp),
            "fdp_exp": rep(fdp),
            "a_all": self.gauss_a[atom],
            "b_all": self.gauss_b[atom],
            "c_all": self.gauss_c[atom],
            "atom": atom,
            "aniso_exp": self.aniso[atom],
        }

    def _density_torch(self, sites_frac, occupancy, u_iso, u_star, fp, fdp):
        """Sample the symmetry-expanded model onto the grid (torch chunk path)."""
        torch = self.torch
        pack = self._expanded_stamp_inputs(sites_frac, occupancy, u_iso, u_star, fp, fdp)
        total = int(np.prod(self.n_real))
        grid_re = torch.zeros(total, dtype=self.dtype, device=self.device)
        has_imag = bool(torch.any(fdp != 0))
        grid_im = torch.zeros(total, dtype=self.dtype, device=self.device) if has_imag else None
        stamp = self._stamp_chunk_fn
        u_extra = grid_re.new_tensor(self.u_extra)

        for bucket, offsets in zip(self.buckets, self.bucket_offsets):
            if bucket.numel() == 0:
                continue
            for aniso_flag in (False, True):
                sel = bucket[pack["aniso_exp"][bucket] == aniso_flag]
                if sel.numel() == 0:
                    continue
                p = offsets.shape[0]
                chunk = max(1, self.params.max_chunk_points // p)
                for start in range(0, sel.numel(), chunk):
                    idx = sel[start : start + chunk]
                    grid_re, grid_im = stamp(
                        grid_re,
                        grid_im,
                        pack["x_exp"][idx],
                        offsets,
                        self.r_cut_exp[idx],
                        pack["u_iso_exp"][idx],
                        pack["u_cart_aniso"][idx],
                        pack["a_all"][idx],
                        pack["b_all"][idx],
                        pack["c_all"][idx],
                        pack["weight"][idx],
                        pack["fp_exp"][idx],
                        pack["fdp_exp"][idx],
                        self.o_mat,
                        self.n_real_f,
                        self.n_real_i,
                        u_extra,
                        aniso_flag,
                    )

        grid = grid_re if grid_im is None else torch.complex(grid_re, grid_im)
        return grid.reshape(self.n_real)

    def density(self, sites_frac, occupancy, u_iso, u_star, fp, fdp):
        """Sample the symmetry-expanded model onto the grid. Returns (complex) grid."""
        kind = self._stamp_kind()
        if kind == "triton":
            from phridge.sfcalc.backends.stamp_triton import triton_density

            return triton_density(self, sites_frac, occupancy, u_iso, u_star, fp, fdp)
        if kind == "numba":
            from phridge.sfcalc.backends.stamp_numba import numba_density

            return numba_density(self, sites_frac, occupancy, u_iso, u_star, fp, fdp)
        if kind == "cpp":
            from phridge.transforms.fourier.ten_eyck.stamp_cpp import cpp_density

            return cpp_density(self, sites_frac, occupancy, u_iso, u_star, fp, fdp)
        if kind == "cuda":
            from phridge.transforms.fourier.ten_eyck.stamp_cuda import cuda_density

            return cuda_density(self, sites_frac, occupancy, u_iso, u_star, fp, fdp)
        return self._density_torch(sites_frac, occupancy, u_iso, u_star, fp, fdp)

    def f_calc(self, sites_frac, occupancy, u_iso, u_star, fp, fdp):
        """Structure factors at self.hkl, complex tensor (N_refl,)."""
        torch = self.torch
        rho = self.density(sites_frac, occupancy, u_iso, u_star, fp, fdp)
        ft = _fftn(rho, self.params.cpu_numpy_fft).reshape(-1)
        return self._f_from_fft(ft)

    def _f_from_fft(self, ft):
        """Gather F_h from the FFT grid; ASU path sums agentsg mates.

        Real-density ``rfftn`` stores the last axis as ``n2//2+1``; missing
        ``-h`` bins are the conjugate of the stored Hermitian partner.
        """
        torch = self.torch
        scale = self.volume / float(np.prod(self.n_real))
        corr = scale * self.u_extra_correction.to(device=ft.device, dtype=ft.real.dtype)
        n_full = int(np.prod(self.n_real))
        if ft.numel() != n_full:
            if self._asu_stamp:
                idx = self.rfft_mate_index.to(device=ft.device)
                conj = self.rfft_mate_conj.to(device=ft.device)
                phase = self.mate_phase.to(device=ft.device, dtype=ft.dtype)
                vals = ft[idx]
                vals = torch.where(conj, vals.conj(), vals)
                return (vals * phase).sum(dim=-1) * corr
            idx = self.rfft_index.to(device=ft.device)
            vals = ft[idx]
            vals = torch.where(self.rfft_conj.to(device=ft.device), vals.conj(), vals)
            return vals * corr
        if self._asu_stamp:
            idx = self.mate_index.to(device=ft.device)
            phase = self.mate_phase.to(device=ft.device, dtype=ft.dtype)
            return (ft[idx] * phase).sum(dim=-1) * corr
        return ft[self.gather_index.to(device=ft.device)] * corr

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
        kind = self._stamp_kind()
        if kind == "cpp":
            from phridge.transforms.fourier.ten_eyck.stamp_cpp import cpp_f_calc_numpy

            return cpp_f_calc_numpy(self)
        if kind == "cuda":
            from phridge.transforms.fourier.ten_eyck.stamp_cuda import cuda_f_calc_numpy

            return cuda_f_calc_numpy(self)
        with torch.no_grad():
            return self.f_calc(*self.tensors()).cpu().numpy().astype(np.complex128)

    def _agarwal_grad_rho(self, d_target_d_f_calc):
        """Gradient map ∂Q/∂ρ from G_h: scatter onto FFT[-h], N·IFFT.

        ``F_h = (V/N) u_extra(h) FFT[ρ]_{-h}``. Adjoint of ``torch.fft.fftn``
        (norm='backward') is ``N · ifftn``.
        """
        torch = self.torch
        # CPU C++ stamp (including MPS auto) stays on host; f16 IFFT uses f32.
        if self._stamp_kind() == "cpp":
            device = "cpu"
            if self.dtype == torch.float16:
                dtype = torch.float32
                cdtype = torch.complex64
            else:
                dtype = self.dtype
                cdtype = self.cdtype
        else:
            device = self.device
            dtype = self.dtype
            cdtype = self.cdtype
        g = torch.as_tensor(np.asarray(d_target_d_f_calc, dtype=np.complex128), dtype=cdtype, device=device)
        n_grid = float(np.prod(self.n_real))
        scale = self.volume / n_grid
        slot = g * (scale * self.u_extra_correction.to(device=device, dtype=dtype))
        use_rfft = not bool(np.any(np.asarray(self.model.fdp)))
        if use_rfft:
            n0, n1, n2r = self.n_rfft
            y = torch.zeros(n0 * n1 * n2r, dtype=cdtype, device=device)
            if self._asu_stamp:
                phase = self.mate_phase.to(device=device, dtype=cdtype)
                contrib = (slot.to(cdtype)[:, None] * phase.conj()).reshape(-1)
                idx = self.rfft_mate_index.to(device=device).reshape(-1)
                conj = self.rfft_mate_conj.to(device=device).reshape(-1)
                uniq = self.rfft_mate_unique.to(device=device).reshape(-1)
            else:
                contrib = slot.to(cdtype)
                idx = self.rfft_index.to(device=device)
                conj = self.rfft_conj.to(device=device)
                uniq = self.rfft_unique.to(device=device)
            contrib = torch.where(conj, contrib.conj(), contrib)
            y = _scatter_add_complex(y, idx, contrib)
            rho0 = torch.zeros(self.n_real, dtype=dtype, device=device, requires_grad=True)
            r = torch.fft.rfftn(rho0).reshape(-1)
            (r * y.conj()).real.sum().backward()
            return rho0.grad.detach()
        y = torch.zeros(int(n_grid), dtype=cdtype, device=device)
        if self._asu_stamp:
            phase = self.mate_phase.to(device=device, dtype=cdtype)
            idx = self.mate_index.to(device=device)
            contrib = (slot.to(cdtype)[:, None] * phase.conj()).reshape(-1)
            y = _scatter_add_complex(y, idx.reshape(-1), contrib)
        else:
            y = _scatter_add_complex(y, self.gather_index.to(device=device), slot.to(cdtype))
        return torch.fft.ifftn(y.reshape(self.n_real)) * n_grid

    def _vjp_from_grad_rho(self, grad_rho, params):
        """Stamp adjoint: ∂Q/∂(sites, occ, U, f', f'') given ∂Q/∂ρ."""
        torch = self.torch
        names = ("site_frac", "occupancy", "u_iso", "u_star", "fp", "fdp")
        kind = self._stamp_kind()
        if kind == "numba":
            from phridge.sfcalc.backends.stamp_numba import _vjp_numpy

            gx, gocc, gu, gustar, gfp, gfdp = _vjp_numpy(self, grad_rho, *params)
            arrays = (gx, gocc, gu, gustar, gfp, gfdp)
            return {k: np.asarray(a, dtype=np.float64) for k, a in zip(names, arrays)}
        if kind == "cpp":
            from phridge.transforms.fourier.ten_eyck.stamp_cpp import _vjp_numpy as _vjp_cpp

            gx, gocc, gu, gustar, gfp, gfdp = _vjp_cpp(self, grad_rho, *params)
            arrays = (gx, gocc, gu, gustar, gfp, gfdp)
            return {k: np.asarray(a, dtype=np.float64) for k, a in zip(names, arrays)}
        if kind == "cuda":
            from phridge.transforms.fourier.ten_eyck.stamp_cuda import _vjp_numpy as _vjp_cuda

            gx, gocc, gu, gustar, gfp, gfdp = _vjp_cuda(self, grad_rho, *params)
            arrays = (gx, gocc, gu, gustar, gfp, gfdp)
            return {k: np.asarray(a, dtype=np.float64) for k, a in zip(names, arrays)}
        leaves = tuple(p.detach().clone().requires_grad_(True) for p in params)
        rho = self.density(*leaves)
        if not torch.is_complex(rho) and torch.is_complex(grad_rho):
            grad_out = grad_rho.real.to(dtype=rho.dtype)
        else:
            grad_out = grad_rho.to(dtype=rho.dtype)
        grads = torch.autograd.grad(rho, leaves, grad_outputs=grad_out, allow_unused=True)
        return {
            k: (torch.zeros_like(p) if gr is None else gr).detach().cpu().numpy().astype(np.float64)
            for k, p, gr in zip(names, leaves, grads)
        }

    def _gradients_autograd(self, d_target_d_f_calc: np.ndarray, params=None):
        """Full-tape VJP (used by ``jvp`` / tests). Prefer :meth:`gradients`."""
        torch = self.torch
        params = self.tensors(requires_grad=True) if params is None else params
        g = torch.as_tensor(
            np.asarray(d_target_d_f_calc, dtype=np.complex128), dtype=self.cdtype, device=self.device
        )
        f = self.f_calc(*params)
        q = (f * g.conj()).real.sum()
        grads = torch.autograd.grad(q, params, allow_unused=True)
        names = ("site_frac", "occupancy", "u_iso", "u_star", "fp", "fdp")
        return {
            k: (torch.zeros_like(p) if gr is None else gr).detach().cpu().numpy().astype(np.float64)
            for k, p, gr in zip(names, params, grads)
        }

    def gradients(self, d_target_d_f_calc: np.ndarray, params=None):
        """dQ/d(params) for Q with given per-reflection complex gradient.

        Convention (cctbx d_target_d_f_calc): G_h = dQ/dA_h + i dQ/dB_h, so
        dQ/dp = sum_h Re[conj(G_h) dF_h/dp].
        Uses a hand-coded Agarwal IFFT + stamp VJP (no autograd tape).
        Returns dict of numpy arrays keyed site_frac, occupancy, u_iso, u_star, fp, fdp.
        """
        if params is None:
            if self._stamp_kind() in ("cpp", "cuda"):
                m = self.model
                params = (m.sites_frac, m.occupancy, m.u_iso, m.u_star, m.fp, m.fdp)
            else:
                params = self.tensors(requires_grad=False)
        grad_rho = self._agarwal_grad_rho(d_target_d_f_calc)
        return self._vjp_from_grad_rho(grad_rho, params)

    def jvp(self, tangents, params=None):
        """Directional derivative (dF/dp) . v as a complex tensor (N_refl,).

        Uses the double-vjp identity jvp(v) = d/du [ v . vjp(u) ] so that no
        functorch transform is needed (also works if the optional numpy FFT
        custom Function is enabled).
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

    def gauss_newton_diagonal(
        self,
        curv_radial,
        curv_tangential,
        n_probes: int = 8,
        seed: int = 0,
        params=None,
    ):
        """Hutchinson estimate of diag(J^T H_F J) with Rademacher probes.

        ``diag(H) ≈ (1/m) Σ_k z_k ⊙ H z_k``, ``z_k ∈ {-1,+1}^n``. Inactive
        parameters (u_iso of anisotropic atoms, u_star of isotropic ones) are
        left at zero. Returns a dict like ``gradients``.
        """
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
                if i == 2:  # u_iso
                    z[aniso] = 0.0
                elif i == 3:  # u_star
                    z[~aniso] = 0.0
                tangents.append(z)
            hv = self.gauss_newton_hvp(
                tangents, curv_radial, curv_tangential, params=leaf_params
            )
            for i, name in enumerate(names):
                acc[i] += tangents[i] * hv[name]
        out = {name: acc[i] / float(m) for i, name in enumerate(names)}
        out["u_iso"][aniso] = 0.0
        out["u_star"][~aniso] = 0.0
        return out


    def gauss_newton_blocks(self, curv_radial, curv_tangential, params=None):
        """Exact per-atom Gauss-Newton blocks via the Tronrud sum/difference split.

        Returns dict with:
          site_frac (N,3,3), occupancy (N,), u_iso (N,), u_star (N,6,6),
          fp (N,), fdp (N,). Inactive ADP blocks/entries are zero.

        ``∂F/∂x = a_h κ(h) e^{2πi h·x}`` with symmetry folded into the
        Jacobian; the GN block is
        ``Σ_h [w^(-) Re[conj(∂F/∂x)∂F/∂y] + w^(+) Re[∂F/∂x ∂F/∂y e^{-2iφ}]]``
        (Tronrud/REFMAC D+S split, including cross-symmetry terms).
        """
        torch = self.torch
        if params is None:
            sites, occ, u_iso, u_star, fp, fdp = (
                np.asarray(x, dtype=np.float64) for x in (
                    self.model.sites_frac, self.model.occupancy, self.model.u_iso,
                    self.model.u_star, self.model.fp, self.model.fdp,
                )
            )
        else:
            sites, occ, u_iso, u_star, fp, fdp = (
                p.detach().cpu().numpy().astype(np.float64) for p in params
            )
        with torch.no_grad():
            f = self.f_calc(
                *[torch.as_tensor(x, dtype=self.dtype, device=self.device) for x in (sites, occ, u_iso, u_star, fp, fdp)]
            ).cpu().numpy().astype(np.complex128)
        amp = np.abs(f)
        phase = f / np.maximum(amp, 1e-300)
        e_m2iphi = np.conj(phase) ** 2
        if hasattr(curv_radial, "detach"):
            curv_radial = curv_radial.detach().cpu().numpy()
        if hasattr(curv_tangential, "detach"):
            curv_tangential = curv_tangential.detach().cpu().numpy()
        cr = np.asarray(curv_radial, dtype=np.float64)
        ct = np.asarray(curv_tangential, dtype=np.float64)
        # H_ab = sum_h [ c_r Re(d_a u*) Re(d_b u*) + c_t Im(d_a u*) Im(d_b u*) ]
        #      = sum_h [ (c_r + c_t)/2 Re(conj(d_a) d_b) + (c_r - c_t)/2 Re(d_a d_b e^{-2i phi}) ]
        # (Tronrud sum/difference split; the two weights coincide only when c_t = 0.)
        wp = 0.5 * (cr + ct)  # multiplies the conjugate product
        wm = 0.5 * (cr - ct)  # multiplies the phase-squared product

        h = self.hkl.astype(np.float64)
        n_h = h.shape[0]
        n = sites.shape[0]
        s = self.model.n_sym
        rot = np.asarray(self.model.rot, dtype=np.float64)
        trans = np.asarray(self.model.trans, dtype=np.float64)
        two_pi = 2.0 * math.pi
        aniso = np.asarray(self.model.anisotropic, dtype=bool)
        dstar2 = np.sum(reciprocal_cartesian(self.model.unit_cell, h) ** 2, axis=1)

        # Expanded atomic SF and sites (weight already includes 1/n_sym)
        a_exp = np.zeros((n * s, n_h), dtype=np.complex128)
        x_exp = np.zeros((n * s, 3), dtype=np.float64)
        R_exp = np.zeros((n * s, 3, 3), dtype=np.float64)
        stol2 = dstar2 / 4.0
        for j in range(n):
            ga = self.model.gauss_a[self.model.type_index[j]]
            gb = self.model.gauss_b[self.model.type_index[j]]
            gc = float(self.model.gauss_c[self.model.type_index[j]])
            ff = gc + float(fp[j]) + np.sum(ga[:, None] * np.exp(-gb[:, None] * stol2[None, :]), axis=0)
            weight = float(occ[j]) * float(self.model.multiplicity[j]) / float(self.model.n_sym)
            for si in range(s):
                e = j * s + si
                R_exp[e] = rot[si]
                x_exp[e] = rot[si] @ sites[j] + trans[si]
                if aniso[j]:
                    u_mat = sym6_to_mat(u_star[j : j + 1])[0]
                    u_sym = rot[si] @ u_mat @ rot[si].T
                    dw = np.exp(-TWO_PI2 * np.einsum("hi,ij,hj->h", h, u_sym, h))
                else:
                    dw = np.exp(-TWO_PI2 * float(u_iso[j]) * dstar2)
                a_exp[e] = weight * (ff + 1j * float(fdp[j])) * dw

        site_blocks = np.zeros((n, 3, 3), dtype=np.float64)
        occ_diag = np.zeros(n, dtype=np.float64)
        u_iso_diag = np.zeros(n, dtype=np.float64)
        u_star_blocks = np.zeros((n, 6, 6), dtype=np.float64)
        fp_diag = np.zeros(n, dtype=np.float64)
        fdp_diag = np.zeros(n, dtype=np.float64)

        # --- site 3x3 via dF construction (D + S, full symmetry) ---
        for j in range(n):
            dF = np.zeros((3, n_h), dtype=np.complex128)
            for si in range(s):
                e = j * s + si
                eph = np.exp(2j * math.pi * (h @ x_exp[e]))
                # ∂F/∂x_exp = a * 2πi * h * eph
                dF_exp = (a_exp[e] * eph)[None, :] * (1j * two_pi * h.T)  # (3, H)
                dF += R_exp[e].T @ dF_exp
            site_blocks[j] = _gn_block_from_df(dF, wm, wp, e_m2iphi)

            # occupancy
            dF_occ = np.zeros(n_h, dtype=np.complex128)
            for si in range(s):
                e = j * s + si
                eph = np.exp(2j * math.pi * (h @ x_exp[e]))
                dF_occ += (a_exp[e] / max(float(occ[j]), 1e-300)) * eph
            occ_diag[j] = _gn_scalar_from_df(dF_occ, wm, wp, e_m2iphi)

            # fp / fdp: ∂a/∂fp = weight * dw (real), ∂a/∂fdp = i * weight * dw
            w_wo = float(self.model.multiplicity[j]) / float(self.model.n_sym) * float(occ[j])
            dF_fp = np.zeros(n_h, dtype=np.complex128)
            dF_fdp = np.zeros(n_h, dtype=np.complex128)
            for si in range(s):
                e = j * s + si
                eph = np.exp(2j * math.pi * (h @ x_exp[e]))
                if aniso[j]:
                    u_mat = sym6_to_mat(u_star[j : j + 1])[0]
                    u_sym = rot[si] @ u_mat @ rot[si].T
                    dw = np.exp(-TWO_PI2 * np.einsum("hi,ij,hj->h", h, u_sym, h))
                else:
                    dw = np.exp(-TWO_PI2 * float(u_iso[j]) * dstar2)
                dF_fp += w_wo * dw * eph
                dF_fdp += 1j * w_wo * dw * eph
            fp_diag[j] = _gn_scalar_from_df(dF_fp, wm, wp, e_m2iphi)
            fdp_diag[j] = _gn_scalar_from_df(dF_fdp, wm, wp, e_m2iphi)

            u_star_blocks[j], u_iso_diag[j] = _gn_adp_row(
                j, bool(aniso[j]), s, h, x_exp, a_exp, rot, u_star, u_iso, dstar2, wm, wp, e_m2iphi,
            )


        return {
            "site_frac": site_blocks,
            "occupancy": occ_diag,
            "u_iso": u_iso_diag,
            "u_star": u_star_blocks,
            "fp": fp_diag,
            "fdp": fdp_diag,
        }


