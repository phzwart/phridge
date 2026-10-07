"""Grid stamping and FFT helpers for the Ten Eyck structure-factor engine."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from phridge.transforms.fourier.ten_eyck.symmetry import identity_ops, site_multiplicities

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
    space_group_hall: Optional[str] = None

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
    dtype: str = "float64"  # float64 | float32 | float16
    # Default: torch.fft.fftn everywhere (CUDA cuFFT / CPU pocketfft via torch) so the
    # density→FFT→gather graph stays in PyTorch and per-atom grads use autograd.
    # Opt-in numpy FFT only for rare CPU torch/MKL threading bugs (see docs/engine.md).
    cpu_numpy_fft: bool = False
    n_radius_buckets: int = 4  # group expanded atoms by cutoff radius to shrink sampling boxes
    compile_stamp: bool = False  # torch.compile the per-chunk stamp (off in unit tests)
    stamp_backend: str = "auto"  # auto | fast | torch | triton | numba | cpp | cuda
    p1_expand: bool = False  # True: paint all sym copies; False: ASU + agentsg gather


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


def _points_core(xf, offsets, r_cut, o_mat, n_real_f, n_real_i):
    """Grid points around atoms xf (C,3): cartesian offsets, flat indices, cutoff mask."""
    import torch

    g0 = torch.round(xf.detach() * n_real_f).to(torch.int64)
    g = g0[:, None, :] + offsets[None, :, :]
    r_frac = g.to(xf.dtype) / n_real_f - xf[:, None, :]
    r_cart = r_frac @ o_mat.T
    inside = (r_cart.detach() ** 2).sum(-1) <= r_cut[:, None] ** 2
    flat = ((g[..., 0] % n_real_i[0]) * n_real_i[1] + (g[..., 1] % n_real_i[1])) * n_real_i[2] + (g[..., 2] % n_real_i[2])
    return r_cart, flat, inside


def _iso_terms_core(r_cart, u_iso, a, b, u_extra):
    r2 = (r_cart**2).sum(-1)
    sigma2_base = u_iso + u_extra
    base = _iso_gauss(r2, sigma2_base)
    sigma2_k = sigma2_base[:, None] + b / EIGHT_PI2
    norm = (2.0 * math.pi * sigma2_k) ** (-1.5)
    gauss_k = norm[:, :, None] * (-0.5 * r2[:, None, :] / sigma2_k[:, :, None]).exp()
    terms = (a[:, :, None] * gauss_k).sum(1)
    return base, terms


def _gauss_core(r_cart, u_cart, extra_iso):
    import torch

    eye = torch.eye(3, dtype=r_cart.dtype, device=r_cart.device)
    if torch.is_tensor(extra_iso) and extra_iso.ndim == 1:
        cov = u_cart + extra_iso[:, None, None] * eye
    else:
        cov = u_cart + extra_iso * eye
    prec, det = _inv_det_3x3(cov)
    x, y, z = r_cart[..., 0], r_cart[..., 1], r_cart[..., 2]
    p = lambda i, j: prec[:, i, j][:, None]  # noqa: E731
    q = p(0, 0) * x * x + p(1, 1) * y * y + p(2, 2) * z * z + 2.0 * (p(0, 1) * x * y + p(0, 2) * x * z + p(1, 2) * y * z)
    norm = (2.0 * math.pi) ** (-1.5) * det.rsqrt()
    return norm[:, None] * torch.exp(-0.5 * q)


def _aniso_terms_core(r_cart, u_cart, a, b, u_extra):
    base = _gauss_core(r_cart, u_cart, u_extra)
    c, k = int(a.shape[0]), int(a.shape[1])
    p = int(r_cart.shape[1])
    extra = b / EIGHT_PI2 + u_extra
    r_rep = r_cart[:, None, :, :].expand(c, k, p, 3).reshape(c * k, p, 3)
    u_rep = u_cart[:, None, :, :].expand(c, k, 3, 3).reshape(c * k, 3, 3)
    gauss_k = _gauss_core(r_rep, u_rep, extra.reshape(c * k)).reshape(c, k, p)
    terms = (a[:, :, None] * gauss_k).sum(1)
    return base, terms


def _stamp_chunk(
    grid_re,
    grid_im,
    xf,
    offsets,
    r_cut,
    u_iso,
    u_cart,
    a,
    b,
    c,
    w,
    fp,
    fdp,
    o_mat,
    n_real_f,
    n_real_i,
    u_extra,
    aniso: bool,
):
    """Paint one chunk of expanded atoms onto the flat real/imag grids."""
    r_cart, flat, inside = _points_core(xf, offsets, r_cut, o_mat, n_real_f, n_real_i)
    if aniso:
        base, terms = _aniso_terms_core(r_cart, u_cart, a, b, u_extra)
    else:
        base, terms = _iso_terms_core(r_cart, u_iso, a, b, u_extra)
    rho_re = (terms + (c + fp)[:, None] * base) * w[:, None] * inside
    grid_re = grid_re.index_add(0, flat.reshape(-1), rho_re.reshape(-1))
    if grid_im is not None:
        rho_im = fdp[:, None] * base * w[:, None] * inside
        grid_im = grid_im.index_add(0, flat.reshape(-1), rho_im.reshape(-1))
    return grid_re, grid_im


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
    """Optional CPU fftn via numpy (pocketfft), as a pair of autograd Functions.

    Only used when ``EngineParams.cpu_numpy_fft`` is True. Default path is
    ``torch.fft.fftn``. Kept for rare torch 2.x / MKL CPU builds where
    multi-threaded ``torch.fft`` returned garbage intermittently. Forward and
    unnormalized inverse are each other's adjoints (Gauss-Newton HVP).
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


def _full_flat_to_rfft(flat: np.ndarray, n0: int, n1: int, n2: int):
    """Map full-grid flat index of ``-h`` to packed ``rfftn`` index + conj/unique."""
    flat = np.asarray(flat, dtype=np.int64)
    n2r = n2 // 2 + 1
    i2 = flat % n2
    tmp = flat // n2
    i1 = tmp % n1
    i0 = tmp // n1
    conj = i2 >= n2r
    j0 = np.where(conj, (-i0) % n0, i0)
    j1 = np.where(conj, (-i1) % n1, i1)
    j2 = np.where(conj, (-i2) % n2, i2)
    idx = (j0 * n1 + j1) * n2r + j2

    def _nyq(i, n):
        return (i == 0) | ((n % 2 == 0) & (i == n // 2))

    unique = _nyq(j0, n0) & _nyq(j1, n1) & _nyq(j2, n2)
    return idx.astype(np.int64), conj, unique


def _scatter_add_complex(y, index, contrib):
    """``scatter_add_`` that works when MPS refuses complex scatter."""
    try:
        y.scatter_add_(0, index, contrib)
        return y
    except RuntimeError:
        import torch

        re = y.real.contiguous()
        im = y.imag.contiguous()
        re.scatter_add_(0, index, contrib.real)
        im.scatter_add_(0, index, contrib.imag)
        return torch.complex(re, im)


def _fftn(x, cpu_numpy: bool = False):
    """3D FFT of the density grid. Real maps use ``rfftn``."""
    import torch

    global _NUMPY_FFTN
    if not x.is_complex() and x.dtype == torch.float16:
        x = x.to(torch.float32)
    elif x.is_complex() and x.dtype not in (torch.complex64, torch.complex128):
        x = x.to(torch.complex64)
    if not x.is_complex():
        return torch.fft.rfftn(x)
    if cpu_numpy and not x.is_cuda:
        if _NUMPY_FFTN is None:
            _NUMPY_FFTN = _make_numpy_fftn()
        return _NUMPY_FFTN.apply(x)
    return torch.fft.fftn(x)

