"""Optional CUDA Triton fused isotropic Ten Eyck stamp.

Aniso atoms stay on the torch chunk path. Backward recomputes ``_density_torch``
so existing autograd gradients stay correct.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

EIGHT_PI2 = 8.0 * math.pi * math.pi

_TRITON_OK: bool | None = None


def triton_available() -> bool:
    global _TRITON_OK
    if _TRITON_OK is not None:
        return _TRITON_OK
    try:
        import torch
        import triton  # noqa: F401
        import triton.language as tl  # noqa: F401

        _TRITON_OK = bool(torch.cuda.is_available())
    except Exception:
        _TRITON_OK = False
    return _TRITON_OK


def _iso_stamp_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(
        xf_ptr,
        u_iso_ptr,
        a_ptr,
        b_ptr,
        c_ptr,
        w_ptr,
        fp_ptr,
        fdp_ptr,
        r_cut_ptr,
        off_ptr,
        o_ptr,
        grid_re_ptr,
        grid_im_ptr,
        n0,
        n1,
        n2,
        nf0,
        nf1,
        nf2,
        P,
        K,
        u_extra,
        has_imag,
        stride_xf,
        stride_a,
        C,
    ):
        atom = tl.program_id(0)
        if atom >= C:
            return
        x = tl.load(xf_ptr + atom * stride_xf + 0)
        y = tl.load(xf_ptr + atom * stride_xf + 1)
        z = tl.load(xf_ptr + atom * stride_xf + 2)
        u_iso = tl.load(u_iso_ptr + atom)
        c = tl.load(c_ptr + atom)
        w = tl.load(w_ptr + atom)
        fp = tl.load(fp_ptr + atom)
        fdp = tl.load(fdp_ptr + atom)
        r_cut = tl.load(r_cut_ptr + atom)
        o00 = tl.load(o_ptr + 0)
        o01 = tl.load(o_ptr + 1)
        o02 = tl.load(o_ptr + 2)
        o10 = tl.load(o_ptr + 3)
        o11 = tl.load(o_ptr + 4)
        o12 = tl.load(o_ptr + 5)
        o20 = tl.load(o_ptr + 6)
        o21 = tl.load(o_ptr + 7)
        o22 = tl.load(o_ptr + 8)
        g0 = tl.math.round(x * nf0).to(tl.int32)
        g1 = tl.math.round(y * nf1).to(tl.int32)
        g2 = tl.math.round(z * nf2).to(tl.int32)
        sigma2_base = u_iso + u_extra
        twopi = 6.283185307179586
        rcut2 = r_cut * r_cut
        for p in range(0, P):
            ox = tl.load(off_ptr + p * 3 + 0)
            oy = tl.load(off_ptr + p * 3 + 1)
            oz = tl.load(off_ptr + p * 3 + 2)
            gi0 = g0 + ox
            gi1 = g1 + oy
            gi2 = g2 + oz
            rf0 = gi0.to(tl.float32) / nf0 - x
            rf1 = gi1.to(tl.float32) / nf1 - y
            rf2 = gi2.to(tl.float32) / nf2 - z
            rcx = o00 * rf0 + o01 * rf1 + o02 * rf2
            rcy = o10 * rf0 + o11 * rf1 + o12 * rf2
            rcz = o20 * rf0 + o21 * rf1 + o22 * rf2
            r2 = rcx * rcx + rcy * rcy + rcz * rcz
            if r2 > rcut2:
                continue
            # wrap like Python % for positive n
            wi0 = ((gi0 % n0) + n0) % n0
            wi1 = ((gi1 % n1) + n1) % n1
            wi2 = ((gi2 % n2) + n2) % n2
            flat = (wi0 * n1 + wi1) * n2 + wi2
            inv_s = 1.0 / sigma2_base
            base = (twopi * sigma2_base) ** (-1.5) * tl.exp(-0.5 * r2 * inv_s)
            terms = 0.0
            for k in range(0, K):
                ak = tl.load(a_ptr + atom * stride_a + k)
                bk = tl.load(b_ptr + atom * stride_a + k)
                sigk = sigma2_base + bk / EIGHT_PI2
                terms += ak * ((twopi * sigk) ** (-1.5) * tl.exp(-0.5 * r2 / sigk))
            rho = (terms + (c + fp) * base) * w
            tl.atomic_add(grid_re_ptr + flat, rho)
            if has_imag:
                tl.atomic_add(grid_im_ptr + flat, fdp * base * w)

    return kernel


_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = _iso_stamp_kernel()
    return _KERNEL


def _stamp_iso_triton(
    x_exp,
    u_iso,
    a,
    b,
    c,
    w,
    fp,
    fdp,
    r_cut,
    offsets,
    o_mat,
    n_real,
    u_extra: float,
    grid_re,
    grid_im,
):
    import torch

    if x_exp.numel() == 0:
        return
    C = int(x_exp.shape[0])
    P = int(offsets.shape[0])
    K = int(a.shape[1])
    n0, n1, n2 = (int(v) for v in n_real)
    xf = x_exp.contiguous().to(torch.float32)
    kernel = _kernel()
    has_imag = 1 if grid_im is not None else 0
    dummy_im = grid_re if grid_im is None else grid_im
    kernel[(C,)](
        xf,
        u_iso.contiguous().to(torch.float32),
        a.contiguous().to(torch.float32),
        b.contiguous().to(torch.float32),
        c.contiguous().to(torch.float32),
        w.contiguous().to(torch.float32),
        fp.contiguous().to(torch.float32),
        fdp.contiguous().to(torch.float32),
        r_cut.contiguous().to(torch.float32),
        offsets.contiguous().to(torch.int32),
        o_mat.contiguous().to(torch.float32).reshape(-1),
        grid_re,
        dummy_im,
        n0,
        n1,
        n2,
        float(n0),
        float(n1),
        float(n2),
        P,
        K,
        float(u_extra),
        has_imag,
        xf.stride(0),
        a.stride(0),
        C,
    )


def _density_triton_forward(engine, sites_frac, occupancy, u_iso, u_star, fp, fdp):
    import torch

    pack = engine._expanded_stamp_inputs(sites_frac, occupancy, u_iso, u_star, fp, fdp)
    total = int(np.prod(engine.n_real))
    grid_re = torch.zeros(total, dtype=torch.float32, device=engine.device)
    has_imag = bool(torch.any(fdp != 0))
    grid_im = torch.zeros(total, dtype=torch.float32, device=engine.device) if has_imag else None

    for bucket, offsets in zip(engine.buckets, engine.bucket_offsets):
        if bucket.numel() == 0:
            continue
        iso = bucket[~pack["aniso_exp"][bucket]]
        aniso = bucket[pack["aniso_exp"][bucket]]
        if iso.numel():
            _stamp_iso_triton(
                pack["x_exp"][iso],
                pack["u_iso_exp"][iso],
                pack["a_all"][iso],
                pack["b_all"][iso],
                pack["c_all"][iso],
                pack["weight"][iso],
                pack["fp_exp"][iso],
                pack["fdp_exp"][iso],
                engine.r_cut_exp[iso],
                offsets,
                engine.o_mat,
                engine.n_real,
                float(engine.u_extra),
                grid_re,
                grid_im,
            )
        if aniso.numel():
            chunk = max(1, engine.params.max_chunk_points // max(int(offsets.shape[0]), 1))
            u_extra_t = grid_re.new_tensor(engine.u_extra)
            for start in range(0, aniso.numel(), chunk):
                idx = aniso[start : start + chunk]
                grid_re, grid_im = engine._stamp_chunk_fn(
                    grid_re,
                    grid_im,
                    pack["x_exp"][idx],
                    offsets,
                    engine.r_cut_exp[idx],
                    pack["u_iso_exp"][idx],
                    pack["u_cart_aniso"][idx],
                    pack["a_all"][idx],
                    pack["b_all"][idx],
                    pack["c_all"][idx],
                    pack["weight"][idx],
                    pack["fp_exp"][idx],
                    pack["fdp_exp"][idx],
                    engine.o_mat,
                    engine.n_real_f,
                    engine.n_real_i,
                    u_extra_t,
                    True,
                )

    if engine.dtype != grid_re.dtype:
        grid_re = grid_re.to(engine.dtype)
        if grid_im is not None:
            grid_im = grid_im.to(engine.dtype)
    grid = grid_re if grid_im is None else torch.complex(grid_re, grid_im)
    return grid.reshape(engine.n_real)


def triton_density(engine, sites_frac, occupancy, u_iso, u_star, fp, fdp):
    """Forward: Triton iso (+ torch aniso). Backward: torch density VJP."""
    import torch

    class _TritonDensity(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, *params):
            ctx.engine = engine
            ctx.save_for_backward(*params)
            with torch.no_grad():
                return _density_triton_forward(engine, *params)

        @staticmethod
        def backward(ctx: Any, grad_rho):
            leaves = tuple(t.detach().requires_grad_(True) for t in ctx.saved_tensors)
            rho = ctx.engine._density_torch(*leaves)
            rho.backward(grad_rho)
            return tuple(t.grad for t in leaves)

    return _TritonDensity.apply(sites_frac, occupancy, u_iso, u_star, fp, fdp)
