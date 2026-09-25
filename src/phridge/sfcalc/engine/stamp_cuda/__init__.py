"""Optional CUDA Ten Eyck stamp (one grid, exp table, algebraic gather on GPU).

JIT-compiled via ``torch.utils.cpp_extension.load``. Opt-in as
``stamp_backend="cuda"``; ``auto`` selects it on NVIDIA when the extension
builds. First-order backward is the same analytical VJP as the CPU C++
stamp. There is no Metal port — MPS ``auto`` uses the CPU C++ stamp.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from phridge.sfcalc.engine.stamp_cpp import _frac_scale, _pack, _reduce_expanded

_CUDA_OK: bool | None = None
_MOD = None
_EXP_CPU: Optional[np.ndarray] = None


def cuda_available() -> bool:
    global _CUDA_OK
    if _CUDA_OK is not None:
        return _CUDA_OK
    try:
        import torch

        if not torch.cuda.is_available():
            _CUDA_OK = False
            return False
        _module()
        _CUDA_OK = True
    except Exception:
        _CUDA_OK = False
    return _CUDA_OK


def _exp_cpu() -> np.ndarray:
    global _EXP_CPU
    if _EXP_CPU is None:
        _EXP_CPU = np.exp(np.arange(200000, dtype=np.float64) / -100.0)
    return _EXP_CPU


def _module():
    global _MOD
    if _MOD is not None:
        return _MOD
    from torch.utils.cpp_extension import load

    here = Path(__file__).resolve().parent
    _MOD = load(
        name="phridge_stamp_cuda",
        sources=[str(here / "stamp.cu")],
        extra_include_paths=[str(here)],
        extra_cuda_cflags=["-O3", "-std=c++17"],
        verbose=False,
    )
    return _MOD


def _as_cuda(arr, device, dtype) -> Any:
    import torch

    if hasattr(arr, "detach"):
        t = arr.detach()
        if t.dtype != dtype:
            t = t.to(dtype=dtype)
        return t.contiguous().to(device=device)
    np_dt = {torch.float16: np.float16, torch.float32: np.float32}.get(dtype, np.float64)
    return torch.as_tensor(np.ascontiguousarray(arr, dtype=np_dt), dtype=dtype, device=device)


def _exp_cuda(engine, device, dtype):
    import torch

    table = getattr(engine, "_cuda_exp_table", None)
    acc = torch.float32 if dtype == torch.float16 else dtype
    if table is None or table.device != device or table.dtype != acc:
        src = _exp_cpu() if acc == torch.float64 else np.exp(np.arange(200000, dtype=np.float32) / np.float32(-100.0))
        engine._cuda_exp_table = torch.as_tensor(src, dtype=acc, device=device)
    return engine._cuda_exp_table


def _ensure_cuda_grid(engine, total: int, has_imag: bool, device, dtype):
    import torch

    re = getattr(engine, "_cuda_grid_re", None)
    if re is None or re.numel() != total or re.device != device or re.dtype != dtype:
        engine._cuda_grid_re = torch.zeros(total, dtype=dtype, device=device)
    if has_imag:
        im = getattr(engine, "_cuda_grid_im", None)
        if im is None or im.numel() != total or im.device != device or im.dtype != dtype:
            engine._cuda_grid_im = torch.zeros(total, dtype=dtype, device=device)
    elif getattr(engine, "_cuda_grid_im", None) is None or engine._cuda_grid_im.dtype != dtype:
        engine._cuda_grid_im = torch.zeros(1, dtype=dtype, device=device)


def _to_cuda_pack(engine, p: dict, device, dtype) -> dict:
    import torch

    return {
        "x_exp": _as_cuda(p["x_exp"], device, dtype),
        "u_iso": _as_cuda(p["u_iso"], device, dtype),
        "u_cart": _as_cuda(p["u_cart"], device, dtype),
        "a": _as_cuda(p["a"], device, dtype),
        "b": _as_cuda(p["b"], device, dtype),
        "c": _as_cuda(p["c"], device, dtype),
        "w": _as_cuda(p["w"], device, dtype),
        "fp": _as_cuda(p["fp"], device, dtype),
        "fdp": _as_cuda(p["fdp"], device, dtype),
        "r_cut": _as_cuda(p["r_cut"], device, dtype),
        "aniso": torch.as_tensor(np.ascontiguousarray(p["aniso"], dtype=np.uint8), device=device),
        "has_imag": p["has_imag"],
    }


def _call_forward(engine, p: dict):
    import torch

    device = torch.device(engine.device)
    dtype = engine.dtype
    gpu = _to_cuda_pack(engine, p, device, dtype)
    n0, n1, n2 = (int(v) for v in engine.n_real)
    total = n0 * n1 * n2
    _ensure_cuda_grid(engine, total, gpu["has_imag"], device, dtype)
    grid_re = engine._cuda_grid_re
    grid_im = engine._cuda_grid_im
    grid_re.zero_()
    if gpu["has_imag"]:
        grid_im.zero_()
    fs = _frac_scale(engine)
    o = engine.o_mat.detach().to(device=device, dtype=dtype).reshape(3, 3).contiguous()
    _module().stamp_forward(
        grid_re,
        grid_im,
        gpu["x_exp"],
        gpu["u_iso"],
        gpu["u_cart"],
        gpu["a"],
        gpu["b"],
        gpu["c"],
        gpu["w"],
        gpu["fp"],
        gpu["fdp"],
        gpu["r_cut"],
        gpu["aniso"],
        o,
        _exp_cuda(engine, device, dtype),
        n0,
        n1,
        n2,
        float(fs[0]),
        float(fs[1]),
        float(fs[2]),
        float(engine.u_extra),
        1 if gpu["has_imag"] else 0,
    )
    re = grid_re.to(dtype=engine.dtype)
    if gpu["has_imag"]:
        im = grid_im.to(dtype=engine.dtype)
        if engine.dtype == torch.float16:
            return torch.complex(re.to(torch.float32), im.to(torch.float32)).reshape(engine.n_real)
        return torch.complex(re, im).reshape(engine.n_real)
    return re.reshape(engine.n_real)


def _density_cuda_forward(engine, sites_frac, occupancy, u_iso, u_star, fp, fdp):
    return _call_forward(engine, _pack(engine, sites_frac, occupancy, u_iso, u_star, fp, fdp))


def _vjp_numpy(engine, grad_rho, sites_frac, occupancy, u_iso, u_star, fp, fdp):
    import torch

    p = _pack(engine, sites_frac, occupancy, u_iso, u_star, fp, fdp)
    device = torch.device(engine.device)
    dtype = engine.dtype
    gpu = _to_cuda_pack(engine, p, device, dtype)
    n0, n1, n2 = (int(v) for v in engine.n_real)
    gre_t = grad_rho.real if grad_rho.is_complex() else grad_rho
    gre = gre_t.detach().to(device=device, dtype=dtype).reshape(-1).contiguous()
    if gpu["has_imag"] and grad_rho.is_complex():
        gim = grad_rho.imag.detach().to(device=device, dtype=dtype).reshape(-1).contiguous()
    else:
        gim = torch.zeros_like(gre)
    n_exp = int(gpu["x_exp"].shape[0])
    gx = torch.zeros((n_exp, 3), dtype=dtype, device=device)
    gu = torch.zeros(n_exp, dtype=dtype, device=device)
    guc = torch.zeros((n_exp, 3, 3), dtype=dtype, device=device)
    gw = torch.zeros(n_exp, dtype=dtype, device=device)
    gfp = torch.zeros(n_exp, dtype=dtype, device=device)
    gfdp = torch.zeros(n_exp, dtype=dtype, device=device)
    fs = _frac_scale(engine)
    o = engine.o_mat.detach().to(device=device, dtype=dtype).reshape(3, 3).contiguous()
    _module().stamp_vjp(
        gre,
        gim,
        gpu["x_exp"],
        gpu["u_iso"],
        gpu["u_cart"],
        gpu["a"],
        gpu["b"],
        gpu["c"],
        gpu["w"],
        gpu["fp"],
        gpu["fdp"],
        gpu["r_cut"],
        gpu["aniso"],
        o,
        _exp_cuda(engine, device, dtype),
        n0,
        n1,
        n2,
        float(fs[0]),
        float(fs[1]),
        float(fs[2]),
        float(engine.u_extra),
        1 if gpu["has_imag"] else 0,
        gx,
        gu,
        guc,
        gw,
        gfp,
        gfdp,
    )
    return _reduce_expanded(
        engine,
        gx.cpu().numpy(),
        gu.cpu().numpy(),
        guc.cpu().numpy(),
        gw.cpu().numpy(),
        gfp.cpu().numpy(),
        gfdp.cpu().numpy(),
        int(sites_frac.shape[0]),
    )


def cuda_f_calc_numpy(engine) -> np.ndarray:
    import torch

    m = engine.model
    with torch.no_grad():
        rho = _density_cuda_forward(engine, m.sites_frac, m.occupancy, m.u_iso, m.u_star, m.fp, m.fdp)
        from phridge.sfcalc.engine.engine import _fftn

        ft = _fftn(rho, engine.params.cpu_numpy_fft).reshape(-1)
        return engine._f_from_fft(ft).detach().cpu().numpy().astype(np.complex128)


def cuda_density(engine, sites_frac, occupancy, u_iso, u_star, fp, fdp):
    import torch

    class _CudaDensity(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, *params):
            ctx.engine = engine
            ctx.save_for_backward(*params)
            with torch.no_grad():
                return _density_cuda_forward(engine, *params)

        @staticmethod
        def backward(ctx: Any, grad_rho):
            saved = ctx.saved_tensors
            if grad_rho.requires_grad:
                leaves = tuple(t.detach().requires_grad_(True) for t in saved)
                rho = ctx.engine._density_torch(*leaves)
                return torch.autograd.grad(rho, leaves, grad_outputs=grad_rho, allow_unused=True, create_graph=True)
            gx, gocc, gu, gustar, gfp, gfdp = _vjp_numpy(ctx.engine, grad_rho, *saved)
            to = lambda a, ref: torch.as_tensor(a, dtype=ref.dtype, device=ref.device)
            return (
                to(gx, saved[0]),
                to(gocc, saved[1]),
                to(gu, saved[2]),
                to(gustar, saved[3]),
                to(gfp, saved[4]),
                to(gfdp, saved[5]),
            )

    return _CudaDensity.apply(sites_frac, occupancy, u_iso, u_star, fp, fdp)
