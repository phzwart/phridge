"""Optional CPU C++ Ten Eyck stamp (one grid, exp table).

JIT-compiled via ``torch.utils.cpp_extension.load``. Opt-in only
(``stamp_backend="cpp"``); ``auto`` is unchanged. First-order backward is
an analytical VJP over the same boxes. Higher-order autograd falls back
to the torch stamp.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from phridge.sfcalc.engine.cell import sym6_to_mat

_CPP_OK: bool | None = None
_MOD = None
_EXP_F64: Optional[np.ndarray] = None
_EXP_F32: Optional[np.ndarray] = None


def cpp_available() -> bool:
    global _CPP_OK
    if _CPP_OK is not None:
        return _CPP_OK
    try:
        _module()
        _CPP_OK = True
    except Exception:
        _CPP_OK = False
    return _CPP_OK


def _precision(engine) -> tuple[np.dtype, str]:
    import torch

    dt = engine.dtype
    if dt == torch.float16:
        return np.dtype(np.float16), "f16"
    if dt == torch.float32:
        return np.dtype(np.float32), "f32"
    return np.dtype(np.float64), "f64"


def _exp_table(suffix: str) -> np.ndarray:
    global _EXP_F64, _EXP_F32
    n = 200000
    if suffix == "f64":
        if _EXP_F64 is None:
            _EXP_F64 = np.exp(np.arange(n, dtype=np.float64) / -100.0)
        return _EXP_F64
    if _EXP_F32 is None:
        _EXP_F32 = np.exp(np.arange(n, dtype=np.float32) / np.float32(-100.0))
    return _EXP_F32


def _kernel(name: str, suffix: str):
    mod = _module()
    if suffix == "f16":
        if not bool(getattr(mod, "has_f16")()):
            suffix = "f32"
        else:
            return getattr(mod, f"{name}_f16")
    if suffix == "f32":
        return getattr(mod, f"{name}_f32")
    return getattr(mod, name)


def _compile_flags() -> tuple[list[str], list[str]]:
    """No OpenMP: torch already ships libomp; a second runtime SIGSEGVs on macOS."""
    return ["-O3", "-std=c++17", "-march=native"], []


def _module():
    global _MOD
    if _MOD is not None:
        return _MOD
    from torch.utils.cpp_extension import load

    here = Path(__file__).resolve().parent
    cflags, ldflags = _compile_flags()
    kwargs = dict(
        sources=[str(here / "stamp.cpp")],
        extra_include_paths=[str(here)],
        verbose=False,
    )
    try:
        _MOD = load(
            name="phridge_stamp_cpp_st",
            extra_cflags=cflags,
            extra_ldflags=ldflags,
            **kwargs,
        )
    except Exception:
        _MOD = load(
            name="phridge_stamp_cpp_st_serial",
            extra_cflags=["-O3", "-std=c++17", "-march=native"],
            extra_ldflags=[],
            **kwargs,
        )
    return _MOD


def _as_np(t) -> np.ndarray:
    if hasattr(t, "detach"):
        arr = t.detach()
        if str(arr.device) != "cpu":
            arr = arr.cpu()
        out = arr.numpy()
        return out if out.flags.c_contiguous else np.ascontiguousarray(out)
    return np.ascontiguousarray(np.asarray(t, dtype=np.float64))


def _frac_scale(engine) -> np.ndarray:
    fs = getattr(engine, "frac_scale", None)
    if fs is None:
        o = _as_np(engine.o_mat)
        n = np.asarray(engine.n_real, dtype=np.float64)
        o_inv = np.linalg.inv(o)
        fs = np.linalg.norm(o_inv, axis=1) * n
    return np.asarray(fs, dtype=np.float64).reshape(3)


def _ensure_buffers(engine, total: int, has_imag: bool, np_dt) -> None:
    re = getattr(engine, "_cpp_grid_re", None)
    if re is None or re.shape[0] != total or re.dtype != np_dt:
        engine._cpp_grid_re = np.zeros(total, dtype=np_dt)
    if has_imag:
        im = getattr(engine, "_cpp_grid_im", None)
        if im is None or im.shape[0] != total or im.dtype != np_dt:
            engine._cpp_grid_im = np.zeros(total, dtype=np_dt)
    else:
        im = getattr(engine, "_cpp_grid_im", None)
        if im is None or im.dtype != np_dt:
            engine._cpp_grid_im = np.zeros(1, dtype=np_dt)


def _o_np(engine, np_dt=np.float64) -> np.ndarray:
    return np.ascontiguousarray(_as_np(engine.o_mat).reshape(3, 3).astype(np_dt, copy=False))


def _pack(engine, sites_frac, occupancy, u_iso, u_star, fp, fdp) -> dict:
    if getattr(engine, "_asu_stamp", False):
        rot = np.eye(3, dtype=np.float64)[None]
        trans = np.zeros((1, 3), dtype=np.float64)
    else:
        rot = np.asarray(engine.model.rot, dtype=np.float64)
        trans = np.asarray(engine.model.trans, dtype=np.float64)
    sites = _as_np(sites_frac).astype(np.float64, copy=False).reshape(-1, 3)
    occ = _as_np(occupancy).astype(np.float64, copy=False).reshape(-1)
    uiso = _as_np(u_iso).astype(np.float64, copy=False).reshape(-1)
    ustar = _as_np(u_star).astype(np.float64, copy=False).reshape(-1, 6)
    fp_a = _as_np(fp).astype(np.float64, copy=False).reshape(-1)
    fdp_a = _as_np(fdp).astype(np.float64, copy=False).reshape(-1)
    n_atoms = sites.shape[0]
    n_sym = int(rot.shape[0])
    x_exp = np.einsum("sij,nj->nsi", rot, sites) + trans[None]
    x_exp = np.ascontiguousarray(x_exp.reshape(-1, 3))
    u_mat = sym6_to_mat(ustar)
    u_exp = np.einsum("sij,njk,slk->nsil", rot, u_mat, rot)
    o = _o_np(engine)
    u_cart = np.einsum("ij,nsjk,lk->nsil", o, u_exp, o)
    u_cart = np.ascontiguousarray(u_cart.reshape(-1, 3, 3))
    w_wo = engine.model.multiplicity.astype(np.float64) / float(engine.model.n_sym)
    atom = np.repeat(np.arange(n_atoms), n_sym)
    a_all = np.ascontiguousarray(engine.model.gauss_a[engine.model.type_index][atom])
    b_all = np.ascontiguousarray(engine.model.gauss_b[engine.model.type_index][atom])
    c_all = np.ascontiguousarray(engine.model.gauss_c[engine.model.type_index][atom])
    aniso = np.ascontiguousarray(engine.model.anisotropic.astype(np.uint8)[atom])
    r_cut = _as_np(engine.r_cut_exp).astype(np.float64, copy=False)
    if r_cut.shape[0] != x_exp.shape[0]:
        r_cut = np.repeat(_as_np(engine.r_cut).astype(np.float64, copy=False), n_sym)
    return {
        "x_exp": x_exp,
        "u_iso": np.ascontiguousarray(np.repeat(uiso, n_sym)),
        "u_cart": u_cart,
        "a": a_all,
        "b": b_all,
        "c": c_all,
        "w": np.ascontiguousarray(np.repeat(occ * w_wo, n_sym)),
        "fp": np.ascontiguousarray(np.repeat(fp_a, n_sym)),
        "fdp": np.ascontiguousarray(np.repeat(fdp_a, n_sym)),
        "r_cut": np.ascontiguousarray(r_cut),
        "aniso": aniso,
        "has_imag": bool(np.any(fdp_a != 0)),
    }


def _cast_pack(p: dict, np_dt):
    out = dict(p)
    for key in ("x_exp", "u_iso", "u_cart", "a", "b", "c", "w", "fp", "fdp", "r_cut"):
        out[key] = np.ascontiguousarray(p[key], dtype=np_dt)
    return out


def _call_forward(engine, p: dict) -> Any:
    import torch

    np_dt, suffix = _precision(engine)
    if suffix == "f16" and not bool(getattr(_module(), "has_f16")()):
        np_dt, suffix = np.dtype(np.float32), "f32"
    p = _cast_pack(p, np_dt)
    n0, n1, n2 = (int(v) for v in engine.n_real)
    total = n0 * n1 * n2
    _ensure_buffers(engine, total, p["has_imag"], np_dt)
    grid_re = engine._cpp_grid_re
    grid_im = engine._cpp_grid_im
    grid_re.fill(0)
    if p["has_imag"]:
        grid_im.fill(0)
    fs = _frac_scale(engine)
    _kernel("stamp_forward", suffix)(
        grid_re,
        grid_im,
        p["x_exp"],
        p["u_iso"],
        p["u_cart"],
        p["a"],
        p["b"],
        p["c"],
        p["w"],
        p["fp"],
        p["fdp"],
        p["r_cut"],
        p["aniso"],
        _o_np(engine, np_dt),
        _exp_table(suffix),
        n0,
        n1,
        n2,
        float(fs[0]),
        float(fs[1]),
        float(fs[2]),
        float(engine.u_extra),
        1 if p["has_imag"] else 0,
    )
    device = engine.device
    dtype = engine.dtype
    if str(device).startswith("mps"):
        device = "cpu"
    re = torch.from_numpy(np.ascontiguousarray(grid_re)).to(device=device, dtype=dtype)
    if p["has_imag"]:
        im = torch.from_numpy(np.ascontiguousarray(grid_im)).to(device=device, dtype=dtype)
        if dtype == torch.float16:
            return torch.complex(re.to(torch.float32), im.to(torch.float32)).reshape(engine.n_real)
        return torch.complex(re, im).reshape(engine.n_real)
    return re.reshape(engine.n_real)


def _density_cpp_forward(engine, sites_frac, occupancy, u_iso, u_star, fp, fdp):
    return _call_forward(engine, _pack(engine, sites_frac, occupancy, u_iso, u_star, fp, fdp))


def _reduce_expanded(engine, gx_exp, gu_iso_exp, gucart_exp, gw_exp, gfp_exp, gfdp_exp, n_atoms: int):
    if getattr(engine, "_asu_stamp", False):
        rot = np.eye(3, dtype=np.float64)[None]
    else:
        rot = np.asarray(engine.model.rot, dtype=np.float64)
    o = _o_np(engine)
    w_wo = engine.model.multiplicity.astype(np.float64) / float(engine.model.n_sym)
    n_sym = int(rot.shape[0])
    gx = np.zeros((n_atoms, 3), dtype=np.float64)
    gu_iso = np.zeros(n_atoms, dtype=np.float64)
    gu_star = np.zeros((n_atoms, 6), dtype=np.float64)
    gocc = np.zeros(n_atoms, dtype=np.float64)
    gfp = np.zeros(n_atoms, dtype=np.float64)
    gfdp = np.zeros(n_atoms, dtype=np.float64)
    o_t = o.T
    guc = gucart_exp.reshape(-1, 3, 3)
    for i in range(n_atoms * n_sym):
        a = i // n_sym
        s = i % n_sym
        r = rot[s]
        gx[a] += r.T @ gx_exp[i]
        gu_iso[a] += gu_iso_exp[i]
        gocc[a] += gw_exp[i] * w_wo[a]
        gfp[a] += gfp_exp[i]
        gfdp[a] += gfdp_exp[i]
        g_uexp = o_t @ guc[i] @ o
        g_u = r.T @ g_uexp @ r
        gu_star[a, 0] += g_u[0, 0]
        gu_star[a, 1] += g_u[1, 1]
        gu_star[a, 2] += g_u[2, 2]
        gu_star[a, 3] += g_u[0, 1] + g_u[1, 0]
        gu_star[a, 4] += g_u[0, 2] + g_u[2, 0]
        gu_star[a, 5] += g_u[1, 2] + g_u[2, 1]
    aniso = np.asarray(engine.model.anisotropic, dtype=bool)
    gu_iso[aniso] = 0.0
    gu_star[~aniso] = 0.0
    return gx, gocc, gu_iso, gu_star, gfp, gfdp


def _vjp_numpy(engine, grad_rho, sites_frac, occupancy, u_iso, u_star, fp, fdp):
    p = _pack(engine, sites_frac, occupancy, u_iso, u_star, fp, fdp)
    n0, n1, n2 = (int(v) for v in engine.n_real)
    np_dt, suffix = _precision(engine)
    if suffix == "f16" and not bool(getattr(_module(), "has_f16")()):
        np_dt, suffix = np.dtype(np.float32), "f32"
    p = _cast_pack(p, np_dt)
    gre_t = grad_rho.real if grad_rho.is_complex() else grad_rho
    gre = np.ascontiguousarray(_as_np(gre_t).reshape(-1).astype(np_dt))
    if p["has_imag"] and grad_rho.is_complex():
        gim = np.ascontiguousarray(_as_np(grad_rho.imag).reshape(-1).astype(np_dt))
    else:
        gim = np.zeros_like(gre)
    n_exp = p["x_exp"].shape[0]
    gx = np.zeros((n_exp, 3), dtype=np_dt)
    gu = np.zeros(n_exp, dtype=np_dt)
    guc = np.zeros((n_exp, 3, 3), dtype=np_dt)
    gw = np.zeros(n_exp, dtype=np_dt)
    gfp = np.zeros(n_exp, dtype=np_dt)
    gfdp = np.zeros(n_exp, dtype=np_dt)
    fs = _frac_scale(engine)
    _kernel("stamp_vjp", suffix)(
        gre,
        gim,
        p["x_exp"],
        p["u_iso"],
        p["u_cart"],
        p["a"],
        p["b"],
        p["c"],
        p["w"],
        p["fp"],
        p["fdp"],
        p["r_cut"],
        p["aniso"],
        _o_np(engine, np_dt),
        _exp_table(suffix),
        n0,
        n1,
        n2,
        float(fs[0]),
        float(fs[1]),
        float(fs[2]),
        float(engine.u_extra),
        1 if p["has_imag"] else 0,
        gx,
        gu,
        guc,
        gw,
        gfp,
        gfdp,
    )
    return _reduce_expanded(
        engine,
        np.asarray(gx, dtype=np.float64),
        np.asarray(gu, dtype=np.float64),
        np.asarray(guc, dtype=np.float64),
        np.asarray(gw, dtype=np.float64),
        np.asarray(gfp, dtype=np.float64),
        np.asarray(gfdp, dtype=np.float64),
        int(sites_frac.shape[0]),
    )


def cpp_f_calc_numpy(engine) -> np.ndarray:
    """Kept-engine F from ``engine.model`` numpy — no ``tensors().clone()``."""
    import torch

    m = engine.model
    with torch.no_grad():
        rho = _density_cpp_forward(engine, m.sites_frac, m.occupancy, m.u_iso, m.u_star, m.fp, m.fdp)
        from phridge.sfcalc.engine.engine import _fftn

        ft = _fftn(rho, engine.params.cpu_numpy_fft).reshape(-1)
        f = engine._f_from_fft(ft)
        return f.detach().cpu().numpy().astype(np.complex128)


def cpp_density(engine, sites_frac, occupancy, u_iso, u_star, fp, fdp):
    import torch

    class _CppDensity(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, *params):
            ctx.engine = engine
            ctx.save_for_backward(*params)
            with torch.no_grad():
                return _density_cpp_forward(engine, *params)

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

    return _CppDensity.apply(sites_frac, occupancy, u_iso, u_star, fp, fdp)
