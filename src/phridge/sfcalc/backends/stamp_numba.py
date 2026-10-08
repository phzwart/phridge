"""Optional CPU Numba Ten Eyck stamp (per-atom box loop).

Forward paints with a parallel loop and thread-local grids (reduction after).
Iso terms use the cctbx ``a exp(b r²)`` form and a separable ``O @ Δx``.
First-order backward is an analytical VJP over the same boxes. Higher-order
autograd (``create_graph=True``, Gauss-Newton HVP) falls back to the torch stamp.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

EIGHT_PI2 = 8.0 * math.pi * math.pi
TWOPI = 2.0 * math.pi
NORM0 = TWOPI ** (-1.5)

_NUMBA_OK: bool | None = None
_STAMP = None
_VJP = None


def numba_available() -> bool:
    global _NUMBA_OK
    if _NUMBA_OK is not None:
        return _NUMBA_OK
    try:
        import numba  # noqa: F401

        _NUMBA_OK = True
    except Exception:
        _NUMBA_OK = False
    return _NUMBA_OK


def _compile():
    """Jit the stamp kernels. Helpers stay inline; the two entry points run in parallel."""
    import numba as nb

    from phridge.sfcalc.backends import stamp_numba_kernels as kernels

    kernels.prange = nb.prange
    for name in ("iround", "inv_det", "iso_gauss", "wrap", "aniso_gauss", "_vjp_aniso"):
        setattr(kernels, name, nb.njit(inline="always")(getattr(kernels, name)))
    forward = nb.njit(parallel=True, cache=True)(kernels.stamp_forward)
    vjp = nb.njit(parallel=True, cache=True)(kernels.stamp_vjp)
    return forward, vjp




def _kernels():
    global _STAMP, _VJP
    if _STAMP is None:
        _STAMP, _VJP = _compile()
    return _STAMP, _VJP


def _np(t):
    return np.ascontiguousarray(t.detach().cpu().numpy())


def _frac_scale(engine):
    fs = getattr(engine, "frac_scale", None)
    if fs is None:
        o = engine.o_mat.detach().cpu().numpy()
        n = np.asarray(engine.n_real, dtype=np.float64)
        o_inv = np.linalg.inv(o)
        fs = np.linalg.norm(o_inv, axis=1) * n
    return np.asarray(fs, dtype=np.float64).reshape(3)


def _pack_numpy(engine, sites_frac, occupancy, u_iso, u_star, fp, fdp):
    pack = engine._expanded_stamp_inputs(sites_frac, occupancy, u_iso, u_star, fp, fdp)
    has_imag = bool(np.any(_np(fdp) != 0))
    return {
        "x_exp": _np(pack["x_exp"]).astype(np.float64, copy=False),
        "u_iso": _np(pack["u_iso_exp"]).astype(np.float64, copy=False),
        "u_cart": _np(pack["u_cart_aniso"]).astype(np.float64, copy=False),
        "a": _np(pack["a_all"]).astype(np.float64, copy=False),
        "b": _np(pack["b_all"]).astype(np.float64, copy=False),
        "c": _np(pack["c_all"]).astype(np.float64, copy=False),
        "w": _np(pack["weight"]).astype(np.float64, copy=False),
        "fp": _np(pack["fp_exp"]).astype(np.float64, copy=False),
        "fdp": _np(pack["fdp_exp"]).astype(np.float64, copy=False),
        "r_cut": _np(engine.r_cut_exp).astype(np.float64, copy=False),
        "aniso": _np(pack["aniso_exp"]).astype(np.uint8, copy=False),
        "has_imag": has_imag,
    }


def _density_numba_forward(engine, sites_frac, occupancy, u_iso, u_star, fp, fdp):
    import torch

    import numba

    stamp, _ = _kernels()
    p = _pack_numpy(engine, sites_frac, occupancy, u_iso, u_star, fp, fdp)
    n0, n1, n2 = (int(v) for v in engine.n_real)
    total = n0 * n1 * n2
    n_thr = max(1, int(numba.get_num_threads()))
    local_re = np.zeros((n_thr, total), dtype=np.float64)
    local_im = np.zeros((n_thr, total), dtype=np.float64) if p["has_imag"] else np.zeros((n_thr, 1), dtype=np.float64)
    o = _np(engine.o_mat).astype(np.float64, copy=False)
    fs = _frac_scale(engine)
    stamp(
        local_re,
        local_im,
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
        o,
        n0,
        n1,
        n2,
        float(fs[0]),
        float(fs[1]),
        float(fs[2]),
        float(engine.u_extra),
        1 if p["has_imag"] else 0,
    )
    grid_re = local_re.sum(axis=0)
    re = torch.from_numpy(np.ascontiguousarray(grid_re)).to(device=engine.device, dtype=engine.dtype)
    if p["has_imag"]:
        grid_im = local_im.sum(axis=0)
        im = torch.from_numpy(np.ascontiguousarray(grid_im)).to(device=engine.device, dtype=engine.dtype)
        return torch.complex(re, im).reshape(engine.n_real)
    return re.reshape(engine.n_real)


def _reduce_expanded(
    engine,
    gx_exp,
    gu_iso_exp,
    gucart_exp,
    gw_exp,
    gfp_exp,
    gfdp_exp,
    n_atoms: int,
):
    if getattr(engine, "_asu_stamp", False):
        rot = np.eye(3, dtype=np.float64)[None]
    else:
        rot = np.asarray(engine.model.rot, dtype=np.float64)
    o = _np(engine.o_mat).astype(np.float64, copy=False)
    w_wo = engine.model.multiplicity.astype(np.float64) / float(engine.model.n_sym)
    n_sym = int(rot.shape[0])
    gx = np.zeros((n_atoms, 3), dtype=np.float64)
    gu_iso = np.zeros(n_atoms, dtype=np.float64)
    gu_star = np.zeros((n_atoms, 6), dtype=np.float64)
    gocc = np.zeros(n_atoms, dtype=np.float64)
    gfp = np.zeros(n_atoms, dtype=np.float64)
    gfdp = np.zeros(n_atoms, dtype=np.float64)
    o_t = o.T
    for i in range(n_atoms * n_sym):
        a = i // n_sym
        s = i % n_sym
        r = rot[s]
        gx[a] += r.T @ gx_exp[i]
        gu_iso[a] += gu_iso_exp[i]
        gocc[a] += gw_exp[i] * w_wo[a]
        gfp[a] += gfp_exp[i]
        gfdp[a] += gfdp_exp[i]
        g_uexp = o_t @ gucart_exp[i] @ o
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
    _, vjp = _kernels()
    p = _pack_numpy(engine, sites_frac, occupancy, u_iso, u_star, fp, fdp)
    n0, n1, n2 = (int(v) for v in engine.n_real)
    gre_t = grad_rho.real if grad_rho.is_complex() else grad_rho
    gre = np.ascontiguousarray(gre_t.detach().cpu().numpy().reshape(-1).astype(np.float64))
    if p["has_imag"] and grad_rho.is_complex():
        gim = np.ascontiguousarray(grad_rho.imag.detach().cpu().numpy().reshape(-1).astype(np.float64))
    else:
        gim = np.zeros_like(gre)
    n_exp = p["x_exp"].shape[0]
    gx = np.zeros((n_exp, 3), dtype=np.float64)
    gu = np.zeros(n_exp, dtype=np.float64)
    guc = np.zeros((n_exp, 3, 3), dtype=np.float64)
    gw = np.zeros(n_exp, dtype=np.float64)
    gfp = np.zeros(n_exp, dtype=np.float64)
    gfdp = np.zeros(n_exp, dtype=np.float64)
    o = _np(engine.o_mat).astype(np.float64, copy=False)
    fs = _frac_scale(engine)
    vjp(
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
        o,
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
    n_atoms = int(sites_frac.shape[0])
    return _reduce_expanded(engine, gx, gu, guc, gw, gfp, gfdp, n_atoms)


def numba_density(engine, sites_frac, occupancy, u_iso, u_star, fp, fdp):
    """Forward: Numba per-atom stamp. Backward: Numba VJP, or torch if create_graph."""
    import torch

    class _NumbaDensity(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, *params):
            ctx.engine = engine
            ctx.save_for_backward(*params)
            with torch.no_grad():
                return _density_numba_forward(engine, *params)

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

    return _NumbaDensity.apply(sites_frac, occupancy, u_iso, u_star, fp, fdp)
