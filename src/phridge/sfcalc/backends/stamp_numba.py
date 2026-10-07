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
    import numba as nb

    @nb.njit(inline="always")
    def iround(x):
        return int(np.rint(x))

    @nb.njit(inline="always")
    def inv_det(m00, m01, m02, m11, m12, m22):
        c00 = m11 * m22 - m12 * m12
        c01 = m02 * m12 - m01 * m22
        c02 = m01 * m12 - m02 * m11
        c11 = m00 * m22 - m02 * m02
        c12 = m01 * m02 - m00 * m12
        c22 = m00 * m11 - m01 * m01
        det = m00 * c00 + m01 * c01 + m02 * c02
        return c00 / det, c01 / det, c02 / det, c11 / det, c12 / det, c22 / det, det

    @nb.njit(inline="always")
    def iso_gauss(r2, s2):
        return (TWOPI * s2) ** (-1.5) * math.exp(-0.5 * r2 / s2)

    @nb.njit(inline="always")
    def wrap(i, n):
        r = i % n
        return r + n if r < 0 else r

    @nb.njit(inline="always")
    def aniso_gauss(rcx, rcy, rcz, u00, u01, u02, u11, u12, u22, extra):
        p00, p01, p02, p11, p12, p22, det = inv_det(
            u00 + extra, u01, u02, u11 + extra, u12, u22 + extra
        )
        q = (
            p00 * rcx * rcx
            + p11 * rcy * rcy
            + p22 * rcz * rcz
            + 2.0 * (p01 * rcx * rcy + p02 * rcx * rcz + p12 * rcy * rcz)
        )
        g = NORM0 * det ** (-0.5) * math.exp(-0.5 * q)
        pr0 = p00 * rcx + p01 * rcy + p02 * rcz
        pr1 = p01 * rcx + p11 * rcy + p12 * rcz
        pr2 = p02 * rcx + p12 * rcy + p22 * rcz
        return g, p00, p01, p02, p11, p12, p22, pr0, pr1, pr2

    @nb.njit(parallel=True, cache=True)
    def stamp_forward(
        local_re,
        local_im,
        xf,
        u_iso,
        u_cart,
        a,
        b,
        c,
        w,
        fp,
        fdp,
        r_cut,
        aniso,
        o,
        n0,
        n1,
        n2,
        fs0,
        fs1,
        fs2,
        u_extra,
        has_imag,
    ):
        n_exp = xf.shape[0]
        n_thr = local_re.shape[0]
        k_max = a.shape[1]
        nf0 = float(n0)
        nf1 = float(n1)
        nf2 = float(n2)
        o00, o01, o02 = o[0, 0], o[0, 1], o[0, 2]
        o10, o11, o12 = o[1, 0], o[1, 1], o[1, 2]
        o20, o21, o22 = o[2, 0], o[2, 1], o[2, 2]
        for tid in nb.prange(n_thr):
            start = (tid * n_exp) // n_thr
            end = ((tid + 1) * n_exp) // n_thr
            for i in range(start, end):
                x0 = xf[i, 0]
                x1 = xf[i, 1]
                x2 = xf[i, 2]
                rc = r_cut[i]
                rcut2 = rc * rc
                h0 = int(math.ceil(rc * fs0))
                h1 = int(math.ceil(rc * fs1))
                h2 = int(math.ceil(rc * fs2))
                g0 = iround(x0 * nf0)
                g1 = iround(x1 * nf1)
                g2 = iround(x2 * nf2)
                wi = w[i]
                cfp = c[i] + fp[i]
                fd = fdp[i]
                if not aniso[i]:
                    s2b = u_iso[i] + u_extra
                    nterm = k_max + 1
                    as_real = np.empty(nterm, dtype=np.float64)
                    bs_real = np.empty(nterm, dtype=np.float64)
                    for kk in range(k_max):
                        s2 = s2b + b[i, kk] / EIGHT_PI2
                        as_real[kk] = wi * a[i, kk] * (TWOPI * s2) ** (-1.5)
                        bs_real[kk] = -0.5 / s2
                    as_real[k_max] = wi * cfp * (TWOPI * s2b) ** (-1.5)
                    bs_real[k_max] = -0.5 / s2b
                    as_im = wi * fd * (TWOPI * s2b) ** (-1.5) if has_imag else 0.0
                    for di in range(-h0, h0 + 1):
                        gi0 = wrap(g0 + di, n0)
                        rf0 = (g0 + di) / nf0 - x0
                        t0x, t0y, t0z = o00 * rf0, o10 * rf0, o20 * rf0
                        row0 = gi0 * n1
                        for dj in range(-h1, h1 + 1):
                            gi1 = wrap(g1 + dj, n1)
                            rf1 = (g1 + dj) / nf1 - x1
                            t1x, t1y, t1z = o01 * rf1, o11 * rf1, o21 * rf1
                            row01 = (row0 + gi1) * n2
                            for dk in range(-h2, h2 + 1):
                                gi2 = wrap(g2 + dk, n2)
                                rf2 = (g2 + dk) / nf2 - x2
                                rcx = t0x + t1x + o02 * rf2
                                rcy = t0y + t1y + o12 * rf2
                                rcz = t0z + t1z + o22 * rf2
                                r2 = rcx * rcx + rcy * rcy + rcz * rcz
                                if r2 > rcut2:
                                    continue
                                contr = 0.0
                                for kk in range(nterm):
                                    contr += as_real[kk] * math.exp(bs_real[kk] * r2)
                                flat = row01 + gi2
                                local_re[tid, flat] += contr
                                if has_imag:
                                    local_im[tid, flat] += as_im * math.exp(bs_real[k_max] * r2)
                else:
                    u00 = u_cart[i, 0, 0]
                    u01 = u_cart[i, 0, 1]
                    u02 = u_cart[i, 0, 2]
                    u11 = u_cart[i, 1, 1]
                    u12 = u_cart[i, 1, 2]
                    u22 = u_cart[i, 2, 2]
                    for di in range(-h0, h0 + 1):
                        gi0 = wrap(g0 + di, n0)
                        rf0 = (g0 + di) / nf0 - x0
                        t0x, t0y, t0z = o00 * rf0, o10 * rf0, o20 * rf0
                        row0 = gi0 * n1
                        for dj in range(-h1, h1 + 1):
                            gi1 = wrap(g1 + dj, n1)
                            rf1 = (g1 + dj) / nf1 - x1
                            t1x, t1y, t1z = o01 * rf1, o11 * rf1, o21 * rf1
                            row01 = (row0 + gi1) * n2
                            for dk in range(-h2, h2 + 1):
                                gi2 = wrap(g2 + dk, n2)
                                rf2 = (g2 + dk) / nf2 - x2
                                rcx = t0x + t1x + o02 * rf2
                                rcy = t0y + t1y + o12 * rf2
                                rcz = t0z + t1z + o22 * rf2
                                r2 = rcx * rcx + rcy * rcy + rcz * rcz
                                if r2 > rcut2:
                                    continue
                                base, _, _, _, _, _, _, _, _, _ = aniso_gauss(
                                    rcx, rcy, rcz, u00, u01, u02, u11, u12, u22, u_extra
                                )
                                terms = 0.0
                                for kk in range(k_max):
                                    gk, _, _, _, _, _, _, _, _, _ = aniso_gauss(
                                        rcx,
                                        rcy,
                                        rcz,
                                        u00,
                                        u01,
                                        u02,
                                        u11,
                                        u12,
                                        u22,
                                        b[i, kk] / EIGHT_PI2 + u_extra,
                                    )
                                    terms += a[i, kk] * gk
                                flat = row01 + gi2
                                local_re[tid, flat] += (terms + cfp * base) * wi
                                if has_imag:
                                    local_im[tid, flat] += fd * base * wi

    @nb.njit(parallel=True, cache=True)
    def stamp_vjp(
        gre,
        gim,
        xf,
        u_iso,
        u_cart,
        a,
        b,
        c,
        w,
        fp,
        fdp,
        r_cut,
        aniso,
        o,
        n0,
        n1,
        n2,
        fs0,
        fs1,
        fs2,
        u_extra,
        has_imag,
        gx,
        gu_iso,
        gucart,
        gw,
        gfp,
        gfdp,
    ):
        n_exp = xf.shape[0]
        k_max = a.shape[1]
        nf0 = float(n0)
        nf1 = float(n1)
        nf2 = float(n2)
        for i in nb.prange(n_exp):
            x0 = xf[i, 0]
            x1 = xf[i, 1]
            x2 = xf[i, 2]
            rc = r_cut[i]
            rcut2 = rc * rc
            h0 = int(math.ceil(rc * fs0))
            h1 = int(math.ceil(rc * fs1))
            h2 = int(math.ceil(rc * fs2))
            g0 = iround(x0 * nf0)
            g1 = iround(x1 * nf1)
            g2 = iround(x2 * nf2)
            wi = w[i]
            cfp = c[i] + fp[i]
            fd = fdp[i]
            acc_x0 = 0.0
            acc_x1 = 0.0
            acc_x2 = 0.0
            acc_u = 0.0
            acc_uc00 = 0.0
            acc_uc01 = 0.0
            acc_uc02 = 0.0
            acc_uc11 = 0.0
            acc_uc12 = 0.0
            acc_uc22 = 0.0
            acc_w = 0.0
            acc_fp = 0.0
            acc_fdp = 0.0
            if not aniso[i]:
                s2b = u_iso[i] + u_extra
                for di in range(-h0, h0 + 1):
                    gi0 = g0 + di
                    rf0 = gi0 / nf0 - x0
                    for dj in range(-h1, h1 + 1):
                        gi1 = g1 + dj
                        rf1 = gi1 / nf1 - x1
                        for dk in range(-h2, h2 + 1):
                            gi2 = g2 + dk
                            rf2 = gi2 / nf2 - x2
                            rcx = o[0, 0] * rf0 + o[0, 1] * rf1 + o[0, 2] * rf2
                            rcy = o[1, 0] * rf0 + o[1, 1] * rf1 + o[1, 2] * rf2
                            rcz = o[2, 0] * rf0 + o[2, 1] * rf1 + o[2, 2] * rf2
                            r2 = rcx * rcx + rcy * rcy + rcz * rcz
                            if r2 > rcut2:
                                continue
                            base = iso_gauss(r2, s2b)
                            terms = 0.0
                            d_terms_dr2 = 0.0
                            d_terms_ds2 = 0.0
                            for kk in range(k_max):
                                s2k = s2b + b[i, kk] / EIGHT_PI2
                                gk = iso_gauss(r2, s2k)
                                terms += a[i, kk] * gk
                                d_terms_dr2 += a[i, kk] * gk * (-0.5 / s2k)
                                d_terms_ds2 += a[i, kk] * gk * (0.5 * r2 / (s2k * s2k) - 1.5 / s2k)
                            d_base_dr2 = base * (-0.5 / s2b)
                            d_base_ds2 = base * (0.5 * r2 / (s2b * s2b) - 1.5 / s2b)
                            flat = (wrap(gi0, n0) * n1 + wrap(gi1, n1)) * n2 + wrap(gi2, n2)
                            er = gre[flat]
                            ei = gim[flat] if has_imag else 0.0
                            rho_re = terms + cfp * base
                            acc_w += er * rho_re + ei * (fd * base)
                            acc_fp += er * wi * base
                            acc_fdp += ei * wi * base
                            dL_dr2 = er * wi * (d_terms_dr2 + cfp * d_base_dr2) + ei * wi * fd * d_base_dr2
                            # d(r2)/d(x) = -2 O.T @ r_cart
                            ot0 = o[0, 0] * rcx + o[1, 0] * rcy + o[2, 0] * rcz
                            ot1 = o[0, 1] * rcx + o[1, 1] * rcy + o[2, 1] * rcz
                            ot2 = o[0, 2] * rcx + o[1, 2] * rcy + o[2, 2] * rcz
                            acc_x0 += dL_dr2 * (-2.0 * ot0)
                            acc_x1 += dL_dr2 * (-2.0 * ot1)
                            acc_x2 += dL_dr2 * (-2.0 * ot2)
                            acc_u += er * wi * (d_terms_ds2 + cfp * d_base_ds2) + ei * wi * fd * d_base_ds2
            else:
                u00 = u_cart[i, 0, 0]
                u01 = u_cart[i, 0, 1]
                u02 = u_cart[i, 0, 2]
                u11 = u_cart[i, 1, 1]
                u12 = u_cart[i, 1, 2]
                u22 = u_cart[i, 2, 2]
                for di in range(-h0, h0 + 1):
                    gi0 = g0 + di
                    rf0 = gi0 / nf0 - x0
                    for dj in range(-h1, h1 + 1):
                        gi1 = g1 + dj
                        rf1 = gi1 / nf1 - x1
                        for dk in range(-h2, h2 + 1):
                            gi2 = g2 + dk
                            rf2 = gi2 / nf2 - x2
                            rcx = o[0, 0] * rf0 + o[0, 1] * rf1 + o[0, 2] * rf2
                            rcy = o[1, 0] * rf0 + o[1, 1] * rf1 + o[1, 2] * rf2
                            rcz = o[2, 0] * rf0 + o[2, 1] * rf1 + o[2, 2] * rf2
                            r2 = rcx * rcx + rcy * rcy + rcz * rcz
                            if r2 > rcut2:
                                continue
                            flat = (wrap(gi0, n0) * n1 + wrap(gi1, n1)) * n2 + wrap(gi2, n2)
                            er = gre[flat]
                            ei = gim[flat] if has_imag else 0.0

                            base, p00, p01, p02, p11, p12, p22, pr0, pr1, pr2 = aniso_gauss(
                                rcx, rcy, rcz, u00, u01, u02, u11, u12, u22, u_extra
                            )
                            terms = 0.0
                            dx0 = 0.0
                            dx1 = 0.0
                            dx2 = 0.0
                            duc00 = 0.0
                            duc01 = 0.0
                            duc02 = 0.0
                            duc11 = 0.0
                            duc12 = 0.0
                            duc22 = 0.0
                            scale_base = er * wi * cfp + ei * wi * fd
                            dx0 += scale_base * base * (o[0, 0] * pr0 + o[1, 0] * pr1 + o[2, 0] * pr2)
                            dx1 += scale_base * base * (o[0, 1] * pr0 + o[1, 1] * pr1 + o[2, 1] * pr2)
                            dx2 += scale_base * base * (o[0, 2] * pr0 + o[1, 2] * pr1 + o[2, 2] * pr2)
                            sb = scale_base * base
                            duc00 += sb * (-0.5 * p00 + 0.5 * pr0 * pr0)
                            duc01 += sb * (-0.5 * p01 + 0.5 * pr0 * pr1)
                            duc02 += sb * (-0.5 * p02 + 0.5 * pr0 * pr2)
                            duc11 += sb * (-0.5 * p11 + 0.5 * pr1 * pr1)
                            duc12 += sb * (-0.5 * p12 + 0.5 * pr1 * pr2)
                            duc22 += sb * (-0.5 * p22 + 0.5 * pr2 * pr2)
                            for kk in range(k_max):
                                extra = b[i, kk] / EIGHT_PI2 + u_extra
                                gk, p00, p01, p02, p11, p12, p22, pr0, pr1, pr2 = aniso_gauss(
                                    rcx, rcy, rcz, u00, u01, u02, u11, u12, u22, extra
                                )
                                terms += a[i, kk] * gk
                                sk = er * wi * a[i, kk]
                                dx0 += sk * gk * (o[0, 0] * pr0 + o[1, 0] * pr1 + o[2, 0] * pr2)
                                dx1 += sk * gk * (o[0, 1] * pr0 + o[1, 1] * pr1 + o[2, 1] * pr2)
                                dx2 += sk * gk * (o[0, 2] * pr0 + o[1, 2] * pr1 + o[2, 2] * pr2)
                                sg = sk * gk
                                duc00 += sg * (-0.5 * p00 + 0.5 * pr0 * pr0)
                                duc01 += sg * (-0.5 * p01 + 0.5 * pr0 * pr1)
                                duc02 += sg * (-0.5 * p02 + 0.5 * pr0 * pr2)
                                duc11 += sg * (-0.5 * p11 + 0.5 * pr1 * pr1)
                                duc12 += sg * (-0.5 * p12 + 0.5 * pr1 * pr2)
                                duc22 += sg * (-0.5 * p22 + 0.5 * pr2 * pr2)
                            acc_w += er * (terms + cfp * base) + ei * (fd * base)
                            acc_fp += er * wi * base
                            acc_fdp += ei * wi * base
                            acc_x0 += dx0
                            acc_x1 += dx1
                            acc_x2 += dx2
                            acc_uc00 += duc00
                            acc_uc01 += duc01
                            acc_uc02 += duc02
                            acc_uc11 += duc11
                            acc_uc12 += duc12
                            acc_uc22 += duc22
            gx[i, 0] = acc_x0
            gx[i, 1] = acc_x1
            gx[i, 2] = acc_x2
            gu_iso[i] = acc_u
            gucart[i, 0, 0] = acc_uc00
            gucart[i, 0, 1] = acc_uc01
            gucart[i, 0, 2] = acc_uc02
            gucart[i, 1, 0] = acc_uc01
            gucart[i, 1, 1] = acc_uc11
            gucart[i, 1, 2] = acc_uc12
            gucart[i, 2, 0] = acc_uc02
            gucart[i, 2, 1] = acc_uc12
            gucart[i, 2, 2] = acc_uc22
            gw[i] = acc_w
            gfp[i] = acc_fp
            gfdp[i] = acc_fdp

    return stamp_forward, stamp_vjp


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
