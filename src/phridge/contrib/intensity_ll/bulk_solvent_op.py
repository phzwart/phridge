"""Worker op ``ml_i_bulk_solvent_fit``: binned k_mask by maximum likelihood.

mmtbx's fast scaler does not fit ``(k_sol, B_sol)``. It grid-searches one scalar
``k_mask`` per resolution bin, smooths, and interpolates onto every reflection:

    F_model = k_model (F_calc + k_mask(s) F_mask)

This op does the same thing to the intensity NLL. ``k_mask(s)`` is linear
interpolation of one value per resolution bin; ``k_model`` is held. σ_A and β are
*profiled* per shell (same bins) and discarded, so an overall scale cannot masquerade
as a better solvent — only the F_calc / F_mask mix is left for ``k_mask`` to explain.

The caller's k_mask (mmtbx's least-squares curve) is profiled the same way and is the
reference: the fit is installed only if it beats that curve on the fit set.

``k_sol`` / ``B_sol`` in the return are a two-parameter caption of the fitted curve
(Gaussian fit on ``d ≥ 4 Å``), not the model.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from phridge.packing import PackedMiller

BULK_SOLVENT_OP_NAME = "ml_i_bulk_solvent_fit"

K_MASK_MAX = 1.0

_BULK_SOLVENT_INPUTS = {
    "f_calc": "MillerArray",
    "f_obs": "MillerArray",
    "f_mask": "array",
    "k_model": "array",
    "k_mask": "array",
    "fit_mask": "array",
    "ss": "array",
    "epsilon": "array",
    "centric": "array",
    "sigma_wilson": "array",
    "nu": "array",
    "k_sol": "json",
    "b_sol": "json",
    "n_shells": "json",
    "smooth": "json",
    "max_iter": "json",
}
_BULK_SOLVENT_OUTPUTS = {
    "k_sol": "json",
    "b_sol": "json",
    "k_mask": "array",
    "accepted": "json",
    "stats": "json",
}


def _logit(frac: float) -> float:
    frac = min(max(float(frac), 1e-4), 1.0 - 1e-4)
    return math.log(frac / (1.0 - frac))


def _k_sol_b_sol_from_curve(ss: np.ndarray, k_mask: np.ndarray) -> tuple[float, float]:
    """Two-number caption of a k_mask curve, matching mmtbx's d ≥ 4 Å convention."""
    sel = ss <= 1.0 / (4.0 * 16.0)
    if int(sel.sum()) < 10:
        sel = np.ones(ss.shape, dtype=bool)
    y = np.asarray(k_mask, dtype=np.float64)[sel]
    x = np.asarray(ss, dtype=np.float64)[sel]
    pos = np.isfinite(y) & np.isfinite(x) & (y > 1e-6)
    if int(pos.sum()) < 5:
        return float(np.nanmax(y)) if y.size else 0.35, 46.0
    a = np.stack([np.ones(int(pos.sum())), -x[pos]], axis=1)
    coef, *_ = np.linalg.lstsq(a, np.log(y[pos]), rcond=None)
    ks = float(min(max(math.exp(float(coef[0])), 0.0), 0.6))
    bs = float(min(max(float(coef[1]), 0.0), 150.0))
    return ks, bs


def _k_at_percentiles(ss: np.ndarray, k_mask: np.ndarray, sel: np.ndarray) -> list[float]:
    """k_mask at the 10 / 50 / 90th percentiles of ss (low / mid / high resolution)."""
    ss_f = np.asarray(ss, dtype=np.float64)[sel]
    k_f = np.asarray(k_mask, dtype=np.float64)[sel]
    if ss_f.size == 0:
        return [float("nan"), float("nan"), float("nan")]
    order = np.argsort(ss_f, kind="stable")
    out: list[float] = []
    for q in (0.10, 0.50, 0.90):
        i = int(round(q * (ss_f.size - 1)))
        out.append(float(k_f[order[i]]))
    return out


def _bin_edges(ss_fit: np.ndarray, n_shells: int) -> np.ndarray:
    """Equal-count resolution edges on the fit set, spanning every reflection's ss."""
    n_sh = max(1, int(n_shells))
    qs = np.linspace(0.0, 1.0, n_sh + 1)
    edges = np.quantile(ss_fit, qs)
    return np.asarray(edges, dtype=np.float64)


def ml_i_bulk_solvent_fit(
    f_calc: PackedMiller,
    f_obs: PackedMiller,
    f_mask: Any,
    k_model: Any,
    k_mask: Optional[Any] = None,
    fit_mask: Optional[Any] = None,
    ss: Optional[Any] = None,
    epsilon: Optional[Any] = None,
    centric: Optional[Any] = None,
    sigma_wilson: Optional[Any] = None,
    nu: Optional[Any] = None,
    k_sol: Optional[Any] = None,
    b_sol: Optional[Any] = None,
    n_shells: Optional[Any] = None,
    smooth: Optional[Any] = 1.0,
    max_iter: Optional[Any] = 60,
) -> dict[str, Any]:
    """Fit a binned k_mask to the intensity NLL on ``fit_mask``.

    Returns the interpolated curve, a ``(k_sol, B_sol)`` caption of it, and ``accepted``:
    whether the fit beats the caller's ``k_mask`` on the fit set. ``stats`` carries mean
    −log p(Z) for both on the fit set and on the rest (each with its own profiled
    shells), plus the low/mid/high k_mask values of both curves.
    """
    import torch

    from phridge.contrib.intensity_ll import free_beta as _fb
    from phridge.contrib.intensity_ll.mli import log_likelihood_normal, log_likelihood_t, normalize
    from phridge.sfcalc.ops import _DEVICE, _np

    if not np.array_equal(_np(f_calc.hkl), _np(f_obs.hkl)):
        raise ValueError("f_calc and f_obs must be on identical hkl lists")
    n = len(f_obs.data)

    def _vec(value: Any, dtype: Any, default: Any, name: str) -> np.ndarray:
        if value is None:
            return np.full(n, default, dtype=dtype)
        arr = np.asarray(_np(value, dtype)).ravel()
        if arr.shape != (n,):
            raise ValueError(f"{name} must have {n} values, got shape {arr.shape}")
        return arr

    io = _vec(f_obs.data, np.float64, 0.0, "f_obs")
    sig = _vec(f_obs.sigmas, np.float64, 1.0, "sigmas")
    fc = _vec(f_calc.data, np.complex128, 0.0, "f_calc")
    fm = _vec(f_mask, np.complex128, 0.0, "f_mask")
    km = _vec(k_model, np.float64, 1.0, "k_model")
    k_ref = _vec(k_mask, np.float64, 0.0, "k_mask")
    ss_np = _vec(ss, np.float64, 0.0, "ss")
    eps = _vec(epsilon, np.float64, 1.0, "epsilon")
    cen = _vec(centric, bool, False, "centric")
    sw = _vec(sigma_wilson, np.float64, 1.0, "sigma_wilson")
    fit = _vec(fit_mask, bool, True, "fit_mask")
    nu_np: Optional[np.ndarray] = None
    if nu is not None:
        nu_arr = _vec(nu, np.float64, np.nan, "nu")
        if np.any(np.isfinite(nu_arr) & (nu_arr < 199.0)):
            nu_np = np.where(np.isfinite(nu_arr), nu_arr, 200.0)

    usable = (
        (sig > 0)
        & (eps > 0)
        & (sw > 0)
        & np.isfinite(io)
        & np.isfinite(sw)
        & np.isfinite(km)
        & np.isfinite(fc)
        & np.isfinite(fm)
    )
    sel_fit = fit & usable
    sel_rest = ~fit & usable
    n_fit = int(sel_fit.sum())
    if n_fit < 50:
        raise ValueError(f"only {n_fit} usable reflections in fit_mask")

    n_sh = int(n_shells) if n_shells else int(min(20, max(6, n_fit // 250)))
    n_sh = max(1, n_sh)
    edges = _bin_edges(ss_np[sel_fit], n_sh)
    # Span the whole list so held-out reflections land in a real bin.
    edges = edges.copy()
    edges[0] = min(float(edges[0]), float(ss_np.min())) - 1e-12
    edges[-1] = max(float(edges[-1]), float(ss_np.max())) + 1e-12
    edges = np.unique(edges)
    n_sh = int(edges.size - 1)
    shell = np.clip(np.digitize(ss_np, edges[1:-1], right=False), 0, n_sh - 1)
    centers_np = 0.5 * (edges[:-1] + edges[1:])
    lam = 0.0 if smooth is None else max(0.0, float(smooth))
    iters = int(max_iter) if max_iter else 60

    dev = _DEVICE["device"]
    dtype = torch.float32 if dev.startswith("mps") else torch.float64

    def _t(arr: np.ndarray, sel: np.ndarray) -> Any:
        return torch.as_tensor(np.ascontiguousarray(arr[sel]), dtype=dtype, device=dev)

    # Real and imaginary parts, never a complex tensor: MPS complex support is partial.
    def _pack(sel: np.ndarray) -> dict[str, Any]:
        a = km * fc
        m = km * fm
        return {
            "a_re": _t(a.real, sel),
            "a_im": _t(a.imag, sel),
            "m_re": _t(m.real, sel),
            "m_im": _t(m.imag, sel),
            "ss": _t(ss_np, sel),
            "k_ref": _t(k_ref, sel),
            "io": _t(io, sel),
            "sig": _t(sig, sel),
            "eps": _t(eps, sel),
            "sw": _t(sw, sel),
            "cen": torch.as_tensor(cen[sel], device=dev),
            "shell": torch.as_tensor(shell[sel], dtype=torch.long, device=dev),
            "nu": None if nu_np is None else _t(nu_np, sel),
            "n": int(sel.sum()),
        }

    def _f_abs(d: dict[str, Any], g: Any) -> Any:
        re = d["a_re"] + g * d["m_re"]
        im = d["a_im"] + g * d["m_im"]
        return torch.sqrt(re * re + im * im + 1e-30)

    def _nll_sum(d: dict[str, Any], f_abs: Any, u_sa: Any, v_b: Any) -> Any:
        sa = (_fb.SIGMA_A_LO + _fb.SIGMA_A_SPAN * torch.sigmoid(u_sa))[d["shell"]]
        beta = (_fb.BETA_LO + _fb.BETA_SPAN * torch.sigmoid(v_b))[d["shell"]]
        ec, sa_n, zo, sz = normalize(f_abs, d["io"], d["sig"], d["eps"], d["sw"], sa)
        ec, sa_n = _fb.rice_inputs(ec, sa, beta)
        if d["nu"] is not None:
            ll = log_likelihood_t(ec, sa_n, zo, sz, d["cen"], nu=d["nu"], n_u=10)
        else:
            ll = log_likelihood_normal(ec, sa_n, zo, sz, d["cen"])
        return -ll.sum()

    def _penalty(u_sa: Any, v_b: Any, u_k: Optional[Any] = None) -> Any:
        if lam <= 0.0:
            return u_sa.sum() * 0.0
        pen = _fb.second_difference_penalty(u_sa) + _fb.second_difference_penalty(v_b)
        if u_k is not None:
            pen = pen + _fb.second_difference_penalty(u_k)
        return lam * pen

    centers_t = torch.as_tensor(centers_np, dtype=dtype, device=dev)

    def _interp_k(ss_t: Any, values: Any) -> Any:
        if values.numel() == 1:
            return values.expand_as(ss_t)
        ss_c = torch.clamp(ss_t, centers_t[0], centers_t[-1])
        idx = torch.bucketize(ss_c, centers_t, right=True)
        idx = torch.clamp(idx, 1, int(centers_t.numel()) - 1)
        left = idx - 1
        c0, c1 = centers_t[left], centers_t[idx]
        w = (ss_c - c0) / (c1 - c0).clamp_min(1e-30)
        return (1.0 - w) * values[left] + w * values[idx]

    failures: list[str] = []

    def _run(params: list[Any], loss_fn: Any, label: str) -> None:
        opt = torch.optim.LBFGS(params, max_iter=iters, line_search_fn="strong_wolfe")

        def closure() -> Any:
            opt.zero_grad()
            loss = loss_fn()
            loss.backward()
            return loss

        try:
            opt.step(closure)
        except Exception as exc:
            failures.append(f"{label}: {type(exc).__name__}: {exc}")

    def _shell_start(d_np_sel: np.ndarray, g_np: np.ndarray) -> tuple[Any, Any]:
        f = np.abs(km * (fc + g_np * fm))[d_np_sel]
        denom = np.maximum(eps[d_np_sel] * sw[d_np_sel], 1e-30)
        sa0, b0 = _fb.moment_init_shells(
            z_obs=io[d_np_sel] / denom,
            e_c_sq=f**2 / denom,
            sigma_z=sig[d_np_sel] / denom,
            shell_idx=shell[d_np_sel],
            n_shells=n_sh,
        )
        u = torch.as_tensor(_fb.logit_from_sigma_a(sa0), dtype=dtype, device=dev)
        v = torch.as_tensor(_fb.logit_from_beta(b0), dtype=dtype, device=dev)
        return u.clone().requires_grad_(True), v.clone().requires_grad_(True)

    def _profiled_ref(d: dict[str, Any], sel: np.ndarray, label: str) -> tuple[float, float]:
        """(penalized objective, mean −log p(Z)) of the caller's k_mask, shells fitted."""
        u, v = _shell_start(sel, k_ref)
        f = _f_abs(d, d["k_ref"]).detach()
        _run([u, v], lambda: _nll_sum(d, f, u, v) + _penalty(u, v), label)
        with torch.no_grad():
            nll = float(_nll_sum(d, f, u, v).item())
            obj = nll + float(_penalty(u, v).item())
        return obj, nll / max(d["n"], 1)

    k_bin = np.array(
        [
            float(np.mean(k_ref[sel_fit & (shell == i)]))
            if np.any(sel_fit & (shell == i))
            else 0.0
            for i in range(n_sh)
        ],
        dtype=np.float64,
    )
    k_bin = np.clip(k_bin, 0.02 * K_MASK_MAX, 0.98 * K_MASK_MAX)
    # k_sol / b_sol are unused: the start is the LS curve, not a two-parameter pair.
    del k_sol, b_sol

    d_fit = _pack(sel_fit)
    ref_obj, ref_fit = _profiled_ref(d_fit, sel_fit, "reference shells")

    u_k = torch.tensor(
        [_logit(float(k) / K_MASK_MAX) for k in k_bin],
        dtype=dtype,
        device=dev,
        requires_grad=True,
    )
    u_sa, v_b = _shell_start(sel_fit, k_ref)

    def _k_vals() -> Any:
        return K_MASK_MAX * torch.sigmoid(u_k)

    def _joint() -> Any:
        g = _interp_k(d_fit["ss"], _k_vals())
        f = _f_abs(d_fit, g)
        return _nll_sum(d_fit, f, u_sa, v_b) + _penalty(u_sa, v_b, u_k)

    _run([u_k, u_sa, v_b], _joint, "k_mask")
    with torch.no_grad():
        k_vals = _k_vals()
        k_vals_np = np.asarray(_np(k_vals), dtype=np.float64).ravel()
        g_fit_t = _interp_k(d_fit["ss"], k_vals)
        new_fit_sum = float(_nll_sum(d_fit, _f_abs(d_fit, g_fit_t), u_sa, v_b).item())
        new_obj = new_fit_sum + float(_penalty(u_sa, v_b, u_k).item())
    k_mask_new = np.interp(ss_np, centers_np, k_vals_np)
    new_fit = new_fit_sum / max(n_fit, 1)
    accepted = bool(np.isfinite(new_obj) and new_obj < ref_obj)
    ks_fit, bs_fit = _k_sol_b_sol_from_curve(ss_np, k_mask_new)

    stats: dict[str, Any] = {
        "n_fit": n_fit,
        "n_shells": int(n_sh),
        "n_bins": int(n_sh),
        "k_ls": _k_at_percentiles(ss_np, k_ref, sel_fit),
        "k_new": _k_at_percentiles(ss_np, k_mask_new, sel_fit),
        "k_sol_start": float(k_bin[0]) if k_bin.size else None,
        "b_sol_start": None,
        "nll_fit_ref": ref_fit,
        "nll_fit_new": new_fit,
        "lbfgs_failures": failures,
    }
    # Held-out check with shells fitted on those reflections themselves: k_mask is
    # the only thing carried over, so this is what the curve buys there.
    n_rest = int(sel_rest.sum())
    if n_rest >= 50:
        d_rest = _pack(sel_rest)
        _, rest_ref = _profiled_ref(d_rest, sel_rest, "held-out reference shells")
        u_r, v_r = _shell_start(sel_rest, k_mask_new)
        f_rest = _f_abs(
            d_rest, torch.as_tensor(k_mask_new[sel_rest], dtype=dtype, device=dev)
        )
        _run(
            [u_r, v_r],
            lambda: _nll_sum(d_rest, f_rest, u_r, v_r) + _penalty(u_r, v_r),
            "held-out shells",
        )
        with torch.no_grad():
            rest_new = float(_nll_sum(d_rest, f_rest, u_r, v_r).item()) / max(n_rest, 1)
        stats.update({"n_rest": n_rest, "nll_rest_ref": rest_ref, "nll_rest_new": rest_new})

    return {
        "k_sol": ks_fit,
        "b_sol": bs_fit,
        "k_mask": k_mask_new,
        "accepted": accepted,
        "stats": stats,
    }


ml_i_bulk_solvent_fit.compute_dtype = "float64"  # type: ignore[attr-defined]
