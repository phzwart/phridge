"""Per-atom Luzzati weights and the modified model (spec eqs 13–14, 25).

λ̄ and U_0 are absorbed into the shell table D_0(s). The fields λ, κ are
mean-zero, so w_j = exp(λ_j) and U_j = (κ_j / 8π²) I.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from phridge.sfcalc.engine.cell import orthogonalization_matrix
from phridge.sfcalc.engine.engine import ScatteringModel

EIGHT_PI2 = 8.0 * math.pi * math.pi
TWO_PI2 = 2.0 * math.pi * math.pi


def presence_weight(lambda_j: Any) -> Any:
    """w_j = exp(λ_j). Spec eq. 13 with λ̄ absorbed into D_0."""
    try:
        import torch

        if isinstance(lambda_j, torch.Tensor):
            return torch.exp(lambda_j)
    except ImportError:
        pass
    return np.exp(np.asarray(lambda_j, dtype=np.float64))


def error_u_iso(kappa_j: Any) -> Any:
    """Isotropic U_j (Å²) from excess error-B κ. Spec eq. 13, U_0 absorbed."""
    try:
        import torch

        if isinstance(kappa_j, torch.Tensor):
            return kappa_j / EIGHT_PI2
    except ImportError:
        pass
    return np.asarray(kappa_j, dtype=np.float64) / EIGHT_PI2


def luzzati_d_iso(u_iso: Any, s2: Any) -> Any:
    """D_j = exp(−2π² U s²) = exp(−κ s²/4) when U = κ/8π². Spec eq. 2, isotropic."""
    try:
        import torch

        if isinstance(u_iso, torch.Tensor) or isinstance(s2, torch.Tensor):
            def _as_like(x: Any, dtype: Any, device: Any) -> Any:
                widen = dtype in (torch.float64, torch.complex128)
                if str(device).startswith("mps") and widen:
                    dtype = torch.float32 if dtype == torch.float64 else torch.complex64
                    widen = False
                if isinstance(x, torch.Tensor):
                    if x.device.type == "mps" and widen:
                        x = x.detach().cpu()
                    if x.dtype != dtype:
                        x = x.to(dtype=dtype)
                    if x.device != device:
                        x = x.to(device=device)
                    return x
                t = torch.as_tensor(np.asarray(x), dtype=dtype)
                return t.to(device=device) if t.device != device else t

            if isinstance(s2, torch.Tensor):
                u = _as_like(u_iso, s2.dtype, s2.device)
                ss = s2
            else:
                u = u_iso
                ss = _as_like(s2, u.dtype, u.device)
            return torch.exp(-TWO_PI2 * u.reshape(-1, 1) * ss.reshape(1, -1))
    except ImportError:
        pass
    u = np.asarray(u_iso, dtype=np.float64).reshape(-1, 1)
    ss = np.asarray(s2, dtype=np.float64).reshape(1, -1)
    return np.exp(-TWO_PI2 * u * ss)


def effective_weight(
    d0: Any,
    lambda_j: Any,
    kappa_j: Any,
    s2: Any,
) -> Any:
    """d_j(h) = D_0(s) exp(λ_j − κ_j s²/4). Spec eq. 14."""
    try:
        import torch

        tensors = [x for x in (d0, lambda_j, kappa_j, s2) if isinstance(x, torch.Tensor)]
        if tensors:
            dtype, device = tensors[0].dtype, tensors[0].device

            def _t(x: Any) -> Any:
                dt = dtype
                widen = dt in (torch.float64, torch.complex128)
                if str(device).startswith("mps") and widen:
                    dt = torch.float32 if dt == torch.float64 else torch.complex64
                    widen = False
                if isinstance(x, torch.Tensor):
                    if x.device.type == "mps" and widen:
                        x = x.detach().cpu()
                    if x.dtype != dt:
                        x = x.to(dtype=dt)
                    if x.device != device:
                        x = x.to(device=device)
                    return x
                t = torch.as_tensor(np.asarray(x), dtype=dt)
                return t.to(device=device) if t.device != device else t

            d0_t, lam, kap, ss = _t(d0), _t(lambda_j), _t(kappa_j), _t(s2)
            return d0_t.reshape(1, -1) * torch.exp(lam.reshape(-1, 1) - kap.reshape(-1, 1) * ss.reshape(1, -1) / 4.0)
    except ImportError:
        pass
    d0_n = np.asarray(d0, dtype=np.float64).reshape(1, -1)
    lam = np.asarray(lambda_j, dtype=np.float64).reshape(-1, 1)
    kap = np.asarray(kappa_j, dtype=np.float64).reshape(-1, 1)
    ss = np.asarray(s2, dtype=np.float64).reshape(1, -1)
    return d0_n * np.exp(lam - kap * ss / 4.0)


def iso_u_star(unit_cell: tuple, u_iso: np.ndarray) -> np.ndarray:
    """u_star for a Cartesian isotropic U = u_iso I."""
    o = orthogonalization_matrix(unit_cell)
    g_inv = np.linalg.inv(o.T @ o)  # O^{-1} O^{-T}
    u = np.asarray(u_iso, dtype=np.float64).reshape(-1)
    # cctbx order 11,22,33,12,13,23
    out = np.zeros((u.shape[0], 6), dtype=np.float64)
    out[:, 0] = u * g_inv[0, 0]
    out[:, 1] = u * g_inv[1, 1]
    out[:, 2] = u * g_inv[2, 2]
    out[:, 3] = u * g_inv[0, 1]
    out[:, 4] = u * g_inv[0, 2]
    out[:, 5] = u * g_inv[1, 2]
    return out


def modified_model(
    model: ScatteringModel,
    w: np.ndarray,
    u_err_iso: np.ndarray,
) -> ScatteringModel:
    """Occupancy w_j, ADP U_ADP + U_j. Spec eq. 25.

    Modelled occupancy is kept: occ_eff = occ * w_j. Isotropic error U_j is
    added to u_iso or, for anisotropic atoms, to u_star as an isotropic increment.
    """
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    u_err = np.asarray(u_err_iso, dtype=np.float64).reshape(-1)
    n = model.n_scatterers
    if w.shape[0] != n or u_err.shape[0] != n:
        raise ValueError("w / U_j length must match n_scatterers")
    occ = model.occupancy * w
    u_iso = model.u_iso.copy()
    u_star = model.u_star.copy()
    aniso = np.asarray(model.anisotropic, dtype=bool)
    u_iso[~aniso] = u_iso[~aniso] + u_err[~aniso]
    if np.any(aniso):
        u_star[aniso] = u_star[aniso] + iso_u_star(model.unit_cell, u_err[aniso])
    return ScatteringModel(
        unit_cell=model.unit_cell,
        sites_frac=np.asarray(model.sites_frac, dtype=np.float64).copy(),
        occupancy=occ,
        u_iso=u_iso,
        u_star=u_star,
        anisotropic=np.asarray(model.anisotropic, dtype=bool).copy(),
        fp=np.asarray(model.fp, dtype=np.float64).copy(),
        fdp=np.asarray(model.fdp, dtype=np.float64).copy(),
        type_index=np.asarray(model.type_index).copy(),
        gauss_a=model.gauss_a,
        gauss_b=model.gauss_b,
        gauss_c=model.gauss_c,
        rot=model.rot,
        trans=model.trans,
        multiplicity=None if model.multiplicity is None else np.asarray(model.multiplicity).copy(),
    )


def interpolate_shell(
    s2: np.ndarray,
    edges: np.ndarray,
    values: np.ndarray,
    default: float,
) -> np.ndarray:
    """Piecewise-constant shell table on s² = |d*|². Empty table → ``default``."""
    ss = np.asarray(s2, dtype=np.float64).reshape(-1)
    val = np.asarray(values, dtype=np.float64).reshape(-1)
    ed = np.asarray(edges, dtype=np.float64).reshape(-1)
    if val.size == 0:
        return np.full(ss.shape, float(default))
    if ed.size == val.size + 1:
        idx = np.clip(np.searchsorted(ed, ss, side="right") - 1, 0, val.size - 1)
        return val[idx]
    if val.size == 1:
        return np.full(ss.shape, float(val[0]))
    raise ValueError("shell table edges must have length n_shells + 1")


def d0_and_sigma_miss(
    s2: np.ndarray,
    d0_shell: Optional[np.ndarray],
    sigma_miss_shell: Optional[np.ndarray],
    edges: Optional[np.ndarray],
    *,
    d0_default: float = 1.0,
    miss_default: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    ed = np.asarray(edges, dtype=np.float64) if edges is not None else np.zeros((0,))
    d0 = interpolate_shell(s2, ed, np.asarray(d0_shell if d0_shell is not None else []), d0_default)
    miss = interpolate_shell(
        s2, ed, np.asarray(sigma_miss_shell if sigma_miss_shell is not None else []), miss_default
    )
    return d0, miss
