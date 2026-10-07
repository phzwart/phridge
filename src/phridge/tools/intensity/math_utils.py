"""Mathematical and optimization utilities for intensity modeling."""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
import torch


def _fom_rice_woolfson(
    Ec: np.ndarray,
    sA: np.ndarray,
    Zo: np.ndarray,
    centric: np.ndarray,
) -> np.ndarray:
    """Figure of merit m = <cos(phi - phi_c)> under the Rice/Woolfson conditional distribution.

    For acentric: m = I_1(2X) / I_0(2X) where X = sA * Ec * sqrt(max(Zo, 0)) / (1 - sA^2).
    For centric:  m = tanh(X).
    """
    sA_clamp = np.clip(sA, 1e-4, 0.999)
    denom = np.maximum(1.0 - sA_clamp**2, 1e-6)
    zo_pos = np.maximum(Zo, 0.0)
    X = sA_clamp * Ec * np.sqrt(zo_pos) / denom
    X = np.clip(X, 0.0, 50.0)  # avoid overflow in exp/cosh

    # Centric: tanh(X)
    cen_m = np.tanh(X)

    # Acentric: I_1(2X) / I_0(2X)
    two_X = 2.0 * X
    # Ratio using exponentially scaled Bessel functions i1e(y)/i0e(y) via PyTorch
    two_X_t = torch.as_tensor(two_X, dtype=torch.float64)
    acen_m = (torch.special.i1e(two_X_t) / torch.special.i0e(two_X_t).clamp_min(1e-300)).cpu().numpy()

    fom = np.where(centric, cen_m, acen_m)
    return np.clip(fom, 0.0, 1.0)


def golden_section_search_torch(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-2,
    max_iter: int = 20,
) -> float:
    """1D bounded scalar minimization using Golden Section Search in PyTorch."""
    invphi = (math.sqrt(5) - 1.0) / 2.0
    invphi2 = (3.0 - math.sqrt(5)) / 2.0
    c = a + invphi2 * (b - a)
    d = a + invphi * (b - a)
    yc = f(c)
    yd = f(d)
    for _ in range(max_iter):
        if (b - a) < tol:
            break
        if yc < yd:
            b = d
            d = c
            yd = yc
            c = a + invphi2 * (b - a)
            yc = f(c)
        else:
            a = c
            c = d
            yc = yd
            d = a + invphi * (b - a)
            yd = f(d)
    return float((a + b) / 2.0)


def tv_denoise_1d(
    y: np.ndarray,
    weights: Optional[np.ndarray] = None,
    lam: float = 0.05,
    lower: float = 0.01,
    upper: float = 0.999,
) -> np.ndarray:
    """1D Total Variation Denoising (TVD) using Huber-smoothed L1 difference penalty in PyTorch.

    Solves:
        min_{x in [lower, upper]^n} 0.5 * sum_i w_i (x_i - y_i)^2 + lam * sum_i sqrt((x_{i+1} - x_i)^2 + eps^2)
    """
    y_np = np.asarray(y, dtype=np.float64)
    n = len(y_np)
    if lam <= 0.0 or n <= 1:
        return np.clip(y_np, lower, upper)
    w_np = np.ones(n, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    w_np = w_np / max(float(np.mean(w_np)), 1e-12)

    y_t = torch.tensor(y_np, dtype=torch.float64)
    w_t = torch.tensor(w_np, dtype=torch.float64)

    span = upper - lower
    y_clip = np.clip(y_np, lower + 1e-5, upper - 1e-5)
    u_init = np.log((y_clip - lower) / (upper - y_clip))
    u = torch.tensor(u_init, dtype=torch.float64, requires_grad=True)

    opt = torch.optim.LBFGS([u], lr=1.0, max_iter=25, line_search_fn="strong_wolfe")
    eps = 1e-6

    def closure():
        opt.zero_grad()
        x = lower + span * torch.sigmoid(u)
        diffs = x[1:] - x[:-1]
        loss = 0.5 * (w_t * (x - y_t) ** 2).sum() + lam * torch.sqrt(diffs**2 + eps).sum()
        loss.backward()
        return loss

    opt.step(closure)
    x_opt = lower + span * torch.sigmoid(u)
    return np.asarray(x_opt.detach().cpu().numpy(), dtype=np.float64)


def isotonic_non_increasing(
    y: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Weighted isotonic regression enforcing non-increasing order: x_0 >= x_1 >= ... >= x_{n-1}
    using the Pool Adjacent Violators Algorithm (PAVA).
    """
    n = len(y)
    if n <= 1:
        return np.asarray(y, dtype=np.float64)
    w = [float(x) for x in (weights if weights is not None else np.ones(n))]
    v = [float(x) for x in y]
    blocks = [[i] for i in range(n)]

    i = 0
    while i < len(blocks) - 1:
        if v[i] < v[i + 1]:  # Violation of non-increasing order
            w_new = w[i] + w[i + 1]
            v_new = (w[i] * v[i] + w[i + 1] * v[i + 1]) / w_new
            w[i] = w_new
            v[i] = v_new
            blocks[i].extend(blocks[i + 1])
            del w[i + 1]
            del v[i + 1]
            del blocks[i + 1]
            if i > 0:
                i -= 1
        else:
            i += 1

    out = np.zeros(n, dtype=np.float64)
    for block_idx, block in enumerate(blocks):
        for idx in block:
            out[idx] = v[block_idx]
    return out
