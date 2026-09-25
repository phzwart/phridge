"""Thin FINUFFT wrapper with fixed crystallographic conventions.

Type-1 computes ``Σ_j w_tj exp(+2πi h·x_j)`` for ``h`` on the FFT-order
mode grid (``modeord=1``: mode ``h`` at index ``h % n``). Type-2 is the
adjoint (opposite ``isign``). The engine never calls the library API.

CPU backward is implemented as a pair of ``torch.autograd.Function``s so
type-1 and type-2 are each other's adjoints and ``create_graph`` (JVP via
double VJP) works. ``pytorch_finufft``'s own CPU backward goes through
numpy and is first-order only.
"""

from __future__ import annotations

import math
import os
from typing import Sequence

# Torch and FINUFFT each ship an OpenMP runtime; on macOS the second init aborts.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_NUFFT_OK: bool | None = None


def nufft_available() -> bool:
    """True if ``pytorch_finufft`` (and thus ``finufft``) can be imported."""
    global _NUFFT_OK
    if _NUFFT_OK is not None:
        return _NUFFT_OK
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        import finufft  # noqa: F401
        import pytorch_finufft.functional  # noqa: F401

        _NUFFT_OK = True
    except Exception:
        _NUFFT_OK = False
    return _NUFFT_OK


def _require_nufft() -> None:
    if not nufft_available():
        raise RuntimeError(
            "pytorch-finufft is required for the NUFFT structure-factor engine; "
            "install with pip install 'phridge[nufft]'"
        )


def _frac_to_finufft_points(x_frac):
    """Fractional sites ``(M, 3)`` → FINUFFT points ``(3, M)`` in ``[-π, π)``."""
    import torch

    frac = x_frac - torch.floor(x_frac)
    twopi = 2.0 * math.pi
    pts = twopi * frac
    pts = torch.where(pts >= math.pi, pts - twopi, pts)
    return pts.transpose(0, 1).contiguous()


def _maybe_move(tensor, device):
    if tensor.device != device:
        return tensor.to(device)
    return tensor


def _raw_type1(points, values, n_modes: tuple[int, int, int], eps: float):
    """FINUFFT type-1, ``isign=+1``, ``modeord=1``. No autograd."""
    import numpy as np
    import torch

    if points.device.type == "cuda":
        import pytorch_finufft.functional as pfinufft

        with torch.no_grad():
            return pfinufft.finufft_type1(points, values, n_modes, eps=float(eps), isign=1, modeord=1)

    import finufft

    pts = points.detach().cpu().contiguous().numpy()
    val = values.detach().resolve_conj().cpu().contiguous().numpy()
    out = finufft.nufft3d1(pts[0], pts[1], pts[2], val, n_modes, eps=float(eps), isign=1, modeord=1, nthreads=1)
    return torch.as_tensor(np.ascontiguousarray(out), dtype=values.dtype, device=values.device)


def _raw_type2(points, grid, eps: float):
    """FINUFFT type-2, ``isign=-1``, ``modeord=1``. No autograd."""
    import numpy as np
    import torch

    if points.device.type == "cuda":
        import pytorch_finufft.functional as pfinufft

        with torch.no_grad():
            return pfinufft.finufft_type2(points, grid, eps=float(eps), isign=-1, modeord=1)

    import finufft

    pts = points.detach().cpu().contiguous().numpy()
    g = grid.detach().resolve_conj().cpu().contiguous().numpy()
    out = finufft.nufft3d2(pts[0], pts[1], pts[2], g, eps=float(eps), isign=-1, modeord=1, nthreads=1)
    return torch.as_tensor(np.ascontiguousarray(out), dtype=grid.dtype, device=grid.device)


def _mode_axes(n_modes, device, dtype):
    import torch

    axes = []
    for n in n_modes:
        i = torch.arange(int(n), device=device)
        axes.append(torch.where(i < (int(n) + 1) // 2, i, i - int(n)).to(dtype))
    return torch.meshgrid(*axes, indexing="ij")


class _Type1Fn:
    """Lazy wrapper so torch is imported only when the Function is built."""

    _cls = None

    @classmethod
    def cls(cls):
        if cls._cls is not None:
            return cls._cls
        import torch

        class Type1(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x_frac, weights, n1, n2, n3, eps):
                modes = (int(n1), int(n2), int(n3))
                points = _frac_to_finufft_points(x_frac)
                ctx.save_for_backward(x_frac, weights)
                ctx.n_modes = modes
                ctx.eps = float(eps)
                return _raw_type1(points, weights, modes, ctx.eps)

            @staticmethod
            def backward(ctx, grad_grid):
                x_frac, weights = ctx.saved_tensors
                grad_w = _Type2Fn.cls().apply(x_frac, grad_grid, ctx.eps) if ctx.needs_input_grad[1] else None
                grad_x = None
                if ctx.needs_input_grad[0]:
                    h, k, l = _mode_axes(ctx.n_modes, grad_grid.device, x_frac.dtype)
                    # pytorch-finufft type-1 isign=+1: ramped = k * grad * 1j * (-1)
                    ramps = torch.stack([h, k, l], dim=0)
                    ramped = grad_grid.unsqueeze(1) * ramps.to(grad_grid.dtype) * (-1j)
                    t, _n1, _n2, _n3 = grad_grid.shape
                    ramped = ramped.reshape(t * 3, _n1, _n2, _n3)
                    x_rep = x_frac
                    back = _Type2Fn.cls().apply(x_rep, ramped, ctx.eps).reshape(t, 3, x_frac.shape[0]).conj()
                    grads_pts = (back * weights.unsqueeze(1)).real.sum(0)
                    grad_x = grads_pts.transpose(0, 1).contiguous() * (2.0 * math.pi)
                return grad_x, grad_w, None, None, None, None

        cls._cls = Type1
        return cls._cls


class _Type2Fn:
    _cls = None

    @classmethod
    def cls(cls):
        if cls._cls is not None:
            return cls._cls
        import torch

        class Type2(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x_frac, grid, eps):
                points = _frac_to_finufft_points(x_frac)
                ctx.save_for_backward(x_frac, grid)
                ctx.eps = float(eps)
                return _raw_type2(points, grid, ctx.eps)

            @staticmethod
            def backward(ctx, grad_values):
                x_frac, grid = ctx.saved_tensors
                n1, n2, n3 = (int(x) for x in grid.shape[-3:])
                grad_grid = (
                    _Type1Fn.cls().apply(x_frac, grad_values, n1, n2, n3, ctx.eps)
                    if ctx.needs_input_grad[1]
                    else None
                )
                grad_x = None
                if ctx.needs_input_grad[0]:
                    h, k, l = _mode_axes((n1, n2, n3), grid.device, x_frac.dtype)
                    ramps = torch.stack([h, k, l], dim=0)
                    # type-2 isign=-1: opposite ramp sign vs type-1
                    ramped = grid.unsqueeze(1) * ramps.to(grid.dtype) * (1j)
                    t = grid.shape[0]
                    ramped = ramped.reshape(t * 3, n1, n2, n3)
                    back = _Type2Fn.cls().apply(x_frac, ramped, ctx.eps).reshape(t, 3, x_frac.shape[0]).conj()
                    grads_pts = (back * grad_values.unsqueeze(1)).real.sum(0)
                    grad_x = grads_pts.transpose(0, 1).contiguous() * (2.0 * math.pi)
                return grad_x, grad_grid, None

        cls._cls = Type2
        return cls._cls


def nufft_type1(x_frac, weights, n_modes: Sequence[int], eps: float):
    """Type-1 NUFFT: ``Σ_j w_tj exp(+2πi h·x_j)``.

    Parameters
    ----------
    x_frac:
        Fractional coordinates, shape ``(M, 3)``.
    weights:
        Complex strengths, shape ``(T, M)`` (or ``(M,)`` for a single transform).
    n_modes:
        Mode counts ``(n1, n2, n3)``.
    eps:
        FINUFFT relative tolerance.

    Returns
    -------
    Complex grid of shape ``(T, n1, n2, n3)`` (or ``(n1, n2, n3)`` if
    ``weights`` was 1-D).
    """
    _require_nufft()
    out_device = x_frac.device
    if str(out_device).startswith("mps"):
        x_frac = x_frac.cpu()
        weights = weights.cpu()
    squeezed = weights.ndim == 1
    if squeezed:
        weights = weights[None, :]
    n1, n2, n3 = (int(n) for n in n_modes)
    grid = _Type1Fn.cls().apply(x_frac, weights, n1, n2, n3, float(eps))
    if squeezed:
        grid = grid[0]
    return _maybe_move(grid, out_device)


def nufft_type2(x_frac, grid, eps: float):
    """Type-2 NUFFT: the adjoint of :func:`nufft_type1` (``isign=-1``).

    Parameters
    ----------
    x_frac:
        Fractional coordinates, shape ``(M, 3)``.
    grid:
        Complex Fourier modes, shape ``(T, n1, n2, n3)`` (or ``(n1, n2, n3)``).
    eps:
        FINUFFT relative tolerance.

    Returns
    -------
    Complex strengths of shape ``(T, M)`` (or ``(M,)`` if ``grid`` was 3-D).
    """
    _require_nufft()
    out_device = x_frac.device
    if str(out_device).startswith("mps"):
        x_frac = x_frac.cpu()
        grid = grid.cpu()
    squeezed = grid.ndim == 3
    if squeezed:
        grid = grid[None, ...]
    values = _Type2Fn.cls().apply(x_frac, grid, float(eps))
    if squeezed:
        values = values[0]
    return _maybe_move(values, out_device)
