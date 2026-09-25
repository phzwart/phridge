"""Thin FINUFFT wrapper with fixed crystallographic conventions.

Type-1 computes ``Σ_j w_tj exp(+2πi h·x_j)`` for ``h`` on the FFT-order
mode grid (``modeord=1``: mode ``h`` at index ``h % n``). Type-2 is the
adjoint (opposite ``isign``). The engine never calls the library API.
"""

from __future__ import annotations

import math
import os
from typing import Sequence

# Torch and FINUFFT each ship an OpenMP runtime; on macOS the second init aborts.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_NUFFT_OK: bool | None = None


def nufft_available() -> bool:
    """True if ``pytorch_finufft`` can be imported."""
    global _NUFFT_OK
    if _NUFFT_OK is not None:
        return _NUFFT_OK
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
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


def _run_device(device) -> str:
    name = str(device)
    if name.startswith("cuda"):
        return "cuda"
    return "cpu"


def _maybe_move(tensor, device):
    if tensor.device != device:
        return tensor.to(device)
    return tensor


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
    import pytorch_finufft.functional as pfinufft

    out_device = x_frac.device
    if _run_device(out_device) == "cpu" and str(out_device) != "cpu":
        x_frac = x_frac.cpu()
        weights = weights.cpu()

    squeezed = weights.ndim == 1
    if squeezed:
        weights = weights[None, :]
    points = _frac_to_finufft_points(x_frac)
    modes = tuple(int(n) for n in n_modes)
    grid = pfinufft.finufft_type1(
        points, weights, modes, eps=float(eps), isign=1, modeord=1, nthreads=1
    )
    if squeezed:
        grid = grid[0]
    return _maybe_move(grid, out_device) if grid.device != out_device else grid


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
    import pytorch_finufft.functional as pfinufft

    out_device = x_frac.device
    if _run_device(out_device) == "cpu" and str(out_device) != "cpu":
        x_frac = x_frac.cpu()
        grid = grid.cpu()

    squeezed = grid.ndim == 3
    if squeezed:
        grid = grid[None, ...]
    points = _frac_to_finufft_points(x_frac)
    values = pfinufft.finufft_type2(points, grid, eps=float(eps), isign=-1, modeord=1, nthreads=1)
    if squeezed:
        values = values[0]
    return _maybe_move(values, out_device) if values.device != out_device else values
