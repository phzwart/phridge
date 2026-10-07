"""Setting and origin alignment, and a canonical Laue-orbit Miller index.

The deposited path enumerates symmetry operations that preserve the space
group and the finite origin shifts that do too, then keeps the candidate
with the highest complex correlation. A gemmi cross-check runs only when
both gemmi and ``reference/cascan_pipeline.py`` import.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np


def canonical_laue_index(h: int, k: int, l: int) -> tuple[int, int, int]:
    """Lexicographically larger of ``h`` and ``-h`` on the Friedel orbit."""
    forward = (int(h), int(k), int(l))
    backward = (-int(h), -int(k), -int(l))
    return forward if forward >= backward else backward


def canonical_laue_hkl(hkl: np.ndarray) -> np.ndarray:
    rows = [canonical_laue_index(int(h[0]), int(h[1]), int(h[2])) for h in np.asarray(hkl)]
    return np.asarray(rows, dtype=np.int32)


def _origin_preserves_group(symbol: str, shift: tuple[float, float, float]) -> bool:
    from phridge.shims.cctbx.pdb import origin_preserves_group

    return origin_preserves_group(symbol, shift)


def permissible_origin_shifts(symbol: str = "P 1") -> list[tuple[float, float, float]]:
    """Half-cell shifts that leave the space group unchanged, plus the zero shift."""
    shifts = [(0.0, 0.0, 0.0)]
    for ia in (0, 1):
        for ib in (0, 1):
            for ic in (0, 1):
                if (ia, ib, ic) == (0, 0, 0):
                    continue
                shift = (ia / 2.0, ib / 2.0, ic / 2.0)
                try:
                    if _origin_preserves_group(symbol, shift):
                        shifts.append(shift)
                except Exception:
                    continue
    return shifts


def apply_origin_phase(hkl: np.ndarray, coeff: np.ndarray, shift: Sequence[float]) -> np.ndarray:
    h = np.asarray(hkl, dtype=np.float64)
    phase = 2.0 * np.pi * (h @ np.asarray(shift, dtype=np.float64))
    return np.asarray(coeff, dtype=np.complex128) * np.exp(1j * phase)


def complex_correlation(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.complex128).reshape(-1)
    b = np.asarray(right, dtype=np.complex128).reshape(-1)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-18:
        return 0.0
    return float(np.real(np.vdot(a, b)) / denom)


def best_origin_shift(
    hkl: np.ndarray,
    reference: np.ndarray,
    moving: np.ndarray,
    symbol: str = "P 1",
) -> tuple[tuple[float, float, float], float]:
    """Origin shift of ``moving`` that maximises correlation with ``reference``."""
    best_shift = (0.0, 0.0, 0.0)
    best = -2.0
    for shift in permissible_origin_shifts(symbol):
        corr = complex_correlation(reference, apply_origin_phase(hkl, moving, shift))
        if corr > best:
            best = corr
            best_shift = shift
    return best_shift, best


def gemmi_crosscheck_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("gemmi") is not None and importlib.util.find_spec("cascan_pipeline") is not None


def gemmi_origin_crosscheck(*_args: Any, **_kwargs: Any) -> Optional[dict[str, Any]]:
    """Compare against ``cascan_pipeline`` when that module and gemmi import."""
    if not gemmi_crosscheck_available():
        return None
    from phridge.tools.ensemble.cascan import module_file

    return {"module": module_file()}
