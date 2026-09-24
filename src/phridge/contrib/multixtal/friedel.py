"""Friedel convention for anomalous structure factors.

With ``S`` and ``G`` the transforms of the real and imaginary densities:

    F(+h) = S + i G
    conj(F(-h)) = S - i G
    I+ - I- = -4 Im(conj(S) G)
    I+ + I- = 2 (|S|^2 + |G|^2)

``S`` and ``G`` are recovered from a pair of evaluations, or from a real-density
pass (``fdp = 0``) plus the full ``F``:

    S = F(fdp=0)
    G = -i (F_full - S)
"""

from __future__ import annotations

from typing import Any


def split_s_g(f_plus: Any, f_minus: Any) -> tuple[Any, Any]:
    """``S = (F+ + conj(F-)) / 2``, ``G = (F+ - conj(F-)) / (2i)``."""
    s = 0.5 * (f_plus + f_minus.conj())
    g = (f_plus - f_minus.conj()) / (2.0j)
    return s, g


def s_g_from_real_and_full(f_real: Any, f_full: Any) -> tuple[Any, Any]:
    """Split using a real-density evaluation and the full (anomalous) ``F``."""
    return f_real, -1.0j * (f_full - f_real)


def reconstruct_f(s: Any, g: Any) -> tuple[Any, Any]:
    """Return ``(F(+h), F(-h))`` from ``S`` and ``G`` at ``+h``."""
    f_plus = s + 1.0j * g
    f_minus = (s - 1.0j * g).conj()
    return f_plus, f_minus


def bijvoet_intensity_difference(s: Any, g: Any) -> Any:
    """``I+ - I- = -4 Im(conj(S) G)``."""
    if hasattr(s, "real"):
        return -4.0 * (s.conj() * g).imag
    import numpy as np

    return -4.0 * np.asarray(np.conj(s) * g).imag


def bijvoet_intensity_sum(s: Any, g: Any) -> Any:
    """``I+ + I- = 2 (|S|^2 + |G|^2)``."""
    if hasattr(s, "abs"):
        return 2.0 * (s.abs() ** 2 + g.abs() ** 2)
    import numpy as np

    s_arr = np.asarray(s)
    g_arr = np.asarray(g)
    return 2.0 * (np.abs(s_arr) ** 2 + np.abs(g_arr) ** 2)
