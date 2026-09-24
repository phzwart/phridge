"""Per-dataset F_calc and G_calc via the sfcalc FFT engine.

``F_mask`` is computed on the cctbx client (``dt_mask.build_f_masks``) and shipped
in; the worker never imports cctbx. ``k_sol`` / ``B_sol`` are applied here as the
classical two-parameter solvent, not the binned ``k_mask`` of ``ml_i_bulk_solvent_fit``.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from phridge.contrib.multixtal.friedel import s_g_from_real_and_full
from phridge.sfcalc.engine.engine import EngineParams, ScatteringModel, StructureFactorEngine


def model_with_cell(model: ScatteringModel, unit_cell: Sequence[float], sites_frac: np.ndarray) -> ScatteringModel:
    """Clone a scattering model onto another cell / fractional sites."""
    return ScatteringModel(
        unit_cell=tuple(float(x) for x in unit_cell),
        sites_frac=np.asarray(sites_frac, dtype=np.float64).reshape(-1, 3),
        occupancy=np.asarray(model.occupancy, dtype=np.float64),
        u_iso=np.asarray(model.u_iso, dtype=np.float64),
        u_star=np.asarray(model.u_star, dtype=np.float64).reshape(-1, 6),
        anisotropic=np.asarray(model.anisotropic, dtype=bool),
        fp=np.asarray(model.fp, dtype=np.float64),
        fdp=np.asarray(model.fdp, dtype=np.float64),
        type_index=np.asarray(model.type_index, dtype=np.int64),
        gauss_a=np.asarray(model.gauss_a, dtype=np.float64),
        gauss_b=np.asarray(model.gauss_b, dtype=np.float64),
        gauss_c=np.asarray(model.gauss_c, dtype=np.float64),
        rot=np.asarray(model.rot, dtype=np.float64),
        trans=np.asarray(model.trans, dtype=np.float64),
    )


def engine_params(d_min: float, dtype: str = "float64") -> EngineParams:
    return EngineParams(d_min=float(d_min), dtype=dtype)


def f_calc_s_g(
    model: ScatteringModel,
    hkl: np.ndarray,
    params: Optional[EngineParams] = None,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(F_full, S, G)`` at ``hkl`` (typically the ASU ``+h`` list)."""
    import torch

    hkl = np.asarray(hkl, dtype=np.int64).reshape(-1, 3)
    if params is None:
        s_sq = _s_sq(model.unit_cell, hkl)
        d_min = float(1.0 / np.sqrt(np.max(s_sq))) if s_sq.size else 2.0
        params = engine_params(d_min * 0.95)
    eng = StructureFactorEngine(model, hkl, params, device=device)
    sites, occ, u_iso, u_star, fp, fdp = eng.tensors()
    with torch.no_grad():
        f_full_t = eng.f_calc(sites, occ, u_iso, u_star, fp, fdp)
        f_real_t = eng.f_calc(sites, occ, u_iso, u_star, fp, torch.zeros_like(fdp))
    f_full = f_full_t.detach().cpu().numpy().astype(np.complex128)
    f_real = f_real_t.detach().cpu().numpy().astype(np.complex128)
    s, g = s_g_from_real_and_full(f_real, f_full)
    return f_full, np.asarray(s, dtype=np.complex128), np.asarray(g, dtype=np.complex128)


def apply_bulk_solvent(
    f_calc: Any,
    f_mask: Any,
    s_sq: Any,
    k_sol: Any,
    b_sol: Any,
) -> Any:
    """``Fc = Fcalc + k_sol exp(-B_sol s^2 / 4) Fmask``."""
    if hasattr(s_sq, "exp"):
        return f_calc + k_sol * (-b_sol * s_sq / 4.0).exp() * f_mask
    return f_calc + k_sol * np.exp(-np.asarray(b_sol) * np.asarray(s_sq) / 4.0) * f_mask


def _s_sq(unit_cell: Sequence[float], hkl: np.ndarray) -> np.ndarray:
    from phridge.contrib.intensity_ll.wilson import reciprocal_cartesian

    s_cart = reciprocal_cartesian(unit_cell, hkl)
    return np.sum(np.asarray(s_cart, dtype=np.float64) ** 2, axis=1)


def resolution_s2(unit_cell: Sequence[float], hkl: np.ndarray) -> np.ndarray:
    return _s_sq(unit_cell, hkl)


def s_hat(unit_cell: Sequence[float], hkl: np.ndarray) -> np.ndarray:
    from phridge.contrib.intensity_ll.aniso_rice import unit_directions
    from phridge.contrib.intensity_ll.wilson import reciprocal_cartesian

    return unit_directions(reciprocal_cartesian(unit_cell, hkl))
