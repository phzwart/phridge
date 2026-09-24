"""Alternative indexing vs the model space group. Torch-free; cctbx only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class IndexingChoice:
    operator: str
    correlation: float
    margin: float
    n_tried: int


def _shell_mean_corr(i_obs: np.ndarray, i_calc: np.ndarray, s_sq: np.ndarray, n_shells: int = 8) -> float:
    i_o = np.asarray(i_obs, dtype=np.float64)
    i_c = np.asarray(i_calc, dtype=np.float64)
    ss = np.asarray(s_sq, dtype=np.float64)
    ok = np.isfinite(i_o) & np.isfinite(i_c)
    if int(ok.sum()) < 8:
        return float("nan")
    n_shells = max(1, min(int(n_shells), int(ok.sum()) // 4))
    order = np.argsort(ss[ok], kind="stable")
    vals_o = i_o[ok][order]
    vals_c = i_c[ok][order]
    n = vals_o.size
    corrs: list[float] = []
    for k in range(n_shells):
        lo = (k * n) // n_shells
        hi = ((k + 1) * n) // n_shells
        if hi - lo < 4:
            continue
        a, b = vals_o[lo:hi], vals_c[lo:hi]
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            continue
        corrs.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(corrs)) if corrs else float("nan")


def reindexing_operators(data_symmetry: Any, model_symmetry: Any, max_delta: float = 3.0) -> list[Any]:
    """Lattice-compatible change-of-basis operators taking data indices toward the model."""
    from cctbx import sgtbx
    from cctbx.sgtbx import lattice_symmetry

    cb_data = data_symmetry.change_of_basis_op_to_niggli_cell()
    cb_model = model_symmetry.change_of_basis_op_to_niggli_cell()
    niggli_model = model_symmetry.change_basis(cb_model)
    group = lattice_symmetry.group(niggli_model.unit_cell(), max_delta=float(max_delta))
    identity = sgtbx.change_of_basis_op()
    ops = [identity]
    seen = {identity.as_hkl()}
    for op in group.all_ops():
        combined = cb_model.inverse() * sgtbx.change_of_basis_op(op) * cb_data
        key = combined.as_hkl()
        if key not in seen:
            seen.add(key)
            ops.append(combined)
    return ops


def choose_indexing(
    intensity: Any,
    f_calc_abs2: Any,
    data_symmetry: Any,
    model_symmetry: Any,
) -> tuple[Any, IndexingChoice]:
    """Pick the reindexing operator maximizing shell-averaged corr(I, |Fc|²)."""
    ops = reindexing_operators(data_symmetry, model_symmetry)
    scores: list[tuple[float, Any, str]] = []
    miller = intensity
    for op in ops:
        try:
            reindexed = miller.change_basis(op).map_to_asu()
        except Exception:
            continue
        common = reindexed.common_set(f_calc_abs2)
        fc = f_calc_abs2.common_set(reindexed)
        if common.size() < 8:
            continue
        i_obs = np.asarray(list(common.data()), dtype=np.float64)
        i_calc = np.asarray(list(fc.data()), dtype=np.float64)
        try:
            ss = np.asarray(list(common.d_star_sq().data()), dtype=np.float64)
        except Exception:
            ss = np.zeros_like(i_obs)
        corr = _shell_mean_corr(i_obs, i_calc, ss)
        if np.isfinite(corr):
            scores.append((corr, op, op.as_hkl()))
    if not scores:
        identity = intensity
        return identity, IndexingChoice(operator="h,k,l", correlation=float("nan"), margin=0.0, n_tried=len(ops))
    scores.sort(key=lambda t: t[0], reverse=True)
    best_corr, best_op, best_hkl = scores[0]
    second = scores[1][0] if len(scores) > 1 else float("nan")
    margin = float(best_corr - second) if np.isfinite(second) else float("nan")
    chosen = miller.change_basis(best_op).map_to_asu()
    return chosen, IndexingChoice(operator=best_hkl, correlation=float(best_corr), margin=margin, n_tried=len(ops))
