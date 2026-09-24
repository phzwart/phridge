"""Coot-facing volumes for the band-limited λ / κ fields.

λ and κ are sampled on a real-space grid with the same Fourier basis used
at the atoms. ``w = exp(λ)`` is taken pointwise after that (not an FFT of
e^λ — spec §6). κ is already error-B in Å² (B = 8π² U = κ).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from phridge.contrib.spatial_sigmaa_v2.fields import FieldBasis, grid_n_real, sample_on_grid
from phridge.contrib.spatial_sigmaa_v2.packing import PackedSpatialSigmaAV2, PackedSpatialSigmaAV2Result, _to_numpy
from phridge.contrib.spatial_sigmaa_v2.per_atom import EIGHT_PI2
from phridge.sfcalc.engine.engine import ScatteringModel


def field_volumes(
    block: PackedSpatialSigmaAV2,
    model: ScatteringModel,
    *,
    n_real: Optional[tuple[int, int, int]] = None,
    basis: Optional[FieldBasis] = None,
) -> dict[str, Any]:
    """``lambda``, ``kappa``, ``w`` on a ``(nx, ny, nz)`` grid, plus ``n_real``."""
    if basis is None:
        basis = FieldBasis.build(model.unit_cell, model.rot, model.trans, block.meta.field_cutoff)
    n = n_real or grid_n_real(model.unit_cell, block.meta.field_cutoff)
    lam = sample_on_grid(basis, _to_numpy(block.lambda_c), n)
    kap = sample_on_grid(basis, _to_numpy(block.kappa_c), n)
    return {
        "lambda": lam,
        "kappa": kap,
        "w": np.exp(np.clip(lam, -3.0, 3.0)),
        "n_real": n,
        "unit_cell": tuple(float(x) for x in model.unit_cell),
        "basis": basis,
    }


def atom_display_columns(result: PackedSpatialSigmaAV2Result) -> tuple[np.ndarray, np.ndarray]:
    """Per-atom ``(occupancy=w, B_error=κ)`` for a diagnostic PDB.

    ``B_error = 8π² U_j = κ_j``. Coot colour-by-B then shows the error field.
    """
    w = np.asarray(_to_numpy(result.w), dtype=np.float64).reshape(-1)
    tr = np.asarray(_to_numpy(result.tr_U), dtype=np.float64).reshape(-1)
    b_err = EIGHT_PI2 * (tr / 3.0)
    return w, b_err
