"""Shared synthetic case for the intensity-likelihood split."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _hkl_to_d_min(cell: tuple[float, ...], d_min: float) -> np.ndarray:
    """Integer triples inside the resolution sphere, excluding (0, 0, 0)."""
    a, b, c = cell[0], cell[1], cell[2]
    hmax, kmax, lmax = (int(math.floor(x / d_min)) for x in (a, b, c))
    h, k, l = np.meshgrid(
        np.arange(-hmax, hmax + 1),
        np.arange(-kmax, kmax + 1),
        np.arange(0, lmax + 1),
        indexing="ij",
    )
    hkl = np.stack([h.ravel(), k.ravel(), l.ravel()], axis=1)
    hkl = hkl[np.any(hkl != 0, axis=1)]
    s = np.stack(
        [hkl[:, 0] / a, hkl[:, 1] / b, hkl[:, 2] / c],
        axis=1,
    )
    d = 1.0 / np.maximum(np.linalg.norm(s, axis=1), 1e-12)
    return hkl[d >= d_min].astype(np.int32)


def mli_case() -> dict[str, Any]:
    """Tiny fixed P1 case: shim random structure, direct F_calc, one seed.

    Numpy arrays are the contract. ``xray``, ``table``, and ``params`` are the
    packed objects the legacy ops need to score the same structure.
    """
    from phridge.contracts.models import ObservationType, SfEngineParams
    from phridge.contracts.packing import PackedMiller
    from phridge.q.error_models.wilson_aniso import reciprocal_cartesian
    from phridge.sfcalc.ops import scattering_model
    from phridge.shims.cctbx import (
        random_xray_structure,
        scattering_table_from_cctbx,
        xray_from_cctbx,
    )
    from phridge.transforms.fourier.ten_eyck import EngineParams, StructureFactorEngine

    xs = random_xray_structure(
        "P 1",
        ["C", "C", "C", "C"],
        volume_per_atom=50.0,
        random_u_iso=False,
        random_occupancy=False,
        seed=1,
    )
    xray = xray_from_cctbx(xs)
    table = scattering_table_from_cctbx(xs)
    crystal = xray.meta.crystal
    unit_cell = tuple(float(x) for x in crystal.unit_cell)
    d_min = 1.8
    hkl = _hkl_to_d_min(unit_cell, d_min)
    params = SfEngineParams(d_min=d_min, quality_factor=100.0, dtype="float64")
    engine = StructureFactorEngine(
        scattering_model(xray, table),
        hkl,
        EngineParams(d_min=d_min, quality_factor=100.0, dtype="float64"),
    )
    f_calc = np.asarray(engine.f_calc_numpy(), dtype=np.complex128)
    n = int(hkl.shape[0])
    rng = np.random.default_rng(1)
    i_true = np.abs(f_calc) ** 2
    sig_i = np.maximum(0.08 * i_true + 0.5, 0.5)
    i_obs = i_true + rng.normal(0.0, sig_i)
    epsilon = np.ones(n, dtype=np.float64)
    centric = np.zeros(n, dtype=bool)
    s_cart = reciprocal_cartesian(unit_cell, hkl)
    s_sq = np.sum(np.asarray(s_cart, dtype=np.float64) ** 2, axis=1)
    flags = (np.arange(n) % 5 == 0)
    f_obs = PackedMiller(
        crystal=crystal,
        hkl=hkl,
        data=i_obs,
        sigmas=sig_i,
        observation_type=ObservationType.intensity,
    )
    f_calc_packed = PackedMiller(
        crystal=crystal,
        hkl=hkl,
        data=f_calc,
        observation_type=ObservationType.complex,
    )
    return {
        "hkl": hkl,
        "i_obs": np.asarray(i_obs, dtype=np.float64),
        "sig_i": np.asarray(sig_i, dtype=np.float64),
        "f_calc": f_calc,
        "epsilon": epsilon,
        "centric": centric,
        "s_sq": np.asarray(s_sq, dtype=np.float64),
        "unit_cell": np.asarray(unit_cell, dtype=np.float64),
        "flags": flags,
        "xray": xray,
        "table": table,
        "params": params,
        "f_obs_packed": f_obs,
        "f_calc_packed": f_calc_packed,
    }
