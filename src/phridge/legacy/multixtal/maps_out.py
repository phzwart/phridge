"""Plug-in map coefficients: model phases of ``S``, FOM from each dataset's σ_A.

Does not call ``intensity_map_coefficients`` (that is the integrated posterior).
Reuses ``_fom_at`` from ``maps.py`` for the figure of merit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class MapCoefficients:
    mean_anomalous: np.ndarray
    factor_anomalous: np.ndarray
    factor_ordinary: np.ndarray
    fom: np.ndarray
    phase_s: np.ndarray


def plugin_fom(e_c: np.ndarray, sigma_a: np.ndarray, centric: np.ndarray) -> np.ndarray:
    """Classical FOM ``m(E=E_c)`` averaged over datasets."""
    from phridge.contrib.intensity_ll.maps import _fom_at

    ec = np.asarray(e_c, dtype=np.float64)
    sa = np.asarray(sigma_a, dtype=np.float64)
    cen = np.asarray(centric, dtype=bool)
    if ec.ndim == 2:
        ec_1d = np.nanmean(ec, axis=0)
    else:
        ec_1d = ec
    if sa.ndim == 2:
        sa_1d = np.nanmean(sa, axis=0)
    else:
        sa_1d = sa
    ec_t = torch.as_tensor(ec_1d, dtype=torch.float64)
    sa_t = torch.as_tensor(sa_1d, dtype=torch.float64).clamp(0.01, 0.999)
    cen_t = torch.as_tensor(cen.reshape(-1), dtype=torch.bool)
    return _fom_at(ec_t, ec_t, sa_t, cen_t).cpu().numpy()


def map_coefficients(
    s: np.ndarray,
    intercept: np.ndarray,
    slopes: np.ndarray,
    loadings: np.ndarray,
    e_c: np.ndarray,
    sigma_a: np.ndarray,
    centric: np.ndarray,
    strong: np.ndarray,
) -> MapCoefficients:
    """Back-project along ``i S`` (anomalous) or ``S`` (ordinary in-phase)."""
    s_arr = np.asarray(s, dtype=np.complex128)
    if s_arr.ndim == 2:
        # Mean S over datasets (first finite).
        s_use = np.nanmean(np.where(np.abs(s_arr) > 0, s_arr, np.nan), axis=0)
        s_use = np.where(np.isfinite(s_use), s_use, 0.0)
    else:
        s_use = s_arr
    amp = np.abs(s_use)
    phase = np.where(amp > 0, s_use / amp, 0.0 + 0.0j)
    fom = plugin_fom(e_c, sigma_a, centric)
    fom = np.where(strong, fom, 0.0)
    i_phase = 1.0j * phase
    mean_anom = fom * np.asarray(intercept, dtype=np.float64) * i_phase
    rank_a = slopes.shape[0] if slopes.size else 0
    fac_anom = np.zeros((rank_a, s_use.size), dtype=np.complex128)
    for m in range(rank_a):
        fac_anom[m] = fom * np.asarray(slopes[m], dtype=np.float64) * i_phase
    rank_o = loadings.shape[0] if loadings.size else 0
    fac_ord = np.zeros((rank_o, s_use.size), dtype=np.complex128)
    for m in range(rank_o):
        fac_ord[m] = fom * np.asarray(loadings[m], dtype=np.float64) * phase
    return MapCoefficients(
        mean_anomalous=mean_anom,
        factor_anomalous=fac_anom,
        factor_ordinary=fac_ord,
        fom=fom,
        phase_s=phase,
    )


def damage_map_coefficients(
    s: np.ndarray,
    loadings: np.ndarray,
    e_c: np.ndarray,
    sigma_a: np.ndarray,
    centric: np.ndarray,
    strong: np.ndarray,
) -> np.ndarray:
    """In-phase damage loading maps: ``m · ℓ_k · S / |S|``. History must say so."""
    s_arr = np.asarray(s, dtype=np.complex128)
    if s_arr.ndim == 2:
        s_use = np.nanmean(np.where(np.abs(s_arr) > 0, s_arr, np.nan), axis=0)
        s_use = np.where(np.isfinite(s_use), s_use, 0.0)
    else:
        s_use = s_arr
    amp = np.abs(s_use)
    phase = np.where(amp > 0, s_use / amp, 0.0 + 0.0j)
    fom = plugin_fom(e_c, sigma_a, centric)
    fom = np.where(strong, fom, 0.0)
    ell = np.asarray(loadings, dtype=np.float64)
    if ell.ndim == 1:
        ell = ell.reshape(1, -1)
    out = np.zeros((ell.shape[0], s_use.size), dtype=np.complex128)
    for k in range(ell.shape[0]):
        out[k] = fom * ell[k] * phase
    return out
