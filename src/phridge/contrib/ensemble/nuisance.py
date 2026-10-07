"""Initial D and σ²_Δ.

The per-sweep update is the latent-moment shell formula in ``mstep.py``.
Initial σ_A and Σ_W come from ``ml_i_nuisance_fit`` when ``fit_nuisance`` is
set; otherwise from the identities

    D = σ_A sqrt(Σ_W / Σ_calc),
    σ²_Δ = (1 - σ_A²) ε Σ_W

with the caller's ``sigma_a0``. ``multixtal.nuisance.fit_dataset_nuisance`` is
the Rice quasi-likelihood this package replaces, so it is not called.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from phridge.contracts.models import CrystalSymmetry, ObservationType


def identities(
    sigma_a: np.ndarray,
    sigma_wilson: np.ndarray,
    f_calc: np.ndarray,
    epsilon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-reflection D and σ²_Δ from σ_A and Σ_W."""
    sa = np.clip(np.asarray(sigma_a, dtype=np.float64), 1e-3, 1.0 - 1e-3)
    sw = np.maximum(np.asarray(sigma_wilson, dtype=np.float64), 1e-6)
    eps = np.maximum(np.asarray(epsilon, dtype=np.float64), 1e-6)
    sigma_calc = np.maximum(np.abs(np.asarray(f_calc, dtype=np.complex128)) ** 2 / eps, 1e-6)
    d = sa * np.sqrt(sw / sigma_calc)
    sigma2 = (1.0 - sa**2) * eps * sw
    return d.astype(np.float64), sigma2.astype(np.float64)


def _p1_crystal(cell: tuple[float, ...]) -> CrystalSymmetry:
    return CrystalSymmetry(unit_cell=[float(x) for x in cell], space_group_hall="P 1", space_group_number=1)


def fit_one_dataset(
    hkl: np.ndarray,
    i_obs: np.ndarray,
    sig_i: np.ndarray,
    f_calc: np.ndarray,
    epsilon: np.ndarray,
    centric: np.ndarray,
    cell: tuple[float, ...],
    *,
    tune_mask: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """One ``ml_i_nuisance_fit`` call. Falls back to the identities on failure."""
    from phridge.contrib.intensity_ll.ops import ml_i_nuisance_fit
    from phridge.packing import PackedMiller

    crystal = _p1_crystal(cell)
    hkl_i = np.asarray(hkl, dtype=np.int32)
    f_c = PackedMiller(
        crystal,
        hkl_i,
        np.asarray(f_calc, dtype=np.complex128),
        anomalous=False,
        observation_type=ObservationType.complex,
        label="F_calc",
    )
    f_o = PackedMiller(
        crystal,
        hkl_i,
        np.asarray(i_obs, dtype=np.float64),
        sigmas=np.asarray(sig_i, dtype=np.float64),
        anomalous=False,
        observation_type=ObservationType.intensity,
        label="I_obs",
    )
    tune = np.ones(hkl_i.shape[0], dtype=bool) if tune_mask is None else np.asarray(tune_mask, dtype=bool)
    try:
        fitted = ml_i_nuisance_fit(
            f_c,
            f_o,
            tune,
            epsilon=np.asarray(epsilon, dtype=np.float64),
            centric=np.asarray(centric, dtype=bool),
            fit_nu=False,
            sigma_a_tensor=False,
            wilson_model="isotropic",
            wilson_max_iter=8,
            n_sigma_a_bins=3,
            fit_sigma_wilson=True,
            smooth_sigma_a=0.1,
            smooth_beta=0.1,
        )
        sigma_a = np.asarray(fitted["sigma_a"], dtype=np.float64).reshape(-1)
        sigma_w = np.asarray(fitted["sigma_wilson"], dtype=np.float64).reshape(-1)
        d, sigma2 = identities(sigma_a, sigma_w, f_calc, epsilon)
        return {"sigma_a": sigma_a, "sigma_wilson": sigma_w, "d": d, "sigma2": sigma2, "source": "ml_i_nuisance_fit"}
    except Exception as exc:  # noqa: BLE001 — caller still gets a usable initialisation
        sw = np.full(hkl_i.shape[0], float(np.mean(np.maximum(i_obs, 1.0))))
        sa = np.full(hkl_i.shape[0], 0.9)
        d, sigma2 = identities(sa, sw, f_calc, epsilon)
        return {
            "sigma_a": sa,
            "sigma_wilson": sw,
            "d": d,
            "sigma2": sigma2,
            "source": "identities",
            "fallback": str(exc),
        }
