"""Population-criterion strong reflection set (phase-1 ``ReflectionSelector``).

Selection uses only population statistics, never a single dataset's own values.
Selecting per dataset conditions on the outcome and biases loadings and scores.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


class PopulationSelector:
    """Observed in enough datasets, and median I/σ at least ``strong_isig``."""

    def __init__(
        self,
        strong_isig: float = 5.0,
        strong_min_datasets: float = 0.5,
        e_calc_floor: Optional[float] = None,
    ) -> None:
        self.strong_isig = float(strong_isig)
        self.strong_min_datasets = float(strong_min_datasets)
        self.e_calc_floor = None if e_calc_floor is None else float(e_calc_floor)

    def select(
        self,
        i_bar: Any,
        sig_bar: Any,
        observed: Any,
        e_calc: Optional[Any] = None,
    ) -> np.ndarray:
        i = np.asarray(i_bar, dtype=np.float64)
        sig = np.asarray(sig_bar, dtype=np.float64)
        obs = np.asarray(observed, dtype=bool)
        n_data, n_h = i.shape
        frac = obs.sum(axis=0) / max(float(n_data), 1.0)
        isig = np.full((n_data, n_h), np.nan, dtype=np.float64)
        good = obs & np.isfinite(i) & np.isfinite(sig) & (sig > 0)
        isig[good] = i[good] / sig[good]
        with np.errstate(all="ignore"):
            med = np.nanmedian(np.where(good, isig, np.nan), axis=0)
        chosen = (frac + 1e-15 >= self.strong_min_datasets) & np.isfinite(med) & (med >= self.strong_isig)
        if e_calc is not None and self.e_calc_floor is not None:
            e = np.asarray(e_calc, dtype=np.float64)
            if e.ndim == 2:
                e = np.nanmedian(np.where(obs, e, np.nan), axis=0)
            chosen = chosen & np.isfinite(e) & (e >= self.e_calc_floor)
        return chosen


def per_dataset_selector(
    i_bar: np.ndarray,
    sig_bar: np.ndarray,
    observed: np.ndarray,
    strong_isig: float,
) -> np.ndarray:
    """The *wrong* selector: each dataset keeps its own I/σ ≥ threshold.

    Used only by the selection-bias guard test.
    """
    i = np.asarray(i_bar, dtype=np.float64)
    sig = np.asarray(sig_bar, dtype=np.float64)
    obs = np.asarray(observed, dtype=bool)
    isig = np.zeros_like(i)
    good = obs & (sig > 0)
    isig[good] = i[good] / sig[good]
    return good & (isig >= float(strong_isig))
