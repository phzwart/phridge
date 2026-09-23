"""NLL binned k_mask fit: recovers a planted curve, and knows when to decline."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from phridge.models import CrystalSymmetry, ObservationType
from phridge.packing import PackedMiller

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch required for the worker op"
)

_K_SOL, _B_SOL = 0.38, 55.0


def _problem(seed=0, n=6000, sigma_a=0.9, k_mask=None):
    """Intensities from F_calc + planted solvent, drawn from the Rice model with σ_A.

    ``d`` spans 25 Å to 2 Å so the solvent term matters in the low-resolution shells
    and has died away well before the high ones, as it does in real data.
    """
    rng = np.random.default_rng(seed)
    d = 1.0 / np.sqrt(rng.uniform(1.0 / 25.0**2, 1.0 / 2.0**2, n))
    ss = 1.0 / (4.0 * d**2)
    amp_c = 30.0 * np.exp(-20.0 * ss) * np.sqrt(rng.exponential(1.0, n))
    fc = amp_c * np.exp(2j * np.pi * rng.random(n))
    # The mask transform is large only at low resolution and runs roughly against F_calc
    # there (the solvent fills what the protein leaves empty).
    amp_m = 90.0 * np.exp(-60.0 * ss) * np.sqrt(rng.exponential(1.0, n))
    fm = -amp_m * np.exp(1j * (np.angle(fc) + rng.normal(0.0, 0.8, n)))
    if k_mask is None:
        k_mask = _K_SOL * np.exp(-_B_SOL * ss)
    f_true = fc + k_mask * fm

    order = np.argsort(ss)
    sigma_w = np.empty(n)
    for chunk in np.array_split(order, 30):
        sigma_w[chunk] = np.mean(np.abs(f_true[chunk]) ** 2)
    noise = np.sqrt(0.5 * (1.0 - sigma_a**2) * sigma_w) * (
        rng.normal(size=n) + 1j * rng.normal(size=n)
    )
    i_true = np.abs(sigma_a * f_true + noise) ** 2
    sig = 0.05 * i_true + 0.02 * sigma_w
    i_obs = i_true + rng.normal(0.0, sig)

    crystal = CrystalSymmetry(unit_cell=[80.0, 80.0, 80.0, 90.0, 90.0, 90.0], space_group_hall="P 1")
    hkl = np.stack([np.arange(n), np.zeros(n), np.zeros(n)], axis=1).astype(np.int32)
    f_obs = PackedMiller(
        crystal=crystal, hkl=hkl, data=i_obs, sigmas=sig, observation_type=ObservationType.intensity
    )
    f_calc = PackedMiller(crystal=crystal, hkl=hkl, data=fc, observation_type=ObservationType.complex)
    fit = rng.random(n) >= 0.1
    return dict(
        f_calc=f_calc,
        f_obs=f_obs,
        fm=fm,
        ss=ss,
        sigma_w=sigma_w,
        fit=fit,
        n=n,
        k_true=k_mask,
    )


def _fit(p, k_mask, k_model=None, **kw):
    from phridge.contrib.intensity_ll.bulk_solvent_op import ml_i_bulk_solvent_fit

    return ml_i_bulk_solvent_fit(
        p["f_calc"],
        p["f_obs"],
        f_mask=p["fm"],
        k_model=np.ones(p["n"]) if k_model is None else k_model,
        k_mask=k_mask,
        fit_mask=p["fit"],
        ss=p["ss"],
        epsilon=np.ones(p["n"]),
        centric=np.zeros(p["n"], dtype=bool),
        sigma_wilson=p["sigma_w"],
        **kw,
    )


def test_recovers_the_planted_solvent_from_a_wrong_start():
    p = _problem()
    wrong = 0.2 * np.exp(-120.0 * p["ss"])
    out = _fit(p, wrong)
    assert out["accepted"]
    # The model is the curve, not the two-number caption.
    rms = float(np.sqrt(np.mean((out["k_mask"] - p["k_true"]) ** 2)))
    assert rms < 0.06
    st = out["stats"]
    assert st["nll_fit_new"] < st["nll_fit_ref"]
    assert st["nll_rest_new"] < st["nll_rest_ref"]
    assert not st["lbfgs_failures"]
    assert st["n_bins"] >= 6
    assert len(st["k_new"]) == 3


def test_recovers_a_non_exponential_k_mask():
    """The point of the binned protocol: a curve LS would caption badly is recoverable."""
    rng_ss = np.random.default_rng(4)
    # Build ss the same way as _problem so the planted step sits on real resolution.
    n = 6000
    d = 1.0 / np.sqrt(rng_ss.uniform(1.0 / 25.0**2, 1.0 / 2.0**2, n))
    ss = 1.0 / (4.0 * d**2)
    # High at low resolution, nearly gone past ~6 Å — not an exponential.
    planted = 0.45 / (1.0 + np.exp((ss - 0.007) / 0.0015))
    p = _problem(seed=4, n=n, k_mask=planted)
    # A two-parameter LS-like start that cannot represent the step.
    wrong = 0.30 * np.exp(-40.0 * p["ss"])
    out = _fit(p, wrong)
    assert out["accepted"]
    err_new = float(np.sqrt(np.mean((out["k_mask"] - p["k_true"]) ** 2)))
    err_start = float(np.sqrt(np.mean((wrong - p["k_true"]) ** 2)))
    assert err_new < 0.6 * err_start
    assert out["stats"]["nll_fit_new"] < out["stats"]["nll_fit_ref"]


def test_an_overall_scale_does_not_move_the_answer():
    p = _problem(seed=1)
    wrong = 0.2 * np.exp(-120.0 * p["ss"])
    base = _fit(p, wrong)
    scaled = _fit(p, wrong, k_model=np.full(p["n"], 3.0))
    # High-resolution k_mask is unidentified (F_mask has died); only the mix where
    # the mask still contributes is a property of the solvent, not of σ_A.
    w = np.abs(p["fm"])
    sel = w > np.median(w)
    rms = float(np.sqrt(np.mean((base["k_mask"][sel] - scaled["k_mask"][sel]) ** 2)))
    assert rms < 0.05


def test_no_solvent_signal_leaves_nothing_to_win():
    """With F_mask zero the curve is unidentified; the reference must not be beaten by noise."""
    p = _problem(seed=2)
    p["fm"] = np.zeros(p["n"], dtype=np.complex128)
    out = _fit(p, np.zeros(p["n"]))
    st = out["stats"]
    assert abs(st["nll_fit_new"] - st["nll_fit_ref"]) < 1e-6


def test_rejects_mismatched_lengths():
    p = _problem(seed=3, n=500)
    with pytest.raises(ValueError, match="k_model"):
        _fit(p, np.zeros(p["n"]), k_model=np.ones(10))
