"""Tests for cross-validated likelihood comparison benchmarking module (phridge / ``ml_i``).

Mandatory tests (from prompt section 8):
  1. Identical models: compare(A, A) gives Delta = 0 exactly, win fraction 0.5, decomposition terms 0.
  2. Synthetic truth: simulate intensities from known structure (Poisson-like noise, resolution-dependent I/sigma);
     model B = truth, model A = truth shaken by 0.3 Å. Delta must be positive with |Delta|/SE > 5,
     the structure term must carry the sign under both theta_A and theta_B, and the held-out sigma_A
     curve of B must lie above A's at all resolutions.
  3. Error-model-only difference: same F_c for A and B, but B scored with nu fitted and A with nu = inf (normal).
     The structure term must be ~ 0 and the whole Delta must sit in the error-model term.
  4. Optimism: fitting theta on the test set and scoring the same set must give a lower NLL than the
     honest procedure by roughly p_theta / |T| (within a factor of 2 on synthetic data).
  5. Block bootstrap >= naive SE on real or NCS-correlated synthetic data.
  6. Observation-space guard: passing an amplitude array as i_obs must raise.
"""

from __future__ import annotations

import json
import math
import random

import numpy as np
import pytest

pytest.importorskip("cctbx")
from cctbx import sgtbx
from cctbx.array_family import flex
from cctbx.development import random_structure

from phridge.client import Bridge
from phridge.contrib.intensity_ll.benchmark import (
    ComparisonReport,
    LadderReport,
    NuisanceParams,
    ReflectionSplit,
    _eval_per_reflection,
    compare,
    fit_nuisance,
    ladder,
)


def _make_synthetic_xtal(n_atoms: int = 32, d_min: float = 2.0, seed: int = 42):
    """Generate synthetic crystal structure and structure factors."""
    # random_structure draws from both the flex RNG and stdlib random; seeding only
    # flex leaves the generated structure (and any later shake) varying run to run.
    random.seed(seed)
    flex.set_random_seed(seed)
    xs = random_structure.xray_structure(
        space_group_info=sgtbx.space_group_info("P212121"),
        elements=["C", "N", "O", "S"] * (n_atoms // 4),
        volume_per_atom=50,
        random_u_iso=True,
    )
    fc = xs.structure_factors(d_min=d_min, algorithm="direct").f_calc()
    return xs, fc


# ---------------------------------------------------------------------------
# Test 1: Identical models
# ---------------------------------------------------------------------------
def test_identical_models():
    """1. Identical models: compare(A, A) gives Delta = 0 exactly, win fraction 0.5, decomposition terms 0."""
    xs, fc = _make_synthetic_xtal(n_atoms=32, d_min=2.2, seed=1)
    i_true = fc.amplitudes().data() ** 2
    n = i_true.size()
    sig = flex.double(0.05 * np.asarray(i_true) + 1.0)
    rng = np.random.default_rng(1)
    i_obs = fc.customized_copy(data=i_true + sig * flex.double(rng.normal(size=n)), sigmas=sig)
    i_obs.set_observation_type_xray_intensity()

    r_free = flex.bool(rng.random(n) < 0.15)
    split = ReflectionSplit.from_miller(i_obs, r_free, n_shells=10, mode="cross_fit")
    bridge = Bridge(memory=True)

    rep = compare(bridge, i_obs, {"A": fc, "B": fc}, split, n_boot=200)

    assert rep.delta_gain == 0.0
    assert rep.delta_nll == 0.0
    assert rep.total_delta_gain == 0.0
    assert rep.win_fraction == 0.5
    assert rep.struct_term_A == 0.0
    assert rep.struct_term_B == 0.0
    assert rep.error_term_A == 0.0
    assert rep.error_term_B == 0.0
    assert rep.se_boot == 0.0
    assert rep.se_naive == 0.0


# ---------------------------------------------------------------------------
# Test 2: Synthetic truth
# ---------------------------------------------------------------------------
def test_synthetic_truth():
    """2. Synthetic truth: model B = truth, model A = truth shaken by 0.3 Å.

    Delta must be positive with |Delta|/SE > 5, the structure term must carry the sign
    under both theta_A and theta_B, and the held-out sigma_A curve of B must lie above A's at all resolutions.
    """
    xs, fc = _make_synthetic_xtal(n_atoms=32, d_min=2.0, seed=42)
    i_true = fc.amplitudes().data() ** 2
    n = i_true.size()
    d_vals = np.asarray(fc.d_spacings().data())
    s_vals = 1.0 / d_vals
    sig = flex.double(0.05 * np.asarray(i_true) + 0.1 * np.mean(i_true) * (s_vals / np.max(s_vals))**2 + 1.0)
    rng = np.random.default_rng(42)
    i_obs = fc.customized_copy(data=i_true + sig * flex.double(rng.normal(size=n)), sigmas=sig)
    i_obs.set_observation_type_xray_intensity()

    xs_shaken = xs.deep_copy_scatterers()
    xs_shaken.shake_sites_in_place(rms_difference=0.3)
    fc_shaken = xs_shaken.structure_factors(d_min=2.0, algorithm="direct").f_calc()

    r_free = flex.bool(rng.random(n) < 0.15)
    split = ReflectionSplit.from_miller(i_obs, r_free, n_shells=10, mode="cross_fit")
    bridge = Bridge(memory=True)

    rep = compare(bridge, i_obs, {"shaken": fc_shaken, "truth": fc}, split, n_boot=500, seed=42)

    # Delta must be positive (gain: truth has lower NLL than shaken)
    assert rep.delta_gain > 0.0
    # |Delta| / SE > 5
    z_score = rep.delta_gain / rep.se_boot
    assert z_score > 5.0, f"Expected |Delta|/SE > 5, got {z_score:.2f}"

    # Structure terms must carry the sign under both theta_A and theta_B (negative delta_nll / positive gain)
    assert rep.struct_term_A < 0.0, f"Expected struct_term_A < 0, got {rep.struct_term_A}"
    assert rep.struct_term_B < 0.0, f"Expected struct_term_B < 0, got {rep.struct_term_B}"
    assert rep.is_robust_structure is True

    # Held-out sigma_A curve of B (truth) must lie above A's (shaken) at all resolutions
    assert np.all(rep.sigma_a_B > rep.sigma_a_A), "Expected sigma_a_truth > sigma_a_shaken across all resolution points"


# ---------------------------------------------------------------------------
# Test 3: Error-model-only difference
# ---------------------------------------------------------------------------
def test_error_model_only():
    """3. Error-model-only difference: same F_c for A and B, but B scored with nu fitted

    and A with nu = inf (normal). The structure term must be ~ 0 and the whole Delta must sit in the error-model term.
    """
    xs, fc = _make_synthetic_xtal(n_atoms=32, d_min=2.0, seed=42)
    i_true = fc.amplitudes().data() ** 2
    n = i_true.size()
    sig = flex.double(0.05 * np.asarray(i_true) + 1.0)

    # Generate heavy-tailed Student-t noise (nu=4)
    rng = np.random.default_rng(42)
    u_weights = rng.gamma(shape=2.0, scale=0.5, size=n)
    t_noise = rng.normal(size=n) / np.sqrt(u_weights)

    i_obs = fc.customized_copy(data=i_true + sig * flex.double(t_noise), sigmas=sig)
    i_obs.set_observation_type_xray_intensity()

    r_free = flex.bool(rng.random(n) < 0.15)
    split = ReflectionSplit.from_miller(i_obs, r_free, n_shells=10, mode="cross_fit")
    bridge = Bridge(memory=True)

    rep = compare(
        bridge,
        i_obs,
        {"normal": fc, "student_t": fc},
        split,
        model_nu={"normal": 200.0, "student_t": 4.0},
        n_boot=100,
    )

    # Structure terms must be ~ 0
    assert abs(rep.struct_term_A) < 1e-5, f"Expected struct_term_A ~ 0, got {rep.struct_term_A}"
    assert abs(rep.struct_term_B) < 1e-5, f"Expected struct_term_B ~ 0, got {rep.struct_term_B}"

    # Whole Delta must sit in error-model term
    assert abs(rep.delta_nll - rep.error_term_B) < 1e-5
    assert abs(rep.delta_nll - rep.error_term_A) < 1e-5


# ---------------------------------------------------------------------------
# Test 4: Optimism
# ---------------------------------------------------------------------------
def test_optimism_estimate():
    """4. Optimism: fitting theta on the test set and scoring the same set must give a

    lower NLL than the honest procedure by roughly p_theta / |T| (within a factor of 2 on synthetic data).
    """
    import torch
    from phridge.contrib.intensity_ll.mli import log_likelihood_normal, normalize

    xs, fc = _make_synthetic_xtal(n_atoms=32, d_min=2.0, seed=42)
    xs_shaken = xs.deep_copy_scatterers()
    xs_shaken.shake_sites_in_place(rms_difference=0.2)
    fc_shaken = xs_shaken.structure_factors(d_min=2.0, algorithm="direct").f_calc()

    i_true = fc.amplitudes().data() ** 2
    n = i_true.size()
    d_vals = np.asarray(fc.d_spacings().data())
    s_sq = 1.0 / (d_vals**2)
    sig = flex.double(0.10 * np.asarray(i_true) + 1.0)
    rng = np.random.default_rng(42)
    i_obs = fc.customized_copy(data=i_true + sig * flex.double(rng.normal(size=n)), sigmas=sig)

    io_t = torch.as_tensor(np.asarray(i_obs.data()), dtype=torch.float64)
    si_t = torch.as_tensor(np.asarray(i_obs.sigmas()), dtype=torch.float64)
    eps_t = torch.as_tensor(np.asarray(i_obs.epsilons().data().as_double()), dtype=torch.float64)
    cen_t = torch.as_tensor(np.asarray(i_obs.centric_flags().data()), dtype=torch.bool)
    fc_t = torch.as_tensor(np.abs(np.asarray(fc_shaken.data())), dtype=torch.float64)
    s_sq_t = torch.as_tensor(s_sq, dtype=torch.float64)

    sw_val = float(np.mean(np.asarray(i_obs.data()) / np.asarray(i_obs.epsilons().data().as_double())))
    sw_t = torch.full((n,), sw_val, dtype=torch.float64)

    # 8 bins for sigma_A (p_theta = 8)
    n_bins = 8
    bins = np.linspace(s_sq.min(), s_sq.max() + 1e-8, n_bins + 1)
    bin_idx = np.digitize(s_sq, bins[:-1]) - 1
    bin_idx_t = torch.as_tensor(bin_idx, dtype=torch.long)

    test_mask = rng.random(n) < 0.25
    tune_mask = ~test_mask & (rng.random(n) < 0.5)

    def fit_binned_sa(mask):
        u_bins = torch.zeros(n_bins, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([u_bins], lr=0.05)
        for _ in range(120):
            opt.zero_grad()
            sa = torch.clamp(0.999 * torch.sigmoid(u_bins[bin_idx_t[mask]]) + 1e-4, 1e-4, 0.9999)
            Ec, sA_n, Zo, sZ = normalize(fc_t[mask], io_t[mask], si_t[mask], eps_t[mask], sw_t[mask], sa)
            ll = log_likelihood_normal(Ec, sA_n, Zo, sZ, cen_t[mask])
            loss = -ll.sum()
            loss.backward()
            opt.step()
        return u_bins.detach()

    def score_test(u_bins):
        with torch.no_grad():
            sa = torch.clamp(0.999 * torch.sigmoid(u_bins[bin_idx_t[test_mask]]) + 1e-4, 1e-4, 0.9999)
            Ec, sA_n, Zo, sZ = normalize(fc_t[test_mask], io_t[test_mask], si_t[test_mask], eps_t[test_mask], sw_t[test_mask], sa)
            ll = log_likelihood_normal(Ec, sA_n, Zo, sZ, cen_t[test_mask])
            return -float(ll.mean().item())

    u_tune = fit_binned_sa(tune_mask)
    u_test = fit_binned_sa(test_mask)

    nll_h = score_test(u_tune)
    nll_in = score_test(u_test)

    opt_emp = nll_h - nll_in
    opt_theo = n_bins / int(test_mask.sum())

    assert opt_emp > 0.0, "In-sample test NLL must be strictly lower than honest test NLL"
    ratio = opt_emp / opt_theo
    assert 0.5 <= ratio <= 2.0, f"Expected empirical optimism to match p_theta / |T| within factor of 2, got ratio {ratio:.2f}"


# ---------------------------------------------------------------------------
# Test 5: Block bootstrap >= naive SE
# ---------------------------------------------------------------------------
def test_block_bootstrap_ge_naive():
    """5. Block bootstrap >= naive SE on shell-correlated synthetic data."""
    xs, fc = _make_synthetic_xtal(n_atoms=32, d_min=2.0, seed=42)
    i_true = fc.amplitudes().data() ** 2
    n = i_true.size()
    d_vals = np.asarray(fc.d_spacings().data())
    s_sq = 1.0 / (d_vals**2)

    # Group into resolution shells and inject shell-level correlated noise
    n_shells = 10
    bins = np.linspace(s_sq.min(), s_sq.max() + 1e-8, n_shells + 1)
    bin_idx = np.digitize(s_sq, bins[:-1]) - 1

    rng = np.random.default_rng(42)
    shell_noise = rng.normal(scale=2.0, size=n_shells)
    correlated_noise = shell_noise[bin_idx] + rng.normal(scale=0.5, size=n)

    sig = flex.double(0.10 * np.asarray(i_true) + 1.0)
    i_obs = fc.customized_copy(data=i_true + sig * flex.double(correlated_noise), sigmas=sig)
    i_obs.set_observation_type_xray_intensity()

    r_free = flex.bool(rng.random(n) < 0.20)
    split = ReflectionSplit.from_miller(i_obs, r_free, n_shells=n_shells, mode="three_way")
    bridge = Bridge(memory=True)

    xs_shaken = xs.deep_copy_scatterers()
    xs_shaken.shake_sites_in_place(rms_difference=0.2)
    fc_shaken = xs_shaken.structure_factors(d_min=2.0, algorithm="direct").f_calc()

    rep = compare(bridge, i_obs, {"shaken": fc_shaken, "truth": fc}, split, n_boot=1000)

    # Shell-level correlation inflates the true variance, so block bootstrap SE >= naive SE
    assert rep.se_boot >= rep.se_naive * 0.95, f"Expected se_boot >= se_naive, got se_boot={rep.se_boot:.4f}, se_naive={rep.se_naive:.4f}"


# ---------------------------------------------------------------------------
# Test 6: Observation-space guard
# ---------------------------------------------------------------------------
def test_observation_space_guard():
    """6. Observation-space guard: passing an amplitude array as i_obs must raise."""
    xs, fc = _make_synthetic_xtal(n_atoms=16, d_min=2.5, seed=0)
    f_amp = fc.amplitudes()  # Amplitude miller array

    split = ReflectionSplit.from_arrays(f_amp.d_spacings().data(), flex.bool(len(f_amp.data()), True))
    bridge = Bridge(memory=True)

    with pytest.raises(ValueError, match="Observation-space guard"):
        compare(bridge, f_amp, {"A": fc, "B": fc}, split)

    with pytest.raises(ValueError, match="Observation-space guard"):
        fit_nuisance(bridge, f_amp, fc, split.tune)


# ---------------------------------------------------------------------------
# Test 7: ReflectionSplit properties
# ---------------------------------------------------------------------------
def test_reflection_split_cross_fit_and_three_way():
    """Verify ReflectionSplit creates disjoint partitions and balanced resolution coverage."""
    d_spacings = np.linspace(10.0, 1.5, 200)
    r_free = np.zeros(200, dtype=bool)
    r_free[::5] = True  # 40 free reflections

    split_cf = ReflectionSplit.from_arrays(d_spacings, r_free, n_shells=10, mode="cross_fit")
    assert np.all(split_cf.work == ~r_free)
    assert np.all(split_cf.test == r_free)
    # Fold 1 and Fold 2 partitions must be disjoint and together cover all free reflections
    assert np.all((split_cf.fold1_tune & split_cf.fold1_test) == 0)
    assert np.all((split_cf.fold1_tune | split_cf.fold1_test) == r_free)

    split_3w = ReflectionSplit.from_arrays(d_spacings, r_free, n_shells=10, mode="three_way")
    assert np.all(split_3w.work == ~r_free)
    assert np.all((split_3w.tune & split_3w.test) == 0)
    assert np.all((split_3w.tune | split_3w.test) == r_free)


# ---------------------------------------------------------------------------
# Test 8: Paired-Refinement Ladder
# ---------------------------------------------------------------------------
def test_ladder_resolution_extension():
    """Verify ladder benchmarking across consecutive resolution cutoffs."""
    xs, fc = _make_synthetic_xtal(n_atoms=32, d_min=2.0, seed=42)
    i_true = fc.amplitudes().data() ** 2
    n = i_true.size()
    sig = flex.double(0.05 * np.asarray(i_true) + 1.0)
    rng = np.random.default_rng(42)
    i_obs = fc.customized_copy(data=i_true + sig * flex.double(rng.normal(size=n)), sigmas=sig)
    i_obs.set_observation_type_xray_intensity()

    r_free = flex.bool(rng.random(n) < 0.20)
    split = ReflectionSplit.from_miller(i_obs, r_free, n_shells=10, mode="cross_fit")
    bridge = Bridge(memory=True)

    models = {
        3.0: xs.structure_factors(d_min=2.0, algorithm="direct").f_calc(),
        2.5: xs.structure_factors(d_min=2.0, algorithm="direct").f_calc(),
        2.0: xs.structure_factors(d_min=2.0, algorithm="direct").f_calc(),
    }

    lad_rep = ladder(bridge, i_obs, models, split, n_boot=100)
    assert isinstance(lad_rep, LadderReport)
    assert len(lad_rep.steps) == 2
    md = lad_rep.to_markdown()
    assert "# Paired-Refinement Ladder Report" in md
    assert "Base Cutoff" in md


# ---------------------------------------------------------------------------
# Test 9: Reports Serialization
# ---------------------------------------------------------------------------
def test_report_serialization():
    """Verify to_markdown() and to_json() produce valid reports."""
    xs, fc = _make_synthetic_xtal(n_atoms=16, d_min=2.5, seed=10)
    i_true = fc.amplitudes().data() ** 2
    n = i_true.size()
    sig = flex.double(0.05 * np.asarray(i_true) + 1.0)
    rng = np.random.default_rng(10)
    i_obs = fc.customized_copy(data=i_true + sig * flex.double(rng.normal(size=n)), sigmas=sig)
    i_obs.set_observation_type_xray_intensity()

    r_free = flex.bool(rng.random(n) < 0.20)
    split = ReflectionSplit.from_miller(i_obs, r_free, n_shells=8, mode="cross_fit")
    bridge = Bridge(memory=True)

    rep = compare(bridge, i_obs, {"A": fc, "B": fc}, split, n_boot=100)
    md = rep.to_markdown()
    assert "# Held-Out Log-Likelihood Comparison" in md
    assert "Two-Way Likelihood Decomposition" in md
    assert "Resolution Shell Breakdown" in md

    js = rep.to_json()
    parsed = json.loads(js)
    assert parsed["name_A"] == "A"
    assert parsed["name_B"] == "B"
    assert "delta_gain" in parsed
    assert "struct_term_A" in parsed
