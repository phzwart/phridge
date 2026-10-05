# Ensemble perturbation modes — Phase 3

In-repo helix campaign and the qualitative acceptance gates. The note's
stored numbers stay xfailed until `reference/results*.json` exists.

## What was done

- [`synthetic.py`](../../src/phridge/contrib/ensemble/synthetic.py): direct
  summation of a 12-site helix plus elastic-network displacements. No cctbx
  on this path. `campaign_with_engine` evaluates the same sites through
  `f_calc_s_g`.
- Fixed campaign constants, also written in
  [`ensemble_modes.md`](../../src/phridge/contrib/ensemble/ensemble_modes.md):
  cell \((30, 30, 40, 90, 90, 90)\), \(K_{\mathrm{true}}=3\),
  \(z\sim N(0,\mathrm{diag}(1, 0.6, 0.3))\) then centred, `hmax=2` (62
  half-set reflections), intensity noise \(0.05|F|^2+0.5\), seed 0.
  `N_NOTE = 128` is the note's dataset count and is only a constant check.

## Spec equations

Structure factors are the direct sum over sites. Modes are finite-difference
displacements, then a thin SVD so the linearised \(\Delta_k\) are
orthonormal under the real embedding. Principal angles are
\(\arccos\) of the singular values of \(Q_a^\mathsf{T} Q_b\), in degrees.

## Tests

[`tests/contrib/test_ensemble_designs.py`](../../tests/contrib/test_ensemble_designs.py):

- `test_holdout_k_scan_and_half_split` on known-F, \(N=8\), \(K_{\mathrm{true}}=2\),
  `hmax=1`, seed 7, noise 0. Held-out mean log-likelihood:
  \(K=0:-0.431\), \(K=1:-0.268\), \(K=2:-0.209\), \(K=3:-0.198\) nats.
  Gain from \(K=0\) to \(K_{\mathrm{true}}\) is \(0.222\) nats; the next
  step is \(0.011\) nats. The gate is `gain_to_truth > gain_after`.
- Half-split correlations are finite. `sqrt(2C/(1+C))` is stored under
  `correlation_optimistic_note_s6` and labelled `optimistic`. A one-reflection
  shell on `hmax=1` has correlation 1, so monotonicity in resolution is not
  asserted on this campaign.
- `test_pca_phase_gap_is_qualitative`, \(N=12\), `hmax=2`, seed 5. Mean
  principal angle of a PCA on the true complex difference: \(28.0^\circ\)
  (components \(24.6^\circ\), \(31.3^\circ\)). The same PCA after replacing
  the phase by \(\arg F_\mu\): \(72.8^\circ\) (\(55.9^\circ\), \(89.8^\circ\)).
  Gate: oracle mean angle smaller than the model-phased mean.
- `test_slow_campaign_qualitative_gates` (`slow`): EM, \(N=12\), \(K=3\),
  `hmax=2`, 8 sweeps, `dof_correct=False`. Smallest history step
  \(+105.8\) nats (gate: no drop larger than \(10^{-4}\)). Principal angles
  to the planted modes: \(76.2^\circ\), \(83.2^\circ\), \(89.7^\circ\). They
  are finite. This run does not recover the planted subspace.
- `test_engine_campaign_shape`: `f_calc_s_g` length matches `half_set(1)`.
- `test_note_numbers_xfail_until_reference` xfails. The note's §7.1–§7.3
  tolerances (\(\pm 3^\circ\), \(\pm 0.05\) nats) are not checked, because
  `reference/results*.json` is absent. No stored number was invented.

## Deviations

- V6 and V8–V12 are the xfail above. When a results file appears, a miss
  keeps the test, xfails with the observed value, and both numbers are
  quoted here.
- The cctbx campaign writer in `multixtal.synthetic` is not used for these
  gates. See Phase 5 for the `adopt_set` failure.
- No speed-up is claimed.
