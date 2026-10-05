# Ensemble perturbation modes — Phase 4

Phase injection, real-space projections, hold-out, and the latent cloud.

## What was done

- `PhaseInjectionOptions.hl_weight` is in \([0,1]\). When injection is on,
  \(\sigma_A\) for the HL vector comes from the same nuisance values as the
  likelihood (the `ml_i_nuisance_fit` free-flag fit when `fit_nuisance` is
  set). The fit records `loglik_comparable_across_hl_weight: false`.
- [`projection.py`](../../src/phridge/contrib/ensemble/projection.py):
  support mask (optional dilation), F(000) = 0, a Gaussian-weight band
  limit, and a positivity lift of \(\rho_\mu + \sum z\,u\) over the observed
  score box. Each prox is its own function. All are off unless
  `ProjectionOptions.enabled` is set. Gauge (orthonormality) and these
  projectors are separate options. The fit can record both the projected
  and the unprojected log-likelihood; the report says they are different
  objectives.
- [`holdout.py`](../../src/phridge/contrib/ensemble/holdout.py): crystal
  k-fold (fit \(z\) on work, score free: marginal log-likelihood and the
  scale-free unexplained-variance fraction), reflection roles, half-split
  with Procrustes and principal angles, stratified bootstrap (modes aligned
  before any spread), audit PIT per shell.
- [`cloud.py`](../../src/phridge/contrib/ensemble/cloud.py): deconvolved
  covariance, quadratic \(R^2\), positivity margin. No clustering.

## Spec equations

Roles: 0 work, 1 calibration, 2 audit, −1 absent. Calibration reflections
enter only the \(D\) and \(\sigma^2\) update. Audit reflections never enter
a parameter.

The smooth projector is a band limit, not a Gaussian multiply. A coefficient
is kept when \(\exp(-2\pi^2\sigma^2 s^2)\ge 1/2\) and zeroed otherwise, which
is idempotent. Friedel mates stay conjugates, so the density is real.

Optimistic half-split correlation, printed only with that label:

\[
\sqrt{2C/(1+C)}.
\]

## Tests

[`tests/contrib/test_ensemble_designs.py`](../../tests/contrib/test_ensemble_designs.py):

- V13 `test_v13_projectors_are_idempotent_and_friedel`: Friedel mates match
  to `pytest.approx`; imaginary density under \(10^{-10}\); F(000), support,
  dilation, smooth, and positivity are idempotent; the positivity lift stays
  non-negative on the observed score box (margin \(10^{-10}\)).
- `test_roles_and_pit_and_cloud`: roles land in \(\{0,1,2\}\); a matched
  intensity PIT is \(1/2\); the calibration curve's bin means increase;
  quadratic \(R^2\) of a linear score column is 0; the positivity margin
  is smaller on the score that drives the density negative.
- V11 and V12 are `test_note_numbers_xfail_until_reference`
  (`reference/ not shipped`). No note number is quoted as a result.

## Deviations

- `project_smooth` zeros high-angle coefficients. Multiplying by a Gaussian
  weight is not idempotent, so it is not the projector.
- Projected and unprojected log-likelihoods are different objectives and are
  not compared.
- Training values at different `hl_weight` are not compared.
