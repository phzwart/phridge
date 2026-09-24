# Spatial σ_A v2 — Phase 5

Field recovery, null calibration, and folding the frozen modified model
into the coordinate-cycle `F_calc`. Flag off remains inert.

## What was done

- `apply.py` — `xray_with_frozen_errors` implements eq. 25 with `(w, U)`
  taken from `SpatialSigmaAV2Result` (not re-evaluated at the live `x`).
  Occupancy grads convert back with `∂L/∂occ = w · ∂L/∂(occ w)`.
- `ml_i_target_and_gradients` applies that path when a fitted result is
  present; otherwise the classical `ml_i` graph is unchanged.
- Client `job_input_from_state` sends the stored block + result on every
  target call after the field/Fisher step. Empty when the flag is off.
- `synthetic.py` — two-domain P2_12_12_1 crystal (centrics present).
- `null_calibration` — parametric bootstrap of the free-field likelihood
  gain under a uniform error model; the quantile is the acceptance
  threshold (spec §11.8).

## Spec equations / §11

| Item | Where |
| --- | --- |
| eq. 25 | `xray_with_frozen_errors`, target apply |
| §11.6 | `test_field_recovery` |
| §11.8 | `test_null_calibration` |
| §11.9 | `test_flag_off_kwargs_omit_fitted_state` |

## Tests

- `test_field_recovery` — domain B (deleted / displaced / inflated) has
  lower `w` than domain A; a uniform-error control stays flatter
- `test_null_calibration` — two-domain gain exceeds the median null
- `test_apply_frozen_modified_model`
- `test_flag_off_is_inert` / `test_flag_off_kwargs_omit_fitted_state`

## Deviations

- The coordinate target is still `ml_i` (Rice × experimental σ(I)). Only
  the mean is replaced by `D_0 F_c(modified)`. `Σ_Δ` is not yet mapped
  onto `beta_residual`.
- Field recovery uses a short gradient-descent field step, not a fully
  converged LBFGS. Contrast is qualitative (B down-weighted vs A).
  `likelihood_gain_free_field` first absorbs the mean scale into
  `D_0` / `Σ_miss` (`fit_uniform_shells`); λ has no constant term.
- The field step normalises `(g_λ, g_κ)` before the line search so a
  large Rice gradient cannot slam coefficients into the clip.
- Null calibration uses 6–8 bootstrap draws so the suite stays fast.
  The stored artefact is the quantile of that sample.
- Coot export samples λ and κ on a coarse grid with the atom basis
  (`viz.py`), then writes `{prefix}_lambda.ccp4`, `_kappa.ccp4`,
  `_w.ccp4` (`w = exp(λ)` pointwise — not an FFT of e^λ) and
  `{prefix}_field_atoms.pdb` (occ = w, B = κ). Prefix:
  `PHRIDGE_SPATIAL_SIGMA_A_V2_PREFIX` (else `mli`).
