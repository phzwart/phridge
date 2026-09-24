# Spatial σ_A v2 — Phase 4

Alternation, Fisher closure, and diagnostics. With the flag off, no step
runs and `ml_i` is unchanged. The coordinate cycle still uses the
classical `ml_i` target; `(w_j, U_j)` are computed and stored, not yet
folded into Phenix LBFGS (that needs the modified model on every
`f_calc`, Phase 5 / a follow-up).

## What was done

- `fisher.py` — intensity-LS 3×3 site blocks (eq. 22) in the fractional
  frame, Cartesian `U_j = [M⁻¹]_{xx}` (eq. 23), damped fixed-point
  iteration. Off-diagonal blocks omitted (usual sparse-inverse approx.).
- `alternate.py` — field half-step (atoms frozen), Fisher half-step
  (fields stay at zero), `FrozenAtomErrors` so λ is not re-evaluated at
  updated x, diagnostics pack.
- `op.py` — worker op `spatial_sigma_a_v2_step`.
- Client: `_run_spatial_sigma_a_v2_step` at the end of
  `update_all_scales`, after the nuisance fit and before LBFGS. Wrapped
  so a failure cannot abort the macrocycle. No-op when the flag is off.

## Spec equations

| Eq. | Where |
| --- | --- |
| 21–22 | `ls_curvatures`, `intensity_ls_site_blocks` |
| 23 | `cartesian_covariance` |
| §9 iteration | `iterate_fisher_closure` |
| §10 schedule | `field_step`, `FrozenAtomErrors`, `run_macrocycle_step` |

## Tests

- `test_fisher_closure` — spec §11.5: high- and low-U starts converge
  toward one finite attractor of the same order as the generating U
- `test_alternation_freezes_weights_during_atom_step`
- `test_fisher_flag_dispatches_closure`
- `test_diagnostics_pack`
- `test_flag_off_is_inert` still holds

## Deviations

- Fisher uses the isotropic `tr(U)/3` and a damped update
  (`U ← (1−α)U + α M⁻¹`, α=½). Raw `α=1` can overshoot on small
  reflection sets.
- Diagonal-block inverse only; neighbour couplings are not inverted.
- LBFGS still scores `ml_i` on the unmodified `F_c`. The v2 step writes
  `PackedSpatialSigmaAV2` / `PackedSpatialSigmaAV2Result` on the f-model
  (`_spatial_sigma_a_v2_block`, `_spatial_sigma_a_v2_result`) for the
  next phase to consume.
