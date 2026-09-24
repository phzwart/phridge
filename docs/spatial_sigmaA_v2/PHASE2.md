# Spatial σ_A v2 — Phase 2

Per-atom error model, moments, and the closed-form Rice/Woolfson target
(spec §1–§4, §6). Worker ops still ignore the optional block: an empty
Phase-1 payload has `n_shells=0` and would make `Σ_Δ = 0`. Existing
`ml_i` / spatial v1 paths are unchanged.

## What was done

New modules under
[`src/phridge/contrib/spatial_sigmaa_v2/`](../../src/phridge/contrib/spatial_sigmaa_v2/):

- `fields.py` — symmetry-adapted Fourier basis for mean-zero `λ(x)`, `κ(x)`.
  Unique ASU reals, `c_0` excluded, Hermitian pairing, centric phase
  restriction. Evaluated at atom positions only (spec §6).
- `per_atom.py` — `w_j = exp(λ_j)`, `U_j = κ_j/(8π²) I`, `d_j` (eq. 14),
  modified model occupancy / ADP (eq. 25).
- `moments.py` — `F_eff` (eq. 3) by direct sum or FFT on the modified
  model × `D_0(s)`; phase-free `Σ_Δ` (eq. 4).
- `target.py` — intensity NLL from eqs 7–8 (eq. 9), plus `∂L/∂F_eff*`
  and `∂L/∂Σ_Δ`. Not `ml_i` (no experimental σ(I)).

## Spec equations

| Eq. | Where |
| --- | --- |
| 2, 14 | `luzzati_d_iso`, `effective_weight` |
| 3, 25 | `f_eff_direct`, `f_eff_fft`, `modified_model` |
| 4 | `sigma_delta` (independence (A1), uniform unmodelled (A5)) |
| 5–6 | `draw_circular_normal` (CLT (A3), circularity (A4)) |
| 7–9 | `rice_nll_intensity` |
| 10–12 | `test_uniform_limit` |
| 13 | `presence_weight`, `error_u_iso` (`λ̄`, `U_0` absorbed in `D_0`) |
| 20–21 | `intensity_variance_limit`, `least_squares_intensity_nll` |

## Tests

[`tests/contrib/test_spatial_sigmaa_v2.py`](../../tests/contrib/test_spatial_sigmaa_v2.py),
named as spec §11 where that section applies now:

- `test_circularity` — §11.1 / (A4)
- `test_gaussianity` — §11.2 / (A3)
- `test_uniform_limit` — §11.3 / §5
- `test_least_squares_limit` — §11.4 / §8 (NLL and Var(I); Fisher closure is Phase 4)
- Field / eq. 3–4 / 7–8 / 14 / 25 unit tests
- `test_flag_off_is_inert` still holds (Phase 1)

Held for later phases: Fisher closure (§11.5), field recovery (§11.6),
parameter gradients (§11.7), null calibration (§11.8).

## Deviations

- Worker `ml_i_*` functions accept `spatial_sigma_a_v2` and do not read it.
  Consuming an empty block would set `Σ_miss = 0` and singularise Rice.
- `Σ_Δ` uses the ASU modelled ADP for `T_j`. Exact for isotropic atoms;
  anisotropic copies would need a rotated `U*` per symop (not used in
  Phase 2 tests).
- Intensity NLL is `−log p(I)` with `p(I) = p(|F|)/(2|F|)` as written in
  §4. Amplitude-only terms that do not depend on `(F_eff, Σ_Δ)` are kept
  on the centric branch (the `2|F|` Jacobian).
- Bessel `I_0`, `I_1` are Abramowitz–Stegun 9.8 polynomials (no scipy).
