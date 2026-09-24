# Spatial σ_A v2 — Phase 3

Per-atom and field-coefficient gradients (spec §10) and the regulariser
(eqs 15–16). Worker ops still ignore the optional block. Alternation and
the Fisher closure stay in Phase 4.

## What was done

- `gradients.py`
  - Mean term: numpy VJP of eq. 3 and Agarwal–Ten Eyck via
    `StructureFactorEngine.gradients` on the modified model (eq. 25),
    coefficients `D_0 · ∂L/∂F_eff*`.
  - Variance term: isotropic eqs 26–28. `∂Σ_Δ/∂x_j = 0` exactly.
  - Anisotropic variance-gradient map raises `NotImplementedError`
    (self-Patterson route, spec §10).
  - Field chain rule (eq. 29) through the real design matrix:
    `∂L/∂λ_j = (∂L/∂w_j) w_j`, `∂L/∂κ_j = (∂L/∂U_j) / 8π²`.
- `regulariser.py` — spectral taper `½ Σ |c_k|² / v_k` and
  `α Σ_s n_s KL(p_j(s) ∥ uniform)` (eqs 15–16).

## Spec equations

| Eq. | Where |
| --- | --- |
| 24 | `add_atom_grads` |
| 25 | `mean_gradients_fft` / `modified_model` |
| 26 | `variance_gradients_iso.site_frac` ≡ 0 |
| 27–28 | `variance_gradients_iso` (isotropic contraction of 28) |
| 29 | `field_gradients` |
| 15–16 | `spectral_taper`, `explained_power_distribution`, `regulariser` |

## Tests

[`tests/contrib/test_spatial_sigmaa_v2.py`](../../tests/contrib/test_spatial_sigmaa_v2.py):

- `test_gradients` — spec §11.7: FD of mean, variance, and field coefficients
- `test_variance_dx_is_zero` — eq. 26 to machine precision
- `test_anisotropic_variance_xfail` — stub
- `test_regulariser_minimum_at_uniform` — both terms minimised at c = 0
- optional FFT vs direct mean-term comparison when torch is present

## Deviations

- Isotropic `∂L/∂U_j` is the scalar contraction of eq. 28 with `U = u I`
  (`hᵀ (∂/∂u) h = s²`), not a stored 3×3. Cartesian `u_err_star` packing
  is unchanged and still unused by the worker.
- Entropy (eq. 16) uses table `f_j T_j` without an extra occupancy factor,
  as written. Toy tests have occupancy 1.
- Worker ops still do not consume `spatial_sigma_a_v2`.
