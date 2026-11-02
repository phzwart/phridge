# Plan E — Split the `intensity_ll` monolith

Branch: `refactor/E`. Three stages. This is the only plan where mathematics
is re-homed rather than moved, so every stage has a numeric regression gate
against the original op, run on CPU float64.

Reference: `src/phridge/contrib/intensity_ll/ops.py` (≈2.5k lines) contains
`ml_i_nuisance_fit` (≈1.4k lines), `ml_i_target_and_gradients`, `ml_i_maps`,
`ml_i_omit_windows` and helpers. The regression fixture is the 9RRL data in
`examples/` (or the synthetic case in `tests/test_intensity_ll_synthetic.py`
if 9RRL inputs are not in the tree — check first and say which you used).

---

## Stage E1 — freeze the reference

**Goal.** Before touching anything, record the original outputs so later
stages have something exact to compare against.

**Do.**
1. `git mv src/phridge/contrib/intensity_ll/ops.py src/phridge/legacy/intensity_ll_ops.py`
   and fix the two or three importers to point at legacy (temporarily).
2. Write `tests/regression/conftest.py` with a fixture `mli_case()` that loads
   the 9RRL inputs (or the synthetic case) as numpy and returns a dict with
   `hkl, i_obs, sig_i, f_calc (complex), epsilon, centric, s_sq, unit_cell, flags`.
3. Write `tests/regression/test_e1_freeze.py` that calls the legacy
   `ml_i_nuisance_fit`, `ml_i_target_and_gradients`, `ml_i_maps` on CPU
   float64 with fixed options (write them down in the test as a dict) and
   saves outputs to `tests/regression/fixtures/mli_reference.npz`:
   per-shell `sigma_a, beta, nu`, `sigma_w` tensor, total NLL, `d_target_d_f_calc`,
   map coefficients. Commit the npz. The test asserts the file exists and has
   the expected keys; it is the reference, not a check.

**Verify.**
```
python -m pytest -q tests/regression/test_e1_freeze.py
ls -la tests/regression/fixtures/mli_reference.npz
```
**Commit.** `regression: freeze intensity_ll reference outputs before the split`

---

## Stage E2 — three Q components and a calibrate policy

**Goal.** Replace `ml_i_nuisance_fit` with three Q components and one policy
that sequences them. Same mathematics, same options, same outputs.

**Do.**
1. Read `legacy/intensity_ll_ops.py::ml_i_nuisance_fit` top to bottom and list
   its phases in a comment block at the top of `src/phridge/policies/calibrate_intensity.py`
   (expected: Wilson/Σ_W fit, per-shell σA/β fit, ν grid screen with refits,
   smoothing). Report the list before writing code.
2. Create or extend (D3 already made these files; add the fit routines to them):
   - `q/error_models/wilson_aniso.py`: `class WilsonAniso(QComponent)` with blocks
     `aniso_tensor[6]`, `forward(obs: Observed) -> Predictive{sigma_w}` and a
     `fit(obs, mask) -> None` method containing the Σ_W phase, model-free.
   - `q/error_models/sigma_a_beta.py`: `class SigmaABeta(QComponent)` with blocks
     `sigma_a_shell`, `beta_shell` (logit transform, bounds (0,1)), `forward(location, sigma_w) -> Predictive{sigma_a, beta}`,
     `fit(location, obs, mask, nu) -> None` containing the per-shell fit and smoothing.
   - `q/error_models/nu_screen.py`: `class NuScreen(QComponent)` with block
     `nu_shell`, `screen(grid, refit: Callable, criterion: Callable) -> None`
     containing the grid logic; Gaussian (`1/nu = 0`) is a grid node.
   Move helper functions (`_wilson_bin_ids`, `_fit_binned_wilson`, `_shell_standard_errors`,
   `parse_nu_grid`, …) into the module that uses them; delete nothing from legacy.
3. `policies/calibrate_intensity.py`: `class CalibrateIntensity(Policy)` whose
   `run(components, obs, mask)` calls `WilsonAniso.fit`, then alternates
   `NuScreen.screen` with `SigmaABeta.fit` exactly as the legacy function did.
   Options are a pydantic model mirroring the legacy option dict field for field.
4. Manifests for the three components (`slot=q_component`, `fit_on="calibration"`)
   and the policy.
5. `tests/regression/test_e2_calibrate.py`: run the new policy on `mli_case()`
   with the same options as E1; compare to `mli_reference.npz`: `sigma_a`, `beta`
   per shell `atol=1e-8`; `nu` selection identical; `sigma_w` `atol=1e-8`.

**Don't.** Do not change a formula. If a formula must change to fit the
component shape, STOP and report it.

**Verify.**
```
python -m pytest -q tests/regression/test_e2_calibrate.py tests/q tests/conformance
```
**Commit.** `refactor: ml_i_nuisance_fit as three Q components and a calibrate policy`

---

## Stage E3 — target, maps, and the single-crystal identity

**Do.**
1. Move the body of `ml_i_target_and_gradients` into `d/rules/mli_quad/rule.py`
   as `MliQuad.score(pred, obs, mask)` returning `(value, {"location": d_location})`;
   the identity observation operator supplies `pred` from Q fields. Keep the
   quadrature code (`mli.py`, `rint.py`) untouched.
2. Move the body of `ml_i_maps` into `d/maps.py::posterior_map_coefficients(pred, obs, d_location)`;
   the difference coefficient must equal `score.d_location` up to the
   `sqrt(eps Sigma_W)` scaling (assert this in a test).
3. `tests/regression/test_e3_rule.py`: NLL, `d_target_d_f_calc`, map coefficients
   vs `mli_reference.npz`, `atol=1e-8`.
4. Remove the `ml_i_omit_windows` op from the plugin registry (its code is in
   `tools/intensity/omit_windows.py`); it becomes a tool, not an op.
5. Delete `legacy/intensity_ll_ops.py` ONLY if every regression test passes;
   otherwise leave it and report what differs.

**Verify.**
```
python -m pytest -q tests/regression tests/d tests/conformance tests/boundaries
```
**Commit.** `refactor: mli_quad as a scoring rule; posterior maps as its adjoint; legacy op retired`
