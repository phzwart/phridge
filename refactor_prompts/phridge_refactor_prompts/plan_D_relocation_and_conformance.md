# Plan D — Relocate plugins into their slots; fill the conformance suites

Branch: `refactor/D`. Five stages. The largest plan by file count; every stage
is `git mv` + import fixes + a manifest + a suite. `contrib/` is empty at the end.

Target directories (create as needed):
```
src/phridge/r/{components,generators,restraints}
src/phridge/q/{scales,error_models}
src/phridge/d/{rules}
src/phridge/transforms/{fourier,observation,build}
src/phridge/optim  src/phridge/policies  src/phridge/report  src/phridge/tools
```
Every moved plugin module gets a module-level `MANIFEST: PluginManifest` and a
`register()` that calls `register_manifest(MANIFEST, impl=...)`. Entry points
in `pyproject.toml` under `[project.entry-points."phridge.plugins"]` point at
each `register`; delete the old `phridge.targets` and `phridge.ops` groups once
nothing uses them.

---

## Stage D1 — Fourier operators → `transforms/fourier/`

**Do.**
1. `git mv src/phridge/sfcalc/engine src/phridge/transforms/fourier/ten_eyck`
   (contains `engine.py`, `stamp_numba.py`, `stamp_triton.py`, `stamp_cpp/`,
   `stamp_cuda/`, `sg_gather.py`, `cell.py`, `symmetry.py`; move
   `nufft_engine.py` and `nufft_backend.py` out to step 2).
2. `git mv` `nufft_engine.py`, `nufft_backend.py` and `src/phridge/contrib/nufft_sf/op.py`,
   `options.py` → `src/phridge/transforms/fourier/nufft/`; `contrib/nufft_sf/benchmark.py`
   → `benchmarks/nufft_sf_benchmark.py` (repo root `benchmarks/`, outside the package).
3. Add `src/phridge/transforms/fourier/base.py`: `class FourierOperator(Protocol)` with
   `forward(r: RObject, hkl: LongTensor) -> ComplexArray`,
   `adjoint(r: RObject, hkl, d_values: ComplexTensor) -> RObject` (gradient layout),
   optional `jvp`, `gauss_newton_hvp`, `gauss_newton_diagonal`, `gauss_newton_blocks`.
   Wrap the existing `StructureFactorEngine` in `ten_eyck/operator.py` implementing it;
   the wrapper converts nothing — it calls the engine with tensors it already takes.
4. Manifests: `ten_eyck` (`tier=native`, `provides=["adjoint","jvp","hvp","block_diag"]`,
   `from_space=R, to_space=Q`), `nufft` (`tier=native`, `provides=["adjoint"]`),
   `direct` — move the direct-summation reference from wherever
   `tests/test_agarwal_gradients.py` builds it into `transforms/fourier/direct.py`
   marked `tier=native, provides=["adjoint"]`, documented as the reference.
5. Fill `phridge/testing/conformance/fourier.py`: `test_forward_matches_direct`
   (cosine of gradient direction ≥ cos(0.3°), magnitude ratio within 5e-4,
   R(F) < 2e-3 — the numbers from the existing tests), `test_adjoint_is_vjp`
   (finite-difference check on a 20-atom P1 case), `test_dtype_declared`.
   Register both ten_eyck and nufft suites in `tests/conformance/test_fourier.py`.
6. Fix imports (`grep -rn "sfcalc.engine\|contrib.nufft_sf"`); update
   `pyproject.toml` force-includes for the stamp kernels to the new paths.

**Verify.**
```
python -m pytest -q tests/test_agarwal_gradients.py tests/test_asu_gather.py tests/test_sf_bind.py tests/conformance/test_fourier.py tests/contrib/test_nufft_sf.py tests/boundaries
```
**Commit.** `refactor: Fourier operators under transforms/fourier with conformance suite`

---

## Stage D2 — restraints → `r/restraints/`; mask → `r/components/`

**Do.**
1. `git mv src/phridge/worker/geometry src/phridge/r/restraints/torch_geometry`
   (`energy.py`, `curvature.py`, `adp.py`). Add `r/restraints/base.py` with
   `class RestraintProvider(Protocol)`: `energy(r: RObject) -> tuple[Tensor, RObject]`,
   optional `hvp`, `block_diag`; `covers() -> set[str]` returning parameter kinds
   (`"sites","u_iso","u_star","occ"`).
2. `worker/ops/geometry_ops.py` → `r/restraints/ops.py` (worker-side glue);
   `cctbx_worker/ops/geometry_ops.py` → `shims/cctbx/restraint_ops.py`.
3. `contrib/intensity_ll/dt_mask.py` → `r/components/mask.py` (continuum
   component; manifest `slot=r_component, produces=["mask_density"]`).
   `client/intensity/model.py` → `r/components/atomic.py` ONLY if it is the
   atom-list container; otherwise `tools/`. Decide by reading it; report which.
4. Manifests for `torch_geometry` (`slot=restraint_provider, tier=hybrid`
   — tables from cctbx once, evaluated by torch) and for the ADP restraints.
5. Fill `conformance/restraint_provider.py`: `test_gradient_matches_finite_difference`
   (relative 1e-6 at float64), `test_energy_nonnegative`, `test_covers_declared`.
6. Fix imports; move `tests/test_geometry_minimize.py`, `test_adp_prior.py`,
   `tests/contrib/test_dt_mask.py` next to their code (`tests/r/`).

**Verify.**
```
python -m pytest -q tests/r tests/test_geometry_curvature.py tests/conformance tests/boundaries
```
**Commit.** `refactor: restraint providers under r/restraints; mask as R continuum component`

---

## Stage D3 — scoring rules → `d/rules/`; error models and scales → `q/`

**Do.**
1. `git mv src/phridge/sfcalc/targets src/phridge/d/rules/amplitude`
   (`base.py`, `least_squares.py`, `maximum_likelihood.py`). Rename classes'
   module paths only; keep `register_target` working by aliasing it to
   `register_manifest` with `slot=scoring_rule`.
2. From `contrib/intensity_ll/`: `mli.py`, `target.py`, `rint.py`, `aniso_rice.py`
   → `d/rules/mli_quad/`; `wilson.py` → `q/error_models/wilson_aniso.py`;
   `free_beta.py` → `q/error_models/sigma_a_beta.py`; `local_sigma_a.py` →
   `q/error_models/local_sigma_a.py`; `bulk_solvent_op.py` → `q/scales/solvent.py`.
   `contrib/spatial_sigmaa_v2/*` → `q/error_models/spatial/`.
   `contrib/intensity_ll/maps.py` and `client/intensity/maps.py` → `d/maps.py`
   (merge only if they are the same functions; otherwise keep both files and
   report the overlap).
3. Leave `contrib/intensity_ll/ops.py` where it is — Plan E splits it.
   Leave `surrogate*.py`, `omit_windows*.py`, `benchmark.py`, `synthetic.py`
   for D5.
4. Add `d/rules/base.py` `class ScoringRule(Protocol)`: `needs: tuple[str,...]`,
   `score(pred: Predictive, obs: Observed, mask: BoolTensor) -> tuple[Tensor, dict[str, Tensor]]`
   (value and `d_pred` per field), `statistics(pred, obs, partitions) -> dict`.
   Adapt the existing `Target.evaluate` surface with a thin adapter class,
   no mathematics changed.
5. Manifests: `ls`, `ml_f`, `mli_quad` (`needs=["location","sigma_a","beta","sigma_w","nu"]`),
   `wilson_aniso`, `sigma_a_beta`, `local_sigma_a`, `spatial_sigma_a`, `solvent`.
6. Fill `conformance/scoring_rule.py`: `test_needs_subset_of_predictive`,
   `test_d_location_matches_finite_difference` (complex-step on a 50-reflection
   synthetic case), `test_score_is_sum_over_mask`. Fill `conformance/q_component.py`:
   `test_produces_declared_fields`, `test_adjoint_shapes`.
7. Fix imports; move tests (`test_ml_i_target_k_model.py`, `test_sigma_a_nuisance_bins.py`,
   `tests/contrib/test_aniso_rice.py`, `test_bulk_solvent_*.py`, `test_free_beta.py`,
   `test_local_sigma_a.py`, `test_wilson_aniso.py`, `test_spatial_sigmaa_v2.py`,
   `test_rint.py`, `test_intensity_maps.py`) under `tests/d/` and `tests/q/`.

**Verify.**
```
python -m pytest -q tests/d tests/q tests/conformance tests/boundaries
```
**Commit.** `refactor: scoring rules under d/rules; error models and scales under q/`

---

## Stage D4 — observation operators, generators, report

**Do.**
1. `contrib/multixtal/`: `indexing.py`, `friedel.py`, `reflection_model.py`,
   `dose_model.py`, `dose_io.py`, `damage.py`, `anom_damage.py` → split into
   `transforms/observation/stacking.py` (indexing + friedel + reflection_model),
   `transforms/observation/dose.py` (dose_model + dose_io), and
   `q/error_models/radiation_damage.py` (damage + anom_damage). `interfaces.py`
   → `transforms/observation/interfaces.py` (its `Protocol` classes are the
   pattern; keep them). Remaining multixtal modules (`client.py`, `cli.py`,
   `report.py`, `diagnostics.py`, `influence.py`, `baselines.py`, `explain.py`,
   `residuals.py`, `systematics.py`, `maps_out.py`, `structure.py`, `selector.py`,
   `spline.py`, `factors.py`, `nuisance.py`, `anomalous.py`, `unmerged.py`,
   `latent.py`, `loadings.py`, `options.py`, `op.py`, `synthetic.py`) → `tools/multixtal/`
   until Plan I decides their fate. Add `transforms/observation/identity.py`
   (trivial pass-through, `from_space=Q, to_space=D`).
2. `contrib/ensemble/`: `loadings.py`, `latent.py`, `gauge.py`, `projection.py`,
   `align.py`, `moments.py`, `mstep.py` → `r/generators/perturbation_modes/`
   (one package); `helix_ensemble.py`, `synthetic.py`, `holdout.py`, `cloud.py`,
   `report.py`, `client.py`, `cli.py`, `options.py`, `ops.py`, `nuisance.py`
   → `tools/ensemble/`.
3. `client/intensity/stats_report.py`, `nll_log.py`, `heartbeat.py` →
   `report/` (`s_family.py`, `nll_log.py`, `heartbeat.py`). Add `report/base.py`
   `class Statistic(Protocol)`: `compute(pred, obs, partitions) -> dict`.
4. Manifests: `stacking` (`observation`, `from Q to D`), `dose`, `identity`,
   `perturbation_modes` (`r_component`, blocks `loadings`, `amplitudes`),
   `s_family` (`statistic`).
5. Fill `conformance/observation.py` (`test_identity_roundtrip`,
   `test_adjoint_shapes`, `test_record_indexing_declared`) and
   `conformance/statistic.py` (`test_reads_only_masks`: a statistic receives
   masks, never raw partition labels — assert by signature).
6. `rm -r src/phridge/contrib` once empty except `intensity_ll/ops.py` and the
   D5 leftovers; move `contrib/AGENTS.md` to repo root as `AGENTS.md` (Plan G
   rewrites it). Fix imports; move tests to `tests/transforms/`, `tests/r/`, `tests/report/`.

**Verify.**
```
python -m pytest -q tests/transforms tests/r tests/report tests/conformance tests/boundaries tests/contrib/test_multixtal.py tests/contrib/test_multixtal_damage.py tests/contrib/test_ensemble_designs.py tests/contrib/test_ensemble_helix.py
```
**Commit.** `refactor: observation operators, perturbation-mode generator, report slot`

---

## Stage D5 — tools, benchmarks, examples, entry points

**Do.**
1. `contrib/intensity_ll/{surrogate.py,surrogate_op.py,omit_windows.py,omit_windows_op.py,synthetic.py}`
   and `client/intensity/omit.py`, `math_utils.py` → `tools/intensity/`.
   `contrib/intensity_ll/benchmark.py` → `benchmarks/intensity_benchmark.py`.
   `client/viewer.py` + `client/assets/` → `tools/viewer/`.
   The remaining `client/api.py` → `src/phridge/api.py` (Plan G rewrites it).
   `src/phridge/client/` must now be empty; delete it.
2. `git rm -r` every results directory under `examples/` (`rerefined_no_shake`,
   `synthetic_*`, `benchmark_results`, `drift_results`, `shake_recover_*`,
   `water_omission_study`); keep input files and configs. Add
   `examples/README.md` listing what was removed and that outputs live in
   release assets.
3. `pyproject.toml`: `[project.scripts]` keeps only `phridge-worker`,
   `phridge-cctbx-worker`, `phridge-ops`, `phridge-view` (new path). Remove
   `phridge-multixtal`, `phridge-ensemble` (their CLIs are in `tools/` and get
   `python -m` invocation for now). Replace entry-point groups with a single
   `[project.entry-points."phridge.plugins"]` listing every `register` from D1–D4.
4. Update `src/phridge/ops.py::load_entry_points` default group to `"phridge.plugins"`
   and make `bootstrap_plugins` call `register()` hooks that may register either
   ops or manifests.
5. Run `python scripts/generate_models.py --all` and commit regenerated docs.

**Verify.**
```
ls src/phridge/client 2>&1 | head -1    # expect: No such file or directory
python -m pytest -q -m "not gpu and not slow and not legacy"
python -m pytest -q tests/boundaries
```
Expected: full suite passes with the same number of non-legacy tests as at the
end of Plan C (report the two numbers).

**Commit.** `refactor: tools/, benchmarks/, examples/ cleaned; single plugin entry-point group`
