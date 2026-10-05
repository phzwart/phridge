# Ensemble perturbation modes — Phase 1

Plumbing, options, registration, the torch-free client, the known-F ALS
op, maps, and the CLI. The intensity E-step is Phase 2.

## What was done

- Package [`src/phridge/contrib/ensemble/`](../../src/phridge/contrib/ensemble/)
  with pydantic options (`EnsembleOptions`, `ProjectionOptions`,
  `HoldoutOptions`, `PhaseInjectionOptions`, `BootstrapOptions`,
  `DatasetRoles`), all `extra="forbid"`.
- Ops `ensemble_fit`, `ensemble_scores`, and `ensemble_maps`, each with
  `compute_dtype = "float64"`. `register()` is idempotent and runs at import.
- Entry point `ensemble = "phridge.contrib.ensemble:register"` under
  `phridge.ops`, and script `phridge-ensemble`.
- `prepare_ensemble_job` wraps `multixtal.client.prepare_job` (common set,
  reindexing, `f_mask`, optional per-dataset models packed as `f_var`).
  The client module imports no torch.
- `likelihood="known_f"` skips the E-step. Weighted ALS is the M-step, then
  the gauge. `ensemble_maps` returns Δ0 as a P1 `MillerArray` plus mode and
  residual arrays.
- CLI subcommands `prepare | fit | scores | maps | holdout`.

## Spec equations

Known-F target, with \(D = 1\) and the reference phase (or a per-dataset
phase model when one is supplied):

\[
R_n = \bigl(|I_n|^{1/2} e^{i\phi_n} - F_{\mathrm{var},n}\bigr) - F_\mu.
\]

The M-step is the real normal equation \(W = [1 \mid z]\), weights
\(\mathrm{mask}\), complex right-hand side, solved per reflection.

A joint fit that would exceed `DEFAULT_MAX_OBJECT_BYTES` (64 MiB) raises.
`chunk_dataset_indices` is the scores path.

## Tests

[`tests/contrib/test_ensemble.py`](../../tests/contrib/test_ensemble.py):

- `test_register_is_idempotent`, `test_pydantic_accept_reject`,
  `test_client_import_is_torch_free`, `test_job_envelope_validates`
  (schema form: `inputs`/`outputs` are arrays).
- `test_known_f_matches_numpy_transcription`: torch vs an independent numpy
  transcription. On a full-rank draw (seed 2), the mode residual is
  \(2.3\times 10^{-16}\) and the ALS reconstruction residual is
  \(6.9\times 10^{-15}\). Tolerance \(10^{-6}\).
- `test_op_roundtrip_known_f`: `Bridge(memory=True, timeout=5)`.
- `test_chunk_and_cap`.
- `test_reference_oracles_absent` xfails with reason `reference/ not shipped`.
  The `cascan_pipeline.als_fit` comparison is that xfail.

## Deviations

- `reference/` is not in the tree. The numpy transcription lives in the test
  module. Stored note angles and likelihoods are not invented.
- The generated JobEnvelope JSON schema stores `inputs` and `outputs` as
  arrays. The pydantic model uses dicts; the schema test uses the array form.
- `write_synthetic_benchmark` raises `AssertionError` inside
  `miller.adopt_set` when a perturbed cell changes the reflection count.
  `multixtal` is unchanged. The CLI fit on a fixed-cell P1 campaign then
  hits `CCTBX_ASSERT(t_.den() == rhs.t_.den())` in
  `multixtal.indexing.reindexing_operators`. `test_cli_fit_on_pdb_campaign`
  xfails with that message. `test_cli_holdout_writes_s_post` runs the CLI
  on an npz and passes.
