# Legacy retirement (stage H1)

`src/phridge/legacy/` stays on the tree. Nothing outside that package may import it. The boundary is `tests/boundaries/test_no_legacy_imports.py`: an import of `phridge.legacy` is allowed only under `src/phridge/legacy/` or in a test module with `pytestmark = pytest.mark.legacy`.

`python -m pytest -q tests/boundaries` passed.

`python -m pytest -q -m legacy` collected and finished: **375 passed, 15 failed, 3 skipped, 3 xfailed**. The same 15 failures fail on `refactor/G` (parent of this branch). They are recorded on the rows they belong to. No new failure was introduced. Two test modules were both named `test_roundtrip.py`, which stops collection under pytest's default import mode; the shim module is now `tests/shims/cctbx/test_cctbx_roundtrip.py`.

`q/predictive.py` and `transforms/observation/stacking.py` are not in the tree and were not added.

| Legacy | Replacement | Tests that were run | Status |
| --- | --- | --- | --- |
| `intensity_engine.py` | `q/scales/solvent.py`, `report/` | `tests/q/test_bulk_solvent_nll.py` passed (`k_sol` / `B_sol` from `ml_i_bulk_solvent_fit`). Legacy side: `tests/q/test_bulk_solvent_engine.py` passed. `tests/test_intensity_engine.py` still fails `test_model_and_scale_updates`, `test_client_clean_import_no_torch`, `test_synthetic_scale_and_b_study`, `test_residual_scale_folded_after_apply_back_trace`, and `test_preconditioners_and_nu_toggles` (same failures on `refactor/G`). | `parity` |
| `intensity_phenix_hook.py` | none | Marked `legacy`. `tests/test_phenix_refine_hook.py` fails `test_clean_import_no_torch` and `test_weight_metric_helpers_and_patches` (same on `refactor/G`; `phenix` is not installed). | `superseded-no-parity-needed` |
| `intensity_refine.py` | `policies/calibrate_intensity.py`, `core/controller.py` | `tests/regression/test_e2_calibrate.py` and `tests/core/test_controller.py` passed. | `parity` |
| `intensity_interleaved.py` | `policies/calibrate_intensity.py`, `core/controller.py` | `tests/contrib/test_interleaved.py` fails `test_interleaved_matches_exact_end_to_end` (coordinate drift 0.041 Å against a 0.02 Å bound; same on `refactor/G`). The other tests in that module passed. | `parity` |
| `intensity_cli.py` | `cli.py` (`validate` / `refine` / `plugins` / `graph`) | Legacy CLI entry points in `tests/test_intensity_tool.py` fail `test_cli_entrypoint`, `test_cli_target_amplitude`, `test_cli_adam_refinement`, and `test_cli_lbfgs_refinement` on a Triton `tl.math.round` error inside structure-factor stamping (same on `refactor/G`). | `superseded-no-parity-needed` |
| `intensity_tool.py` | `cli.py` | Same Triton failures, plus `test_pipeline_end_to_end_from_files` (same on `refactor/G`). The shim imports the modules that still define its names and does not require the live package to import `phridge.legacy`. | `superseded-no-parity-needed` |
| `client_joint.py` | `policies/stock_macrocycle.py` | `tests/policies/test_stock.py` passed, including `weight_by_grad_norm is True`. Legacy side: `tests/test_joint_refinement.py` passed. | `parity` |
| `intensity_ll_ops.py` | none on the live plugin path | Still the worker for `ml_i_maps`, `ml_i_nuisance_fit`, and `ml_i_target_and_gradients`. Those ops register when the module is imported. `contrib/intensity_ll.register()` does not import it. Callers are legacy-marked, including `tests/d/test_intensity_maps_op.py`. | `superseded-no-parity-needed` |
| `multixtal_pipeline.py` | controller, once `transforms/observation/stacking.py` exists | That module is not in the tree. | `pending-plan-I` |
| `multixtal/` | controller, once `transforms/observation/stacking.py` exists | `tests/legacy/test_multixtal.py::test_synthetic_if_cctbx` and `tests/legacy/test_multixtal_damage.py::test_synthetic_unmerged_if_cctbx` fail (same on `refactor/G`). | `pending-plan-I` |
| `ensemble_fit.py` | `r/generators/perturbation_modes/` plus a policy | Known-F and hold-out checks call this module directly: `tests/r/test_ensemble.py::test_op_roundtrip_known_f` and `tests/r/test_ensemble_designs.py::test_cli_holdout_writes_s_post` passed. The live worker op raises until Plan I. | `pending-plan-I` |
| `__init__.py` | none | Package marker only. | `superseded-no-parity-needed` |
