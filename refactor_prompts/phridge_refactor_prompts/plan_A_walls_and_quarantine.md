# Plan A — Walls, quarantine, dead shims

Branch: `refactor/A`. Three stages. No behaviour changes; only tests added,
files moved, files deleted.

---

## Stage A1 — import-boundary tests (they must fail first)

**Goal.** Add `tests/boundaries/test_import_walls.py` that walks every
`.py` under `src/phridge/` with `ast`, collects `import` / `from ... import`
targets, and asserts four rules. Run it; it is expected to FAIL on the current
tree (that proves it works). Do not fix the failures in this stage.

**Do.**

1. Create `tests/boundaries/__init__.py` (empty) and
   `tests/boundaries/test_import_walls.py`.
2. Implement `iter_modules(root) -> list[tuple[Path, set[str]]]` returning each
   module path and the set of top-level imported package names (first dotted
   component), using `ast.parse`; skip `__pycache__`.
3. Rules (each its own test function so failures are named):
   - `test_no_phenix_anywhere`: no module imports `phenix`. Exempt: nothing.
   - `test_cctbx_only_behind_doors`: modules importing any of
     `cctbx, mmtbx, iotbx, scitbx, libtbx` must live under
     `src/phridge/io/`, `src/phridge/shims/`, `src/phridge/worker/cctbx_runner.py`
     or `src/phridge/legacy/`.
   - `test_no_torch_in_io_or_cctbx_shims`: modules under `src/phridge/io/` or
     `src/phridge/shims/cctbx/` do not import `torch`.
   - `test_core_tier_only_in_spaces`: modules under `src/phridge/core/`,
     `r/`, `q/`, `d/`, `transforms/`, `optim/`, `policies/`, `report/` import
     only from: stdlib, `torch`, `numpy`, `pydantic`, `redis`, `phridge`.
     (Use `sys.stdlib_module_names`.) Directories that do not exist yet pass
     trivially.
4. A module-level constant `LEGACY_EXEMPT = ("src/phridge/legacy/",)` used by
   rule 2 only. Plan H deletes it.
5. Add marker `boundaries` to `pyproject.toml` `[tool.pytest.ini_options].markers`
   and mark the module `pytestmark = pytest.mark.boundaries`.

**Don't.** Do not modify any file under `src/`.

**Verify.**
```
python -m pytest -q tests/boundaries
```
Expected: `test_no_phenix_anywhere` FAILS listing
`src/phridge/client/intensity/phenix_hook.py` (and only that file);
`test_cctbx_only_behind_doors` FAILS listing many files; the other two PASS.
Paste the failing file lists into the commit message body.

**Commit.** `tests: add import-boundary rules (currently failing by design)`

---

## Stage A2 — quarantine the Phenix-era loops, delete dead shims

**Goal.** Create `src/phridge/legacy/` and move the code that exists only to
run inside `phenix.refine` or that duplicates a loop. Delete star-import shims.
After this stage, `test_no_phenix_anywhere` passes.

**Do.**

1. `mkdir src/phridge/legacy` with an `__init__.py` whose docstring is exactly:
   `"""Quarantined code awaiting replacement. Do not add lines here. See refactor plan."""`
2. `git mv` these into `src/phridge/legacy/` keeping file names, flattening
   directories (`client/intensity/engine.py` → `legacy/intensity_engine.py`,
   etc.). Prefix with the original package to keep names unique:
   - `src/phridge/client/intensity/engine.py` → `legacy/intensity_engine.py`
   - `src/phridge/client/intensity/phenix_hook.py` → `legacy/intensity_phenix_hook.py`
   - `src/phridge/client/intensity/refine.py` → `legacy/intensity_refine.py`
   - `src/phridge/client/intensity/interleaved.py` → `legacy/intensity_interleaved.py`
   - `src/phridge/client/intensity/cli.py` → `legacy/intensity_cli.py`
   - `src/phridge/client/intensity_tool.py` → `legacy/intensity_tool.py`
   - `src/phridge/client/joint.py` → `legacy/client_joint.py`
   - `src/phridge/contrib/multixtal/pipeline.py` → `legacy/multixtal_pipeline.py`
   - `src/phridge/contrib/ensemble/fit.py` → `legacy/ensemble_fit.py`
3. Fix every import of the moved modules across `src/` and `tests/`
   (`grep -rn "client.intensity.engine\|client.intensity.phenix_hook\|client.intensity.refine\|client.intensity.interleaved\|client.intensity.cli\|client.intensity_tool\|client.joint\|multixtal.pipeline\|ensemble.fit"`).
   Importers that are themselves staying (e.g. `client/intensity/__init__.py`,
   `contrib/multixtal/cli.py`, `contrib/ensemble/cli.py`, `contrib/ensemble/__init__.py`)
   must import from `phridge.legacy.<name>` for now — this is the ONE place a
   non-legacy module may import legacy, and Plan H removes it.
4. Delete (`git rm`) the star-import compatibility shims:
   `src/phridge/worker/targets/` (whole dir), `src/phridge/worker/xtal/` (whole dir),
   `src/phridge/worker/ops/xtal_ops.py`, `src/phridge/client/xtal_engine.py`.
   Fix any importer to use `phridge.sfcalc.targets`, `phridge.sfcalc.engine`,
   `phridge.sfcalc.ops`, `phridge.sfcalc.client` directly.
5. Delete the empty `src/phridge/contrib/twinned/` directory.
6. In `pyproject.toml`, remove `[project.scripts]` entries
   `phridge-intensity` and `phridge-refine` (both point at legacy).
7. Mark tests that import from `legacy`: add `pytestmark = pytest.mark.legacy`
   at the top of `tests/test_phenix_refine_hook.py`,
   `tests/test_intensity_engine.py`, `tests/test_intensity_tool.py`,
   `tests/test_joint_refinement.py`, `tests/contrib/test_interleaved.py`,
   `tests/contrib/test_ensemble.py` (only if it imports `fit`), and any test
   that `grep -ln "legacy" tests/` finds. Add marker `legacy` to pyproject.

**Don't.** Do not edit the content of any moved file beyond its import lines.
Do not move `client/intensity/model.py`, `maps.py`, `omit.py`, `math_utils.py`,
`stats_report.py`, `nll_log.py`, `heartbeat.py` — those relocate in Plan D.

**Verify.**
```
python -m pytest -q tests/boundaries/test_import_walls.py::test_no_phenix_anywhere
python -m pytest --co -q
python -m pytest -q -m "not gpu and not slow and not legacy"
```
Expected: first command PASSES; collection has no errors; third command passes
with the same pass count as before this stage minus the legacy-marked tests.

**Commit.** `refactor: quarantine Phenix-era loops under legacy/, delete dead shims`

---

## Stage A3 — multixtal tests off the pipeline

**Goal.** `tests/contrib/test_multixtal.py` and `test_multixtal_damage.py`
must not depend on `legacy/multixtal_pipeline.py`. Rewrite them against the
operator-level functions so the pipeline can be deleted in Plan H.

**Do.**

1. `grep -n "pipeline" tests/contrib/test_multixtal*.py`. For each use, find
   the operator-level function the pipeline calls (in `contrib/multixtal/op.py`,
   `dose_model.py`, `damage.py`, `anom_damage.py`, `nuisance.py`, `indexing.py`)
   and call that directly with the same fixtures.
2. If a test genuinely exercises end-to-end orchestration and nothing else,
   move it to `tests/legacy/test_multixtal_pipeline.py` and mark it `legacy`.
3. Remove the `legacy` marker from any multixtal test that no longer imports legacy.

**Don't.** Do not change `contrib/multixtal/*.py`.

**Verify.**
```
grep -rn "legacy" tests/contrib/test_multixtal*.py ; echo "exit=$?"   # expect no matches, exit=1
python -m pytest -q tests/contrib/test_multixtal.py tests/contrib/test_multixtal_damage.py
```

**Commit.** `tests: multixtal tests target operators, not the quarantined pipeline`
