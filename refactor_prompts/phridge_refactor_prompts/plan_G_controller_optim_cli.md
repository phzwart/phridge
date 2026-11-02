# Plan G — Controller, optimizers, policies, builders, CLI

Branch: `refactor/G`. Five stages. Ends with configs A and B running end to
end from one command and producing depositable files through `io/`.

---

## Stage G1 — optimizers

**Do.**
1. `src/phridge/optim/base.py`: `class Optimizer(Protocol)`: `requires: tuple[str,...]`
   (any of `"grad","hvp","preconditioner"`), `step(objective, x0, bounds, budget: Budget) -> StepResult(x1, trace, n_evals)`.
   `Budget(iterations: int|None, seconds: float|None)`.
2. `optim/lbfgs.py`: L-BFGS-B via `torch.optim.LBFGS` with bounds handled by
   the registry's transforms (`requires=("grad",)`).
3. `optim/newton_cg.py`: move `_preconditioned_cg` from `sfcalc/client.py`
   here; `requires=("grad","hvp","preconditioner")`.
4. `optim/gauss_newton.py`: GN using the Fourier operator's `gauss_newton_blocks`
   as block-diagonal preconditioner (code exists in the ten_eyck engine);
   `requires=("grad","hvp","preconditioner")`.
5. `optim/grid.py`: one-block grid search with a user criterion (`requires=()`),
   used by `nu_screen`.
6. Manifests (`slot=optimizer`), `conformance/optimizer.py`:
   `test_decreases_quadratic`, `test_respects_bounds`, `test_refuses_missing_capability`.
7. Delete what remains of `src/phridge/sfcalc/client.py` after moving
   `RemoteRefinementTarget`'s objective wrapper logic into `core/objective.py`
   (if not already) and `StructureFactorServer` into `worker/client.py`.
   `src/phridge/sfcalc/` must now be empty; delete it.

**Verify.** `python -m pytest -q tests/optim tests/conformance tests/boundaries; ls src/phridge/sfcalc 2>&1 | head -1`
**Commit.** `optim: optimizer slot with L-BFGS, Newton-CG, Gauss-Newton, grid`

---

## Stage G2 — controller state machine

**Do.**
1. `src/phridge/core/controller.py`: states
   `Load, Configure, Calibrate, Ready, Step, Evaluate, Report, Reconfigure, Finalize, Done`;
   actions `configure, calibrate, step, evaluate, accept, reject, freeze, unfreeze, report, continue_, reconfigure, finalize, query, propose, apply`;
   guards exactly as the spec's state-machine section: `step` refused on a
   refused move set or failed audit rule; `report` twice without an accepted
   step refused; `step` after `report` refused until `continue_`; `reconfigure`
   refused from Step/Evaluate; `propose` only from Ready; `apply` only after a
   `propose` and recorded as a structural event.
   Every action appends a `Receipt` and returns an `Observation` pydantic model
   (scalars, per-shell tables, counts, versions; never arrays).
2. `tests/core/test_controller.py` with a toy graph: every guard; the receipt
   log replays to the same final block values (CPU float64, bitwise);
   observations contain no tensors (walk the model and assert).

**Verify.** `python -m pytest -q tests/core`
**Commit.** `core: controller state machine with guards, receipts, observations`

---

## Stage G3 — policies and the minimal loop

**Do.**
1. `src/phridge/policies/base.py`: `class Policy(Protocol)`: `next(obs_history) -> Action | None`.
2. `policies/stock_macrocycle.py`: N cycles of (calibrate scales) → step
   `sites` with L-BFGS → step `u_iso` with L-BFGS → report; weights by
   gradient-norm ratio as an objective option (document that this is the
   stock default and the open design question lives in the spec).
3. `policies/calibrate_intensity.py` (from Plan E) adopts the `Policy` protocol.
4. `src/phridge/io/`: `read_model.py` (iotbx.pdb + mmtbx.model → `RObject`
   via `shims/cctbx`), `read_data.py` (reflection file → `Observed`; intensity
   or amplitude kinds; flags), `write_model.py`, `write_mtz.py`. These are the
   only modules importing iotbx/mmtbx besides shims.
5. `src/phridge/api.py`: `Refinement.from_config(path)`, `.run(policy)`, `.finalize(out_dir)`.
6. `tests/integration/test_minimal_loop.py` (marked `slow`): run
   `examples/configs/amplitude_ml.yaml` and `intensity_mli.yaml` on the
   lysozyme example for 2 cycles; both produce a PDB/mmCIF and MTZ; the receipt
   log replays; the amplitude run's R_free change per cycle is reported in the
   test output (compared against `phenix.refine` manually, not in the test).

**Verify.** `python -m pytest -q tests/policies tests/io tests/core; python -m pytest -q -m slow tests/integration/test_minimal_loop.py`
**Commit.** `policies, io, api: minimal loop runs configs A and B end to end`

---

## Stage G4 — builders

**Do.**
1. `src/phridge/transforms/build/base.py`: `class Builder(Protocol)`:
   `propose(maps: dict[str, Tensor], stats: dict, selection) -> RChange` where
   `RChange(add: Optional[RObject], remove: BoolTensor, reinit: dict[str, Tensor])`;
   never applies.
2. `transforms/build/water.py`: peak search on the difference map + distance
   and B filters, lifted from `mmtbx.solvent.ordered_solvent` criteria but
   implemented in torch; cctbx is used only to supply vdW radii via `shims/cctbx/`.
3. `transforms/build/rotamer.py`: rotamer fitting on a selection using the
   rotamer library through `shims/cctbx/rotamers.py` (new shim, `tier=hybrid`:
   library loaded once, scoring in torch).
4. Controller: `apply(RChange)` performs the structural event through the
   registry (`reshape`) and bumps the atom-list node.
5. `conformance/builder.py`: `test_propose_does_not_mutate`, `test_idempotent_on_unchanged_map`,
   `test_apply_then_revert_bitwise`.

**Verify.** `python -m pytest -q tests/transforms tests/conformance tests/boundaries`
**Commit.** `transforms/build: Builder slot with water and rotamer plugins`

---

## Stage G5 — CLI, AGENTS.md, size lint

**Do.**
1. `src/phridge/cli.py` with `phridge refine <config.yaml> [--policy ...] [--out DIR]`,
   `phridge validate <config.yaml>`, `phridge plugins list`, `phridge graph <config.yaml> --node <name>` (prints `stale`, `cost`, `readers`). Register as `[project.scripts] phridge = "phridge.cli:main"`.
2. Rewrite repo-root `AGENTS.md` from the spec's "Repo-wide rules": import walls
   as tests, manifest before code, one slot one suite, size caps (800/150),
   no loops outside `core/controller.py` and `policies/`, tensors in/out,
   pydantic at boundaries, receipts not logging, examples hold inputs only,
   legacy may not gain a line. Mirror to `.cursor/rules/phridge.mdc`.
3. `tests/boundaries/test_size_caps.py`: no module > 800 lines, no function
   > 150 lines, outside `legacy/`. `tests/boundaries/test_no_loops_outside_controller.py`:
   grep for `optimizer.step(` / `.step(objective` outside `core/` and `policies/`.
4. Update `README.md` top section: what phridge is now (torch-native refinement
   object; the bridge is `phridge.worker`), install extras, the `phridge` CLI.

**Verify.** `python -m pytest -q tests/boundaries; phridge validate examples/configs/intensity_mli.yaml`
**Commit.** `cli: phridge command; AGENTS.md; size and loop lints`
