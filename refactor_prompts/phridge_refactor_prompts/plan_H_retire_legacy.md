# Plan H — Retire `legacy/`

Branch: `refactor/H`. Two stages.

---

## Stage H1 — confirm replacements

**Do.**
1. For each file in `src/phridge/legacy/`, name its replacement:
   `intensity_engine.py` → `q/scales/solvent.py` + `q/predictive.py` + `report/`;
   `intensity_phenix_hook.py` → none (deleted without replacement);
   `intensity_refine.py`, `intensity_interleaved.py` → `policies/calibrate_intensity.py` + controller;
   `intensity_cli.py`, `intensity_tool.py` → `cli.py`;
   `client_joint.py` → `policies/stock_macrocycle.py`;
   `multixtal_pipeline.py` → controller + `transforms/observation/stacking.py` (Plan I);
   `ensemble_fit.py` → `r/generators/perturbation_modes/` + a policy (Plan I).
2. For each replacement that exists, run the legacy module's tests
   (`-m legacy`) and the replacement's tests side by side; record in
   `docs/legacy_retirement.md` a table: legacy file, replacement, tests that
   prove parity, status (`parity`, `superseded-no-parity-needed`, `pending-plan-I`).
3. Salvage before deletion: from `intensity_engine.py`, the k_sol/B_sol fit
   against the intensity objective (`IntensityScaleMixin`) must already be in
   `q/scales/solvent.py` — verify by test, not by reading. From `client_joint.py`,
   its weight heuristics must be the documented defaults of `stock_macrocycle.py`.

**Verify.** `python -m pytest -q -m legacy` (passes or is skipped, nothing errors); `cat docs/legacy_retirement.md`
**Commit.** `docs: legacy retirement table with parity evidence`
