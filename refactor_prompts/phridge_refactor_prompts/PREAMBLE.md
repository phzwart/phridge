# PREAMBLE — paste this above every stage prompt

You are refactoring the Python package `phridge` (repo root has `pyproject.toml`,
`src/phridge/`, `tests/`, `schema/`). Work ONLY on the files the stage names.

Rules that apply to every stage:

- Make the smallest change that satisfies the stage. Do not improve, rename,
  reformat or "clean up" anything the stage does not name.
- Never import `phenix` or any `phenix.*` module anywhere.
- `cctbx`, `mmtbx`, `iotbx`, `scitbx`, `libtbx` may be imported ONLY under
  `src/phridge/io/`, `src/phridge/shims/`, `src/phridge/worker/cctbx_runner.py`
  and `src/phridge/legacy/`. Everywhere else is forbidden.
- `torch` may NOT be imported under `src/phridge/io/` or `src/phridge/shims/cctbx/`.
- Every module starts with `from __future__ import annotations`. Public
  functions have full type annotations. No `**kwargs` at package boundaries.
  Options that cross a boundary are pydantic `BaseModel` with `extra="forbid"`.
- Tensors in, tensors out: public functions in `r/`, `q/`, `d/`, `transforms/`,
  `optim/` take and return torch tensors or the named-tuple types in
  `phridge.contracts.tensors`. Never a cctbx object. Never numpy on a per-step path.
- When moving a module, use `git mv`, fix imports with a search over the whole
  tree (`grep -rn "old.path"`), move its tests alongside, and leave NO
  re-export shim behind unless the stage says so.
- Do not touch anything under `src/phridge/legacy/` except to move things into
  it. Fixes go into the replacement, never into legacy.
- Modules over 800 lines and functions over 150 lines are not allowed in new
  or moved code outside `legacy/`.
- Run the stage's Verify block exactly as written. Report the output. If it
  fails, fix only what is needed and re-run. Do not mark the stage done until
  it passes.
- Finish with: `git add -A && git commit -m "<the commit message the stage gives>"`.
- If a stage instruction conflicts with what you find in the tree, STOP and
  report the conflict with file paths and line numbers. Do not improvise.

Tests: `python -m pytest -q -m "not gpu and not slow"` is the standard run.
If cctbx or torch are missing in your environment, run
`python -m pytest --co -q` to at least prove collection works, and say so.
