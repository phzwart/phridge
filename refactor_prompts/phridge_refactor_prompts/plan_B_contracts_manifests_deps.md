# Plan B — Contracts package, manifests, dependency tiers

Branch: `refactor/B`. Four stages. Introduces the typed vocabulary every later
plan uses. No science changes.

---

## Stage B1 — `phridge.contracts` package

**Goal.** Move the generated pydantic models and the LinkML schemas under one
package and add the tensor contract types.

**Do.**

1. `mkdir src/phridge/contracts`. `git mv src/phridge/models.py src/phridge/contracts/models.py`.
   Create `src/phridge/contracts/__init__.py` that re-exports everything
   `models.py` exported (`from .models import *` plus an explicit `__all__`
   copied from models). Fix all importers (`grep -rn "phridge.models\|from phridge import models"`).
2. Move the LinkML sources: `git mv schema src/phridge/contracts/schema`.
   Update `pyproject.toml`: `[tool.hatch.build.targets.sdist].include` and the
   `force-include` entry for `schema` (now inside the package, drop the
   force-include; keep the stamp kernel force-includes). Update
   `scripts/generate_models.py` and `tests/test_linkml_validate.py` paths.
   Update README paths with a search-and-replace of `schema/` → `src/phridge/contracts/schema/`.
3. Create `src/phridge/contracts/tensors.py` with these `typing.NamedTuple`
   classes (fields are `torch.Tensor` or `Optional[torch.Tensor]`; import torch
   lazily under `TYPE_CHECKING` plus `Any` at runtime so the module imports
   without torch):
   - `RObject(sites_frac, u_iso, u_star, occupancy, fp, fdp, scatterer_type: LongTensor, uij_defined: BoolTensor)`
     with docstring: "Named tensors describing an atom list. Never an xray_structure."
   - `DensityObject(values, origin, spacing, cell)` for continuum components.
   - `ComplexArray(hkl: LongTensor[N,3], values: ComplexTensor[N])`
     docstring: "Complex coefficients over a Miller table the sandbox owns."
   - `Predictive(fields: dict[str, Tensor])` — a thin wrapper with
     `.require(*names)` that raises `KeyError` naming the missing field, and
     `.has(name)`.
   - `Observed(index: dict[str, LongTensor], data: dict[str, Tensor], partitions: dict[str, BoolTensor])`.
   Each class gets a `validate()` method checking shapes agree on the leading
   axis. No other logic.
4. Create `tests/contracts/test_tensors.py`: construct each type from random
   tensors, `validate()` passes; mismatched leading axis raises; `Predictive.require`
   raises KeyError naming the field. Skip the whole module if torch is absent.

**Verify.**
```
python -m pytest -q tests/test_schema_roundtrip.py tests/test_linkml_validate.py tests/test_register_op.py tests/contracts
python -m pytest -q tests/boundaries
```

**Commit.** `refactor: phridge.contracts package with schema, models and tensor contracts`

---

## Stage B2 — plugin manifest

**Goal.** Add `PluginManifest` and make the existing `OpSpec` a manifest
subtype so current ops register unchanged.

**Do.**

1. Create `src/phridge/contracts/manifest.py`:
   ```python
   class Slot(str, Enum): r_component, restraint_provider, q_component, fourier, observation, builder, scoring_rule, optimizer, policy, statistic, io_adapter
   class Space(str, Enum): R, Q, D, transform
   class Tier(str, Enum): shim, hybrid, native
   class BlockDecl(_Strict): id: str; shape: list[int|str]; bounds: Optional[tuple[float,float]]; transform: Literal["identity","log","logit"] = "identity"; frozen: bool = False
   class PluginManifest(_Strict):
       name: str; slot: Slot; space: Space
       from_space: Optional[Space] = None; to_space: Optional[Space] = None   # required when space == transform
       reads: list[str] = []; produces: list[str] = []
       blocks: list[BlockDecl] = []
       needs: list[str] = []          # capability names required of neighbours, e.g. "predictive.sigma_w", "objective.hvp"
       provides: list[str] = []       # capability names offered, e.g. "adjoint", "hvp", "block_diag"
       tier: Tier = Tier.native; dtype: Literal["float32","float64"] = "float64"; device: Literal["cpu","cuda","mps","any"] = "any"
       version: str = "0"
   ```
   with a `@model_validator` that requires `from_space`/`to_space` iff
   `space == transform`, and forbids `space == transform` for slots other than
   `fourier, observation, builder`.
2. In `contracts/models.py`, make `OpSpec` carry an optional
   `manifest: Optional[PluginManifest] = None`. Do not change its other fields.
3. In `src/phridge/ops.py`, add `register_manifest(m: PluginManifest, impl=None)`
   that stores manifests in a `_MANIFESTS: dict[str, PluginManifest]` registry
   and, when `impl` is given, also calls `register_impl`. Add `get_manifest`,
   `list_manifests`, `manifests_for_slot(slot)`. Existing `register_op` is
   untouched.
4. Add LinkML for `PluginManifest` in `contracts/schema/phridge.yaml` and
   regenerate: `python scripts/generate_models.py --all`. If generation changes
   `models.py` in ways beyond adding the new classes, STOP and report.
5. Tests: `tests/contracts/test_manifest.py` — valid transform manifest
   accepts; transform without `from_space` rejects; `scoring_rule` with
   `space=transform` rejects; `register_manifest` then `manifests_for_slot`
   round-trips; `jsonschema.validate` of a manifest dict against the generated
   schema.

**Verify.**
```
python -m pytest -q tests/contracts tests/test_register_op.py tests/test_linkml_validate.py tests/boundaries
```

**Commit.** `contracts: PluginManifest with slots, spaces, tiers; registry hooks`

---

## Stage B3 — dependency tiers in `pyproject.toml`

**Goal.** Make the four dependency tiers explicit and make `agentsg` a shim.

**Do.**

1. `[project.dependencies]` = exactly `numpy`, `torch`, `pydantic>=2.4`, `redis>=5.0`
   (torch moves from optional to core; the spec is torch-native).
2. `[project.optional-dependencies]`: one extra per science shim and tool set:
   `cctbx = ["cctbx-base"]`, `agentsg = [...]` (keep the git URL),
   `finufft = ["pytorch-finufft"]`, `triton = ["triton"]`, `numba = ["numba"]`,
   `tools = ["matplotlib"]`, `dev = ["pytest>=7.4","fakeredis>=2.23","linkml>=1.8","jsonschema"]`.
   Remove `client`, `worker`, `nufft` extras (they are now core or duplicated).
   Update README install lines accordingly (search `pip install -e`).
3. Create `src/phridge/shims/__init__.py` (empty) and `src/phridge/shims/agentsg/__init__.py`
   with a guarded import and a `SymmetryProvider`-shaped surface:
   `available() -> bool`, `niggli_reduce(cell: Tensor[6]) -> Tensor[6]`,
   `root_invariant_distance(cell_a, cell_b) -> float`,
   `reindex_branches(cell_a, cell_b) -> list[Tensor[3,3]]`. Each function
   imports `agentsg` inside the body and raises `RuntimeError("install phridge[agentsg]")`
   if missing. Map each to the agentsg call currently used in
   `src/phridge/sfcalc/engine/sg_gather.py` and `engine.py`; move those usages
   to call the shim. Add a `PluginManifest(name="agentsg_symmetry", slot=... )`
   — use slot `io_adapter` for now (a `symmetry_provider` slot is added in Plan D).
4. Move `tests/test_agentsg_plugin.py` → `tests/shims/test_agentsg.py`; it must
   skip cleanly when `agentsg` is not installed.
5. Add `tests/boundaries/test_dependency_tiers.py`: parse `pyproject.toml`,
   assert core deps are exactly the four; assert every top-level non-stdlib
   import found by `iter_modules` (reuse A1's helper) in `core/ r/ q/ d/ transforms/ optim/ policies/ report/`
   is in the core set; assert `agentsg` is imported only under `shims/agentsg/`.

**Verify.**
```
python -m pytest -q tests/boundaries tests/shims tests/test_asu_gather.py
pip install -e . --dry-run 2>&1 | tail -3
```

**Commit.** `deps: four dependency tiers; agentsg behind shims/agentsg`

---

## Stage B4 — conformance test scaffolding

**Goal.** Create the importable conformance package the spec requires, with
one empty suite per slot. Plan D fills them.

**Do.**

1. Create `src/phridge/testing/__init__.py` and
   `src/phridge/testing/conformance/__init__.py` with
   `suite_for(slot: Slot) -> type` returning the suite class for a slot.
2. One module per slot under `src/phridge/testing/conformance/`:
   `r_component.py, restraint_provider.py, q_component.py, fourier.py,
   observation.py, builder.py, scoring_rule.py, optimizer.py, policy.py,
   statistic.py, io_adapter.py`. Each defines an abstract
   `class <Slot>Suite:` with `plugin: PluginManifest`, `make() -> Any`
   (abstract) and at least `test_manifest_valid(self)` which asserts the
   manifest's slot matches and `reads/produces` are non-empty where the slot
   requires it. Add `TOLERANCES: dict[str, float]` per module (empty for now).
3. `tests/conformance/__init__.py` and `tests/conformance/test_scaffold.py`:
   `suite_for` returns a class for every `Slot`; each suite has `test_manifest_valid`.
4. Pyproject: make `phridge.testing` part of the wheel (it is under `src/`, so
   it is by default; just add a test that `import phridge.testing.conformance` works).

**Verify.**
```
python -m pytest -q tests/conformance tests/boundaries
```

**Commit.** `testing: conformance suite scaffolding, one suite per slot`
