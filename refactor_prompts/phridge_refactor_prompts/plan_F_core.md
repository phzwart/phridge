# Plan F — Core: registry, move sets, graph, objective, partitions, config, receipts

Branch: `refactor/F`. Five stages of new code. Pure Python + torch; no cctbx.
Each stage has unit tests on toy inputs before anything scientific is wired.

---

## Stage F1 — parameter registry and blocks

**Do.**
1. `src/phridge/core/registry.py`:
   ```python
   @dataclass
   class Block: id: str; component: str; space: Space; shape: tuple[int,...]; bounds: Optional[tuple[Tensor,Tensor]]; transform: Literal["identity","log","logit"]; frozen: bool; value_version: int; structural_version: int; value: Tensor
   class Registry:
       def register(self, block: Block) -> None            # refuses duplicate id
       def block(self, id) -> Block
       def set_value(self, id, value: Tensor) -> None        # bumps value_version; shape must match
       def reshape(self, id, new_shape, new_value) -> None  # bumps structural_version
       def freeze(self, id) / unfreeze(self, id)
       def epoch(self) -> int ; def bump_epoch(self) -> None
       def pack(self, entries: list[Entry]) -> Packed       # Entry = (block_id, view_mask|None); Packed = (vector, slices, bounds, transform_ops)
       def unpack(self, packed_vector, entries) -> None     # writes values, bumps versions of touched blocks only
   ```
   Transforms (`log`, `logit`) are applied in `pack`/`unpack` so optimizers see
   unconstrained variables; store the inverse in `Packed`.
2. `tests/core/test_registry.py`: duplicate id refused; `set_value` with wrong
   shape refused; `pack`/`unpack` round-trip bitwise for identity transform and
   to 1e-12 for log/logit; only touched blocks' versions bump; `reshape` bumps
   structural but not other blocks; freeze excludes from pack.

**Verify.** `python -m pytest -q tests/core tests/boundaries`
**Commit.** `core: parameter registry with blocks, packing, value/structural versions`

---

## Stage F2 — move sets and views

**Do.**
1. `src/phridge/core/moveset.py`:
   ```python
   class View(BaseModel): block: str; selection: Optional[str] = None; index: Optional[list[int]] = None; predicate: Optional[str] = None
   class Entry(BaseModel): block: str; view: Optional[View] = None; bounds: Optional[tuple[float,float]] = None; scale: float = 1.0; frozen: bool = False
   class MoveSet(BaseModel): name: str; entries: list[Entry]; coupled: list[list[str]] = []
   def resolve(ms: MoveSet, registry: Registry, selector: Callable[[str], BoolTensor]) -> list[ResolvedEntry]   # mask per entry
   def validate(ms, registry, graph_capabilities) -> None   # raises MoveSetRefused(entry, reason)
   ```
   Refusals: frozen block; structural version changed since the given
   `ready_epoch`; generator-plus-output overlap (needs a `generator_outputs`
   map argument from components); missing adjoint capability.
   `selection` strings are resolved by an injected `selector` callable
   (the cctbx selection language will be the first provider in `shims/cctbx/`;
   here it is just a callable).
2. `tests/core/test_moveset.py`: toy registry with four blocks; each refusal
   case; view by index and by predicate; `scale` appears in `Packed`.

**Verify.** `python -m pytest -q tests/core`
**Commit.** `core: move sets with views, per-entry options and refusal rules`

---

## Stage F3 — dependency graph with version stamps

**Do.**
1. `src/phridge/core/graph.py`:
   ```python
   class Node: name; kind: Literal["block","derived"]; reads: list[str]; compute: Optional[Callable]; value_version; structural_version; last_input_versions: dict[str, tuple[int,int]]; cached: Any
   class Graph:
       def from_manifests(self, manifests: list[PluginManifest], registry: Registry) -> Graph   # edges = union of reads
       def get(self, name) -> Any                     # pull-based; recompute stale chain only
       def stale(self, name) -> list[str]             # which inputs moved
       def affected(self, blocks_or_moveset) -> set[str]
       def readers(self, name) -> set[str]
       def cost(self, name) -> list[str]              # the chain that would recompute
       def structural_bump(self, name) ; def rebuild(self) # epoch
   ```
   Staleness: any input's current version > recorded, or epoch moved.
   Structural bumps clear a `shape_cache` dict on downstream nodes.
2. Receipts: `src/phridge/core/receipts.py` — append-only list of
   `Receipt(action, node, inputs_versions, output_version, wall_time, partitions_consulted)`;
   `Graph.get` emits one per recompute.
3. `tests/core/test_graph.py` (arithmetic toy graph, no physics):
   recompute count equals the stale chain exactly; unrelated branch untouched;
   structural bump drops caches downstream only; placement change (`rebuild`)
   recomputes everything; `readers`, `affected`, `cost` correct on the toy;
   receipts count matches recomputes.

**Verify.** `python -m pytest -q tests/core`
**Commit.** `core: dependency graph with version stamps, pull-based recompute, receipts`

---

## Stage F4 — objective and partitions

**Do.**
1. `src/phridge/core/partitions.py`: `Partitions(scheme: "two"|"four"|"halfmap", fractions, seed, unit: "reflection"|"particle"|"crystal_consistent")`
   producing `dict[str, BoolTensor]` masks over an `Observed`; `audit_rule(graph) -> None`
   raises if `readers("partition.audit") != {"statistics"}`; merge rules
   (calibration→tune allowed, audit never).
2. `src/phridge/core/objective.py`: `Objective(graph, moveset_resolved, score_node, restraint_nodes, weights, mask)`
   with `value(x)`, `grad(x)`, optional `hvp(x, v)` and `preconditioner()` —
   each sets block values via `registry.unpack`, pulls `graph.get(score)` and
   restraint nodes, masks and scales gradients per entry, and chains generator
   Jacobians (a node that `produces` an atom list exposes `adjoint`).
   `hvp` is available iff every node on the path declares `provides: ["hvp"]`.
3. `tests/core/test_partitions.py`, `tests/core/test_objective.py` on a toy
   quadratic "score" node and a toy "restraint" node: gradient equals analytic;
   masked entries have zero gradient; `hvp` unavailable when a node lacks it;
   audit rule refuses a graph where a fit node reads `partition.audit`.

**Verify.** `python -m pytest -q tests/core`
**Commit.** `core: objective over move sets; partitions and the audit rule`

---

## Stage F5 — configuration loader

**Do.**
1. `src/phridge/core/config.py`: pydantic models mirroring the spec's YAML
   (`components`, `granularity`, `partitions`, `controller`, `observed`,
   `move_sets`), `load(path) -> Config`, `build(config) -> (Registry, Graph, Partitions, components)`:
   instantiates plugins by `impl` path, validates placement (`reads`/`produces`
   closure; wall rule via transform `from`/`to`; `needs` ⊆ produced Predictive
   fields; audit rule), raises `ConfigRefused(path, reason)`.
2. Two example configs under `examples/configs/`: `amplitude_ml.yaml`
   (stock path: French–Wilson preprocessing, `ml_f`, two partitions) and
   `intensity_mli.yaml` (intensity path: mask in R, `wilson_aniso`,
   `sigma_a_beta`, `nu_screen`, `mli_quad`, four partitions), using the
   manifests from Plans D–E.
3. `tests/core/test_config.py`: both examples build; a config wiring audit into
   a fit is refused with the right path; a rule whose `needs` are not produced
   is refused; a Q component reading an R node directly is refused.

**Verify.** `python -m pytest -q tests/core tests/boundaries`
**Commit.** `core: configuration loader and validation; two example configs`
