# Extending phridge (external ops and workers)

phridge is meant to be a **protocol + runtime**, not a closed app. Your own
package can add ops without forking this repo. Redis remains the server;
`phridge-worker` / `phridge-cctbx-worker` are the consumers; Phenix or a
torch driver talks through `Bridge`.

There is no separate “phridge server” process — start `redis-server` and
workers that share the URL. Ops are routed by `OpSpec.runtime` (`torch` or
`cctbx`) onto separate Redis streams so workers never steal each other's
jobs. See [redis.md](redis.md).

## Register an op

Prefer payloads that already exist (`array`, `json`, or packed types such as
`MillerArray`). New science object kinds still need schema + codec work
inside phridge; new **ops** do not.

```python
# mypkg/plugin.py
import numpy as np
from phridge.ops import register_op

def denoise(array, strength=0.1):
    # torch / your model here
    return array  # same shape

def register():
    register_op(
        "denoise",
        denoise,
        inputs={"array": "array", "strength": "json"},
        outputs={"array": "array"},
        runtime="torch",  # default; use "cctbx" for mmtbx-side ops
    )

# Import-time registration (works with --preload)
register()
```

`register_op` updates the shared **OpSpec catalog** (so `Bridge` will enqueue
the name) and optionally the **worker implementation**. The impl is stored
without importing `phridge.worker`, so a Phenix client can import your plugin
and stay torch-free if the callable itself does not pull torch at import
time — or register catalog-only on the client and full `register_op(..., impl)`
on the worker via the same module’s `register()`.

Lower-level helpers: `register(OpSpec(...))` and `register_impl(name, fn)`.

## Run a worker that loads your package

```bash
# discovery options (any combination):
phridge-worker --redis-url redis://gpu:6379/0 --preload mypkg.plugin
PHRIDGE_PRELOAD=mypkg.plugin phridge-worker --device cuda
phridge-worker --runtime cctbx   # or: phridge-cctbx-worker

# dump catalog including plugins
phridge-ops --preload mypkg.plugin
```

| Flag / env | Role |
|------------|------|
| `--runtime torch\|cctbx` | Job stream / impl table (`PHRIDGE_RUNTIME`) |
| `--preload MOD` | Import module(s) before serving (repeatable; comma-separated ok) |
| `PHRIDGE_PRELOAD` | Same as `--preload` |
| `--no-entry-points` | Skip `phridge.ops` and `phridge.targets` entry-point groups |

Entry points (optional) so `pip install mypkg` is enough:

```toml
# mypkg pyproject.toml
[project.entry-points."phridge.ops"]
denoise = "mypkg.plugin:register"

# reciprocal-space targets (register_target):
[project.entry-points."phridge.targets"]
my_nll = "mypkg.my_likelihood:register"
```

`phridge-worker` and `phridge-ops` call each entry point (zero-arg) at
startup for **both** groups, then apply `--preload` (preload wins on name
clashes).

## Package map: `phridge.sfcalc` vs `phridge.contrib`

| Package | Role |
|---------|------|
| [`phridge.sfcalc`](../src/phridge/sfcalc/) | Structure-factor **core**: FFT engine, built-in targets (`ls`, `ml_f`), SF ops, packing, `StructureFactorServer` |
| [`phridge.contrib`](../src/phridge/contrib/) | Optional in-tree specialized targets/ops (e.g. `intensity_ll` → `ml_i`) |

Preferred imports:

```python
from phridge.sfcalc import StructureFactorServer
from phridge.sfcalc.targets import Target, register_target
```

Compatibility shims still exist (`phridge.worker.targets`, `phridge.client.xtal_engine`, …).

### In-tree contrib subpackages

Add specialized science under `src/phridge/contrib/<name>/` with a zero-arg
`register()` and a `phridge.targets` (or `phridge.ops`) entry point in
`pyproject.toml`. **Mandatory agent rules** (typing, pydantic options,
LinkML/jsonschema when schema changes) live in
[`src/phridge/contrib/AGENTS.md`](../src/phridge/contrib/AGENTS.md).

Built-in contrib example: `phridge.contrib.intensity_ll` registers
`{"name": "ml_i", ...}` (crystallographic intensity likelihood via adaptive quadrature)
and the `ml_i_maps` op (posterior map coefficients; client wrapper
`phridge.contrib.intensity_ll.client.RemoteIntensityMaps`) — a worked example of a
contrib package that ships both a target and an op without new schema.

## Client side

The client must know the op **name** before enqueue (`get_op` gate):

```python
import mypkg.plugin  # registers OpSpec
from phridge.client import Bridge

bridge = Bridge("redis://gpu:6379/0")
out = bridge.call("denoise", array=..., strength=0.2)
```

In-process demos: `Bridge(memory=True)` runs `process_envelope` in the same
process — still call `register_op` (with impl) first.

## Custom targets (existing SF ops)

For a new reciprocal-space target used by `target_eval` /
`refine_gradients`, subclass `phridge.sfcalc.targets.Target` and
`@register_target("my_name")`. Prefer an in-tree package under
`phridge.contrib` (see above) or an external module; the worker must load
it via entry points or `--preload`. Step-by-step:
[tutorial_sf_targets.md](tutorial_sf_targets.md). Math and built-ins:
[engine.md](engine.md).

## CCTBX-runtime ops

Ops that need mmtbx / the monomer library register with `runtime="cctbx"`
and ship an impl under a module the CCTBX worker loads. Built-in example:
`build_geometry_restraints` (PDB / hierarchy → packed `GeometryRestraints`).
Torch drivers use `RemoteRestraintBuilder` without importing cctbx.

## LinkML / contract tests

Treat `schema/*.yaml` as the source of truth for envelopes and JSON meta.
Extension PRs / plugins should add fixtures that `jsonschema.validate`
against `schema/generated/*.schema.json` (see `tests/test_linkml_validate.py`).
Packed npz / i_seq checks stay in Python pack constructors — not LinkML.

## Checklist for a third-party op

1. Implement a pure function `(decoded inputs) -> outputs` (numpy/torch ok).
2. `register_op(name, fn, inputs=..., outputs=...)`.
3. Prefer `array` / `json` / existing packed types.
4. `--preload your.module` on the worker; import the same module on the client.
5. Optional: `phridge.ops` entry point + contract tests on `JobEnvelope` JSON.

## Worked example: agentsg Niggli reduction

A full runnable demo lives under [`examples/`](../examples/):

| File | Role |
|------|------|
| [`examples/plugins/agentsg_niggli.py`](../examples/plugins/agentsg_niggli.py) | External plugin (`register_op`) |
| [`examples/agentsg_cell_reduce.py`](../examples/agentsg_cell_reduce.py) | Client + markdown report |

```bash
pip install -e ".[agentsg]"   # git+https://github.com/phzwart/agentsg …/subdirectory=agentsg
make example-agentsg
```

Flow: JSON unit cell → `Bridge(memory=True)` → worker calls
`agentsg.cell.niggli_reduce` → JSON `{unit_cell, change_of_basis}`. For a live
worker: `PYTHONPATH=examples phridge-worker --preload plugins.agentsg_niggli`.

Phenix-facing helper (same module) converts cctbx ↔ JSON:

```python
from cctbx import uctbx
from plugins.agentsg_niggli import niggli_cell
from phridge.client import Bridge

bridge = Bridge(memory=True)
uc_red = niggli_cell(uctbx.unit_cell((9, 5, 7, 80, 100, 95)), bridge=bridge)
# → cctbx.uctbx.unit_cell  (worker never imported cctbx)
```
