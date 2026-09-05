# Extending phridge (external ops and workers)

phridge is meant to be a **protocol + runtime**, not a closed app. Your own
PyTorch package can add ops without forking this repo. Redis remains the
server; `phridge-worker` (or a thin wrapper) is the consumer; Phenix talks
through `Bridge`.

There is no separate “phridge server” process — start `redis-server` and a
worker that shares the URL.

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

# dump catalog including plugins
phridge-ops --preload mypkg.plugin
```

| Flag / env | Role |
|------------|------|
| `--preload MOD` | Import module(s) before serving (repeatable; comma-separated ok) |
| `PHRIDGE_PRELOAD` | Same as `--preload` |
| `--no-entry-points` | Skip the `phridge.ops` entry-point group |

Entry points (optional) so `pip install mypkg` is enough:

```toml
# mypkg pyproject.toml
[project.entry-points."phridge.ops"]
denoise = "mypkg.plugin:register"
```

`phridge-worker` and `phridge-ops` call each entry point (zero-arg) at
startup, then apply `--preload` (preload wins on name clashes).

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
`refine_gradients`, subclass `phridge.worker.targets.Target` and
`@register_target("my_name")`. The worker must import that module (same
`--preload` / entry-point path). Step-by-step:
[tutorial_sf_targets.md](tutorial_sf_targets.md). Math and built-ins:
[engine.md](engine.md).

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
