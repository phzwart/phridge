# phridge

Bidirectional bridge between Phenix/cctbx and PyTorch via Redis. Redis is
mandatory and the only shared store: blobs, per-runtime job streams
(`phridge:jobs:torch` / `phridge:jobs:cctbx`), and a per-job ready list.
Phenix never imports torch; the torch worker never imports cctbx; the
CCTBX worker never imports torch.

More detail: [docs/README.md](docs/README.md) (including
[Redis / `Bridge(memory=True)`](docs/redis.md)). The torch worker also
carries a differentiable FFT structure-factor engine and ML /
least-squares targets that drop in for `cctbx.xray.structure_factors` on
the Phenix side — often on a **separate GPU machine** via
[`StructureFactorServer`](docs/sf_server.md):
[docs/engine.md](docs/engine.md). Short how-to:
[docs/tutorial_sf_targets.md](docs/tutorial_sf_targets.md).

## Contract

LinkML schemas are the source of truth (`schema_version` / LinkML `version` = `1`):

| Schema | Types |
|--------|--------|
| [`schema/phridge.yaml`](schema/phridge.yaml) | job envelope, `ObjectRef`, op specs |
| [`schema/cctbx.yaml`](schema/cctbx.yaml) | `CrystalSymmetry` |
| [`schema/cctbx_reflections.yaml`](schema/cctbx_reflections.yaml) | miller, HL, MTZ-like `ReflectionFile` |
| [`schema/cctbx_maps.yaml`](schema/cctbx_maps.yaml) | real/complex maps, gridding, `EmMap` |
| [`schema/cctbx_coordinates.yaml`](schema/cctbx_coordinates.yaml) | sites, `Hierarchy`, `XrayStructure` |
| [`schema/cctbx_geometry.yaml`](schema/cctbx_geometry.yaml) | `GeometryRestraints`, `ModelGeometry` |
| [`schema/cctbx_scattering.yaml`](schema/cctbx_scattering.yaml) | `ScatteringTable`, `SfEngineParams`, `SfGradients`, `SfCurvatures`, `TargetResult` |

Redis stores science objects with `kind=cctbx` and `cctbx_type` set to the
class name. There is no `sparse_miller` or `map_grid` kind.

| Class | Module | Packed payload |
|-------|--------|----------------|
| `CrystalSymmetry` | core | JSON: unit cell + Hall + IT number |
| `MillerArray`, `HendricksonLattman`, `ReflectionFile` | reflections | npz: `hkl` plus columns |
| `RealMap`, `ComplexMap`, `MapCoefficients`, `CrystalGridding` | maps | npz grid / miller coeffs; gridding is JSON |
| `EmMap` | maps | npz `data` plus wrapping, voxels, `origin_cart` |
| `CartesianSites`, `FractionalSites` | coordinates | npz `xyz` |
| `Hierarchy` | coordinates | npz `xyz`, `occupancy`, `b_iso`; optional `u_cart` / `uij_defined` |
| `XrayStructure` | coordinates | npz `sites_frac`, `occupancy`, `u_iso`, `u_star` (`N×6` fractional) |
| `GeometryRestraints` | geometry | npz proxy tables keyed by `i_seq` |
| `ModelGeometry` | geometry | JSON correspondence header |
| `ScatteringTable` | scattering | npz Gaussian form-factor coefficients per type |
| `SfEngineParams` | scattering | JSON: d_min, gridding, quality factor |
| `SfGradients` | scattering | npz per-scatterer gradients (site, occ, u_iso, u_star, fp, fdp) |
| `SfCurvatures` | scattering | npz per-atom GN blocks (site 3×3, u_star 6×6, occ/u_iso/fp/fdp) |
| `TargetResult` | scattering | npz per-reflection target, `d_target_d_f_calc`, curvatures |

LinkML holds metadata. Multi-buffer objects are **one packed npz** at
`ObjectRef.key`. Do not pickle. Do not store flex or torch objects.

Canonical store dtypes are little-endian **float64 / int32**. The worker
may compute in float32; it casts back before `put`.

**Model–geometry correspondence:** `Hierarchy.atoms[i].i`,
`XrayStructure.scatterers[i].i`, and every restraint `i_seq` share one
index space. Use `phridge.client.geometry.model_geometry(...)` to check
that and emit `ModelGeometry`. See [docs/types.md](docs/types.md).

## Install

```bash
make venv                        # .venv with tests + cctbx-base
pip install -e ".[dev]"          # same, into an existing venv
pip install -e ".[cctbx]"        # pip cctbx-base only (not a Phenix bundle)
pip install -e ".[worker]"       # torch worker
```

On aarch64 (and generally for real geometry restraints), prefer a conda
env with `cctbx-base` plus the monomer library package:

```bash
conda create -n phridge-cctbx -c conda-forge python=3.13 cctbx-base
conda install -n phridge-cctbx -c chem_data chem_data   # ~5 GB; mon_lib + geostd
conda run -n phridge-cctbx pip install -e ".[dev]"
```

`chem_data` lands in `site-packages/chem_data`; mmtbx finds it without
extra env vars. (`MMTBX_CCP4_MONOMER_LIB` still works if you point at
`…/chem_data/geostd` from a Phenix tree instead.)

`cctbx-base` on PyPI is the library, not Phenix. A Phenix install cannot
be assumed to `pip install` into the bundle. Pip has no aarch64
`cctbx-base` wheel.

Runtime needs `numpy`, `redis`, and `pydantic` v2. `linkml-runtime` is
not required at runtime; [`src/phridge/models.py`](src/phridge/models.py)
is committed. After editing YAML, update `models.py` and regenerate JSON
Schema plus the class catalog:

```bash
make schema-docs
# or: python scripts/generate_models.py --all
```

That writes [`schema/generated/*.schema.json`](schema/generated/) and
[`docs/schema/`](docs/schema/). `linkml-validate` / pytest check JSON
metadata only (not packed npz).

### Phenix client

`pip install phridge[client]` often **does not work** inside a Phenix
bundle. Use a side conda env that can import `cctbx`, or put this repo
on `PYTHONPATH`:

```bash
export PYTHONPATH=/path/to/phridge/src:$PYTHONPATH
```

Phenix must still import `redis`, `numpy`, and `pydantic`. If those
cannot be installed into the bundle, run the client in a separate Python
that talks to the same Redis.

## Run

```bash
redis-server
phridge-worker --redis-url redis://localhost:6379/0 --device cuda
phridge-cctbx-worker --redis-url redis://localhost:6379/0
phridge-worker --preload mypkg.plugin   # third-party torch ops
phridge-ops   # dump OpSpec JSON
```

Workers create per-runtime consumer groups on startup:

```text
XGROUP CREATE phridge:jobs:torch phridge-workers:torch 0 MKSTREAM
XGROUP CREATE phridge:jobs:cctbx phridge-workers:cctbx 0 MKSTREAM
```

```python
from phridge.client import Bridge
import numpy as np

bridge = Bridge(redis_url="redis://localhost:6379/0")
out = bridge.call("scale_array", array=np.arange(4, dtype=np.float64), scale=2.0)
```

Without a Redis server, use explicit **memory mode** (same protocol,
in-process store + worker; still needs the `redis` package):

```python
bridge = Bridge(memory=True)
out = bridge.call("scale_array", array=np.arange(4, dtype=np.float64), scale=2.0)
```

`call` blocks until `done` or `error` (envelope poll / `BLPOP` on
`phridge:job:{id}:ready`, not Pub/Sub). `submit` / `result` are the
non-blocking pair. Full Redis / memory-mode notes:
[docs/redis.md](docs/redis.md).

Built-in ops cover SF / targets / geometry minimize (torch) and
`build_geometry_restraints` (cctbx); add your own with `register_op` and
`--preload` ([docs/extending.md](docs/extending.md)). Client converters
and EM helpers are in [docs/client.md](docs/client.md).

## Redis

| Key | Role |
|-----|------|
| `phridge:job:{id}` | `JobEnvelope` JSON |
| `phridge:obj:{id}:{name}` | raw bytes (npy / npz / JSON) |
| `phridge:jobs:torch` | Redis Stream (torch-runtime ops) |
| `phridge:jobs:cctbx` | Redis Stream (cctbx-runtime ops) |
| `phridge:job:{id}:ready` | list; worker `RPUSH`, client `BLPOP` |

Keys expire after 1 hour by default.

`put` rejects blobs larger than `max_object_bytes` (default 64 MiB).
Real maps will exceed this. Raise the cap or plan a later filesystem/S3
blob backend behind the same `ObjectRef.key`. Set Redis `maxmemory`
accordingly; Redis is a poor store for hundreds of MB.

`Bridge(memory=True)` skips `redis-server` / workers for local work; it
does not replace the `redis` Python dependency and cannot reach a remote
worker. See [docs/redis.md](docs/redis.md).

## Tests

```bash
make test         # excludes gpu/slow markers
make test-sf-gpu  # ~1000-atom CUDA SF gradients vs CCTBX (needs CUDA)
pytest
```

cctbx converter tests skip unless `cctbx` is importable. Round-trips use
`Bridge(memory=True)`. Stream loopback tests may use fakeredis; if
consumer groups are incomplete the stream test skips. Use a real Redis to
verify `XREADGROUP`. Torch worker tests skip unless `phridge[worker]` is
installed. Large-N GPU SF gradient checks are marked `gpu`/`slow` — see
[docs/engine.md](docs/engine.md).

## Examples

```bash
make example-restraints
# → examples/restraint_minimization.md  (torch LBFGS/Adam/SGD)

make example-torch-build
# → examples/torch_build_restraints.md  (CCTBX build → torch minimize)

make example-latent-adam
# → examples/latent_restraints_adam.md  (torch latents + CCTBX E/grads via phridge)

make example-sf-server
# → examples/sf_server_demo.md  (cctbx-feel SF server → remote GPU)

make example-sf-gradients
# → examples/sf_gradient_benchmark.md  (CUDA SF grads vs CCTBX)

pip install -e ".[agentsg]"   # optional third-party cell package
make example-agentsg
# → examples/agentsg_cell_reduce.md  (external op via register_op)
```

See [examples/README.md](examples/README.md). Restraint and SF demos use
built-in ops; the agentsg demo is the canonical **external package** path
(`register_op` + `--preload`, no edits under `src/phridge`) — details in
[docs/extending.md](docs/extending.md).

## Layout

```
schema/                      LinkML (source of truth)
docs/                        Contract, packing, client helpers, extending
examples/                    demos + examples/plugins/ (third-party ops)
src/phridge/models.py        committed pydantic
src/phridge/packing*.py      npz encode/decode
src/phridge/client/          Bridge, converters, EM + geometry helpers
src/phridge/worker/          stream loop, torch helpers, science ops
src/phridge/ops.py           OpSpec catalog + register_op / preload
```
