# phridge

Bridge between Phenix/cctbx and a PyTorch worker. Redis is the only shared
store: blobs, a job stream, and a per-job ready list. The client never
imports torch. The worker never imports cctbx.

More detail: [docs/README.md](docs/README.md). The worker also carries a
differentiable FFT structure-factor engine and ML / least-squares
targets that drop in for `cctbx.xray.structure_factors` on the Phenix
side: [docs/engine.md](docs/engine.md).

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
| [`schema/cctbx_scattering.yaml`](schema/cctbx_scattering.yaml) | `ScatteringTable`, `SfEngineParams`, `SfGradients`, `TargetResult` |

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

`cctbx-base` on PyPI is the library, not Phenix. A Phenix install cannot
be assumed to `pip install` into the bundle.

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
phridge-worker --redis-url redis://localhost:6379/0
phridge-ops   # dump OpSpec JSON
```

The worker creates the consumer group on startup:

```text
XGROUP CREATE phridge:jobs phridge-workers 0 MKSTREAM
```

```python
from phridge.client import Bridge
import numpy as np

bridge = Bridge(redis_url="redis://localhost:6379/0")
out = bridge.call("scale_array", array=np.arange(4, dtype=np.float64), scale=2.0)
```

`call` blocks until `done` or `error` (envelope poll / `BLPOP` on
`phridge:job:{id}:ready`, not Pub/Sub). `submit` / `result` are the
non-blocking pair.

v1 ships one dummy op, `scale_array`. No science kernels. Client
converters and EM helpers are documented in [docs/client.md](docs/client.md).

## Redis

| Key | Role |
|-----|------|
| `phridge:job:{id}` | `JobEnvelope` JSON |
| `phridge:obj:{id}:{name}` | raw bytes (npy / npz / JSON) |
| `phridge:jobs` | Redis Stream |
| `phridge:job:{id}:ready` | list; worker `RPUSH`, client `BLPOP` |

Keys expire after 1 hour by default.

`put` rejects blobs larger than `max_object_bytes` (default 64 MiB).
Real maps will exceed this. Raise the cap or plan a later filesystem/S3
blob backend behind the same `ObjectRef.key`. Set Redis `maxmemory`
accordingly; Redis is a poor store for hundreds of MB.

## Tests

```bash
make test    # uses .venv
pytest
```

cctbx converter tests skip unless `cctbx` is importable. Stream loopback
uses fakeredis; if consumer groups are incomplete the stream test skips.
Use a real Redis to verify `XREADGROUP`. Torch worker tests skip unless
`phridge[worker]` is installed.

## Layout

```
schema/                      LinkML (source of truth)
docs/                        Contract, packing, client helpers
src/phridge/models.py        committed pydantic
src/phridge/packing*.py      npz encode/decode
src/phridge/client/          Bridge, converters, EM + geometry helpers
src/phridge/worker/          stream loop, torch helpers, scale_array
```
