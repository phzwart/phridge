# Redis

Redis is **mandatory** as the production protocol: the `redis` Python
package is a hard runtime dependency, and a reachable Redis server is
required for any real client / worker split across processes. phridge does
not fall back to an in-memory store when `import redis` fails.

Redis is the shared process boundary between drivers (Phenix/cctbx or
torch) and workers. The Phenix client never imports torch; the torch
worker never imports cctbx; the CCTBX worker never imports torch.
Everything that crosses that line — packed science objects, job metadata,
and completion signals — goes through Redis.

It is not a cache layer and not a Celery/RQ-style task broker with its own
worker framework. phridge uses Redis directly as an object store, a job
dispatch stream (per worker runtime), and a per-job ready list.

## What it stores

| Key | Role |
|-----|------|
| `phridge:job:{id}` | `JobEnvelope` JSON (op, status, input/output refs, errors) |
| `phridge:obj:{id}:{name}` | raw bytes (npy / npz / JSON) for arrays and packed cctbx types |
| `phridge:jobs:torch` | Redis Stream for torch-runtime ops |
| `phridge:jobs:cctbx` | Redis Stream for cctbx-runtime ops |
| `phridge:job:{id}:ready` | list; worker `RPUSH`, client `BLPOP` |

Consumer groups are `phridge-workers:torch` and `phridge-workers:cctbx`.
(Older docs referred to a single `phridge:jobs` stream; that name is
retired — torch jobs use `phridge:jobs:torch`.)

Keys expire after one hour by default (`ttl_seconds=3600`). Blobs larger
than `max_object_bytes` (default 64 MiB) are rejected: Redis is a poor
store for full maps. Raise the cap for large jobs, or plan a later
filesystem/S3 backend behind the same `ObjectRef.key`.

An `ObjectRef` is a pointer at those bytes plus enough metadata to decode
them. Science types live in `schema/cctbx*.yaml`; Redis only holds the
refs and the payloads they name. See [types.md](types.md) for packing and
[`schema/phridge.yaml`](../schema/phridge.yaml) for the job schema.

## Bidirectional workers

Each `OpSpec` carries a `runtime` of `torch` or `cctbx` (default `torch`).
`Bridge` enqueues onto the matching stream so workers never steal each
other's jobs:

| Runtime | Stream | CLI |
|---------|--------|-----|
| `torch` | `phridge:jobs:torch` | `phridge-worker` (default) / `--runtime torch` |
| `cctbx` | `phridge:jobs:cctbx` | `phridge-cctbx-worker` / `phridge-worker --runtime cctbx` |

Example production topology:

```bash
redis-server
phridge-worker --redis-url redis://localhost:6379/0 --device cuda
phridge-cctbx-worker --redis-url redis://localhost:6379/0
```

A torch driver can call `build_geometry_restraints` (cctbx stream) then
`geometry_minimize` (torch stream) on the same `Bridge`.

## Remote jobs

A remote job is one envelope on the runtime stream. The client packs
kwargs into canonical form, writes blobs and a `JobEnvelope` with
`status=queued`, then `XADD`s the job id onto `phridge:jobs:{runtime}`.
The matching worker reads with `XREADGROUP`, sets the envelope to
`running`, decodes inputs, runs the registered op, encodes outputs, and
marks the envelope `done` or `error`. When it finishes it `RPUSH`es the
ready list and `XACK`s the stream message.

The client does not use Pub/Sub. `Bridge.call` blocks by `BLPOP` on the
ready key (with short polls) and then reads the envelope. `submit` /
`result` are the non-blocking pair: enqueue now, wait later.

```python
from phridge.client import Bridge
import numpy as np

bridge = Bridge("redis://localhost:6379/0")
# or Bridge("redis://gpu-box:6379/0") for a remote worker host

out = bridge.call("scale_array", array=np.arange(4, dtype=np.float64), scale=2.0)

job_id = bridge.submit("scale_array", array=..., scale=2.0)
out = bridge.result(job_id)
```

Higher-level façades (`RemoteGeometry`, `RemoteRestraintBuilder`,
`RemoteStructureFactors`, `RemoteTargetFunctor`, `RemoteRefinementTarget`)
all go through `bridge.call`. See [client.md](client.md) and
[engine.md](engine.md).

Registered ops include `scale_array`, `sf_calc`, `sf_gradients`,
`target_eval`, `refine_gradients`, `gauss_newton_hvp`,
`geometry_minimize` (torch), and `build_geometry_restraints` (cctbx).
Unknown ops fail on the client before enqueue. Third-party packages can
add more via `register_op(..., runtime=...)` and
`phridge-worker --preload` — see [extending.md](extending.md).

## Running with a server

Start Redis and workers that share the same URL:

```bash
redis-server
phridge-worker --redis-url redis://localhost:6379/0 --device cuda
phridge-cctbx-worker --redis-url redis://localhost:6379/0
# with an external op package:
phridge-worker --preload mypkg.plugin --device cuda
```

Worker flags (env overrides in parentheses):

| Flag | Env | Default |
|------|-----|---------|
| `--runtime` | `PHRIDGE_RUNTIME` | `torch` (`phridge-worker` only) |
| `--redis-url` | `PHRIDGE_REDIS_URL` | `redis://localhost:6379/0` |
| `--device` | `PHRIDGE_DEVICE` | `auto` (torch only) |
| `--consumer` | `PHRIDGE_CONSUMER` | hostname |
| `--max-object-bytes` | `PHRIDGE_MAX_OBJECT_BYTES` | 64 MiB |
| `--preload MOD` | `PHRIDGE_PRELOAD` | (none) |
| `--no-entry-points` | | load `phridge.ops` entry points |

`Bridge` takes `redis_url` (or an injected `store`); it does not read
`PHRIDGE_REDIS_URL` itself. Point both sides at the same instance so the
client on a cctbx machine and the worker on a GPU box can share jobs.

Phenix must import `redis` (and usually `numpy` / `pydantic`). If those
packages cannot live in the Phenix bundle, run the client in a separate
Python that shares the same Redis URL with the worker. That is still a
real Redis path — not memory mode.

## Explicit memory mode (no server)

If you do not want to run `redis-server` or `phridge-worker`, pass
`memory=True`. That is an **opt-in** in-process path: same pack / enqueue /
decode protocol, but an in-memory key store and the worker op runs in the
calling process on `submit`. Dispatch follows `OpSpec.runtime` (torch vs
cctbx `process_envelope`).

```python
from phridge.client import Bridge

bridge = Bridge(memory=True)           # CPU by default
bridge = Bridge(memory=True, device="cuda")

out = bridge.call("scale_array", array=..., scale=2.0)
```

Memory mode cannot talk to a remote GPU worker in another process. It still
requires the `redis` Python package (and whatever the op needs — torch
and/or cctbx). There is no silent switch to memory mode when Redis
is unreachable or when `import redis` fails.

Under the hood this uses `RedisStore.memory()` (`MemoryRedis`) and the
runtime-matched `process_envelope` after each `submit`. Do not also pass
`store=`.

The restraint minimization example
([`examples/restraint_minimization.py`](../examples/restraint_minimization.py))
uses `Bridge(memory=True)`. The torch-driver build+minimize example is
[`examples/torch_build_restraints.py`](../examples/torch_build_restraints.py).

## Tests with fakeredis

Unit tests that exercise Streams / `XREADGROUP` may still wrap
`fakeredis.FakeRedis` in `RedisStore` and drive `consume_one`. Install
`fakeredis` via the `[dev]` extra. Prefer `Bridge(memory=True)` for
ordinary round-trip tests that only need the pack → op → unpack path.
If fakeredis Streams are incomplete, stream tests may skip; use a real
Redis to verify worker claiming.
