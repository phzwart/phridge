# Redis

Redis is **mandatory** as the production protocol: the `redis` Python
package is a hard runtime dependency, and a reachable Redis server is
required for any real client / worker split across processes. phridge does
not fall back to an in-memory store when `import redis` fails.

Redis is the only shared process boundary between the Phenix/cctbx client
and the PyTorch worker. The client never imports torch; the worker never
imports cctbx. Everything that crosses that line — packed science objects,
job metadata, and completion signals — goes through Redis.

It is not a cache layer and not a Celery/RQ-style task broker with its own
worker framework. phridge uses Redis directly as an object store, a job
dispatch stream, and a per-job ready list.

## What it stores

Four kinds of keys carry the protocol:

| Key | Role |
|-----|------|
| `phridge:job:{id}` | `JobEnvelope` JSON (op, status, input/output refs, errors) |
| `phridge:obj:{id}:{name}` | raw bytes (npy / npz / JSON) for arrays and packed cctbx types |
| `phridge:jobs` | Redis Stream; workers claim jobs with a consumer group |
| `phridge:job:{id}:ready` | list; worker `RPUSH`, client `BLPOP` |

Keys expire after one hour by default (`ttl_seconds=3600`). Blobs larger
than `max_object_bytes` (default 64 MiB) are rejected: Redis is a poor
store for full maps. Raise the cap for large jobs, or plan a later
filesystem/S3 backend behind the same `ObjectRef.key`.

An `ObjectRef` is a pointer at those bytes plus enough metadata to decode
them. Science types live in `schema/cctbx*.yaml`; Redis only holds the
refs and the payloads they name. See [types.md](types.md) for packing and
[`schema/phridge.yaml`](../schema/phridge.yaml) for the job schema.

## Remote jobs

A remote job is one envelope on the stream. The Phenix side packs kwargs
into canonical form, writes blobs and a `JobEnvelope` with
`status=queued`, then `XADD`s the job id onto `phridge:jobs`. The worker
reads with `XREADGROUP` on the `phridge-workers` consumer group, sets the
envelope to `running`, decodes inputs, runs the registered op, encodes
outputs, and marks the envelope `done` or `error`. When it finishes it
`RPUSH`es the ready list and `XACK`s the stream message.

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

Higher-level façades (`RemoteGeometry`, `RemoteStructureFactors`,
`RemoteTargetFunctor`, `RemoteRefinementTarget`) all go through
`bridge.call`. From Phenix they look like local helpers; under the hood
they pack cctbx objects to Redis and wait for the torch worker. See
[client.md](client.md) and [engine.md](engine.md).

Registered ops include `scale_array`, `sf_calc`, `sf_gradients`,
`target_eval`, `refine_gradients`, `gauss_newton_hvp`, and
`geometry_minimize`. Unknown ops fail on the client before enqueue.
Third-party packages can add more via `register_op` and
`phridge-worker --preload` — see [extending.md](extending.md).

## Running with a server

Start Redis and a worker that shares the same URL:

```bash
redis-server
phridge-worker --redis-url redis://localhost:6379/0 --device cuda
# with an external op package:
phridge-worker --preload mypkg.plugin --device cuda
```

Worker flags (env overrides in parentheses):

| Flag | Env | Default |
|------|-----|---------|
| `--redis-url` | `PHRIDGE_REDIS_URL` | `redis://localhost:6379/0` |
| `--device` | `PHRIDGE_DEVICE` | `auto` |
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
calling process on `submit`.

```python
from phridge.client import Bridge

bridge = Bridge(memory=True)           # CPU by default
bridge = Bridge(memory=True, device="cuda")

out = bridge.call("scale_array", array=..., scale=2.0)
```

Memory mode cannot talk to a remote GPU worker in another process. It still
requires the `redis` Python package (and whatever the op needs, e.g. torch
for science kernels). There is no silent switch to memory mode when Redis
is unreachable or when `import redis` fails.

Under the hood this uses `RedisStore.memory()` (`MemoryRedis`) and
`process_envelope` after each `submit`. Do not also pass `store=`.

The restraint minimization example
([`examples/restraint_minimization.py`](../examples/restraint_minimization.py))
uses `Bridge(memory=True)`.

## Tests with fakeredis

Unit tests that exercise Streams / `XREADGROUP` may still wrap
`fakeredis.FakeRedis` in `RedisStore` and drive `consume_one`. Install
`fakeredis` via the `[dev]` extra. Prefer `Bridge(memory=True)` for
ordinary round-trip tests that only need the pack → op → unpack path.
If fakeredis Streams are incomplete, stream tests may skip; use a real
Redis to verify worker claiming.
