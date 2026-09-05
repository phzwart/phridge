# Structure-factor server (GPU, possibly another machine)

`StructureFactorServer` is the **cctbx-feel** object for structure factors
and their derivatives when the FFT work should run on a **torch GPU
worker** — often a **separate machine** from Phenix.

Phenix / cctbx stay on the CPU box. The worker that owns CUDA can live on
a lab GPU server, a cluster node, or the same host. Redis is the wire;
the Phenix process never imports torch.

```
  Phenix workstation                    GPU server (other machine OK)
  ┌─────────────────────┐               ┌──────────────────────────┐
  │ StructureFactorServer│── Redis ──▶  │ phridge-worker --device  │
  │  .f_calc()           │  (URL)       │   cuda                   │
  │  .gradients()        │◀──────────── │  Ten Eyck FFT + autograd │
  │  .target_and_gradients│             └──────────────────────────┘
  └─────────────────────┘
```

## Start the server on the GPU box

On the machine that has the GPU (and `phridge[worker]` / torch):

```bash
# GPU server — can be a different host than Phenix
redis-server
phridge-worker --redis-url redis://0.0.0.0:6379/0 --device cuda
```

Point Redis at an address the Phenix host can reach (bind / firewall /
`redis.conf` `bind` / TLS as your site requires). The worker and Phenix
must share the **same Redis URL**.

## Use it from Phenix (no CUDA on this side)

```python
# import cctbx before torch if both ever share a process; Phenix need not import torch at all
from phridge.client import StructureFactorServer

# Production: URL of the GPU machine's Redis
sf = StructureFactorServer("redis://gpu-lab.example.edu:6379/0")

# Local demo only (in-process worker; no Redis, not a remote server):
# sf = StructureFactorServer(memory=True, device="cuda")  # or "cpu"
```

### Like `from_scatterers(...).f_calc()` / `.gradients(...)`

```python
f_calc = sf.f_calc(xray_structure, miller_set, d_min=2.0)
# → cctbx miller.array (complex)

grads = sf.gradients(xray_structure, miller_set, d_target_d_f_calc, d_min=2.0)
sites = grads.d_target_d_site_cart()   # or .d_target_d_site_frac()
packed = grads.packed()                # packing_order_convention 2 for LBFGS
```

Or keep a bound engine (same names as cctbx):

```python
engine = sf.from_scatterers(xray_structure, miller_set, d_min=2.0)
f_calc = engine.f_calc()
grads = engine.gradients(d_target_d_f_calc)
```

### One round trip per minimizer step

```python
target, packed = sf.target_and_gradients(
    xray_structure,
    f_obs,
    {"name": "ls", "obs_type": "F"},   # or {"name": "ml_f"} + alpha/beta
    d_min=2.0,
)
# feed (target, packed) to scitbx.lbfgs
```

Optional second-order helpers (same remote worker):

```python
refiner = sf.refinement_target(xray_structure, f_obs, {"name": "ls", "obs_type": "F"}, d_min=2.0)
inv_diag = refiner.diagonal()          # L-BFGS Hk0 (diag_mode="once")
# result = refiner.newton_cg(xray_structure)
```

## Why a separate machine

| | Phenix host | SF server (GPU host) |
|--|-------------|----------------------|
| Imports | cctbx / mmtbx only | torch (+ phridge worker) |
| Hardware | CPU workstation / cluster login | CUDA GPU |
| Role | model, restraints, LBFGS, I/O | \(F_\mathrm{calc}\), \(G_h\to\) param grads |
| Coupling | Redis URL only | Redis URL only |

Importing torch before cctbx in one process can crash Boost.Python; keeping
SF on a remote worker avoids that and lets you size the GPU box
independently of the Phenix install.

## Accuracy and knobs

`SfEngineParams(d_min=..., quality_factor=1000, wing_cutoff=1e-4)` trades
grid / sampling cost for agreement with `algorithm="direct"`. Defaults
match the engine doc ([engine.md](engine.md)). Raise `quality_factor` for
tighter gradients; use GPU when \(N\) atoms × box points is large.

## Related

- Math and ops: [engine.md](engine.md)
- Custom likelihoods: [tutorial_sf_targets.md](tutorial_sf_targets.md)
- Redis / `Bridge(memory=True)`: [redis.md](redis.md)
- Runnable demo: `examples/sf_server_demo.py` → `examples/sf_server_demo.md`
