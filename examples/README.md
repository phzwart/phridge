# Examples

Runnable demos: **cctbx** packs molecule + restraints into Redis, the
**phridge worker** runs torch `LBFGS` / `Adam` / `SGD`, the client blocks and
converts results back to cctbx. A torch-driver demo builds restraints on the
**CCTBX** runtime then minimizes on torch. A separate demo compares large-N
CUDA structure-factor gradients to CCTBX.

## Environment

```bash
conda activate phridge-cctbx
pip install -e ".[dev]"
make chem-data   # once — monomer library (+ rotarama cache for helix builder)
```

Production path also needs Redis + workers:

```bash
redis-server
phridge-worker --redis-url redis://localhost:6379/0 --device cuda
phridge-cctbx-worker --redis-url redis://localhost:6379/0
```

The examples below use **`Bridge(memory=True)`** (in-process store + worker)
so you can run without a live Redis server.

## Restraint minimization

```bash
make example-restraints
# or:
python examples/restraint_minimization.py
```

Writes [`restraint_minimization.md`](restraint_minimization.md) with a results
table for **LBFGS**, **Adam**, **AdamW**, and **SGD** on the same distorted
30-residue poly-Ala α-helix (call counts + optimizer memory).

Phenix-facing API:

```python
from phridge.client import Bridge, RemoteGeometry

bridge = Bridge(memory=True)  # or Bridge("redis://localhost:6379/0") + worker
geo = RemoteGeometry(bridge, hierarchy, restraints_manager)
hierarchy_out = geo.minimize(max_iterations=100, optimizer="lbfgs")  # blocks
```

## Torch driver: build restraints then minimize

```bash
make example-torch-build
# or:
python examples/torch_build_restraints.py
```

Writes [`torch_build_restraints.md`](torch_build_restraints.md). Flow:
`RemoteRestraintBuilder` (cctbx runtime) → packed hierarchy + restraints →
`RemoteGeometry.minimize` (torch runtime).

```python
from phridge.client import Bridge, RemoteGeometry, RemoteRestraintBuilder

bridge = Bridge(memory=True)
out = RemoteRestraintBuilder(bridge).build(pdb_string=pdb)
geo = RemoteGeometry(bridge, out["hierarchy"], out["restraints"])
sites = geo.minimize(max_iterations=80, optimizer="lbfgs", update_hierarchy=False)
```

## Structure-factor server (cctbx-feel → remote GPU)

```bash
make example-sf-server
# or:
python examples/sf_server_demo.py
```

Writes [`sf_server_demo.md`](sf_server_demo.md). Documents the architecture
where **Phenix stays on the CPU host** and
[`StructureFactorServer`](../docs/sf_server.md) sends \(F_\mathrm{calc}\) /
gradients to a **GPU worker that can live on another machine**. The demo
itself uses `memory=True` (no Redis); production uses
`StructureFactorServer("redis://gpu-host:6379/0")`.

```python
from phridge.client import StructureFactorServer

sf = StructureFactorServer("redis://gpu-lab.example.edu:6379/0")
f_calc = sf.f_calc(xray_structure, miller_set, d_min=2.0)
grads = sf.gradients(xray_structure, miller_set, d_target_d_f_calc, d_min=2.0)
target, packed = sf.target_and_gradients(
    xray_structure, f_obs, {"name": "ls", "obs_type": "F"}, d_min=2.0
)
```

## Structure-factor gradients (GPU)

```bash
make example-sf-gradients
# or:
python examples/sf_gradient_benchmark.py
# pytest: make test-sf-gpu
```

Writes [`sf_gradient_benchmark.md`](sf_gradient_benchmark.md): ~1000-atom
structures over several space groups, F_obs from a known model, small
Gaussian site shake, then CCTBX `gradients_direct` vs phridge CUDA site
gradients (cosine / length ratio + timings). Requires CUDA.

## External package: agentsg cell reduction

Canonical **bring-your-own-package** walkthrough. Install
[agentsg](https://github.com/phzwart/agentsg), register a plugin op, call it
through `Bridge` — no edits under `src/phridge`.

```bash
pip install -e ".[agentsg]"
# or: pip install "git+https://github.com/phzwart/agentsg.git#subdirectory=agentsg"

make example-agentsg
# or:
PYTHONPATH=examples python examples/agentsg_cell_reduce.py
```

Writes [`agentsg_cell_reduce.md`](agentsg_cell_reduce.md). Plugin source:
[`plugins/agentsg_niggli.py`](plugins/agentsg_niggli.py) (worker JSON op +
Phenix-facing `niggli_cell()` that returns `uctbx.unit_cell`). See
[docs/extending.md](../docs/extending.md).
