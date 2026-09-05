# Examples

Runnable demos: **cctbx** packs molecule + restraints into Redis, the
**phridge worker** runs torch `LBFGS` / `Adam` / `SGD`, the client blocks and
converts results back to cctbx.

## Environment

```bash
conda activate phridge-cctbx
pip install -e ".[dev]"
make chem-data   # once — monomer library (+ rotarama cache for helix builder)
```

Production path also needs Redis + a worker:

```bash
redis-server
phridge-worker --redis-url redis://localhost:6379/0 --device cuda
```

The example below uses **`Bridge(memory=True)`** (in-process store + worker)
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
