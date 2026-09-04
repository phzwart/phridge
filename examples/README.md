# Examples

Runnable demos: **cctbx** packs molecule + restraints into Redis, the
**phridge worker** runs torch `LBFGS` / `Adam` / `SGD`, the client blocks and
converts results back to cctbx.

## Environment

```bash
conda activate phridge-cctbx
pip install -e ".[dev]"
make chem-data   # once — monomer library for the peptide demo
```

Production path also needs Redis + a worker:

```bash
redis-server
phridge-worker --redis-url redis://localhost:6379/0 --device cuda
```

The example below uses an **in-process** worker loopback (fakeredis) so you
can run without a live Redis.

## Restraint minimization

```bash
make example-restraints
# or:
python examples/restraint_minimization.py
python examples/restraint_minimization.py --optimizer adam --max-iterations 500
```

Writes [`restraint_minimization.md`](restraint_minimization.md) (code bits + checks).

Phenix-facing API:

```python
from phridge.client import Bridge, RemoteGeometry

bridge = Bridge("redis://localhost:6379/0")
geo = RemoteGeometry(bridge, hierarchy, restraints_manager)
hierarchy_out = geo.minimize(max_iterations=100, optimizer="lbfgs")  # blocks
```
