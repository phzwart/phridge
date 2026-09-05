# Tutorial: FFT targets and gradients (and your own likelihood)

Super short guide to calling phridge’s FFT structure-factor target / gradient
path from Phenix, then plugging in a custom likelihood. Details and math live
in [engine.md](engine.md); the **remote GPU “structure-factor server”** story
in [sf_server.md](sf_server.md); Redis / workers in [redis.md](redis.md).

## 0. Setup

```bash
conda activate phridge-cctbx
pip install -e ".[dev]"

# On the GPU machine (can be a different host than Phenix):
redis-server
phridge-worker --redis-url redis://0.0.0.0:6379/0 --device cuda
```

```python
# import cctbx before torch in one process
from phridge.client import StructureFactorServer
from phridge.models import SfEngineParams

# production: Redis on the GPU server (other machine OK)
sf = StructureFactorServer("redis://gpu-lab.example.edu:6379/0")

# local / demo (no Redis; in-process worker):
# sf = StructureFactorServer(memory=True, device="cuda")  # or "cpu"
```

You need a cctbx `xray_structure` and an `f_obs` miller array (amplitudes).

---

## 1. One call: target + packed gradients

This is the usual refinement step: FFT \(F_\mathrm{calc}\), evaluate the
target, backprop to scatterer parameters, return cctbx packing-order-2
gradients for `scitbx.lbfgs`.

```python
params = SfEngineParams(d_min=2.0, quality_factor=100)  # raise quality for tighter F

target, packed = sf.target_and_gradients(
    xray_structure,
    f_obs,
    {"name": "ls", "obs_type": "F"},   # built-in least squares
    params=params,
)
# target: float (work-set)
# packed: flex.double — sites (cart), U, occ, … per grad flags
```

Built-in targets:

| `target_spec` | Needs |
|---------------|--------|
| `{"name": "ls", "obs_type": "F"}` or `"I"` | `f_obs` amplitudes (or intensities for `"I"`) |
| `{"name": "ml_f"}` | also `alpha`, `beta` (and epsilons / centric flags; filled from `f_obs` if omitted) |

ML example:

```python
target, packed = sf.target_and_gradients(
    xray_structure, f_obs,
    {"name": "ml_f"},
    params=params,
    alpha=alpha, beta=beta,           # flex or numpy, one per reflection
    r_free_flags=r_free_flags,        # optional
)
```

---

## 2. Split API: \(F_\mathrm{calc}\) then your own \(G_h\)

If you already have `d_target_d_f_calc` (cctbx convention:
\(G_h = \partial Q/\partial A_h + i\,\partial Q/\partial B_h\)):

```python
f_calc = sf.f_calc(xray_structure, miller_set, params=params)   # miller.array
grads = sf.gradients(xray_structure, miller_set, d_target_d_f_calc, params=params)

sites = grads.d_target_d_site_frac()              # or .d_target_d_site_cart()
packed = grads.packed()                           # for LBFGS
```

Or evaluate a registered target on a given `f_calc` without re-running the FFT:

```python
functor = sf.target_functor(f_obs, {"name": "ls", "obs_type": "F"})
res = functor(f_calc)
print(res.target_work(), res.d_target_d_f_calc())
```

---

## 3. Add your favorite likelihood

You only write the **per-reflection loss** in torch. Autograd supplies
`d_target_d_f_calc` and (for amplitude-only targets) Gauss–Newton
curvatures. The FFT chain then turns that into per-atom gradients.

```python
# mypkg/my_likelihood.py  — must run on the *worker* (preload / memory Bridge)
from phridge.worker.targets import Observations, Target, register_target


@register_target("my_nll")
class MyNegLogLikelihood(Target):
    """Example: simple Gaussian NLL on |F| with fixed sigma."""

    amplitude_only = True  # True → depends on F only through |F|

    def __init__(self, sigma: float = 1.0, **_):
        super().__init__(sigma=sigma)
        self.sigma = float(sigma)

    def per_reflection(self, f_calc, obs: Observations):
        # f_calc: complex tensor; obs.data: |F_obs| (same length / order)
        resid = obs.data - f_calc.abs()
        return 0.5 * (resid / self.sigma) ** 2


# register at import time
def register():
    pass  # decorator already registered the class
```

**Wire it in**

1. Worker loads your module (so the target registry knows `"my_nll"`):

```bash
phridge-worker --preload mypkg.my_likelihood --device cuda
# or with StructureFactorServer(memory=True): import mypkg.my_likelihood before the call
```

2. Client uses the same name in the target spec (no torch on the Phenix side):

```python
import mypkg.my_likelihood  # only needed for memory Bridge / same-process demos

target, packed = sf.target_and_gradients(
    xray_structure, f_obs,
    {"name": "my_nll", "sigma": 2.5},
    params=SfEngineParams(d_min=2.0),
)
```

Tips:

- Put anything that needs torch in `per_reflection` / `prepare` / `reduce`.
- Use `obs.data`, `obs.weights`, `obs.r_free`, `obs.alpha`, `obs.beta`, …
  (see `Observations` in `phridge.worker.targets`). Pass extras through
  `sf.target_and_gradients(..., alpha=..., beta=..., weights=...)`.
- Override `reduce` if you do not want the default mean over work reflections.
- Set `amplitude_only = False` if your loss uses the complex \(F_h\) (phase),
  not only \(|F_h|\).

For registering brand-new **ops** (not just targets), see
[extending.md](extending.md).

---

## 4. Minimal LBFGS sketch

```python
import scitbx.lbfgs
from cctbx.array_family import flex

class Minimizer:
    def __init__(self, xs):
        self.xs = xs
        self.x = xs.sites_cart().as_double()

    def compute_functional_and_gradients(self):
        self.xs.set_sites_cart(flex.vec3_double(self.x))
        return sf.target_and_gradients(
            self.xs, f_obs, {"name": "ls", "obs_type": "F"},
            params=SfEngineParams(d_min=2.0),
        )

m = Minimizer(xs)
scitbx.lbfgs.run(target_evaluator=m)
```

---

## Cheat sheet

| Want | Call |
|------|------|
| Remote GPU handle | `StructureFactorServer("redis://gpu-host:6379/0")` |
| \(F_\mathrm{calc}\) only | `sf.f_calc(xs, miller_set, ...)` |
| Grads given \(G_h\) | `sf.gradients(xs, miller_set, dtdf, ...)` |
| Target on fixed \(F_c\) | `sf.target_functor(...)(f_calc)` |
| Full step (FFT + target + grads) | `sf.target_and_gradients(...)` |
| Custom likelihood | `@register_target("name")` + `{"name": "name", ...}` |
| Architecture write-up | [sf_server.md](sf_server.md) |
