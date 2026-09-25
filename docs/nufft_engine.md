# NUFFT structure-factor engine

A second, additive forward model (`NufftStructureFactorEngine`) that is
API-compatible with `StructureFactorEngine`. It does **not** replace the
stamp/FFT engine; `sf_calc` / `sf_gradients` are unchanged. Select it
explicitly via the contrib ops `nufft_sf_calc` and `nufft_sf_gradients`.

## Why NUFFT

The stamp engine paints each atom as real-space Gaussians, FFTs, and
divides out a *shared* extra smear `u_extra`. That reciprocal correction
is only exact when the s-dependent factor is the same for every atom.
Per-atom form factors and (an)isotropic ADPs cannot be pulled out of the
sum, so a single shared-kernel splat is approximate for mixed atoms.

The NUFFT engine groups atoms so that within a group the s-dependent
factor is common, Taylor-expands the residual ADP `ΔU_j`, and evaluates
every remaining sum `Σ_j w_j exp(2πi h·x_j)` as a type-1 NUFFT (Kaiser–Bessel
spreader, no `u_extra`, no box, no aliasing correction).

## Mathematics

Conventions match the stamp engine. `s² = |d*|²`, `stol² = s²/4`, eltbx
Gaussians `f_t(stol²) = Σ_k a_tk exp(-b_tk stol²) + c_t`. Expanded atoms
carry `w_j = occ_j · multiplicity_j / n_sym`.

```
F(h) = Σ_g f_{t(g)}(s) exp(-2π² U_g s²)
         · Σ_{j∈g} w_j (1 + (fp_j + i fdp_j)/f_t(s))
                · exp(-2π² d*^T ΔU_j d*) exp(2πi h·x_j)
```

`U_j = U_g I + ΔU_j` with `U_g` the shell median. The residual Debye–Waller
factor is expanded to order `N`. Isotropic terms use `(s²)^n · NUFFT[w δ^n]`.
Anisotropic terms use the 6-component Voigt form (6 transforms at `n=1`,
21 at `n=2`). `n ≥ 3` is not implemented for aniso; shells are tightened
instead. `fp`/`fdp` add a second scalar-weight family per group.

Truncation bound (per group, relative):

```
|R_{N+1}| ≤ (2π² s_max² λ_max(ΔU))^{N+1} / (N+1)!
```

This bound drives automatic U-shell construction (`GroupPlan.from_model`).

## API

```python
from phridge.sfcalc.engine import NufftEngineParams, NufftStructureFactorEngine, GroupPlan

eng = NufftStructureFactorEngine(model, hkl, NufftEngineParams(d_min=2.0, tau=1e-4, n_max=2, eps=1e-6))
f = eng.f_calc_numpy()
g = eng.gradients(d_target_d_f_calc)  # keys: site_frac, occupancy, u_iso, u_star, fp, fdp
eng.replan()  # after large ADP changes
```

Public surface matches `StructureFactorEngine`: `f_calc`, `tensors`,
`f_calc_numpy`, `gradients`, `jvp`, `gauss_newton_hvp`,
`gauss_newton_diagonal`. `gauss_newton_blocks` raises `NotImplementedError`.

`symmetry="expand"` expands to P1 with the same `expand()` as the stamp
engine. `symmetry="asu"` is reserved and not implemented.

### Plan / replan

The group plan is built from the model at construction and reused.
`U_g` is a detached constant; gradients w.r.t. `u_iso` / `u_star` flow
through `ΔU_j` only. A stale plan degrades **accuracy**, never correctness
of the derivative of the model actually evaluated. Call `replan()` after
large ADP changes.

## Contrib ops

```python
from phridge.contrib.nufft_sf import register
register()  # or pip install + worker entry point phridge.ops:nufft_sf

# inputs: xray, table, hkl, params (json NufftEngineOptions)
# outputs: f_calc (MillerArray)
bridge.call("nufft_sf_calc", xray=..., table=..., hkl=..., params={"engine": "nufft", "d_min": 2.0})

# inputs: xray, table, d_target_d_f_calc, params
# outputs: gradients (SfGradients)
bridge.call("nufft_sf_gradients", ...)
```

`NufftEngineOptions` (`extra="forbid"`): `engine="nufft"`, `d_min`, `tau=1e-4`,
`n_max=2`, `eps=1e-6`, `dtype="float64"`, `t_chunk=16`, `symmetry="expand"`.

Install: `pip install 'phridge[nufft]'` (pulls `pytorch-finufft` / `finufft`).

## Benchmark

Phenix / cctbx stays in `phenix.python`. Torch + FINUFFT stay in a phridge
worker. Do not install FINUFFT into Phenix.

```bash
# Redis (once)
redis-server --daemonize yes

# torch worker (mamba / conda with pytorch-finufft)
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src python -m phridge.worker.runner \
    --device cpu --preload phridge.contrib.nufft_sf

# cctbx client
PYTHONPATH=src phenix.python -m phridge.contrib.nufft_sf.benchmark \
    --redis-url redis://localhost:6379/0 \
    --cases 1ee2,6czg,synthetic \
    --d-min 2.5,2.0,1.5 \
    --device cpu \
    --out docs/nufft_engine.md
```

In-process (same interpreter has both cctbx and FINUFFT):

```bash
PYTHONPATH=src python -m phridge.contrib.nufft_sf.benchmark \
    --cases 1ee2,6czg,synthetic --d-min 2.5,2.0,1.5 --device cpu
```

Results (this environment has no cctbx and no CUDA; 1ee2 / 6czg / cctbx-direct
rows were skipped. Re-run the commands above on a cctbx+CUDA box to fill the
full table):

    PYTHONPATH=src python -m phridge.contrib.nufft_sf.benchmark --cases synthetic --d-min 2.5 --device cpu --repeats 3 --n-synthetic 80

| engine | case | d_min | device | n_atoms | n_refl | n_groups | T | R(F) vs cctbx direct | t(F) | t(F+grad) | peak mem (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stamp_qf100 | synthetic_p1 | 2.5 | cpu | 80 | 20256 | 0 | 0 | n/a | 0.0028 | 0.0085 | ~520 |
| stamp_qf1000 | synthetic_p1 | 2.5 | cpu | 80 | 20256 | 0 | 0 | n/a | 0.0028 | 0.0082 | ~530 |
| nufft_tau1e-3 | synthetic_p1 | 2.5 | cpu | 80 | 20256 | 1 | 3 | n/a | 0.0035 | 0.0199 | ~580 |
| nufft_tau1e-4 | synthetic_p1 | 2.5 | cpu | 80 | 20256 | 2 | 6 | n/a | 0.0062 | 0.0338 | ~600 |

On this CPU smoke (80 atoms, dense P1 sphere, no GPU) NUFFT is **not** 5×
faster than the stamp engine; the go/no-go for a CUDA default remains open
until the 1ee2 / 6czg / 20k-atom suite is run with cctbx.

The default worker engine is **not** changed in this PR. A go/no-go for
making `engine: "nufft"` the CUDA default is a ≥5× `F+grad` speedup over
the stamp engine at equal `R(F)` on the largest case.

Results (fill after running the commands above):

    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src python -m phridge.worker.runner --device cpu --preload phridge.contrib.nufft_sf
    PYTHONPATH=src phenix.python -m phridge.contrib.nufft_sf.benchmark --redis-url redis://127.0.0.1:6379/1 --cases synthetic --d-min 2.0 --device cpu --repeats 3

| engine | case | d_min | device | n_atoms | n_refl | n_groups | T | R(F) vs cctbx FFT | t(F) | t(F+grad) | peak mem (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stamp_qf100 | synthetic_1000 | 2.0 | cpu+bridge | 1000 | 14098 | 0 | 0 | 3.148e-03 | 0.0186 | 0.0448 | n/a |
| stamp_qf1000 | synthetic_1000 | 2.0 | cpu+bridge | 1000 | 14098 | 0 | 0 | 4.949e-03 | 0.0201 | 0.0472 | n/a |
| stamp_torch_qf1000 | synthetic_1000 | 2.0 | cpu+bridge | 1000 | 14098 | 0 | 0 | 4.949e-03 | 0.1044 | 0.3282 | n/a |
| stamp_cpp_qf1000 | synthetic_1000 | 2.0 | cpu+bridge | 1000 | 14098 | 0 | 0 | 4.990e-03 | 0.0145 | 0.0412 | n/a |
| nufft_tau1e-3 | synthetic_1000 | 2.0 | cpu+bridge | 1000 | 14098 | 0 | 0 | 5.597e-03 | 0.3287 | 1.9522 | n/a |
| nufft_tau1e-4 | synthetic_1000 | 2.0 | cpu+bridge | 1000 | 14098 | 0 | 0 | 5.597e-03 | 0.6584 | 3.9980 | n/a |
| cctbx_fft | synthetic_1000 | 2.0 | cpu | 1000 | 14098 | 0 | 0 | 0.000e+00 | 0.0141 | n/a | n/a |

The default worker engine is **not** changed in this PR. A go/no-go for
making `engine: "nufft"` the CUDA default is a ≥5× `F+grad` speedup over
the stamp engine at equal `R(F)` on the largest case.
