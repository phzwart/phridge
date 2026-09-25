# Structure-factor engine benchmark

_2026-09-25. Kept-engine phridge worker (Redis db 1) vs local cctbx FFT.
~1000 ASU atoms (C/N/O/S), `d_min=2.0` Å, `quality_factor=1000`,
`volume_per_atom=50`. Median of 3 timed calls after one warmup.
`R(F)` is vs cctbx `algorithm='fft'`, not vs direct summation._

## Summary

| | |
|---|---|
| Protocol | cctbx `random_structure` → bind once → `sf_calc` / `sf_gradients` |
| Space groups | **P1**, **P2₁2₁2₁**, **R3** (hex, SG 146), **F423** (SG 209 `F 4 3 2`) |
| Default worker engine | **unchanged** (`stamp`, `auto`) |
| This machine | Apple Silicon; no NVIDIA. CUDA rows are options, not timings. |
| Fast path here | CPU C++ stamp, ASU paint + algebraic gather, real `rfftn` |

On this laptop the C++ stamp matches or beats cctbx FFT on P2₁2₁2₁ and R3,
and is ~3× faster on F423 (96 P1 images; we splat the ASU only). The
eager torch splat is the correctness reference, not the speed path.

## How the C++ route matches cctbx

The C++ path is the same algorithm as cctbx `sampled_model_density` +
`maptbx.structure_factors.from_map`, not a P1 expansion of every atom.

1. **Splat the ASU only.** Ten Eyck Gaussians for the asymmetric-unit
   atoms are painted onto the crystal grid (periodic wrap). Occupancy is
   still `occ · multiplicity / n_sym`, so special positions are not
   over-counted. We do **not** stamp the 4 / 9 / 96 symmetry copies in
   real space.
2. **Space-group algebra from agentsg.** `sg_gather` asks agentsg for
   the space group (Hall, HM, or number), the closed operator list, and
   centering. That is the ASU → crystal map: each Seitz op is a
   point-group rotation \(W\) plus a translation \(w\).
3. **FFT the ASU density**, then **add mates algebraically**:

   \[
   F(h)=\sum_s \mathrm{FFT}[\rho_\mathrm{ASU}][-hW_s]\;e^{2\pi i\,h\cdot w_s}
   \]

   same sum cctbx does in `from_map`. Agarwal is the adjoint of that
   gather (scatter \(G_h\) onto the mate slots, then the stamp VJP).

`p1_expand=True` turns this off and paints every symmetry copy, which is
what the old torch/Numba path did and why F423 used to be ~96× the splat.
ASU splat is on whenever `n_sym > 1` and agentsg is installed; C++,
Numba, torch, and CUDA all use the same gather.

## 1. Options

`auto` on CPU is still Numba. **phridge refine** (and any client that
omits `params`) now uses `SfEngineParams.fastest`: `stamp_backend='fast'`
(CUDA if the worker is NVIDIA, otherwise C++), `dtype=float32`,
`quality_factor=1000`, ASU splat + agentsg gather, kept `sf_bind`.
Pass an explicit `SfEngineParams(...)` to override.

Pick the rest through `SfEngineParams` (JSON on the wire) or
`EngineParams` (in-process). `sf_calc` / `sf_gradients` stay on the
stamp engine.

### 1.1 Stamp backend (`stamp_backend`)

| value | what it is | when `auto` picks it |
|---|---|---|
| `auto` | dispatch below | — |
| `fast` | CUDA if present, else C++ | refine default (`SfEngineParams.fastest`) |
| `numba` | parallel thread-local CPU grids | CPU if Numba is installed |
| `cpp` | one-grid OpenMP/C++ stamp, exp table | **MPS** (no Metal kernel) |
| `cuda` | same stamp/VJP in `.cu` (`atomicAdd`) | NVIDIA if the extension builds |
| `triton` | CUDA Triton splat | CUDA if `.cu` is unavailable |
| `torch` | broadcast `(atoms × box)` + `index_add` | fallback everywhere |

`auto` on **CPU does not select C++** — Numba stays the default. Pass
`stamp_backend="cpp"` to opt in. There is **no Metal stamp**; MPS `auto`
runs the CPU C++ kernel and, today, also keeps the FFT/Agarwal on the
host (`stamp_cpp` pins the grid to CPU). `stamp_backend="torch"` is the
only path that actually uses the GPU on this Mac (Metal `rfftn` +
real/imag `scatter_add`).

CUDA today is a 1-thread-per-atom port of the C++ kernel. It is opt-in
and untimed here. To be fast it needs an on-device pack, a
block-per-atom (or tile+bin) splat, f32 iso math, and a CUDA graph
around the kept-engine `rfftn` + Agarwal VJP — see the notes in
`src/phridge/sfcalc/engine/stamp_cuda/`.

### 1.2 Precision (`dtype`)

| `dtype` | stamp storage | box math | FFT | notes |
|---|---|---|---|---|
| `float64` | f64 | f64 | `rfftn` / `fftn` | default; **not on MPS** (lowered to f32) |
| `float32` | f32 | f32 | `rfftn` / `fftn` | fast path on this machine |
| `float16` | f16 | f32 | promoted to f32 | `R(F)` ~1.5×10⁻²; `rfftn` Half unsupported (torch/numba skip) |

C++/CUDA f16 keep an f32 accumulator. `quality_factor` (100 / 1000) and
`wing_cutoff` (default 1e-4) still dominate accuracy vs cctbx FFT
(`R(F)` ~5×10⁻³ at qf=1000, f32/f64).

### 1.3 Device

| device | stamp `auto` | FFT / Agarwal |
|---|---|---|
| `cpu` | Numba | torch `rfftn` on CPU |
| `mps` | C++ on CPU | C++ path: CPU. torch path: Metal |
| `cuda` | CUDA `.cu` if built | cuFFT `rfftn` on device |

Worker: `--device cpu|mps|cuda`. Client and worker must agree; the
Phenix process does **not** load torch.

### 1.4 Symmetry and FFT

| switch | default | effect |
|---|---|---|
| ASU splat + agentsg gather | on when `n_sym>1` | cctbx-style: paint ASU, sum mates over Seitz ops |
| `p1_expand=True` | off | old expand-every-atom splat |
| real `rfftn` | on when all `fdp=0` | packed last axis; Hermitian gather / Agarwal scatter |
| complex `fftn` | `fdp≠0` | full grid |
| `cpu_numpy_fft` | off | numpy FFT escape hatch (CPU only) |

R3 is hexagonal **R 3 :H** (9 ops including centering). F423 in this
table is HM **F 4 3 2** / Hall **F 4 2 3** (SG 209, 96 ops). cctbx
rejects the compact symbol `F423`; the harness maps `f423` → `F 4 3 2`.

### 1.5 Engines and ops

| engine | ops | grads | default? |
|---|---|---|---|
| stamp / FFT | `sf_bind`, `sf_calc`, `sf_gradients` | hand Agarwal + stamp VJP | **yes** |
| NUFFT | `nufft_sf_bind`, `nufft_sf_calc`, `nufft_sf_gradients` | autograd through FINUFFT | no |
| cctbx FFT | in-process `algorithm='fft'` | not timed here | baseline `R(F)` |

NUFFT is skipped on MPS (no FINUFFT on Metal) and was not re-run in
this sweep (`--no-nufft`). Bind once per `(xray, hkl, params)` and reuse
the handle — rebuild-every-RPC is not these numbers.

### 1.6 How to run

```bash
# worker (isolated Redis db)
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src python -m phridge.worker.runner \
  --redis-url redis://127.0.0.1:6379/1 --device cpu \
  --preload phridge.contrib.nufft_sf --consumer nufft-bench

# Phenix / cctbx client
PYTHONPATH=src phenix.python -m phridge.contrib.nufft_sf.benchmark \
  --redis-url redis://127.0.0.1:6379/1 \
  --cases p1,p212121,r3,f423 --n-synthetic 1000 --d-min 2.0 \
  --device cpu --dtypes float32 --repeats 3 --no-nufft \
  --out examples/sf_gradient_benchmark.md
```

`--dtypes float64,float32,float16` iterates precision. `--device mps`
skips f64. Cases also accept `synthetic` (= P2₁2₁2₁), `r3h`, PDB
`1ee2` / `6czg`.

The older LS-vs-`gradients_direct` harness
(`examples/sf_gradient_benchmark.py`, `tests/sf_gradient_bench.py`) is
still there for cosine / site-rel checks. It is **not** what produced
the tables below.

## 2. CPU (bridge, float32)

`stamp_qf1000_f32` is `auto` → **Numba**. `stamp_cpp_*` is opt-in C++.
`stamp_torch_*` is the eager splat. cctbx FFT is local in Phenix (no
Redis).

| engine | SG | n_atoms | n_refl | R(F) vs cctbx | t(F) | t(F+grad) |
|---|---|---|---|---|---|---|
| stamp auto (Numba) f32 | P1 | 1000 | 13090 | 6.90e-03 | 0.0145 | 0.0410 |
| stamp cpp f32 | P1 | 1000 | 13090 | 6.95e-03 | **0.0096** | **0.0294** |
| stamp torch f32 | P1 | 1000 | 13090 | 6.90e-03 | 0.0909 | 0.2981 |
| cctbx FFT | P1 | 1000 | 13090 | 0 | 0.0083 | — |
| stamp auto (Numba) f32 | P2₁2₁2₁ | 1000 | 14098 | 4.95e-03 | 0.0209 | 0.0494 |
| stamp cpp f32 | P2₁2₁2₁ | 1000 | 14098 | 4.99e-03 | **0.0103** | **0.0332** |
| stamp torch f32 | P2₁2₁2₁ | 1000 | 14098 | 4.95e-03 | 0.0742 | 0.2159 |
| cctbx FFT | P2₁2₁2₁ | 1000 | 14098 | 0 | 0.0129 | — |
| stamp auto (Numba) f32 | R3:H | 1000 | 13109 | 4.21e-03 | 0.0368 | 0.0786 |
| stamp cpp f32 | R3:H | 1000 | 13109 | 4.27e-03 | **0.0138** | **0.0460** |
| stamp torch f32 | R3:H | 1000 | 13109 | 4.21e-03 | 0.0940 | 0.2794 |
| cctbx FFT | R3:H | 1000 | 13109 | 0 | 0.0270 | — |
| stamp auto (Numba) f32 | F423 | 1000 | 14458 | 6.56e-03 | 0.2098 | 0.4136 |
| stamp cpp f32 | F423 | 1000 | 14458 | 6.62e-03 | **0.0486** | **0.2414** |
| stamp torch f32 | F423 | 1000 | 14458 | 6.56e-03 | 0.1138 | 0.4191 |
| cctbx FFT | F423 | 1000 | 14458 | 0 | 0.1658 | — |

C++ vs cctbx `t(F)`: P1 0.86×, P2₁2₁2₁ **0.80×**, R3 **0.51×**, F423 **0.29×**.

## 3. MPS (bridge, float32)

`auto` and `cpp` are the same kernel (CPU C++). `torch` is Metal.

| engine | SG | n_atoms | n_refl | R(F) vs cctbx | t(F) | t(F+grad) |
|---|---|---|---|---|---|---|
| stamp auto (=cpp) f32 | P1 | 1000 | 13090 | 6.95e-03 | 0.0117 | 0.0345 |
| stamp cpp f32 | P1 | 1000 | 13090 | 6.95e-03 | 0.0108 | 0.0334 |
| stamp torch f32 | P1 | 1000 | 13090 | 6.90e-03 | 0.0432 | 0.1194 |
| cctbx FFT | P1 | 1000 | 13090 | 0 | 0.0086 | — |
| stamp auto (=cpp) f32 | P2₁2₁2₁ | 1000 | 14098 | 4.99e-03 | 0.0115 | 0.0388 |
| stamp cpp f32 | P2₁2₁2₁ | 1000 | 14098 | 4.99e-03 | 0.0140 | 0.0393 |
| stamp torch f32 | P2₁2₁2₁ | 1000 | 14098 | 4.95e-03 | 0.0311 | 0.0830 |
| cctbx FFT | P2₁2₁2₁ | 1000 | 14098 | 0 | 0.0138 | — |
| stamp auto (=cpp) f32 | R3:H | 1000 | 13109 | 4.27e-03 | 0.0147 | 0.0478 |
| stamp cpp f32 | R3:H | 1000 | 13109 | 4.27e-03 | 0.0152 | 0.0472 |
| stamp torch f32 | R3:H | 1000 | 13109 | 4.21e-03 | 0.0353 | 0.0993 |
| cctbx FFT | R3:H | 1000 | 13109 | 0 | 0.0254 | — |
| stamp auto (=cpp) f32 | F423 | 1000 | 14458 | 6.62e-03 | 0.0541 | 0.2487 |
| stamp cpp f32 | F423 | 1000 | 14458 | 6.62e-03 | 0.0521 | 0.2478 |
| stamp torch f32 | F423 | 1000 | 14458 | 6.56e-03 | **0.0339** | **0.0945** |
| cctbx FFT | F423 | 1000 | 14458 | 0 | 0.1732 | — |

On F423 the Metal torch path wins `t(F)` and especially `F+grad`: the
map is large enough that `rfftn` + packed Agarwal on GPU beat a host
C++ stamp that never uploads the grid. On P1 / P2₁2₁2₁ / R3 the C++
stamp is still faster.

## 4. Other options (not re-timed in this sweep)

P2₁2₁2₁ / 1k / 2 Å, earlier the same day, same harness:

| engine | device | t(F) | t(F+grad) | R(F) |
|---|---|---|---|---|
| cpp f64 | cpu | 0.0118 | 0.0415 | 5.0e-03 |
| cpp f32 | cpu | 0.0096 | 0.0334 | 5.0e-03 |
| cpp f16 | cpu | 0.0106 | 0.0355 | 1.5e-02 |
| Numba f64 | cpu | 0.0176 | 0.0498 | 4.9e-03 |
| torch f32 | cpu | 0.0762 | 0.2247 | 4.9e-03 |
| torch / Numba f16 | cpu | skip | skip | `rfftn` Half unsupported |
| cctbx FFT | cpu | 0.0129 | — | 0 |
| CUDA `.cu` | — | — | — | not run (no NVIDIA) |
| NUFFT τ=1e-3 / 1e-4 | cpu | — | — | contrib only; `--no-nufft` here |

## 5. What the times include

- Redis hop + `sf_bind` once (not in `t(F)`).
- C++ (and the other stamp backends) splat the ASU only, `rfftn`, then
  agentsg mate gather; first-order grads are Agarwal (scatter onto
  packed `−h`, `rfftn` VJP, stamp VJP). No autograd tape through the FFT.
- cctbx column is a fresh `from_scatterers(..., algorithm='fft')` each
  call (setup + Ten Eyck). No cctbx direct summation.

The 2026-09-05 in-process CUDA / LS-vs-direct table that used to live
in this file is retired. Those F / grad times (0.2–2 s) were a
rebuild-every-call torch splat, not the kept C++ engine.
