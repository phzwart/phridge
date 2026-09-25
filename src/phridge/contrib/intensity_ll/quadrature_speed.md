# Speeding up the intensity-likelihood quadrature

How `mli.py` got faster without changing the integrator (24-node Gauss–Legendre on weak/centric data, 7-node adaptive Gauss–Hermite on strong acentrics, log-γ mixture for Student-\(t\)). Algorithm and error tables stay in [`intensity_likelihood_quadrature.md`](intensity_likelihood_quadrature.md).

All times below are **value-only** `log_likelihood_*` on one Mac CPU, float64, generative grid (15% centric, \(\sigma_A\) and \(\sigma_Z\) as in the quadrature notes). Same box for “before” and “after”.

## What was expensive

Per reflection the target is

\[
L = \int_0^\infty f_{\mathrm{Rice/Woolfson}}(E \mid E_C,\sigma_A)\, f_{\mathrm{noise}}(Z_o \mid E^2,\sigma_Z[,\,\nu])\,dE.
\]

The old loop paid for a lot that is not that integral:

1. **Newton on the full batch until the last row converged.** Mean acentric iters ~3.4, mean centric ~7, cap 20. One flat-topped centric kept every reflection in the integrand for 20 steps.
2. **Both Rice and Woolfson every Newton step**, then `where(centric, …)`.
3. **I-mode Newton on every reflection**, then a mask so only strong acentrics (\(Z_o/\sigma_Z \ge 5\), not boundary) used \(I_0\).
4. **Quadrature asked for \(g\) and \(H\)** (`i1e`, Hessian) when it only needs \(\log f\). `i0e` was also computed twice.
5. **Student-\(t\)** = 12 serial normal-noise solves (one per \(u=\log\lambda\) node).
6. **Gauss nodes** re-wrapped `numpy → tensor` on every call.
7. **No memory of last \(E_0\)**. Refine reuses the same Miller order; the next cycle started from the Rice guess again.

The 24 / 7 quadrature nodes themselves were already cheap and were left alone.

## What changed

| change | where | same \(\log L\)? |
|---|---|---|
| Split acentric / centric Newton (12 vs 20 iters) | `_newton_E_split` | yes |
| Peel stragglers: full SIMD for 4 steps, then gather the unconverged tail only when ≥90% of the *current* set has converged **or** ≤256 remain | `_newton_on` | yes (window-tolerant) |
| I-Newton only on strong-acentric candidates; start from last \(I_0\) when cached | `quadrature_terms_normal` | yes |
| Log-only Rice / Woolfson / \(I\) at the nodes; share one `i0e` in Newton | `acen_E_log`, `cen_E_log`, `acen_I_log` | yes |
| One batched normal-noise call per distinct \(\nu\) (`repeat_interleave` over \(u\)) | `quadrature_terms_t` | yes |
| Cache Legendre / Hermite tensors by `(n, dtype, device)` | `_gauss_t` | yes |
| Warm start: last \(E_0\), \(I_0\) keyed by `(N, device, dtype)`; Rice start if \(E_C,\sigma_A,Z_o,\sigma_Z\) jumped | `_warm_x0` / `_store_warm` | yes (\(10^{-13}\) on replay) |

Peeling on the first converged row would copy ~99% of the batch to drop 1%. Waiting for a dense tail is cheaper. `clear_mode_cache()` drops the warm table (tests / new dataset).

Related, not quadrature math: C++ stamp keeps \(\rho\) and \(F\) on CPU even when the worker is MPS. `Observations.to_like(F)` moves \(I_{\mathrm{obs}}\) onto \(F\)'s device so evaluate does not die with `mps:0 and cpu`.

## Times

### 80 000 reflections (this machine)

| evaluator | Gaussian | Student-\(t\) \(\nu=5\), \(n_u=12\) |
|---|---|---|
| Before these changes | 245 ms | 3387 ms |
| After (cold cache) | 96 ms (**2.6×**) | 676 ms (**5.0×**) |
| After (replay, identical inputs) | 63 ms (**3.9×**) | 409 ms (**8.3×**) |
| After (1% \(E_C\) jitter, typical refine step) | 81 ms (**3.0×**) | 504 ms (**6.7×**) |

Per-reflection, replay Gaussian is ~0.8 µs; original was ~3.1 µs.

### 20 000 reflections (after the vectorized / split pass, before warm start was wired)

| evaluator | Gaussian | Student-\(t\) \(\nu=5\) |
|---|---|---|
| Before | 71 ms | 967 ms |
| After that pass | 25 ms | 175 ms |

### Newton iteration counts (40 000 reflections, optimized + warm)

| call | mean iters | max | warm hits | wall |
|---|---|---|---|---|
| Cold | 3.76 | 20 | 0 | 73 ms |
| Replay | 0.43 | 20 | 39 132 | 44 ms |
| 1% \(E_C\) drift | 2.19 | 20 | 39 132 | 60 ms |

A few centrics still hit the cap. The Legendre window is built for that; replay \(\log L\) matched to \(1\times10^{-13}\).

## What a refine cycle sees

- **First target eval** (empty cache): ~2.5–3× Gaussian, ~5× \(t\).
- **Later evals / next macrocycle** (same Miller list, \(E_C\) moved a little): ~3–4× Gaussian, ~7–8× \(t\).
- **Nuisance fit** (many nearby \(\sigma_A\) steps): closer to the replay column.

`Target.evaluate` with `compute_curvature=True` still runs the quadrature twice (value+grad, then Hessian). Refine `target_and_gradients` defaults `precondition=False`, so that double pass is off unless you turn it on.

## MPS / CUDA

The C++ stamp still builds ρ on the host (no Metal kernel). When the worker device is `mps` or `cuda`, `ml_i` evaluate uploads the 1-D miller arrays (`_accel_like`) and runs Newton + quadrature there, then downloads \(G_h\). MPS is float32 only.

Same 80 000-reflection generative grid, value-only (this machine):

| | CPU float64 | MPS float32 |
|---|---|---|
| Gaussian (warm) | 56 ms | 40 ms |
| Student-\(t\) \(\nu=5\) cold / warm | 631 / 392 ms | 181 / 83 ms |

\(\lvert\log L_{\mathrm{cpu64}} - \log L_{\mathrm{mps32}}\rvert\): median \(2\times10^{-7}\), 99th percentile \(1\times10^{-5}\), max \(6\times10^{-5}\) (8k draw). That is inside the documented Legendre error. Maps and `evaluate(..., compute_curvature=True)` also run on Metal.

## What this does not save

The 24-point (weak / centric) and 7-point (strong) integrands are now the bulk of the time. Warm start only removes Newton. Cutting node count, a Laplace/GH fallback on mid-SNR acentrics, or a float32 Legendre branch would be the next chunk — those change the quadrature error, unlike the work above.
