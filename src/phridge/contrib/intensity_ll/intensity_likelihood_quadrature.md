# Intensity-based crystallographic likelihoods by adaptive quadrature

Notes on simplifying the quadrature scheme of Zwart & Perryman (2020), *Acta Cryst.* D76, 736–750, with a numerical study on a generative parameter grid, a vectorized PyTorch implementation (`mli.py`), and regression reference values.

## 1. Problem

Per reflection, in normalized units ($E_C$ model amplitude, $\sigma_A$, $Z_o$ observed normalized intensity, $\sigma_Z$ its standard deviation), the intensity-based likelihood marginalizes the error-free amplitude $E$:

$$L(E_C) = \int_0^\infty f(E \mid E_C, \sigma_A)\, f_{\text{noise}}(Z_o \mid E^2, \sigma_Z)\, dE$$

with the acentric Rice density

$$f_a(E \mid E_C, \sigma_A) = \frac{2E}{a}\exp\!\left(-\frac{E^2 + \sigma_A^2 E_C^2}{a}\right) I_0\!\left(\frac{2\sigma_A E E_C}{a}\right), \qquad a = 1 - \sigma_A^2$$

the centric Woolfson density

$$f_c(E \mid E_C, \sigma_A) = \sqrt{\frac{2}{\pi a}}\exp\!\left(-\frac{E^2 + \sigma_A^2 E_C^2}{2a}\right)\cosh\!\left(\frac{\sigma_A E E_C}{a}\right)$$

and either normal noise $\mathcal{N}(Z_o \mid E^2, \sigma_Z^2)$ or Student-$t$ noise with $\nu = N_{\text{eff}} - 1$ degrees of freedom.

The 2020 paper evaluates this with a power transform ($E = u^{1/\gamma}$, $\gamma = 2$ found best), a logistic compression centred on the integrand's maximum (found by Newton, "within 50 evaluations"), an equispaced trapezoid on $[0, 1]$ with 7–49 points, and hand-derived gradients that carry $\partial w_j / \partial Y$ terms because the nodes move with $E_C$.

## 2. What can be simplified

### 2.1 Work in intensity: the noise term and the Rice exponential merge

Written in $I = E^2$ the acentric Rice density has no singularity and its exponential factor combines exactly with normal noise:

$$\exp\!\left(-\frac{I}{a}\right)\mathcal{N}(Z_o \mid I, \sigma_Z^2) = \exp\!\left(-\frac{Z_o}{a} + \frac{\sigma_Z^2}{2a^2}\right)\mathcal{N}\!\left(I \,\middle|\, Z_o - \frac{\sigma_Z^2}{a},\, \sigma_Z^2\right)$$

so the whole likelihood is one Gaussian window times one smooth, monotone factor:

$$L = \frac{1}{a}\exp\!\left(-\frac{\sigma_A^2 E_C^2 + Z_o}{a} + \frac{\sigma_Z^2}{2a^2}\right)\int_0^\infty \mathcal{N}(I \mid \mu', \sigma_Z^2)\, I_0\!\left(c\sqrt{I}\right) dI, \qquad c = \frac{2\sigma_A E_C}{a}$$

This is why $\gamma = 2$ works: it is the variable in which the noise is exactly Gaussian. With $I = \sigma_Z t$ the integral is a two-parameter function $T(m, k) = \int_0^\infty \mathcal{N}(t \mid m, 1)\, I_0(k\sqrt{t})\, dt$ with $m = \mu'/\sigma_Z$ and $k = c\sqrt{\sigma_Z}$; $E_C$ enters only through $k$ and the prefactor. (A 2-D table of $\log T$ minus its Laplace approximation is a viable alternative evaluator; it was checked to be smooth over $m \in [-12, 40]$, $k \in [0, 150]$ but not pursued further because the quadrature below is already cheap.)

### 2.2 Adaptive Gauss–Hermite instead of transform + trapezoid

For a unimodal integrand the standard tool is Gauss–Hermite about the Laplace point (Liu & Pierce, *Biometrika* 1994): find the mode $x_0$ and curvature $H = g''(x_0)$ of $g = \log f$, set $\sigma = (-H)^{-1/2}$, and use nodes $x_0 + \sqrt{2}\sigma\xi_j$ with Hermite weights. The mode is one Newton solve away, and nothing else is needed.

### 2.3 Weak data is the only hard regime, and it needs a shape-free rule

When the Gaussian window straddles $I = 0$ (roughly $Z_o/\sigma_Z \lesssim 5$) the Laplace Gaussian is a poor shape: skew from $E \leftrightarrow E^2$, truncation at zero, flat-topped folded normals for centrics. The paper's compression transform fights this. A rule that assumes no shape does better: Gauss–Legendre on the window $[\max(0, E_0 - k\sigma),\, E_0 + k\sigma]$ in $E$, with $E_0$ and $\sigma$ from the same Newton solve.

### 2.4 Student-$t$ tails from a scale mixture, not from more nodes

The $t$ density is a Gamma mixture of normals:

$$t_\nu(Z_o \mid I, \sigma_Z^2) = \int_0^\infty \mathcal{N}\!\left(Z_o \mid I, \sigma_Z^2/\lambda\right)\mathrm{Gamma}\!\left(\lambda \mid \tfrac{\nu}{2}, \tfrac{\nu}{2}\right) d\lambda$$

so the $t$ likelihood is the normal result with $\sigma_Z^2 \to \sigma_Z^2/\lambda$ integrated over $\lambda$. Gauss–Laguerre in $\lambda$ converges badly for small $\nu$; a Gauss rule for the weight of $u = \log\lambda$ (density $\propto \exp(\tfrac{\nu}{2}u - \tfrac{\nu}{2}e^{u})$, built once per $\nu$ by Golub–Welsch) converges fast. $\nu$ takes few distinct integer-redundancy values, so the rules are cached.

### 2.5 No hand-derived derivatives

In torch the evaluator is a `logsumexp` over nodes; autograd differentiates through the integrand at the nodes. The mode and window are computed detached (nodes treated as fixed), so the gradient is that of the quadrature approximation with frozen nodes; the omitted term is of the order of the quadrature error. Tracking the window through the Newton steps is available (`differentiate_window=True`) at roughly double the cost.

### 2.6 Numerical hygiene

Bessel functions appear only as $I_0$ via `i0e` and the ratio $I_1/I_0$ via `i1e/i0e`; $\log I_0(x) = \log i0e(x) + x$. $\cosh$ is written as $|x| + \log(1 + e^{-2|x|}) - \log 2$. All sums are log-sum-exp. The Newton derivatives are analytic, using $\frac{d}{dx}\frac{I_1}{I_0} = 1 - \frac{1}{x}\frac{I_1}{I_0} - \left(\frac{I_1}{I_0}\right)^2$.

## 3. Numerical study

### 3.1 Test grid

Sampled from the generative model so that the density of cases matches real data: $\sigma_A \in \{0.3, 0.6, 0.8, 0.9, 0.95, 0.99\}$; $E_{\text{true}}$ from the Wilson distribution (acentric: $E^2 \sim \text{Exp}(1)$; centric: $|\mathcal{N}(0,1)|$); $E_C$ from the Rice/Woolfson conditional given $E_{\text{true}}$ and $\sigma_A$; $\sigma_Z$ log-uniform on $[0.02, 3]$; $Z_o = E_{\text{true}}^2 + \sigma_Z\epsilon$. Resulting ranges: $E_C \in [0.004, 3.2]$; $Z_o/\sigma_Z$ 1st/50th/99th percentiles $-1.6 / 2.6 / 98$; 13% negative intensities. Reference values: 200 000–400 000-point trapezoid in $y = \log E$. Errors below are absolute errors in $\log L$, maxima over 1200–2000 draws.

### 3.2 Mode finding

Newton on the analytic log-integrand in $E$, start $\max\!\big(\sqrt{\max(Z_o, 0)},\ \sqrt{\sigma_A^2 E_C^2 + a/2}\big)$ (observed amplitude or Rice mode, whichever is larger), step damped so $x$ never crosses zero. The integrand was unimodal on every draw checked.

| kind | mean iterations | max iterations | final gradient |
|------|-----------------|----------------|----------------|
| acentric | 3.4 | 8 | $10^{-12}$ |
| centric | 7 | 20 (cap; flat-topped cases) | — |

An imprecise centric mode is harmless because the Legendre window tolerates it.

### 3.3 Which rule where (normal noise)

| regime | rule | acentric max error | centric max error |
|--------|------|--------------------|-------------------|
| $Z_o/\sigma_Z \geq 5$, acentric | 7-node Gauss–Hermite in $I$ about the Laplace point | $5\times10^{-7}$ (9 nodes: $8\times10^{-8}$) | — |
| all other | 24-node Gauss–Legendre on $[\max(0, E_0 - 8\sigma), E_0 + 8\sigma]$ | $2\times10^{-6}$ | $2\times10^{-5}$ (99th percentile); $6\times10^{-4}$ worst |

Failed alternatives, for the record: plain Gauss–Hermite in $E$ with negative nodes dropped plateaus at $2\times10^{-2}$ (lost mass); Gauss–Hermite in $\log E$ converges slowly (skew); a truncated-normal Gauss rule tabulated in $\mu/\sigma$ works but is beaten by the clipped Legendre window; Gauss–Legendre on the full $[0, E_{\max}]$ interval under-resolves narrow integrands ($10^{-1}$ errors at $Z_o/\sigma_Z \approx 4$ with small $\sigma_Z$).

Two safety rails on the window were needed: the upper limit is capped by the physical cutoff $E_{\max} = \min\!\big(\sqrt{\max(Z_o,0) + 9\sigma_Z},\ \sigma_A E_C + 6\sqrt{a}\big)$ but never below $E_0 + 3\sigma$ (a model that disagrees with the data must not have its mode cut off), and reflections whose mode is at the origin (gradient negative at $E \to 0$) use $[0, E_{\max}]$ with the curvature-based upper cap when available.

### 3.4 Student-$t$ noise

Outer rule in $u = \log\lambda$, inner evaluator as above; acentric, 500 draws:

| $\nu$ | 6 nodes | 10 nodes | 16 nodes |
|-------|---------|----------|----------|
| 2 | $4.6\times10^{-2}$ | $7\times10^{-3}$ | $1.6\times10^{-4}$ |
| 3 | $2.6\times10^{-2}$ | $1.9\times10^{-3}$ | $5\times10^{-5}$ |
| 8 | $8\times10^{-4}$ | $1.6\times10^{-5}$ | $6\times10^{-7}$ |

For comparison, Gauss–Laguerre directly in $\lambda$ gave $10^{-2}$ at 24 nodes for $\nu = 3$.

### 3.5 Gradients

Autograd (window detached) against central differences of the dense reference, 200 acentric draws: $\partial\log L/\partial E_C$ max relative error $1.6\times10^{-4}$ (99th percentile $2\times10^{-5}$); $\partial\log L/\partial\sigma_A$ max $5\times10^{-4}$. With the window tracked: $E_C$ max $4\times10^{-5}$.

### 3.6 Cost

Single-threaded numpy prototype: 15 µs per reflection (1.5 s per 100 000), Newton alone 4 µs. PyTorch on two CPU threads, float64, 100 000 reflections: values 2.5 s; value plus gradients in $E_C$ and $\sigma_A$ 1.7 s; $t$ noise with 12 outer nodes 14 s. Every operation is elementwise over arrays of size $N$ or $N \times 24$, so CUDA cost should be low milliseconds per 100 000; float32 is adequate for the Legendre branch.

## 4. Recommended algorithm

Precompute once: Legendre nodes (24), Hermite nodes (7), log-gamma rules per $\nu$ (12–16). Per batch of reflections, vectorized:

1. Newton in $E$ on the merged acentric/centric log-integrand (mask-selected), 12–20 iterations max, damped; obtain $E_0$, $H_E$.
2. Classify: `boundary` if $H_E \geq 0$, the gradient at the origin is negative, or the Laplace width $\sigma_E$ exceeds the physical cutoff $E_{\max}$ (flat-topped centric integrands); `strong` if acentric, not boundary, and $Z_o/\sigma_Z \geq 5$.
3. Strong: a few Newton steps in $I$ from $E_0^2$; 7-node Gauss–Hermite in $I$.
4. Otherwise: 24-node Gauss–Legendre on the clipped, capped window in $E$.
5. $t$ noise: wrap 1–4 in the log-$\lambda$ rule with $\sigma_Z \to \sigma_Z e^{-u_j/2}$.
6. `logsumexp` per reflection; autograd for $\partial/\partial E_C$, $\partial/\partial\sigma_A$ and, through phridge's `Target` base class, $G_h$ and the curvatures.

What is kept from the paper unchanged: the $\sigma_A$ normalization, the $\nu = N_{\text{eff}} - 1$ argument, the Sivia intensity-to-amplitude conversion as a fallback, and the validation methodology. What is dropped: the power and logistic transforms, the trapezoid, the mode search inside the transform, and the hand-derived gradient with node-derivative terms.

## 5. PyTorch implementation (`mli.py`)

```python
from mli_quad import log_likelihood_normal, log_likelihood_t, normalize

# tensors (N,), any device; centric is a bool tensor (N,)
logL = log_likelihood_normal(E_C, sigma_A, Z_o, sigma_Z, centric)
logL_t = log_likelihood_t(E_C, sigma_A, Z_o, sigma_Z, centric, nu, n_u=12)

# raw data -> normalized: E_C = |F_c| / sqrt(eps Sigma), Z_o = I / (eps Sigma), sigma_Z = sigma_I / (eps Sigma)
E_C, sigma_A, Z_o, sigma_Z = normalize(f_calc_abs, i_obs, sig_i, epsilon, sigma_wilson, sigma_a)
```

Keyword options: `snr_strong=5.0`, `n_hermite=7`, `n_legendre=24`, `k_window=8.0`, `differentiate_window=False`, `return_stats=False` (returns Newton iteration counts and branch populations).

As a phridge target: an `ml_i` subclass of `Target` whose `per_reflection` returns `-log_likelihood_normal(...)` (or the $t$ version) on the normalized observations; `amplitude_only = True` holds since the dependence on $F_h$ is through $|F_h|$, so the curvatures come for free.

Open items: a gradient test for the $t$ model; an extreme-parameter sweep ($\sigma_A > 0.99$, $E_C > 4$) where the Rice factor becomes narrow enough that the window should use the Rice width explicitly; GPU timing.

## 6. Regression reference values

Reference $\log L$ from an 800 001-point trapezoid in $\log E$, independent of the quadrature code. Units normalized; `nu = None` is normal noise.

| # | kind | $E_C$ | $\sigma_A$ | $Z_o$ | $\sigma_Z$ | $\nu$ | log L (reference) | log L (`mli_quad`) | diff |
|---|------|-------|------------|-------|------------|-------|-------------------|--------------------|------|
| 1 | acen | 1.0 | 0.8 | 1.0 | 0.1 | — | -0.7224563217 | -0.7224563217 | -3.3e-16 |
| 2 | acen | 1.0 | 0.8 | 1.0 | 1.0 | — | -1.1292240582 | -1.1292240582 | -7.2e-12 |
| 3 | acen | 0.5 | 0.9 | 4.0 | 0.2 | — | -12.6745099709 | -12.6745099709 | +1.8e-15 |
| 4 | acen | 2.5 | 0.9 | 1.0 | 0.2 | — | -8.4753016262 | -8.4753016284 | -2.1e-09 |
| 5 | acen | 1.2 | 0.5 | 0.05 | 0.05 | — | -0.4098844610 | -0.4098844610 | +3.4e-11 |
| 6 | acen | 0.8 | 0.95 | -0.3 | 0.5 | — | -1.7016432112 | -1.7016432112 | +1.7e-11 |
| 7 | acen | 3.0 | 0.99 | 9.0 | 0.05 | — | -0.4487587321 | -0.4487587321 | -5.7e-14 |
| 8 | acen | 0.05 | 0.3 | 0.0 | 0.02 | — | -0.6165280078 | -0.6165280078 | -4.9e-13 |
| 9 | acen | 1.5 | 0.7 | 2.0 | 3.0 | — | -2.0961729587 | -2.0961729587 | +2.5e-13 |
| 10 | cen | 1.0 | 0.8 | 1.0 | 0.1 | — | -1.1418004952 | -1.1418003769 | +1.2e-07 |
| 11 | cen | 1.0 | 0.8 | 1.0 | 1.0 | — | -1.2408982398 | -1.2408981883 | +5.1e-08 |
| 12 | cen | 0.3 | 0.9 | 0.5 | 0.5 | — | -0.5245635522 | -0.5245634137 | +1.4e-07 |
| 13 | cen | 0.2 | 0.6 | -0.5 | 0.8 | — | -1.3784319882 | -1.3784318075 | +1.8e-07 |
| 14 | cen | 2.0 | 0.95 | 4.0 | 0.1 | — | -1.1947262485 | -1.1947263473 | -9.9e-08 |
| 15 | cen | 0.05 | 0.3 | 0.0 | 0.02 | — | 0.9280925597 | 0.9280933018 | +7.4e-07 |
| 16 | acen | 1.0 | 0.8 | 1.0 | 0.3 | 3.0 | -0.7853041400 | -0.7853047598 | -6.2e-07 |
| 17 | acen | 0.8 | 0.95 | -0.3 | 0.5 | 3.0 | -1.6879317350 | -1.6879317614 | -2.6e-08 |
| 18 | acen | 2.0 | 0.9 | 3.0 | 0.2 | 8.0 | -1.0464830298 | -1.0464830301 | -2.6e-10 |
| 19 | cen | 1.0 | 0.8 | 1.0 | 0.3 | 4.0 | -1.1047172878 | -1.1047199675 | -2.7e-06 |

Gradient references (central differences of the dense reference, $h = 10^{-5}$), normal-noise cases:

| # | ∂logL/∂E_C (ref) | autograd | ∂logL/∂σ_A (ref) | autograd |
|---|---|---|---|---|
| 1 | 0.32541550 | 0.32541550 | 1.95342008 | 1.95342008 |
| 2 | -0.15000226 | -0.15000226 | 0.47909619 | 0.47909618 |
| 3 | 13.25296948 | 13.25296948 | -101.51838425 | -101.51838326 |
| 4 | -11.10741553 | -11.10741558 | -92.59095059 | -92.59095068 |
| 5 | -0.73432123 | -0.73432123 | -1.07643958 | -1.07643958 |
| 6 | -3.42474866 | -3.42474866 | -3.35765804 | -3.35765801 |
| 7 | 2.79525610 | 2.79525610 | 53.42949442 | 53.42947810 |
| 8 | -0.00971840 | -0.00971840 | 0.64611636 | 0.64611636 |
| 9 | -0.00346036 | -0.00346036 | 0.13373586 | 0.13373586 |
| 10 | 0.37941924 | 0.37941920 | 2.21032660 | 2.21032664 |
| 11 | -0.16203645 | -0.16203653 | 0.51032139 | 0.51032116 |
| 12 | 0.01602851 | 0.01602833 | 0.34717686 | 0.34717712 |
| 13 | -0.07587507 | -0.07587508 | 0.59772820 | 0.59772825 |
| 14 | 0.96362844 | 0.96362855 | 10.73263718 | 10.73263687 |
| 15 | -0.00489343 | -0.00489343 | 0.32533320 | 0.32533320 |

Python literal, `(kind, E_C, sigma_A, Z_o, sigma_Z, nu, log_L)`:

```python
REFERENCE = [
    ("acen", 1.0, 0.8, 1.0, 0.1, None, -0.7224563217),
    ("acen", 1.0, 0.8, 1.0, 1.0, None, -1.1292240582),
    ("acen", 0.5, 0.9, 4.0, 0.2, None, -12.6745099709),
    ("acen", 2.5, 0.9, 1.0, 0.2, None, -8.4753016262),
    ("acen", 1.2, 0.5, 0.05, 0.05, None, -0.4098844610),
    ("acen", 0.8, 0.95, -0.3, 0.5, None, -1.7016432112),
    ("acen", 3.0, 0.99, 9.0, 0.05, None, -0.4487587321),
    ("acen", 0.05, 0.3, 0.0, 0.02, None, -0.6165280078),
    ("acen", 1.5, 0.7, 2.0, 3.0, None, -2.0961729587),
    ("cen", 1.0, 0.8, 1.0, 0.1, None, -1.1418004952),
    ("cen", 1.0, 0.8, 1.0, 1.0, None, -1.2408982398),
    ("cen", 0.3, 0.9, 0.5, 0.5, None, -0.5245635522),
    ("cen", 0.2, 0.6, -0.5, 0.8, None, -1.3784319882),
    ("cen", 2.0, 0.95, 4.0, 0.1, None, -1.1947262485),
    ("cen", 0.05, 0.3, 0.0, 0.02, None, 0.9280925597),
    ("acen", 1.0, 0.8, 1.0, 0.3, 3.0, -0.7853041400),
    ("acen", 0.8, 0.95, -0.3, 0.5, 3.0, -1.6879317350),
    ("acen", 2.0, 0.9, 3.0, 0.2, 8.0, -1.0464830298),
    ("cen", 1.0, 0.8, 1.0, 0.3, 4.0, -1.1047172878),
]

GRADIENT_REFERENCE = [  # (dlogL/dE_C, dlogL/dsigma_A) for REFERENCE[0..14]
    (0.32541550, 1.95342008), (-0.15000226, 0.47909619), (13.25296948, -101.51838425),
    (-11.10741553, -92.59095059), (-0.73432123, -1.07643958), (-3.42474866, -3.35765804),
    (2.79525610, 53.42949442), (-0.00971840, 0.64611636), (-0.00346036, 0.13373586),
    (0.37941924, 2.21032660), (-0.16203645, 0.51032139), (0.01602851, 0.34717686),
    (-0.07587507, 0.59772820), (0.96362844, 10.73263718), (-0.00489343, 0.32533320),
]
```

Suggested tolerances: `abs=1e-8` on acentric normal-noise values, `abs=2e-6` on centric and $t$-noise values (case 19 is the loosest at $2.7\times10^{-6}$), `rel=1e-5` on gradients (case 7's $\sigma_A$ gradient is the loosest at $3\times10^{-7}$ relative; cases 11–12 at $5\times10^{-7}$ to $10^{-5}$). Coverage: strong and noisy consistent data (1, 2, 10, 11); model weaker or stronger than the data (3, 4); weak reflection and negative intensities (5, 6, 13); sharp Rice at $\sigma_A = 0.99$ (7); near-zero corner (8, 15); very large $\sigma_Z$ (9); flat-topped centric (12); $t$ noise at $\nu = 3, 4, 8$ (16–19).

## 7. Files

- `mli_quad.py` — PyTorch implementation (`log_likelihood_normal`, `log_likelihood_t`, `normalize`).
- `quad.py` — numpy prototype including the dense `reference()` integrator used for validation and the truncated-normal rules that were tried and dropped.
- `run1.py` — generative test grid (`sample(kind, N)`).

Sources: Zwart & Perryman, [IUCr full text](https://journals.iucr.org/d/issues/2020/08/00/rr5195/index.html), [bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2020.01.12.903690v2.full); Liu & Pierce, *Biometrika* 81 (1994) 624–629.
