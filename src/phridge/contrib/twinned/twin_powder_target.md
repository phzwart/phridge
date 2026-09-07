# Likelihoods for mixed-intensity observations: twinning and powder overlap

Math and numerical tests for the characteristic-function (CF) route to the marginal intensity likelihood when one observation mixes several reflections. Companion to `intensity_likelihood_quadrature.md`; implementation in `twin_cf.py` (PyTorch).

## 1. Model

Normalized units throughout. One observation sees $K$ mates with known weights $\alpha_k$ (twin fractions; for powders multiplicity × Lorentz-polarization × profile weight):

$$Z_o = \sum_{k=1}^{K}\alpha_k I_k + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, \sigma_Z^2)$$

Given the model, the true intensities are independent with the Rice (acentric) or Woolfson (centric) prior in intensity form, parameterized by $E_{C,k}$ and $\sigma_A$; write $a = 1 - \sigma_A^2$ and $\lambda_k = \sigma_A^2 E_{C,k}^2$. The likelihood is the density of $Z_o$:

$$L = \int_{\mathbb{R}_+^K}\prod_{k=1}^{K} p(I_k \mid E_{C,k}, \sigma_A)\;\mathcal{N}\!\Big(Z_o \,\Big|\, \sum_k\alpha_k I_k,\ \sigma_Z^2\Big)\, dI_1\cdots dI_K$$

For $K = 1$ this is the untwinned integral. For a powder cluster the scalar $Z_o$ is replaced by the vector of Pawley-extracted intensities $\hat{\mathbf{I}}$ with covariance $C$ (Section 4).

## 2. Characteristic functions

The density of a sum of independent variables is a convolution; in Fourier space it is a product. With $\varphi_X(t) = \mathbb{E}[e^{itX}]$:

$$\varphi_{Z_o}(t) = e^{-\sigma_Z^2t^2/2}\prod_{k=1}^{K}\varphi_{I_k}(\alpha_k t)$$

The normalized acentric intensity given the model is $\tfrac{a}{2}\chi^2_2(2\lambda/a)$ (squared modulus of $\sigma_AE_Ce^{i\phi}$ plus a complex Gaussian of total variance $a$); the centric one is $a\,\chi^2_1(\lambda/a)$. From $\varphi_{\chi^2_\nu(\delta)}(u) = (1-2iu)^{-\nu/2}\exp\big(i\delta u/(1-2iu)\big)$:

$$\varphi_{\text{acen}}(t) = \frac{1}{1 - iat}\exp\!\left(\frac{i\lambda t}{1 - iat}\right)$$

$$\varphi_{\text{cen}}(t) = \frac{1}{\sqrt{1 - 2iat}}\exp\!\left(\frac{i\lambda t}{1 - 2iat}\right)$$

Both have mean $a + \lambda$; variances are $a^2 + 2a\lambda$ (acentric) and $2a^2 + 4a\lambda$ (centric). No Bessel functions appear: $I_0$ and $\cosh$ in the amplitude densities are the result of integrating the phase out, which in Fourier space is already done. Both denominators have real part 1, so the principal branches of $\log$ and $\sqrt{\cdot}$ are the correct ones with no branch tracking.

## 3. Inversion and quadrature

$$L = \frac{1}{2\pi}\int_{-\infty}^{\infty} e^{-itZ_o}\,e^{-\sigma_Z^2t^2/2}\prod_k\varphi_k(\alpha_k t)\,dt$$

Substituting $t = s/\sigma_Z$ turns the noise factor into the Gauss–Hermite weight $e^{-s^2/2}$:

$$L = \frac{1}{2\pi\sigma_Z}\int_{-\infty}^{\infty} e^{-s^2/2}\,\exp\!\Big[-\frac{isZ_o}{\sigma_Z} + \sum_k\log\varphi_k\!\Big(\frac{\alpha_k s}{\sigma_Z}\Big)\Big]\,ds \approx \frac{1}{2\pi\sigma_Z}\sum_j W_j\,\mathrm{Re}\big[\cdots\big]_{s = S_j}$$

with fixed Hermite nodes $S_j$, weights $W_j$. There is no mode search and no per-reflection window. Derivatives with respect to $E_{C,k}$, $\sigma_A$ and $\alpha_k$ are one autograd pass through complex arithmetic.

**Resolution requirement.** The integrand's phase is $e^{-is(Z_o - \mu_T)/\sigma_Z}$ times slowly varying factors, where $\mu_T = \sum_k\alpha_k(a + \lambda_k)$ is the model's expected observation (the CF's own linear phase supplies the $\mu_T$ automatically; explicitly centring the phase changes nothing numerically). The remaining structure of $\prod_k\varphi_k$ lives on the scale $t \sim 1/\sqrt{\mathrm{Var}_{\text{prior}}}$, i.e. $s \sim \sigma_Z/\sqrt{\mathrm{Var}_{\text{prior}}}$ where

$$\mathrm{Var}_{\text{prior}} = \sum_k\alpha_k^2\,\mathrm{Var}(I_k)$$

So the controlling parameter is the noise-to-prior-width ratio $r = \sigma_Z/\sqrt{\mathrm{Var}_{\text{prior}}}$: when the measurement is at least as uncertain as the prior, the integrand is smooth on the Hermite window; when the measurement is much sharper than the prior, the CF varies faster than the weight can resolve and the node count grows like $r^{-2}$. This is the precise meaning of "weak data" for this route, and it is the opposite regime from the one where direct-space adaptive quadrature is hard. The two are complementary along $r$.

## 4. Vector observations with covariance (powder clusters)

For a cluster of $K$ overlapping reflections and $m$ profile points, $\mathbf{y} = \mathbf{b} + W\mathbf{I} + \boldsymbol{\varepsilon}$ with $\boldsymbol{\varepsilon} \sim \mathcal{N}(0, \Sigma)$. Because the map is linear and the noise Gaussian, the dependence on $\mathbf{I}$ passes through the weighted least-squares estimate:

$$\hat{\mathbf{I}} = (W^\top\Sigma^{-1}W)^{-1}W^\top\Sigma^{-1}(\mathbf{y} - \mathbf{b}), \qquad C = (W^\top\Sigma^{-1}W)^{-1}$$

and $\mathcal{N}(\mathbf{y} \mid \mathbf{b} + W\mathbf{I}, \Sigma) = \mathcal{N}(\hat{\mathbf{I}} \mid \mathbf{I}, C) \times (\text{term independent of } \mathbf{I})$. These are the Pawley-extracted intensities and their covariance, so the marginal Rietveld likelihood of the cluster is the twin integral with a covariance:

$$L = \int_{\mathbb{R}_+^K}\prod_k p(I_k \mid E_{C,k}, \sigma_A)\;\mathcal{N}(\hat{\mathbf{I}} \mid \mathbf{I}, C)\,d\mathbf{I} = \frac{1}{(2\pi)^K}\int e^{-i\mathbf{t}^\top\hat{\mathbf{I}}}\,e^{-\mathbf{t}^\top C\mathbf{t}/2}\prod_k\varphi_k(t_k)\,d\mathbf{t}$$

Whitening with $C = LL^\top$, $\mathbf{t} = L^{-\top}\mathbf{s}$, $d\mathbf{t} = \det(C)^{-1/2}d\mathbf{s}$ gives a product Gauss–Hermite rule in $K$ dimensions. The three regimes of $C$: diagonal (separated peaks) reduces to independent single-reflection problems; singular along difference directions (exact overlap) reduces to the twin sum, with $\alpha_k \propto$ multiplicity × Lorentz factor; banded with negative neighbour correlations (partial overlap) is the genuinely new case. In the last case the *sum* direction is typically well determined and the *difference* directions poorly, so after whitening the sum direction has small $r$ — which is exactly where the CF rule needs many nodes. This is why the CF route alone does not finish the powder problem: the precise directions want direct-space treatment (Laplace point plus low-order Hermite in intensity space, or the polar/Legendre rule for pairs) and the imprecise ones want the CF.

## 5. Exact limits used as tests

Equal fractions and all-acentric mates: $\sum_k\tfrac{1}{K}I_k = \tfrac{a}{2K}\chi^2_{2K}(\Lambda)$ with $\Lambda = \sum_k 2\lambda_k/a$, so the prior density of the twinned intensity is a scaled noncentral $\chi^2_{2K}$ in closed form and the problem reduces to the single-domain solver with $I_{K-1}$ in place of $I_0$. With $E_{C,k} = 0$, $K = 2$, $a = 1$ this is $p(T) = 4Te^{-2T}$, the Rees–Yeates twinned Wilson distribution. With $K = 1$ the CF result must agree with the direct quadrature of `mli_quad.py`.

## 6. Numerical results

Reference values: dense brute-force integrals on log-$E$ grids (2-D: $1200^2$–$2500^2$ points; 3-D: $110^3$) or the noncentral-$\chi^2$ closed form with a 400 001-point outer integral. Errors are absolute errors in $\log L$. Draws follow the generative model ($\sigma_A \in \{0.5, 0.7, 0.85, 0.95\}$, Wilson $E_{\text{true}}$, $E_C$ from the Rice conditional, $\sigma_Z$ set by the ratio $r$ defined above).

### 6.1 Single reflection, CF vs direct quadrature ($N = 1500$)

| Hermite nodes | $r \geq 1$ | $0.5 \leq r < 1$ | $0.3 \leq r < 0.5$ | $0.2 \leq r < 0.3$ |
|---|---|---|---|---|
| 32 | $3.9\times10^{-4}$ | fails | fails | fails |
| 64 | $1.8\times10^{-6}$ | $1.0\times10^{-1}$ | $9.5\times10^{-1}$ | fails |
| 96 | $4.5\times10^{-7}$ | $3.2\times10^{-3}$ | $2.7\times10^{-2}$ | fails |
| 128 | $4.5\times10^{-7}$ | $1.5\times10^{-4}$ | $5.0\times10^{-3}$ | $5.8\times10^{-2}$ |

Operating rule: CF with 64 nodes for $r \geq 1$, 128 nodes down to $r \approx 0.5$, direct-space quadrature below. In typical normalized units a reflection with $I/\sigma \approx 10$ has $r \approx 0.1$, so the CF route is for the genuinely weak tail of a data set and for prior-dominated overlap directions, not for the bulk.

### 6.2 Two-mate twins, CF vs brute-force 2-D ($N = 150$, $r \in [0.7, 4]$, one third with a centric mate)

| nodes | max | median | acentric/acentric | acentric/centric |
|---|---|---|---|---|
| 48 | $4.7\times10^{-5}$ | $1.3\times10^{-8}$ | $4.7\times10^{-5}$ | $1.9\times10^{-5}$ |
| 64 | $9.2\times10^{-6}$ | $5.0\times10^{-10}$ | $8.5\times10^{-6}$ | $9.2\times10^{-6}$ |
| 96 | $9.2\times10^{-6}$ | $1.3\times10^{-10}$ | $6.2\times10^{-7}$ | $9.2\times10^{-6}$ |

### 6.3 Perfect $K$-fold twins, CF vs noncentral $\chi^2_{2K}$ closed form ($N = 400$ each, $r \in [0.5, 4]$)

| $K$ | nodes | max | 99th pct | median | $r \geq 1$ | $0.5 \leq r < 1$ |
|---|---|---|---|---|---|---|
| 2 | 64 | $2.8\times10^{-3}$ | $7.3\times10^{-5}$ | $4\times10^{-10}$ | $2.1\times10^{-7}$ | $2.8\times10^{-3}$ |
| 2 | 128 | $9.1\times10^{-6}$ | $3.4\times10^{-7}$ | $8\times10^{-11}$ | $4.8\times10^{-9}$ | $9.1\times10^{-6}$ |
| 3 | 64 | $8.9\times10^{-4}$ | $1.2\times10^{-4}$ | $9\times10^{-16}$ | $1.2\times10^{-9}$ | $8.9\times10^{-4}$ |
| 3 | 128 | $1.4\times10^{-6}$ | $1.2\times10^{-7}$ | $4\times10^{-16}$ | $2.2\times10^{-13}$ | $1.4\times10^{-6}$ |
| 4 | 64 | $6.6\times10^{-4}$ | $2.9\times10^{-5}$ | $6\times10^{-16}$ | $1.3\times10^{-8}$ | $6.6\times10^{-4}$ |
| 4 | 128 | $1.5\times10^{-6}$ | $2.3\times10^{-8}$ | $7\times10^{-16}$ | $2.5\times10^{-14}$ | $1.5\times10^{-6}$ |
| 6 | 64 | $6.8\times10^{-5}$ | $8.9\times10^{-6}$ | $4\times10^{-16}$ | $1.2\times10^{-11}$ | $6.8\times10^{-5}$ |
| 6 | 128 | $1.0\times10^{-8}$ | $4.7\times10^{-10}$ | $4\times10^{-16}$ | $3.3\times10^{-14}$ | $1.0\times10^{-8}$ |

The cost is independent of $K$ except for the $K$ CF evaluations per node; more mates make the sum more Gaussian and the CF integrand smoother, so accuracy improves with $K$.

### 6.4 Clusters with covariance, CF (product Hermite) vs brute-force $K$-D ($N = 60$, $r \geq 1$ per dimension, neighbour correlations $-0.2$ to $-0.7$)

| $K$ | nodes/dim | max | median |
|---|---|---|---|
| 2 | 24 | $1.0\times10^{-2}$ | $6.5\times10^{-6}$ |
| 2 | 32 | $2.7\times10^{-3}$ | $7.4\times10^{-7}$ |
| 2 | 48 | $2.8\times10^{-4}$ | $1.3\times10^{-8}$ |
| 3 | 24 | $3.6\times10^{-2}$ | $1.7\times10^{-5}$ |
| 3 | 32 | $6.9\times10^{-3}$ | $1.2\times10^{-5}$ |

Slower convergence than the scalar case, for the reason given in Section 4: negative correlations make the sum direction sharper than the per-dimension $r$ suggests. Medians are fine; the tails are the well-determined sums, which should be handed to the direct-space rule.

### 6.5 Gradients and cost

Autograd against central differences of the brute-force 2-D reference (twin, $K = 2$, 96 nodes, 30 draws): $\partial\log L/\partial E_{C,1}$ max relative error $1.0\times10^{-7}$, $\partial\log L/\partial\alpha_1$ max $1.5\times10^{-7}$. Two CPU threads, float64, 100 000 observations, 64 nodes: $K = 2$ value 9.4 s, value plus gradient 3.8 s (first call includes warm-up); $K = 4$ value 8.7 s, value plus gradient 6.9 s. The work is $N \times 64 \times K$ complex exponentials, embarrassingly parallel.

## 7. Recommended use

For twins: two domains — the polar/Legendre direct-space rule for strong observations and the CF for $r \gtrsim 0.5$; three or more domains — CF for $r \gtrsim 0.5$, Laplace plus low-order product Hermite in intensity space for strong observations; equal fractions — the noncentral-$\chi^2_{2K}$ closed form as fast path and oracle. For powders: Pawley extraction with covariance, then per cluster whiten $C$, treat eigen-directions with $r \geq 1$ by CF and the sharp ones in direct space; exactly degenerate overlaps are twin sums. Everything shares the priors, $\sigma_A$ handling and Student-$t$ mixture of `mli_quad.py`, and all derivatives come from autograd.

## 8. Regression reference values

Twin likelihood, `log_likelihood_twin_cf` with 128 nodes; references independent of the CF code.

| # | $E_C$ | $\sigma_A$ | $\alpha$ | centric | $Z_o$ | $\sigma_Z$ | log L (reference) | log L (CF) | diff | reference |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [1.0, 1.0] | 0.8 | [0.5, 0.5] | [F, F] | 1.0 | 1.0 | -1.042960464 | -1.042960464 | +3.3e-11 | brute-force 2-D |
| 2 | [1.5, 0.5] | 0.8 | [0.7, 0.3] | [F, F] | 1.2 | 0.8 | -0.999685310 | -0.999685310 | +6.9e-11 | brute-force 2-D |
| 3 | [0.8, 1.2] | 0.9 | [0.6, 0.4] | [F, T] | 0.5 | 1.5 | -1.414319728 | -1.414319167 | +5.6e-07 | brute-force 2-D |
| 4 | [2.0, 0.3] | 0.95 | [0.5, 0.5] | [F, F] | -0.4 | 1.2 | -2.853142835 | -2.853142834 | +1.9e-10 | brute-force 2-D |
| 5 | [1.0, 1.0, 1.0] | 0.7 | [1/3, 1/3, 1/3] | [F, F, F] | 1.0 | 1.0 | -1.027607087 | -1.027607087 | -4.4e-16 | noncentral $\chi^2_6$ |
| 6 | [0.5, 1.5, 1.0, 2.0] | 0.85 | [0.25]×4 | [F]×4 | 1.5 | 1.0 | -1.017721960 | -1.017721960 | -4.4e-16 | noncentral $\chi^2_8$ |

```python
TWIN_REFERENCE = [  # (E_C, sigma_A, alpha, centric, Z_o, sigma_Z, log_L)
    ([1.0, 1.0], 0.8, [0.5, 0.5], [False, False], 1.0, 1.0, -1.042960464),
    ([1.5, 0.5], 0.8, [0.7, 0.3], [False, False], 1.2, 0.8, -0.999685310),
    ([0.8, 1.2], 0.9, [0.6, 0.4], [False, True], 0.5, 1.5, -1.414319728),
    ([2.0, 0.3], 0.95, [0.5, 0.5], [False, False], -0.4, 1.2, -2.853142835),
    ([1.0, 1.0, 1.0], 0.7, [1/3, 1/3, 1/3], [False]*3, 1.0, 1.0, -1.027607087),
    ([0.5, 1.5, 1.0, 2.0], 0.85, [0.25]*4, [False]*4, 1.5, 1.0, -1.017721960),
]
```

Suggested tolerance `abs=1e-6` (case 3, with a centric mate, is the loosest at $5.6\times10^{-7}$). The Rees–Yeates identity is a further unit test: with $E_C = 0$, $\sigma_A \to 0$, $\alpha = (\tfrac12, \tfrac12)$ and $\sigma_Z = 1$, the CF value must equal $\int 4Te^{-2T}\,\mathcal{N}(Z_o \mid T, 1)\,dT$.

## 9. Files
# Likelihoods for mixed-intensity observations: twinning and powder overlap

Math and numerical tests for the characteristic-function (CF) route to the marginal intensity likelihood when one observation mixes several reflections. Companion to `intensity_likelihood_quadrature.md`; implementation in `twin_cf.py` (PyTorch).

## 1. Model

Normalized units throughout. One observation sees $K$ mates with known weights $\alpha_k$ (twin fractions; for powders multiplicity × Lorentz-polarization × profile weight):

$$Z_o = \sum_{k=1}^{K}\alpha_k I_k + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, \sigma_Z^2)$$

Given the model, the true intensities are independent with the Rice (acentric) or Woolfson (centric) prior in intensity form, parameterized by $E_{C,k}$ and $\sigma_A$; write $a = 1 - \sigma_A^2$ and $\lambda_k = \sigma_A^2 E_{C,k}^2$. The likelihood is the density of $Z_o$:

$$L = \int_{\mathbb{R}_+^K}\prod_{k=1}^{K} p(I_k \mid E_{C,k}, \sigma_A)\;\mathcal{N}\!\Big(Z_o \,\Big|\, \sum_k\alpha_k I_k,\ \sigma_Z^2\Big)\, dI_1\cdots dI_K$$

For $K = 1$ this is the untwinned integral. For a powder cluster the scalar $Z_o$ is replaced by the vector of Pawley-extracted intensities $\hat{\mathbf{I}}$ with covariance $C$ (Section 4).

## 2. Characteristic functions

The density of a sum of independent variables is a convolution; in Fourier space it is a product. With $\varphi_X(t) = \mathbb{E}[e^{itX}]$:

$$\varphi_{Z_o}(t) = e^{-\sigma_Z^2t^2/2}\prod_{k=1}^{K}\varphi_{I_k}(\alpha_k t)$$

The normalized acentric intensity given the model is $\tfrac{a}{2}\chi^2_2(2\lambda/a)$ (squared modulus of $\sigma_AE_Ce^{i\phi}$ plus a complex Gaussian of total variance $a$); the centric one is $a\,\chi^2_1(\lambda/a)$. From $\varphi_{\chi^2_\nu(\delta)}(u) = (1-2iu)^{-\nu/2}\exp\big(i\delta u/(1-2iu)\big)$:

$$\varphi_{\text{acen}}(t) = \frac{1}{1 - iat}\exp\!\left(\frac{i\lambda t}{1 - iat}\right)$$

$$\varphi_{\text{cen}}(t) = \frac{1}{\sqrt{1 - 2iat}}\exp\!\left(\frac{i\lambda t}{1 - 2iat}\right)$$

Both have mean $a + \lambda$; variances are $a^2 + 2a\lambda$ (acentric) and $2a^2 + 4a\lambda$ (centric). No Bessel functions appear: $I_0$ and $\cosh$ in the amplitude densities are the result of integrating the phase out, which in Fourier space is already done. Both denominators have real part 1, so the principal branches of $\log$ and $\sqrt{\cdot}$ are the correct ones with no branch tracking.

## 3. Inversion and quadrature

$$L = \frac{1}{2\pi}\int_{-\infty}^{\infty} e^{-itZ_o}\,e^{-\sigma_Z^2t^2/2}\prod_k\varphi_k(\alpha_k t)\,dt$$

Substituting $t = s/\sigma_Z$ turns the noise factor into the Gauss–Hermite weight $e^{-s^2/2}$:

$$L = \frac{1}{2\pi\sigma_Z}\int_{-\infty}^{\infty} e^{-s^2/2}\,\exp\!\Big[-\frac{isZ_o}{\sigma_Z} + \sum_k\log\varphi_k\!\Big(\frac{\alpha_k s}{\sigma_Z}\Big)\Big]\,ds \approx \frac{1}{2\pi\sigma_Z}\sum_j W_j\,\mathrm{Re}\big[\cdots\big]_{s = S_j}$$

with fixed Hermite nodes $S_j$, weights $W_j$. There is no mode search and no per-reflection window. Derivatives with respect to $E_{C,k}$, $\sigma_A$ and $\alpha_k$ are one autograd pass through complex arithmetic.

**Resolution requirement.** The integrand's phase is $e^{-is(Z_o - \mu_T)/\sigma_Z}$ times slowly varying factors, where $\mu_T = \sum_k\alpha_k(a + \lambda_k)$ is the model's expected observation (the CF's own linear phase supplies the $\mu_T$ automatically; explicitly centring the phase changes nothing numerically). The remaining structure of $\prod_k\varphi_k$ lives on the scale $t \sim 1/\sqrt{\mathrm{Var}_{\text{prior}}}$, i.e. $s \sim \sigma_Z/\sqrt{\mathrm{Var}_{\text{prior}}}$ where

$$\mathrm{Var}_{\text{prior}} = \sum_k\alpha_k^2\,\mathrm{Var}(I_k)$$

So the controlling parameter is the noise-to-prior-width ratio $r = \sigma_Z/\sqrt{\mathrm{Var}_{\text{prior}}}$: when the measurement is at least as uncertain as the prior, the integrand is smooth on the Hermite window; when the measurement is much sharper than the prior, the CF varies faster than the weight can resolve and the node count grows like $r^{-2}$. This is the precise meaning of "weak data" for this route, and it is the opposite regime from the one where direct-space adaptive quadrature is hard. The two are complementary along $r$.

## 4. Vector observations with covariance (powder clusters)

For a cluster of $K$ overlapping reflections and $m$ profile points, $\mathbf{y} = \mathbf{b} + W\mathbf{I} + \boldsymbol{\varepsilon}$ with $\boldsymbol{\varepsilon} \sim \mathcal{N}(0, \Sigma)$. Because the map is linear and the noise Gaussian, the dependence on $\mathbf{I}$ passes through the weighted least-squares estimate:

$$\hat{\mathbf{I}} = (W^\top\Sigma^{-1}W)^{-1}W^\top\Sigma^{-1}(\mathbf{y} - \mathbf{b}), \qquad C = (W^\top\Sigma^{-1}W)^{-1}$$

and $\mathcal{N}(\mathbf{y} \mid \mathbf{b} + W\mathbf{I}, \Sigma) = \mathcal{N}(\hat{\mathbf{I}} \mid \mathbf{I}, C) \times (\text{term independent of } \mathbf{I})$. These are the Pawley-extracted intensities and their covariance, so the marginal Rietveld likelihood of the cluster is the twin integral with a covariance:

$$L = \int_{\mathbb{R}_+^K}\prod_k p(I_k \mid E_{C,k}, \sigma_A)\;\mathcal{N}(\hat{\mathbf{I}} \mid \mathbf{I}, C)\,d\mathbf{I} = \frac{1}{(2\pi)^K}\int e^{-i\mathbf{t}^\top\hat{\mathbf{I}}}\,e^{-\mathbf{t}^\top C\mathbf{t}/2}\prod_k\varphi_k(t_k)\,d\mathbf{t}$$

Whitening with $C = LL^\top$, $\mathbf{t} = L^{-\top}\mathbf{s}$, $d\mathbf{t} = \det(C)^{-1/2}d\mathbf{s}$ gives a product Gauss–Hermite rule in $K$ dimensions. The three regimes of $C$: diagonal (separated peaks) reduces to independent single-reflection problems; singular along difference directions (exact overlap) reduces to the twin sum, with $\alpha_k \propto$ multiplicity × Lorentz factor; banded with negative neighbour correlations (partial overlap) is the genuinely new case. In the last case the *sum* direction is typically well determined and the *difference* directions poorly, so after whitening the sum direction has small $r$ — which is exactly where the CF rule needs many nodes. This is why the CF route alone does not finish the powder problem: the precise directions want direct-space treatment (Laplace point plus low-order Hermite in intensity space, or the polar/Legendre rule for pairs) and the imprecise ones want the CF.

## 5. Exact limits used as tests

Equal fractions and all-acentric mates: $\sum_k\tfrac{1}{K}I_k = \tfrac{a}{2K}\chi^2_{2K}(\Lambda)$ with $\Lambda = \sum_k 2\lambda_k/a$, so the prior density of the twinned intensity is a scaled noncentral $\chi^2_{2K}$ in closed form and the problem reduces to the single-domain solver with $I_{K-1}$ in place of $I_0$. With $E_{C,k} = 0$, $K = 2$, $a = 1$ this is $p(T) = 4Te^{-2T}$, the Rees–Yeates twinned Wilson distribution. With $K = 1$ the CF result must agree with the direct quadrature of `mli_quad.py`.

## 6. Numerical results

Reference values: dense brute-force integrals on log-$E$ grids (2-D: $1200^2$–$2500^2$ points; 3-D: $110^3$) or the noncentral-$\chi^2$ closed form with a 400 001-point outer integral. Errors are absolute errors in $\log L$. Draws follow the generative model ($\sigma_A \in \{0.5, 0.7, 0.85, 0.95\}$, Wilson $E_{\text{true}}$, $E_C$ from the Rice conditional, $\sigma_Z$ set by the ratio $r$ defined above).

### 6.1 Single reflection, CF vs direct quadrature ($N = 1500$)

| Hermite nodes | $r \geq 1$ | $0.5 \leq r < 1$ | $0.3 \leq r < 0.5$ | $0.2 \leq r < 0.3$ |
|---|---|---|---|---|
| 32 | $3.9\times10^{-4}$ | fails | fails | fails |
| 64 | $1.8\times10^{-6}$ | $1.0\times10^{-1}$ | $9.5\times10^{-1}$ | fails |
| 96 | $4.5\times10^{-7}$ | $3.2\times10^{-3}$ | $2.7\times10^{-2}$ | fails |
| 128 | $4.5\times10^{-7}$ | $1.5\times10^{-4}$ | $5.0\times10^{-3}$ | $5.8\times10^{-2}$ |

Operating rule: CF with 64 nodes for $r \geq 1$, 128 nodes down to $r \approx 0.5$, direct-space quadrature below. In typical normalized units a reflection with $I/\sigma \approx 10$ has $r \approx 0.1$, so the CF route is for the genuinely weak tail of a data set and for prior-dominated overlap directions, not for the bulk.

### 6.2 Two-mate twins, CF vs brute-force 2-D ($N = 150$, $r \in [0.7, 4]$, one third with a centric mate)

| nodes | max | median | acentric/acentric | acentric/centric |
|---|---|---|---|---|
| 48 | $4.7\times10^{-5}$ | $1.3\times10^{-8}$ | $4.7\times10^{-5}$ | $1.9\times10^{-5}$ |
| 64 | $9.2\times10^{-6}$ | $5.0\times10^{-10}$ | $8.5\times10^{-6}$ | $9.2\times10^{-6}$ |
| 96 | $9.2\times10^{-6}$ | $1.3\times10^{-10}$ | $6.2\times10^{-7}$ | $9.2\times10^{-6}$ |

### 6.3 Perfect $K$-fold twins, CF vs noncentral $\chi^2_{2K}$ closed form ($N = 400$ each, $r \in [0.5, 4]$)

| $K$ | nodes | max | 99th pct | median | $r \geq 1$ | $0.5 \leq r < 1$ |
|---|---|---|---|---|---|---|
| 2 | 64 | $2.8\times10^{-3}$ | $7.3\times10^{-5}$ | $4\times10^{-10}$ | $2.1\times10^{-7}$ | $2.8\times10^{-3}$ |
| 2 | 128 | $9.1\times10^{-6}$ | $3.4\times10^{-7}$ | $8\times10^{-11}$ | $4.8\times10^{-9}$ | $9.1\times10^{-6}$ |
| 3 | 64 | $8.9\times10^{-4}$ | $1.2\times10^{-4}$ | $9\times10^{-16}$ | $1.2\times10^{-9}$ | $8.9\times10^{-4}$ |
| 3 | 128 | $1.4\times10^{-6}$ | $1.2\times10^{-7}$ | $4\times10^{-16}$ | $2.2\times10^{-13}$ | $1.4\times10^{-6}$ |
| 4 | 64 | $6.6\times10^{-4}$ | $2.9\times10^{-5}$ | $6\times10^{-16}$ | $1.3\times10^{-8}$ | $6.6\times10^{-4}$ |
| 4 | 128 | $1.5\times10^{-6}$ | $2.3\times10^{-8}$ | $7\times10^{-16}$ | $2.5\times10^{-14}$ | $1.5\times10^{-6}$ |
| 6 | 64 | $6.8\times10^{-5}$ | $8.9\times10^{-6}$ | $4\times10^{-16}$ | $1.2\times10^{-11}$ | $6.8\times10^{-5}$ |
| 6 | 128 | $1.0\times10^{-8}$ | $4.7\times10^{-10}$ | $4\times10^{-16}$ | $3.3\times10^{-14}$ | $1.0\times10^{-8}$ |

The cost is independent of $K$ except for the $K$ CF evaluations per node; more mates make the sum more Gaussian and the CF integrand smoother, so accuracy improves with $K$.

### 6.4 Clusters with covariance, CF (product Hermite) vs brute-force $K$-D ($N = 60$, $r \geq 1$ per dimension, neighbour correlations $-0.2$ to $-0.7$)

| $K$ | nodes/dim | max | median |
|---|---|---|---|
| 2 | 24 | $1.0\times10^{-2}$ | $6.5\times10^{-6}$ |
| 2 | 32 | $2.7\times10^{-3}$ | $7.4\times10^{-7}$ |
| 2 | 48 | $2.8\times10^{-4}$ | $1.3\times10^{-8}$ |
| 3 | 24 | $3.6\times10^{-2}$ | $1.7\times10^{-5}$ |
| 3 | 32 | $6.9\times10^{-3}$ | $1.2\times10^{-5}$ |

Slower convergence than the scalar case, for the reason given in Section 4: negative correlations make the sum direction sharper than the per-dimension $r$ suggests. Medians are fine; the tails are the well-determined sums, which should be handed to the direct-space rule.

### 6.5 Gradients and cost

Autograd against central differences of the brute-force 2-D reference (twin, $K = 2$, 96 nodes, 30 draws): $\partial\log L/\partial E_{C,1}$ max relative error $1.0\times10^{-7}$, $\partial\log L/\partial\alpha_1$ max $1.5\times10^{-7}$. Two CPU threads, float64, 100 000 observations, 64 nodes: $K = 2$ value 9.4 s, value plus gradient 3.8 s (first call includes warm-up); $K = 4$ value 8.7 s, value plus gradient 6.9 s. The work is $N \times 64 \times K$ complex exponentials, embarrassingly parallel.

## 7. Recommended use

For twins: two domains — the polar/Legendre direct-space rule for strong observations and the CF for $r \gtrsim 0.5$; three or more domains — CF for $r \gtrsim 0.5$, Laplace plus low-order product Hermite in intensity space for strong observations; equal fractions — the noncentral-$\chi^2_{2K}$ closed form as fast path and oracle. For powders: Pawley extraction with covariance, then per cluster whiten $C$, treat eigen-directions with $r \geq 1$ by CF and the sharp ones in direct space; exactly degenerate overlaps are twin sums. Everything shares the priors, $\sigma_A$ handling and Student-$t$ mixture of `mli_quad.py`, and all derivatives come from autograd.

## 8. Regression reference values

Twin likelihood, `log_likelihood_twin_cf` with 128 nodes; references independent of the CF code.

| # | $E_C$ | $\sigma_A$ | $\alpha$ | centric | $Z_o$ | $\sigma_Z$ | log L (reference) | log L (CF) | diff | reference |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [1.0, 1.0] | 0.8 | [0.5, 0.5] | [F, F] | 1.0 | 1.0 | -1.042960464 | -1.042960464 | +3.3e-11 | brute-force 2-D |
| 2 | [1.5, 0.5] | 0.8 | [0.7, 0.3] | [F, F] | 1.2 | 0.8 | -0.999685310 | -0.999685310 | +6.9e-11 | brute-force 2-D |
| 3 | [0.8, 1.2] | 0.9 | [0.6, 0.4] | [F, T] | 0.5 | 1.5 | -1.414319728 | -1.414319167 | +5.6e-07 | brute-force 2-D |
| 4 | [2.0, 0.3] | 0.95 | [0.5, 0.5] | [F, F] | -0.4 | 1.2 | -2.853142835 | -2.853142834 | +1.9e-10 | brute-force 2-D |
| 5 | [1.0, 1.0, 1.0] | 0.7 | [1/3, 1/3, 1/3] | [F, F, F] | 1.0 | 1.0 | -1.027607087 | -1.027607087 | -4.4e-16 | noncentral $\chi^2_6$ |
| 6 | [0.5, 1.5, 1.0, 2.0] | 0.85 | [0.25]×4 | [F]×4 | 1.5 | 1.0 | -1.017721960 | -1.017721960 | -4.4e-16 | noncentral $\chi^2_8$ |

```python
TWIN_REFERENCE = [  # (E_C, sigma_A, alpha, centric, Z_o, sigma_Z, log_L)
    ([1.0, 1.0], 0.8, [0.5, 0.5], [False, False], 1.0, 1.0, -1.042960464),
    ([1.5, 0.5], 0.8, [0.7, 0.3], [False, False], 1.2, 0.8, -0.999685310),
    ([0.8, 1.2], 0.9, [0.6, 0.4], [False, True], 0.5, 1.5, -1.414319728),# Likelihoods for mixed-intensity observations: twinning and powder overlap

Math and numerical tests for the characteristic-function (CF) route to the marginal intensity likelihood when one observation mixes several reflections. Companion to `intensity_likelihood_quadrature.md`; implementation in `twin_cf.py` (PyTorch).

## 1. Model

Normalized units throughout. One observation sees $K$ mates with known weights $\alpha_k$ (twin fractions; for powders multiplicity × Lorentz-polarization × profile weight):

$$Z_o = \sum_{k=1}^{K}\alpha_k I_k + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, \sigma_Z^2)$$

Given the model, the true intensities are independent with the Rice (acentric) or Woolfson (centric) prior in intensity form, parameterized by $E_{C,k}$ and $\sigma_A$; write $a = 1 - \sigma_A^2$ and $\lambda_k = \sigma_A^2 E_{C,k}^2$. The likelihood is the density of $Z_o$:

$$L = \int_{\mathbb{R}_+^K}\prod_{k=1}^{K} p(I_k \mid E_{C,k}, \sigma_A)\;\mathcal{N}\!\Big(Z_o \,\Big|\, \sum_k\alpha_k I_k,\ \sigma_Z^2\Big)\, dI_1\cdots dI_K$$

For $K = 1$ this is the untwinned integral. For a powder cluster the scalar $Z_o$ is replaced by the vector of Pawley-extracted intensities $\hat{\mathbf{I}}$ with covariance $C$ (Section 4).

## 2. Characteristic functions

The density of a sum of independent variables is a convolution; in Fourier space it is a product. With $\varphi_X(t) = \mathbb{E}[e^{itX}]$:

$$\varphi_{Z_o}(t) = e^{-\sigma_Z^2t^2/2}\prod_{k=1}^{K}\varphi_{I_k}(\alpha_k t)$$

The normalized acentric intensity given the model is $\tfrac{a}{2}\chi^2_2(2\lambda/a)$ (squared modulus of $\sigma_AE_Ce^{i\phi}$ plus a complex Gaussian of total variance $a$); the centric one is $a\,\chi^2_1(\lambda/a)$. From $\varphi_{\chi^2_\nu(\delta)}(u) = (1-2iu)^{-\nu/2}\exp\big(i\delta u/(1-2iu)\big)$:

$$\varphi_{\text{acen}}(t) = \frac{1}{1 - iat}\exp\!\left(\frac{i\lambda t}{1 - iat}\right)$$

$$\varphi_{\text{cen}}(t) = \frac{1}{\sqrt{1 - 2iat}}\exp\!\left(\frac{i\lambda t}{1 - 2iat}\right)$$

Both have mean $a + \lambda$; variances are $a^2 + 2a\lambda$ (acentric) and $2a^2 + 4a\lambda$ (centric). No Bessel functions appear: $I_0$ and $\cosh$ in the amplitude densities are the result of integrating the phase out, which in Fourier space is already done. Both denominators have real part 1, so the principal branches of $\log$ and $\sqrt{\cdot}$ are the correct ones with no branch tracking.

## 3. Inversion and quadrature

$$L = \frac{1}{2\pi}\int_{-\infty}^{\infty} e^{-itZ_o}\,e^{-\sigma_Z^2t^2/2}\prod_k\varphi_k(\alpha_k t)\,dt$$

Substituting $t = s/\sigma_Z$ turns the noise factor into the Gauss–Hermite weight $e^{-s^2/2}$:

$$L = \frac{1}{2\pi\sigma_Z}\int_{-\infty}^{\infty} e^{-s^2/2}\,\exp\!\Big[-\frac{isZ_o}{\sigma_Z} + \sum_k\log\varphi_k\!\Big(\frac{\alpha_k s}{\sigma_Z}\Big)\Big]\,ds \approx \frac{1}{2\pi\sigma_Z}\sum_j W_j\,\mathrm{Re}\big[\cdots\big]_{s = S_j}$$

with fixed Hermite nodes $S_j$, weights $W_j$. There is no mode search and no per-reflection window. Derivatives with respect to $E_{C,k}$, $\sigma_A$ and $\alpha_k$ are one autograd pass through complex arithmetic.

**Resolution requirement.** The integrand's phase is $e^{-is(Z_o - \mu_T)/\sigma_Z}$ times slowly varying factors, where $\mu_T = \sum_k\alpha_k(a + \lambda_k)$ is the model's expected observation (the CF's own linear phase supplies the $\mu_T$ automatically; explicitly centring the phase changes nothing numerically). The remaining structure of $\prod_k\varphi_k$ lives on the scale $t \sim 1/\sqrt{\mathrm{Var}_{\text{prior}}}$, i.e. $s \sim \sigma_Z/\sqrt{\mathrm{Var}_{\text{prior}}}$ where

$$\mathrm{Var}_{\text{prior}} = \sum_k\alpha_k^2\,\mathrm{Var}(I_k)$$

So the controlling parameter is the noise-to-prior-width ratio $r = \sigma_Z/\sqrt{\mathrm{Var}_{\text{prior}}}$: when the measurement is at least as uncertain as the prior, the integrand is smooth on the Hermite window; when the measurement is much sharper than the prior, the CF varies faster than the weight can resolve and the node count grows like $r^{-2}$. This is the precise meaning of "weak data" for this route, and it is the opposite regime from the one where direct-space adaptive quadrature is hard. The two are complementary along $r$.

## 4. Vector observations with covariance (powder clusters)

For a cluster of $K$ overlapping reflections and $m$ profile points, $\mathbf{y} = \mathbf{b} + W\mathbf{I} + \boldsymbol{\varepsilon}$ with $\boldsymbol{\varepsilon} \sim \mathcal{N}(0, \Sigma)$. Because the map is linear and the noise Gaussian, the dependence on $\mathbf{I}$ passes through the weighted least-squares estimate:

$$\hat{\mathbf{I}} = (W^\top\Sigma^{-1}W)^{-1}W^\top\Sigma^{-1}(\mathbf{y} - \mathbf{b}), \qquad C = (W^\top\Sigma^{-1}W)^{-1}$$

and $\mathcal{N}(\mathbf{y} \mid \mathbf{b} + W\mathbf{I}, \Sigma) = \mathcal{N}(\hat{\mathbf{I}} \mid \mathbf{I}, C) \times (\text{term independent of } \mathbf{I})$. These are the Pawley-extracted intensities and their covariance, so the marginal Rietveld likelihood of the cluster is the twin integral with a covariance:

$$L = \int_{\mathbb{R}_+^K}\prod_k p(I_k \mid E_{C,k}, \sigma_A)\;\mathcal{N}(\hat{\mathbf{I}} \mid \mathbf{I}, C)\,d\mathbf{I} = \frac{1}{(2\pi)^K}\int e^{-i\mathbf{t}^\top\hat{\mathbf{I}}}\,e^{-\mathbf{t}^\top C\mathbf{t}/2}\prod_k\varphi_k(t_k)\,d\mathbf{t}$$

Whitening with $C = LL^\top$, $\mathbf{t} = L^{-\top}\mathbf{s}$, $d\mathbf{t} = \det(C)^{-1/2}d\mathbf{s}$ gives a product Gauss–Hermite rule in $K$ dimensions. The three regimes of $C$: diagonal (separated peaks) reduces to independent single-reflection problems; singular along difference directions (exact overlap) reduces to the twin sum, with $\alpha_k \propto$ multiplicity × Lorentz factor; banded with negative neighbour correlations (partial overlap) is the genuinely new case. In the last case the *sum* direction is typically well determined and the *difference* directions poorly, so after whitening the sum direction has small $r$ — which is exactly where the CF rule needs many nodes. This is why the CF route alone does not finish the powder problem: the precise directions want direct-space treatment (Laplace point plus low-order Hermite in intensity space, or the polar/Legendre rule for pairs) and the imprecise ones want the CF.

## 5. Exact limits used as tests

Equal fractions and all-acentric mates: $\sum_k\tfrac{1}{K}I_k = \tfrac{a}{2K}\chi^2_{2K}(\Lambda)$ with $\Lambda = \sum_k 2\lambda_k/a$, so the prior density of the twinned intensity is a scaled noncentral $\chi^2_{2K}$ in closed form and the problem reduces to the single-domain solver with $I_{K-1}$ in place of $I_0$. With $E_{C,k} = 0$, $K = 2$, $a = 1$ this is $p(T) = 4Te^{-2T}$, the Rees–Yeates twinned Wilson distribution. With $K = 1$ the CF result must agree with the direct quadrature of `mli_quad.py`.

## 6. Numerical results

Reference values: dense brute-force integrals on log-$E$ grids (2-D: $1200^2$–$2500^2$ points; 3-D: $110^3$) or the noncentral-$\chi^2$ closed form with a 400 001-point outer integral. Errors are absolute errors in $\log L$. Draws follow the generative model ($\sigma_A \in \{0.5, 0.7, 0.85, 0.95\}$, Wilson $E_{\text{true}}$, $E_C$ from the Rice conditional, $\sigma_Z$ set by the ratio $r$ defined above).

### 6.1 Single reflection, CF vs direct quadrature ($N = 1500$)

| Hermite nodes | $r \geq 1$ | $0.5 \leq r < 1$ | $0.3 \leq r < 0.5$ | $0.2 \leq r < 0.3$ |
|---|---|---|---|---|
| 32 | $3.9\times10^{-4}$ | fails | fails | fails |
| 64 | $1.8\times10^{-6}$ | $1.0\times10^{-1}$ | $9.5\times10^{-1}$ | fails |
| 96 | $4.5\times10^{-7}$ | $3.2\times10^{-3}$ | $2.7\times10^{-2}$ | fails |
| 128 | $4.5\times10^{-7}$ | $1.5\times10^{-4}$ | $5.0\times10^{-3}$ | $5.8\times10^{-2}$ |

Operating rule: CF with 64 nodes for $r \geq 1$, 128 nodes down to $r \approx 0.5$, direct-space quadrature below. In typical normalized units a reflection with $I/\sigma \approx 10$ has $r \approx 0.1$, so the CF route is for the genuinely weak tail of a data set and for prior-dominated overlap directions, not for the bulk.

### 6.2 Two-mate twins, CF vs brute-force 2-D ($N = 150$, $r \in [0.7, 4]$, one third with a centric mate)

| nodes | max | median | acentric/acentric | acentric/centric |
|---|---|---|---|---|
| 48 | $4.7\times10^{-5}$ | $1.3\times10^{-8}$ | $4.7\times10^{-5}$ | $1.9\times10^{-5}$ |
| 64 | $9.2\times10^{-6}$ | $5.0\times10^{-10}$ | $8.5\times10^{-6}$ | $9.2\times10^{-6}$ |
| 96 | $9.2\times10^{-6}$ | $1.3\times10^{-10}$ | $6.2\times10^{-7}$ | $9.2\times10^{-6}$ |

### 6.3 Perfect $K$-fold twins, CF vs noncentral $\chi^2_{2K}$ closed form ($N = 400$ each, $r \in [0.5, 4]$)

| $K$ | nodes | max | 99th pct | median | $r \geq 1$ | $0.5 \leq r < 1$ |
|---|---|---|---|---|---|---|
| 2 | 64 | $2.8\times10^{-3}$ | $7.3\times10^{-5}$ | $4\times10^{-10}$ | $2.1\times10^{-7}$ | $2.8\times10^{-3}$ |
| 2 | 128 | $9.1\times10^{-6}$ | $3.4\times10^{-7}$ | $8\times10^{-11}$ | $4.8\times10^{-9}$ | $9.1\times10^{-6}$ |
| 3 | 64 | $8.9\times10^{-4}$ | $1.2\times10^{-4}$ | $9\times10^{-16}$ | $1.2\times10^{-9}$ | $8.9\times10^{-4}$ |
| 3 | 128 | $1.4\times10^{-6}$ | $1.2\times10^{-7}$ | $4\times10^{-16}$ | $2.2\times10^{-13}$ | $1.4\times10^{-6}$ |
| 4 | 64 | $6.6\times10^{-4}$ | $2.9\times10^{-5}$ | $6\times10^{-16}$ | $1.3\times10^{-8}$ | $6.6\times10^{-4}$ |
| 4 | 128 | $1.5\times10^{-6}$ | $2.3\times10^{-8}$ | $7\times10^{-16}$ | $2.5\times10^{-14}$ | $1.5\times10^{-6}$ |
| 6 | 64 | $6.8\times10^{-5}$ | $8.9\times10^{-6}$ | $4\times10^{-16}$ | $1.2\times10^{-11}$ | $6.8\times10^{-5}$ |
| 6 | 128 | $1.0\times10^{-8}$ | $4.7\times10^{-10}$ | $4\times10^{-16}$ | $3.3\times10^{-14}$ | $1.0\times10^{-8}$ |

The cost is independent of $K$ except for the $K$ CF evaluations per node; more mates make the sum more Gaussian and the CF integrand smoother, so accuracy improves with $K$.

### 6.4 Clusters with covariance, CF (product Hermite) vs brute-force $K$-D ($N = 60$, $r \geq 1$ per dimension, neighbour correlations $-0.2$ to $-0.7$)

| $K$ | nodes/dim | max | median |
|---|---|---|---|
| 2 | 24 | $1.0\times10^{-2}$ | $6.5\times10^{-6}$ |
| 2 | 32 | $2.7\times10^{-3}$ | $7.4\times10^{-7}$ |
| 2 | 48 | $2.8\times10^{-4}$ | $1.3\times10^{-8}$ |
| 3 | 24 | $3.6\times10^{-2}$ | $1.7\times10^{-5}$ |
| 3 | 32 | $6.9\times10^{-3}$ | $1.2\times10^{-5}$ |

Slower convergence than the scalar case, for the reason given in Section 4: negative correlations make the sum direction sharper than the per-dimension $r$ suggests. Medians are fine; the tails are the well-determined sums, which should be handed to the direct-space rule.

### 6.5 Gradients and cost

Autograd against central differences of the brute-force 2-D reference (twin, $K = 2$, 96 nodes, 30 draws): $\partial\log L/\partial E_{C,1}$ max relative error $1.0\times10^{-7}$, $\partial\log L/\partial\alpha_1$ max $1.5\times10^{-7}$. Two CPU threads, float64, 100 000 observations, 64 nodes: $K = 2$ value 9.4 s, value plus gradient 3.8 s (first call includes warm-up); $K = 4$ value 8.7 s, value plus gradient 6.9 s. The work is $N \times 64 \times K$ complex exponentials, embarrassingly parallel.

## 7. Recommended use

For twins: two domains — the polar/Legendre direct-space rule for strong observations and the CF for $r \gtrsim 0.5$; three or more domains — CF for $r \gtrsim 0.5$, Laplace plus low-order product Hermite in intensity space for strong observations; equal fractions — the noncentral-$\chi^2_{2K}$ closed form as fast path and oracle. For powders: Pawley extraction with covariance, then per cluster whiten $C$, treat eigen-directions with $r \geq 1$ by CF and the sharp ones in direct space; exactly degenerate overlaps are twin sums. Everything shares the priors, $\sigma_A$ handling and Student-$t$ mixture of `mli_quad.py`, and all derivatives come from autograd.

## 8. Regression reference values

Twin likelihood, `log_likelihood_twin_cf` with 128 nodes; references independent of the CF code.

| # | $E_C$ | $\sigma_A$ | $\alpha$ | centric | $Z_o$ | $\sigma_Z$ | log L (reference) | log L (CF) | diff | reference |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [1.0, 1.0] | 0.8 | [0.5, 0.5] | [F, F] | 1.0 | 1.0 | -1.042960464 | -1.042960464 | +3.3e-11 | brute-force 2-D |
| 2 | [1.5, 0.5] | 0.8 | [0.7, 0.3] | [F, F] | 1.2 | 0.8 | -0.999685310 | -0.999685310 | +6.9e-11 | brute-force 2-D |
| 3 | [0.8, 1.2] | 0.9 | [0.6, 0.4] | [F, T] | 0.5 | 1.5 | -1.414319728 | -1.414319167 | +5.6e-07 | brute-force 2-D |
| 4 | [2.0, 0.3] | 0.95 | [0.5, 0.5] | [F, F] | -0.4 | 1.2 | -2.853142835 | -2.853142834 | +1.9e-10 | brute-force 2-D |
| 5 | [1.0, 1.0, 1.0] | 0.7 | [1/3, 1/3, 1/3] | [F, F, F] | 1.0 | 1.0 | -1.027607087 | -1.027607087 | -4.4e-16 | noncentral $\chi^2_6$ |
| 6 | [0.5, 1.5, 1.0, 2.0] | 0.85 | [0.25]×4 | [F]×4 | 1.5 | 1.0 | -1.017721960 | -1.017721960 | -4.4e-16 | noncentral $\chi^2_8$ |

```python
TWIN_REFERENCE = [  # (E_C, sigma_A, alpha, centric, Z_o, sigma_Z, log_L)
    ([1.0, 1.0], 0.8, [0.5, 0.5], [False, False], 1.0, 1.0, -1.042960464),
    ([1.5, 0.5], 0.8, [0.7, 0.3], [False, False], 1.2, 0.8, -0.999685310),
    ([0.8, 1.2], 0.9, [0.6, 0.4], [False, True], 0.5, 1.5, -1.414319728),
    ([2.0, 0.3], 0.95, [0.5, 0.5], [False, False], -0.4, 1.2, -2.853142835),
    ([1.0, 1.0, 1.0], 0.7, [1/3, 1/3, 1/3], [False]*3, 1.0, 1.0, -1.027607087),
    ([0.5, 1.5, 1.0, 2.0], 0.85, [0.25]*4, [False]*4, 1.5, 1.0, -1.017721960),
]
```

Suggested tolerance `abs=1e-6` (case 3, with a centric mate, is the loosest at $5.6\times10^{-7}$). The Rees–Yeates identity is a further unit test: with $E_C = 0$, $\sigma_A \to 0$, $\alpha = (\tfrac12, \tfrac12)$ and $\sigma_Z = 1$, the CF value must equal $\int 4Te^{-2T}\,\mathcal{N}(Z_o \mid T, 1)\,dT$.

## 9. Files

- `twin_cf.py` — `log_cf_intensity`, `log_likelihood_twin_cf` (scalar observation, any $K$), `log_likelihood_cluster_cf` (covariance, $K \leq 3$), and the reference integrators `reference_twin_2d`, `reference_equal_fraction_ncx2`, `reference_cluster_nd`.
- `mli_quad.py` — the single-reflection direct-space quadrature used for the $K = 1$ cross-check and for the strong-observation branch.
    ([2.0, 0.3], 0.95, [0.5, 0.5], [False, False], -0.4, 1.2, -2.853142835),
    ([1.0, 1.0, 1.0], 0.7, [1/3, 1/3, 1/3], [False]*3, 1.0, 1.0, -1.027607087),
    ([0.5, 1.5, 1.0, 2.0], 0.85, [0.25]*4, [False]*4, 1.5, 1.0, -1.017721960),
]
```

Suggested tolerance `abs=1e-6` (case 3, with a centric mate, is the loosest at $5.6\times10^{-7}$). The Rees–Yeates identity is a further unit test: with $E_C = 0$, $\sigma_A \to 0$, $\alpha = (\tfrac12, \tfrac12)$ and $\sigma_Z = 1$, the CF value must equal $\int 4Te^{-2T}\,\mathcal{N}(Z_o \mid T, 1)\,dT$.

## 9. Files

- `twin_cf.py` — `log_cf_intensity`, `log_likelihood_twin_cf` (scalar observation, any $K$), `log_likelihood_cluster_cf` (covariance, $K \leq 3$), and the reference integrators `reference_twin_2d`, `reference_equal_fraction_ncx2`, `reference_cluster_nd`.
- `mli_quad.py` — the single-reflection direct-space quadrature used for the $K = 1$ cross-check and for the strong-observation branch.
- `twin_cf.py` — `log_cf_intensity`, `log_likelihood_twin_cf` (scalar observation, any $K$), `log_likelihood_cluster_cf` (covariance, $K \leq 3$), and the reference integrators `reference_twin_2d`, `reference_equal_fraction_ncx2`, `reference_cluster_nd`.
- `mli_quad.py` — the single-reflection direct-space quadrature used for the $K = 1$ cross-check and for the strong-observation branch.