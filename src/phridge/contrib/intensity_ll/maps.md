# Map coefficients from the intensity likelihood

`maps.py` turns the `ml_i` posterior into map coefficients and per-reflection
weights, and exposes them to Phenix through the `ml_i_maps` op and
`RemoteIntensityMaps`. This note records the math and the conventions.

## 1. Why there is no $F_o$

An intensity target never forms $|F_o|$. For each reflection the observation
enters only through the posterior of the true normalized amplitude $E$,

$$
p(E \mid Z_o, E_C) \propto p(E \mid E_C, \sigma_A)\, p(Z_o \mid E^2, \sigma_Z[, \nu]),
$$

where $p(E \mid E_C, \sigma_A)$ is Rice (acentric) or Woolfson (centric) and
the noise term is normal or Student-$t$ in intensity. Everything in a map is a
posterior average $\langle\cdot\rangle$ under this density, evaluated on the
same quadrature nodes `mli` uses for $\log L$, so the maps are exactly
consistent with the gradient the refinement sees. French–Wilson is the
special case $E_C = 0$, $\nu = \infty$; $m|F_o|$ is that limit with
$\sigma_Z \to 0$ as well.

## 2. Score by Fisher's identity

The gradient of a log-marginal is the posterior mean of the gradient of the
log-prior. With $a = 1 - \sigma_A^2$ this gives, for acentric reflections,

$$
\frac{\partial \log L}{\partial E_C} = \frac{2\sigma_A}{a}\Bigl(\langle E\, m(E)\rangle - \sigma_A E_C\Bigr)
$$

with the conditional figure of merit

$$
m(E) = \frac{I_1}{I_0}\!\left(\frac{2\sigma_A E_C E}{a}\right),
$$

and for centric reflections

$$
\frac{\partial \log L}{\partial E_C} = \frac{\sigma_A}{a}\Bigl(\langle E\, m(E)\rangle - \sigma_A E_C\Bigr)
$$

with

$$
m(E) = \tanh\!\left(\frac{\sigma_A E_C E}{a}\right).
$$

`posterior_moments` returns this closed form; the test suite checks it against
autograd of the quadrature $\log L$ to $10^{-10}$ (it is an identity on
frozen nodes). The same nodes give $\langle E\rangle$, $\langle E^2\rangle$,
the figure of merit $\langle E m\rangle / \langle E\rangle$ and, for
Student-$t$ noise, the scale-mixture posterior $\langle\lambda\rangle$ and
$\langle\log\lambda\rangle$.

## 3. The four maps (F units)

With $F = \sqrt{\varepsilon\Sigma}\,E$ and $D|F_c| = \sqrt{\varepsilon\Sigma}\,\sigma_A E_C$:

| name | coefficient | reads as |
|---|---|---|
| `difference` | $\sqrt{\varepsilon\Sigma}\,(\langle E m\rangle - \sigma_A E_C)\,e^{i\varphi_c}$ | $mF_o - DF_c$ with the posterior in place of $F_o$ |
| `model` | $\sqrt{\varepsilon\Sigma}\,(2\langle E m\rangle - \sigma_A E_C)\,e^{i\varphi_c}$; centric $\sqrt{\varepsilon\Sigma}\,\langle E m\rangle\,e^{i\varphi_c}$ | $2mF_o - DF_c$ |
| `gradient` | $\partial \log L / \partial F_c^{*}$ = score $/\sqrt{\varepsilon\Sigma}\; e^{i\varphi_c}$ | the Agarwal gradient map |
| `newton` | `gradient` $/\,(\max(\kappa,0) + \mu)$, $\kappa = -\partial^2 \log L/\partial|F_c|^2$ | diagonal (damped) Newton step |

`gradient` equals $-N_\text{work}\,$`d_target_d_f_calc` of the `ml_i` target
(the target is the mean NLL), and $\kappa$ equals $N_\text{work}\,$`curv_radial`;
both are asserted in tests. The gradient map is therefore the
information-weighted difference map,

$$
G_h = \frac{2\sigma_A}{a\sqrt{\varepsilon\Sigma}}\Bigl(\langle E m\rangle - \sigma_A E_C\Bigr)e^{i\varphi_c},
$$

and dividing by the curvature undoes that weighting. The damping
$\mu = $ `newton_damping` $\times$ median positive curvature (default 0.1) is a
Levenberg-style guard: reflections with vanishing or negative curvature would
otherwise be inflated without bound.

## 4. What the intensity posterior changes relative to $mF_o - DF_c$

For a well-measured reflection the posterior is a spike at $\sqrt{Z_o}$ and
`difference` reproduces $(m|F_o| - D|F_c|)e^{i\varphi_c}$ (median relative
deviation $6\times10^{-4}$ at $I/\sigma > 30$ in the tests). When $\sigma_I$ is
large compared to the prior width the posterior collapses onto the prior, the
prior score integrates to zero and the coefficient vanishes; in the Gaussian
limit this is the $2D/(v + \sigma_F^2)$ variance-inflated form. So the map does
not gain high-resolution signal, it loses high-resolution noise, which is why
it looks cleaner and slightly sharper than a conventional $mF_o - DF_c$ built
with $m$ from $\sigma_A$ alone.

For Student-$t$ noise the posterior additionally carries the robust weight

$$
w_h = \langle\lambda \mid Z_o\rangle \approx \frac{\nu + 1}{\nu + r_h^2},
$$

which multiplies the difference coefficient of a discordant reflection toward
zero. This is redescending: a wrongly measured reflection and a genuinely
missing feature both live in the tail. `robust_weight` is returned so the two
can be told apart, and the recommended comparison is the same map at
$\nu \to \infty$ (evaluate with `nu` omitted from the target spec).

## 5. Refining $\nu$

$\nu$ does not touch $F_c$, so it has no map component, but its derivative
follows from the Gamma mixture on the same nodes:

$$
\frac{\partial \log L}{\partial \nu} = \tfrac12\Bigl[\log\tfrac{\nu}{2} + 1 - \psi\!\left(\tfrac{\nu}{2}\right)\Bigr] + \tfrac12\Bigl\langle \log\lambda - \lambda\Bigr\rangle .
$$

`d_loglik_d_nu` (per reflection) and `d_loglik_d_nu_total` are returned; the
total agrees with central finite differences of `log_likelihood_t` to
$10^{-3}$ relative (per-reflection finite differences are noisy across
quadrature-branch switches, the analytic form is not). Refining $\nu$ jointly
with $\sigma_A$ and the model on the working set has a soft degeneracy toward
small $\nu$ and high $\sigma_A$ (heavier tails absorb model error as noise),
which makes maps quieter for the wrong reason; keep $\nu$ global, bounded away
from the Cauchy end, and preferably estimated with $\sigma_A$ on the free set.

## 6. Use from Phenix

```python
from phridge.client import Bridge
from phridge.contrib.intensity_ll.client import RemoteIntensityMaps

maps = RemoteIntensityMaps(
    Bridge("redis://gpu:6379/0"), i_obs, {"name": "ml_i", "nu": 6.0},
    alpha=sigma_a, beta=sigma_wilson, r_free_flags=flags.data(),
    maps={"newton_damping": 0.1, "include_free": True},
)
res = maps(f_calc)
res.difference, res.model, res.gradient, res.newton   # cctbx complex miller arrays
res.fom, res.robust_weight, res.curvature, res.f_post  # flex.double
res.d_loglik_d_nu_total                                 # float or None
```

`f_post` is $\langle F\rangle$, the model-conditioned robust French–Wilson
amplitude, for anyone who needs a column of numbers; it is not used in the
maps. Free reflections are included by default (maps are not a refinement
target); `include_free=False` zeroes them.

## 7. Files and tests

`maps.py` (posterior moments, coefficients), `ops.py` (`ml_i_maps` op, worker
impl), `client.py` (torch-free cctbx wrapper). `mli.py` gained
`quadrature_terms_normal`, which returns the nodes and log-terms behind
`log_likelihood_normal` (bit-identical results), and a window fix for
flat-topped centric integrands (mode at the origin with Laplace width larger
than the physical support; previously up to $10^{-2}$ in $\log L$, now
$2\times10^{-5}$). Tests: `tests/test_intensity_maps.py`.