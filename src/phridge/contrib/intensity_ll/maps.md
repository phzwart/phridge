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
## 8. Windowed omit coefficients (`ml_i_omit_windows`)

Optional post-scale feature (`PHRIDGE_OMIT_WINDOWS=1` / `--omit-windows`): for each
real-space window covering the ASU, compute reciprocal-space map coefficients
as if that window's atoms were absent, then **stitch** those maps in real space
(keep each window's density only inside its box / Voronoi cell) and FFT back to
one set of composite Fourier coefficients.

### Structure factors

$$
\Delta F_{h,w} = F_c^{\mathrm{atoms}\in w}
$$

via an occupancy-masked FFT (same engine as refinement). Then

$$
F^{\mathrm{omit,model}}_{h,w} = F_{\mathrm{model},h} - k_h\,\Delta F_{h,w}
$$

with solvent and scales of the **full** refined model held fixed (no per-window
mask recompute).

### Anisotropic Wilson normalization: exactly one anisotropy carrier is free

$\Sigma_W$ is fitted as a tensor, $\Sigma_0\exp(-\tfrac12\,\mathbf{s}^{\mathsf T} B\,\mathbf{s})$
with $\mathbf{s}=O^{-\mathsf T}\mathbf{h}$ the Cartesian reciprocal vector — **this is the
default** (`PHRIDGE_WILSON_MODEL=anisotropic`), because real data are anisotropic and a
scalar normalization is a modelling error rather than a conservative choice. The isotropic
form $\Sigma_0\exp(-\tfrac12 B_W s^2)$ remains available on request. See
[`wilson.py`](wilson.py) for the parameterization.

The one automatic exception is identifiability: six components need reflection directions
that span the sphere, so a set confined to a line, a plane or a narrow cone falls back to
the scalar and records why in `sigma_wilson_params["wilson_fallback"]`, rather than
reporting a tensor whose unconstrained directions are arbitrary.

**Design decision.** An anisotropic $\Sigma_W$ and `f_model`'s `k_anisotropic` are
*degenerate*: both absorb the same directional falloff, so fitting both yields a good
fit and two individually meaningless tensors. Exactly one may be free, and it must be
the normalization, because $\Sigma_W$ is the **prior** and three downstream quantities
are only interpretable if the prior describes the observations' directional falloff:

1. $\sigma_Z=\sigma_I/(\varepsilon\Sigma_W)$ and hence the reported `data%`, otherwise
   too high in weak directions and too low in strong ones;
2. the prior variance $\beta=\Sigma_W(1-\sigma_A^2)$, mis-priced the same way, which
   biases the per-reflection posterior weights and so the map coefficients
   *directionally*;
3. the per-shell $\sigma_A(s)$, a scalar, which otherwise absorbs the residual as
   "model error".

With the prior carrying it, `k_anisotropic` has nothing legitimate left to carry, so it
is constrained to isotropic and the constraint is logged. Concretely, both
`bulk_solvent_and_scale.anisotropic_scaling` and
`bulk_solvent_and_scale.minimization_b_cart` are set to `False` — clearing only the first
leaves the `b_cart` minimizer free to fit the tensor anyway. Because the default is now
anisotropic, this happens on every run. The guard measures the anisotropy *actually
applied* to $F_{\mathrm{calc}}$ after scaling and raises if both carriers are live — a
flag records the intent, not the outcome, and across mmtbx versions setting one is
best-effort.

**Centric normalization is already correct and needs no factor.** The Wilson mean is
$\varepsilon\Sigma$ for both parities; only the variance differs (1 acentric, 2
centric). The centric factor therefore belongs in the likelihood's centric branch and
*not* in $Z=I/(\varepsilon\Sigma)$. Verified numerically: the stage-1 NLL minimizes at
the true $\Sigma$ for pure-acentric and pure-centric sets alike, and the fitted tensor
does not move when centrics are excluded (`test_centrics_do_not_drive_the_tensor`). Had
the factor leaked into the normalization, a tensor fit would have absorbed the
acentric/centric imbalance into $B$.

**Gauge.** $\Sigma_0$ and $\mathrm{tr}\,B$ are partially degenerate, so reporting fixes
the convention: the eigen-decomposition, with $B_{\mathrm{iso}}=\mathrm{tr}B/3$ and the
comparable anisotropy $\Delta B=\lambda_{\max}-\lambda_{\min}$ separated. Raw $B$
components are basis-dependent and are not the primary output.

### Free $\beta$: the intercept is a parameter, not a constraint

In normalized units the Rice first moment is exact:

$$
E[Z_o \mid E_C] = \sigma_A^2 E_C^2 + \beta
$$

so within a shell $\sigma_A^2$ is the **slope** and $\beta$ the **intercept**. The classical
$\beta = 1-\sigma_A^2$ forces that line through $(1,1)$, which is exactly the assertion that
the Wilson normalization is correct. When it is not — residual anisotropy, an inadequate
solvent model, tNCS — the constrained fit can only reach the data by tilting the slope, and
the normalization error is **laundered into $\sigma_A$**. Measured on synthetic data with
$\beta$ inflated 1.5x at a true $\sigma_A=0.85$: the constrained fit returns $0.80$ and
reports $\sigma_A^2+\beta = 1.000$ by construction, so it cannot even signal the problem;
the free fit returns $0.856$ and reports $1.09$, which is the diagnostic
(`tests/contrib/test_free_beta.py`). Where the normalization is right the two agree, so
freeing $\beta$ costs nothing.

Both $Z_o$ and $E_C^2$ are divided by the same $\Sigma_W$, so the **slope is invariant** to
an error in $\Sigma_0$ and only the intercept scales. That is why $\sigma_A$ is recoverable
at all, and why $\sigma_A^2+\beta$ is the column to read.

**No monotonicity.** $\sigma_A$ is commonly depressed at low resolution where the solvent
model is poor and can dip mid-range (ice rings, detector artifacts). A cumulative-drop
parameterization can only fall, so it pushes such structure into neighbouring shells. It is
replaced by a **second-difference penalty on the logits**, which charges curvature and
leaves a straight trend free — a claim that the profile is smooth, which is defensible,
rather than that it never rises, which is not. The strengths are regularization and are
chosen on the tune set, never on the audit set.

**Exactness.** Freeing $\beta$ required **no change to the quadrature, target or
gradients**. The integrands use $(E_C,\sigma_A)$ only through the product $\sigma_A E_C$ and
through the prior variance $a = 1-\sigma_A^2$, so the substitution

$$
\sigma_A^{\text{eff}} = \sqrt{1-\beta}, \qquad
E_C^{\text{eff}} = \sigma_A E_C/\sqrt{1-\beta}
$$

gives $a=\beta$ while preserving the product, hence a Rice prior with
$E[E^2]=\sigma_A^2E_C^2+\beta$. Verified to machine precision for both parities. See
[`free_beta.py`](free_beta.py).

**Identifiability.** With $\beta$ free, $\sigma_A$ and an overall $F_c$ scale $k$ are exactly
degenerate — only $\sigma_A^2k^2$ enters the slope — so stage 2 refuses to refine a joint
scale rather than report one of infinitely many equally good splits.

**Where $\beta$ lands.** The fitted $\beta$ is not a diagnostic; it is the prior variance, so
it reaches the refinement target, the gradients, the map coefficients, the omit maps and the
Rice surrogate. It travels as `obs.beta_residual` (normalized units — for `ml_i` the `beta`
slot already holds $\Sigma_W$) and every consumer reads it through
`IntensityLogLikelihood.normalized`, which is the single place the reparameterization is
applied. An absent array means the classical $1-\sigma_A^2$, bit-for-bit.

Two chain rules matter. Autograd differentiates the reparameterization itself, so the
**target** gradient needs nothing. A **closed-form** score does:
`posterior_moments().score` is $\partial\log L/\partial E_C$ in whatever parameterization it
was handed, so the maps multiply it by $\mathrm{d}E_C^{\text{eff}}/\mathrm{d}E_C =
\sigma_A/\sqrt{1-\beta}$, and the surrogate — which must live in the true $t=E_C$ because
`amplitude_units` maps it to `ml_f` through $F=SE$ — scales its score by that factor and its
curvature by its square (the factor is independent of $E_C$, so there is no extra term).
Omitting it biases every map gradient by ~6% on the measured case: small, silent, and
checked against finite differences of the exact likelihood in
`test_map_gradient_matches_finite_differences_under_a_free_beta`.

Measured effect of the correctly-plumbed $\beta$ on the laundering case: NLL improves 0.025
nats/reflection, mean $|\partial\log L/\partial|F_c||$ moves 28%, and the mean figure of
merit **falls** from 0.758 to 0.700 — the right direction, because the true residual
variance exceeds $1-\sigma_A^2$ and the constrained prior was overconfident about phases.

### Residual-variance adjustment

Maps use Wilson $\Sigma$ in the `beta` slot and $\sigma_A$ in `alpha`. Residual
$\beta^{\mathrm{res}}=\Sigma(1-\sigma_A^2)$. For window $w$:

$$
\beta^{\mathrm{res}}_{h,w} = \Sigma_h\,(1-\sigma_{A,h}^2) + \sigma_{A,h}^2\,\Sigma^{\mathrm{omit}}_{h,w}
$$

$$
\sigma_{A,h,w} = \sigma_{A,h}\sqrt{\max\bigl(0,\, 1 - \Sigma^{\mathrm{omit}}_{h,w}/\Sigma_h\bigr)}
$$

$\Sigma^{\mathrm{omit}}$ is the B-weighted atomic scattering power of the
window (calibrated by a global $\kappa$ so omitting all atoms approaches the
Wilson prior). Student-$t$ $\nu$ is unchanged across windows.

### Artifact

Default: single MTZ ``{prefix}_omit_windows.mtz`` — Fourier coefficients of the
**stitched** composite omit map, with columns:

- ``FWT`` / ``PHWT`` — omit-model 2mFo−DFc analog
- ``DELFWT`` / ``PHDELWT`` — omit-model mFo−DFc analog

plus ``{prefix}_omit_windows.window_meta.json``. Open the MTZ in Coot/PyMOL like
any map-coefficient file.

Optional: ``PHRIDGE_OMIT_SAVE_NPZ=1`` keeps a diagnostic per-window ``.npz``
(reader: `phridge.contrib.intensity_ll.omit_windows.load_omit_windows`).
``PHRIDGE_OMIT_RESOLUTION_FACTOR`` (default ``0.25``) sets the FFT grid for
stitching.

### Limitation

Atoms were refined in the presence of the omitted region (phase-memory bias).
Do not treat density at the omit site as fully unbiased.

## 9. The S family: S_post and S_prior (reporting only)

Any R factor evaluated at a posterior point estimate (mode or mean) shrinks toward
$\sigma_A E_C$ as noise grows, so the number *improves* when the data get worse.
The integrated residual avoids that pathology:

$$
k_S = \frac{\sum_S \langle E\rangle E_C}{\sum_S E_C^2},\qquad
S_{\mathrm{post}} = \frac{\sum_S \langle |E - k_S E_C|\rangle}{\sum_S \langle E\rangle}
$$

Expectations use the same quadrature nodes as map coefficients / NLL
(`phridge.contrib.intensity_ll.rint`). Work and free each get their own $k_S$;
resolution shells **reuse the parent set's $k_S$**. $k_S$ is least squares on the
posterior means and converges to $\sigma_A$ in the no-data limit.

$S_{\mathrm{prior}}$ is the same pair under the prior only (`prior_only=True`),
implemented by evaluating the identical node machinery with $\sigma_Z = 10^6$
(flat likelihood) — the $\sigma_A$-implied no-data floor. Report the two as a
pair; the gap may be negative when $\sigma_A$ is conservative.

JSON / accessors: `s_post_work`, `s_post_free`, `s_post_all`, `s_prior_work`,
`s_prior_free`, `s_prior_all` (never labeled `r_work` / `r_free`). Direct
intensity R remains `r_intensity_*`.

### 9.0 Notation: why S, and never a bare S

These are **not** the crystallographic R factor. The functional is an expectation
under a probability distribution, and point-estimate variants of it are biased low
by posterior shrinkage. Named with the letter R they get pasted into R columns and
compared against deposited values no matter what the caption says — the field
already paid for that lesson with Rmerge / Rmeas / Rpim, and NMR deliberately
picked "Q-factor" for its R-like statistic. The defining sentence is: *the same
residual functional, evaluated under the posterior versus under the prior.*

The rules, in full:

1. `S_post` — expected residual under the per-reflection posterior. Work/free
   variants `S_post(work)` / `S_post(free)`; fields `s_post_work` / `s_post_free`.
2. `S_prior` — the same functional under the prior only, the $\sigma_A$-implied
   no-data floor. Fields `s_prior_work` / `s_prior_free`.
3. **S never appears bare.** Every user-visible occurrence carries its subscript,
   because bare capital S is the small-molecule goodness-of-fit
   (`_refine_ls_goodness_of_fit`) and lowercase $s$ is $\sin\theta/\lambda$, which
   appears throughout this package as $\sigma_A(s)$.
4. `s_vis` — the visible-numerator (plug-in) diagnostic, evaluating the residual at
   the posterior mean instead of integrating over it. **Internal**: never printed in
   the default report, kept on the stats object for debugging.
5. `rho2`, `xi`, `omega` keep their names — already non-R letters.
6. The only statistic still labeled "R" is the legacy French–Wilson amplitude R
   (plus the direct-intensity R, which is a genuine point-estimate R on intensities
   with no shrinkage). The FW R is the deliberate legacy bridge and fills the
   mandated `R VALUE` / `FREE R VALUE` fields in REMARK 3. Posterior mean and mode
   agreement statistics are demoted to a verbose block labeled
   `shrunken-amplitude agreement (biased low — diagnostic, not an R factor)`.

For readers coming from conventional refinement:
**`S_post(free)` plays the role conventionally played by R_free, with the
shrinkage pathology removed.**

Deprecated aliases (`s_post_work`, `s_prior_work`, `k_S_work`, …) survive as
read-only properties on `IntensityStatsReport` and emit a `DeprecationWarning`
naming the replacement. They will be removed in the next minor version; the
printed report and the worker payload use S names exclusively.

Neither S statistic is the legacy R functional, and neither must be compared
row-for-row against deposited R values.

### 9.1 Visible / latent split ($\rho^2$)

$S_{\mathrm{post}}$ is an $L_1$ quantity and admits no variance decomposition. Its
$L_2$ companion does, exactly:

$$
\sum_S \langle (E - k E_C)^2 \rangle
= \underbrace{\sum_S (\langle E\rangle - k E_C)^2}_{v_{\mathrm{vis}}}
+ \underbrace{\sum_S \operatorname{Var}(E)}_{v_{\mathrm{lat}}},
\qquad
\rho^2 = \frac{v_{\mathrm{vis}}}{v_{\mathrm{vis}} + v_{\mathrm{lat}}}
$$

$v_{\mathrm{vis}}$ is the part of the residual the posterior mean actually commits
to; $v_{\mathrm{lat}}$ is what stays inside the posterior. The reports print
$1-\rho^2$, the latent share, which rises monotonically as the data weaken.

**$\rho^2$ is descriptive, not a calibrated test.** In the weak-data limit the
posterior mean still carries the Rice Jacobian lift, so $v_{\mathrm{vis}}$ does not
go to zero even when the observations constrain nothing. Read $1-\rho^2$ as "how
much of the disagreement the model declines to commit to", never as a p-value.

JSON: `v_vis_{work,free,all}`, `v_lat_{work,free,all}`, `rho2_{work,free,all}`, plus
`rho2_work_bins` / `rho2_free_bins` per shell.

### 9.2 Score test ($\xi$, $\omega$) — opt-in

Set `PHRIDGE_SCORE_TEST=1` to enable. It is off by default because the reference
variance costs a dozen extra posterior solves per reflection.

The numerator is the difference-map coefficient itself,
$\Delta_h = \langle E m\rangle_{\mathrm{post}} - \sigma_A E_C$ — the same expression
`intensity_map_coefficients` uses, evaluated on the same nodes. The reference is its
variance under the predictive distribution of $Z_o$ at the claimed $\sigma_A$:

$$
\xi_S = \frac{\sum_S |\Delta_h|^2}{\sum_S \mathbb{E}_{\mathrm{model}}\!\left[|\Delta_h|^2\right]},
\qquad \omega = \frac{\xi_{\mathrm{free}}}{\xi_{\mathrm{work}}}
$$

**The reference variance must be computed by quadrature, never by Monte Carlo.**
It is evaluated by Gauss-Hermite integration on a rule moment-matched to the exact
predictive mean and variance of $Z_o$ (with $a = 1-\sigma_A^2$, $\langle I\rangle =
\sigma_A^2 E_C^2 + a$ and $\operatorname{Var}(I) = a^2 + 2a\sigma_A^2 E_C^2$ for
acentrics, twice that for centrics, plus $\sigma_Z^2$ of noise). A sampled reference
variance is inflated by Jensen's inequality; that inflation cancels in the ratio
$\omega$ but not in $\xi_{\mathrm{work}}$ or $\xi_{\mathrm{free}}$ separately, which
would leave both individually uninterpretable. Twelve nodes suffice — the value is
converged to better than $10^{-3}$ against a 32-node rule.

Reading the numbers:

- $\xi_{\mathrm{free}} \approx 1$ means the model explains the free data as well as
  it claims to. $\xi_{\mathrm{free}} > 1$ is model error.
- $1 - \xi_{\mathrm{work}}$ estimates the effective number of fitted parameters per
  work reflection.
- **$\omega$'s null value is not 1.** For $p$ effective parameters fitted on
  $N_{\mathrm{work}}$ reflections it is
  $(1 + p/N_{\mathrm{work}})/(1 - p/N_{\mathrm{work}})$. Compare $\omega$ against
  that, not against unity, or ordinary fitting will read as model error.

Per shell, $\xi_{\mathrm{free}}$ and $\omega$ are blanked below
`min_free_per_shell` (default 30) free reflections rather than reported as noise;
the counts stay visible so the blanks are self-explanatory.

JSON: `xi_{work,free,all}`, `omega`, and `xi_work_bins` / `xi_free_bins` /
`omega_bins`.

### 9.3 Work / free

Free-set values are the honest ones; work-set values are optimistically biased by
fitting. $R_{\mathrm{int,free}} - R_{\mathrm{int,work}}$ is twice the optimism, so
their midpoint estimates the true error budget — it is invariant under fitting even
when the two halves separate, which is what the perturbation test in
`tests/contrib/test_rint.py` checks.
