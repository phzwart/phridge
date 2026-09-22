# The Rice surrogate and interleaved refinement

`surrogate.py` builds a per-reflection surrogate for the exact marginal intensity
likelihood, and `phridge.client.intensity.interleaved` spends the exact target only at
block boundaries. This note records the math, the conventions, and the reasons the
design is the way it is.

## 1. The problem

The exact target integrates a Rice/Woolfson prior on the true normalized amplitude $E$
against the intensity noise model, one adaptive quadrature per reflection per
evaluation (see [`maps.md`](maps.md) and `mli.py`). A refinement macro cycle asks for
that hundreds of times: bulk-solvent scaling, the weight scan, every line-search step
inside LBFGS, simulated annealing. Almost none of those calls need the exact number —
they need a target that agrees with it locally and points the same way.

## 2. EM, and the curvature it gets wrong

Write $t = E_C$ for the normalized model amplitude and let $q$ be the exact posterior
$p(E \mid Z_o, t_0)$ at a checkpoint model $t_0$. The EM / variational bound (the
$Q$-function, equivalently the ELBO) is

$$
Q(t) = \mathbb{E}_q\bigl[\log p(E \mid t)\bigr] \le \log L(t) + \text{const}.
$$

By **Fisher's identity** its gradient at $t_0$ is the *exact* gradient:

$$
Q'(t_0) = \mathbb{E}_q\!\left[\frac{\partial}{\partial t}\log p(E \mid t_0)\right]
        = \frac{\partial}{\partial t}\log L(t_0).
$$

Its curvature is not. **Louis' identity** says

$$
\frac{\partial^2}{\partial t^2}\log L(t_0)
= \underbrace{\mathbb{E}_q\!\left[\frac{\partial^2}{\partial t^2}\log p(E \mid t_0)\right]}_{Q''(t_0)}
+ \underbrace{\operatorname{Var}_q\!\left(\frac{\partial}{\partial t}\log p(E \mid t_0)\right)}_{\text{missing information}} ,
$$

so the EM bound understates the curvature by exactly the missing information, the
posterior variance of the complete-data score. An inner loop run on $Q$ therefore takes
systematically short steps and converges at the EM rate.

We instead fit the Rice family to match **both** the exact score and the exact curvature
at $t_0$. That trades EM's global lower-bound guarantee for Newton-quality inner
convergence, and buys the safety back a different way: every block is adjudicated by an
exact evaluation, and the exact NLL is the sole arbiter of whether the block is kept.

For Student-$t$ noise the complete data is $(E, \lambda)$ and $\log p(\lambda)$ carries
no $t$, so both identities may be taken against the joint posterior on the $(u, E)$
nodes with no extra terms. That is why `exact_score_and_curvature` needs nothing beyond
the nodes the likelihood already integrated on — `posterior_node_cache` hands over the
same frozen nodes and posterior weights that `posterior_moments` consumes.

## 3. The parameterization

The surrogate is $m(t; F_p, \alpha_p, \beta_p)$, the acentric Rice log-likelihood exactly
as `ml_f` consumes it (centrics: the Woolfson / $\tanh$ analog throughout):

$$
m_{\text{acen}} = \log\frac{2F_p}{\beta_p} - \frac{F_p^2 + \alpha_p^2 t^2}{\beta_p}
  + \log I_0\!\left(\frac{2\alpha_p F_p t}{\beta_p}\right),
$$

$$
m_{\text{cen}} = \tfrac12\log\frac{2}{\pi\beta_p} - \frac{F_p^2 + \alpha_p^2 t^2}{2\beta_p}
  + \log\cosh\!\left(\frac{\alpha_p F_p t}{\beta_p}\right).
$$

Note the structure: this is the Rice *prior* with $E$ replaced by the constant $F_p$ and
$1 - \sigma_A^2$ replaced by $\beta_p$.

### $\alpha_p$ is shell-wise and pinned

$\alpha_p(s) = \sigma_A(s)$, never fitted per reflection. Two reasons, both fatal if
ignored:

* **Identifiability.** $\alpha_p$ and $F_p$ enter the Bessel argument only through their
  product, so two derivative conditions cannot determine the pair.
* **Meaning.** $\alpha_p$ *is* the model-quality parameter. A per-reflection $\alpha_p$
  would no longer be $\sigma_A$, and anything downstream that reads it as $\sigma_A$
  would be reading a fitted nuisance number instead.

### $F_p$ and $\beta_p$ are per-reflection

They are solved so that $m'(t_0)$ and $m''(t_0)$ equal the exact score and curvature.
With $r(X) = I_1(X)/I_0(X)$ and $X = 2\alpha_p F_p t/\beta_p$:

$$
m' = -\frac{2\alpha_p^2 t}{\beta_p} + \frac{2\alpha_p F_p}{\beta_p} r(X),
\qquad
m'' = -\frac{2\alpha_p^2}{\beta_p} + \left(\frac{2\alpha_p F_p}{\beta_p}\right)^{\!2} r'(X),
\qquad
r'(X) = 1 - \frac{r}{X} - r^2 .
$$

`_ratio_prime` switches to the series $r' = \tfrac12 - \tfrac{3}{16}X^2 + \tfrac{5}{96}X^4 - \dots$
below $X = 0.3$, where the closed form cancels.

A per-reflection $\beta_p$ is not a liberty. Collapsing it to a shell median degrades the
maximum relative gradient error over $t \in [0.75, 1.25]\,t_0$ by a factor of **4–5** on a
mixed shell — measured, and guarded by
`test_per_reflection_beta_is_necessary`. With per-reflection $(F_p, \beta_p)$ that error
has p90 below 0.08–0.10 on a shell including a substantial tail of negative intensities.
Per-reflection $\beta_p$ is also the standard variance-inflation convention, so it is
interface-legal: cctbx target functors take `alpha` / `beta` as per-reflection flex
arrays.

### Units

With $S = \sqrt{\epsilon \Sigma_W}$ so that $F = S E$, the conversion to the units `ml_f`
consumes is

$$
F_p^{(F)} = S\,F_p, \qquad \beta_p^{(F)} = \Sigma_W \beta_p, \qquad \alpha_p = \sigma_A .
$$

Substituting into the `ml_f` form (whose residual scale is $e_b = \epsilon \beta$)
reproduces the normalized expressions term for term, so
$\partial/\partial|F_c| = S^{-1}\,\partial/\partial t$ follows automatically. The check
that this is right is in `test_surrogate_fit_op_registers_and_round_trips`: the fitted
arrays pushed through the stock `ml_f` target reproduce the exact `ml_i`
`d_target_d_f_calc` and `curv_radial` to better than $10^{-6}$ relative.

## 4. Initialization and the fallback ladder

Per reflection, in order:

1. **Closed-form EM / Jensen values** $F_p = \langle E \rangle_{\text{post}}$,
   $\beta_p = 1 - \sigma_A^2$. Free from what the checkpoint already computed, and a
   genuine lower bound with the exact gradient.
2. **Newton on $(F_p, \log \beta_p)$** from that initialization — damped, with
   backtracking, vectorized over the batch (no Python loop over reflections) and
   compressed onto the still-unconverged subset each iteration. Without that compression
   a handful of hard reflections keeps the whole batch iterating and the fit costs an
   order of magnitude more than the exact evaluation it replaces.
3. **On fit failure, keep the initialization** and set `FIT_FALLBACK_INIT`.
4. **If the initialization is unusable too**, set `FIT_EXACT_ROUTE`; the controller
   routes those reflections through the exact target inside the inner loop
   (`IntensityFModel._splice_exact_route`).

The Rice family is *not* globally concave in $t$: at $X = 0$ the curvature is
$2\alpha_p^2(F_p^2/\beta_p - 1)/\beta_p$, positive whenever $F_p^2 > \beta_p$. So the
positive exact curvature that weak and negative reflections show near $t = 0$ is
matchable, and the fit does not refuse it up front — reachability is decided by the
residual after Newton. Those reflections live on the far side of the family and are
unreachable from the EM start, so they get additional vectorized starts including one
placed analytically on the $X = 1$ convex branch. In practice ~99% of a mixed shell
converges; the remainder falls to step 3, and the per-shell counts are in the telemetry.

## 5. The interleaved loop

Per macro cycle, with `refinement.target_mode=interleaved`:

1. **Checkpoint (exact).** One exact evaluation gives $\mathrm{NLL}_0$, the scores, the
   curvatures and the posterior moments; the surrogate is fitted here. Every statistic
   and every nuisance refit ($\sigma_A(s)$, $\nu$) happens here and only here, from the
   exact posterior. `update_all_scales` drops the surrogate when it refits the nuisance
   arrays, so a stale $\alpha_p$ can never be used.
2. **Inner block (surrogate).** The macro cycle's inner machinery runs on the stock
   `ml_f` path fed with $(F_p, \alpha_p(s), \beta_p)$. No quadrature is evaluated.
3. **Adjudication (exact).** One exact evaluation gives $\mathrm{NLL}_1$. Accept if
   $\mathrm{NLL}_1 \le \mathrm{NLL}_0 + \texttt{tol}$ (default $\texttt{tol} = 0$, so no
   uphill step is ever accepted). On rejection: refresh the surrogate at the midpoint
   model and re-run the block with the iteration count and step scale halved. A second
   rejection falls back to running the macro cycle fully exact.

**No block is ever accepted on the surrogate's own word, and no statistic is ever
computed from a surrogate value.** The final macro cycle always runs fully exact, so
deposited and published statistics never touch the surrogate.

Measured on the synthetic end-to-end test: 168 exact evaluations become 43 (a factor of
3.9), the final exact NLL agrees to six decimals, and the coordinates agree to 0.0015 Å.
The saving grows with block length, because it is the ratio of inner evaluations to
block boundaries.

## 6. Student-$t$ caveat

With Student-$t$ noise the robust down-weighting is frozen into $(F_p, \beta_p)$ at the
checkpoint. A reflection that becomes an outlier partway through a block is over-trusted
until the next refresh, because Rice tails are lighter than the $t$-marginal's. The
accept/reject is the containment. **A rising rejection rate is the signal to shorten
blocks, not to widen the tolerance** — `InterleavedTelemetry.report` says so out loud
above 50%.

## 7. Asymmetric refresh (v2, off by default)

`refresh="visible_fraction"` refits only reflections whose visible fraction $\rho^2$ is
below `rho2_threshold` or which have moved more than `delta_ec_threshold` in $E_C$ since
the last refresh. The theory note: the EM contraction rate per reflection is the
missing-information fraction $1 - \rho^2$, so low-$\rho^2$ reflections are exactly the
ones whose surrogates go stale fastest. v1 ships with `refresh="full"` because a full
refresh costs one exact evaluation that the checkpoint is already paying for.

## 8. Hard rules

* **`F_p` and `beta_p` are internal arrays.** They must never appear in any printed
  report, output file, or user-visible label: `F_p` is one paste away from being
  mistaken for an observed amplitude and `beta_p` for an experimental variance. Same
  discipline as the S-family naming (see [`rint.py`](rint.py)), and guarded by
  `test_internal_surrogate_arrays_are_never_user_visible`.
* **No change to the exact target, quadrature, gradients, or any computation on the
  exact path.** The one structural edit is the mechanical extraction of
  `posterior_node_cache` out of `posterior_moments`, which returns identical numbers.
* **The exact NLL is the sole arbiter.**
