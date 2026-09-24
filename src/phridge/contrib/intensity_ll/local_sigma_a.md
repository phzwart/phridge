# Localized $\sigma_A$ from the inverse solvent mask

A design note. Nothing here is implemented yet. The current nuisance fit
(`ml_i_nuisance_fit`, [`ops.py`](ops.py), [`free_beta.py`](free_beta.py),
[`aniso_rice.py`](aniso_rice.py)) is a *stationary* error model: $\sigma_A$ and
$\beta$ vary with resolution and direction, not with position in the cell. This
note is how to add a real-space field $\log\sigma_A(x)$ whose Fourier
coefficients are tied to the inverse of the bulk-solvent mask, and how that
field re-enters the existing Rice / intensity likelihood without breaking
identifiability.

Related notes: [`maps.md`](maps.md) (posterior coefficients, Wilson / free
$\beta$ conventions), [`omit_windows.py`](omit_windows.py) (hard spatial
windows), [`bulk_solvent_op.py`](bulk_solvent_op.py) (NLL $k_{\mathrm{mask}}$).

## 1. Why a spatial field, and why not a new $\sigma_A(h)$ column

The Rice first moment in normalized units is exact,

$$
\mathbb{E}[Z_o \mid E_C] = \sigma_A^2 E_C^2 + \beta,
$$

and today $\sigma_A$ and $\beta$ are per-shell scalars, optionally modulated by
Laue-class tensors $M_A$, $M_\beta$ ([`aniso_rice.py`](aniso_rice.py)). That is
equivalent to: the error field has the same statistics everywhere in the cell,
and a length scale (hence a resolution falloff). It cannot say *where* the
model is wrong.

A real-space field $\alpha(x)$ does. Write

$$
\rho_{\mathrm{true}}(x) = \alpha(x)\,\rho_{\mathrm{model}}(x) + \eta(x),
\qquad \mathrm{Var}(\eta(x)) = \beta(x).
$$

Multiplication in $x$ is convolution in $h$:

$$
F_{\mathrm{true}}(h) = (\hat\alpha * F_{\mathrm{model}})(h) + \varepsilon(h).
$$

If $\eta$ is spatially white, $\mathrm{Cov}(\varepsilon(h),\varepsilon(h')) =
\hat\beta(h-h')$ and the *diagonal* is only $\int\beta(x)\,dx$ — independent of
$h$. A localized white variance is invisible to a per-reflection $\beta$.
Localization of $\sigma_A$ is useful because it changes the mean model the
likelihood sees (the next section); localization of $\beta$ is not, unless
off-diagonal covariances are modelled. This note therefore puts the spatial
field on $\sigma_A$ and leaves $\beta$ as the existing shells.

Do **not** implement localization as another per-Miller $\sigma_A(h)$. That is
still a function of $s$, not of $x$.

## 2. What $F_{\mathrm{eff}}$ is

Today the intensity target has one complex Miller array in the $F_c$ slot. That
array is $F_{\mathrm{model}}$, the ordinary calculated structure factor
(atoms + bulk solvent, already scaled). The Rice model uses it only as

$$
E_C(h) = \frac{|F_{\mathrm{model}}(h)|}{\sqrt{\varepsilon_h\Sigma_W(h)}},
\qquad
\mathbb{E}[Z_o\mid E_C] = \sigma_A^2 E_C^2 + \beta.
$$

$F_{\mathrm{eff}}$ is the array that occupies that same $F_c$ slot once a
real-space field $\alpha(x)$ is in play. It is not a new kind of observation
and not a map coefficient. It is **the model structure factor after local
trust has been multiplied onto the density**:

$$
F_{\mathrm{eff}}(h)
\;=\;
\int_V \alpha(x)\,\rho_{\mathrm{model}}(x)\,e^{2\pi i h\cdot x}\,dx
\;=\;
\bigl(\hat\alpha * F_{\mathrm{model}}\bigr)(h).
$$

$\alpha(x)$ is the spatial part of $\sigma_A$, stripped of the shell mean
($\alpha=\exp\varphi$, $\varphi$ defined in §5.1). Regions the model trusts
have $\alpha\approx 1$ and contribute to $F_{\mathrm{eff}}$ as they do to
$F_{\mathrm{model}}$. Regions it does not trust have $\alpha< 1$ and are
down-weighted *before* the Fourier transform. The likelihood then does exactly
what it does now, with $F_{\mathrm{model}}$ replaced by $F_{\mathrm{eff}}$:

$$
E_C(h) = \frac{|F_{\mathrm{eff}}(h)|}{\sqrt{\varepsilon_h\Sigma_W(h)}}.
$$

Two things stay what they were:

- $F_{\mathrm{model}}$ is still the structural model. Coordinates, ADPs and
  $k_{\mathrm{mask}}$ still generate $F_{\mathrm{calc}}$ and $F_{\mathrm{mask}}$
  the usual way. $F_{\mathrm{eff}}$ is derived from that, then thrown away
  after the NLL / maps / gradients have used it.
- The shell $\sigma_A(s)$ is still the mean correlation. It is not baked into
  $\alpha$. If it were, $F_{\mathrm{eff}}$ and $\sigma_A$ would both scale
  $E_C$ and the slope would be counted twice.

When $\alpha\equiv 1$, $F_{\mathrm{eff}}=F_{\mathrm{model}}$ and nothing has
changed.

**v1 does not build this integral on a grid.** $\alpha(x)$ tied to the inverse
mask is two-valued (molecule vs solvent), and the model is already split into
those supports, so the integral collapses to a two-term mix, then a per-shell
rms gauge that stops $u$ looking like a scale:

$$
F_{\mathrm{mix}}(h) = w_{\mathrm{mol}}\,F_{\mathrm{atoms}}(h)
                    + w_{\mathrm{sol}}\,F_{\mathrm{bulk}}(h),
\qquad
F_{\mathrm{eff}}(h) = F_{\mathrm{mix}}(h)\,/\,n_{k(h)}.
$$

That is the same $F_{\mathrm{eff}}$ as the integral, evaluated for a
piecewise-constant $\alpha$. v2 (a $\psi(x)$ that varies inside the molecule)
is the integral for real, because that $\alpha$ is no longer constant on
$F_{\mathrm{atoms}}$.

$F_{\mathrm{mix}}$ is an intermediate. $F_{\mathrm{eff}}$ is what is passed as
`f_calc` into `normalized`.

## 3. The inverse mask is the generating object

cctbx / mmtbx convention in this repo: the bulk-solvent mask is $m(x)=1$ in
solvent, $0$ on the molecule. Structure factors already split as

$$
F_{\mathrm{model}} = k_{\mathrm{model}}\bigl(F_{\mathrm{calc}} + k_{\mathrm{mask}}(s)\,F_{\mathrm{mask}}\bigr).
$$

Call the pieces (both already on the observation scale once $k_{\mathrm{model}}$
is applied)

$$
F_{\mathrm{atoms}} = k_{\mathrm{model}}\,F_{\mathrm{calc}},
\qquad
F_{\mathrm{bulk}} = k_{\mathrm{model}}\,k_{\mathrm{mask}}(s)\,F_{\mathrm{mask}},
$$

so $F_{\mathrm{model}} = F_{\mathrm{atoms}} + F_{\mathrm{bulk}}$. The molecular
envelope is the inverse mask

$$
e(x) = 1 - m(x).
$$

Its Fourier coefficients are already in hand: $F_e(0) = V_{\mathrm{mol}}$ and

$$
F_e(h) = -F_{\mathrm{mask}}(h)\qquad(h\neq 0).
$$

$e(x)$ is space-group invariant, already an envelope, and already low-resolution
in character (solvent radius + shrink). A free low-res Fourier series for
$\log\sigma_A(x)$ would spend most of its coefficients rediscovering $e$. Tie
them instead.

Atomic density already lives on $e$. Multiplying $\rho_{\mathrm{calc}}$ by
$e(x)$ is a near no-op. The mask tie is useful because it gives the two
channels *different* correlations, not because it re-masks $F_{\mathrm{calc}}$.
That is a partition of scattering, and it stays diagonal in $h$:

$$
F_{\mathrm{mix}}(h) = w_{\mathrm{mol}}\,F_{\mathrm{atoms}}(h) + w_{\mathrm{sol}}\,F_{\mathrm{bulk}}(h).
$$

No density-grid FFT is required for the leading term. Omit windows
([`omit_windows.py`](omit_windows.py)) are the discrete, hard-mask limit of the
same idea on the atomic side only.

## 4. How locality survives a sum over reflections

The refinement target is still $\sum_h -\log p(Z_o\mid F_{\mathrm{eff}}(h))$,
independent $h$, with $F_{\mathrm{eff}}$ as in §2. Locality is not stored in a
per-reflection $\sigma_A(h)$. It is stored in **how $F_{\mathrm{eff}}$ is built
from the density**. If that construction is a real-space multiply and then an
FFT, the chain rule carries $\alpha(x)$ onto every coordinate gradient. If it
is collapsed to a diagonal $\sigma_A(h)$ and $F_c$ is left as
$F_{\mathrm{model}}$, the *where* is gone.

### 4.1 What is kept: the Jacobian, not the $\sigma_A$ column

$$
F_{\mathrm{eff}} = \mathrm{FFT}\bigl(\alpha(x)\,\rho_{\mathrm{model}}(x)\bigr)
\qquad\Longleftrightarrow\qquad
F_{\mathrm{eff}}(h) = (\hat\alpha * F_{\mathrm{model}})(h).
$$

The likelihood never sees $\alpha(x)$. It sees $|F_{\mathrm{eff}}(h)|$. After
that the Rice model is the usual one. Locality is in the *forward map from
atoms to $F_{\mathrm{eff}}$*:

$$
\frac{\partial L}{\partial x_i}
= \sum_h \frac{\partial L}{\partial F_{\mathrm{eff}}(h)}
         \frac{\partial F_{\mathrm{eff}}(h)}{\partial x_i},
\qquad
\frac{\partial F_{\mathrm{eff}}}{\partial x_i}
= \mathrm{FFT}\bigl(\alpha(x)\,\partial_i\rho_i(x)\bigr).
$$

Atom $i$ sitting in a low-$\alpha$ region is down-weighted in every reflection
and its coordinate / B / occupancy gradient shrinks by that same $\alpha$. The
sum over $h$ does not smear $\alpha$ into a resolution curve; it smears the
*weighted atom* in the usual Fourier way. That is ordinary crystallography.
The extra fact is that the weight is a function of $x$, applied *before* the
FFT.

Two-channel v1 is the same statement with a two-valued $\alpha$:

$$
\frac{\partial F_{\mathrm{eff}}}{\partial x_i}
= w_{\mathrm{mol}}\,\frac{\partial F_{\mathrm{atoms}}}{\partial x_i}
\quad\text{(atoms)},
\qquad
\frac{\partial F_{\mathrm{eff}}}{\partial k_{\mathrm{mask}}}
= w_{\mathrm{sol}}\,\frac{\partial F_{\mathrm{bulk}}}{\partial k_{\mathrm{mask}}}.
$$

All atoms share one $w_{\mathrm{mol}}$. Intra-molecular “this loop is worse
than that domain” is **not** present in v1. The only locality v1 keeps is
molecule vs solvent — the scale of the inverse mask. That is why $\psi$ (v2)
has to be a product FFT on $\rho_{\mathrm{atoms}}$, not another pair of
channel weights.

### 4.2 What is thrown away: residual covariance

Even with a correct $F_{\mathrm{eff}}$, independent Rice says the leftover
errors $\varepsilon(h)$ are independent. A missing blob at $x_0$ correlates
many $h$ ($\mathrm{Cov}(\varepsilon(h),\varepsilon(h'))=\hat\beta(h-h')$). The
diagonal target

- does **not** know that those reflections share one real-space error,
- therefore over-counts the information in the data (errors too tight),
- and still Fourier-smears the residual gradient of that blob onto every atom
  in the usual way.

So: **which atoms count** is localized (through $\alpha$ in the Jacobian).
**Which reflections share the same leftover error** is not. Recovering the
second thing requires either a non-diagonal covariance or a windowed /
real-space likelihood (omit windows are the discrete version of that). v1 and
v2 do not try.

### 4.3 The way that *does* lose locality

Form $\varphi(x)$, take its power spectrum or $|c_h|$, and write a new
$\sigma_A(s)$ or $\sigma_A(h)$ while leaving $F_c = F_{\mathrm{model}}$. The
likelihood is then the current one with a slightly different resolution
curve. Nothing in the Jacobian depends on $x$. That is the construction this
note is written to forbid.

A useful test of any implementation: move one isolated atom into a region
where $\alpha$ is small. Its $|F|$ contribution and its refinement gradient
must drop. If only the shell $\sigma_A$ moved, locality was lost.

## 5. Parameterization

Keep the shell curve as the mean correlation. Put only *contrast* in the map.

### 5.1 Log field

$$
\varphi(x) = \log\frac{\sigma_A(x)}{\sigma_A^{\mathrm{shell}}(s)}
         = u\,\bigl(e_{\mathrm{lp}}(x) - \bar e\bigr).
$$

$e_{\mathrm{lp}}$ is $e$ low-passed to a cutoff $d_{\min}^{\varphi}$ (default
$15$–$20\,\text{Å}$, or the native mask shrink if that is already as soft).
$\bar e$ is the cell mean of $e_{\mathrm{lp}}$ (the molecular volume fraction
of the low-passed envelope). $u$ is a single unconstrained scalar. $u>0$ means
the atomic model is more trusted than the solvent model.

Optional second number: a B on the envelope coefficients, so the field is
slightly more blurred than the solvent mask,

$$
c_0 = 0,
\qquad
c_h = u\,e^{-B s_h^2/4}\,(-F_{\mathrm{mask}}(h))
\quad(h\neq 0,\; d_h \ge d_{\min}^{\varphi}).
$$

$\varphi = \mathrm{IDFT}(c)$ is then exactly the map in §5.1 with a
resolution-dependent blur. Default $B=0$ (use the mask as is, only low-pass by
dropping $h$).

Origin $c_0$ stays zero. The shells own the mean $\sigma_A$. This is the same
gauge as refusing `fit_scale` when $\beta$ is free ([`free_beta.py`](free_beta.py)).

### 5.2 Channel weights

On a sharp (or softly interpolated) two-region field the voxel multiplier is
$g(x)=\exp\varphi(x)$, hence the two values

$$
w_{\mathrm{mol}} = \exp\bigl(u(1-\bar e)\bigr),
\qquad
w_{\mathrm{sol}} = \exp\bigl(u(0-\bar e)\bigr).
$$

At $u=0$, $w_{\mathrm{mol}}=w_{\mathrm{sol}}=1$ and $F_{\mathrm{mix}}=
F_{\mathrm{model}}$: current behaviour, bit-identical.

Jensen: mean-centering $\varphi$ does **not** mean-center $g$. $u$ therefore
leaks a global scale into $|F_{\mathrm{mix}}|$, which is degenerate with the
shell $\sigma_A$. Kill that leak by a scattering-weighted (or per-shell rms)
renormalization, not by another free $k$:

$$
n_k = \frac{\mathrm{rms}_{h\in k}|F_{\mathrm{mix}}|}{\mathrm{rms}_{h\in k}|F_{\mathrm{model}}|},
\qquad
F_{\mathrm{eff}}(h) = F_{\mathrm{mix}}(h)\,/\,n_{k(h)}.
$$

$u$ then changes the *direction* in the $(F_{\mathrm{atoms}},F_{\mathrm{bulk}})$
plane and the per-reflection |F| pattern (solvent cancellation moves), but not
the shell-average model power. $\sigma_A^{\mathrm{shell}}$ remains the slope of
$Z_o$ on $E_C^2$ with $E_C$ from $F_{\mathrm{eff}}$.

A single global rms ratio is acceptable as a v1 simplification; per-shell $n_k$
is the one that cannot tilt the Wilson plot and should be the default.

### 5.3 What the Rice model sees

$F_{\mathrm{eff}}$ is defined in §2. Replace $F_c$ with that array *before*
`IntensityLogLikelihood.normalized`.
The shells, free $\beta$, tensors, $\nu$, quadrature, maps, omit bookkeeping and
surrogate do not change. They already consume whatever $F_c$ they are handed
([`target.py`](target.py) `normalized`, [`maps.py`](maps.py)
`posterior_moments`).

$$
E_C = |F_{\mathrm{eff}}|/\sqrt{\varepsilon\Sigma_W},
\qquad
\mathbb{E}[Z_o\mid E_C] = \sigma_A^{\mathrm{shell}\,2}\,E_C^2 + \beta.
$$

Phase of $F_{\mathrm{eff}}$ is the model phase for the maps. When
$w_{\mathrm{mol}}\neq w_{\mathrm{sol}}$, $F_{\mathrm{atoms}}$ and
$F_{\mathrm{bulk}}$ are not parallel and $\varphi_{\mathrm{eff}}$ rotates
relative to $\varphi_{\mathrm{model}}$, mostly at low resolution. That is
intended.

When $\beta$ is free the existing `rice_inputs` transform still applies to
$(E_C,\sigma_A^{\mathrm{shell}},\beta)$. No new reparameterization.

## 6. Optional residual inside the molecule (v2)

If the envelope mode is not enough, gate extra low-res contrast by the same
inverse mask:

$$
\varphi(x) = u\bigl(e_{\mathrm{lp}}(x)-\bar e\bigr) + e_{\mathrm{lp}}(x)\,\psi(x).
$$

$\psi$ is a mean-zero Fourier series on a coarse unique-under-SG Miller set
(the $F_{\mathrm{mask}}$ list restricted to $d\ge d_{\min}^{\psi}$, or an even
coarser box $|h|,|k|,|l|\le 2$), with reality $c_{-h}=\overline{c_h}$ and
centric phase restrictions. Solvent stays pinned; leftover coefficients can
only talk about local model quality where there are atoms.

This residual *does* need a product FFT,

$$
F_{\psi} = \mathrm{FFT}\bigl(e_{\mathrm{lp}}\,\psi\cdot\rho_{\mathrm{atoms}}\bigr),
$$

and then $F_{\mathrm{mix}} = w_{\mathrm{mol}}F_{\mathrm{atoms}} +
w_{\mathrm{sol}}F_{\mathrm{bulk}} + F_{\psi}$ with the same rms gauge. Do not
put free $\psi$ on the solvent side: that region is owned by $k_{\mathrm{mask}}$
and the low-resolution $\beta$ shells.

v1 is $u$ (and optional $B$) only. Do not implement $\psi$ until $u$ is shown
to move on real data and not to steal from $k_{\mathrm{mask}}$.

## 7. Identifiability

The same class of degeneracies as stage 2 of `ml_i_nuisance_fit`. Refuse
rather than report an arbitrary split.

| Pair | Why they fight | Rule |
|---|---|---|
| $u$ and $k_{\mathrm{mask}}$ | both reweight $F_{\mathrm{bulk}}$ vs $F_{\mathrm{atoms}}$ | freeze $k_{\mathrm{mask}}$ before fitting $u$ (already true: NLL $k_{\mathrm{mask}}$ runs first, then nuisance) |
| $c_0$ and $\sigma_A^{\mathrm{shell}}$ | mean of $\varphi$ is a global scale | $c_0=0$ always |
| $n_k$ omitted and $\sigma_A^{\mathrm{shell}}$ | $\exp\varphi$ has a Jensen scale | per-shell rms gauge, §5.2 |
| $u$ and `fit_scale` | same slope $\sigma_A^2 k^2$ | `fit_scale` already off when $\beta$ is free; keep it off |
| $B$ and low-res $\sigma_A$ shells | both are a low-$s$ envelope | default $B=0$; if $B$ is free, freeze the lowest one or two $\sigma_A$ shells or put a tight prior on $B$ |
| $\psi$ and omit / local error | both eat intra-molecular disagreement | v2 only, and not in the same macrocycle as omit windows |
| $u$ and $\beta$ at low $s$ | a worse solvent mix can be read as extra variance | $\beta$ free is the *reason* this is safe: the intercept can absorb what $u$ does not, instead of laundering it into $\sigma_A$ |

$\Sigma_W$ remains the only data-anisotropy carrier ([`maps.md`](maps.md) §8).
$M_A$ stays a function of $\hat s$, not of $x$. Cubic crystals already have
$M_A=I$; $u$ is still identified there because it is spatial, not directional.

No mask $\Rightarrow$ no field. If there is no single solvent mask (the
bulk-solvent NLL op already refuses `len(f_masks)!=1`), skip $u$ and record
`local_sigma_a_fallback`, same pattern as `wilson_fallback` /
`sigma_a_tensor_fallback`.

## 8. Staging

Insert as **stage 2b** of the existing nuisance sequence, after $k_{\mathrm{mask}}$
and $\Sigma_W$ are held, jointly with or immediately after the shell
$\sigma_A/\beta$ fit. Do not open a third free scale.

Current order in [`engine.py`](../../client/intensity/engine.py)
`update_all_scales`:

1. mmtbx bulk-solvent + scale (anisotropic $k$ constrained off when $\Sigma_W$
   will carry anisotropy).
2. `ml_i_bulk_solvent_fit`: binned $k_{\mathrm{mask}}$, $\sigma_A/\beta$
   *profiled and discarded* ([`bulk_solvent_op.py`](bulk_solvent_op.py)).
3. `ml_i_nuisance_fit` stage 1: $\Sigma_W$ from intensities, no $F_c$.
4. `ml_i_nuisance_fit` stage 2: shell $\sigma_A$, $\beta$, optional $\nu$,
   optional $M_A,M_\beta$, on the tune set (free flags when there are enough).

Proposed:

4. Stage 2 as now, but $F_c$ is $F_{\mathrm{eff}}(u)$ instead of $F_{\mathrm{model}}$.
   $u$ (and optional $B$) sit in the same LBFGS as the shell logits.
5. Maps, surrogate, omit windows consume the same $F_{\mathrm{eff}}$ through
   `normalized`.

$u$ is estimated on the tune set, like $\sigma_A$. On the work set the model
has already been fitted to those reflections; a spatial field would otherwise
claim the overfitting as “the molecule is better than the solvent.”

Initialize $u=0$ (and $B=0$). The start is the current model, so the fit can
only improve the profiled NLL. A weak $u^2$ prior (default $\lambda_u = 1$)
keeps the field from running to a hard mask on thin data; it is regularization
and is chosen on the tune set, never on the audit set — same rule as
`smooth_sigma_a`.

## 9. The inspectable map

The object to look at is $\varphi(x)$, not a full-resolution $\sigma_A$ voxel
map and not a sigma-scaled density.

Coefficients (v1):

$$
c_0 = 0,
\qquad
c_h = u\,e^{-B s_h^2/4}\,(-F_{\mathrm{mask}}(h))
\quad(d_h \ge d_{\min}^{\varphi}).
$$

Inverse FFT on a coarse grid (the mask grid, or `resolution_factor` against
$d_{\min}^{\varphi}$, not against the data $d_{\min}$). Write CCP4 **without**
`apply_sigma_scaling`: $\varphi$ is already in log-correlation units. Labels
`phridge log_sigma_a` / `phridge inv_mask`.

Optional companion map: $\varphi(x)+\log\sigma_A^{\mathrm{ref}}$ at a nominated
shell (e.g. the shell nearest $d=3\,\text{Å}$) so the numbers read as
$\log\sigma_A$ rather than as contrast about the shell mean. The coefficients
of that map differ only at the origin.

Report in `sigma_a_params`:

```
local_sigma_a: {
  enabled, u, b_blur, d_min, e_bar,
  w_mol, w_sol, n_shell_rms[],
  n_envelope_hkl, lambda_u,
  fallback  // absent, or why it was skipped
}
```

The coefficient array itself can ride along as a packed Miller array on the
same crystal as $F_{\mathrm{mask}}$, unique under the space group, for anyone
who wants to re-FFT later.

## 10. What this is not

- **Not a second $k_{\mathrm{mask}}$.** The mask still builds $F_{\mathrm{model}}$.
  $\varphi$ only says where that model is believed. $k_{\mathrm{mask}}$ stays
  frozen in stage 2b.
- **Not a voxel-wise $\sigma_A(h)$.** Localization is a function of $x$, and it
  enters refinement only through $F_{\mathrm{eff}}=\mathrm{FFT}(\alpha\rho)$,
  not through a new $\sigma_A$ column (§2, §4).
- **Not a real-space $\beta(x)$ on the Rice diagonal.** A white $\beta(x)$ is
  invisible there (§1). A two-region $\beta_{\mathrm{mol}}/\beta_{\mathrm{sol}}$
  is a separate question and is not required for v1.
- **Not a recompute of the solvent mask per evaluation.** $F_{\mathrm{mask}}$
  is the one already on the fmodel, same caveat as omit windows: scales and
  mask of the full model are held.
- **Not omit windows.** Omit windows punch a hole in $F_{\mathrm{atoms}}$ and
  inflate residual $\beta$ by $\sigma_A^2\Sigma^{\mathrm{omit}}$. This field
  reweights the two channels of the *full* model. They compose: omit still
  subtracts $k_{\mathrm{scale}}\Delta F_w$ from $F_{\mathrm{eff}}$, and the
  residual-$\beta$ increment is unchanged.

## 11. Implementation sketch

v1 needs no new LinkML kind. Inputs are existing packed types (`MillerArray`,
`array`, `json`). Follow [`AGENTS.md`](../AGENTS.md): pydantic options,
`extra="forbid"`, typed public API, registration + accept/reject + numeric
smoke tests.

### 11.1 Options

```python
class LocalSigmaAOptions(BaseModel):
    model_config = {"extra": "forbid"}

    enabled: bool = False
    d_min: float = 15.0          # Å; envelope cutoff
    blur_b: float = 0.0          # optional extra B on c_h
    lambda_u: float = 1.0        # u^2 prior, tune-set only
    rms_gauge: Literal["shell", "global"] = "shell"
    residual: bool = False       # v2; refuse unless explicitly on
```

Env knob `PHRIDGE_LOCAL_SIGMA_A=1` to turn it on from the client, matching
`PHRIDGE_OMIT_WINDOWS` / `PHRIDGE_BULK_SOLVENT_NLL`.

### 11.2 Worker pieces (new module `local_sigma_a.py`)

Torch-free helpers that cctbx clients can import:

- `envelope_coefficients(f_mask, s_sq, d_min, blur_b) -> (hkl, c_h)`
  drops $h=0$ and $d < d_{\min}$, sets $c_h = -F_{\mathrm{mask}}(h)\,
  e^{-B s^2/4}$.
- `channel_weights(u, e_bar) -> (w_mol, w_sol)`.
- `mix_f_eff(f_atoms, f_bulk, w_mol, w_sol, shell_id, gauge) -> (f_eff, n_k)`
  the two-channel mix plus the rms gauge. At $u=0$ this is the identity on
  `f_atoms + f_bulk`.
- `phi_coefficients(f_mask, u, ...) -> Miller-like (hkl, c_h)` for the
  inspectable map.
- `describe_local_sigma_a(...)` JSON fragment for `sigma_a_params`.

`e_bar` from the mask volume fraction if the real-space mask is available
(`n_solvent / n_grid`); otherwise from $F_{\mathrm{mask}}(0)/V$ if the origin
term is present, else a stored metadata value. Do not estimate $\bar e$ from
high-resolution $F_{\mathrm{mask}}(h)$.

### 11.3 Nuisance op

`ml_i_nuisance_fit` gains optional inputs `f_atoms`, `f_bulk` (or `f_mask` +
`k_mask` + `k_model` and form the pieces on the worker) and `local_sigma_a`
JSON. Today's call site passes `f_calc=f_model` only; when the local field is
off, keep that path and do not require the split.

When enabled:

1. Validate a single mask; else set fallback and run as now.
2. Include `u` (logit-unconstrained) in the stage-2 LBFGS. Each NLL evaluation
   builds $F_{\mathrm{eff}}(u)$ and hands $|F_{\mathrm{eff}}|$ to the existing
   `normalize` / quadrature. Penalty $\lambda_u u^2$.
3. Return `f_eff` (optional, same hkl as `f_obs`) and the JSON in §9.

Do not change `_INPUTS` kinds: `f_atoms` / `f_bulk` are `MillerArray` or
`array` on the same hkl list as `f_obs`.

### 11.4 Client

In `update_all_scales`, after the current $F_{\mathrm{model}}$ checkpoint:

- pack `F_{\mathrm{atoms}}$ and $F_{\mathrm{bulk}}$ from `f_calc`, `f_mask`,
  `k_mask`, `k_model` / `k1` (the same split `_fit_bulk_solvent_nll` already
  has).
- pass `local_sigma_a=LocalSigmaAOptions(enabled=...)`.
- after the fit, if coefficients came back, FFT and write
  `{prefix}_log_sigma_a.ccp4` next to the other maps in
  [`client/intensity/maps.py`](../../client/intensity/maps.py). No sigma
  scaling.

The refinement target keeps using $F_{\mathrm{model}}$ as the *structural*
model (coordinates still generate $F_{\mathrm{calc}}$). Only the likelihood's
$E_C$ uses $F_{\mathrm{eff}}$. That is the same split we already make between
“the fmodel” and “the $F_c$ the nuisance fit / maps see” (today those happen
to be equal).

### 11.5 Maps and omit

`ml_i_maps` already takes `f_calc`. Pass $F_{\mathrm{eff}}$ when the field is
on, so posterior coefficients, FOM and the gradient map are consistent with
the NLL. Omit windows subtract $\Delta F_w$ from that same $F_{\mathrm{eff}}$
and keep the current residual-$\beta$ increment.

## 12. Tests

Minimum set, next to `tests/contrib/test_free_beta.py` /
`tests/contrib/test_bulk_solvent_nll.py`.

1. **Identity.** $u=0$ $\Rightarrow$ $F_{\mathrm{eff}}=F_{\mathrm{model}}$
   (bit-identical) and NLL unchanged.
2. **Envelope coefficients.** $c_h = -F_{\mathrm{mask}}(h)$ for $h\neq 0$
   inside the cutoff; $c_0=0$; hermiticity.
3. **Gauge.** Per-shell $\mathrm{rms}|F_{\mathrm{eff}}| =
   \mathrm{rms}|F_{\mathrm{model}}|$ to $10^{-12}$.
4. **Pydantic.** accept/reject on `LocalSigmaAOptions` (`extra` forbid,
   `d_min>0`, `residual=True` without v2 support raises).
5. **Fallback.** no mask / several masks $\Rightarrow$ field off,
   `local_sigma_a_fallback` set, existing $\sigma_A$ path unchanged.
6. **Identifiability.** enabled field + `fit_scale=True` + `beta_mode=free`
   raises with the same class of message as today's scale/$\beta$ guard.
7. **Synthetic direction.** Build $I$ from $D_{\mathrm{mol}}F_{\mathrm{atoms}}
   + D_{\mathrm{sol}}F_{\mathrm{bulk}}$ with $D_{\mathrm{mol}}>D_{\mathrm{sol}}$,
   fit $u$; recovered $u>0$. Reverse the $D$'s; recovered $u<0$. Shell
   $\sigma_A$ stays near the true mean correlation.
8. **Tune-set only.** Fitting $u$ on work flags after a coordinate fit must
   not be the advertised path; a smoke that the client passes the free mask
   when it is large enough (mirrors the existing $\sigma_A$ tune rule).
9. **Jacobian keeps $x$.** Shrink $\alpha$ on one isolated atom (v2) or drop
   $w_{\mathrm{mol}}$ (v1). That atom's $\partial L/\partial x_i$ must fall;
   the shell $\sigma_A$ must not be the thing that moved. If it did, locality
   was collapsed to a resolution curve (§4.3).

v2 adds: $\psi=0$ reduces to v1; a compact intra-molecular dip in synthetic
$\alpha(x)$ is recovered in $\psi$ and not in $u$.

## 13. Defaults and knobs

| knob | default | notes |
|---|---|---|
| `enabled` / `PHRIDGE_LOCAL_SIGMA_A` | off | current runs reproduce exactly |
| `d_min` | $15\,\text{Å}$ | truly low resolution; raise to $20$ on small cells if $\|\{h\}\|$ is tiny |
| `blur_b` | $0$ | only if the raw mask is jagged |
| `lambda_u` | $1$ | tune-set regularization |
| `rms_gauge` | `shell` | `global` only as a debug comparison |
| `residual` | off | v2 |

On a typical protein at $15\,\text{Å}$ the unique envelope list is a few dozen
reflections, often fewer under a non-P1 group. $u$ is one number. That is the
point.

## 14. Suggested build order

1. `LocalSigmaAOptions` + `envelope_coefficients` + `mix_f_eff` + identity /
   gauge / coefficient tests. No op changes.
2. Thread $F_{\mathrm{atoms}}$, $F_{\mathrm{bulk}}$ and $u$ through
   `ml_i_nuisance_fit` stage 2; synthetic direction test.
3. Client pack/unpack, env knob, `sigma_a_params` JSON, CCP4 write of
   $\varphi(x)$.
4. Point `ml_i_maps` at $F_{\mathrm{eff}}$ when the field is on; confirm
   gradient-map vs autograd still holds (it must: same `normalized`).
5. Only then consider $\psi$ (v2) or a two-region $\beta$.

Until step 2 lands, the inspectable map can already be written from
$-u F_{\mathrm{mask}}$ at $u$ hand-set, as a visual check that the inverse
mask at $d_{\min}^{\varphi}$ is the object we think it is.
