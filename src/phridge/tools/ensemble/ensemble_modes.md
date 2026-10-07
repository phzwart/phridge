# Ensemble perturbation modes

Datasets \(n = 1..N\), Friedel half-set reflections \(h\), modes \(k = 1..K\).

\[
F_n(h) = F_{\mathrm{var},n}(h) + D_n(s)\bigl[F_\mu(h) + \Delta_0(h) + \sum_k z_{nk}\Delta_k(h)\bigr] + \varepsilon_n(h)
\]

\(\varepsilon_n \sim \mathrm{CN}(0,\sigma^2_{\Delta,n}(s))\), \(z_{nk}\) real, \(\Delta_k(-h) = \overline{\Delta_k(h)}\).

\(F_\mu\) is the reference protein \(F_{\mathrm{calc}}\) on the common ASU set from `multixtal.client.prepare_job`. \(F_{\mathrm{var},n}\) is the per-dataset ligand or water \(F_{\mathrm{calc}}\) plus the bulk-solvent term. With no per-dataset model it is the solvent term only.

\[
D = \sigma_A\sqrt{\Sigma_W/\Sigma_{\mathrm{calc}}},\qquad
\sigma^2_\Delta = (1-\sigma_A^2)\,\varepsilon\,\Sigma_W.
\]

The M-step replaces the shell values from the latent moments and, when `dof_correct` is set, multiplies \(\sigma^2\) by \(N/(N-K-1)\) once per sweep. That factor is not the maximum-likelihood update. The monotone EM test therefore runs with `dof_correct=False`.

Missing reflections have weight 0. Audit reflections (role 2) never enter a parameter. Calibration reflections (role 1) enter only the \(D\) and \(\sigma^2\) update.

## E-step

The amplitude integral is `mli`'s quadrature. In the model frame \(\arg\mu = 0\),

\[
\kappa_{\mathrm{ac}}(r) = \frac{2 r |\mu|}{\sigma^2_\Delta},\qquad
\kappa_{\mathrm{cen}}(r) = \frac{r |\mu|}{\sigma^2_\Delta}.
\]

Acentric conditional phase, with optional HL vector \(A+iB\):

\[
v = \kappa_{\mathrm{ac}} + (A+iB),\qquad
\langle e^{i\phi}\mid r\rangle = e^{i\arg v}\frac{I_1(|v|)}{I_0(|v|)}.
\]

The amplitude nodes are reweighted by \(I_0(|v|)/I_0(\kappa)\). `hl is None` does not take this branch, so `E_mean`, `I_mean` and `log_lik` stay on the original path. `log_lik_hl` is the reweighted normaliser and is not comparable across `hl_weight`.

Centric reflections are a two-point distribution on the centric axis (phases \(0\) and \(\pi\) after the axis is rotated onto the real line):

\[
w_+ \propto \exp(\kappa_{\mathrm{cen}}+A),\qquad
w_- \propto \exp(-\kappa_{\mathrm{cen}}-A),\qquad
\langle e^{i\phi}\rangle = \tanh(\kappa_{\mathrm{cen}}+A).
\]

\(A\) here is the HL vector projected on the centric axis. \(F = \sqrt{\varepsilon\Sigma_W}\,E\).

The Fisher identity used by the direct fit is

\[
\frac{\partial L}{\partial \mathrm{Re}\,\mu} + i\frac{\partial L}{\partial \mathrm{Im}\,\mu}
  = \frac{2\bigl(\mathbb{E}[F]-\mu\bigr)}{\sigma^2_\Delta}.
\]

The finite-difference check freezes the amplitude nodes. `posterior_moments` already treats its window as fixed; differentiating the moving window is not the identity being checked.

## M-step and gauge

Design \(W = [1\mid z]\in\mathbb{R}^{N\times(K+1)}\), weights \(w_{nh} = \mathrm{mask}_{nh} D_n^2/\sigma^2_{\Delta,n}(h)\). Each reflection solves a real \((K+1)\) system with a complex right-hand side via `torch.linalg.solve`.

Scores:

\[
\mathrm{Re}[\overline{\Delta}\,\Omega_n\Delta^\mathsf{T}]\,z_n
  = \mathrm{Re}[\overline{\Delta}\,\Omega_n(R_n-\Delta_0)].
\]

The covariance is the inverse of that real Gram matrix. \(\sum_n z_{nk} = 0\); \(\Delta_0\) carries the mean.

Gauge: thin SVD of the real embedding \([\mathrm{Re}\,\Delta\mid\mathrm{Im}\,\Delta]\) of \(\sum z_k\Delta_k\). Modes are orthonormal under \(\mathrm{Re}\sum_h \overline{u}v\). Score columns are uncorrelated with descending variances. Gauge does not change the likelihood.

## Projections

Support, F(000) removal, the Gaussian-weight band limit, and the positivity shift are projections (idempotent). Multiplying coefficients by a Gaussian is not idempotent, so `project_smooth` zeros reflections with \(\exp(-2\pi^2\sigma^2 s^2) < 1/2\). Positivity lifts \(\rho + \sum z_k u_k\) on the box of observed scores. Projected and unprojected log-likelihoods are different objectives.

## Campaign used by the always-on tests

Orthogonal cell \((30, 30, 40, 90, 90, 90)\), 12 sites on a helix, three displacement modes, \(z\sim N(0,\mathrm{diag}(1, 0.6, 0.3))\) then centred, half-set `hmax=2`, intensity noise \(0.05|F|^2+0.5\), seed 0. The note's \(N=128\) stored numbers are read from `reference/results*.json` when that file exists.

## Shells and S

Reporting shells use `rint.equal_count_shell_indices`. \(\sigma_A\) bins, when the nuisance fit runs, are those of `ml_i_nuisance_fit`. The integrated residual is `S_post` or `S_prior`, never a bare S.

## Deviations

- `reference/` (the numpy oracle and the note's stored numbers) is not in the tree. Those tests xfail until it is added.
- Per-dataset nuisances are initialised from `ml_i_nuisance_fit` or from the identities above, not from `multixtal.nuisance.fit_dataset_nuisance`.
- The joint fit is one payload. Dataset chunking applies to `ensemble_scores`. A fit that would exceed 64 MiB raises.
- Stacked modes are a complex array. `ensemble_maps` returns \(\Delta_0\) as a `MillerArray`.
