# Ensemble perturbation modes — Phase 2

Exact E-step, generalised EM, and the Fisher-identity L-BFGS path.

## What was done

- [`posterior_moments`](../../src/phridge/contrib/intensity_ll/maps.py) gained
  keyword-only `hl=None`, a complex first moment `e_complex`, and
  `log_lik_hl`. `hl is None` keeps the previous sums for `E_mean`, `I_mean`,
  and `log_lik`.
- With HL, amplitude-node weights are multiplied by \(I_0(|v|)/I_0(\kappa)\)
  (centric: the two-point normaliser ratio). `log_lik` stays the unweighted
  `mli` value. `log_lik_hl` is reported separately and is not comparable
  across `hl_weight`.
- [`moments.py`](../../src/phridge/contrib/ensemble/moments.py) converts E
  back to F with \(\sqrt{\varepsilon\Sigma_W}\) and documents the centric
  two-point formulae. `fixed_grid_posterior` freezes the Gauss–Legendre
  window on \([0, 6]\) for the finite-difference check.
- `FullComplexLoadings` and `ComplexGLS` live only in this package.
  Design \(W=[1\mid z]\), weights \(\mathrm{mask}\,D^2/\sigma^2_\Delta\),
  `torch.linalg.solve`. Score covariance is the inverse real Gram.
  \(\sum_n z_{nk}=0\).
- Shell \(D\) and \(\sigma^2_\Delta\) from latent second moments. When
  `dof_correct` is set, \(\sigma^2\) is multiplied by \(N/(N-K-1)\) once
  per sweep. Default `EnsembleOptions.dof_correct` is true. The monotone EM
  test turns it off, because the factor is not an M-step maximizer.
- Gauge: thin SVD of \([\mathrm{Re}\,\Delta\mid\mathrm{Im}\,\Delta]\) scaled
  by \(z\). `em_fit` and `direct_fit` (`torch.optim.LBFGS`). Fisher identity
  \(\partial L/\partial\mu = 2(\mathbb{E}[F]-\mu)/\sigma^2\).

## Spec equations

Acentric, model frame \(\arg\mu=0\):

\[
v = \kappa + (A+iB),\qquad
\langle e^{i\phi}\mid r\rangle = e^{i\arg v}\,I_1(|v|)/I_0(|v|).
\]

Centric, phases \(0\) and \(\pi\) on the axis:

\[
\langle e^{i\phi}\rangle = \tanh(\kappa_{\mathrm{cen}}+A).
\]

Scores:

\[
\mathrm{Re}[\overline{\Delta}\,\Omega\Delta^\mathsf{T}]\,z
  = \mathrm{Re}[\overline{\Delta}\,\Omega(R-\Delta_0)].
\]

## Tests

Always-on, in [`tests/contrib/test_ensemble.py`](../../tests/contrib/test_ensemble.py):

| Check | Measured | Gate |
| --- | --- | --- |
| V1 `hl=None` vs omitted | `torch.equal` on `E_mean`, `I_mean`, `log_lik` | bit-identical |
| V2 vs trapezoid transcription (8000 nodes) | \(\lvert E_m\rvert\) abs \(6.92\times 10^{-7}\); log-lik abs \(1.40\times 10^{-7}\) | \(10^{-4}\) |
| V3 Fisher vs frozen-grid FD | relative \(5.42\times 10^{-11}\) | \(10^{-6}\) |
| V4 EM, 20 sweeps, `dof_correct=False`, \(N=4\), `hmax=1` | history length 21; smallest step \(+0.317\) nats | no drop \(>10^{-6}\) |
| V5 gauge | field change \(6.0\times 10^{-15}\); log-lik change \(4\times 10^{-29}\) | \(10^{-10}\) |
| V7 centric vs two-point sum | under \(10^{-8}\) on the tested grid | \(10^{-8}\) |
| V14 missing column | parameters unchanged when the masked column is set to \(10^6\) | \(10^{-8}\) |
| V15 score covariance | max abs vs `inv(Gram)` under \(10^{-10}\) | \(10^{-10}\) |
| V16 audit role 2 | parameters unchanged when audit entries are overwritten | \(10^{-8}\) |

`tests/test_intensity_maps.py`, `tests/contrib/test_free_beta.py`, and
`tests/contrib/test_interleaved.py` (`not slow and not gpu`): 59 passed,
1 skipped. V2's `deltaem.estep_moments` import is the `reference/` xfail.

## Deviations

- Nuisances are initialised from `ml_i_nuisance_fit` when `fit_nuisance` is
  set, otherwise from \(D=\sigma_A\sqrt{\Sigma_W/\Sigma_{\mathrm{calc}}}\) and
  \(\sigma^2_\Delta=(1-\sigma_A^2)\varepsilon\Sigma_W\).
  `multixtal.nuisance.fit_dataset_nuisance` is not called.
- The finite-difference check uses the frozen window in
  `fixed_grid_posterior`, not a moving `mli` window.
- V4 and the slow campaign set `dof_correct=False`. A separate test checks
  that the multiplier equals \(N/(N-K-1)\) when the flag is on
  (\(N=8\), \(K=2\): factor \(8/5\)).
- No speed-up is claimed. Evaluation counts were not compared.
