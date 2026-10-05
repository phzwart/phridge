# Ensemble perturbation modes — Phase 5

Deposited-structure alignment, the CLI, and the text report.

## What was done

- [`align.py`](../../src/phridge/contrib/ensemble/align.py): canonical index
  on the Laue orbit (the mate with the first nonzero index positive).
  Permissible origin shifts are the half-cell translations that preserve
  the space group, from cctbx. The shift is chosen by correlation of the
  phase-ramped common set against the reference. `gemmi_origin_crosscheck`
  runs only when both gemmi and `reference/cascan_pipeline.py` import.
- [`report.py`](../../src/phridge/contrib/ensemble/report.py): per-shell
  per-mode half-split correlation, the K-scan table, and the latent-cloud
  summary. Agreement statistics are `S_post` and `S_prior`. The optimistic
  conversion is labelled. A nonzero `hl_weight` and an enabled projection
  each print that the two likelihoods are different objectives.
- CLI `phridge-ensemble` subcommands `prepare`, `fit`, `scores`, `maps`,
  and `holdout`. `fit` uses `Bridge(memory=True)` when `--memory` is set.
  Default likelihood is `known_f`; `fit_nuisance` defaults off.

## Spec equations

Origin phase on the common set:

\[
F'(h) = F(h)\,\exp\bigl(2\pi i\, h\cdot t\bigr),
\]

with \(t\) restricted to the shifts returned for that space group.
Laue canonicalisation maps \(h\) to \(-h\) when the first nonzero component
is negative.

## Tests

[`tests/contrib/test_ensemble_designs.py`](../../tests/contrib/test_ensemble_designs.py):

- `test_alignment_recovers_origin_and_laue_index`.
  `canonical_laue_index(-1, 2, -3) = (1, -2, 3)` and the map is idempotent.
  `P 1` includes the zero shift. A `P 1 21 1` shift applied to a random
  common set is recovered at correlation \(> 0.99\), and undoing it
  restores the reference to \(10^{-6}\). `gemmi_crosscheck_available()` is
  false in this environment (gemmi is not installed, and
  `reference/cascan_pipeline.py` is absent).
- `test_report_names_s_post`: the report contains `S_post`, `S_prior`, and
  `optimistic`, and no bare `S`.
- `test_cli_holdout_writes_s_post`: `phridge-ensemble holdout` on an npz
  from `simulate_campaign` writes a report containing `S_post` and `K scan`.
- `test_cli_fit_on_pdb_campaign` xfails. See deviations.

## Deviations

- `multixtal.synthetic.write_synthetic_benchmark` raises `AssertionError`
  in `miller.adopt_set` (perturbed cells change the reflection count).
  That helper is not patched.
- A fixed-cell P1 PDB plus MTZ written by the test then fails inside
  `prepare_job` → `reindexing_operators` with
  `CCTBX_ASSERT(t_.den() == rhs.t_.den())` on this cctbx build. The CLI
  fit test xfails with that message. The hold-out CLI, which does not call
  `prepare_job`, passes.
- Gemmi is not installed here, so the gemmi origin cross-check did not run.
- `ensemble_maps` returns Δ0 as a `MillerArray`. Stacked modes stay a
  complex array.
- No speed-up is claimed.
