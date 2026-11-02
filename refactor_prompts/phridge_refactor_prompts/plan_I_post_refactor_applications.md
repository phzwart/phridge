# Plan I — First applications: multi-crystal pieces, EM observed kinds

Branch: `refactor/I`. Two stages. These are the spec's two worked
configurations, scheduled as the proof that the slot design holds. If any
stage needs `core/` to change, STOP and report what and why — that is the
finding, not a thing to work around.

---

## Stage I1 — multi-crystal

Five pieces, in order; each has its own test before the next starts.

1. **Structured `Predictive` fields.** Extend `contracts/tensors.py::Predictive`
   so a field may be a `StructuredField(low_rank: Tensor[h,N,K], diag: Tensor[h,N])`
   alongside plain tensors; `require` and `has` unchanged. Memory plan: never
   materialise `h×N×N`.
2. **Stacking transformation.** `transforms/observation/stacking.py` (moved in D4):
   build the consistent indexing table at configure through the `SymmetryProvider`
   (`shims/agentsg/` or `shims/cctbx/symmetry.py`, chosen by config), store as
   index tensors; `forward` gathers per-crystal Q fields onto `(crystal, h)`
   records; `adjoint` scatters back. Changing the table is a structural event.
3. **`mode_covariance` Q component.** Reads `f_loadings.per_crystal` and
   `amplitudes`, produces `gamma` and `c_pseudo` as `StructuredField`s; the
   mathematics is the improper-complex-normal estimand in the project notes
   (do not re-derive; implement the stated formulas and cite the note in the docstring).
4. **`joint_perturbation` rule.** `d/rules/joint_perturbation.py` with
   `needs = ("location","sigma_w","sigma_a","beta","gamma","c_pseudo")`.
   First conformance test: with one crystal and zero loadings it must equal
   `mli_quad` to 1e-10 on `mli_case()`.
5. **Generator.** `r/generators/perturbation_modes/` gets `forward` emitting N
   atom lists and `adjoint` distributing gradients, with the gauge constraint
   (orthonormal loadings, zero-mean amplitudes) applied inside.

Config: `examples/configs/multixtal_modes.yaml` per the spec sketch. Policy:
`policies/alternating_multixtal.py` (shared blocks → per-crystal amplitudes →
loadings → per-crystal scales). Partitions `unit: crystal_consistent`.

**Verify.** `python -m pytest -q tests/transforms tests/q tests/d tests/r tests/conformance tests/boundaries; phridge validate examples/configs/multixtal_modes.yaml`
**Commit.** `multi-crystal: stacking, mode covariance, joint rule, perturbation-mode generator`

---

## Stage I2 — EM observed kinds and projection

1. `io/read_em.py`: MRC map and half-maps → `Observed(kind="em_map")` on the
   box lattice (FFT once at load; P1; half-map difference → per-shell variance;
   FSC → per-shell table); STAR file → `Observed(kind="em_particles")` with
   pose, shift, CTF, optics group as index/data tensors. `mrcfile` and
   `starfile` are a new science-shim extra `em = ["mrcfile","starfile"]`,
   imported only in `io/read_em.py`.
2. `r/generators/rigid_box_placement.py` (six-parameter rigid placement).
3. `q/error_models/halfmap_noise.py` (model-free variance from half-maps),
   `q/error_models/fsc_trust.py` (σA from FSC, formula cited in docstring).
4. `d/rules/complex_gaussian.py` (`needs=("location","variance","sigma_a")`).
5. `transforms/observation/project.py`: central-slice extraction at pose
   (trilinear on the 3D Fourier grid, differentiable in slice and pose),
   phase ramp for shift, CTF multiplication; blocks `pose`, `shift`, `defocus`
   on the transformation; native torch, no shim.
6. Configs: `examples/configs/em_map.yaml`, `em_particles.yaml`; partitions
   `scheme: halfmap` and `unit: particle` respectively.

**Verify.** `python -m pytest -q tests/io tests/transforms tests/q tests/d tests/conformance tests/boundaries; phridge validate examples/configs/em_map.yaml examples/configs/em_particles.yaml`
**Commit.** `em: map and particle data spaces, projection transform, complex Gaussian rule`
