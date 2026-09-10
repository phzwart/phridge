# Interfacing with `phenix.refine` for Intensity-Based Likelihood (`mli_quad`)

This document provides a comprehensive technical reference on how **Phridge** interfaces with `phenix.refine` and `mmtbx` to enable direct intensity-based likelihood refinement (`mli_quad` / `mli`).

---

## 1. Executive Summary & Problem Formulation

### The Traditional Amplitude Paradigm
In conventional macromolecular refinement with `phenix.refine`, experimental intensities $I_{\text{obs}} \pm \sigma(I)$ are converted into structure factor amplitudes $|F_{\text{obs}}| \pm \sigma(F)$ via French-Wilson truncation prior to refinement. The model is then optimized against amplitude-based maximum likelihood targets (`ml`, `mlhl`, `ml_sad`, `ls`):

$$-\ln L(|F_c|) = -\sum_h \ln P(|F_{\text{obs}, h}| \mid |F_{c, h}|, \sigma_A, \Sigma_N)$$

This conversion suffers from fundamental statistical flaws:
1. **Truncation of Negative Observations**: In background-subtracted diffraction experiments, weak reflections often have $I_{\text{obs}} \le 0$. French-Wilson imposes an artificial non-negative prior, squashing observation variance and creating an artificial intensity/amplitude noise floor.
2. **Phase and Contrast Erasure**: High-resolution Fourier phases depend critically on the contrast provided by weak reflections. Artificially inflating weak $|F_{\text{obs}}|$ values obscures faint electron density features (e.g. ordered water networks, sub-stoichiometric ligands).
3. **Improper Scoring**: Legacy $R$-factors ($R_{\text{work}}$, $R_{\text{free}}$) are dominated by low-resolution, high-intensity reflections, obscuring genuine coordinate improvements in high-resolution shells.

### The Direct Intensity Likelihood Solution
Phridge's intensity engine directly evaluates the joint marginalization integral over the unobserved error-free normalized amplitude $E$:

$$L(E_c) = \int_0^\infty f_{\text{prior}}(E \mid E_c, \sigma_A) \cdot f_{\text{noise}}\!\left(Z_o \mid E^2, \sigma_Z, \nu\right) dE$$

where:
- $f_{\text{prior}}$ is the **Rice distribution** for acentric reflections and the **Woolfson distribution** for centric reflections.
- $f_{\text{noise}}$ is either a normal noise distribution or a heavy-tailed **Student-$t$ noise distribution** with degrees of freedom $\nu$.
- $Z_o = I_{\text{obs}} / (\varepsilon \Sigma)$ and $\sigma_Z = \sigma(I_{\text{obs}}) / (\varepsilon \Sigma)$ are normalized experimental intensities.
- Derivatives w.r.t. structure factors $\frac{\partial \text{LL}}{\partial F_c}$ are evaluated analytically via **Fisher's identity** or automatic differentiation without ever forming $|F_{\text{obs}}|$.

---

## 2. System Architecture & Process Separation

To prevent environment incompatibilities between CCTBX (frequently running in Python 3.9 or 3.10 with specialized C++ shared objects) and PyTorch (running modern CUDA/MPS autograd kernels), Phridge uses a decoupled client-server architecture:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      PHENIX / CCTBX CLIENT RUNTIME                      │
│                                                                         │
│   phenix.refine / mmtbx                                                 │
│       ├── refinement.main.target = mli_quad                             │
│       ├── IntensityFModel (subclass of mmtbx.f_model.manager)           │
│       ├── IntensityTargetFunctor & IntensityTargetResult                │
│       ├── IntensityElectronDensityMap                                   │
│       └── IntensityFModelInfo (S_post/S_prior & R_intensity)           │
│                                                                         │
│   phridge.client.convert / Bridge                                       │
│       └── LinkML Serialization: PackedMiller, PackedXrayStructure       │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                     Redis Streams / In-Memory Queue
                     RPC Operations:
                       • ml_i_target_and_gradients
                       • target_eval
                       • ml_i_maps
                       • ml_i_nuisance_fit
                                     │
┌────────────────────────────────────▼────────────────────────────────────┐
│                       PYTORCH / PHRIDGE WORKER                          │
│                                                                         │
│   phridge.contrib.intensity_ll.ops                                      │
│       ├── Differentiable Structure Factor Calculation (direct/fft)      │
│       ├── Adaptive Hybrid Quadrature (Gauss-Hermite & Gauss-Legendre)   │
│       ├── Student-t Heavy-Tailed Precision Mixture                      │
│       ├── Two-stage nuisance: intensity ML Wilson Σ, then σ_A (+ β)     │
│       ├── Fisher's Identity Exact Derivatives w.r.t. F_calc             │
│       ├── Block-Diagonal Gauss-Newton Curvatures & Preconditioning     │
│       └── Posterior Mode Amplitudes (E_mode -> F_mode)                  │
└─────────────────────────────────────────────────────────────────────────┘
```

The Phenix client requires **zero PyTorch dependencies**; all heavy tensor calculations are dispatched over the `Bridge`. For unified environments or test pipelines, `Bridge(memory=True)` executes the worker in-process.

---

## 3. The Interception Mechanism (`phenix_hook.py`)

The integration hook (`phridge.client.intensity.phenix_hook`) dynamically patches six core touchpoints in `mmtbx` and `phenix.refine` when `enable_intensity_in_phenix()` is invoked:

```mermaid
flowchart TD
    CLI["User Command: phenix.refine model.pdb data.mtz refinement.main.target=mli_quad"]
    CLI --> H1["1. Register Target Names (mmtbx.refinement.targets.target_names)"]
    H1 --> H2["2. Patch PHIL Master Parameters (phenix.refinement.master_params)"]
    H2 --> H3["3. Intercept FModel Manager (mmtbx.utils.fmodel_manager2)"]
    H3 --> H4["4. Patch Target Functor (mmtbx.refinement.targets.target_functor)"]
    H4 --> H5["5. Intercept Density Maps (mmtbx.f_model.manager.electron_density_map)"]
    H5 --> H6["6. Intercept Statistics (mmtbx.f_model.manager.info)"]
```

### Touchpoint 1: Target Registration
CCTBX maintains a global dictionary of recognized refinement targets in `mmtbx.refinement.targets.target_names`. The hook injects:
```python
target_names["mli_quad"] = target_attributes(family="ml", specialization="i")
target_names["mli"] = target_names["mli_quad"]
```
This registers `mli_quad` and its alias `mli` with the target attributes required by CCTBX macro cycles.

### Touchpoint 2: PHIL Parameter Schema Augmentation
When `phenix.refine` parses input arguments, `libtbx.phil` validates choices strictly. Without patching, `refinement.main.target=mli_quad` raises a choice error. The hook intercepts `phenix.refinement.master_params()`:
1. Locates `refinement.main.target`.
2. Injects the tokens `mli_quad` and `mli` into `node.words`.
3. Updates `node.caption` to include `MLI_QUAD`.
4. Wraps `phenix.refinement.misc.set_data_target_type` so any detection of `mli` standardizes to `"mli_quad"`.

### Touchpoint 3: FModel Manager Construction (`IntensityFModel`)
When `phenix.refine` sets up data managers via `mmtbx.utils.fmodel_manager2`:
```python
def patched_fmodel_manager2(f_obs, r_free_flags, ..., twin_law=None, target_name=None, i_obs=None, **kwargs):
    is_intensity_target = target_name in ("mli", "mli_quad", "ml_i")
    has_intensity_data = i_obs is not None or getattr(f_obs, "is_xray_intensity_array", lambda: False)()
    has_twin = bool(twin_law) or bool(kwargs.get("twin")) or bool(kwargs.get("is_twin"))

    if is_intensity_target or (has_intensity_data and target_name is None):
        if has_twin:
            raise IntensityTwinningError(
                "Twinning is not supported for intensity-based likelihood ('mli_quad') refinement: "
                "twin is True or twin_law was provided."
            )
        return IntensityFModel(
            i_obs=i_obs if i_obs is not None else f_obs,
            f_obs=f_obs,
            xray_structure=xray_structure,
            r_free_flags=r_free_flags,
            target_name=target_name or "mli_quad",
            ...
        )
```
- **Twinning Guard (Hard Error)**: Twinning is explicitly not supported for `mli_quad`. If `twin_law`, `twin=True`, or `is_twin=True` is encountered, a hard error (`IntensityTwinningError`, subclassing `RuntimeError`, `ValueError`, and `NotImplementedError`) is immediately raised.
- `IntensityFModel` subclasses `mmtbx.f_model.manager`.
- It maintains **amplitude scaffolding** (`self.f_obs()`) for legacy CCTBX routines (e.g. resolution binner, symmetry checks) while preserving genuine observed intensities `self.i_obs()` for the refinement engine.
- This bypasses French-Wilson amplitude conversion completely when raw intensities are supplied in the MTZ.

### Touchpoint 4: Target Functor Evaluation & Gradient Routing
During refinement macro cycles (coordinate minimization, simulated annealing, rigid body, B-factor optimization), `phenix.refine` calls `mmtbx.refinement.targets.target_functor(manager)`. The hook wraps `target_functor.__new__`:
```python
def patched_new(cls, manager, alpha_beta=None):
    if getattr(manager, "target_name", None) in ("mli", "mli_quad", "ml_i"):
        return IntensityTargetFunctor(manager, alpha_beta=alpha_beta)
    return orig_new(cls)
```

`IntensityTargetFunctor` calls the Phridge worker:
1. Dispatches `target_eval` with the current $F_{\text{model}}$ and target parameters.
2. Returns an `IntensityTargetResult` conforming to `mmtbx.refinement.targets.target_result_mixin`.
3. Returns target values:
   - `target_work()`: Negative log-likelihood on working reflections $W$.
   - `target_test()`: Negative log-likelihood on audit reflections $A$.
4. Computes exact derivatives w.r.t. structure factors:
   $$\frac{\partial T}{\partial F_{\text{calc}}} = \text{scale} \cdot \frac{\partial T}{\partial F_{\text{model}}}$$
5. By default, CCTBX `structure_factor_gradients_w` turns those into site / ADP / occupancy gradients.
6. With `--precondition` / `PHRIDGE_PRECONDITION=1`, `IntensityTargetResult.gradients_wrt_atomic_parameters` instead returns **packed Gauss–Newton–preconditioned** atomic gradients from the intensity worker (Cartesian sites, occupancy, $U_{\mathrm{iso}}$, and $U^*$). Weight selection should be re-run under that metric (see `PHRIDGE_WEIGHT_METRIC`).

### Touchpoint 5: Electron Density Map Synthesis
At the conclusion of macro cycles or when generating map coefficients, `phenix.refine` calls `fmodel.electron_density_map()`. The hook routes this to `IntensityElectronDensityMap`:
- Intercepts requests for `2mFo-DFc` and `mFo-DFc` maps.
- Dispatches `ml_i_maps` to the PyTorch worker.
- Evaluates the **Bayesian posterior mode amplitude** $E_{\text{mode}}$ via Newton's method on the posterior density:
  $$F_{\text{mode}} = \text{scale} \cdot \sqrt{\varepsilon \Sigma} \cdot E_{\text{mode}}$$
- Returns Fourier coefficients using the MAP estimator:
  - $2mF_o - DF_c$ equivalent: $(2 F_{\text{mode}} - |F_c|) \exp(i \phi_c)$
  - $mF_o - DF_c$ equivalent: $(F_{\text{mode}} - |F_c|) \exp(i \phi_c)$
  - Gradient and Newton difference density maps for rapid ligand and solvent placement.

### Touchpoint 6: Statistical Reporting (the S family)
Legacy $R$-factors do not represent proper scoring metrics under direct intensity modeling. `IntensityFModelInfo` overrides `mmtbx.f_model.manager.info`:
1. **S_post / S_prior** (`s_post_*`, `s_prior_*`): the expected residual $|E - k_S E_C|$ under the posterior, and the same functional under the prior. Same functional, different measure — **not** an R factor, and never labeled `r_work`/`r_free`. See [`maps.md` §9](../src/phridge/contrib/intensity_ll/maps.md).
2. **Direct Intensity R** (`r_intensity_*`): $\sum|I_{\mathrm{obs}}-|F_c|^2|/\sum|I_{\mathrm{obs}}|$ — a genuine point-estimate R on intensities, with no shrinkage, so it keeps the R label.
3. **Legacy French–Wilson amplitude R** (`r_work`/`r_free`/`r_all`): fills the mandated `R VALUE` / `FREE R VALUE` fields in REMARK 3. This is the deliberate legacy bridge and the only amplitude R in the output.
4. **Shrunken-amplitude diagnostics** (`r_post_*`, `r_mode_*`): biased low by posterior shrinkage, so they are demoted to a verbose block and never presented as R factors.
5. **`show_all`** prints the overall banner plus a resolution table
   (`show_rfactors_targets_in_bins`) with `S_post` / `S_prior`, `data%`, $\sigma_A$, and `scale_k1`.
   The post-scale stats report (`PHRIDGE_STATS_REPORT`) adds `S_post` / `S_prior` columns to the I/σ bins.
6. Formats an integrated summary banner directly into the `phenix.refine` output log:
```
+----------------------------------------------------------------------------+
| Intensity Likelihood Refinement (mli_quad)                                 |
|                                                                            |
| Direct Intensity R: r_work= 0.2814   r_free= 0.3120   r_all= 0.2831         |
| S_post:             work= 0.2700   free= 0.2900   S_prior_w/f= 0.2700/0.2700|
| S diagnostics:      k_S= 0.9628   E_C outliers= 0                           |
| Target NLL:         target_work= 12450.2104   target_free= 1238.9402        |
| Scale Factor k:     scale_k1= 1.0421   Student-t nu: 7.0                    |
+----------------------------------------------------------------------------+
```
`S_post(free)` plays the role conventionally played by R_free, with the shrinkage
pathology removed.

---

## 4. Complete Execution Lifecycle

The following sequence illustrates the complete life cycle of an iteration during refinement:

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant PR as phenix.refine
    participant FM as IntensityFModel
    participant TF as IntensityTargetFunctor
    participant BR as Phridge Bridge
    participant WK as PyTorch Worker

    User->>PR: phenix.refine model.pdb data.mtz refinement.main.target=mli_quad
    Note over PR: Hooks activated via enable_intensity_in_phenix()
    PR->>FM: Construct IntensityFModel(i_obs, xray_structure, r_free)
    FM->>BR: Register X-ray structure & scattering table
    PR->>TF: target_functor(fmodel)
    loop Macro Cycle Iteration
        PR->>TF: __call__(compute_gradients=True)
        TF->>BR: call("target_eval", f_calc, target_spec)
        BR->>WK: Calculate quadrature & exact derivatives
        WK-->>BR: Target NLL + dLL/dF_calc + curvatures
        BR-->>TF: RemoteTargetResult
        TF-->>PR: IntensityTargetResult
        PR->>PR: Compute atomic coordinate & ADP gradients
        PR->>FM: update_xray_structure(new_sites, new_u)
    end
    PR->>FM: electron_density_map()
    FM->>BR: call("ml_i_maps", target_spec)
    BR->>WK: Find posterior mode E_mode, form map coefficients
    WK-->>BR: Map coefficients (2mFo-DFc, mFo-DFc, f_mode)
    BR-->>PR: CCTBX Miller array of map coefficients
    PR->>FM: info()
    FM-->>PR: IntensityFModelInfo (S_post, R_intensity, NLL)
    PR-->>User: Refined PDB, MTZ map coefficients, log report
```

---

## 5. How to Run `phenix.refine` with `mli_quad`

### Method 1: The `phridge-refine` CLI (Recommended)
Phridge provides a dedicated entrypoint `phridge-refine` that automatically installs the hooks into the current Python runtime before executing `phenix.refine`:

```bash
phridge-refine model.pdb data.mtz \
    refinement.main.target=mli_quad \
    refinement.main.number_of_macro_cycles=5
```

### Method 2: In-Process Scripting with `phenix.python`
If executing inside custom refinement pipelines or Jupyter notebooks:

```python
import phenix.refinement
from phridge.client.intensity import enable_intensity_in_phenix

# 1. Activate the mli_quad hooks
enable_intensity_in_phenix()

# 2. Run standard phenix.refine command line runner
from phenix.command_line import refine
refine.run([
    "model.pdb",
    "data.mtz",
    "refinement.main.target=mli_quad",
    "refinement.main.number_of_macro_cycles=3",
])
```

### Method 3: Direct API / Custom Minimizer Integration
For researchers writing custom refinement workflows using `cctbx` and `mmtbx.f_model`:

```python
from iotbx import reflection_file_reader
from iotbx.pdb import combined_status
import iotbx.pdb
from phridge.client.intensity import IntensityFModel, enable_intensity_in_phenix

enable_intensity_in_phenix()

# Read inputs
pdb_io = iotbx.pdb.input(file_name="model.pdb")
xray_structure = pdb_io.xray_structure_simple()

mtz_io = reflection_file_reader.any_reflection_file("data.mtz")
miller_arrays = mtz_io.as_miller_arrays()
i_obs = next(a for a in miller_arrays if a.is_xray_intensity_array())
r_free_flags = next(a for a in miller_arrays if "free" in a.info().label_string().lower())

# Build the IntensityFModel
fmodel = IntensityFModel(
    i_obs=i_obs,
    xray_structure=xray_structure,
    r_free_flags=r_free_flags,
    target_name="mli_quad",
    nu=7.0,  # Student-t degrees of freedom
    memory=True,  # In-process bridge for development
)

# Macro-cycle target and gradients
tg = fmodel.target_and_gradients(preconditioned=True)
print(f"Target NLL: {tg.target():.4f}")
print(f"Cartesian gradients (first 3 atoms):\n{list(tg.gradients.d_target_d_site_cart()[:3])}")

# Compute electron density maps
edm = fmodel.electron_density_map()
map_coeffs_2fofc = edm.map_coefficients(map_type="2mFo-DFc")
map_coeffs_fofc = edm.map_coefficients(map_type="mFo-DFc")
```

---

## 6. Environment knobs

| Variable | Default | Meaning |
|---|---|---|
| `PHRIDGE_REDIS_URL` | (required for Redis bridge) | Worker Redis URL |
| `PHRIDGE_NU` | (engine default) | Student-t ν |
| `PHRIDGE_FIT_NU` | off | Fit ν during `update_all_scales` |
| `PHRIDGE_NU_MODE` | **`bins`** | ν fit: same resolution **bins** as σ_A (default) or scalar `global` |
| `PHRIDGE_PRECONDITION` | off | Gauss–Newton diagonal preconditioning of **XYZ, occupancy, and ADP** grads (Phenix uses packed worker grads instead of CCTBX SF→atom) |
| `PHRIDGE_MEMORY` | **on** | In-process `Bridge(memory=True)` (avoids Redis socket-read hangs). Set `0` / `--redis` for Redis worker |
| `PHRIDGE_HEARTBEAT` | **on** | Alive prints while waiting (target evals, scale updates, Redis waits) |
| `PHRIDGE_HEARTBEAT_INTERVAL` | **30** | Seconds between heartbeat lines |
| `PHRIDGE_STATS_REPORT` | **on** | After each `update_all_scales`, print resolution bins of I/σ, negatives, I/σ&lt;1..5, **σ_A(s)**, and data-fraction vs Wilson |
| `PHRIDGE_STATS_BIN_SIZE` | `500` | Reflections per stats bin |
| `PHRIDGE_SIGMA_A_MODE` | **`bins`** | σ_A(s) fit: per-resolution **bins** (monotone, default) or stiff Read curve (`read`) |
| `PHRIDGE_SIGMA_A_BINS` | auto | Number of σ_A shells when mode=`bins` |
| `PHRIDGE_SIGMA_A_TV_NORM` | `0` | Total-variation penalty λ_TV on adjacent σ_A **and** ν bin values (bins mode; e.g. `0.04`) |
| `PHRIDGE_FIT_SIGMA_WILSON` | **on** | Intensity-only ML Wilson fit of \(\Sigma_0>0\), \(B_W\) (no model / no \(\sigma_A\)); then freeze \(\Sigma\) and fit \(\sigma_A\). Set `0` to keep the moment-plot \(\Sigma\) only |
| `PHRIDGE_OMIT_WINDOWS` | off | After each `update_all_scales`, write stitched omit map MTZ (`{prefix}_omit_windows.mtz`) |
| `PHRIDGE_OMIT_BOX_SIZE` | `10` | Omit box edge length in Å |
| `PHRIDGE_OMIT_MODE` | **`boxes`** | `boxes` or `residue_blocks` |
| `PHRIDGE_OMIT_PREFIX` | `mli_omit` | Output prefix → ``{prefix}_omit_windows.mtz`` |
| `PHRIDGE_OMIT_CHUNK_SIZE` | auto | Windows per map-coefficient chunk |
| `PHRIDGE_OMIT_RESOLUTION_FACTOR` | `0.25` | FFT grid for real-space stitching |
| `PHRIDGE_OMIT_SAVE_NPZ` | off | Also write diagnostic per-window ``*_omit_windows.npz`` |
| `PHRIDGE_VERBOSE_TARGET` | **off** | Per-eval `mli_quad` banners (client + worker); set `1` to enable |
| `PHRIDGE_WEIGHT_METRIC` | **`nll`** | Rank XYZ/ADP weight trials by free-set NLL (`nll`) or classic R-free (`rfree`) |

**Intensities / French–Wilson:** `mli_quad` **never** runs French–Wilson and **never** reconstructs `I` as `F²`. The driver forces `xray_data.french_wilson_scale=False`, extract patches build a `√max(I,0)` amplitude scaffold without dropping reflections, and any `as_intensity_array()` / `french_wilson_scale` fallback raises `IntensityDataError`.

Weight selection rewiring (when `PHRIDGE_WEIGHT_METRIC=nll` and target is `mli_quad`):

- **XYZ** — Phenix still records R-factors (needed for its post-select assert). The scorer additionally stores free/work NLL per trial and picks the lowest **free-set NLL** among geometry-acceptable trials.
- **ADP** — Trial ranking columns are filled with NLL (so the usual R-gap filters become no-ops at NLL scale); the winner is the lowest free-set NLL. True R is still printed in the trial table.

CLI aliases: `--verbose-target`, `--weight-metric=nll|rfree`, `--sigma-a-bins=N`, `--tv-norm=λ`, `--fit-nu`, `--nu-mode=bins|global`, `--fit-sigma-wilson` / `--no-fit-sigma-wilson`, `--omit-windows` / `--omit-box-size` / `--omit-prefix` on `phenix_refine_mli.py` / `run_phenix_intensity.sh`.

Windowed omit coefficients (`ml_i_omit_windows`): see [`maps.md` §8](../src/phridge/contrib/intensity_ll/maps.md). Solvent/scales stay those of the full model; residual β is increased by omitted scattering and encoded as an effective $\sigma_A$ for the Rice maps API. Per-window coeffs are stitched in real space; default artifact is ``{prefix}_omit_windows.mtz`` (``FWT`` / ``DELFWT`` of the composite map); optional npz via ``PHRIDGE_OMIT_SAVE_NPZ=1``.

---

## 6b. Two-stage nuisance fit (`ml_i_nuisance_fit`)

On each `update_all_scales`, Phridge fits Wilson scale and $\sigma_A$ on the **tune** set (work reflections) in two stages. This is the default path that keeps $\sigma_A$ well-behaved in practice.

### Why not joint $(\Sigma_0, \sigma_A)$?

Maximizing the intensity NLL over both overall Wilson scale $\Sigma_0$ and $\sigma_A$ is degenerate: the optimizer can drive $\sigma_A \to 1$ while $\Sigma_0 \to \infty$ (Rice width $a = 1-\sigma_A^2$ collapses). Freeing those two together is what produced $\sigma_A$ stuck at $0.999$.

The classical remedy is the Read / cctbx split into **correlation** and **residual scale**:

| Component | Symbol | Role |
|---|---|---|
| Correlation | $\alpha = \sigma_A$ | Model–data agreement |
| Residual Wilson | $\beta = \Sigma\,(1-\sigma_A^2)$ | Unexplained intensity variance |

Phridge reports $\beta_h$ after the fit. A future joint residual/correlation parameterization should free $(\sigma_A,\beta)$, not $(\sigma_A,\Sigma_0)$.

### Stage 1 — Intensity-only ML Wilson (no model)

$$
\Sigma(s) = \Sigma_0\,\exp(-0.5\,B_W s^2),\qquad \Sigma_0 = e^{\ell} > 0
$$

1. Moment Wilson plot initializes $(\Sigma_0, B_W)$.
2. L-BFGS refines them under a **pure Wilson prior** ($\sigma_A \to 0$) times Gaussian noise on $I_{\mathrm{obs}}\pm\sigma_I$ — **no $F_{\mathrm{calc}}$**.
3. The quadrature returns $\log p(Z)$ with $Z = I/(\varepsilon\Sigma)$. Stage 1 uses the intensity density
   $$
   \log p(I) = \log p(Z) - \log(\varepsilon\Sigma).
   $$
   Omitting the Jacobian sends $\Sigma_0\to\infty$.

Disable with `--no-fit-sigma-wilson` / `PHRIDGE_FIT_SIGMA_WILSON=0` to keep the moment-plot $\Sigma$ only.

### Stage 2 — Freeze $\Sigma$, fit $\sigma_A$ (+ optional $\nu$)

With $\Sigma(s)$ held fixed:

- Default: monotone decreasing $\sigma_A$ in resolution bins (`PHRIDGE_SIGMA_A_MODE=bins`).
- Optional: Read-style curve (`=read`), TV on adjacent bins (`--tv-norm` / `PHRIDGE_SIGMA_A_TV_NORM`), shell count (`--sigma-a-bins`).
- Optional Student-$t$ $\nu$ (`--fit-nu`):
  - **`PHRIDGE_NU_MODE=bins`** (default): same resolution shells as $\sigma_A$, co-refined in L-BFGS; `--tv-norm` also penalizes adjacent $\nu$ jumps (scaled by $1/10$ so $\nu$ and $\sigma_A$ TV terms share $\lambda$).
  - **`=global`**: single scalar $\nu$ via bounded 1-D search (previous behaviour).
  - Per-reflection $\nu(s)$ is returned and used in subsequent target / map evaluations.

Standalone `phridge-intensity` retains overlapping bins / PAVA / TV for its own shell-wise $\sigma_A$ / $\nu$ path; the Phenix worker uses the two-stage procedure above.

---

## 7. Verification and Regression Testing

The interface is validated through dedicated test suites covering all integration layers:

| Test File | Verified Touchpoints |
|---|---|
| `tests/test_sigma_a_nuisance_bins.py` | Intensity-only ML Wilson ($\Sigma_0>0$, Jacobian), frozen-$\Sigma$ $\sigma_A$ bins / Read / TV, bin-wise $\nu$ + TV, $\beta=\Sigma(1-\sigma_A^2)$, no $\sigma_A\to 1$ collapse. |
| `tests/test_intensity_engine.py` | `IntensityFModel` inheritance from `mmtbx.f_model.manager`, structure factor caching, `target_and_gradients` with and without preconditioning, component-wise gradient accessors, posterior mode $E_{\text{mode}}$, and L-BFGS convergence. |
| `tests/test_phenix_refine_hook.py` | Registration of `mli_quad` in `mmtbx.refinement.targets.target_names`, PHIL choice validation, `fmodel_manager2` returning `IntensityFModel`, `target_functor` returning `IntensityTargetFunctor`, map synthesis via `compute_map_coefficients`, and clean import verification (confirming zero PyTorch imports on client side). |

To run the full test suite:
```bash
pytest tests/test_phenix_refine_hook.py tests/test_intensity_engine.py tests/test_sigma_a_nuisance_bins.py -v
```
