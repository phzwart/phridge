# Torch structure-factor engine and refinement targets

The worker implements a differentiable FFT structure-factor engine and a
small library of reciprocal-space targets. Together they replace the
`cctbx.xray.structure_factors` / target-functor pair for a refinement
loop that stays in Phenix but computes on a (GPU) torch worker.

## Information flow

A refinement target has the form

$$Q(\mathbf{x}) = \sum_h w_h\, g\big(\text{obs}_h, F_h(\mathbf{x})\big)$$

with $\mathbf{x}$ the scatterer parameters (sites, occupancies, $U_{iso}$,
$U^*$, $f'$, $f''$). The chain rule splits into a reciprocal-space part
and a model part:

$$\frac{dQ}{dx} = \sum_h \mathrm{Re}\!\left[\overline{G_h}\,\frac{\partial F_h}{\partial x}\right], \qquad G_h = \frac{\partial g_h}{\partial A_h} + i\,\frac{\partial g_h}{\partial B_h}$$

where $F_h = A_h + iB_h$. $G_h$ is exactly cctbx's `d_target_d_f_calc`,
and it is also what `torch.autograd` returns for a real loss of a complex
tensor, so the two conventions line up with no factors of two.

| Stage | Where | How |
|-------|-------|-----|
| $F_h(\mathbf{x})$ | worker `xtal.engine` | atoms sampled as real-space Gaussians on a grid, `torch.fft.fftn`, gather at $h$ |
| $G_h = \partial g/\partial F_h$ | worker `targets` | autograd of the target w.r.t. the complex `f_calc` tensor |
| $\partial F_h/\partial x$ chain | worker `xtal.engine` | vector-Jacobian product through the FFT (the Agarwal gradient-map trick, for free) |
| packing for a minimizer | client `xtal_engine` | cctbx `packing_order_convention == 2`, cartesian site / $U_{cart}$ gradients |

## Forward model

Per symmetry-expanded atom $j$, the scattering density is a sum of
normalized 3-D Gaussians with covariance

$$M_{jk} = U_{cart,j} + \left(\frac{b_k}{8\pi^2} + u_{extra}\right) I$$

one per form-factor term $a_k \exp(-b_k s^2)$, plus one for the constant
term $c + f'$ and (imaginary) $f''$. Space-group symmetry is applied by
expanding every atom over the full operator list with weight
$\text{occ} \cdot \text{multiplicity} / n_{sym}$, which matches cctbx on
special positions. The extra $u_{extra}$ (cctbx `u_base`, same formula and
`quality_factor`) smooths the density before sampling and is divided out
per reflection:

$$F_h = \frac{V}{N_{grid}}\, \mathrm{FFT}[\rho]_{-h}\; \exp\!\big(2\pi^2 u_{extra}\, |d^*_h|^2\big)$$

Accuracy against `algorithm="direct"` is at the level of cctbx's own FFT
engine: relative error $\sim 10^{-3}$ on $F$ and site / occupancy
gradients, $\sim 10^{-2}$ on $U$ gradients at `quality_factor=1000`.

## Second derivatives

Differentiating once more,

$$\frac{\partial^2 Q}{\partial x\,\partial y} = \sum_h \left[\left(\frac{\partial u_h}{\partial x}\right)^{\!\top} H_h \left(\frac{\partial u_h}{\partial y}\right) + G_h^\top \frac{\partial^2 u_h}{\partial x\,\partial y}\right]$$

with $u_h = (A_h, B_h)$. The first (Gauss-Newton) term is what
`gauss_newton_hvp` computes as a Hessian-vector product: one `jvp`
through the engine, the $2\times 2$ curvature per reflection, one `vjp`
back. For amplitude-only targets the curvature is

$$H_h = g''(|F_h|)\, e_h e_h^\top + \frac{g'(|F_h|)}{|F_h|}\, e_h^\perp e_h^{\perp\top}$$

and `TargetResult` carries `curv_radial` ($g''$) and `curv_tangential`
($g'/|F|$). The second (residual) term is block-diagonal by atom and is
not computed; it vanishes in expectation at convergence.

## Targets

Targets are registered by name and built from a JSON spec:

| name | spec options | matches |
|------|--------------|---------|
| `ls` | `obs_type` "F" or "I", `scale_factor`, `compute_scale_using_all_data`, `use_sigmas_as_weights` | `cctbx.xray.ext.targets_least_squares_residual[_for_intensity]` |
| `ml_f` | `scale_factor` | `cctbx.xray.ext.mlf_target_and_gradients` (needs `alpha`, `beta`, `epsilon`, `centric`) |

A new target is a subclass of `phridge.worker.targets.Target` with a
`per_reflection(f_calc, obs)` method; autograd supplies `d_target_d_f_calc`
and the curvatures.

## Ops

| op | inputs | outputs |
|----|--------|---------|
| `sf_calc` | `xray`, `table`, `params`, `hkl` | `f_calc` (complex MillerArray) |
| `sf_gradients` | `xray`, `table`, `params`, `d_target_d_f_calc` | `gradients` (SfGradients) |
| `target_eval` | `f_calc`, `f_obs`, `target`, optional `weights`, `r_free`, `alpha`, `beta`, `epsilon`, `centric` | `target` (TargetResult) |
| `refine_gradients` | `xray`, `table`, `params`, `f_obs`, `target`, optional arrays | `f_calc`, `target`, `gradients` |
| `gauss_newton_hvp` | `xray`, `table`, `params`, `target`, `hkl`, `v` (SfGradients layout) | `hv` (SfGradients) |

`table` is a `ScatteringTable` (Gaussian coefficients per scattering
type, `scattering_table_from_cctbx(xray_structure)`), and
`CrystalSymmetry.symops` must be present (default for
`crystal_from_cctbx`).

## Phenix side

```python
from phridge.client import Bridge
from phridge.client.xtal_engine import RemoteStructureFactors, RemoteTargetFunctor, RemoteRefinementTarget

bridge = Bridge("redis://gpu-box:6379/0")

# like xray.structure_factors.from_scatterers(...).f_calc() / .gradients(...)
engine = RemoteStructureFactors(bridge, xray_structure, miller_set, d_min=2.0)
f_calc = engine.f_calc()
grads = engine.gradients(d_target_d_f_calc)      # .d_target_d_site_frac(), .packed(), ...

# like a cctbx target functor
functor = RemoteTargetFunctor(bridge, f_obs, {"name": "ml_f"}, alpha=alpha, beta=beta, r_free_flags=flags)
res = functor(f_calc)                            # .target_work(), .gradients_work(), .curvatures_work()

# one round trip per minimizer step
refiner = RemoteRefinementTarget(bridge, xray_structure, f_obs, {"name": "ls", "obs_type": "F"}, d_min=2.0)
target, packed_gradients = refiner.target_and_gradients(xray_structure)   # feed to scitbx.lbfgs
```

Bulk solvent and overall scaling are not part of the engine; use
`mmtbx.f_model` on the client and pass the resulting per-reflection
`d_target_d_f_calc` through `sf_gradients`, or feed `F_model`-scaled data
to the target ops.

## Performance and platform notes

Sampling cost is (expanded atoms) x (box points) x (Gaussian terms); atoms
are bucketed by cutoff radius and isotropic atoms take a spherical fast
path. On a 2-core CPU in float64 a 2000-atom P212121 structure at 2 A
takes ~10 s for F_calc and ~45 s for gradients — this engine is meant for
a GPU worker (`phridge-worker --device cuda`), where the same work is a
few hundred million fused exp evaluations. `SfEngineParams.wing_cutoff`
(default 1e-4; cctbx uses 1e-3) and `quality_factor` trade accuracy for
box size and grid size.

On CPU the FFT goes through numpy (`EngineParams.cpu_numpy_fft`) and the
Gaussian quadratic forms are written out elementwise: with the torch
2.14 / MKL build used for development, multi-threaded `torch.fft` and
batched GEMM backward returned wrong results intermittently once other
parallel kernels had run in the process. Both workarounds are exact and
cost nothing on CUDA, which uses cuFFT.

Note: in one process, import cctbx before torch. Importing torch first
crashes cctbx's Boost.Python extensions, which is one more reason the
client and worker are separate processes.
