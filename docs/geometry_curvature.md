# Second derivatives and preconditioning for the geometry term

The torch restraint energy is a weighted sum of squares,

$$
E(x) = \sum_r w_r\, \rho_r(x)^2 ,
$$

with residuals $\rho_r$ for bonds ($|d| - d_0$ beyond the slack), angles ($\theta - \theta_0$
in degrees) and dihedrals ($\sqrt{2}\sin\tfrac{n(\varphi-\varphi_0)}{2}$, so that
$w\rho^2 = w(1-\cos n(\varphi-\varphi_0))$). Its Gauss–Newton matrix

$$
H_{GN} = 2 J^\top W J, \qquad J_{ri} = \partial \rho_r / \partial x_i ,
$$

is positive semi-definite and sparse: each residual touches at most four atoms, so
$H_{GN}$ only couples bonded neighbours. `phridge.worker.geometry.curvature` builds it
exactly and cheaply by evaluating every residual on *gathered* coordinates
$P_r \in \mathbb{R}^{k_r\times 3}$, so that one autograd call on $\sum_r \rho_r$ returns every
per-residual gradient (the gathered Jacobian is block diagonal), and scatter-adding
$2w_r\, g_r g_r^\top$ onto the atoms. From the same gradients come the per-atom $3\times3$
blocks, the diagonal, the COO triplets of the full sparse matrix, Hessian–vector products
by double vjp (Gauss–Newton or the full Hessian including $\sum_r 2w_r\rho_r\nabla^2\rho_r$,
which is indefinite away from the minimum), and two factorised solvers.

## What actually preconditions the geometry term

The obvious candidate, a Jacobi (diagonal) preconditioner, does nothing here, and it is worth
being explicit about why. The stiffness of a restrained chain sits in the *relative*
displacement of bonded atoms — it is off-diagonal by construction — while the per-atom
diagonal of a randomly oriented bond network is nearly isotropic. Measured on a 300-atom
restrained chain (bond $\sigma$ 0.02 Å, angle $\sigma$ 3°, dihedrals, five long-range
bonds), conjugate-gradient iterations to $10^{-6}$ on the damped Gauss–Newton system:

| preconditioner | CG iterations |
|---|---|
| none | 380 |
| diagonal (Jacobi) | 450 |
| per-atom $3\times3$ blocks | ~ same as diagonal |
| residue block-diagonal | 170 |
| residue block-tridiagonal | 12 |
| full sparse factorisation | 1 |

The two that work are the ones that keep inter-atom coupling. `GaussNewtonPreconditioner`
factorises $w\,2J^\top WJ + \mathrm{diag}(\text{extra}) + \mu I$ with scipy's sparse LU
(CHOLMOD when `sksparse` is importable). `BlockTridiagonalPreconditioner` keeps the
residue diagonal blocks and the couplings between consecutive residues and factorises them
by block $LDL^\top$ in $O(n)$; for a linear chain whose restraints bridge at most one
residue boundary (peptide bond, the three angles across the link, ω, planarity) it is
exact, and disulfides, H-bond / base-pair restraints, NCS and non-bonded contacts are the
dropped terms. Its band structure is fixed, so the blocks are precomputable per residue
type / pair at ideal geometry (rotation-covariant) and batch on the GPU where a general
sparse Cholesky does not.

The `preconditioner="diagonal"` option of `energy_and_sites` (Jacobi scaling
$y = s\,x$, $s = \sqrt{\mathrm{diag}\,H_{GN}}$) is kept for completeness and for
first-order methods, with the measurement above as the caveat.

## Gauss–Newton minimisation of the geometry term

With the factorisation in hand, the geometry-only problem is best solved directly:
`energy_and_sites(..., optimizer="gauss_newton")` runs Levenberg–Marquardt,

$$
(2J^\top WJ + \mu\,\bar d\, I)\, p = -\nabla E ,
$$

with $\mu$ relative to the median diagonal $\bar d$ (start $10^{-3}$, $\times 4$ on a
rejected step, $\div 3$ on success). On the shaken 2000-atom chain it goes from
$E = 10^6$ to $10^{-9}$ in 40 steps (1.6 s on two CPU threads); L-BFGS with 400
iterations is still at $E \approx 3$, and the Jacobi-scaled L-BFGS is no better.

## Joint x-ray + geometry refinement

`phridge.client.joint.JointSiteRefinement(refiner, geometry, weight)` runs damped
Newton–CG on cartesian sites for

$$
Q(x) = Q_\text{xray}(x) + w\,E(x), \qquad H = F^\top H^{GN}_\text{xray} F + w\,2J^\top WJ ,
$$

with $F$ the fractionalisation matrix. Each CG iteration costs two worker calls
(`gauss_newton_hvp` for the FFT engine, `geometry_hvp` for the restraints); the
preconditioner is `geometry_gn_solve` with the x-ray per-atom block diagonal (from
`gauss_newton_blocks`, transformed to cartesian) passed as `extra_diag`, either the full
sparse factorisation (`method="sparse"`) or the residue block-tridiagonal one
(`method="tridiagonal"`, `groups` from `RemoteGeometry.residue_groups()`). On a synthetic
36-atom test with a geometry-dominated weight it reaches 0.001 Å from a 0.25 Å shake in
six steps and 77 HVPs, against 0.028 Å and 150 HVPs for the same Newton–CG with a
diagonal preconditioner.

Two details matter for the x-ray half. First, $J^\top H_F J$ with the exact
$(A,B)$-space curvatures ($c_r = g''$, $c_t = g'/|F|$) is indefinite whenever
$c_t < 0$ (least squares with $|F_o| > k|F_c|$, most ML targets in places); for
preconditioning and Newton–CG the clients use the positive semi-definite surrogate with
$c_t \leftarrow \max(c_t,0)$, $c_r \leftarrow \max(c_r,0)$ (`psd_target`, the usual
Tronrud / REFMAC choice; `psd=False` restores the exact operator). Second, this work
uncovered a sign error in `gauss_newton_blocks`: the Tronrud sum/difference weights were
swapped, which is invisible at zero residual ($c_t = 0$) but gives 30–50 % errors in the
blocks otherwise. The correct split is

$$
H_{ab} = \sum_h \Bigl[\tfrac{c_r+c_t}{2}\,\mathrm{Re}(\bar d_a d_b) + \tfrac{c_r-c_t}{2}\,\mathrm{Re}(d_a d_b e^{-2i\varphi})\Bigr],
$$

and the blocks now agree with the HVP and with a finite-difference Jacobian reference at
nonzero residual (`test_gauss_newton_blocks_match_hvp_at_nonzero_residual`).

## API summary

| where | what |
|---|---|
| `worker/geometry/curvature.py` | `RestraintCurvature` (residuals, `gn_blocks`, `gn_diagonal`, `gn_hvp`, `full_hvp`), `gn_sparse_coo`, `GaussNewtonPreconditioner`, `BlockTridiagonalPreconditioner`, `jacobi_scale` |
| `worker/geometry/energy.py` | `optimizer="gauss_newton"`, `preconditioner="diagonal"`, `precond_refresh` |
| ops | `geometry_curvature` (diagonal, blocks, gradient, COO), `geometry_hvp` (`hessian` gn/full), `geometry_gn_solve` (`method` sparse/tridiagonal, `extra_diag`, `weight`, `damping`) |
| `client/geometry.py` | `RemoteGeometry.curvature / hvp / gn_solve / residue_groups`, `minimize(optimizer="gauss_newton", preconditioner=...)` |
| `client/joint.py` | `JointSiteRefinement.newton_cg(precondition, method, groups, psd)` |
| `sfcalc/client.py` | `psd_target`, `psd=` on `curvatures / diagonal / newton_cg` |

A footnote on CPU torch: on builds where `torch.fft.fftn` misbehaves with more than one
intra-op thread (torch 2.14 + MKL in the test container), `torch.set_num_threads(k>1)`
silently corrupts the FFT engine's $F_\text{calc}$. Leave the thread count alone or set
`EngineParams(cpu_numpy_fft=True)`; the tests never touch it.
