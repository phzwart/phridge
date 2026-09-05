"""Phenix-facing drop-ins for cctbx.xray.structure_factors and target functors.

All computation happens on the torch worker; this module only converts
cctbx objects to wire types and results back to flex arrays in the
shapes cctbx / mmtbx expect.

    engine = RemoteStructureFactors(bridge, xray_structure, miller_set, d_min=2.0)
    f_calc = engine.f_calc()                       # miller.array, like from_scatterers(...).f_calc()
    grads = engine.gradients(d_target_d_f_calc)    # .d_target_d_site_frac(), ..., .packed()

    functor = RemoteTargetFunctor(bridge, f_obs, {"name": "ml_f"}, alpha=..., beta=..., r_free_flags=...)
    result = functor(f_calc, compute_gradients=True)   # .target_work(), .gradients_work(), .d_target_d_f_calc()

    refiner = RemoteRefinementTarget(bridge, xray_structure, f_obs, target_spec, d_min=2.0)
    target, packed_gradients = refiner.target_and_gradients(xray_structure)  # one round trip
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from phridge.client.convert import crystal_from_cctbx, miller_from_cctbx
from phridge.client.convert_xtal import scattering_table_from_cctbx, xray_from_cctbx
from phridge.models import SfEngineParams
from phridge.packing import PackedMiller
from phridge.packing_scattering import PackedSfCurvatures, PackedSfGradients, PackedTargetResult


def _flex():
    from cctbx.array_family import flex

    return flex


def _hkl_of(miller_set: Any) -> np.ndarray:
    return np.asarray(list(miller_set.indices()), dtype=np.int32)


def _miller_template(miller_set: Any) -> PackedMiller:
    """MillerArray carrying only indices (data ignored by the worker)."""
    return PackedMiller(
        crystal=crystal_from_cctbx(miller_set.crystal_symmetry()),
        hkl=_hkl_of(miller_set),
        data=np.zeros(miller_set.size(), dtype=np.float64),
        anomalous=bool(miller_set.anomalous_flag()),
    )


def _packed_xray(xray_structure: Any, table: Optional[str]):
    return xray_from_cctbx(xray_structure), scattering_table_from_cctbx(xray_structure, table)


class RemoteGradients:
    """Mirror of cctbx.xray.structure_factors.gradients_direct results."""

    def __init__(self, xray_structure: Any, packed: PackedSfGradients) -> None:
        self.xray_structure = xray_structure
        self.raw = packed

    def d_target_d_site_frac(self):
        flex = _flex()
        return flex.vec3_double([tuple(float(x) for x in row) for row in self.raw.d_site_frac])

    def d_target_d_site_cart(self):
        """Cartesian site gradients: F^T @ d/dx_frac (cctbx convention)."""
        flex = _flex()
        uc = self.xray_structure.unit_cell()
        f = np.asarray(uc.fractionalization_matrix(), dtype=np.float64).reshape(3, 3)
        cart = self.raw.d_site_frac @ f
        return flex.vec3_double([tuple(float(x) for x in row) for row in cart])

    def d_target_d_u_iso(self):
        return _flex().double(self.raw.d_u_iso.tolist())

    def d_target_d_u_star(self):
        flex = _flex()
        return flex.sym_mat3_double([tuple(float(x) for x in row) for row in self.raw.d_u_star])

    def d_target_d_u_cart(self):
        from cctbx import adptbx

        flex = _flex()
        uc = self.xray_structure.unit_cell()
        return flex.sym_mat3_double(
            [adptbx.grad_u_star_as_u_cart(uc, tuple(float(x) for x in row)) for row in self.raw.d_u_star]
        )

    def d_target_d_occupancy(self):
        return _flex().double(self.raw.d_occupancy.tolist())

    def d_target_d_fp(self):
        return _flex().double(self.raw.d_fp.tolist())

    def d_target_d_fdp(self):
        return _flex().double(self.raw.d_fdp.tolist())

    def packed(self):
        """Gradients packed in cctbx packing_order_convention 2.

        Per scatterer: site (cartesian, 3) if grad_site; u_iso if
        use_u_iso and grad_u_iso; u_cart (6) if use_u_aniso and
        grad_u_aniso; occupancy if grad_occupancy; fp / fdp if any
        scatterer has grad_fp / grad_fdp (cctbx quirk).
        """
        from cctbx import adptbx

        flex = _flex()
        xs = self.xray_structure
        uc = xs.unit_cell()
        f = np.asarray(uc.fractionalization_matrix(), dtype=np.float64).reshape(3, 3)
        scatterers = xs.scatterers()
        any_fp = any(sc.flags.grad_fp() for sc in scatterers)
        any_fdp = any(sc.flags.grad_fdp() for sc in scatterers)
        out: list[float] = []
        for i, sc in enumerate(scatterers):
            fl = sc.flags
            if fl.grad_site():
                out.extend((self.raw.d_site_frac[i] @ f).tolist())
            if fl.use_u_iso() and fl.grad_u_iso():
                out.append(float(self.raw.d_u_iso[i]))
            if fl.use_u_aniso() and fl.grad_u_aniso():
                out.extend(adptbx.grad_u_star_as_u_cart(uc, tuple(float(x) for x in self.raw.d_u_star[i])))
            if fl.grad_occupancy():
                out.append(float(self.raw.d_occupancy[i]))
            if any_fp:
                out.append(float(self.raw.d_fp[i]))
            if any_fdp:
                out.append(float(self.raw.d_fdp[i]))
        return flex.double(out)


def _pack_diagonal_like_gradients(xray_structure: Any, packed: PackedSfGradients):
    """Pack a Hessian diagonal (SfGradients layout) like RemoteGradients.packed()."""
    return RemoteGradients(xray_structure, packed).packed()


def _invert_diag(curvatures, floor: float = 1e-8):
    """Inverse-Hessian diagonal for L-BFGS Hk0; clamp to keep entries positive."""
    flex = _flex()
    arr = np.asarray(list(curvatures), dtype=np.float64)
    inv = np.where(arr > floor, 1.0 / arr, 1.0 / floor)
    return flex.double(inv.tolist())


class RemoteStructureFactors:
    """Drop-in for cctbx.xray.structure_factors.from_scatterers(...) and .gradients."""

    def __init__(
        self,
        bridge: Any,
        xray_structure: Any,
        miller_set: Any,
        *,
        d_min: Optional[float] = None,
        params: Optional[SfEngineParams] = None,
        table: Optional[str] = None,
    ) -> None:
        self.bridge = bridge
        self.xray_structure = xray_structure
        self.miller_set = miller_set
        if params is None:
            d_min = d_min if d_min is not None else float(miller_set.d_min())
            params = SfEngineParams(d_min=d_min)
        self.params = params
        self.table = table
        self._template = _miller_template(miller_set)

    def _inputs(self) -> dict:
        xray, table = _packed_xray(self.xray_structure, self.table)
        return {"xray": xray, "table": table, "params": self.params}

    def f_calc(self):
        """Complex miller.array on ``miller_set``."""
        packed = self.bridge.call("sf_calc", hkl=self._template, **self._inputs())
        from phridge.client.convert import miller_to_cctbx

        return miller_to_cctbx(packed if isinstance(packed, PackedMiller) else packed)

    def gradients(self, d_target_d_f_calc: Any) -> RemoteGradients:
        """``d_target_d_f_calc``: flex.complex_double or numpy, aligned with ``miller_set``."""
        data = np.asarray(list(d_target_d_f_calc) if not isinstance(d_target_d_f_calc, np.ndarray) else d_target_d_f_calc, dtype=np.complex128)
        dtdf = PackedMiller(
            crystal=self._template.meta.crystal,
            hkl=self._template.hkl,
            data=data,
            anomalous=self._template.meta.anomalous,
        )
        packed = self.bridge.call("sf_gradients", d_target_d_f_calc=dtdf, **self._inputs())
        return RemoteGradients(self.xray_structure, packed)


class RemoteTargetResult:
    """Mirror of cctbx target_and_gradients results (work/test split, gradients_work)."""

    def __init__(self, packed: PackedTargetResult, r_free: Optional[np.ndarray]) -> None:
        self.raw = packed
        self._work = np.ones(packed.meta.n_refl, dtype=bool) if r_free is None else ~r_free

    def target_work(self) -> float:
        return self.raw.meta.value

    def target_test(self) -> Optional[float]:
        return self.raw.meta.value_test

    def target(self) -> float:
        return self.raw.meta.value

    def target_per_reflection(self):
        return _flex().double(self.raw.per_reflection.tolist())

    def d_target_d_f_calc(self):
        """Full-length (zero on free reflections)."""
        return _flex().complex_double(self.raw.d_target_d_f_calc.tolist())

    def derivatives(self):
        return self.d_target_d_f_calc()

    def gradients_work(self):
        return _flex().complex_double(self.raw.d_target_d_f_calc[self._work].tolist())

    def curvatures_work(self):
        """(radial, tangential) flex.double on the work set, or None."""
        if self.raw.curv_radial is None:
            return None
        flex = _flex()
        return (
            flex.double(self.raw.curv_radial[self._work].tolist()),
            flex.double(self.raw.curv_tangential[self._work].tolist()),
        )

    def scale_factor(self) -> Optional[float]:
        return self.raw.meta.scale_factor


class RemoteTargetFunctor:
    """Target functor evaluated on the worker.

    ``target_spec`` e.g. {"name": "ls", "obs_type": "F"} or {"name": "ml_f", "scale_factor": 1.0}.
    Optional per-reflection arrays (flex or numpy) aligned with ``f_obs``.
    """

    def __init__(
        self,
        bridge: Any,
        f_obs: Any,
        target_spec: dict,
        *,
        r_free_flags: Any = None,
        weights: Any = None,
        alpha: Any = None,
        beta: Any = None,
        epsilons: Any = None,
        centric_flags: Any = None,
        compute_curvature: bool = True,
    ) -> None:
        self.bridge = bridge
        self.f_obs = f_obs
        self.target_spec = dict(target_spec)
        self.compute_curvature = compute_curvature
        self._packed_obs = miller_from_cctbx(f_obs)
        name = self.target_spec.get("name")
        if name == "ml_f":
            if epsilons is None:
                epsilons = f_obs.epsilons().data().as_double()
            if centric_flags is None:
                centric_flags = f_obs.centric_flags().data()
        self.arrays = {
            "r_free": _optional_array(r_free_flags, bool),
            "weights": _optional_array(weights, np.float64),
            "alpha": _optional_array(alpha, np.float64),
            "beta": _optional_array(beta, np.float64),
            "epsilon": _optional_array(epsilons, np.float64),
            "centric": _optional_array(centric_flags, bool),
        }

    def kwargs(self) -> dict:
        out = {"f_obs": self._packed_obs, "target": self.target_spec, "compute_curvature": self.compute_curvature}
        out.update({k: v for k, v in self.arrays.items() if v is not None})
        return out

    def __call__(self, f_calc: Any, compute_gradients: bool = True) -> RemoteTargetResult:
        _ = compute_gradients  # gradients are always computed on the worker
        packed_fc = miller_from_cctbx(f_calc)
        result = self.bridge.call("target_eval", f_calc=packed_fc, **self.kwargs())
        return RemoteTargetResult(result, self.arrays["r_free"])


class RemoteRefinementTarget:
    """One round trip: F_calc, target, dQ/dF and packed gradients for a minimizer."""

    def __init__(
        self,
        bridge: Any,
        xray_structure: Any,
        f_obs: Any,
        target_spec: dict,
        *,
        d_min: Optional[float] = None,
        params: Optional[SfEngineParams] = None,
        table: Optional[str] = None,
        **functor_kwargs: Any,
    ) -> None:
        self.bridge = bridge
        self.xray_structure = xray_structure
        self.functor = RemoteTargetFunctor(bridge, f_obs, target_spec, **functor_kwargs)
        if params is None:
            params = SfEngineParams(d_min=d_min if d_min is not None else float(f_obs.d_min()))
        self.params = params
        self.table = table
        self.last: Optional[dict] = None

    def compute(self, xray_structure: Optional[Any] = None) -> dict:
        xs = xray_structure if xray_structure is not None else self.xray_structure
        xray, table = _packed_xray(xs, self.table)
        out = self.bridge.call("refine_gradients", xray=xray, table=table, params=self.params, **self.functor.kwargs())
        self.last = out
        return out

    def target_and_gradients(self, xray_structure: Optional[Any] = None):
        """(target_work, packed flex.double gradients) for the structure's grad flags."""
        xs = xray_structure if xray_structure is not None else self.xray_structure
        out = self.compute(xs)
        grads = RemoteGradients(xs, out["gradients"])
        return out["target"].meta.value, grads.packed()

    def f_calc(self):
        from phridge.client.convert import miller_to_cctbx

        if self.last is None:
            self.compute()
        return miller_to_cctbx(self.last["f_calc"])

    def target_result(self) -> RemoteTargetResult:
        if self.last is None:
            self.compute()
        return RemoteTargetResult(self.last["target"], self.functor.arrays["r_free"])

    def curvatures(self, xray_structure: Optional[Any] = None) -> PackedSfCurvatures:
        """Exact per-atom Gauss-Newton blocks (SfCurvatures)."""
        xs = xray_structure if xray_structure is not None else self.xray_structure
        out = self.compute(xs)
        xray, table = _packed_xray(xs, self.table)
        return self.bridge.call(
            "gauss_newton_blocks",
            xray=xray,
            table=table,
            params=self.params,
            target=out["target"],
            hkl=_miller_template(self.functor.f_obs),
        )

    def diagonal(
        self,
        xray_structure: Optional[Any] = None,
        *,
        method: str = "blocks",
        n_probes: int = 8,
        seed: int = 0,
        as_inverse: bool = True,
        floor: float = 1e-8,
    ):
        """Packed GN Hessian diagonal in cctbx packing order.

        ``method="blocks"`` uses exact per-atom blocks (site / U* transformed
        to cartesian); ``method="hutchinson"`` uses Rademacher probes.
        When ``as_inverse`` (default) the result is the L-BFGS Hk0 diagonal.
        """
        xs = xray_structure if xray_structure is not None else self.xray_structure
        out = self.compute(xs)
        xray, table = _packed_xray(xs, self.table)
        hkl = _miller_template(self.functor.f_obs)
        if method == "blocks":
            curv = self.bridge.call(
                "gauss_newton_blocks",
                xray=xray,
                table=table,
                params=self.params,
                target=out["target"],
                hkl=hkl,
            )
            packed_curv = _pack_block_diagonal(xs, curv)
        elif method == "hutchinson":
            diag = self.bridge.call(
                "gauss_newton_diagonal",
                xray=xray,
                table=table,
                params=self.params,
                target=out["target"],
                hkl=hkl,
                n_probes=n_probes,
                seed=seed,
            )
            packed_curv = _pack_diagonal_like_gradients(xs, diag)
        else:
            raise ValueError("method must be 'blocks' or 'hutchinson'")
        return _invert_diag(packed_curv, floor=floor) if as_inverse else packed_curv

    def newton_cg(
        self,
        xray_structure: Optional[Any] = None,
        *,
        max_iterations: int = 20,
        cg_max_iter: int = 20,
        damping: float = 1e-3,
        step_max: float = 0.05,
    ) -> dict:
        """Damped Newton–CG on sites: solve (J^T H J + λ D) p = -∇Q.

        Uses ``gauss_newton_hvp`` inside CG and the block diagonal as Jacobi
        preconditioner D. Fractional site updates use backtracking line
        search. Returns diagnostics; ``xray_structure`` on the result is a
        refined deep copy.
        """
        from cctbx.array_family import flex

        xs = (xray_structure if xray_structure is not None else self.xray_structure).deep_copy_scatterers()
        for sc in xs.scatterers():
            sc.flags.set_grad_site(True)
        history = []
        for it in range(int(max_iterations)):
            out = self.compute(xs)
            target = float(out["target"].meta.value)
            grads = RemoteGradients(xs, out["gradients"])
            g_frac = np.asarray(grads.raw.d_site_frac, dtype=np.float64)
            xray, table = _packed_xray(xs, self.table)
            curv = self.bridge.call(
                "gauss_newton_blocks",
                xray=xray,
                table=table,
                params=self.params,
                target=out["target"],
                hkl=_miller_template(self.functor.f_obs),
            )
            D = np.diagonal(curv.site_frac, axis1=1, axis2=2).copy()
            D = np.maximum(D, 1e-8)
            g_flat = g_frac.reshape(-1)
            D_flat = D.reshape(-1)
            n = xs.scatterers().size()

            def hvp_flat(v_flat):
                v_site = v_flat.reshape(n, 3)
                v = PackedSfGradients(
                    d_site_frac=v_site,
                    d_occupancy=np.zeros(n),
                    d_u_iso=np.zeros(n),
                    d_u_star=np.zeros((n, 6)),
                    d_fp=np.zeros(n),
                    d_fdp=np.zeros(n),
                )
                xray, table = _packed_xray(xs, self.table)
                hv = self.bridge.call(
                    "gauss_newton_hvp",
                    xray=xray,
                    table=table,
                    params=self.params,
                    target=out["target"],
                    hkl=_miller_template(self.functor.f_obs),
                    v=v,
                )
                return np.asarray(hv.d_site_frac, dtype=np.float64).reshape(-1) + float(damping) * D_flat * v_flat

            p_flat = _preconditioned_cg(hvp_flat, -g_flat, D_flat, max_iter=int(cg_max_iter))
            p = p_flat.reshape(n, 3)
            step_norm = float(np.linalg.norm(p))
            if step_norm > step_max:
                p *= step_max / max(step_norm, 1e-300)
                step_norm = float(np.linalg.norm(p))

            sites0 = np.asarray(list(xs.sites_frac()), dtype=np.float64).reshape(n, 3)
            alpha = 1.0
            accepted = False
            new_target = target
            for _ in range(8):
                trial = sites0 + alpha * p
                xs.set_sites_frac(flex.vec3_double([tuple(r) for r in trial]))
                new_target = float(self.compute(xs)["target"].meta.value)
                if new_target < target:
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                xs.set_sites_frac(flex.vec3_double([tuple(r) for r in sites0]))
                history.append({"iteration": it, "target": target, "step_norm": 0.0, "alpha": 0.0})
                break
            history.append(
                {"iteration": it, "target": target, "step_norm": step_norm * alpha, "alpha": alpha, "target_after": new_target}
            )
            if step_norm * alpha < 1e-6 or new_target < 1e-12:
                break
        self.xray_structure = xs
        final = self.compute(xs)
        return {
            "xray_structure": xs,
            "target": float(final["target"].meta.value),
            "history": history,
            "n_iterations": len(history),
        }


def _pack_block_diagonal(xray_structure: Any, curv: PackedSfCurvatures):
    """Pack per-atom GN block diagonals in cartesian packing order."""
    from cctbx import adptbx

    flex = _flex()
    xs = xray_structure
    uc = xs.unit_cell()
    F = np.asarray(uc.fractionalization_matrix(), dtype=np.float64).reshape(3, 3)
    scatterers = xs.scatterers()
    any_fp = any(sc.flags.grad_fp() for sc in scatterers)
    any_fdp = any(sc.flags.grad_fdp() for sc in scatterers)
    out: list[float] = []
    for i, sc in enumerate(scatterers):
        fl = sc.flags
        if fl.grad_site():
            H_cart = F.T @ curv.site_frac[i] @ F
            out.extend(np.diag(H_cart).tolist())
        if fl.use_u_iso() and fl.grad_u_iso():
            out.append(float(curv.u_iso[i]))
        if fl.use_u_aniso() and fl.grad_u_aniso():
            J = np.zeros((6, 6), dtype=np.float64)
            for mu in range(6):
                e = [0.0] * 6
                e[mu] = 1.0
                J[:, mu] = adptbx.grad_u_star_as_u_cart(uc, tuple(e))
            H_cart = J @ curv.u_star[i] @ J.T
            out.extend(np.diag(H_cart).tolist())
        if fl.grad_occupancy():
            out.append(float(curv.occupancy[i]))
        if any_fp:
            out.append(float(curv.fp[i]))
        if any_fdp:
            out.append(float(curv.fdp[i]))
    return flex.double(out)


def _preconditioned_cg(hvp, b, M_diag, max_iter=20, tol=1e-6):
    """Solve H p = b with Jacobi-preconditioned CG; M_diag ≈ diag(H)."""
    x = np.zeros_like(b)
    r = b.copy()
    z = r / M_diag
    p = z.copy()
    rz = float(np.dot(r, z))
    bnorm = max(float(np.linalg.norm(b)), 1e-300)
    for _ in range(max_iter):
        Hp = hvp(p)
        denom = float(np.dot(p, Hp))
        if abs(denom) < 1e-300:
            break
        alpha = rz / denom
        x = x + alpha * p
        r = r - alpha * Hp
        if float(np.linalg.norm(r)) / bnorm < tol:
            break
        z = r / M_diag
        rz_new = float(np.dot(r, z))
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x


def _optional_array(value: Any, dtype) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.astype(dtype)
    return np.asarray(list(value), dtype=dtype)
