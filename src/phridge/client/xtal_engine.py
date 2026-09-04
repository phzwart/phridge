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
from phridge.packing_scattering import PackedSfGradients, PackedTargetResult


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


def _optional_array(value: Any, dtype) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.astype(dtype)
    return np.asarray(list(value), dtype=dtype)
