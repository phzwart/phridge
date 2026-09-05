"""Phenix/cctbx-side access to ``ml_i_maps`` (no torch import here).

Example::

    from phridge.client import Bridge
    from phridge.contrib.intensity_ll.client import RemoteIntensityMaps

    maps = RemoteIntensityMaps(bridge, i_obs, {"name": "ml_i", "nu": 6.0},
                               alpha=sigma_a_array, beta=sigma_wilson_array,
                               r_free_flags=flags.data())
    result = maps(f_calc)              # f_calc: cctbx complex miller array
    result.difference                  # miller array: (<F m> - D|Fc|) e^{i phi_c}
    result.model                       # (2<F m> - D|Fc|) e^{i phi_c}
    result.newton                      # gradient / curvature (Newton-step map)
    result.gradient                    # d log L / d F_c^*
    result.robust_weight               # flex.double <lambda | I_obs>  (Student-t only)
    result.d_loglik_d_nu_total         # scalar, for a nu refinement step
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from phridge.contrib.intensity_ll.ops import OP_NAME, register_ops
from phridge.sfcalc.client import RemoteTargetFunctor, _flex


class RemoteIntensityMapResult:
    """Map coefficient sets as cctbx miller arrays plus per-reflection weights."""

    def __init__(self, raw: dict[str, Any]) -> None:
        from phridge.client.convert import miller_to_cctbx

        self.raw = raw

        def _miller(value: Any) -> Any:  # packed (default) or already-converted cctbx array
            return miller_to_cctbx(value) if hasattr(value, "pack") else value

        self.difference = _miller(raw["difference"])
        self.model = _miller(raw["model"])
        self.gradient = _miller(raw["gradient"])
        self.newton = _miller(raw["newton"])
        self.stats: dict[str, Any] = dict(raw["stats"])

    def _flex_double(self, key: str):
        return _flex().double(np.asarray(self.raw[key], dtype=np.float64).tolist())

    @property
    def fom(self):
        return self._flex_double("fom")

    @property
    def robust_weight(self):
        return self._flex_double("robust_weight")

    @property
    def curvature(self):
        return self._flex_double("curvature")

    @property
    def f_post(self):
        """Posterior mean amplitude <F> (model-conditioned, robust French-Wilson analogue)."""
        return self._flex_double("f_post")

    @property
    def d_loglik_d_nu(self):
        return self._flex_double("d_loglik_d_nu")

    @property
    def d_loglik_d_nu_total(self) -> Optional[float]:
        """Sum over reflections of d log L / d nu (None for normal noise)."""
        if self.stats.get("nu") is None:
            return None
        return float(np.asarray(self.raw["d_loglik_d_nu"], dtype=np.float64).sum())

    def as_numpy(self, key: str) -> np.ndarray:
        value = self.raw[key]
        return np.asarray(value.data if hasattr(value, "data") else value)


class RemoteIntensityMaps(RemoteTargetFunctor):
    """Same constructor as :class:`RemoteTargetFunctor` (with an ``ml_i`` spec) plus ``maps`` options."""

    def __init__(
        self,
        bridge: Any,
        f_obs: Any,
        target_spec: dict,
        *,
        maps: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        register_ops()
        if target_spec.get("name") != "ml_i":
            raise ValueError("RemoteIntensityMaps needs an ml_i target spec")
        super().__init__(bridge, f_obs, target_spec, **kwargs)
        self.maps = dict(maps or {})

    def __call__(self, f_calc: Any, compute_gradients: bool = True) -> RemoteIntensityMapResult:  # type: ignore[override]
        from phridge.client.convert import miller_from_cctbx

        kw = self.kwargs()
        kw.pop("compute_curvature", None)
        kw["maps"] = self.maps
        raw = self.bridge.call(OP_NAME, f_calc=miller_from_cctbx(f_calc), **kw)
        return RemoteIntensityMapResult(raw)
