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
from phridge.contrib.intensity_ll.omit_windows import OMIT_OP_NAME, OmitWindowStore
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
    def f_mode(self):
        """Posterior mode amplitude F_mode = scale * E_mode (MAP estimate)."""
        if "f_mode" in self.raw and self.raw["f_mode"] is not None:
            return self._flex_double("f_mode")
        return None

    @property
    def r_values(self) -> dict[str, Any]:
        """Raw statistics bag from the worker (S family, direct-intensity R, CCs)."""
        return dict(self.raw.get("r_values", {}))

    def s_post(self, which: str = "work") -> float:
        """``S_post``: expected residual under the posterior (``work``/``free``/``all``)."""
        key = {"work": "s_post_work", "free": "s_post_free", "all": "s_post_all"}.get(which, "s_post_work")
        return float(self.r_values.get(key, float("nan")))

    def s_prior(self, which: str = "work") -> float:
        """``S_prior``: same functional under the prior — the sigma_A no-data floor."""
        key = {"work": "s_prior_work", "free": "s_prior_free", "all": "s_prior_all"}.get(which, "s_prior_work")
        return float(self.r_values.get(key, float("nan")))

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
        maps = dict(self.maps)
        if "bin_size" not in maps:
            try:
                from phridge.client.intensity.stats_report import stats_bin_size

                maps["bin_size"] = stats_bin_size()
            except Exception:
                maps["bin_size"] = 500
        kw["maps"] = maps
        if "d_spacings" not in kw:
            try:
                kw["d_spacings"] = np.asarray(self.f_obs.d_spacings().data(), dtype=np.float64)
            except Exception:
                pass
        raw = self.bridge.call(OP_NAME, f_calc=miller_from_cctbx(f_calc), **kw)
        return RemoteIntensityMapResult(raw)


class RemoteOmitWindowCoefficients:
    """Torch-free client for ``ml_i_omit_windows``."""

    def __init__(
        self,
        bridge: Any,
        f_obs: Any,
        target_spec: dict,
        *,
        omit: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        register_ops()
        if target_spec.get("name") != "ml_i":
            raise ValueError("RemoteOmitWindowCoefficients needs an ml_i target spec")
        self.bridge = bridge
        self.f_obs = f_obs
        self.target_spec = dict(target_spec)
        self.omit = dict(omit or {})
        self._extra = dict(kwargs)

    def __call__(
        self,
        f_model: Any,
        xray: Any,
        table: Any,
        params: Any,
        *,
        k_scale: Any = None,
        residue_ids: Any = None,
    ) -> OmitWindowStore:
        from phridge.client.convert import miller_from_cctbx

        kw = dict(self._extra)
        kw["f_obs"] = miller_from_cctbx(self.f_obs) if hasattr(self.f_obs, "indices") else self.f_obs
        kw["f_model"] = miller_from_cctbx(f_model) if hasattr(f_model, "indices") else f_model
        kw["target"] = self.target_spec
        kw["xray"] = xray
        kw["table"] = table
        kw["params"] = params
        kw["omit"] = self.omit
        if k_scale is not None:
            kw["k_scale"] = k_scale
        if residue_ids is not None:
            kw["residue_ids"] = residue_ids
        raw = self.bridge.call(OMIT_OP_NAME, **kw)
        return OmitWindowStore(
            hkl=np.asarray(raw["hkl"]),
            coef_model=np.asarray(raw["coef_model"]) if raw.get("coef_model") is not None else None,
            coef_difference=np.asarray(raw["coef_difference"]) if raw.get("coef_difference") is not None else None,
            beta_adjust=np.asarray(raw["beta_adjust"]),
            window_meta=dict(raw.get("window_meta") or {}),
            window_ids=np.asarray(raw["window_ids"]),
            atom_to_window=np.asarray(raw["atom_to_window"]),
        )


def __getattr__(name: str) -> Any:
    if name in (
        "IntensityFModel",
        "IntensityFmodel",
        "IntensityLikelihoodEngine",
        "IntensityElectronDensityMap",
        "IntensityGradients",
    ):
        from phridge.client.intensity.engine import (
            IntensityElectronDensityMap,
            IntensityFModel,
            IntensityFmodel,
            IntensityGradients,
            IntensityLikelihoodEngine,
        )

        return {
            "IntensityFModel": IntensityFModel,
            "IntensityFmodel": IntensityFmodel,
            "IntensityLikelihoodEngine": IntensityLikelihoodEngine,
            "IntensityElectronDensityMap": IntensityElectronDensityMap,
            "IntensityGradients": IntensityGradients,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
