"""Intensity likelihood engine and fmodel interface for CCTBX and Phenix.

Architecture Overview
=====================
This module provides a pure-CCTBX client interface (`IntensityFModel`) that seamlessly
plugs into ``phenix.refine`` and ``mmtbx`` while delegating intensive reciprocal-space
operations to a PyTorch worker daemon via the Phridge connector:

                    +------------------------------------------+
                    |             IntensityFModel              |
                    |  (inherits from mmtbx.f_model.manager)   |
                    +------------------------------------------+
                                         |
     +--------------------+--------------+-------------+---------------------+
     |                    |                            |                     |
     v                    v                            v                     v
+------------------+ +------------------+ +--------------------+ +-------------------+
| CctbxCompatMixin | |IntensityScaleMix.| |IntensityRValuesMix.| |    Core Engine    |
|------------------| |------------------| |--------------------| |-------------------|
| • Solvent masks  | | • Analytical LS  | | • Posterior <F> R  | | • f_calc()        |
|   (f_masks)      | |   scale (k1)     | | • MAP mode F R     | | • f_model()       |
| • k_sol, B_sol   | | • Counts <-> e-  | | • Direct I R-value | | • target_and_     |
| • Twinning guards| | • k_anisotropic  | | • Work / free CCs  | |   gradients()     |
| • mmtbx facade   | | • Initial scale  | | • r_factors()      | | • update_all_     |
|   properties     | |   estimation     | |   presentation     | |   scales()        |
+------------------+ +------------------+ +--------------------+ +-------------------+
                                                                          |
                                                               (Phridge Bridge / Redis)
                                                                          |
                                                                          v
                                                               +---------------------+
                                                               |   PyTorch Worker    |
                                                               |---------------------|
                                                               | • Structure factors |
                                                               | • Student-t NLL     |
                                                               | • Autograd dQ/dFc*  |
                                                               | • Hessian precond.  |
                                                               | • Nuisance fitting  |
                                                               +---------------------+
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from cctbx import crystal, miller, xray
    from cctbx.array_family import flex
    from libtbx import group_args
except ImportError:
    flex = None
    group_args = None

try:
    import mmtbx.f_model
    import mmtbx.refinement.targets
    _BaseFModel = mmtbx.f_model.manager
except (ImportError, Exception):
    mmtbx = None
    _BaseFModel = object


def register_mli_targets() -> None:
    """Register 'mli_quad' and 'mli' in mmtbx.refinement.targets.target_names."""
    if mmtbx is not None and hasattr(mmtbx, "refinement") and hasattr(mmtbx.refinement, "targets"):
        tn = mmtbx.refinement.targets.target_names
        if "mli_quad" not in tn:
            if "mli" in tn:
                tn["mli_quad"] = tn["mli"]
            else:
                tn["mli_quad"] = mmtbx.refinement.targets.target_attributes(family="ml", specialization="i")
        if "mli" not in tn:
            tn["mli"] = tn["mli_quad"]


class IntensityTwinningError(RuntimeError, ValueError):
    """Raised when twinning is requested with intensity-based likelihood refinement.

    Twinning is not supported for intensity-based likelihood targets ('mli_quad' / 'mli').
    """
    pass


class IntensityDataError(RuntimeError, ValueError):
    """Raised when genuine I_obs cannot be obtained without French–Wilson or F² reconstruction.

    ``mli_quad`` requires observed intensities (including negatives). Converting amplitudes
    via ``as_intensity_array()`` / ``F²`` or French–Wilson scaling is disabled.
    """
    pass


def require_intensity_array(arr: Any, *, context: str = "mli_quad") -> Any:
    """Return ``arr`` if it is an X-ray intensity Miller array; otherwise raise."""
    if arr is None:
        raise IntensityDataError(
            f"{context}: observed intensities (I_obs) are required; got None."
        )
    is_i = getattr(arr, "is_xray_intensity_array", None)
    if callable(is_i) and is_i():
        return arr
    is_i2 = getattr(arr, "is_x_ray_intensity_array", None)
    if callable(is_i2) and is_i2():
        return arr
    raise IntensityDataError(
        f"{context}: observed intensities (I_obs) are required. "
        "Amplitude-only arrays and F² reconstruction are disabled "
        "(French–Wilson / as_intensity_array)."
    )


def recover_i_obs_from_fmodel(fmodel: Any, *, context: str = "mli_quad") -> Any:
    """Recover genuine I_obs from an fmodel / IntensityFModel without F² reconstruction."""
    i_obs = getattr(fmodel, "_i_obs", None)
    if i_obs is not None:
        return require_intensity_array(i_obs, context=context)

    i_fn = getattr(fmodel, "i_obs", None)
    if callable(i_fn):
        try:
            cand = i_fn()
            if cand is not None:
                return require_intensity_array(cand, context=context)
        except IntensityDataError:
            raise
        except Exception:
            pass

    origin_fn = getattr(fmodel, "origin", None)
    if callable(origin_fn):
        try:
            orig = origin_fn()
        except Exception:
            orig = None
        if orig is not None:
            try:
                mtz_ds = getattr(orig, "mtz_dataset", None)
                if callable(mtz_ds):
                    mtz_ds = mtz_ds()
                mtz_obj = None
                if mtz_ds is not None and hasattr(mtz_ds, "mtz_object"):
                    mtz_obj = mtz_ds.mtz_object()
                    if callable(mtz_obj):
                        mtz_obj = mtz_obj()
                if mtz_obj is None and hasattr(orig, "as_miller_arrays"):
                    mtz_obj = orig
                if mtz_obj is not None and hasattr(mtz_obj, "as_miller_arrays"):
                    f_obs = fmodel.f_obs() if hasattr(fmodel, "f_obs") else None
                    for arr in mtz_obj.as_miller_arrays():
                        if getattr(arr, "is_xray_intensity_array", lambda: False)():
                            if f_obs is not None:
                                return require_intensity_array(
                                    arr.common_set(f_obs), context=context
                                )
                            return require_intensity_array(arr, context=context)
            except IntensityDataError:
                raise
            except Exception:
                pass

    raise IntensityDataError(
        f"{context}: could not recover genuine I_obs from fmodel "
        "(no _i_obs / intensity MTZ column). "
        "F² reconstruction from F_obs is disabled."
    )


def amplitude_scaffold_from_intensities(i_obs: Any) -> Any:
    """Build a CCTBX amplitude scaffold from I_obs without dropping reflections.

    Uses ``√max(I,0)`` for F values only. Does **not** mutate or filter ``i_obs``.
    Negative intensities become F=0 in the scaffold but remain in ``i_obs``.
    """
    require_intensity_array(i_obs, context="amplitude_scaffold_from_intensities")
    _ensure_flex()
    io = np.asarray(i_obs.data(), dtype=np.float64)
    fo = np.sqrt(np.maximum(io, 0.0))
    if i_obs.sigmas() is not None:
        sig_i = np.asarray(i_obs.sigmas(), dtype=np.float64)
        sig_f = sig_i / (2.0 * np.maximum(fo, 1e-6))
    else:
        sig_f = np.ones_like(fo)
    return i_obs.customized_copy(
        data=flex.double(fo),
        sigmas=flex.double(sig_f),
    ).set_observation_type_xray_amplitude()


from phridge.client.api import Bridge
from phridge.client.convert import crystal_from_cctbx, miller_from_cctbx, miller_to_cctbx
from phridge.client.convert_xtal import scattering_table_from_cctbx, xray_from_cctbx
from phridge.contrib.intensity_ll.client import RemoteIntensityMapResult
from phridge.contrib.intensity_ll.ops import (
    MAPS_OP_NAME,
    NUISANCE_FIT_OP_NAME,
    TARGET_AND_GRADIENTS_OP_NAME,
    register_ops,
)
from phridge.models import SfEngineParams
from phridge.sfcalc.client import RemoteGradients, RemoteTargetResult


def _env_flag_enabled(name: str, default: str = "0") -> bool:
    """True when env var is a common truthy token (1/true/yes/on)."""
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def _precondition_enabled() -> bool:
    """Whether Gauss–Newton diagonal preconditioning of XYZ/occ/ADP grads is on."""
    return _env_flag_enabled("PHRIDGE_PRECONDITION", "0")


def _memory_bridge_preferred(explicit_memory: Optional[bool] = None) -> bool:
    """Prefer in-process Bridge(memory=True) over Redis (avoids socket-read hangs).

    Default **on** (``PHRIDGE_MEMORY=1``). Set ``PHRIDGE_MEMORY=0`` or pass
    ``memory=False`` to force Redis.
    """
    if explicit_memory is not None:
        return bool(explicit_memory)
    raw = os.environ.get("PHRIDGE_MEMORY")
    if raw is None or not str(raw).strip():
        return True  # machine default: in-process
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _make_intensity_bridge(
    *,
    redis_url: Optional[str] = None,
    memory: Optional[bool] = None,
    device: str = "auto",
    timeout: float = 3600.0,
) -> Any:
    """Construct Bridge: memory if preferred and torch available, else Redis.

    When ``PHRIDGE_MEMORY`` is on (default) but torch is missing, raise instead of
    silently connecting to Redis (that caused ConnectionRefused after we stopped
    auto-starting redis-server).
    """
    from phridge.client.api import Bridge

    want_mem = _memory_bridge_preferred(memory)
    has_torch = False
    try:
        import torch  # noqa: F401

        has_torch = True
    except ImportError:
        has_torch = False

    if want_mem:
        if has_torch:
            return Bridge(memory=True, device=device, timeout=timeout)
        raise RuntimeError(
            "mli_quad is configured for in-process Bridge (PHRIDGE_MEMORY=1) but "
            "torch is not importable in this interpreter "
            f"({sys.executable}).\n\n"
            "Fix one of:\n"
            "  1) Install torch into Phenix:  phenix.python -m pip install torch\n"
            "  2) Use Redis worker:           ./run_phenix_intensity.sh --redis ...\n"
            "     (or PHRIDGE_MEMORY=0 with redis-server + phridge-worker running)\n"
        )

    resolved = redis_url or os.environ.get("PHRIDGE_REDIS_URL") or "redis://127.0.0.1:6379/0"
    return Bridge(resolved, timeout=timeout)


def _resolve_intensity_engine(manager: Any) -> Optional[Any]:
    """Return IntensityFModel-like object that exposes target_and_gradients."""
    if manager is None:
        return None
    if hasattr(manager, "target_and_gradients") and hasattr(manager, "bridge"):
        return manager
    eng = getattr(manager, "_intensity_engine", None)
    if eng is not None and hasattr(eng, "target_and_gradients"):
        return eng
    return None


def _ensure_flex():
    if flex is None:
        raise ImportError("cctbx is required to use IntensityFModel")
    return flex


def _packed_xray(xray_structure: Any, table: Optional[str]):
    return xray_from_cctbx(xray_structure), scattering_table_from_cctbx(xray_structure, table)


def _to_miller(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "pack") or hasattr(value, "meta"):
        return miller_to_cctbx(value)
    return value


# ==============================================================================
# SECTION 1: PACKED GRADIENTS & DERIVATIVES CONTAINER
# ==============================================================================

class IntensityGradients(flex.double if flex is not None else object):
    """Packed gradients array (subclass of flex.double) with cctbx component accessors.

    Can be passed directly into ``scitbx.lbfgs.run`` or inspected component-wise
    via ``d_target_d_site_cart()``, ``d_target_d_u_iso()``, etc.
    """

    def __init__(
        self,
        values: Any,
        xray_structure: Any,
        raw: Any,
        *,
        preconditioned: bool = False,
        curvatures: Optional[Any] = None,
    ) -> None:
        if flex is not None and isinstance(values, flex.double):
            super().__init__(values)
        self.xray_structure = xray_structure
        self.raw = raw
        self.preconditioned = bool(preconditioned)
        self.curvatures = curvatures
        self._remote = RemoteGradients(xray_structure, raw)

    def packed(self) -> Any:
        """Return the flat flex.double gradient array."""
        return self

    def d_target_d_site_frac(self) -> Any:
        return self._remote.d_target_d_site_frac()

    def d_target_d_site_cart(self) -> Any:
        return self._remote.d_target_d_site_cart()

    def d_target_d_u_iso(self) -> Any:
        return self._remote.d_target_d_u_iso()

    def d_target_d_u_star(self) -> Any:
        return self._remote.d_target_d_u_star()

    def d_target_d_u_cart(self) -> Any:
        return self._remote.d_target_d_u_cart()

    def d_target_d_occupancy(self) -> Any:
        return self._remote.d_target_d_occupancy()

    def d_target_d_fp(self) -> Any:
        return self._remote.d_target_d_fp()

    def d_target_d_fdp(self) -> Any:
        return self._remote.d_target_d_fdp()


# ==============================================================================
# SECTION 2: BAYESIAN ELECTRON DENSITY MAPS
# ==============================================================================

class IntensityElectronDensityMap:
    """Map module mimicking ``mmtbx.map_tools.electron_density_map``.

    Provides Fourier coefficients and real-space maps for:
      - 2mFo-DFc map equivalent (Bayesian posterior model map)
      - mFo-DFc map equivalent (Bayesian posterior difference map)
      - Gradient difference map (d log L / d F_c^*)
      - Newton step map (gradient / curvature)
      - Inferred R-factors from the posterior mode
    """

    def __init__(
        self,
        fmodel: "IntensityFModel",
        *args: Any,
        newton_damping: float = 0.1,
        include_free: bool = True,
        raw_result: Optional[dict[str, Any]] = None,
        map_calculation_helper: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        self.fmodel = fmodel
        self.mch = map_calculation_helper
        if args:
            first = args[0]
            if isinstance(first, (int, float)):
                newton_damping = float(first)
            else:
                self.mch = first
        self.newton_damping = float(newton_damping)
        self.include_free = bool(include_free)
        if raw_result is not None:
            self._map_res = RemoteIntensityMapResult(raw_result)
        else:
            self._map_res = self._compute_maps()

        scale_fn = getattr(self.fmodel, "scale_k1", None)
        k_val = scale_fn() if callable(scale_fn) else getattr(self.fmodel, "scale_factor", 1.0)
        self.scale_to_electrons = 1.0 / float(k_val) if (k_val is not None and float(k_val) > 0) else 1.0
        k = k_val

        # Scale Fourier coefficients to absolute electron scale (matching f_calc & mmtbx)
        self.difference = self._map_res.difference.customized_copy(
            data=self._map_res.difference.data() * self.scale_to_electrons
        )
        self.model = self._map_res.model.customized_copy(
            data=self._map_res.model.data() * self.scale_to_electrons
        )
        self.gradient = self._map_res.gradient.customized_copy(
            data=self._map_res.gradient.data() * float(k if k is not None else 1.0)
        )
        self.newton = self._map_res.newton.customized_copy(
            data=self._map_res.newton.data() * self.scale_to_electrons
        )
        self.stats = self._map_res.stats
        self.r_values = self._map_res.r_values

    def _get_engine(self) -> Any:
        if hasattr(self.fmodel, "_common_eval_kwargs"):
            return self.fmodel
        if not hasattr(self.fmodel, "_intensity_engine") or self.fmodel._intensity_engine is None:
            i_obs = recover_i_obs_from_fmodel(self.fmodel, context="IntensityElectronDensityMap")
            self.fmodel._intensity_engine = IntensityFModel(
                i_obs=i_obs,
                xray_structure=self.fmodel.xray_structure,
                r_free_flags=self.fmodel.r_free_flags(),
                memory=True,
            )
        eng = self.fmodel._intensity_engine
        eng.update_xray_structure(self.fmodel.xray_structure)
        return eng

    def _compute_maps(self) -> RemoteIntensityMapResult:
        """Invoke ml_i_maps op on the PyTorch worker."""
        eng = self._get_engine()
        fc = eng.f_model_scaled_with_k1() if hasattr(eng, "f_model_scaled_with_k1") else eng.f_model()
        kw = eng._common_eval_kwargs()
        kw["target"] = eng.target_spec
        kw["maps"] = {
            "newton_damping": self.newton_damping,
            "include_free": self.include_free,
        }
        raw = eng.bridge.call(MAPS_OP_NAME, f_calc=fc, **kw)
        return RemoteIntensityMapResult(raw)

    def _base_array(self) -> Any:
        io_fn = getattr(self.fmodel, "i_obs", None)
        if io_fn is not None and callable(io_fn):
            return io_fn()
        return getattr(self.fmodel, "_i_obs", None) or self.fmodel.f_obs()

    @property
    def fom(self) -> Any:
        """Figure of merit m = <E m(E)> / <E> as a miller.array."""
        return self._base_array().customized_copy(
            data=self._map_res.fom,
            sigmas=None,
            observation_type=None,
        )

    @property
    def robust_weight(self) -> Any:
        """Student-t scale mixture weights <lambda | I_obs> as a miller.array."""
        return self._base_array().customized_copy(
            data=self._map_res.robust_weight,
            sigmas=None,
            observation_type=None,
        )

    @property
    def curvature(self) -> Any:
        """Radial curvature -d^2 log L / d|F_c|^2 as a miller.array."""
        return self._base_array().customized_copy(
            data=self._map_res.curvature,
            sigmas=None,
            observation_type=None,
        )

    @property
    def f_post(self) -> Any:
        """Posterior mean amplitude <F> in electrons (matching f_calc)."""
        data = self._map_res.f_post * getattr(self, "scale_to_electrons", 1.0)
        return self._base_array().customized_copy(
            data=data,
            sigmas=None,
            observation_type=None,
        )

    @property
    def f_mode(self) -> Optional[Any]:
        """Posterior mode amplitude F_mode in electrons (matching f_calc)."""
        if self._map_res.f_mode is None:
            return None
        data = self._map_res.f_mode * getattr(self, "scale_to_electrons", 1.0)
        return self._base_array().customized_copy(
            data=data,
            sigmas=None,
            observation_type=None,
        )

    def inferred_r_values(self) -> Dict[str, Any]:
        """Dictionary of inferred R-values and correlation coefficients."""
        return dict(self.r_values)

    def map_coefficients(
        self,
        map_type: str = "2mFo-DFc",
        fill_missing: bool = False,
        fill_missing_f_obs: bool = False,
        acentrics_scale: float = 2.0,
        centrics_pre_scale: float = 1.0,
        **kwargs: Any,
    ) -> Any:
        """Return map coefficients as a cctbx.miller.array."""
        mt = str(map_type).strip()

        # Handle anomalous / non-standard requests gracefully
        if mt.lower() in ("anomalous", "anom", "anomalous-diff", "anomalous-residual", "phaser-sad-llg"):
            return None

        coeffs = None
        is_diff = False

        # 1. Parse via mmtbx.map_names if available
        parsed_via_map_names = False
        if mmtbx is not None and hasattr(mmtbx, "map_names"):
            try:
                mn = mmtbx.map_names(map_name_string=mt)
                if mn.anomalous:
                    return None
                if mn.k == 0:
                    coeffs = self.fmodel.f_calc()
                elif mn.ml_map:
                    if mn.k == 1 and (mn.n == -1 or mn.n == 1):
                        coeffs = self.difference
                        is_diff = True
                    elif mn.k == 2 and (mn.n == -1 or mn.n == 1):
                        coeffs = self.model
                    else:
                        fc = self.fmodel.f_calc()
                        coeffs = self.f_post.phase_transfer(fc)
                else:
                    coeffs = self.model
                parsed_via_map_names = True
                fill_missing = fill_missing or bool(mn.f_obs_filled)
            except Exception:
                parsed_via_map_names = False

        # 2. String matching fallback
        if not parsed_via_map_names:
            mt_clean = mt.lower().replace("_", "").replace("-", "").replace(" ", "")
            if mt_clean in ("2mfodfc", "2fofc", "2mfobsdfmodel", "2mfobsdfcalc"):
                coeffs = self.model
            elif mt_clean in ("mfodfc", "fofc", "mfobsdfmodel", "mfobsdfcalc"):
                coeffs = self.difference
                is_diff = True
            elif mt_clean in ("gradient", "score", "dl_dfc"):
                coeffs = self.gradient
            elif mt_clean in ("newton", "curv_step"):
                coeffs = self.newton
            elif mt_clean in ("fcalc", "fc", "fmodel"):
                coeffs = self.fmodel.f_calc()
            else:
                coeffs = self.model

        if coeffs is None:
            return None

        if kwargs.get("exclude_free_r_reflections", False):
            r_free_flags = self.fmodel.r_free_flags()
            if r_free_flags is not None:
                if getattr(coeffs, "anomalous_flag", lambda: False)():
                    coeffs = coeffs.average_bijvoet_mates()
                if getattr(r_free_flags, "anomalous_flag", lambda: False)():
                    r_free_flags = r_free_flags.average_bijvoet_mates()
                coeffs = coeffs.select(~r_free_flags.data())

        if kwargs.get("merge_anomalous", False) and getattr(coeffs, "anomalous_flag", lambda: False)():
            coeffs = coeffs.average_bijvoet_mates()

        # Apply standard mmtbx acentric weighting to difference maps
        if is_diff:
            centric_flags = coeffs.centric_flags().data()
            w = np.where(np.asarray(centric_flags), float(centrics_pre_scale), float(acentrics_scale))
            coeffs = coeffs.customized_copy(data=coeffs.data() * flex.double(w))

        if fill_missing or fill_missing_f_obs:
            try:
                d_min = float(self.fmodel.d_min) if hasattr(self.fmodel, "d_min") and self.fmodel.d_min is not None else float(coeffs.d_min())
                f_calc_complete = self.fmodel.xray_structure.structure_factors(
                    d_min=d_min,
                    algorithm=getattr(self.fmodel.sfg_params, "algorithm", "fft"),
                ).f_calc()
                coeffs = coeffs.complete_with(other=f_calc_complete, scale=True)
            except Exception:
                pass
        return coeffs

    def fft_map(
        self,
        resolution_factor: Union[float, str] = 0.25,
        symmetry_flags: Optional[Any] = None,
        map_coefficients: Optional[Any] = None,
        other_fft_map: Optional[Any] = None,
        map_type: Optional[str] = None,
        force_anomalous_flag_false: Optional[bool] = None,
        acentrics_scale: float = 2.0,
        centrics_pre_scale: float = 1.0,
        use_all_data: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Synthesize real-space cctbx.miller.fft_map from Fourier coefficients."""
        if isinstance(resolution_factor, str):
            if map_type is None:
                map_type = resolution_factor
            res_factor = 0.25
        else:
            res_factor = float(resolution_factor)

        if map_coefficients is None:
            if map_type is None:
                map_type = "2mFo-DFc"
            map_coefficients = self.map_coefficients(
                map_type=map_type,
                acentrics_scale=acentrics_scale,
                centrics_pre_scale=centrics_pre_scale,
                **kwargs,
            )
        if map_coefficients is None:
            return None
        if force_anomalous_flag_false and getattr(map_coefficients, "anomalous_flag", lambda: False)():
            map_coefficients = map_coefficients.average_bijvoet_mates()
        if not use_all_data:
            work_sel = None
            if hasattr(self.fmodel, "arrays") and hasattr(self.fmodel.arrays, "work_sel"):
                work_sel = self.fmodel.arrays.work_sel
            elif hasattr(self.fmodel, "r_free_flags") and self.fmodel.r_free_flags() is not None:
                work_sel = ~self.fmodel.r_free_flags().data()
            if work_sel is not None:
                try:
                    map_coefficients = map_coefficients.select(work_sel)
                except Exception:
                    pass
        if other_fft_map is None:
            return map_coefficients.fft_map(
                resolution_factor=res_factor,
                symmetry_flags=symmetry_flags,
            )
        else:
            return miller.fft_map(
                crystal_gridding=other_fft_map,
                fourier_coefficients=map_coefficients,
            )

    def two_fofc_map(self, resolution_factor: float = 0.25, **kwargs: Any) -> Any:
        """Direct convenience method for real-space 2mFo-DFc map."""
        fft = self.fft_map(map_type="2mFo-DFc", resolution_factor=resolution_factor, **kwargs)
        if fft is None:
            return None
        return fft.apply_sigma_scaling().real_map_unpadded()

    def fofc_map(self, resolution_factor: float = 0.25, **kwargs: Any) -> Any:
        """Direct convenience method for real-space mFo-DFc map."""
        fft = self.fft_map(map_type="mFo-DFc", resolution_factor=resolution_factor, **kwargs)
        if fft is None:
            return None
        return fft.apply_sigma_scaling().real_map_unpadded()


# ==============================================================================
# SECTION 3: CCTBX MINIMIZER BRIDGE (TARGET FUNCTOR & RESULT)
# ==============================================================================

class IntensityTargetResult(
    mmtbx.refinement.targets.target_result_mixin if (mmtbx is not None and hasattr(mmtbx, "refinement")) else object
):
    """Result object conforming to mmtbx.refinement.targets.target_result.

    Evaluates exact likelihood derivatives w.r.t F_calc on the PyTorch worker via
    Phridge. Atomic XYZ/occ/ADP gradients normally come from CCTBX
    ``structure_factor_gradients_w`` applied to ``d_target_d_f_calc``. When
    ``PHRIDGE_PRECONDITION`` / ``--precondition`` is on, Phenix instead receives
    packed Gauss–Newton–preconditioned atomic gradients from the phridge worker
    (sites, occupancy, U_iso, and U*).
    """

    def __init__(self, manager: Any, remote_result: RemoteTargetResult) -> None:
        self.manager = manager
        self.core_result = remote_result
        self._remote = remote_result
        self._atomic_gradients: Optional[IntensityGradients] = None

    def target_per_reflection(self) -> Any:
        return self._remote.target_per_reflection()

    def target_work(self) -> float:
        return float(self._remote.target_work())

    def target_test(self) -> Optional[float]:
        val = self._remote.target_test()
        return float(val) if val is not None else None

    def target(self) -> float:
        return self.target_work()

    def d_target_d_f_model_work(self) -> Any:
        return self.manager.f_obs_work().array(data=self._remote.gradients_work())

    def d_target_d_f_calc_work(self) -> Any:
        # Target is evaluated on residual-scaled f_model (observation counts).
        # Chain-rule back to F_calc: residual_k1 · k_iso · k_aniso.
        grad = self._remote.gradients_work()
        residual_k = 1.0
        try:
            residual_k = float(self.manager.scale_k1())
            if not (np.isfinite(residual_k) and residual_k > 0):
                residual_k = 1.0
        except Exception:
            residual_k = 1.0
        if hasattr(self.manager, "k_anisotropic_work") and hasattr(self.manager, "k_isotropic_work"):
            try:
                k_ani = self.manager.k_anisotropic_work()
                k_iso = self.manager.k_isotropic_work()
                if k_ani is not None and k_iso is not None:
                    grad = grad * (k_ani * k_iso * residual_k)
                else:
                    grad = grad * residual_k
            except Exception:
                grad = grad * residual_k
        else:
            grad = grad * residual_k
        return self.manager.f_obs_work().array(data=grad)

    # RemoteTargetResult compatibility accessors
    @property
    def raw(self) -> Any:
        return self._remote.raw

    def gradients_work(self) -> Any:
        return self._remote.gradients_work()

    def d_target_d_f_calc(self) -> Any:
        return self._remote.d_target_d_f_calc()

    def derivatives(self) -> Any:
        return self._remote.derivatives()

    def curvatures_work(self) -> Any:
        return self._remote.curvatures_work()

    def scale_factor(self) -> Optional[float]:
        return self._remote.scale_factor()

    def gradients_wrt_atomic_parameters(self, **kwargs: Any) -> Any:
        """Atomic grads for Phenix LBFGS (XYZ / occ / ADP).

        With ``PHRIDGE_PRECONDITION``, return packed Gauss–Newton–preconditioned
        site, occupancy, U_iso, and U* gradients from the intensity worker.
        Otherwise defer to the mmtbx mixin (CCTBX ``structure_factor_gradients_w``
        on ``d_target_d_f_calc``).
        """
        if _precondition_enabled():
            if self._atomic_gradients is not None and self._atomic_gradients.preconditioned:
                return self._atomic_gradients.packed()
            eng = _resolve_intensity_engine(self.manager)
            if eng is None:
                raise RuntimeError(
                    "PHRIDGE_PRECONDITION is set but no IntensityFModel engine is available "
                    "for Gauss–Newton preconditioned XYZ/occ/ADP gradients."
                )
            xs = None
            if hasattr(self.manager, "xray_structure"):
                xs = self.manager.xray_structure
            _, ig = eng.target_and_gradients(xray_structure=xs, precondition=True)
            self._atomic_gradients = ig
            return ig.packed()

        mixin_fn = getattr(super(), "gradients_wrt_atomic_parameters", None)
        if callable(mixin_fn):
            return mixin_fn(**kwargs)

        # No mmtbx mixin (unit tests / standalone): packed un-preconditioned grads.
        eng = _resolve_intensity_engine(self.manager)
        if eng is None:
            raise RuntimeError("Cannot compute atomic gradients: no intensity engine on manager")
        xs = getattr(self.manager, "xray_structure", None)
        _, ig = eng.target_and_gradients(xray_structure=xs, precondition=False)
        self._atomic_gradients = ig
        return ig.packed()


class IntensityTargetFunctor:
    """Functor compatible with cctbx minimizers and mmtbx.fmodels (evaluates target on worker)."""

    def __init__(self, fmodel: Any, alpha_beta: Optional[Any] = None) -> None:
        if getattr(fmodel, "is_twin_fmodel_manager", lambda: False)():
            raise IntensityTwinningError(
                "Twinning is not supported for intensity-based likelihood ('mli_quad') refinement."
            )
        twin_law = getattr(fmodel, "twin_law", None)
        if twin_law is not None and twin_law is not False and str(twin_law).strip().lower() not in ("", "none"):
            raise IntensityTwinningError(
                "Twinning is not supported for intensity-based likelihood ('mli_quad') refinement: "
                f"twin_law={twin_law!r} is specified."
            )
        twin = getattr(fmodel, "twin", False)
        if bool(twin):
            raise IntensityTwinningError(
                "Twinning is not supported for intensity-based likelihood ('mli_quad') refinement: "
                "twin is True."
            )
        self.manager = fmodel
        self.fmodel = fmodel
        self.alpha_beta = alpha_beta
        if hasattr(fmodel, "_common_eval_kwargs"):
            self._engine = fmodel
        else:
            if not hasattr(fmodel, "_intensity_engine") or fmodel._intensity_engine is None:
                i_obs = recover_i_obs_from_fmodel(fmodel, context="IntensityTargetFunctor")

                fmodel._intensity_engine = IntensityFModel(
                    i_obs=i_obs,
                    f_obs=fmodel.f_obs(),
                    xray_structure=fmodel.xray_structure,
                    r_free_flags=fmodel.r_free_flags(),
                    origin=getattr(fmodel, "origin", lambda: None)(),
                    redis_url=os.environ.get("PHRIDGE_REDIS_URL"),
                    memory=None,  # PHRIDGE_MEMORY default (on)
                )
            self._engine = fmodel._intensity_engine
            self._engine.update_xray_structure(fmodel.xray_structure)

    def f_obs(self) -> Any:
        return self._engine.i_obs()

    def prepare_for_minimization(self) -> None:
        pass

    def target_function_is_invariant_under_allowed_origin_shifts(self) -> bool:
        return True

    def __call__(
        self,
        f_calc: Optional[Any] = None,
        compute_gradients: bool = True,
        **kwargs: Any,
    ) -> IntensityTargetResult:
        t_start = time.perf_counter()
        if f_calc is None:
            # Likelihood needs F on the I_obs / counts scale. After BSS with
            # apply_back_trace=True, CCTBX often leaves overall scale in residual
            # scale_k1 (not in k_iso); fold that in via f_model_scaled_with_k1.
            if hasattr(self.manager, "f_model_scaled_with_k1"):
                f_calc = self.manager.f_model_scaled_with_k1()
            else:
                f_calc = self.manager.f_model()
        kw = self._engine._common_eval_kwargs()
        kw["target"] = self._engine.target_spec
        kw["compute_curvature"] = bool(compute_gradients)
        label = f"target_eval#{getattr(self._engine, '_mli_eval_count', 0) + 1}"
        if compute_gradients:
            label += "+grad"
        else:
            label += " (line-search)"
        from phridge.client.intensity.heartbeat import mli_heartbeat, maybe_progress_eval

        with mli_heartbeat(label, log=getattr(self.manager, "log", None), announce=False):
            raw = self._engine.bridge.call("target_eval", f_calc=f_calc, **kw)
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        r_free = self._engine.r_free_flags()
        rem = RemoteTargetResult(raw, r_free.data() if r_free is not None else None)

        if not hasattr(self._engine, "_mli_eval_count"):
            self._engine._mli_eval_count = 0
            self._engine._mli_last_target_work = None
            self._engine._mli_total_time_ms = 0.0

        self._engine._mli_eval_count += 1
        eval_idx = self._engine._mli_eval_count
        self._engine._mli_total_time_ms += elapsed_ms

        target_work = float(rem.target_work())
        target_test = rem.target_test()
        target_test_val = float(target_test) if target_test is not None else None

        maybe_progress_eval(
            eval_idx,
            target_work=target_work,
            elapsed_ms=elapsed_ms,
            compute_gradients=compute_gradients,
            log=getattr(self.manager, "log", None),
        )

        delta_str = ""
        if self._engine._mli_last_target_work is not None:
            delta = target_work - self._engine._mli_last_target_work
            delta_str = f" (Δ: {delta:+.6f})"
        self._engine._mli_last_target_work = target_work

        # Stable overall scale (not the residual scale_k1, which can flip if
        # F_model path is inconsistent).
        k_val = float(getattr(self.manager, "scale_factor", 1.0) or 1.0)
        residual_k = None
        try:
            residual_k = float(self.manager.scale_k1())
        except Exception:
            residual_k = None
        nu_val = getattr(self._engine, "nu", None)
        if nu_val is not None:
            if hasattr(nu_val, "numel") and nu_val.numel() > 1:
                nu_str = f"nu={float(nu_val.mean().item()):.1f} (mean)"
            else:
                nu_f = float(nu_val.item()) if hasattr(nu_val, "item") else float(nu_val)
                nu_str = f"nu={nu_f:.1f}"
        else:
            nu_str = "Gaussian"
        test_str = f"{target_test_val:.6f}" if target_test_val is not None else "N/A"
        grad_str = "True" if compute_gradients else "False (line-search)"
        residual_str = f" residual_k1={residual_k:.4f}" if residual_k is not None else ""

        # Opt-in: PHRIDGE_VERBOSE_TARGET=1 (default off — banners flood phenix.refine).
        env_verb = os.environ.get("PHRIDGE_VERBOSE_TARGET", "0").strip().lower()
        if env_verb in ("1", "true", "yes", "on"):
            border = "=" * 80
            line1 = f">>> [mli_quad target #{eval_idx}] Target(work): {target_work:.6f}{delta_str} | Free: {test_str}"
            line2 = f">>>                       Time: {elapsed_ms:.2f} ms (cum: {self._engine._mli_total_time_ms:.1f} ms) | Grad: {grad_str} | Scale k: {k_val:.4f}{residual_str} | {nu_str}"
            msg = f"\n{border}\n{line1}\n{line2}\n{border}"
            out_targets = [sys.stdout]
            mgr_log = getattr(self.manager, "log", None) or getattr(self._engine, "log", None)
            if mgr_log is not None and mgr_log not in out_targets and hasattr(mgr_log, "write"):
                out_targets.append(mgr_log)
            for out in out_targets:
                try:
                    print(msg, file=out)
                    if hasattr(out, "flush"):
                        out.flush()
                except Exception:
                    pass

        return IntensityTargetResult(self.manager, rem)


# ==============================================================================
# SECTION 4: STATISTICAL DISPLAY & REMARK 3 EXPORTER
# ==============================================================================

class IntensityFModelInfo:
    """Statistical summary object mimicking and wrapping mmtbx.f_model.f_model_info.info."""

    def __init__(self, fmodel: Any, **kwargs: Any) -> None:
        self.fmodel = fmodel
        self.target_name = getattr(fmodel, "target_name", "mli_quad")
        self.r_work = fmodel.r_work() if hasattr(fmodel, "r_work") else float("nan")
        self.r_free = fmodel.r_free() if hasattr(fmodel, "r_free") else float("nan")
        self.r_all = fmodel.r_all() if hasattr(fmodel, "r_all") else float("nan")

        self.r_post_work = fmodel.r_post_work() if hasattr(fmodel, "r_post_work") else self.r_work
        self.r_post_free = fmodel.r_post_free() if hasattr(fmodel, "r_post_free") else self.r_free
        self.r_post_all = fmodel.r_post_all() if hasattr(fmodel, "r_post_all") else self.r_all

        self.r_mode_work = fmodel.r_mode_work() if hasattr(fmodel, "r_mode_work") else self.r_work
        self.r_mode_free = fmodel.r_mode_free() if hasattr(fmodel, "r_mode_free") else self.r_free
        self.r_mode_all = fmodel.r_mode_all() if hasattr(fmodel, "r_mode_all") else self.r_all

        if hasattr(fmodel, "r_intensity_work"):
            self.r_intensity_work = fmodel.r_intensity_work()
            self.r_intensity_free = fmodel.r_intensity_free()
            self.r_intensity_all = fmodel.r_intensity_all()
        elif hasattr(fmodel, "_intensity_engine") and fmodel._intensity_engine is not None:
            ie = fmodel._intensity_engine
            self.r_intensity_work = ie.r_intensity_work()
            self.r_intensity_free = ie.r_intensity_free()
            self.r_intensity_all = ie.r_intensity_all()
        else:
            self.r_intensity_work = float("nan")
            self.r_intensity_free = float("nan")
            self.r_intensity_all = float("nan")

        self.cc_work = fmodel.cc_work() if hasattr(fmodel, "cc_work") else float("nan")
        self.cc_free = fmodel.cc_free() if hasattr(fmodel, "cc_free") else float("nan")
        self.cc_intensity_work = fmodel.cc_intensity_work() if hasattr(fmodel, "cc_intensity_work") else float("nan")
        self.cc_intensity_free = fmodel.cc_intensity_free() if hasattr(fmodel, "cc_intensity_free") else float("nan")

        self.target_work = fmodel.target_w() if hasattr(fmodel, "target_w") else float("nan")
        self.target_free = fmodel.target_t() if hasattr(fmodel, "target_t") else float("nan")
        # Prefer stored overall electrons→counts scale; residual scale_k1 is ~1 after BSS fold-in.
        if hasattr(fmodel, "scale_factor") and fmodel.scale_factor is not None:
            self.overall_scale_k1 = float(fmodel.scale_factor)
        elif hasattr(fmodel, "scale_k1"):
            self.overall_scale_k1 = float(fmodel.scale_k1())
        else:
            self.overall_scale_k1 = 1.0

        self._cctbx_info = None
        if mmtbx is not None and hasattr(mmtbx, "f_model") and hasattr(mmtbx.f_model, "f_model_info"):
            try:
                self._cctbx_info = mmtbx.f_model.f_model_info.info(fmodel, **kwargs)
            except Exception:
                pass

    def __getattr__(self, name: str) -> Any:
        if self._cctbx_info is not None and hasattr(self._cctbx_info, name):
            return getattr(self._cctbx_info, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def show_remark_3(self, out: Optional[Any] = None) -> None:
        """Write REMARK 3 refinement header information for PDB export."""
        if out is None:
            out = sys.stdout
        if self._cctbx_info is not None and hasattr(self._cctbx_info, "show_remark_3"):
            self._cctbx_info.show_remark_3(out=out)
            pr = "REMARK   3  "
            print(pr + "PHRIDGE DIRECT INTENSITY LIKELIHOOD (MLI_QUAD).", file=out)
            nu_val = getattr(self.fmodel, "nu", "None")
            print(pr + f" STUDENT-T NU PARAMETER                : {str(nu_val):<8}", file=out)
            print(pr + f" POSTERIOR MEAN R-WORK / R-FREE        : {self.r_post_work:.4f} / {self.r_post_free:.4f}", file=out)
            print(pr + f" POSTERIOR MODE R-WORK / R-FREE        : {self.r_mode_work:.4f} / {self.r_mode_free:.4f}", file=out)
            print(pr + f" DIRECT INTENSITY R-WORK / R-FREE      : {self.r_intensity_work:.4f} / {self.r_intensity_free:.4f}", file=out)
            print(pr, file=out)
        else:
            pr = "REMARK   3  "
            print(pr + "REFINEMENT TARGET : MLI_QUAD", file=out)
            print(pr, file=out)
            print(pr + "FIT TO DATA USED IN REFINEMENT.", file=out)
            print(pr + f" R VALUE     (WORKING + TEST SET) : {self.r_all:.4f}", file=out)
            print(pr + f" R VALUE            (WORKING SET) : {self.r_work:.4f}", file=out)
            print(pr + f" FREE R VALUE                     : {self.r_free:.4f}", file=out)
            print(pr, file=out)
            print(pr + "PHRIDGE DIRECT INTENSITY LIKELIHOOD (MLI_QUAD).", file=out)
            nu_val = getattr(self.fmodel, "nu", "None")
            print(pr + f" STUDENT-T NU PARAMETER                : {str(nu_val):<8}", file=out)
            print(pr + f" POSTERIOR MEAN R-WORK / R-FREE        : {self.r_post_work:.4f} / {self.r_post_free:.4f}", file=out)
            print(pr, file=out)

    def show_rfactors_targets_scales_overall(self, header: Optional[str] = None, out: Optional[Any] = None) -> None:
        if out is None:
            out = sys.stdout
        header_text = f" [{header}]" if header else ""
        print("+" + "-" * 76 + "+", file=out)
        print(f"| Intensity Likelihood Refinement (mli_quad){header_text:<31}|", file=out)
        print("|" + " " * 76 + "|", file=out)
        line_post = f"| Posterior Mean <F>: r_work= {self.r_post_work:6.4f}   r_free= {self.r_post_free:6.4f}   r_all= {self.r_post_all:6.4f}"
        print(f"{line_post:<77}|", file=out)
        line_mode = f"| Posterior Mode:     r_work= {self.r_mode_work:6.4f}   r_free= {self.r_mode_free:6.4f}   r_all= {self.r_mode_all:6.4f}"
        print(f"{line_mode:<77}|", file=out)
        line_ri = f"| Direct Intensity:   r_work= {self.r_intensity_work:6.4f}   r_free= {self.r_intensity_free:6.4f}   r_all= {self.r_intensity_all:6.4f}"
        print(f"{line_ri:<77}|", file=out)
        if np.isfinite(self.cc_work) and np.isfinite(self.cc_free):
            line_cc = f"| Correlation (CC):   cc_work= {self.cc_work:6.4f}   cc_free= {self.cc_free:6.4f}   cc_I_work= {self.cc_intensity_work:6.4f}"
            print(f"{line_cc:<77}|", file=out)
        if np.isfinite(self.target_work) and np.isfinite(self.target_free):
            line_tgt = f"| Target NLL:         target_work= {self.target_work:12.4f}   target_free= {self.target_free:12.4f}"
            print(f"{line_tgt:<77}|", file=out)
        nu_val = getattr(self.fmodel, "nu", "N/A")
        ksol_val, bsol_val = self.fmodel.k_sol_b_sol() if hasattr(self.fmodel, "k_sol_b_sol") else (None, None)
        sol_str = f"   k_sol={ksol_val:.3f}   b_sol={bsol_val:.1f}" if ksol_val is not None and bsol_val is not None else ""
        line_sc = f"| Scale Factor k:     scale_k1= {self.overall_scale_k1:6.4f}{sol_str}   Student-t nu: {str(nu_val):<6}"
        print(f"{line_sc:<77}|", file=out)
        print("+" + "-" * 76 + "+", file=out)

    def show_targets(self, text: str = "Refinement target", out: Optional[Any] = None) -> None:
        if out is None:
            out = sys.stdout
        print(f"{text}: {self.target_name}", file=out)

    def show_all(self, header: str = "", out: Optional[Any] = None) -> None:
        self.show_rfactors_targets_scales_overall(header=header, out=out)


# ==============================================================================
# SECTION 5: CCTBX / PHENIX COMPATIBILITY MIXIN
# ==============================================================================

class CctbxCompatMixin:
    """Provides CCTBX and mmtbx.f_model.manager compatibility methods.

    Encapsulates:
      - Solvent masks (f_masks, f_mask, k_masks, k_mask, k_sol_b_sol, f_bulk)
      - Reflection arrays (i_obs, f_obs, r_free_flags)
      - Unit cell & crystal symmetry accessors
      - Twinning guard properties (hard error if twin is requested)
      - Parameter updating & outlier handling
    """

    def _init_cctbx_scaffolding(
        self,
        f_obs_in: Any,
        mask_params: Any,
        sf_and_grads_accuracy_params: Any,
        alpha_beta_params: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize CCTBX base attributes and dictionary keys for Phenix compatibility."""
        # Always scaffold amplitudes from genuine I_obs (√max(I,0)); never trust
        # French–Wilson / F²-converted F_obs for the working reflection set.
        f_obs_scaffold = amplitude_scaffold_from_intensities(self._i_obs)
        self._f_obs_scaffold = f_obs_scaffold

        # Set CCTBX internal dictionary attributes
        self.xray_structure = self._xray_structure
        self.__dict__["xray_structure"] = self._xray_structure
        self.__dict__["twin"] = False
        self.__dict__["twin_law"] = None
        self.__dict__["twin_fraction"] = None
        self.__dict__["twin_law_str"] = None
        self.__dict__["k_sol"] = None
        self.__dict__["b_sol"] = None
        self.__dict__["b_cart"] = None
        self.__dict__["k_h"] = 0.0
        self.__dict__["b_h"] = 0.0
        self.__dict__["target_name"] = "mli_quad"

        # Parameters
        self.mask_params = mask_params or kwargs.get("mask_params")
        if self.mask_params is None and mmtbx is not None and hasattr(mmtbx, "masks"):
            try:
                self.mask_params = mmtbx.masks.mask_master_params.extract()
            except Exception:
                pass

        self.sf_and_grads_accuracy_params = sf_and_grads_accuracy_params or kwargs.get("sf_and_grads_accuracy_params") or getattr(self, "sf_and_grads_accuracy_params", None)
        if self.sf_and_grads_accuracy_params is None and mmtbx is not None and hasattr(mmtbx, "f_model") and hasattr(mmtbx.f_model, "sf_and_grads_accuracy_master_params"):
            try:
                self.sf_and_grads_accuracy_params = mmtbx.f_model.sf_and_grads_accuracy_master_params.extract()
            except Exception:
                pass
        self.sfg_params = self.sf_and_grads_accuracy_params
        self.alpha_beta_params = alpha_beta_params or kwargs.get("alpha_beta_params")

        # Invoke base class __init__ if mmtbx is present
        if _BaseFModel is not object:
            base_kw = {}
            if hasattr(_BaseFModel, "__init__"):
                import inspect
                arg_names = set(inspect.getfullargspec(_BaseFModel.__init__).args[1:])
                for k, v in kwargs.items():
                    if k in arg_names:
                        base_kw[k] = v
                if "mask_params" in arg_names and self.mask_params is not None:
                    base_kw["mask_params"] = self.mask_params
                if "sf_and_grads_accuracy_params" in arg_names and self.sf_and_grads_accuracy_params is not None:
                    base_kw["sf_and_grads_accuracy_params"] = self.sf_and_grads_accuracy_params
                if "alpha_beta_params" in arg_names and self.alpha_beta_params is not None:
                    base_kw["alpha_beta_params"] = self.alpha_beta_params
            try:
                super().__init__(
                    f_obs=f_obs_scaffold,
                    i_obs=self._i_obs,
                    r_free_flags=self._r_free_flags,
                    xray_structure=self._xray_structure,
                    target_name="mli_quad",
                    **base_kw,
                )
            except Exception:
                pass

    # Reflection arrays
    def i_obs(self) -> Any:
        """Observed intensities miller.array."""
        return self._i_obs

    def f_obs(self) -> Any:
        """Observed amplitudes scaffold on √max(I,0) (CCTBX-compatible; I_obs unchanged)."""
        if hasattr(self, "_f_obs_scaffold") and self._f_obs_scaffold is not None:
            return self._f_obs_scaffold
        res = amplitude_scaffold_from_intensities(self._i_obs)
        try:
            if self.binner is not None:
                res.use_binning(binning=self.binner.binning)
            elif self._i_obs.binner() is not None:
                res.use_binning_of(self._i_obs)
        except Exception:
            pass
        self._f_obs_scaffold = res
        return self._f_obs_scaffold

    def r_free_flags(self) -> Any:
        """Boolean test set flags miller.array."""
        return self._r_free_flags

    # Structure & symmetry
    @property
    def xray_structure(self) -> Any:
        """Current atomic model (xray.structure)."""
        return self._xray_structure

    @xray_structure.setter
    def xray_structure(self, value: Any) -> None:
        self._xray_structure = value
        self._f_calc = None
        self._f_model = None
        self._last_gradients = None
        self._last_maps = None
        self._f_post = None
        self._f_mode = None
        self._r_values = None

    def unit_cell(self) -> Any:
        return self._xray_structure.unit_cell()

    def space_group(self) -> Any:
        return self._xray_structure.space_group()

    def crystal_symmetry(self) -> Any:
        return self._xray_structure.crystal_symmetry()

    # Bulk solvent
    def f_masks(self) -> Any:
        """Bulk solvent mask structure factors (list of miller.array)."""
        if hasattr(self, "arrays") and self.arrays is not None and hasattr(self.arrays, "core"):
            if self.arrays.core is not None and hasattr(self.arrays.core, "f_masks"):
                return self.arrays.core.f_masks
        if _BaseFModel is not object and hasattr(super(), "f_masks"):
            try:
                return super().f_masks()
            except Exception:
                pass
        return []

    def f_mask(self) -> Any:
        """Primary bulk solvent mask structure factor (miller.array)."""
        masks = self.f_masks()
        return masks[0] if (masks is not None and len(masks) > 0) else None

    def k_masks(self) -> Any:
        """Bulk solvent scale factors (list of flex.double)."""
        if hasattr(self, "arrays") and self.arrays is not None and hasattr(self.arrays, "core"):
            if self.arrays.core is not None and hasattr(self.arrays.core, "k_masks"):
                return self.arrays.core.k_masks
        if _BaseFModel is not object and hasattr(super(), "k_masks"):
            try:
                return super().k_masks()
            except Exception:
                pass
        return []

    def k_mask(self) -> Any:
        """Primary bulk solvent scale factor array (flex.double)."""
        kms = self.k_masks()
        return kms[0] if (kms is not None and len(kms) > 0) else None

    def k_sol_b_sol(self) -> Tuple[Optional[float], Optional[float]]:
        """Return (k_sol, b_sol) from k_masks if available."""
        if hasattr(self, "k_sol_b_sol_from_k_mask"):
            try:
                ks, bs = self.k_sol_b_sol_from_k_mask()
                if ks is not None and bs is not None and (ks > 0 or bs > 0):
                    return float(ks), float(bs)
            except Exception:
                pass
        ks = getattr(self, "k_sol", None)
        bs = getattr(self, "b_sol", None)
        if callable(ks):
            try:
                ks = ks()
            except Exception:
                ks = None
        if callable(bs):
            try:
                bs = bs()
            except Exception:
                bs = None
        return (float(ks) if ks is not None else None, float(bs) if bs is not None else None)

    def f_bulk(self) -> Any:
        """Total bulk solvent structure factor F_bulk = sum_i(k_mask_i * F_mask_i)."""
        if hasattr(self, "arrays") and self.arrays is not None and hasattr(self.arrays, "core"):
            if self.arrays.core is not None and hasattr(self.arrays.core, "data"):
                fb_data = getattr(self.arrays.core.data, "f_bulk", None)
                if fb_data is not None:
                    return self._i_obs.customized_copy(data=fb_data)
        f_masks = self.f_masks()
        k_masks = self.k_masks()
        if f_masks and k_masks:
            tot = flex.complex_double(self._i_obs.size(), 0)
            for fm, km in zip(f_masks, k_masks):
                tot += km * fm.data()
            return self._i_obs.customized_copy(data=tot)
        return None

    def remove_outliers(self, log: Optional[Any] = None, use_model: bool = True) -> Any:
        """In intensity likelihood, outliers are handled by Student-t noise without discarding."""
        return self

    # Twinning guards
    def is_twin_fmodel_manager(self) -> bool:
        return False

    @property
    def twin_law(self) -> None:
        return None

    @twin_law.setter
    def twin_law(self, value: Any) -> None:
        if value is not None and value is not False and str(value).strip().lower() not in ("", "none"):
            raise IntensityTwinningError(
                "Twinning is not supported for intensity-based likelihood ('mli_quad') refinement: "
                f"attempted to set twin_law={value!r}"
            )

    @property
    def twin(self) -> bool:
        return False

    @twin.setter
    def twin(self, value: bool) -> None:
        if bool(value):
            raise IntensityTwinningError(
                "Twinning is not supported for intensity-based likelihood ('mli_quad') refinement: "
                "attempted to set twin=True"
            )

    # Target name
    @property
    def target_name(self) -> str:
        return getattr(self, "_target_name", "mli_quad") or "mli_quad"

    @target_name.setter
    def target_name(self, value: str) -> None:
        self.set_target_name(value)

    def set_target_name(self, target_name: str) -> None:
        name = str(target_name).lower()
        if name in ("mli", "mli_quad", "ml_i"):
            self._target_name = "mli_quad"
            if hasattr(self, "target_spec") and isinstance(self.target_spec, dict):
                self.target_spec["name"] = "ml_i"
        else:
            self._target_name = target_name
            if _BaseFModel is not object and hasattr(super(), "set_target_name"):
                try:
                    super().set_target_name(target_name)
                except Exception:
                    pass

    def resolution_filter(self, d_min: float) -> "IntensityFModel":
        """Return a resolution-filtered copy of the intensity model."""
        return IntensityFModel(
            i_obs=self._i_obs.resolution_filter(d_min=d_min),
            xray_structure=self._xray_structure.deep_copy_scatterers(),
            r_free_flags=self._r_free_flags.resolution_filter(d_min=d_min) if self._r_free_flags is not None else None,
            bridge=self.bridge,
            target_spec=dict(self.target_spec),
            nu=self.nu,
            d_min=d_min,
            scale_factor=self.scale_factor,
            n_bins=self.n_bins,
            params=self.params,
            table=self.table,
        )

    def info(self, **kwargs: Any) -> IntensityFModelInfo:
        """Return statistical summary object for phenix.refine and mmtbx reporting."""
        return IntensityFModelInfo(self, **kwargs)


# ==============================================================================
# SECTION 6: INTENSITY SCALING & NORMALIZATION MIXIN
# ==============================================================================

class IntensityScaleMixin:
    """Manages scaling between physical electron units and detector observation counts.

    Encapsulates:
      - Analytical least-squares scale factor k relating F_model to observed data
      - f_model_scaled_with_k1() for likelihood evaluation on the counts scale
      - k_isotropic and k_anisotropic arrays
      - Initial scale estimation on high-resolution data
    """

    def scale_k1(self, selection: Optional[Any] = None) -> float:
        """Residual LS scale relating current F_model to observed amplitudes (mmtbx-compatible).

        Pure function: does **not** mutate ``self.scale_factor``. After bulk-solvent /
        scaling, F_model already includes ``k_isotropic * k_anisotropic``, so this
        residual is typically near 1. Use ``self.scale_factor`` for the overall
        electrons→counts scale stored at ``update_all_scales`` time.
        """
        if not getattr(self, "_scale_fitted", False) or self._i_obs is None:
            return float(self.scale_factor)
        fm = self.f_model()
        if fm is None:
            return float(self.scale_factor)
        fo = flex.sqrt(flex.abs(self._i_obs.data()))
        fc = flex.abs(fm.data())
        if selection is not None:
            fo = fo.select(selection)
            fc = fc.select(selection)
        den = flex.sum(fc * fc)
        if den > 0:
            val = float(flex.sum(fo * fc) / den)
            if np.isfinite(val) and val > 0:
                return val
        return float(self.scale_factor)

    def scale_k1_w(self) -> float:
        """Overall scale factor k on work reflections."""
        flags = self.r_free_flags()
        sel = (flags.data() == False) if flags is not None else None
        return self.scale_k1(selection=sel)

    def scale_k1_t(self) -> float:
        """Overall scale factor k on test/free reflections."""
        flags = self.r_free_flags()
        sel = (flags.data() == True) if flags is not None else None
        return self.scale_k1(selection=sel)

    def f_model_scaled_with_k1(self) -> Any:
        """F_model scaled to observed intensity data counts scale (matching mmtbx)."""
        fm = self.f_model()
        k = self.scale_k1()
        return fm.customized_copy(data=fm.data() * k)

    def _fold_residual_scale_into_k_isotropic(self) -> float:
        """Multiply residual LS scale into ``k_isotropic`` so ``f_model()`` is on counts scale.

        Phenix ``apply_back_trace=True`` often resets mean ``k_iso``≈1 while leaving
        overall electrons→counts scale only in residual ``scale_k1``. Intensity
        likelihood needs that scale inside ``f_model()`` (and residual near 1).

        Returns the residual that was applied (1.0 if nothing changed).
        """
        self._scale_fitted = True
        k1 = self.scale_k1()
        if not (np.isfinite(k1) and k1 > 0):
            return 1.0
        if abs(k1 - 1.0) < 1e-6:
            return float(k1)
        if not (hasattr(self, "update_core") and hasattr(self, "arrays") and self.arrays is not None):
            self.scale_factor = float(self.scale_factor) * float(k1)
            self._f_model = None
            return float(k1)
        try:
            k_iso = self.k_isotropic()
            if k_iso is None:
                k_iso = flex.double(self._i_obs.size(), 1.0)
            self.update_core(k_isotropic=k_iso * float(k1))
            self._f_model = None
        except Exception:
            self.scale_factor = float(self.scale_factor) * float(k1)
            self._f_model = None
        return float(k1)

    def f_model_scaled_with_k1_w(self) -> Any:
        """F_model on work reflections scaled to observed data counts scale."""
        fm_w = self.f_model_work() if hasattr(self, "f_model_work") else self.f_model()
        k = self.scale_k1_w()
        return fm_w.customized_copy(data=fm_w.data() * k)

    def f_model_scaled_with_k1_t(self) -> Any:
        """F_model on test/free reflections scaled to observed data counts scale."""
        fm_t = self.f_model_free() if hasattr(self, "f_model_free") else self.f_model()
        k = self.scale_k1_t()
        return fm_t.customized_copy(data=fm_t.data() * k)

    def k_total(self) -> float:
        """Overall scale factor k on F_calc."""
        return self.scale_k1()

    def k_isotropic(self) -> Any:
        """Overall isotropic scale factor array (matching mmtbx)."""
        if hasattr(self, "arrays") and self.arrays is not None and hasattr(self.arrays, "core"):
            if self.arrays.core is not None and hasattr(self.arrays.core, "k_isotropic"):
                return self.arrays.core.k_isotropic
        n = self._i_obs.size()
        return flex.double(n, float(self.scale_factor))

    def k_anisotropic(self) -> Any:
        """Overall anisotropic scale factor array (matching mmtbx)."""
        if hasattr(self, "arrays") and self.arrays is not None and hasattr(self.arrays, "core"):
            if self.arrays.core is not None and hasattr(self.arrays.core, "k_anisotropic"):
                return self.arrays.core.k_anisotropic
        n = self._i_obs.size()
        return flex.double(n, 1.0)

    def _estimate_initial_scale(self) -> None:
        """Estimate analytical least-squares scale relating F_calc to I_obs."""
        if self._f_calc is None or self._i_obs is None:
            return
        fc_abs = np.abs(np.asarray(self._f_calc.data()))
        io = np.asarray(self._i_obs.data())
        fo = np.sqrt(np.maximum(io, 0.0))
        d_sp = np.asarray(self._i_obs.d_spacings().data())
        s_sq = 1.0 / (4.0 * np.maximum(d_sp, 1e-6)**2)

        # Exclude low-resolution bulk solvent region (d > 5 Å, s_sq < 0.04) where
        # solvent cancellation severely depresses I_obs and biases scale downward
        sel = (s_sq > 0.04) & (fc_abs > 1e-4) & (fo > 0.0)
        if np.sum(sel) > 20:
            fo_sub = fo[sel]
            fc_sub = fc_abs[sel]
            den = np.sum(fc_sub**2)
            if den > 0:
                k0 = float(np.sum(fo_sub * fc_sub) / den)
                if np.isfinite(k0) and k0 > 0:
                    self.scale_factor = k0
                    self._scale_fitted = True
                    self._f_model = None
                    if hasattr(self, "update_core") and hasattr(self, "arrays") and self.arrays is not None:
                        try:
                            k_iso = flex.double(self._i_obs.size(), float(self.scale_factor))
                            self.update_core(k_isotropic=k_iso)
                        except Exception:
                            pass
                    return

        den = np.sum(fc_abs**2)
        if den > 0:
            k0 = float(np.sum(fo * fc_abs) / den)
            if np.isfinite(k0) and k0 > 0:
                self.scale_factor = k0
                self._scale_fitted = True
                self._f_model = None
                if hasattr(self, "update_core") and hasattr(self, "arrays") and self.arrays is not None:
                    try:
                        k_iso = flex.double(self._i_obs.size(), float(self.scale_factor))
                        self.update_core(k_isotropic=k_iso)
                    except Exception:
                        pass


# ==============================================================================
# SECTION 7: INFERRED R-VALUES & STATISTICS MIXIN
# ==============================================================================

class IntensityRValuesMixin:
    """Manages Bayesian posterior mode, posterior mean, and direct intensity R-factors.

    Encapsulates:
      - Posterior mean amplitude <F> and posterior mode F_mode accessors
      - Inferred R-factors: r_work, r_free, r_all (from posterior mean)
      - Posterior mode R-factors: r_mode_work, r_mode_free, r_mode_all
      - Direct intensity R-factors: r_intensity_work, r_intensity_free, r_intensity_all
      - Correlation coefficients: cc_work, cc_free, cc_intensity_work, cc_intensity_free
      - Formatted r_factors() output
    """

    @property
    def f_post(self) -> Any:
        """Posterior mean amplitude <F> miller.array in electrons (matching f_calc)."""
        if not hasattr(self, "_f_post") or self._f_post is None:
            self.electron_density_map()
        return self._f_post

    @property
    def f_mode(self) -> Optional[Any]:
        """Posterior mode amplitude F_mode miller.array in electrons (matching f_calc)."""
        if self._f_mode is None:
            self.electron_density_map()
        return self._f_mode

    def inferred_r_values(self) -> Dict[str, Any]:
        """Dictionary of inferred R-values and correlation coefficients."""
        if self._r_values is None:
            self.electron_density_map()
        return dict(self._r_values or {})

    def r_work(self) -> float:
        """Working set R-factor (defaults to inferred posterior mean <F> R_work)."""
        rv = self.inferred_r_values()
        return float(rv.get("r_post_work", rv.get("r_work", float("nan"))))

    def r_free(self) -> float:
        """Free/test set R-factor (defaults to inferred posterior mean <F> R_free)."""
        rv = self.inferred_r_values()
        return float(rv.get("r_post_free", rv.get("r_free", float("nan"))))

    def r_all(self) -> float:
        """Overall R-factor (defaults to inferred posterior mean <F> R_all)."""
        rv = self.inferred_r_values()
        return float(rv.get("r_post_all", rv.get("r_all", float("nan"))))

    def r_post_work(self) -> float:
        """R_work computed from Bayesian posterior mean amplitude <F>."""
        rv = self.inferred_r_values()
        return float(rv.get("r_post_work", float("nan")))

    def r_post_free(self) -> float:
        """R_free computed from Bayesian posterior mean amplitude <F>."""
        rv = self.inferred_r_values()
        return float(rv.get("r_post_free", float("nan")))

    def r_post_all(self) -> float:
        """R_all computed from Bayesian posterior mean amplitude <F>."""
        rv = self.inferred_r_values()
        return float(rv.get("r_post_all", float("nan")))

    def r_mode_work(self) -> float:
        """R_work computed from posterior mode amplitude F_mode."""
        rv = self.inferred_r_values()
        return float(rv.get("r_mode_work", float("nan")))

    def r_mode_free(self) -> float:
        """R_free computed from posterior mode amplitude F_mode."""
        rv = self.inferred_r_values()
        return float(rv.get("r_mode_free", float("nan")))

    def r_mode_all(self) -> float:
        """R_all computed from posterior mode amplitude F_mode."""
        rv = self.inferred_r_values()
        return float(rv.get("r_mode_all", float("nan")))

    def cc_work(self) -> float:
        """Correlation coefficient (F_calc vs F_post) on work reflections."""
        rv = self.inferred_r_values()
        return float(rv.get("cc_post_work", rv.get("cc_work", float("nan"))))

    def cc_free(self) -> float:
        """Correlation coefficient (F_calc vs F_post) on free reflections."""
        rv = self.inferred_r_values()
        return float(rv.get("cc_post_free", rv.get("cc_free", float("nan"))))

    def r_factors(self, prefix: str = "", as_string: bool = True) -> Union[str, Any]:
        """Format inferred R-factors mimicking mmtbx.f_model.manager.r_factors."""
        rv = self.inferred_r_values()
        rw = float(rv.get("r_post_work", rv.get("r_work", float("nan"))))
        rf = float(rv.get("r_post_free", rv.get("r_free", float("nan"))))
        ra = float(rv.get("r_post_all", rv.get("r_all", float("nan"))))
        cw = float(rv.get("cc_post_work", rv.get("cc_work", float("nan"))))
        cf = float(rv.get("cc_post_free", rv.get("cc_free", float("nan"))))
        fmt = "%s r_work=%6.4f r_free=%6.4f r_all=%6.4f cc_work=%6.4f cc_free=%6.4f"
        if as_string:
            return fmt % (prefix, rw, rf, ra, cw, cf)
        if group_args is not None:
            return group_args(r_work=rw, r_free=rf, r_all=ra, cc_work=cw, cc_free=cf)
        return {"r_work": rw, "r_free": rf, "r_all": ra, "cc_work": cw, "cc_free": cf}

    def r_intensity_work(self) -> float:
        """Direct intensity R-factor on work reflections (I_obs vs I_calc)."""
        rv = self.inferred_r_values()
        return float(rv.get("r_intensity_work", float("nan")))

    def r_intensity_free(self) -> float:
        """Direct intensity R-factor on free/test reflections (I_obs vs I_calc)."""
        rv = self.inferred_r_values()
        return float(rv.get("r_intensity_free", float("nan")))

    def r_intensity_all(self) -> float:
        """Direct intensity R-factor on all reflections (I_obs vs I_calc)."""
        rv = self.inferred_r_values()
        return float(rv.get("r_intensity_all", float("nan")))

    def cc_intensity_work(self) -> float:
        """Direct intensity correlation coefficient on work reflections."""
        rv = self.inferred_r_values()
        return float(rv.get("cc_intensity_work", float("nan")))

    def cc_intensity_free(self) -> float:
        """Direct intensity correlation coefficient on free reflections."""
        rv = self.inferred_r_values()
        return float(rv.get("cc_intensity_free", float("nan")))


# ==============================================================================
# SECTION 8: CORE INTENSITY FMODEL ENGINE
# ==============================================================================

class IntensityFModel(
    CctbxCompatMixin,
    IntensityScaleMixin,
    IntensityRValuesMixin,
    _BaseFModel,
):
    """Core intensity likelihood engine and CCTBX fmodel interface.

    Coordinates:
      1. Structure factor evaluation (f_calc) via PyTorch worker daemon.
      2. Bulk solvent and total model calculation (f_model).
      3. Exact likelihood and gradient evaluation (target_and_gradients).
      4. Macrocycle scaling and nuisance parameter fitting (update_all_scales).
    """

    def __init__(
        self,
        i_obs: Optional[Any] = None,
        xray_structure: Optional[Any] = None,
        r_free_flags: Optional[Any] = None,
        *,
        f_obs: Optional[Any] = None,
        bridge: Optional[Any] = None,
        redis_url: Optional[str] = None,
        memory: Optional[bool] = None,
        device: str = "cpu",
        target_spec: Optional[dict] = None,
        target_name: Optional[str] = "mli_quad",
        nu: Optional[float] = None,
        d_min: Optional[float] = None,
        scale_factor: float = 1.0,
        n_bins: int = 10,
        params: Optional[SfEngineParams] = None,
        table: Optional[str] = None,
        sigma_a: Optional[Any] = None,
        sigma_wilson: Optional[Any] = None,
        binner: Optional[Any] = None,
        timeout: float = 3600.0,
        twin: bool = False,
        twin_law: Optional[Any] = None,
        is_twin: bool = False,
        mask_params: Optional[Any] = None,
        sf_and_grads_accuracy_params: Optional[Any] = None,
        alpha_beta_params: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        _ensure_flex()
        register_ops()
        register_mli_targets()

        # Hard error on twinning
        _twin_requested = (
            bool(twin)
            or bool(is_twin)
            or bool(kwargs.get("twin"))
            or bool(kwargs.get("is_twin"))
            or (twin_law is not None and twin_law is not False and str(twin_law).strip().lower() not in ("", "none"))
            or (kwargs.get("twin_law") is not None and kwargs["twin_law"] is not False and str(kwargs["twin_law"]).strip().lower() not in ("", "none"))
        )
        if _twin_requested:
            raise IntensityTwinningError(
                "Twinning is not supported for intensity-based likelihood ('mli_quad') refinement: "
                "twin is True or twin_law was provided."
            )

        # Handle reversed positional arguments (xray_structure, i_obs)
        if i_obs is not None and hasattr(i_obs, "scatterers") and xray_structure is not None and hasattr(xray_structure, "indices"):
            i_obs, xray_structure = xray_structure, i_obs

        # Handle f_obs passed instead of i_obs — amplitudes / F² reconstruction forbidden.
        if i_obs is None and f_obs is not None:
            if hasattr(f_obs, "is_xray_intensity_array") and f_obs.is_xray_intensity_array():
                i_obs = f_obs
            elif hasattr(f_obs, "is_x_ray_intensity_array") and f_obs.is_x_ray_intensity_array():
                i_obs = f_obs
            else:
                raise IntensityDataError(
                    "IntensityFModel requires observed intensities (I_obs). "
                    "Amplitude-only input and F² reconstruction "
                    "(as_intensity_array / French–Wilson) are disabled."
                )

        if i_obs is None:
            raise ValueError("i_obs (or intensity f_obs) must be provided")
        require_intensity_array(i_obs, context="IntensityFModel")
        if xray_structure is None:
            raise ValueError("xray_structure must be provided")

        self.table = table
        self.d_min = float(d_min) if d_min is not None else float(i_obs.d_min())
        self.params = params or SfEngineParams(d_min=self.d_min)

        # Resolution filter if d_min requested
        if d_min is not None and float(i_obs.d_min()) < d_min:
            self._i_obs = i_obs.resolution_filter(d_min=d_min)
        else:
            self._i_obs = i_obs

        # Ensure intensities have valid sigmas
        if self._i_obs.sigmas() is None:
            mean_i = float(flex.mean(self._i_obs.data()))
            self._i_obs.set_sigmas(flex.double(self._i_obs.size(), max(mean_i * 0.1, 1.0)))

        # Align crystal symmetry
        cs = self._i_obs.crystal_symmetry()
        self._xray_structure = xray_structure.customized_copy(crystal_symmetry=cs)

        # R-free flags
        if r_free_flags is not None:
            self._r_free_flags = r_free_flags.common_set(self._i_obs)
        else:
            self._r_free_flags = self._i_obs.generate_r_free_flags(fraction=0.05)

        self._target_name = "mli_quad"

        # Initialize CCTBX scaffolding via mixin
        self._init_cctbx_scaffolding(
            f_obs_in=f_obs,
            mask_params=mask_params,
            sf_and_grads_accuracy_params=sf_and_grads_accuracy_params,
            alpha_beta_params=alpha_beta_params,
            **kwargs,
        )

        # Bridge connector — prefer in-process (memory) to avoid Redis socket hangs.
        if bridge is not None:
            self.bridge = bridge
        else:
            self.bridge = _make_intensity_bridge(
                redis_url=redis_url,
                memory=memory,
                device=device,
                timeout=timeout,
            )

        # Target spec & Student-t nu
        spec = dict(target_spec or {"name": "ml_i"})
        if spec.get("name") not in ("ml_i", "mli_quad"):
            spec["name"] = "ml_i"
        if nu is None:
            env_nu = os.environ.get("PHRIDGE_NU")
            if env_nu is not None and env_nu.strip().lower() not in ("", "none", "null", "false"):
                try:
                    nu = float(env_nu)
                except ValueError:
                    pass
        if nu is not None:
            spec["nu"] = float(nu)
        self.target_spec = spec
        self.nu = float(nu) if nu is not None else spec.get("nu")
        self.scale_factor = float(scale_factor)
        self.n_bins = int(n_bins)

        # Resolution binning & normalization factors
        self.binner = binner or self._i_obs.setup_binner(n_bins=self.n_bins)
        self.sigma_wilson = self._setup_sigma_wilson(sigma_wilson)
        self.sigma_a = self._setup_sigma_a(sigma_a)

        # Internal evaluation caches
        self._f_calc: Optional[Any] = None
        self._f_model: Optional[Any] = None
        self._target_value: float = float("nan")
        self._target_value_test: float = float("nan")
        self._last_gradients: Optional[IntensityGradients] = None
        self._last_maps: Optional[IntensityElectronDensityMap] = None
        self._last_stats_report: Optional[Any] = None
        self._sigma_a_params: Dict[str, float] = {}
        self._sigma_wilson_params: Dict[str, float] = {}
        self._f_post: Optional[Any] = None
        self._f_mode: Optional[Any] = None
        self._r_values: Optional[Dict[str, Any]] = None
        self._scale_fitted: bool = False

        # Seed overall isotropic scale into CCTBX scaffolding when provided
        # explicitly (before update_all_scales replaces it with fitted k_iso/k_aniso).
        if abs(self.scale_factor - 1.0) > 1e-12 and hasattr(self, "update_core") and hasattr(self, "arrays") and self.arrays is not None:
            try:
                k_iso = flex.double(self._i_obs.size(), float(self.scale_factor))
                self.update_core(k_isotropic=k_iso)
                self._scale_fitted = True
            except Exception:
                pass

    def _setup_sigma_wilson(self, user_sw: Optional[Any]) -> Any:
        if user_sw is not None:
            return user_sw
        eps = self._i_obs.epsilons().data().as_double()
        io_data = self._i_obs.data()
        sw = flex.double(self._i_obs.size(), 1.0)
        for i_bin in self.binner.range_used():
            sel = self.binner.selection(i_bin)
            if sel.count(True) > 0:
                i_sel = io_data.select(sel)
                eps_sel = eps.select(sel)
                m = float(flex.mean(i_sel / eps_sel))
                sw.set_selected(sel, max(m, 1e-4))
        return sw

    def _setup_sigma_a(self, user_sa: Optional[Any]) -> Any:
        if user_sa is not None:
            return user_sa
        ss = 1.0 / flex.pow2(self._i_obs.d_spacings().data())
        s_max = float(flex.max(ss)) if ss.size() > 0 else 1.0
        sa = flex.double(0.95 * np.exp(-0.30 * (np.asarray(ss) / max(s_max, 1e-12))))
        return sa

    # ---------------------------------------------------------------- Core SF & Model
    def f_calc(self, xray_structure: Optional[Any] = None) -> Any:
        """Calculated structure factors F_calc in physical electron units (remote)."""
        if xray_structure is not None:
            self.update_xray_structure(xray_structure)
        if self._f_calc is None:
            px, table = _packed_xray(self._xray_structure, self.table)
            hkl = self._i_obs
            raw = self.bridge.call("sf_calc", xray=px, table=table, hkl=hkl, params=self.params)
            self._f_calc = _to_miller(raw)
            if hasattr(self, "update_core") and hasattr(self, "arrays") and self.arrays is not None:
                try:
                    # Only refresh F_calc — never overwrite CCTBX k_isotropic /
                    # k_anisotropic fitted by update_all_scales (that caused
                    # scale ping-pong during XYZ LBFGS).
                    self.update_core(f_calc=self._f_calc)
                except Exception:
                    pass
            if self.scale_factor == 1.0 and not getattr(self, "_scale_fitted", False):
                self._estimate_initial_scale()
        return self._f_calc

    def f_model(self, xray_structure: Optional[Any] = None) -> Any:
        """Total model SF on the observation scale: k_iso·k_aniso·(F_calc+F_bulk)."""
        if xray_structure is not None:
            self.update_xray_structure(xray_structure)
        if self._f_model is not None:
            return self._f_model
        if hasattr(self, "arrays") and self.arrays is not None and getattr(self.arrays, "f_model", None) is not None:
            self._f_model = self.arrays.f_model
            return self._f_model
        fc = self.f_calc()
        fb = self.f_bulk()
        data = fc.data() + fb.data() if fb is not None else fc.data()
        try:
            k_iso = self.k_isotropic()
            k_aniso = self.k_anisotropic()
            if k_iso is not None and k_aniso is not None:
                data = data * k_iso * k_aniso
        except Exception:
            pass
        self._f_model = fc.customized_copy(data=data)
        return self._f_model

    # ---------------------------------------------------------------- Target & Gradients
    def _common_eval_kwargs(self) -> dict[str, Any]:
        return {
            "f_obs": self._i_obs,
            "alpha": self.sigma_a,
            "beta": self.sigma_wilson,
            "epsilon": self._i_obs.epsilons().data().as_double(),
            "centric": self._i_obs.centric_flags().data(),
            "r_free": self._r_free_flags.data() if self._r_free_flags is not None else None,
        }

    def target_and_gradients(
        self,
        xray_structure: Optional[Any] = None,
        *,
        precondition: bool = False,
        damping: float = 0.05,
        scale_factor: Optional[float] = None,
    ) -> Tuple[float, IntensityGradients]:
        """Compute intensity likelihood target and scatterer gradients.

        Parameters
        ----------
        xray_structure : cctbx.xray.structure, optional
            Updated atomic model.
        precondition : bool, default False
            If True (or ``PHRIDGE_PRECONDITION``), applies exact diagonal Gauss-Newton
            curvature preconditioning to site, occupancy, U_iso, and U* gradients.
        damping : float, default 0.05
            Levenberg damping factor for Gauss-Newton preconditioning.
        scale_factor : float, optional
            Scale factor on F_calc (defaults to self.scale_factor).

        Returns
        -------
        target_work : float
            Negative log-likelihood on work reflections.
        gradients : IntensityGradients
            Packed gradients (flex.double subclass) with component accessors.
        """
        if xray_structure is not None:
            self.update_xray_structure(xray_structure)

        if not precondition:
            precondition = _precondition_enabled()

        k_scale = float(scale_factor) if scale_factor is not None else self.scale_factor
        px, table = _packed_xray(self._xray_structure, self.table)
        kw = self._common_eval_kwargs()
        kw.update({
            "xray": px,
            "table": table,
            "params": self.params,
            "target": self.target_spec,
            "precondition": bool(precondition),
            "damping": float(damping),
            "scale_factor": k_scale,
        })
        fb = self.f_bulk()
        if fb is not None:
            kw["f_bulk"] = fb.data()

        t_start = time.perf_counter()
        out = self.bridge.call(TARGET_AND_GRADIENTS_OP_NAME, **kw)
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        # Update cache
        self._f_calc = _to_miller(out["f_calc"])
        if hasattr(self, "update_core") and hasattr(self, "arrays") and self.arrays is not None:
            try:
                self.update_core(f_calc=self._f_calc)
            except Exception:
                pass

        tgt_obj = out.get("target")
        if hasattr(tgt_obj, "meta"):
            self._target_value = float(tgt_obj.meta.value)
            self._target_value_test = float(tgt_obj.meta.value_test) if tgt_obj.meta.value_test is not None else float("nan")
        elif isinstance(tgt_obj, dict):
            meta = tgt_obj.get("meta", tgt_obj)
            self._target_value = float(meta.get("value", meta.get("target_work", float("nan"))))
            self._target_value_test = float(meta.get("value_test", meta.get("target_test", float("nan"))))
        else:
            self._target_value = float(out.get("target_work", float("nan")))
            self._target_value_test = float(out.get("target_test", float("nan")))

        if not hasattr(self, "_mli_tag_count"):
            self._mli_tag_count = 0
            self._mli_tag_total_time_ms = 0.0
            self._mli_tag_last_work = None
        self._mli_tag_count += 1
        self._mli_tag_total_time_ms += elapsed_ms

        delta_str = ""
        if self._mli_tag_last_work is not None:
            delta = self._target_value - self._mli_tag_last_work
            delta_str = f" (Δ: {delta:+.6f})"
        self._mli_tag_last_work = self._target_value

        env_verb = os.environ.get("PHRIDGE_VERBOSE_TARGET", "0").strip().lower()
        if env_verb in ("1", "true", "yes", "on"):
            test_str = f"{self._target_value_test:.6f}" if np.isfinite(self._target_value_test) else "N/A"
            prec_str = f"True (damping={damping})" if precondition else "False"
            nu_val = self.nu
            if nu_val is not None:
                if hasattr(nu_val, "numel") and nu_val.numel() > 1:
                    nu_str = f"nu={float(nu_val.mean().item()):.1f} (mean)"
                else:
                    nu_f = float(nu_val.item()) if hasattr(nu_val, "item") else float(nu_val)
                    nu_str = f"nu={nu_f:.1f}"
            else:
                nu_str = "Gaussian"
            border = "=" * 80
            line1 = f">>> [mli_quad target_and_gradients #{self._mli_tag_count}] Target (work) = {self._target_value:.6f}{delta_str} | Free = {test_str}"
            line2 = f">>>                                        Time: {elapsed_ms:.2f} ms (cum: {self._mli_tag_total_time_ms:.1f} ms) | Precondition: {prec_str} | Scale k: {k_scale:.4f} | {nu_str}"
            msg = f"\n{border}\n{line1}\n{line2}\n{border}"
            print(msg, file=sys.stdout, flush=True)

        raw_grads = out["gradients"]
        curvs = out.get("curvatures") if precondition else None
        rg = RemoteGradients(self._xray_structure, raw_grads)
        packed_vals = rg.packed()

        ig = IntensityGradients(
            packed_vals,
            self._xray_structure,
            raw_grads,
            preconditioned=precondition,
            curvatures=curvs,
        )
        self._last_gradients = ig
        return self._target_value, ig

    def target_w(self, xray_structure: Optional[Any] = None) -> float:
        """Target NLL on work reflections (matching mmtbx.f_model.manager.target_w)."""
        if xray_structure is not None:
            self.update_xray_structure(xray_structure)
        res = self.target_functor()(compute_gradients=False)
        self._target_value = res.target_work()
        if res.target_test() is not None:
            self._target_value_test = res.target_test()
        return self._target_value

    def target_t(self, xray_structure: Optional[Any] = None) -> float:
        """Target NLL on test/free reflections (matching mmtbx.f_model.manager.target_t)."""
        if xray_structure is not None:
            self.update_xray_structure(xray_structure)
        res = self.target_functor()(compute_gradients=False)
        self._target_value = res.target_work()
        if res.target_test() is not None:
            self._target_value_test = res.target_test()
        return self._target_value_test

    def one_time_gradients_wrt_atomic_parameters(self, **kwargs: Any) -> Any:
        """Component-wise gradients (matching mmtbx.f_model.manager)."""
        atomic_flags = {"site", "u_iso", "u_aniso", "occupancy", "selection", "tan_b_iso_max", "u_iso_refinable_params"}
        if any(k in atomic_flags for k in kwargs):
            return self.target_functor()(compute_gradients=True).gradients_wrt_atomic_parameters(**kwargs)
        return self.target_and_gradients(**kwargs)[1]

    def compute_functional_and_gradients(self, precondition: bool = False) -> Tuple[float, Any]:
        """Direct protocol for scitbx.lbfgs.run target evaluator."""
        f, g = self.target_and_gradients(precondition=precondition)
        return f, g.packed()

    def target_functor(self, alpha_beta: Optional[Any] = None) -> IntensityTargetFunctor:
        """Return target functor for cctbx minimizers and mmtbx.fmodels."""
        return IntensityTargetFunctor(self, alpha_beta=alpha_beta)

    # ---------------------------------------------------------------- Map Module
    def electron_density_map(
        self,
        map_calculation_helper: Optional[Any] = None,
        newton_damping: float = 0.1,
        include_free: bool = True,
        **kwargs: Any,
    ) -> IntensityElectronDensityMap:
        """Map module producing Bayesian 2mFo-DFc, mFo-DFc, gradient, and Newton maps."""
        emap = IntensityElectronDensityMap(
            self,
            map_calculation_helper=map_calculation_helper,
            newton_damping=newton_damping,
            include_free=include_free,
            **kwargs,
        )
        self._last_maps = emap
        self._f_post = emap.f_post
        self._f_mode = emap.f_mode
        self._r_values = emap.r_values
        return emap

    def fft_map(self, *args: Any, **kwargs: Any) -> Any:
        """Synthesize real-space cctbx.miller.fft_map from Fourier coefficients."""
        return self.electron_density_map().fft_map(*args, **kwargs)

    def map_calculation_helper(self, *args: Any, **kwargs: Any) -> Any:
        """Lightweight helper compatible with mmtbx map calculations."""
        return None

    def map_coefficients(self, map_type: str = "2mFo-DFc", **kwargs: Any) -> Any:
        """Direct convenience method for Fourier map coefficients."""
        return self.electron_density_map().map_coefficients(map_type=map_type, **kwargs)

    def two_fofc_map(self, resolution_factor: float = 0.25, **kwargs: Any) -> Any:
        """Direct convenience method for real-space 2mFo-DFc map."""
        return self.electron_density_map().two_fofc_map(resolution_factor=resolution_factor, **kwargs)

    def fofc_map(self, resolution_factor: float = 0.25, **kwargs: Any) -> Any:
        """Direct convenience method for real-space mFo-DFc map."""
        return self.electron_density_map().fofc_map(resolution_factor=resolution_factor, **kwargs)

    # ---------------------------------------------------------------- Macrocycle Scale Update
    def update_all_scales(
        self,
        update_f_part1: bool = True,
        apply_back_trace: bool = False,
        params: Any = None,
        nproc: Any = None,
        cycles: Any = None,
        fast: bool = True,
        optimize_mask: bool = False,
        refine_hd_scattering: bool = True,
        refine_hd_scattering_method: str = "fast",
        bulk_solvent_and_scaling: bool = True,
        remove_outliers: bool = True,
        apply_scale_k1_to_f_obs: bool = True,
        show: bool = False,
        verbose: Any = None,
        log: Any = None,
        fit_nu: bool = False,
        fit_scale: bool = True,
        nu_bounds: Tuple[float, float] = (3.0, 30.0),
        **kwargs: Any,
    ) -> Any:
        """Co-refine bulk solvent mask (k_sol, B_sol, k_aniso), sigma_A(s), Sigma_W(s), and scale factor k."""
        if not fit_nu:
            env_fit_nu = os.environ.get("PHRIDGE_FIT_NU")
            if env_fit_nu is not None and env_fit_nu.strip().lower() in ("1", "true", "yes", "on"):
                fit_nu = True

        # 1. Run CCTBX bulk solvent and scaling if enabled
        from phridge.client.intensity.heartbeat import mli_heartbeat

        if bulk_solvent_and_scaling and mmtbx is not None:
            try:
                from mmtbx.bulk_solvent import f_model_all_scales
                self.__dict__["xray_structure"] = self._xray_structure
                self.__dict__["twin"] = False
                self.__dict__["twin_law"] = None
                self.__dict__["twin_fraction"] = None
                self.__dict__["twin_law_str"] = None
                self.__dict__["k_sol"] = None
                self.__dict__["b_sol"] = None
                self.__dict__["b_cart"] = None
                self.__dict__["k_h"] = 0.0
                self.__dict__["b_h"] = 0.0
                self.__dict__["target_name"] = "mli_quad"
                with mli_heartbeat("bulk_solvent_and_scaling", log=log, announce=True):
                    f_model_all_scales.run(
                        fmodel=self,
                        apply_back_trace=apply_back_trace,
                        remove_outliers=False,
                        fast=fast,
                        params=params,
                        refine_hd_scattering=refine_hd_scattering,
                        log=log,
                    )
                if optimize_mask and hasattr(self, "optimize_mask"):
                    self.optimize_mask(out=log)
                try:
                    ks, bs = self.k_sol_b_sol_from_k_mask()
                    if ks is not None and bs is not None:
                        self.k_sol = float(ks)
                        self.b_sol = float(bs)
                except Exception:
                    pass
            except Exception:
                pass

        # 2. Fold residual LS scale into k_isotropic so f_model() is on I_obs counts
        # scale (Phenix apply_back_trace=True otherwise leaves residual_k1≈overall k).
        self._fold_residual_scale_into_k_isotropic()
        k_core = 1.0
        if hasattr(self, "arrays") and self.arrays is not None and hasattr(self.arrays, "core"):
            c = self.arrays.core
            if c is not None and hasattr(c, "k_isotropic") and hasattr(c, "k_anisotropic"):
                k_iso = c.k_isotropic
                k_aniso = c.k_anisotropic
                if k_iso is not None and k_aniso is not None and len(k_iso) > 0 and len(k_aniso) > 0:
                    val = float(np.mean(np.asarray(k_iso) * np.asarray(k_aniso)))
                    if np.isfinite(val) and val > 0:
                        k_core = val
        self.scale_factor = float(k_core)
        self._scale_fitted = True

        # 3. Model structure factors with bulk solvent (now on observation scale)
        f_model = self.f_model()

        # 4. Refine sigma_A(s), Sigma_W(s), and nu on worker
        tune_mask = ~self._r_free_flags.data() if self._r_free_flags is not None else flex.bool(self._i_obs.size(), True)
        sa_mode = os.environ.get("PHRIDGE_SIGMA_A_MODE", "bins").strip().lower() or "bins"
        n_sa_bins_env = os.environ.get("PHRIDGE_SIGMA_A_BINS")
        n_sa_bins = int(n_sa_bins_env) if (n_sa_bins_env and str(n_sa_bins_env).strip().isdigit()) else None
        raw = None
        with mli_heartbeat(
            f"ml_i_nuisance_fit(sigma_a_mode={sa_mode})",
            log=log,
            announce=True,
        ):
            raw = self.bridge.call(
                NUISANCE_FIT_OP_NAME,
                f_calc=f_model,
                f_obs=self._i_obs,
                tune_mask=tune_mask,
                epsilon=self._i_obs.epsilons().data().as_double(),
                centric=self._i_obs.centric_flags().data(),
                fit_nu=bool(fit_nu),
                fit_scale=bool(fit_scale and not bulk_solvent_and_scaling),
                nu_bounds=list(nu_bounds),
                nu=self.nu,
                sigma_a_mode=sa_mode,
                n_sigma_a_bins=n_sa_bins,
            )
        self.sigma_a = flex.double(np.asarray(raw["sigma_a"], dtype=np.float64))
        self.sigma_wilson = flex.double(np.asarray(raw["sigma_wilson"], dtype=np.float64))
        self._sigma_a_params = dict(raw.get("sigma_a_params") or {})
        self._sigma_wilson_params = dict(raw.get("sigma_wilson_params") or {})
        if raw.get("scale_k") is not None and (not bulk_solvent_and_scaling):
            self.scale_factor = float(raw["scale_k"])
        if raw.get("nu") is not None:
            self.nu = float(raw["nu"])
            self.target_spec["nu"] = self.nu
        self._f_model = None
        self._last_maps = None
        self._r_values = None
        self._print_stats_report(log=log, label="after update_all_scales")
        if show:
            self.show(log=log)
        return self

    def _print_stats_report(self, log: Any = None, label: str = "") -> Optional[Any]:
        """Print resolution-binned I/σ + σ_A(s) report (PHRIDGE_STATS_REPORT, default on)."""
        from phridge.client.intensity.stats_report import (
            print_intensity_stats_report,
            report_from_fmodel,
            stats_report_enabled,
        )

        if not stats_report_enabled(default=True):
            return None
        try:
            if getattr(self, "sigma_a", None) is None or getattr(self, "_i_obs", None) is None:
                return None
            if self._i_obs.sigmas() is None:
                return None
            report = report_from_fmodel(self, label=label)
            self._last_stats_report = report
            extras = []
            if log is not None and hasattr(log, "write"):
                extras.append(log)
            print_intensity_stats_report(report, extra_streams=extras)
            return report
        except Exception as exc:
            try:
                print(f"[mli_quad stats report skipped: {exc}]", file=sys.stderr)
            except Exception:
                pass
            return None

    def stats_report(self, *, bin_size: Optional[int] = None, label: str = "") -> Any:
        """Compute (and return) the resolution-binned intensity / σ_A report."""
        from phridge.client.intensity.stats_report import report_from_fmodel

        return report_from_fmodel(self, bin_size=bin_size, label=label)

    # ---------------------------------------------------------------- Updates & State
    def update(
        self,
        f_calc: Any = None,
        f_obs: Any = None,
        f_mask: Any = None,
        r_free_flags: Any = None,
        f_part1: Any = None,
        k_mask: Any = None,
        k_anisotropic: Any = None,
        sf_and_grads_accuracy_params: Any = None,
        target_name: Any = None,
        abcd: Any = None,
        alpha_beta_params: Any = None,
        twin_fraction: Any = None,
        xray_structure: Any = None,
        epsilons: Any = None,
        mask_params: Any = None,
        **kwargs: Any,
    ) -> "IntensityFModel":
        """Update model parameters, target names, or atomic structures without disrupting intensity engine."""
        if twin_fraction is not None and float(twin_fraction) > 0:
            raise IntensityTwinningError(
                "Twinning is not supported for intensity-based likelihood ('mli_quad') refinement: "
                f"twin_fraction={twin_fraction}"
            )
        if target_name is not None:
            self.set_target_name(target_name)
        if xray_structure is not None:
            self.update_xray_structure(xray_structure=xray_structure)
        if r_free_flags is not None:
            self._r_free_flags = r_free_flags
            if self.arrays is not None and hasattr(self.arrays, "r_free_flags"):
                self.arrays.r_free_flags = r_free_flags
        if f_obs is not None:
            self._f_obs = f_obs
            if self.arrays is not None and hasattr(self.arrays, "f_obs"):
                self.arrays.f_obs = f_obs
        if mask_params is not None:
            self.mask_params = mask_params
        if sf_and_grads_accuracy_params is not None:
            self.sf_and_grads_accuracy_params = sf_and_grads_accuracy_params
            self.sfg_params = sf_and_grads_accuracy_params
        if alpha_beta_params is not None:
            self.alpha_beta_params = alpha_beta_params
        if epsilons is not None:
            self.epsilons = epsilons
        self._r_values = None
        self._f_calc = None
        self._f_model = None
        self._last_gradients = None
        self._last_maps = None
        return self

    def update_xray_structure(
        self,
        xray_structure: Optional[Any] = None,
        update_f_calc: bool = False,
        update_f_mask: bool = False,
        force_update_f_mask: bool = False,
    ) -> None:
        """Update atomic coordinates and ADPs, resetting cached structure factors and maps."""
        if xray_structure is not None:
            self._xray_structure = xray_structure
            self.xray_structure = xray_structure
            self.__dict__["xray_structure"] = xray_structure
        if _BaseFModel is not object and hasattr(super(), "update_xray_structure"):
            try:
                super().update_xray_structure(
                    xray_structure=xray_structure,
                    update_f_calc=update_f_calc,
                    update_f_mask=update_f_mask,
                    force_update_f_mask=force_update_f_mask,
                )
            except Exception:
                pass
        self._f_calc = None
        self._f_model = None
        self._last_gradients = None
        self._last_maps = None
        self._f_post = None
        self._f_mode = None
        self._r_values = None
        self._structure_factor_gradients_w = None
        if update_f_calc:
            fc = self.f_calc()
            if hasattr(self, "update_core") and hasattr(self, "arrays") and self.arrays is not None:
                try:
                    self.update_core(f_calc=fc)
                except Exception:
                    pass

    def deep_copy(self) -> "IntensityFModel":
        """Return a deep copy of the engine."""
        return IntensityFModel(
            i_obs=self._i_obs.deep_copy(),
            f_obs=self._f_obs.deep_copy() if self._f_obs is not None else None,
            xray_structure=self._xray_structure.deep_copy_scatterers(),
            r_free_flags=self._r_free_flags.deep_copy() if self._r_free_flags is not None else None,
            bridge=self.bridge,
            target_spec=dict(self.target_spec),
            target_name=self._target_name,
            nu=self.nu,
            d_min=self.d_min,
            scale_factor=self.scale_factor,
            n_bins=self.n_bins,
            params=self.params,
            table=self.table,
            sigma_a=self.sigma_a.deep_copy() if hasattr(self.sigma_a, "deep_copy") else self.sigma_a,
            sigma_wilson=self.sigma_wilson.deep_copy() if hasattr(self.sigma_wilson, "deep_copy") else self.sigma_wilson,
            binner=self.binner,
            sf_and_grads_accuracy_params=self.sf_and_grads_accuracy_params,
            mask_params=self.mask_params,
            alpha_beta_params=self.alpha_beta_params,
        )

    def show(
        self,
        out: Optional[Any] = None,
        prefix: str = "",
        suffix: str = "",
        log: Optional[Any] = None,
    ) -> None:
        """Display model statistics mimicking mmtbx.f_model.manager.show."""
        target_out = log if log is not None else (out if out is not None else sys.stdout)
        ksol_val, bsol_val = self.k_sol_b_sol()
        sol_str = f" | k_sol: {ksol_val:.3f} B_sol: {bsol_val:.1f} Å²" if ksol_val is not None and bsol_val is not None else ""
        header = f"{prefix}: " if prefix else ""
        tail = f" {suffix}" if suffix else ""
        print("=" * 65, file=target_out)
        print(f"  {header}Intensity Likelihood Model Engine (ml_i){tail}", file=target_out)
        print(f"  Reflections: {self._i_obs.size()} (d_min: {self.d_min:.2f} Å)", file=target_out)
        print(f"  Scale k: {float(self.scale_factor):.4f}{sol_str} | Student-t nu: {self.nu}", file=target_out)
        rv = self.inferred_r_values()
        rw = rv.get("r_post_work", rv.get("r_work", float("nan")))
        rf = rv.get("r_post_free", rv.get("r_free", float("nan")))
        ra = rv.get("r_post_all", rv.get("r_all", float("nan")))
        cw = rv.get("cc_post_work", rv.get("cc_work", float("nan")))
        cf = rv.get("cc_post_free", rv.get("cc_free", float("nan")))
        print(f"  Inferred R-values (<F>):  r_work={rw:.4f} r_free={rf:.4f} r_all={ra:.4f} cc_work={cw:.4f} cc_free={cf:.4f}", file=target_out)
        if "r_mode_work" in rv and "r_mode_free" in rv:
            print(f"  Posterior Mode R-values: r_work={rv['r_mode_work']:.4f} r_free={rv['r_mode_free']:.4f}", file=target_out)
        if "r_intensity_work" in rv and "r_intensity_free" in rv:
            print(f"  Direct Intensity R (I):  r_work={rv['r_intensity_work']:.4f} r_free={rv['r_intensity_free']:.4f} all={rv.get('r_intensity_all', float('nan')):.4f}", file=target_out)
        print("=" * 65, file=target_out)


# Backward compatibility aliases
IntensityFmodel = IntensityFModel
IntensityLikelihoodEngine = IntensityFModel

__all__ = [
    "IntensityTwinningError",
    "IntensityGradients",
    "IntensityElectronDensityMap",
    "IntensityTargetResult",
    "IntensityTargetFunctor",
    "IntensityFModelInfo",
    "CctbxCompatMixin",
    "IntensityScaleMixin",
    "IntensityRValuesMixin",
    "IntensityFModel",
    "IntensityFmodel",
    "IntensityLikelihoodEngine",
    "register_mli_targets",
]
