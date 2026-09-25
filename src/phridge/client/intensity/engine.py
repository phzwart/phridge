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


# Sentinel so a failed French–Wilson conversion caches as None instead of retrying
_FW_UNSET = object()


def _french_wilson_scale_unpatched() -> Any:
    """The real ``cctbx`` French–Wilson, even while the mli_quad disable patch is on.

    The patch keeps FW-converted amplitudes out of the refinement *target*. The
    legacy R is a reporting bridge and is the one sanctioned consumer, so it reaches
    past the patch to the stashed original. Imported lazily: ``phenix_hook`` imports
    from this module.
    """
    try:
        from phridge.client.intensity import phenix_hook

        orig = phenix_hook._ORIGINALS.get("cctbx.french_wilson.french_wilson_scale")
        if orig is not None:
            return orig
    except Exception:
        pass
    try:
        from cctbx import french_wilson

        return french_wilson.french_wilson_scale
    except ImportError:
        return None


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
from phridge.contrib.intensity_ll.bulk_solvent_op import BULK_SOLVENT_OP_NAME
from phridge.contrib.intensity_ll.client import RemoteIntensityMapResult
from phridge.contrib.intensity_ll.ops import (
    MAPS_OP_NAME,
    NUISANCE_FIT_OP_NAME,
    TARGET_AND_GRADIENTS_OP_NAME,
    register_ops,
)
from phridge.contrib.intensity_ll.wilson import (
    wilson_carries_anisotropy as _wilson_carries_anisotropy,
)
from phridge.models import SfEngineParams
from phridge.sfcalc.client import RemoteGradients, RemoteTargetResult


def _env_flag_enabled(name: str, default: str = "0") -> bool:
    """True when env var is a common truthy token (1/true/yes/on)."""
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    """Non-negative float from the environment; unparseable or negative falls back."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return float(default)
    try:
        value = float(raw)
    except ValueError:
        return float(default)
    return value if value >= 0.0 and value == value else float(default)


def _env_int(name: str) -> Optional[int]:
    """Positive int from the environment, or None if unset / unparseable."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _fit_nu_this_cycle(cycle: int) -> bool:
    """Whether this macro cycle should search ν.

    Default is cycle 1 only: later cycles hold the value already on the engine.
    ``PHRIDGE_FIT_NU_EVERY_CYCLE=1`` searches every cycle. ``PHRIDGE_FIT_NU_CYCLES=N``
    searches the first N cycles.
    """
    if _env_flag_enabled("PHRIDGE_FIT_NU_EVERY_CYCLE", "0"):
        return True
    limit = _env_int("PHRIDGE_FIT_NU_CYCLES") or 1
    return int(cycle) <= int(limit)


def _k_mask_is_negligible(k: Any, thresh: float = 0.05) -> bool:
    """True when k_mask is missing, empty, or everywhere below ``thresh``."""
    if k is None:
        return True
    arr = np.asarray(k, dtype=np.float64).ravel()
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return True
    return float(np.nanmax(np.abs(arr))) < thresh


def _nuisance_detail(
    tune_label: str,
    before: tuple[Optional[float], Optional[float]],
    after: tuple[Optional[float], Optional[float]],
) -> str:
    """Stage detail: which reflections were fitted, and how log(εΣ) moved.

    The NLL on the stage line is already -log p(I). Showing the normalization term
    beside it is what makes a remaining jump readable: the part that is just Σ_W
    changing units is written out, and whatever is left is the fit.
    """
    detail = f"tune={tune_label}"
    if before[0] is not None and after[0] is not None:
        detail += f", log(εΣ) work {before[0]:.3f} -> {after[0]:.3f}"
        if before[1] is not None and after[1] is not None:
            detail += f", free {before[1]:.3f} -> {after[1]:.3f}"
    return detail


def _fmt_k_triple(values: Any) -> str:
    return "/".join(f"{float(x):.2f}" for x in values)


def _bulk_solvent_detail(raw: dict, k_sol_in: Optional[float], b_sol_in: Optional[float]) -> str:
    """Stage detail for the NLL binned k_mask fit, and what it bought.

    Both NLLs are profiled over per-shell σ_A and β, so they compare the solvent curves
    alone; the reference is mmtbx's least-squares k_mask. ``k_sol`` / ``B_sol`` are a
    two-parameter caption of the fitted curve, not the model.
    """
    stats = raw.get("stats") or {}
    n_bins = stats.get("n_bins") or stats.get("n_shells")
    parts: list[str] = []
    if n_bins is not None:
        parts.append(f"k_mask bins={int(n_bins)}")
    k_ls, k_new = stats.get("k_ls"), stats.get("k_new")
    if k_ls is not None and k_new is not None:
        parts.append(f"k(low/mid/high) {_fmt_k_triple(k_ls)} -> {_fmt_k_triple(k_new)}")
    ks, bs = raw.get("k_sol"), raw.get("b_sol")
    if ks is not None and bs is not None:
        start = (
            f"{k_sol_in:.3f}/{b_sol_in:.1f} -> "
            if k_sol_in is not None and b_sol_in is not None
            else ""
        )
        parts.append(f"equiv k_sol/B_sol {start}{float(ks):.3f}/{float(bs):.1f}")
    detail = ", ".join(parts) if parts else "k_mask"
    ref, new = stats.get("nll_fit_ref"), stats.get("nll_fit_new")
    if ref is not None and new is not None:
        detail += f", profiled −log p(Z) vs LS k_mask: work {float(new) - float(ref):+.5f}"
    ref_r, new_r = stats.get("nll_rest_ref"), stats.get("nll_rest_new")
    if ref_r is not None and new_r is not None:
        detail += f", free {float(new_r) - float(ref_r):+.5f}"
    detail += ", installed" if raw.get("accepted") else ", kept LS k_mask"
    failures = stats.get("lbfgs_failures")
    if failures:
        detail += ", LBFGS failed: " + "; ".join(str(item) for item in failures)
    return detail


# Smallest test set worth fitting σ_A on: the worker defaults to ``n_tune // 250``
# resolution bins with a floor of 6, so below this the bins are too sparse to be
# better than the (biased) work-set fit.
_MIN_NUISANCE_TUNE = 1000


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
      - Shrunken-amplitude agreement statistics from the posterior mode
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
        if hasattr(eng, "_likelihood_f_calc"):
            fc = eng._likelihood_f_calc()
        else:
            fc = eng.f_model_scaled_with_k1() if hasattr(eng, "f_model_scaled_with_k1") else eng.f_model()
        kw = eng._common_eval_kwargs()
        kw["target"] = eng.target_spec
        from phridge.client.intensity.stats_report import stats_bin_size

        kw["maps"] = {
            "newton_damping": self.newton_damping,
            "include_free": self.include_free,
            "bin_size": stats_bin_size(),
        }
        try:
            i_obs = eng.i_obs() if callable(getattr(eng, "i_obs", None)) else eng._i_obs
            d_sp = np.asarray(i_obs.d_spacings().data(), dtype=np.float64)
            kw["d_spacings"] = d_sp
        except Exception:
            pass
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
        if hasattr(self.manager, "_likelihood_f_calc"):
            f_calc = self.manager._likelihood_f_calc()
        elif f_calc is None:
            # Likelihood needs F on the I_obs / counts scale. After BSS with
            # apply_back_trace=True, CCTBX often leaves overall scale in residual
            # scale_k1 (not in k_iso); fold that in via f_model_scaled_with_k1.
            if hasattr(self.manager, "f_model_scaled_with_k1"):
                f_calc = self.manager.f_model_scaled_with_k1()
            else:
                f_calc = self.manager.f_model()
        surrogate = bool(getattr(self._engine, "surrogate_active", lambda: False)())
        if surrogate:
            # Inner block of an interleaved macro cycle: stock ml_f fed with the
            # per-reflection surrogate arrays fitted at the last exact checkpoint.
            # No quadrature is evaluated here.
            kw = self._engine._surrogate_eval_kwargs()
        else:
            kw = self._engine._common_eval_kwargs()
            kw["target"] = self._engine.target_spec
        kw["compute_curvature"] = bool(compute_gradients)
        prefix = "surrogate_eval#" if surrogate else "target_eval#"
        count = (
            getattr(self._engine, "_surrogate_eval_count", 0)
            if surrogate
            else getattr(self._engine, "_mli_eval_count", 0)
        )
        label = f"{prefix}{count + 1}"
        if compute_gradients:
            label += "+grad"
        else:
            label += " (line-search)"
        from phridge.client.intensity.heartbeat import mli_heartbeat, maybe_progress_eval

        with mli_heartbeat(label, log=getattr(self.manager, "log", None), announce=False):
            raw = self._engine.bridge.call("target_eval", f_calc=f_calc, **kw)
        if surrogate:
            raw = self._engine._splice_exact_route(raw, f_calc, bool(compute_gradients))
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        r_free = self._engine.r_free_flags()
        rem = RemoteTargetResult(raw, r_free.data() if r_free is not None else None)

        if not hasattr(self._engine, "_mli_eval_count"):
            self._engine._mli_eval_count = 0
            self._engine._mli_last_target_work = None
            self._engine._mli_total_time_ms = 0.0

        if surrogate:
            # Surrogate evaluations are counted separately: the whole point of the
            # interleaved mode is that the exact-evaluation count stops tracking the
            # number of times the minimizer asks for a target.
            self._engine._surrogate_eval_count = (
                getattr(self._engine, "_surrogate_eval_count", 0) + 1
            )
            ctrl = getattr(self._engine, "_interleaved", None)
            if ctrl is not None:
                ctrl.telemetry.n_surrogate_evals += 1
        else:
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

        self.s_post_work = fmodel.s_post_work() if hasattr(fmodel, "s_post_work") else float("nan")
        self.s_post_free = fmodel.s_post_free() if hasattr(fmodel, "s_post_free") else float("nan")
        self.s_post_all = fmodel.s_post_all() if hasattr(fmodel, "s_post_all") else float("nan")
        self.s_prior_work = fmodel.s_prior_work() if hasattr(fmodel, "s_prior_work") else float("nan")
        self.s_prior_free = fmodel.s_prior_free() if hasattr(fmodel, "s_prior_free") else float("nan")
        self.s_prior_all = fmodel.s_prior_all() if hasattr(fmodel, "s_prior_all") else float("nan")

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

    def _cctbx_r_disagreement(self) -> str:
        """Describe any material gap between cctbx's R fields and the FW ones.

        Returns an empty string when they agree, when cctbx exposes nothing to compare,
        or when the only difference is the percent-versus-fraction convention.
        """
        info = self._cctbx_info
        if info is None:
            return ""
        parts = []
        for name in ("r_work", "r_free"):
            theirs = getattr(info, name, None)
            ours = getattr(self, name, None)
            try:
                theirs, ours = float(theirs), float(ours)
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(theirs) and np.isfinite(ours)):
                continue
            candidates = (theirs, theirs / 100.0) if theirs > 1.0 else (theirs,)
            if any(abs(c - ours) <= 5e-4 for c in candidates):
                continue
            parts.append(f"{name.upper()} {theirs:.4f} vs {ours:.4f}")
        return "; ".join(parts)

    def show_remark_3(self, out: Optional[Any] = None) -> None:
        """Write REMARK 3 refinement header information for PDB export."""
        if out is None:
            out = sys.stdout
        if self._cctbx_info is not None and hasattr(self._cctbx_info, "show_remark_3"):
            self._cctbx_info.show_remark_3(out=out)
            pr = "REMARK   3  "
            # Restate the mandated R fields explicitly and unambiguously. The block
            # above is written by cctbx; this one says which amplitudes were used, so
            # nobody has to infer it from the target name.
            #
            # cctbx's info object is expected to have taken these from r_work() /
            # r_free(), which are the French-Wilson bridge. If some version computes
            # them itself it would use fmodel.f_obs() -- the sqrt(max(I,0)) scaffold --
            # and REMARK 3 would then carry two different numbers under one field name.
            # Say so loudly rather than let the wrong one be deposited.
            mismatch = self._cctbx_r_disagreement()
            if mismatch:
                print(pr + "WARNING: THE R VALUES WRITTEN ABOVE DISAGREE WITH THE", file=out)
                print(pr + " FRENCH-WILSON R VALUES BELOW: " + mismatch, file=out)
                print(pr + " TRUST THE FRENCH-WILSON VALUES BELOW; THE ONES ABOVE WERE NOT", file=out)
                print(pr + " COMPUTED FROM FRENCH-WILSON AMPLITUDES. DO NOT DEPOSIT THEM.", file=out)
            print(pr + "R VALUES ABOVE ARE FRENCH-WILSON AMPLITUDE R FACTORS:", file=out)
            print(pr + f" R VALUE     (WORKING + TEST SET) : {self.r_all:.4f}  (FRENCH-WILSON)", file=out)
            print(pr + f" R VALUE            (WORKING SET) : {self.r_work:.4f}  (FRENCH-WILSON)", file=out)
            print(pr + f" FREE R VALUE                     : {self.r_free:.4f}  (FRENCH-WILSON)", file=out)
            print(pr + " F_OBS FOR THESE R VALUES = FRENCH-WILSON POSTERIOR AMPLITUDES FROM", file=out)
            print(pr + " I_OBS UNDER THE WILSON PRIOR ALONE (NO MODEL). THE REFINEMENT TARGET", file=out)
            print(pr + " ITSELF NEVER FORMS F_OBS: IT IS THE MARGINAL INTENSITY LIKELIHOOD.", file=out)
            print(pr, file=out)
            print(pr + "PHRIDGE DIRECT INTENSITY LIKELIHOOD (MLI_QUAD).", file=out)
            nu_val = getattr(self.fmodel, "nu", "None")
            print(pr + f" STUDENT-T NU PARAMETER                : {str(nu_val):<8}", file=out)
            print(pr + f" DIRECT INTENSITY R-WORK / R-FREE      : {self.r_intensity_work:.4f} / {self.r_intensity_free:.4f}", file=out)
            print(pr + f" S_POST  (WORK / FREE)                 : {self.s_post_work:.4f} / {self.s_post_free:.4f}", file=out)
            print(pr + f" S_PRIOR (WORK / FREE)                 : {self.s_prior_work:.4f} / {self.s_prior_free:.4f}", file=out)
            print(pr + " S_POST / S_PRIOR ARE NOT R FACTORS: EXPECTED RESIDUAL UNDER THE", file=out)
            print(pr + " POSTERIOR / UNDER THE PRIOR. DO NOT COMPARE WITH DEPOSITED R.", file=out)
            print(pr + " SHRUNKEN-AMPLITUDE AGREEMENT (BIASED LOW — DIAGNOSTIC, NOT AN R FACTOR):", file=out)
            print(pr + f"  POSTERIOR MEAN <F> (WORK / FREE)     : {self.r_post_work:.4f} / {self.r_post_free:.4f}", file=out)
            print(pr + f"  POSTERIOR MODE     (WORK / FREE)     : {self.r_mode_work:.4f} / {self.r_mode_free:.4f}", file=out)
            print(pr, file=out)
            self._write_aniso_b_remark_3(pr, out)
        else:
            pr = "REMARK   3  "
            print(pr + "REFINEMENT TARGET : MLI_QUAD", file=out)
            print(pr, file=out)
            print(pr + "FIT TO DATA USED IN REFINEMENT.", file=out)
            # The mandated PDB fields carry the French–Wilson amplitude R — the one
            # statistic here that is genuinely an R factor. Labelled in the output,
            # not just in this comment.
            print(pr + f" R VALUE     (WORKING + TEST SET) : {self.r_all:.4f}  (FRENCH-WILSON)", file=out)
            print(pr + f" R VALUE            (WORKING SET) : {self.r_work:.4f}  (FRENCH-WILSON)", file=out)
            print(pr + f" FREE R VALUE                     : {self.r_free:.4f}  (FRENCH-WILSON)", file=out)
            print(pr + " F_OBS FOR THESE R VALUES = FRENCH-WILSON POSTERIOR AMPLITUDES FROM", file=out)
            print(pr + " I_OBS UNDER THE WILSON PRIOR ALONE (NO MODEL). THE REFINEMENT TARGET", file=out)
            print(pr + " ITSELF NEVER FORMS F_OBS: IT IS THE MARGINAL INTENSITY LIKELIHOOD.", file=out)
            print(pr, file=out)
            print(pr + "PHRIDGE DIRECT INTENSITY LIKELIHOOD (MLI_QUAD).", file=out)
            nu_val = getattr(self.fmodel, "nu", "None")
            print(pr + f" STUDENT-T NU PARAMETER                : {str(nu_val):<8}", file=out)
            print(pr + f" S_POST  (WORK / FREE)                 : {self.s_post_work:.4f} / {self.s_post_free:.4f}", file=out)
            print(pr + f" S_PRIOR (WORK / FREE)                 : {self.s_prior_work:.4f} / {self.s_prior_free:.4f}", file=out)
            print(pr + " S_POST / S_PRIOR ARE NOT R FACTORS: EXPECTED RESIDUAL UNDER THE", file=out)
            print(pr + " POSTERIOR / UNDER THE PRIOR. DO NOT COMPARE WITH DEPOSITED R.", file=out)
            print(pr, file=out)
            self._write_aniso_b_remark_3(pr, out)

    def _write_aniso_b_remark_3(self, pr: str, out: Any) -> None:
        """Write the mandated OVERALL ANISOTROPIC B VALUE block.

        cctbx cannot supply this one: ``b_cart`` on the fmodel is ``None`` by
        construction, so a cctbx-written block would carry zeros or nothing at all for a
        field that is supposed to describe the applied scale. The values here are fitted
        from ``k_anisotropic`` itself, and the provenance is stated in the block so a
        depositor is not left guessing.
        """
        from phridge.client.intensity.stats_report import format_aniso_scale_remark_3

        try:
            aniso = self.fmodel.aniso_scale() if hasattr(self.fmodel, "aniso_scale") else None
        except Exception:
            return
        if aniso is None:
            return
        lines = format_aniso_scale_remark_3(aniso, prefix=pr)
        if not lines:
            return
        for line in lines:
            print(line, file=out)
        print(pr + " ANISOTROPIC B FITTED FROM THE APPLIED K_ANISOTROPIC SCALE ARRAY,", file=out)
        print(pr + " WHICH IS WHAT MULTIPLIES F_CALC IN F_MODEL. ANISOTROPY (SPREAD OF", file=out)
        print(pr + f" PRINCIPAL VALUES) : {aniso.anisotropy:.5f} A**2", file=out)
        print(pr, file=out)

    def show_rfactors_targets_scales_overall(self, header: Optional[str] = None, out: Optional[Any] = None) -> None:
        if out is None:
            out = sys.stdout
        header_text = f" [{header}]" if header else ""
        print("+" + "-" * 76 + "+", file=out)
        print(f"| Intensity Likelihood Refinement (mli_quad){header_text:<31}|", file=out)
        print("|" + " " * 76 + "|", file=out)
        # The conventional R factor first, and named for the amplitudes it uses. This
        # is the number that is comparable with deposited values; everything below it
        # on this panel is not.
        line_rfw = (
            f"| R (French-Wilson):  r_work= {self.r_work:6.4f}   r_free= {self.r_free:6.4f}   "
            f"r_all= {self.r_all:6.4f}"
        )
        print(f"{line_rfw:<77}|", file=out)
        line_ri = f"| Direct Intensity R: r_work= {self.r_intensity_work:6.4f}   r_free= {self.r_intensity_free:6.4f}   r_all= {self.r_intensity_all:6.4f}"
        print(f"{line_ri:<77}|", file=out)
        line_spost = (
            f"| S_post:             work= {self.s_post_work:6.4f}   free= {self.s_post_free:6.4f}   "
            f"S_prior_w/f= {self.s_prior_work:6.4f}/{self.s_prior_free:6.4f}"
        )
        print(f"{line_spost:<77}|", file=out)
        rv = {}
        if hasattr(self.fmodel, "inferred_r_values"):
            try:
                rv = self.fmodel.inferred_r_values() or {}
            except Exception:
                rv = {}
        k_w = rv.get("k_s_work", float("nan"))
        n_out = int(rv.get("n_ec_outliers", 0) or 0)
        if np.isfinite(float(k_w)):
            # s_vis (the plug-in numerator) is deliberately not shown here: it is a
            # debugging quantity and reads like an agreement statistic if printed.
            line_k = f"| k_S diagnostics:    k_S= {float(k_w):6.4f}   E_C outliers= {n_out}"
            print(f"{line_k:<77}|", file=out)
        rho2_w = float(rv.get("rho2_work", float("nan")))
        rho2_f = float(rv.get("rho2_free", float("nan")))
        if np.isfinite(rho2_w):
            lat_f = f"{1.0 - rho2_f:6.4f}" if np.isfinite(rho2_f) else "   n/a"
            line_lat = (
                f"| Latent share 1-ρ²:  work= {1.0 - rho2_w:6.4f}   free= {lat_f}"
                "   (descriptive)"
            )
            print(f"{line_lat:<77}|", file=out)
        xi_w = float(rv.get("xi_work", float("nan")))
        if np.isfinite(xi_w):
            xi_f = float(rv.get("xi_free", float("nan")))
            om = float(rv.get("omega", float("nan")))
            xi_f_s = f"{xi_f:6.4f}" if np.isfinite(xi_f) else "   n/a"
            om_s = f"{om:6.4f}" if np.isfinite(om) else "   n/a"
            line_xi = f"| Score test ξ:       work= {xi_w:6.4f}   free= {xi_f_s}   ω= {om_s}"
            print(f"{line_xi:<77}|", file=out)
        rv_err = rv.get("s_report_error")
        if rv_err:
            err_line = f"| S_post error: {str(rv_err)[:60]}"
            print(f"{err_line:<77}|", file=out)
        elif not np.isfinite(self.s_post_work) and "r_intensity_work" in rv and "s_post_work" not in rv:
            hint = "| S_post: worker missing S-statistic code — restart phridge-worker or --memory"
            print(f"{hint:<77}|", file=out)
        self._pending_s_report_debug = rv.get("s_report_debug")
        # Two CCs, two different provenances, and the difference matters more than the
        # values: the posterior one correlates the model against a quantity that was
        # itself shrunk toward the model, so it rises toward 1 as the data weaken.
        # Naming the source on each line is the only thing that keeps them apart.
        if np.isfinite(self.cc_work) and np.isfinite(self.cc_free):
            line_cc = (
                f"| CC posterior <F> vs |F_c| (shrunken): work= {self.cc_work:6.4f}   "
                f"free= {self.cc_free:6.4f}"
            )
            print(f"{line_cc:<77}|", file=out)
        if np.isfinite(self.cc_intensity_work):
            line_cci = (
                f"| CC_I  I_obs vs I_calc (model-free):   work= {self.cc_intensity_work:6.4f}   "
                f"free= {self.cc_intensity_free:6.4f}"
            )
            print(f"{line_cci:<77}|", file=out)
        if np.isfinite(self.target_work) and np.isfinite(self.target_free):
            line_tgt = f"| Target NLL:         target_work= {self.target_work:12.4f}   target_free= {self.target_free:12.4f}"
            print(f"{line_tgt:<77}|", file=out)
        nu_val = getattr(self.fmodel, "nu", "N/A")
        ksol_val, bsol_val = self.fmodel.k_sol_b_sol() if hasattr(self.fmodel, "k_sol_b_sol") else (None, None)
        sol_str = f"   k_sol={ksol_val:.3f}   b_sol={bsol_val:.1f}" if ksol_val is not None and bsol_val is not None else ""
        line_sc = f"| Scale Factor k:     scale_k1= {self.overall_scale_k1:6.4f}{sol_str}   Student-t nu: {str(nu_val):<6}"
        print(f"{line_sc:<77}|", file=out)
        # Overall anisotropic B of the global scale: the PDB REMARK 3 quantity, on the
        # same panel as the other scale parameters it was fitted alongside.
        aniso = self.fmodel.aniso_scale() if hasattr(self.fmodel, "aniso_scale") else None
        if aniso is not None and getattr(aniso, "is_valid", False):
            from phridge.client.intensity.stats_report import b_field

            # b_field, not %7.3f: a symmetry-fixed component comes back as -1e-17 and
            # would otherwise read "-0.000", which looks like a measurement.
            b = [f"{b_field(x, 3):>7}" for x in aniso.b_cart]
            line_b1 = (
                f"| Aniso B (scale) Å²: B11= {b[0]}  B22= {b[1]}  B33= {b[2]}"
                f"  aniso={aniso.anisotropy:7.3f}"
            )
            line_b2 = f"|                     B12= {b[3]}  B13= {b[4]}  B23= {b[5]}"
            # Pad and truncate: a large anisotropy must not push the panel border out.
            print(f"{line_b1:<77.77}|", file=out)
            print(f"{line_b2:<77.77}|", file=out)
        print("+" + "-" * 76 + "+", file=out)
        print(
            "  R (French-Wilson) is the conventional R factor: F_obs are French-Wilson\n"
            "  posterior amplitudes from I_obs under the Wilson prior alone (no model),\n"
            "  so it is model-free on the observation side and comparable with deposited\n"
            "  values. It is what r_work() / r_free() / REMARK 3 report. The refinement\n"
            "  target never forms F_obs at all. Direct Intensity R, S_post and S_prior are\n"
            "  different statistics — do not compare any of them with deposited R.\n"
            "  CC posterior <F> is model-conditioned: <F> is shrunk toward sigma_A E_C, so\n"
            "  the model sits on both sides and the value rises toward 1 as the data weaken\n"
            "  (it reaches ~0.99 at I/sigma where CC_I has fallen to ~0.03). Holding\n"
            "  reflections out does not protect the free value: the contamination enters\n"
            "  per reflection through E_C. CC_I is the model-free one — prefer it.",
            file=out,
        )
        dbg = getattr(self, "_pending_s_report_debug", None)
        if dbg:
            print(f"  [S_post debug] {dbg}", file=out)

    def show_targets(self, text: str = "Refinement target", out: Optional[Any] = None) -> None:
        if out is None:
            out = sys.stdout
        print(f"{text}: {self.target_name}", file=out)

    def show_rfactors_targets_in_bins(self, out: Optional[Any] = None) -> None:
        """Phenix-like resolution table with S_post / S_prior."""
        if out is None:
            out = sys.stdout
        rv = {}
        if hasattr(self.fmodel, "inferred_r_values"):
            rv = self.fmodel.inferred_r_values()
        d_max = rv.get("bin_d_max") or []
        if not d_max:
            print("  (no S_post resolution bins — maps not yet computed or missing d_spacings)", file=out)
            return
        d_min = rv.get("bin_d_min") or []
        # Named for what they are. These were r_iw / r_fw / ... before the S-family
        # rename, and "r_fw" in particular reads as "R French-Wilson", which is a
        # different statistic entirely and the one thing this table does not contain.
        s_post_w = rv.get("s_post_work_bins") or []
        s_post_f = rv.get("s_post_free_bins") or []
        s_prior_w = rv.get("s_prior_work_bins") or []
        s_prior_f = rv.get("s_prior_free_bins") or []
        rho2_w = rv.get("rho2_work_bins") or []
        omega_b = rv.get("omega_bins") or []
        min_free = int(rv.get("min_free_per_shell", 30))
        # The score test is opt-in (PHRIDGE_SCORE_TEST); drop the column entirely
        # rather than printing a permanently blank one.
        show_omega = any(np.isfinite(float(v)) for v in omega_b)
        n_w = rv.get("n_work_bins") or []
        n_f = rv.get("n_free_bins") or []
        scale = self.overall_scale_k1
        # Merge data%/σ_A from last stats report if present
        stats_bins = []
        last = getattr(self.fmodel, "_last_stats_report", None)
        if last is not None and getattr(last, "bins", None):
            stats_bins = list(last.bins)
        header = (
            f"{'bin':>4} {'d_max':>7} {'d_min':>7} {'Nw':>5} {'Nf':>5} "
            + (f"{'ω':>6} " if show_omega else "")
            + f"{'S_post_w':>8} {'S_post_f':>8} {'S_pri_w':>8} {'S_pri_f':>8} "
            f"{'1-ρ²':>6} {'data%':>6} {'σ_A':>6} {'k1':>6}"
        )
        print("+" + "-" * len(header) + "+", file=out)
        title = "| S_post / S_prior vs resolution (shells reuse parent-set k_S)"
        print(f"{title:<{len(header) + 1}}|", file=out)
        print("+" + "-" * len(header) + "+", file=out)
        print(header, file=out)
        print("-" * len(header), file=out)
        for i, dm in enumerate(d_max):
            nw = int(n_w[i]) if i < len(n_w) else 0
            nf = int(n_f[i]) if i < len(n_f) else 0
            riw = float(s_post_w[i]) if i < len(s_post_w) else float("nan")
            rif = float(s_post_f[i]) if i < len(s_post_f) else float("nan")
            rfw = float(s_prior_w[i]) if i < len(s_prior_w) else float("nan")
            rff = float(s_prior_f[i]) if i < len(s_prior_f) else float("nan")
            r2w = float(rho2_w[i]) if i < len(rho2_w) else float("nan")
            lat_s = f"{1.0 - r2w:6.3f}" if np.isfinite(r2w) else f"{'n/a':>6}"
            dn = float(d_min[i]) if i < len(d_min) else float("nan")
            data_s, sa_s = "   n/a", "   n/a"
            if i < len(stats_bins):
                b = stats_bins[i]
                if np.isfinite(b.mean_data_frac):
                    data_s = f"{100.0 * b.mean_data_frac:5.1f}%"
                if np.isfinite(b.mean_sigma_a):
                    sa_s = f"{b.mean_sigma_a:6.3f}"
            def _fmt(x: float) -> str:
                return f"{x:8.4f}" if np.isfinite(x) else f"{'n/a':>8}"

            om_s = ""
            if show_omega:
                om = float(omega_b[i]) if i < len(omega_b) else float("nan")
                om_s = (f"{om:6.3f} " if np.isfinite(om) else f"{'n/a':>6} ")
            print(
                f"{i + 1:4d} {dm:7.3f} {dn:7.3f} {nw:5d} {nf:5d} {om_s}"
                f"{_fmt(riw)} {_fmt(rif)} {_fmt(rfw)} {_fmt(rff)} "
                f"{lat_s} {data_s} {sa_s} {scale:6.4f}",
                file=out,
            )
        print("-" * len(header), file=out)
        print(
            "  S_post / S_prior = expected residual under the posterior / under the "
            "prior (σ_A floor). Same functional, different measure. NOT the "
            "crystallographic R factor; do not compare with deposited R values.",
            file=out,
        )
        print(
            "  Free-set values are the honest ones; work-set values are optimistically "
            "biased by fitting. S_post_f - S_post_w is twice the optimism, so their "
            "midpoint estimates the true error budget.",
            file=out,
        )
        print(
            f"  1-ρ² = latent share of the L2 residual (descriptive, not calibrated). "
            f"Free-set shells with Nf < {min_free} are blanked rather than "
            "reported as noise.",
            file=out,
        )
        if show_omega:
            print(
                "  ω = ξ_free/ξ_work on the map coefficients. Its null value is not 1 "
                "but (1+p/Nw)/(1-p/Nw) for p effective fitted parameters, so read ω>1 "
                "as fitting, not automatically as model error.",
                file=out,
            )

    def show_all(self, header: str = "", out: Optional[Any] = None) -> None:
        self.show_rfactors_targets_scales_overall(header=header, out=out)
        self.show_rfactors_targets_in_bins(out=out)


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

    def _model_scale_array(self) -> Optional[np.ndarray]:
        """Per-reflection scale that :meth:`f_model` applies to ``F_calc + F_bulk``.

        mmtbx's core forms ``k_isotropic_exp * k_isotropic * k_anisotropic * (F_calc +
        F_bulk)``; the fallback path of :meth:`f_model` uses ``k_isotropic *
        k_anisotropic``. None when the arrays are unavailable or malformed.
        """
        try:
            n = self._i_obs.size()
            k = np.asarray(self.k_isotropic(), dtype=np.float64) * np.asarray(
                self.k_anisotropic(), dtype=np.float64
            )
            core = getattr(getattr(self, "arrays", None), "core", None)
            k_exp = getattr(core, "k_isotropic_exp", None) if core is not None else None
            if k_exp is not None:
                k = k * np.asarray(k_exp, dtype=np.float64)
        except Exception:
            return None
        if k.shape != (n,) or not np.all(np.isfinite(k)):
            return None
        return k

    def _resync_after_mmtbx_scaling(self, log: Any = None) -> None:
        """Bring this object back in line after ``f_model_all_scales.run``.

        That class copies our ``__dict__``, runs ``mmtbx.f_model.manager`` methods on the
        copy (none of our overrides), and copies the dict back. So our F caches come back
        holding pre-scaling values, and with ``apply_back_trace`` the overall B has been
        moved into a *new* ``xray_structure`` that our ``_xray_structure`` -- the one the
        target op computes F_calc from -- never saw. A stale cached F_model makes the
        residual-scale fold compute k1 from the wrong F, which alternates cycle to cycle.
        """
        self._f_model = None
        self._last_maps = None
        self._r_values = None
        self._f_post = None
        self._f_mode = None
        self._last_gradients = None
        new_xrs = self.__dict__.get("xray_structure")
        if new_xrs is not None and new_xrs is not self._xray_structure:
            # Recompute F_calc with our own engine and put it in the core, so f_model()
            # and the target op share one F_calc rather than cctbx's and ours.
            self.update_xray_structure(new_xrs, update_f_calc=True)
            print(
                "[mli_quad] scaling replaced the model (overall B moved into the atoms); "
                "F_calc recomputed from the updated structure",
                file=log if log is not None else sys.stdout,
            )
        else:
            self._f_calc = None

    def _model_scale_mismatch(self, f_model: Any) -> Optional[float]:
        """Relative RMS of ``|k (F_calc + F_bulk)| - |f_model|``, or None if unmeasurable."""
        k = self._model_scale_array()
        if k is None or f_model is None or self._f_calc is None:
            return None
        base = np.asarray(self._f_calc.data(), dtype=np.complex128)
        fb = self.f_bulk()
        if fb is not None:
            base = base + np.asarray(fb.data(), dtype=np.complex128)
        ref = np.abs(np.asarray(f_model.data(), dtype=np.complex128))
        diff = np.abs(k * base) - ref
        den = float(np.sqrt(np.mean(ref**2)))
        if not np.isfinite(den) or den <= 0.0:
            return None
        return float(np.sqrt(np.mean(diff**2)) / den)

    def _k_mask_curve_from_k_sol(self, k_sol: float, b_sol: float) -> np.ndarray:
        d = np.asarray(self._i_obs.d_spacings().data(), dtype=np.float64)
        ss = 1.0 / (4.0 * np.maximum(d, 1e-6) ** 2)
        return np.clip(float(k_sol) * np.exp(-float(b_sol) * ss), 0.0, 1.0)

    def _k_mask_from_f_bulk(self) -> Optional[np.ndarray]:
        """Per-reflection k_mask recovered as Re(F_bulk · F_mask*) / |F_mask|²."""
        fb = None
        if hasattr(self, "arrays") and self.arrays is not None and hasattr(self.arrays, "core"):
            core = self.arrays.core
            if core is not None and hasattr(core, "data"):
                fb = getattr(core.data, "f_bulk", None)
        fm = self.f_mask()
        if fb is None or fm is None:
            return None
        bulk = np.asarray(fb, dtype=np.complex128).ravel()
        mask = np.asarray(fm.data(), dtype=np.complex128).ravel()
        if bulk.shape != mask.shape or bulk.size == 0:
            return None
        den = np.abs(mask) ** 2
        med = float(np.median(den))
        if med <= 0.0:
            return None
        k = np.where(den > 1e-8 * med, (bulk * np.conjugate(mask)).real / den, 0.0)
        k = np.clip(k, 0.0, 1.0)
        if _k_mask_is_negligible(k):
            return None
        return k

    def _recover_k_mask_after_scaling(self, log: Any = None) -> None:
        """If LS left k_masks ~0, rebuild a start from f_bulk, k_sol, or 0.35/46.

        ``f_model_all_scales`` copies ``__dict__`` onto a working manager and back.
        The R drop 0.38→0.25 is real on that copy, but ``arrays.core.k_masks``
        often comes back as ~0. The NLL fit then starts at (and stays at) no
        solvent. Recover the curve mmtbx actually used, or a conventional start.
        """
        kms = self.k_masks()
        k0 = kms[0] if kms else None
        if not _k_mask_is_negligible(k0):
            return
        recovered = self._k_mask_from_f_bulk()
        source = "f_bulk / F_mask"
        if recovered is None:
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
            if ks is not None and float(ks) > 0.05:
                recovered = self._k_mask_curve_from_k_sol(float(ks), float(bs) if bs else 46.0)
                source = f"k_sol/B_sol {float(ks):.3f}/{float(bs) if bs else 46.0:.1f}"
        if recovered is None:
            recovered = self._k_mask_curve_from_k_sol(0.35, 46.0)
            source = "default 0.35/46"
        if hasattr(self, "update_core") and getattr(self, "arrays", None) is not None:
            self.update_core(k_mask=[flex.double(np.ascontiguousarray(recovered))])
            self._f_model = None
            self._last_maps = None
            self._r_values = None
        if log is not None:
            print(
                f"[mli_quad] LS k_mask was ~0 after f_model_all_scales; "
                f"seeded from {source} (max={float(np.nanmax(recovered)):.3f})",
                file=log if log is not None else sys.stdout,
            )

    def _fit_bulk_solvent_nll(self, log: Any = None) -> Optional[str]:
        """Refit the binned k_mask to the intensity NLL; install it if it beats mmtbx's.

        Same protocol as mmtbx's fast scaler (one value per resolution bin, interpolated)
        scored on −log p(Z) with σ_A / β profiled. Fitted on the work set with the
        current per-reflection scale held. Returns the journal detail, or None when the
        fit did not run (``PHRIDGE_BULK_SOLVENT_NLL=0``, no single solvent mask, or no
        core to install into).
        """
        if not _env_flag_enabled("PHRIDGE_BULK_SOLVENT_NLL", "1"):
            return None
        if not (hasattr(self, "update_core") and getattr(self, "arrays", None) is not None):
            return None
        f_masks, k_masks = self.f_masks(), self.k_masks()
        if not f_masks or not k_masks or len(f_masks) != 1 or len(k_masks) != 1:
            return None
        if _k_mask_is_negligible(k_masks[0]):
            self._recover_k_mask_after_scaling(log=log)
            k_masks = self.k_masks()
            if not k_masks or _k_mask_is_negligible(k_masks[0]):
                return None
        k_model = self._model_scale_array()
        if k_model is None or self.sigma_wilson is None:
            return None
        n = self._i_obs.size()
        d = np.asarray(self._i_obs.d_spacings().data(), dtype=np.float64)
        ss = 1.0 / (4.0 * np.maximum(d, 1e-6) ** 2)
        if self._r_free_flags is not None:
            work = ~np.asarray(self._r_free_flags.data(), dtype=bool)
        else:
            work = np.ones(n, dtype=bool)
        k_sol_in, b_sol_in = self.k_sol_b_sol()
        raw = self.bridge.call(
            BULK_SOLVENT_OP_NAME,
            f_calc=self.f_calc(),
            f_obs=self._i_obs,
            f_mask=f_masks[0].data(),
            k_model=flex.double(np.ascontiguousarray(k_model)),
            k_mask=k_masks[0],
            fit_mask=flex.bool(work.tolist()),
            ss=flex.double(np.ascontiguousarray(ss)),
            epsilon=self._i_obs.epsilons().data().as_double(),
            centric=self._i_obs.centric_flags().data(),
            sigma_wilson=self.sigma_wilson,
            nu=getattr(self, "nu_per_refl", None),
            k_sol=k_sol_in,
            b_sol=b_sol_in,
            n_shells=_env_int("PHRIDGE_BULK_SOLVENT_BINS"),
            smooth=_env_float("PHRIDGE_SMOOTH_SIGMA_A", 1.0),
        )
        if raw.get("accepted"):
            k_mask_new = np.asarray(raw["k_mask"], dtype=np.float64)
            if k_mask_new.shape == (n,) and np.all(np.isfinite(k_mask_new)):
                self.update_core(k_mask=[flex.double(np.ascontiguousarray(k_mask_new))])
                self.k_sol = float(raw["k_sol"])
                self.b_sol = float(raw["b_sol"])
                self._f_model = None
                self._last_maps = None
                self._r_values = None
                self._f_post = None
                self._f_mode = None
                self._last_gradients = None
                # The solvent contribution moved, so the least-squares residual did too.
                self._fold_residual_scale_into_k_isotropic()
            else:
                raw = dict(raw, accepted=False)
        return _bulk_solvent_detail(raw, k_sol_in, b_sol_in)

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
    """Manages posterior-mode / posterior-mean agreement statistics and the S family.

    Encapsulates:
      - Posterior mean amplitude <F> and posterior mode F_mode accessors
      - S family: s_post_work/free/all, s_prior_work/free/all
      - Shrunken-amplitude diagnostics (biased low, not R factors):
        r_post_work/free/all, r_mode_work/free/all
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

    def f_obs_french_wilson(self) -> Any:
        """French–Wilson amplitudes from ``I_obs`` — the legacy reporting bridge.

        This is the one place in ``mli_quad`` where French–Wilson is deliberately
        allowed. The FW-disable patch exists to keep converted amplitudes out of the
        *refinement target*; the legacy R is reporting only, and it has to be a real
        French–Wilson R or it is not the bridge anyone expects. Cached; ``None`` if
        the conversion is unavailable.
        """
        # Depends only on I_obs, so it survives model updates and is not cleared
        # alongside the f_calc / f_model caches.
        cached = getattr(self, "_f_obs_fw", _FW_UNSET)
        if cached is not _FW_UNSET:
            return cached
        self._f_obs_fw = None
        try:
            from libtbx.utils import null_out

            fw_scale = _french_wilson_scale_unpatched()
            if fw_scale is not None and self._i_obs is not None:
                self._f_obs_fw = fw_scale(miller_array=self._i_obs, log=null_out())
        except Exception:
            self._f_obs_fw = None
        return self._f_obs_fw

    def _r_french_wilson(self, which: str = "work") -> float:
        """Legacy amplitude R on FW-converted ``F_obs`` vs the scaled model."""
        f_obs = self.f_obs_french_wilson()
        if f_obs is None:
            return float("nan")
        try:
            f_model = self.f_model()
            fo, fm = f_obs.common_sets(f_model)
            fo_d = flex.abs(fo.data())
            fm_d = flex.abs(fm.data())
            if which != "all":
                flags = self._r_free_flags
                if flags is not None:
                    fo_f, flags_c = fo.common_sets(flags)
                    sel = flags_c.data() if which == "free" else ~flags_c.data()
                    # common_sets may reorder; recompute the model on the same set
                    fo, fm = fo_f.common_sets(f_model)
                    fo_d = flex.abs(fo.data()).select(sel)
                    fm_d = flex.abs(fm.data()).select(sel)
                elif which == "free":
                    return float("nan")
            if fo_d.size() == 0:
                return float("nan")
            num_k = flex.sum(fo_d * fm_d)
            den_k = flex.sum(fm_d * fm_d)
            k = float(num_k / den_k) if den_k > 0 else 1.0
            den = flex.sum(fo_d)
            if den <= 0:
                return float("nan")
            return float(flex.sum(flex.abs(fo_d - k * fm_d)) / den)
        except Exception:
            return float("nan")

    def r_work(self) -> float:
        """Legacy French–Wilson amplitude R on the working set.

        Deliberately *not* the posterior mean: that statistic shrinks toward
        sigma_A E_C as the data weaken, so it improves when the data get worse and
        must never be deposited as an R factor. See ``r_post_work`` for it.
        """
        return self._r_french_wilson("work")

    def r_free(self) -> float:
        """Legacy French–Wilson amplitude R on the free/test set."""
        return self._r_french_wilson("free")

    def r_all(self) -> float:
        """Legacy French–Wilson amplitude R on all reflections."""
        return self._r_french_wilson("all")

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
        """Format R-factors mimicking ``mmtbx.f_model.manager.r_factors``.

        **These are French–Wilson amplitude R factors** and the label says so. They
        must not be the posterior-mean statistic: that one shrinks toward
        ``sigma_A E_C`` as the data weaken, so it *improves* when the data get worse.
        ``r_post_*`` still exists as a diagnostic and is printed under an explicit
        "biased low, not an R factor" heading; it is never what a caller asking for
        "the R factors" receives. See :meth:`r_work` and :meth:`r_post_work`.
        """
        rw, rf, ra = self.r_work(), self.r_free(), self.r_all()
        rv = self.inferred_r_values()
        cw = float(rv.get("cc_post_work", rv.get("cc_work", float("nan"))))
        cf = float(rv.get("cc_post_free", rv.get("cc_free", float("nan"))))
        fmt = "%s r_work=%6.4f r_free=%6.4f r_all=%6.4f cc_work=%6.4f cc_free=%6.4f (French-Wilson amplitudes)"
        if as_string:
            return fmt % (prefix, rw, rf, ra, cw, cf)
        if group_args is not None:
            return group_args(
                r_work=rw, r_free=rf, r_all=ra, cc_work=cw, cc_free=cf, r_source="french_wilson"
            )
        return {
            "r_work": rw,
            "r_free": rf,
            "r_all": ra,
            "cc_work": cw,
            "cc_free": cf,
            "r_source": "french_wilson",
        }

    # Explicit aliases. Anything that wants to be unambiguous in a report, a column
    # header or a REMARK should call these rather than the bare r_work / r_free.
    def r_work_french_wilson(self) -> float:
        """French–Wilson amplitude R on the working set (same as :meth:`r_work`)."""
        return self._r_french_wilson("work")

    def r_free_french_wilson(self) -> float:
        """French–Wilson amplitude R on the free/test set (same as :meth:`r_free`)."""
        return self._r_french_wilson("free")

    def r_all_french_wilson(self) -> float:
        """French–Wilson amplitude R on all reflections (same as :meth:`r_all`)."""
        return self._r_french_wilson("all")

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

    @staticmethod
    def _finite_or_nan(value: Any) -> float:
        if value is None:
            return float("nan")
        try:
            v = float(value)
        except (TypeError, ValueError):
            return float("nan")
        return v if np.isfinite(v) else float("nan")

    def s_post_work(self) -> float:
        """S_post (expected residual under the posterior) on work reflections."""
        return self._finite_or_nan(self.inferred_r_values().get("s_post_work"))

    def s_post_free(self) -> float:
        """S_post (expected residual under the posterior) on free reflections."""
        return self._finite_or_nan(self.inferred_r_values().get("s_post_free"))

    def s_post_all(self) -> float:
        """S_post (expected residual under the posterior) on all reflections."""
        return self._finite_or_nan(self.inferred_r_values().get("s_post_all"))

    def s_prior_work(self) -> float:
        """S_prior (same functional under the prior) on work reflections."""
        return self._finite_or_nan(self.inferred_r_values().get("s_prior_work"))

    def s_prior_free(self) -> float:
        """S_prior (same functional under the prior) on free reflections."""
        return self._finite_or_nan(self.inferred_r_values().get("s_prior_free"))

    def s_prior_all(self) -> float:
        """S_prior (same functional under the prior) on all reflections."""
        return self._finite_or_nan(self.inferred_r_values().get("s_prior_all"))

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
        self.params = params or SfEngineParams.fastest(self.d_min)
        self._sf_handle: Optional[str] = None

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
        self.nu_per_refl: Optional[Any] = None
        self._nu_params: Dict[str, Any] = {}
        self.scale_factor = float(scale_factor)
        self.n_bins = int(n_bins)

        # Resolution binning & normalization factors
        self.binner = binner or self._i_obs.setup_binner(n_bins=self.n_bins)
        self.sigma_wilson = self._setup_sigma_wilson(sigma_wilson)
        # Populated by the nuisance fit; None means β is tied to 1 - σ_A².
        self.beta_residual: Optional[Any] = None
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
        self._spatial_sigma_a: Optional[Dict[str, Any]] = None
        self._spatial_sigma_a_v2_block: Optional[Any] = None
        self._spatial_sigma_a_v2_result: Optional[Any] = None

        # Interleaved refinement (refinement.target_mode). Exact by default: the
        # surrogate path is entirely opt-in and never active unless a controller has
        # been installed and has switched it on.
        from phridge.client.intensity.interleaved import target_mode_from_env

        self.target_mode = target_mode_from_env()
        self._surrogate: Optional[Dict[str, Any]] = None
        self._use_surrogate: bool = False
        self._interleaved: Optional[Any] = None
        self._surrogate_eval_count: int = 0
        # Run-scoped NLL journal (PHRIDGE_NLL_LOG). Created on first use so a run that
        # never reaches a stage boundary never pays for a probe.
        self._nll_journal: Optional[Any] = None
        # The final macro cycle always runs fully exact, so deposited statistics never
        # touch the surrogate. total_macro_cycles is filled in from the Phenix params.
        self.macro_cycle_index: int = 0
        self.total_macro_cycles: Optional[int] = None

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

    def _wilson_bin_ids(self) -> Optional[Any]:
        """Per-reflection index of the binner bin, for the binned Σ_W.

        The same bins as :meth:`_setup_sigma_wilson`, so the binned model with no tensor
        is exactly the starting Σ. None if the binner does not cover every reflection;
        the op then builds its own bins.
        """
        binner = getattr(self, "binner", None)
        if binner is None:
            return None
        try:
            ids = np.full(self._i_obs.size(), -1.0, dtype=np.float64)
            for k, i_bin in enumerate(binner.range_used()):
                sel = np.asarray(binner.selection(i_bin), dtype=bool)
                ids[sel] = float(k)
        except Exception:
            return None
        if np.any(ids < 0):
            return None
        return flex.double(ids)

    def _setup_sigma_a(self, user_sa: Optional[Any]) -> Any:
        if user_sa is not None:
            return user_sa
        ss = 1.0 / flex.pow2(self._i_obs.d_spacings().data())
        s_max = float(flex.max(ss)) if ss.size() > 0 else 1.0
        sa = flex.double(0.95 * np.exp(-0.30 * (np.asarray(ss) / max(s_max, 1e-12))))
        return sa

    # ---------------------------------------------------------------- Core SF & Model
    def _ensure_sf_handle(self, px: Any, table: Any, hkl: Any) -> str:
        """Bind the stamp engine once; later F / grads refresh sites on the handle."""
        hid = getattr(self, "_sf_handle", None)
        if hid is None:
            hid = str(self.bridge.call("sf_bind", xray=px, table=table, hkl=hkl, params=self.params))
            self._sf_handle = hid
        return hid

    def f_calc(self, xray_structure: Optional[Any] = None) -> Any:
        """Calculated structure factors F_calc in physical electron units (remote)."""
        if xray_structure is not None:
            self.update_xray_structure(xray_structure)
        if self._f_calc is None:
            px, table = _packed_xray(self._xray_structure, self.table)
            hkl = self._i_obs
            raw = self.bridge.call(
                "sf_calc",
                xray=px,
                table=table,
                hkl=hkl,
                params=self.params,
                handle=self._ensure_sf_handle(px, table, hkl),
            )
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
        kw: dict[str, Any] = {
            "f_obs": self._i_obs,
            "alpha": self.sigma_a,
            "beta": self.sigma_wilson,
            "epsilon": self._i_obs.epsilons().data().as_double(),
            "centric": self._i_obs.centric_flags().data(),
            "r_free": self._r_free_flags.data() if self._r_free_flags is not None else None,
        }
        nu_pr = getattr(self, "nu_per_refl", None)
        if nu_pr is not None:
            kw["nu"] = nu_pr
        # The fitted residual variance in normalized units. Absent until a nuisance fit has
        # run, and None whenever β is tied, in which case every consumer falls back to
        # 1 - σ_A² exactly as before.
        beta_res = getattr(self, "beta_residual", None)
        if beta_res is not None:
            kw["beta_residual"] = beta_res
        kw.update(self._spatial_sigma_a_v2_kwargs())
        return kw

    # ---------------------------------------------------------------- interleaved mode
    def surrogate_active(self) -> bool:
        """True while an interleaved inner block is running on the surrogate."""
        return bool(self._use_surrogate and self._surrogate is not None)

    def use_surrogate(self, enabled: bool) -> None:
        """Switch the target functor between the exact quadrature and the surrogate.

        Part of the :class:`~phridge.client.intensity.interleaved.InterleavedHost`
        contract; the controller owns every call.
        """
        self._use_surrogate = bool(enabled) and self._surrogate is not None

    def _announce_macro_cycle(self, log: Any = None) -> None:
        """Mark the macro-cycle boundary and say whether it will interleave.

        The controller logs each block, but only once a block starts. This line goes
        out at the scale-update substage, so the log shows the "when" even for a cycle
        whose blocks all end up exact.
        """
        from phridge.client.intensity.interleaved import TargetMode

        mode = getattr(self, "target_mode", TargetMode.exact)
        cycle = getattr(self, "macro_cycle_index", 0)
        total = getattr(self, "total_macro_cycles", None)
        when = f"macro cycle {cycle}/{total}" if total else f"macro cycle {cycle}"
        if mode is not TargetMode.interleaved:
            text = f"[interleaved] {when}: target_mode=exact -- every target call is an exact quadrature"
        elif total is not None and cycle >= total:
            text = (
                f"[interleaved] {when}: FINAL macro cycle -- forced fully exact, "
                "no interleaving from here on; deposited statistics are exact"
            )
        else:
            text = (
                f"[interleaved] {when}: interleaving enabled -- inner blocks in this cycle "
                "run on the surrogate and each is adjudicated by an exact evaluation"
            )
        from phridge.client.intensity.heartbeat import output_streams

        sink = log if log is not None and hasattr(log, "write") else None
        for out in output_streams(sink):
            try:
                print(text, file=out, flush=True)
            except Exception:
                pass

    def _surrogate_eval_kwargs(self) -> Dict[str, Any]:
        """``target_eval`` kwargs for the stock ``ml_f`` path.

        The surrogate arrays occupy the observation slots ``ml_f`` already has for
        them: ``f_obs`` (amplitude), ``alpha`` (= sigma_A, shell-wise and pinned) and
        ``beta`` (residual scale). ``epsilon``, ``centric`` and ``r_free`` are the same
        arrays the exact target uses, so the work-set normalization is identical and
        the two NLLs are directly comparable up to a per-reflection constant.
        """
        sur = self._surrogate
        if sur is None:
            raise RuntimeError("no surrogate is installed; call exact_checkpoint() first")
        return {
            "f_obs": sur["f_p_miller"],
            "alpha": sur["alpha_p"],
            "beta": sur["beta_p"],
            "epsilon": self._i_obs.epsilons().data().as_double(),
            "centric": self._i_obs.centric_flags().data(),
            "r_free": self._r_free_flags.data() if self._r_free_flags is not None else None,
            "target": {"name": "ml_f"},
        }

    def _splice_exact_route(self, raw: Any, f_calc: Any, compute_gradients: bool) -> Any:
        """Route the unfittable reflections through the exact target inside the block.

        Step 4 of the surrogate fallback ladder. These reflections should not occur --
        they need both the Newton fit and the closed-form EM initialization to be
        unusable -- so this costs an extra quadrature evaluation only in a case the
        telemetry is already flagging. Both calls share ``r_free``, hence the same
        ``1/n_work`` factor, so splicing the per-reflection arrays elementwise is exact.
        """
        sur = self._surrogate
        if sur is None:
            return raw
        route = sur.get("exact_route")
        if route is None or not bool(np.any(route)):
            return raw
        kw = self._common_eval_kwargs()
        kw["target"] = self.target_spec
        kw["compute_curvature"] = bool(compute_gradients)
        exact = self.bridge.call("target_eval", f_calc=f_calc, **kw)

        from phridge.sfcalc.packing import PackedTargetResult

        sel = np.asarray(route, dtype=bool)
        per = np.asarray(raw.per_reflection, dtype=np.float64).copy()
        grad = np.asarray(raw.d_target_d_f_calc, dtype=np.complex128).copy()
        per[sel] = np.asarray(exact.per_reflection, dtype=np.float64)[sel]
        grad[sel] = np.asarray(exact.d_target_d_f_calc, dtype=np.complex128)[sel]
        curv_r = curv_t = None
        if raw.curv_radial is not None and exact.curv_radial is not None:
            curv_r = np.asarray(raw.curv_radial, dtype=np.float64).copy()
            curv_t = np.asarray(raw.curv_tangential, dtype=np.float64).copy()
            curv_r[sel] = np.asarray(exact.curv_radial, dtype=np.float64)[sel]
            curv_t[sel] = np.asarray(exact.curv_tangential, dtype=np.float64)[sel]
        work = np.ones(per.shape, dtype=bool)
        if self._r_free_flags is not None:
            work = ~np.asarray(self._r_free_flags.data(), dtype=bool)
        n_work = max(int(work.sum()), 1)
        value = float(per[work].sum() / n_work)
        value_test = None
        if bool((~work).any()):
            value_test = float(per[~work].sum() / max(int((~work).sum()), 1))
        return PackedTargetResult(
            name=raw.meta.name,
            value=value,
            per_reflection=per,
            d_target_d_f_calc=grad,
            curv_radial=curv_r,
            curv_tangential=curv_t,
            value_test=value_test,
            scale_factor=raw.meta.scale_factor,
        )

    def _spatial_sigma_a_enabled(self) -> bool:
        return _env_flag_enabled("PHRIDGE_SPATIAL_SIGMA_A", "0")

    def _spatial_sigma_a_v2_kwargs(self) -> dict[str, Any]:
        """Optional v2 block + fitted result. Empty when the flag is off."""
        from phridge.contrib.spatial_sigmaa_v2.options import job_input_from_state, options_from_env

        return job_input_from_state(
            getattr(self, "_spatial_sigma_a_v2_block", None),
            getattr(self, "_spatial_sigma_a_v2_result", None),
            opts=options_from_env(),
        )

    def _run_spatial_sigma_a_v2_step(self, log: Any = None) -> None:
        """Field or Fisher half-step after nuisance fit. No-op when the flag is off.

        (w_j, U_j) are frozen for the subsequent xyz/ADP cycle (spec §10).
        Failures never abort the macrocycle.
        """
        from phridge.contrib.spatial_sigmaa_v2 import block_from_env, options_from_env

        opts = options_from_env()
        if not opts.enabled:
            return
        try:
            from phridge.client.convert_xtal import scattering_table_from_cctbx, xray_from_cctbx
            from phridge.contrib.spatial_sigmaa_v2.alternate import run_macrocycle_step
            from phridge.sfcalc.ops import scattering_model

            model = scattering_model(
                xray_from_cctbx(self.xray_structure),
                scattering_table_from_cctbx(self.xray_structure),
            )
            hkl = np.array(list(self._i_obs.indices()), dtype=np.int64)
            intensity = np.asarray(self._i_obs.data(), dtype=np.float64)
            epsilon = np.asarray(self._i_obs.epsilons().data(), dtype=np.float64)
            centric = np.asarray(self._i_obs.centric_flags().data(), dtype=bool)
            block = self._spatial_sigma_a_v2_block or block_from_env()
            if block is None:
                return
            packed, result, _frozen = run_macrocycle_step(
                model, hkl, intensity, block, opts, epsilon=epsilon, centric=centric
            )
            self._spatial_sigma_a_v2_block = packed
            self._spatial_sigma_a_v2_result = result
            if log is not None:
                kind = "Fisher" if packed.meta.fisher else "field"
                print(
                    f"[spatial_sigma_a_v2] {kind} step  n_coeff={packed.meta.n_coeff}  "
                    f"mean w={float(np.mean(result.w)):.3f}  mean tr(U)={float(np.mean(result.tr_U)):.4f}",
                    file=log,
                )
            self._write_spatial_sigma_a_v2_maps(packed, result, model, log=log)
        except Exception as exc:
            try:
                print(f"[spatial_sigma_a_v2] step skipped: {exc}", file=sys.stderr if log is None else log)
            except Exception:
                pass

    def _spatial_options_from_env(self) -> Any:
        from phridge.contrib.intensity_ll.local_sigma_a import LocalSigmaAOptions

        return LocalSigmaAOptions(
            enabled=self._spatial_sigma_a_enabled(),
            d_min=_env_float("PHRIDGE_SPATIAL_SIGMA_A_D_MIN", 15.0) or 15.0,
            lambda_u=_env_float("PHRIDGE_SPATIAL_SIGMA_A_LAMBDA_U", 1.0),
        )

    def _atoms_bulk_split(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """``F_atoms = k F_calc``, ``F_bulk = k f_bulk()`` on the observation scale."""
        k = self._model_scale_array()
        try:
            fc = np.asarray(self.f_calc().data(), dtype=np.complex128)
        except Exception:
            return None
        fb_arr = self.f_bulk()
        if fb_arr is None:
            return None
        try:
            fb = np.asarray(fb_arr.data(), dtype=np.complex128)
        except Exception:
            return None
        n = int(self._i_obs.size())
        if fc.shape[0] != n or fb.shape[0] != n:
            return None
        if k is None:
            k = np.ones(n, dtype=np.float64)
        residual_k1 = float(self.scale_k1()) if getattr(self, "_scale_fitted", False) else 1.0
        if residual_k1 > 0 and np.isfinite(residual_k1):
            k = k * residual_k1
        return k * fc, k * fb

    def _spatial_n_per_refl(self, state: dict[str, Any]) -> np.ndarray:
        n = int(self._i_obs.size())
        n_k = np.asarray(state.get("n_shell_rms") or [1.0], dtype=np.float64).reshape(-1)
        edges = state.get("bin_edges_s2")
        if n_k.size <= 1 or not edges:
            return np.full(n, float(n_k[0]) if n_k.size else 1.0)
        d = np.asarray(self._i_obs.d_spacings().data(), dtype=np.float64)
        s_sq = 1.0 / np.maximum(d, 1e-12) ** 2
        ed = np.asarray(edges, dtype=np.float64)
        sid = np.clip(np.digitize(s_sq, ed[1:-1], right=False), 0, n_k.size - 1)
        return n_k[sid]

    def _likelihood_f_calc(self) -> Any:
        """The F the Rice target / maps / surrogate score. ``F_eff`` when spatial σ_A is on."""
        state = getattr(self, "_spatial_sigma_a", None)
        if not state or not state.get("enabled"):
            return self._checkpoint_f_model()
        split = self._atoms_bulk_split()
        if split is None:
            return self._checkpoint_f_model()
        fa, fb = split
        n_per = self._spatial_n_per_refl(state)
        # n_k stored per shell; apply_frozen_mix wants shell n_k + ids, or a scalar.
        # Broadcast the already-expanded per-reflection n as a "global" divide via
        # mixing then dividing elementwise.
        f_mix = float(state["w_mol"]) * fa + float(state["w_sol"]) * fb
        scale = np.where(n_per > 1e-30, n_per, 1.0)
        f_eff = f_mix / scale
        template = self._checkpoint_f_model()
        try:
            from cctbx.array_family import flex as _flex

            return template.customized_copy(data=_flex.complex_double(np.ascontiguousarray(f_eff)))
        except Exception:
            return template

    def _apply_spatial_to_target_kwargs(self, kw: dict[str, Any]) -> dict[str, Any]:
        """Rewrite ``k_model`` / ``f_bulk`` so target_and_gradients sees ``F_eff``."""
        state = getattr(self, "_spatial_sigma_a", None)
        if not state or not state.get("enabled"):
            return kw
        w_mol = float(state.get("w_mol") or 1.0)
        w_sol = float(state.get("w_sol") or 1.0)
        if w_mol == 1.0 and w_sol == 1.0:
            return kw
        n_per = self._spatial_n_per_refl(state)
        if "k_model" in kw and kw["k_model"] is not None:
            k = np.asarray(kw["k_model"], dtype=np.float64).reshape(-1)
            k = k * (w_mol / np.where(n_per > 1e-30, n_per, 1.0))
            kw["k_model"] = flex.double(np.ascontiguousarray(k))
        if "f_bulk" in kw and kw["f_bulk"] is not None and abs(w_mol) > 1e-12:
            fb = np.asarray(kw["f_bulk"], dtype=np.complex128).reshape(-1)
            kw["f_bulk"] = (w_sol / w_mol) * fb
        return kw

    def _write_spatial_sigma_a_v2_maps(
        self,
        packed: Any,
        result: Any,
        model: Any,
        log: Any = None,
    ) -> None:
        """CCP4 λ / κ / w volumes and a diagnostic PDB. Never fails refinement."""
        try:
            from phridge.contrib.spatial_sigmaa_v2.viz import atom_display_columns, field_volumes

            prefix = (
                os.environ.get("PHRIDGE_SPATIAL_SIGMA_A_V2_PREFIX")
                or os.environ.get("PHRIDGE_SPATIAL_SIGMA_A_PREFIX", "mli")
            ).strip() or "mli"
            vols = field_volumes(packed, model)
            cs = self.xray_structure.crystal_symmetry()
            written: list[str] = []
            for name in ("lambda", "kappa", "w"):
                path = f"{prefix}_{name}.ccp4"
                self._write_real_ccp4(path, vols[name], cs, f"phridge spatial_sigma_a_v2 {name}")
                written.append(path)
            w, b_err = atom_display_columns(result)
            pdb_path = f"{prefix}_field_atoms.pdb"
            self._write_field_atoms_pdb(pdb_path, w, b_err)
            written.append(pdb_path)
            out = log if log is not None else sys.stdout
            print(f"[spatial_sigma_a_v2] wrote {' '.join(written)}", file=out)
            print(
                "[spatial_sigma_a_v2] Coot: open λ/κ/w maps; contour λ at ±0.2, "
                "w at 0.7; colour field_atoms.pdb by B (error-B = κ)",
                file=out,
            )
        except Exception as exc:
            try:
                print(f"[spatial_sigma_a_v2] maps skipped: {exc}", file=sys.stderr if log is None else log)
            except Exception:
                pass

    def _write_real_ccp4(self, path: str, data: np.ndarray, crystal_symmetry: Any, label: str) -> None:
        from iotbx.map_manager import map_manager
        from scitbx.array_family import flex

        n = tuple(int(x) for x in np.asarray(data).shape)
        flat = np.ascontiguousarray(np.asarray(data, dtype=np.float64).reshape(-1))
        flex_data = flex.double(flat)
        flex_data.reshape(flex.grid(n))
        mm = map_manager(
            map_data=flex_data,
            unit_cell_grid=n,
            unit_cell_crystal_symmetry=crystal_symmetry,
            wrapping=True,
        )
        if hasattr(mm, "write_map"):
            mm.write_map(path)
        else:
            from iotbx import ccp4_map

            ccp4_map.write_ccp4_map(
                file_name=path,
                unit_cell=crystal_symmetry.unit_cell(),
                space_group=crystal_symmetry.space_group(),
                map_data=flex_data,
                labels=[label],
            )

    def _write_field_atoms_pdb(self, path: str, w: np.ndarray, b_err: np.ndarray) -> None:
        from cctbx.array_family import flex

        xs = self.xray_structure.deep_copy_scatterers()
        n = int(xs.scatterers().size())
        ww = np.asarray(w, dtype=np.float64).reshape(-1)
        bb = np.asarray(b_err, dtype=np.float64).reshape(-1)
        if ww.shape[0] != n or bb.shape[0] != n:
            raise ValueError("field-atom columns do not match xray_structure")
        xs.set_occupancies(flex.double(np.ascontiguousarray(ww)))
        xs.set_b_iso(values=flex.double(np.ascontiguousarray(bb)))
        with open(path, "w") as fh:
            fh.write(xs.as_pdb_file())

    def _write_log_sigma_a_map(self, state: dict[str, Any], log: Any = None) -> None:
        """Write ``{prefix}_log_sigma_a.ccp4`` from φ coefficients. Never fails refinement."""
        try:
            prefix = os.environ.get("PHRIDGE_SPATIAL_SIGMA_A_PREFIX", "mli").strip() or "mli"
            masks = self.f_masks()
            if not masks:
                return
            fm = masks[0]
            d = np.asarray(self._i_obs.d_spacings().data(), dtype=np.float64)
            s_sq = 1.0 / np.maximum(d, 1e-12) ** 2
            from phridge.contrib.intensity_ll.local_sigma_a import phi_on_full_list

            hkl = np.asarray(fm.indices(), dtype=np.int32)
            phi = phi_on_full_list(
                np.asarray(fm.data(), dtype=np.complex128),
                s_sq,
                float(state.get("u") or 0.0),
                float(state.get("d_min") or 15.0),
                float(state.get("b_blur") or 0.0),
                hkl=hkl,
            )
            arr = fm.customized_copy(data=flex.complex_double(np.ascontiguousarray(phi)))
            fft_map = arr.fft_map(resolution_factor=0.25)
            path = f"{prefix}_log_sigma_a.ccp4"
            fft_map.as_ccp4_map(file_name=path, labels=["phridge log_sigma_a"])
            print(f"[mli_spatial_sigma_a] wrote {path}", file=log if log is not None else sys.stdout)
        except Exception as exc:
            try:
                print(f"[mli_spatial_sigma_a] map skipped: {exc}", file=sys.stderr)
            except Exception:
                pass

    def _checkpoint_f_model(self) -> Any:
        """The structural F_model on the observation scale (not the spatial mix)."""
        if hasattr(self, "f_model_scaled_with_k1"):
            try:
                return self.f_model_scaled_with_k1()
            except Exception:
                pass
        return self.f_model()

    def exact_checkpoint(self) -> Any:
        """One exact evaluation at the current model; fit and install the surrogate.

        This is the only place the surrogate arrays are ever created or refreshed, and
        the returned NLL is the value that adjudicates the block that follows. See
        :mod:`phridge.client.intensity.interleaved`.
        """
        from phridge.client.intensity.interleaved import CheckpointResult
        from phridge.contrib.intensity_ll.surrogate_op import SURROGATE_FIT_OP_NAME

        kw = self._common_eval_kwargs()
        kw["target"] = self.target_spec
        f_model = self._likelihood_f_calc()
        raw = self.bridge.call(SURROGATE_FIT_OP_NAME, f_calc=f_model, **kw)

        f_p = np.asarray(raw["f_p"], dtype=np.float64)
        alpha_p = np.asarray(raw["alpha_p"], dtype=np.float64)
        beta_p = np.asarray(raw["beta_p"], dtype=np.float64)
        mask = np.asarray(raw["mask"], dtype=np.int64)
        from phridge.contrib.intensity_ll.surrogate import FIT_EXACT_ROUTE

        self._surrogate = {
            # Internal arrays. Never printed, never written to an output file.
            "f_p_miller": self._i_obs.customized_copy(
                data=flex.double(np.ascontiguousarray(f_p)), sigmas=None
            ).set_observation_type_xray_amplitude(),
            "alpha_p": flex.double(np.ascontiguousarray(alpha_p)),
            "beta_p": flex.double(np.ascontiguousarray(beta_p)),
            "exact_route": (mask & FIT_EXACT_ROUTE) != 0,
            "telemetry": dict(raw.get("telemetry") or {}),
        }
        self._mli_eval_count = getattr(self, "_mli_eval_count", 0) + 1
        nll = dict(raw.get("nll") or {})
        return CheckpointResult(
            nll=float(nll.get("work", float("nan"))),
            nll_free=None if nll.get("free") is None else float(nll["free"]),
            f_p=f_p,
            alpha_p=alpha_p,
            beta_p=beta_p,
            mask=mask,
            e_c=np.asarray(raw["e_c"], dtype=np.float64),
            rho2=np.asarray(raw["rho2"], dtype=np.float64),
            telemetry=dict(raw.get("telemetry") or {}),
        )

    def exact_nll(self) -> float:
        """One exact evaluation at the current model. No refit, no statistics."""
        kw = self._common_eval_kwargs()
        kw["target"] = self.target_spec
        kw["compute_curvature"] = False
        raw = self.bridge.call("target_eval", f_calc=self._likelihood_f_calc(), **kw)
        self._mli_eval_count = getattr(self, "_mli_eval_count", 0) + 1
        return float(raw.meta.value)

    def surrogate_nll(self) -> float:
        """The surrogate's own value at the current model (diagnostic only)."""
        if self._surrogate is None:
            raise RuntimeError("no surrogate is installed")
        kw = self._surrogate_eval_kwargs()
        kw["compute_curvature"] = False
        raw = self.bridge.call("target_eval", f_calc=self._likelihood_f_calc(), **kw)
        return float(raw.meta.value)

    def interleaved_controller(self, log: Any = None) -> Optional[Any]:
        """The run's controller, created on first use; ``None`` in exact mode."""
        from phridge.client.intensity.interleaved import (
            InterleavedController,
            TargetMode,
            interleaved_options_from_env,
        )

        if self.target_mode is not TargetMode.interleaved:
            return None
        if self._interleaved is None:
            self._interleaved = InterleavedController(
                self,
                interleaved_options_from_env(),
                log=log,
                mode=TargetMode.interleaved,
                journal=self.nll_journal(log=log),
            )
        return self._interleaved

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

        # The target must see the same F_model as the functor, the maps and the nuisance
        # fit: f_model·k1, with mmtbx's per-reflection scales inside f_model. A scalar
        # mean of those scales gives a differently shaped F_model whenever k_isotropic
        # has a resolution slope.
        k_model = self._model_scale_array() if scale_factor is None else None
        residual_k1: Any = False
        if k_model is not None:
            # scale_k1() before the first scaling is the stored scale, not an LS fit.
            if getattr(self, "_scale_fitted", False):
                residual_k1 = True
            else:
                k_model = k_model * float(self.scale_factor)
            k_scale = float(np.mean(k_model))
        else:
            k_scale = float(scale_factor) if scale_factor is not None else self.scale_factor
        px, table = _packed_xray(self._xray_structure, self.table)
        kw = self._common_eval_kwargs()
        kw.update({
            "xray": px,
            "table": table,
            "params": self.params,
            "handle": self._ensure_sf_handle(px, table, self._i_obs),
            "target": self.target_spec,
            "precondition": bool(precondition),
            "damping": float(damping),
            "scale_factor": 1.0 if k_model is not None else k_scale,
        })
        if k_model is not None:
            kw["k_model"] = flex.double(np.ascontiguousarray(k_model))
            kw["residual_k1"] = bool(residual_k1)
        fb = self.f_bulk()
        if fb is not None:
            kw["f_bulk"] = fb.data()
        kw = self._apply_spatial_to_target_kwargs(kw)
        kw.update(self._spatial_sigma_a_v2_kwargs())

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

    def nll_point(self) -> Any:
        """Work and free NLL from a **single** exact evaluation, with their counts.

        ``target_w()`` followed by ``target_t()`` is two full quadratures for two numbers
        the same functor call already produced, which matters here because the weight
        scans ask for this pair once per trial.
        """
        from phridge.client.intensity.nll_log import NllPoint

        res = self.target_functor()(compute_gradients=False)
        work = float(res.target_work())
        test = res.target_test()
        # Cache the quadrature's own number. The minimizer's progress line diffs against
        # it, and that number is -log p(Z); mixing in the normalization offset here would
        # show up as a phantom jump on the next gradient evaluation.
        self._target_value = work
        if test is not None:
            self._target_value_test = float(test)
        # The journal reports -log p(I) = -log p(Z) + log(εΣ), which is what stays
        # comparable when the nuisance fit replaces Σ_W. While Σ is fixed the two differ
        # by a constant, so every delta is the same number either way.
        off_w, off_f = self._normalization_offset()
        if off_w is not None and np.isfinite(work):
            work += off_w
        if test is not None and off_f is not None:
            test = float(test) + off_f
        n_free = 0
        if self._r_free_flags is not None:
            try:
                n_free = int(self._r_free_flags.data().count(True))
            except Exception:
                n_free = 0
        n_total = int(self._i_obs.size()) if self._i_obs is not None else None
        return NllPoint(
            work=work,
            free=float(test) if test is not None else None,
            n_work=(n_total - n_free) if n_total is not None else None,
            n_free=n_free or None,
        )

    def _normalization_offset(self) -> tuple[Optional[float], Optional[float]]:
        """``(mean log(εΣ) on work, mean log(εΣ) on free)``. ``(None, None)`` if unavailable."""
        from phridge.client.intensity.nll_log import normalization_offset

        try:
            eps = self._i_obs.epsilons().data().as_double()
            free = self._r_free_flags.data() if self._r_free_flags is not None else None
            return normalization_offset(eps, self.sigma_wilson, free)
        except Exception:
            return None, None

    def nll_journal(self, log: Any = None) -> Any:
        """The run's NLL journal, created on first use and kept for the whole run."""
        from phridge.client.intensity.nll_log import NllJournal

        if self._nll_journal is None:
            self._nll_journal = NllJournal(self.nll_point, log=log)
        self._nll_journal.set_log(log)
        return self._nll_journal

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
        nu_bounds: Tuple[float, float] = (2.5, 200.0),
        **kwargs: Any,
    ) -> Any:
        """Co-refine bulk solvent mask (k_sol, B_sol, k_aniso), sigma_A(s), Sigma_W(s), and scale factor k."""
        # Phenix calls this once per macro cycle at the scale-update substage, so it is
        # also the macro-cycle counter the interleaved controller reads to know when it
        # has reached the final (always fully exact) cycle.
        self.macro_cycle_index = getattr(self, "macro_cycle_index", 0) + 1
        if self.total_macro_cycles is None:
            try:
                n = getattr(getattr(params, "main", None), "number_of_macro_cycles", None)
                if n is not None:
                    self.total_macro_cycles = int(n)
            except Exception:
                pass
        if not fit_nu:
            env_fit_nu = os.environ.get("PHRIDGE_FIT_NU")
            if env_fit_nu is not None and env_fit_nu.strip().lower() in ("1", "true", "yes", "on"):
                fit_nu = True
        if fit_nu and not _fit_nu_this_cycle(self.macro_cycle_index):
            held = getattr(self, "nu", None)
            nu_p = dict(getattr(self, "_nu_params", None) or {})
            if log is not None:
                if nu_p.get("bin_nu"):
                    sel = " ".join(
                        "G" if float(v) >= 199 else f"{float(v):g}" for v in nu_p["bin_nu"]
                    )
                    print(
                        f"[mli_quad] ν held per shell at {sel} "
                        f"(search is cycle 1 only; --fit-nu-every-cycle to refit)",
                        file=log,
                    )
                elif held is not None:
                    print(
                        f"[mli_quad] ν held at {float(held):g} "
                        f"(search is cycle 1 only; --fit-nu-every-cycle to refit)",
                        file=log,
                    )
            fit_nu = False

        # The NLL journal: this is the run's first chance to measure the target, and the
        # two stages below are the only ones that refit nuisances rather than move atoms.
        from phridge.client.intensity.nll_log import StageKind

        journal = self.nll_journal(log=log)
        cycle_label = (
            f"macro cycle {self.macro_cycle_index}/{self.total_macro_cycles}"
            if self.total_macro_cycles
            else f"macro cycle {self.macro_cycle_index}"
        )
        journal.set_prefix(cycle_label)
        # One probe serves as both the run's first NLL and this stage's entry value.
        scale_start = journal.measure()
        journal.mark_start(scale_start)
        scale_t0 = time.monotonic()

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
                params = self._constrain_scaling_to_isotropic(params, log=log)
                if os.environ.get("PHRIDGE_DT_MASK", "").strip().lower() in {"1", "true", "yes", "on"}:
                    if not getattr(self, "_dt_mask_ignored_logged", False):
                        self._dt_mask_ignored_logged = True
                        print(
                            "[mli_quad] PHRIDGE_DT_MASK is set but ignored: "
                            "phenix refine uses the flat Jiang–Brünger F_mask.",
                            file=log if log is not None else sys.stdout,
                        )
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
                    self._resync_after_mmtbx_scaling(log=log)
                if optimize_mask and hasattr(self, "optimize_mask"):
                    self.optimize_mask(out=log)
                try:
                    ks, bs = self.k_sol_b_sol_from_k_mask()
                    kms = self.k_masks()
                    k0 = kms[0] if kms else None
                    # A caption of a ~0 k_mask (0.02/26) must not overwrite mmtbx's k_sol.
                    if ks is not None and bs is not None and not _k_mask_is_negligible(k0):
                        self.k_sol = float(ks)
                        self.b_sol = float(bs)
                except Exception:
                    pass
                self._recover_k_mask_after_scaling(log=log)
            except Exception as exc:
                # Scaling is allowed to fail without stopping refinement, but not silently:
                # a failed run leaves whatever scales were there before.
                print(
                    f"[mli_quad] WARNING: bulk-solvent scaling failed ({type(exc).__name__}: "
                    f"{exc}); keeping the previous scales.",
                    file=log if log is not None else sys.stdout,
                )

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
        try:
            mismatch = self._model_scale_mismatch(f_model)
            if mismatch is not None and mismatch > 1e-6:
                print(
                    f"[mli_quad] WARNING: per-reflection k·(F_calc+F_bulk) differs from "
                    f"f_model() by relative RMS {mismatch:.3g}; the target and the nuisance "
                    f"fit are not seeing the same F_model.",
                    file=log if log is not None else sys.stdout,
                )
        except Exception:
            pass  # a diagnostic must never cost a macro cycle

        # Measured here rather than immediately after the scaling call: the residual fold
        # above invalidates the cached f_model, so this is the first point at which the
        # scales and the model structure factors are consistent again. The same point is
        # the entry value of the next stage.
        after_scaling = journal.measure()
        journal.record_stage(
            "bulk solvent + scaling",
            StageKind.target,
            scale_start,
            after_scaling,
            seconds=time.monotonic() - scale_t0,
            detail="" if bulk_solvent_and_scaling else "scaling disabled; residual fold only",
        )
        nuisance_start = after_scaling
        if bulk_solvent_and_scaling:
            bulk_t0 = time.monotonic()
            bulk_detail: Optional[str] = None
            try:
                with mli_heartbeat("ml_i_bulk_solvent_fit", log=log, announce=True):
                    bulk_detail = self._fit_bulk_solvent_nll(log=log)
            except Exception as exc:
                print(
                    f"[mli_quad] WARNING: NLL bulk-solvent fit failed ({type(exc).__name__}: "
                    f"{exc}); keeping the least-squares k_mask.",
                    file=log if log is not None else sys.stdout,
                )
            if bulk_detail is not None:
                nuisance_start = journal.measure()
                journal.record_stage(
                    "bulk solvent (NLL k_mask)",
                    StageKind.target,
                    after_scaling,
                    nuisance_start,
                    seconds=time.monotonic() - bulk_t0,
                    detail=bulk_detail,
                )
        nuisance_t0 = time.monotonic()
        # Σ_W is about to be replaced. Record log(εΣ) now so the stage line can show how
        # much of the NLL movement is the change of normalization rather than the fit.
        norm_before = self._normalization_offset()

        # 4. Refine sigma_A(s), Sigma_W(s), and nu on worker.
        # σ_A is estimated on the test set, as cctbx does for α/β: on the work set the
        # model has already been fitted to these very reflections, so σ_A absorbs the
        # overfitting and biases toward 1. Σ_W rides along on the same set at no cost —
        # it is a two-parameter Wilson curve, not a per-bin estimate.
        free_flags = self._r_free_flags.data() if self._r_free_flags is not None else None
        n_free = int(free_flags.count(True)) if free_flags is not None else 0
        if free_flags is not None and n_free >= _MIN_NUISANCE_TUNE:
            tune_mask = free_flags
            tune_label = f"free/{n_free}"
        else:
            tune_mask = flex.bool(self._i_obs.size(), True)
            tune_label = f"all/{self._i_obs.size()}"
        sa_mode = os.environ.get("PHRIDGE_SIGMA_A_MODE", "bins").strip().lower() or "bins"
        n_sa_bins_env = os.environ.get("PHRIDGE_SIGMA_A_BINS")
        n_sa_bins = int(n_sa_bins_env) if (n_sa_bins_env and str(n_sa_bins_env).strip().isdigit()) else None
        tv_norm_env = os.environ.get("PHRIDGE_SIGMA_A_TV_NORM", "").strip()
        tv_norm = float(tv_norm_env) if tv_norm_env else 0.0
        fit_sigma_wilson = _env_flag_enabled("PHRIDGE_FIT_SIGMA_WILSON", "1")
        # Before the nuisance fit consumes it: if Σ_W is about to carry the anisotropy,
        # the scale must not still be carrying it too. Raising here, after scaling and
        # before the tensor is fitted, is the last point where the two are separable.
        self._assert_single_anisotropy_carrier(log=log)
        # Fit σ_A and β to the F the target will be evaluated on. The functor, the maps
        # and the checkpoints all use f_model·k1; fitting to bare f_model while k1 ≠ 1
        # fits one amplitude scale and scores another.
        residual_k1 = float(self.scale_k1())
        f_model = self._checkpoint_f_model()
        nu_mode = os.environ.get("PHRIDGE_NU_MODE", "grid").strip().lower() or "grid"
        nu_grid = os.environ.get("PHRIDGE_NU_GRID", "5:50:5")
        spatial_opts = self._spatial_options_from_env()
        spatial_kw: dict[str, Any] = {}
        if spatial_opts.enabled:
            split = self._atoms_bulk_split()
            if split is not None:
                fa, fb = split
                spatial_kw["f_atoms"] = fa
                spatial_kw["f_bulk"] = fb
                spatial_kw["local_sigma_a"] = spatial_opts.model_dump()
            else:
                spatial_kw["local_sigma_a"] = spatial_opts.model_dump()
        raw = None
        with mli_heartbeat(
            f"ml_i_nuisance_fit(sigma_a_mode={sa_mode}, nu_mode={nu_mode if fit_nu else 'off'}, "
            f"wilson={'ml' if fit_sigma_wilson else 'moment'}, tune={tune_label}"
            f"{', spatial σ_A' if spatial_opts.enabled else ''})",
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
                nu_mode=nu_mode,
                nu_grid=nu_grid,
                fit_scale=self._fix_scale_for_free_beta(
                    bool(fit_scale and not bulk_solvent_and_scaling), log=log
                ),
                nu_bounds=list(nu_bounds),
                nu=(
                    np.asarray(self.nu_per_refl, dtype=np.float64).tolist()
                    if getattr(self, "nu_per_refl", None) is not None
                    else self.nu
                ),
                sigma_a_mode=sa_mode,
                n_sigma_a_bins=n_sa_bins,
                tv_norm=tv_norm,
                fit_sigma_wilson=fit_sigma_wilson,
                wilson_model=self.wilson_model,
                wilson_bins=self._wilson_bin_ids(),
                beta_mode=self.beta_mode,
                sigma_a_shape=self.sigma_a_shape,
                smooth_sigma_a=_env_float("PHRIDGE_SMOOTH_SIGMA_A", 1.0),
                smooth_beta=_env_float("PHRIDGE_SMOOTH_BETA", 1.0),
                beta_consistency_prior=_env_float("PHRIDGE_BETA_CONSISTENCY_PRIOR", 0.0),
                sigma_a_tensor=_env_flag_enabled("PHRIDGE_SIGMA_A_TENSOR", "1"),
                sphericity=_env_float("PHRIDGE_SPHERICITY", 1.0),
                **spatial_kw,
                **self._spatial_sigma_a_v2_kwargs(),
            )
        self.sigma_a = flex.double(np.asarray(raw["sigma_a"], dtype=np.float64))
        self.sigma_wilson = flex.double(np.asarray(raw["sigma_wilson"], dtype=np.float64))
        # β reaches the target and the maps only through here. Set it to None when the fit
        # tied it, so switching back to beta_mode=constrained genuinely restores the old
        # prior rather than leaving a stale array behind.
        br = raw.get("beta_residual")
        if br is None:
            self.beta_residual = None
        else:
            br_np = np.asarray(br, dtype=np.float64)
            if br_np.shape == (self._i_obs.size(),) and np.all(np.isfinite(br_np)):
                self.beta_residual = flex.double(np.ascontiguousarray(br_np))
            else:
                self.beta_residual = None
        self._sigma_a_params = dict(raw.get("sigma_a_params") or {})
        loc = dict(self._sigma_a_params.get("local_sigma_a") or {})
        if loc.get("enabled") and loc.get("u") is not None:
            self._spatial_sigma_a = loc
            try:
                self._write_log_sigma_a_map(loc, log=log)
            except Exception:
                pass
        else:
            self._spatial_sigma_a = loc if loc.get("fallback") else None
        self._sigma_wilson_params = dict(raw.get("sigma_wilson_params") or {})
        self._nu_params = dict(raw.get("nu_params") or {})
        if raw.get("scale_k") is not None and (not bulk_solvent_and_scaling):
            self.scale_factor = float(raw["scale_k"])
        if raw.get("nu") is not None:
            self.nu = float(raw["nu"])
            self.target_spec["nu"] = self.nu
            nu_arr = raw.get("nu_per_refl")
            if nu_arr is not None:
                nu_np = np.asarray(nu_arr, dtype=np.float64)
                if nu_np.shape == (self._i_obs.size(),) and np.all(np.isfinite(nu_np)):
                    self.nu_per_refl = flex.double(np.ascontiguousarray(nu_np))
                else:
                    self.nu_per_refl = flex.double(self._i_obs.size(), float(self.nu))
            else:
                self.nu_per_refl = flex.double(self._i_obs.size(), float(self.nu))
        self._f_model = None
        self._last_maps = None
        self._r_values = None
        # sigma_A(s) just moved, and alpha_p is pinned to it, so any surrogate fitted
        # before this point is stale. Drop it rather than refresh it here: the
        # controller's next checkpoint refits from the updated nuisance arrays, which
        # keeps the count at exactly one exact evaluation per block.
        self._surrogate = None
        self._use_surrogate = False
        norm_after = self._normalization_offset()
        detail = _nuisance_detail(tune_label, norm_before, norm_after)
        detail += f", residual k1={residual_k1:.4f}"
        loc = dict(self._sigma_a_params.get("local_sigma_a") or {})
        if loc.get("enabled") and loc.get("u") is not None:
            detail += f", spatial σ_A u={float(loc['u']):+.4f}"
        elif loc.get("fallback"):
            detail += f", spatial σ_A fallback={loc['fallback']}"
        # tune_nll is the objective stage 2 minimized: mean −log p(Z) on the tune
        # set. The stage line is −log p(I). They differ by mean log(εΣ), so printing
        # both is what shows whether the installed model is the one that was fit.
        tune_nll = raw.get("tune_nll") if isinstance(raw, dict) else None
        if tune_nll is not None:
            try:
                detail += f", tune −log p(Z)={float(tune_nll):.4f}"
            except (TypeError, ValueError):
                pass
        nu_p = dict(self._nu_params or {})
        if nu_p.get("mode") in ("grid", "grid_bins") and nu_p.get("grid") and nu_p.get("grid_nll"):
            body = " ".join(
                f"{float(n):g}:{float(v):.4f}"
                for n, v in zip(nu_p["grid"], nu_p["grid_nll"])
            )
            if nu_p.get("mode") == "grid_bins" and nu_p.get("bin_nu"):
                sel = " ".join(
                    "G" if float(v) >= 199 else f"{float(v):g}" for v in nu_p["bin_nu"]
                )
                detail += f", ν grid {body} per-shell {sel}"
            else:
                detail += f", ν grid {body} best={nu_p.get('nu')}"
            g_nll = nu_p.get("grid_nll_gaussian")
            if g_nll is not None and nu_p.get("mode") == "grid" and nu_p.get("nu") is not None:
                chosen = float(nu_p["nu"])
                best_nll = next(
                    (
                        float(v)
                        for n, v in zip(nu_p["grid"], nu_p["grid_nll"])
                        if abs(float(n) - chosen) < 1e-6
                    ),
                    None,
                )
                if best_nll is not None:
                    detail += f" vs Gaussian {float(g_nll):.4f} ({best_nll - float(g_nll):+.4f})"
                else:
                    detail += f" vs Gaussian {float(g_nll):.4f}"
            elif g_nll is not None:
                detail += f" vs Gaussian {float(g_nll):.4f}"
        failures = raw.get("lbfgs_failures") if isinstance(raw, dict) else None
        if failures:
            detail += ", LBFGS failed: " + "; ".join(str(item) for item in failures)
        journal.record_stage(
            "nuisance fit (σ_A, Σ_W, β, ν)",
            StageKind.target,
            nuisance_start,
            journal.measure(),
            seconds=time.monotonic() - nuisance_t0,
            detail=detail,
        )
        # Spatial σ_A v2 field / Fisher half-step: atoms frozen. Inert when the flag is off.
        try:
            self._run_spatial_sigma_a_v2_step(log=log)
        except Exception:
            pass
        # The anisotropic scale was just refitted, so the cached tensor is stale. Report
        # it here, in the scaling method, next to the k_sol / b_sol it belongs with.
        try:
            self._report_aniso_scale(log=log)
        except Exception:
            pass  # a log line must never cost a macro cycle
        try:
            self._announce_macro_cycle(log=log)
        except Exception:
            pass  # a log line must never cost a macro cycle
        self._print_stats_report(log=log, label="after update_all_scales")
        # Phenix's fmodels.update_all_scales passes show=True together with its
        # log, then calls fmodel.show() itself. Showing here too prints the
        # engine summary twice. A caller that asks to show and does not hand us
        # that log still gets the summary.
        if show and log is None:
            self.show(log=log)
        # Optional windowed omit coefficients (read-only; never fails refinement)
        try:
            from phridge.client.intensity.omit import omit_options_from_env, omit_windows_enabled, run_omit_windows_from_fmodel

            if omit_windows_enabled():
                opts = omit_options_from_env()
                when = opts.when
                run_now = when == "every_macrocycle"
                if when == "end_of_refinement":
                    # Overwrite on each scale update; final file reflects the last refined model.
                    run_now = True
                if run_now:
                    run_omit_windows_from_fmodel(self, options=opts, log=log)
        except Exception as exc:
            try:
                print(f"[mli_omit_windows] skipped: {exc}", file=sys.stderr)
            except Exception:
                pass
        return self

    def _print_stats_report(self, log: Any = None, label: str = "") -> Optional[Any]:
        """Print resolution-binned I/σ + σ_A(s) + S_post/S_prior report (PHRIDGE_STATS_REPORT, default on)."""
        from phridge.client.intensity.stats_report import (
            merge_sstat_bins_into_report,
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
            # Always refresh maps for S_post so bins match current scales / σ_A
            try:
                self._r_values = None
                self._last_maps = None
                self.electron_density_map()
                rv = self.inferred_r_values()
                merge_sstat_bins_into_report(report, rv)
                report.s_post_work = float(rv.get("s_post_work", float("nan")))
                report.s_post_free = float(rv.get("s_post_free", float("nan")))
                report.s_prior_work = float(rv.get("s_prior_work", float("nan")))
                report.s_prior_free = float(rv.get("s_prior_free", float("nan")))
                report.k_s_work = float(rv.get("k_s_work", float("nan")))
                report.k_s_prior_work = float(rv.get("k_s_prior_work", float("nan")))
                report.rho2_work = float(rv.get("rho2_work", float("nan")))
                report.rho2_free = float(rv.get("rho2_free", float("nan")))
                report.s_report_version = rv.get("s_report_version")
                if rv.get("s_report_error"):
                    print(
                        f"[mli_quad] S_post unavailable: {rv['s_report_error']}",
                        file=sys.stderr,
                        flush=True,
                    )
                elif rv.get("s_report_version") != 9:
                    print(
                        "[mli_quad] S_post: stale phridge-worker (missing s_report_version=9) — "
                        "re-run with ./scripts/run_phenix_intensity.sh --redis",
                        file=sys.stderr,
                        flush=True,
                    )
                elif rv.get("s_report_debug"):
                    print(
                        f"[mli_quad] S_post debug: {rv['s_report_debug']}",
                        file=sys.stderr,
                        flush=True,
                    )
                elif not any(np.isfinite(b.s_post) for b in report.bins):
                    print(
                        "[mli_quad] S_post bins missing after maps "
                        "(restart phridge-worker if you just pulled S-statistic code)",
                        file=sys.stderr,
                        flush=True,
                    )
            except Exception as exc:
                print(f"[mli_quad] S_post merge skipped: {exc}", file=sys.stderr, flush=True)
            self._merge_agreement_stats_into_report(report)
            self._attach_cc_isig_table(report)
            self._last_stats_report = report
            # Phenix's log already writes to the terminal. Printing to stdout and
            # handing that log in as an extra stream emits the whole table twice.
            from phridge.client.intensity.heartbeat import output_streams
            from phridge.client.intensity.stats_report import format_intensity_stats_report

            text = format_intensity_stats_report(report)
            for stream in output_streams(log):
                try:
                    print(text, file=stream)
                    if hasattr(stream, "flush"):
                        stream.flush()
                except Exception:
                    pass
            return report
        except Exception as exc:
            try:
                print(f"[mli_quad stats report skipped: {exc}]", file=sys.stderr)
            except Exception:
                pass
            return None

    def _merge_agreement_stats_into_report(self, report: Any) -> None:
        """Copy the R factor and the posterior point estimates onto the report.

        The French-Wilson R comes from ``r_work()`` / ``r_free()`` (the model-free
        bridge); the posterior estimates come from the maps ``r_values``. Both are
        reporting-only, so a failure here must never cost a macro cycle.
        """
        try:
            report.r_fw_work = self.r_work()
            report.r_fw_free = self.r_free()
        except Exception:
            pass
        try:
            report.aniso_scale = self.aniso_scale()
        except Exception:
            pass
        try:
            rv = self.inferred_r_values()
        except Exception:
            return
        for attr, key in (
            ("r_post_work", "r_post_work"),
            ("r_post_free", "r_post_free"),
            ("r_mode_work", "r_mode_work"),
            ("r_mode_free", "r_mode_free"),
            ("r_intensity_work", "r_intensity_work"),
            ("r_intensity_free", "r_intensity_free"),
        ):
            try:
                setattr(report, attr, float(rv.get(key, float("nan"))))
            except (TypeError, ValueError):
                pass

    @property
    def wilson_model(self) -> str:
        """``"binned"`` (default), ``"anisotropic"`` or ``"isotropic"`` — the Σ_W form.

        Read from ``PHRIDGE_WILSON_MODEL``. Binned keeps each resolution bin's mean
        intensity and fits a global traceless anisotropic B on top; the two single-curve
        forms must be asked for explicitly. This is the switch that
        decides which of the two degenerate anisotropy carriers is free, so it is resolved
        in one place and every consumer reads it from here.
        """
        cached = getattr(self, "_wilson_model", None)
        if cached is not None:
            return str(cached)
        from phridge.contrib.intensity_ll.wilson import normalize_wilson_model

        value = normalize_wilson_model(os.environ.get("PHRIDGE_WILSON_MODEL"))
        self._wilson_model = value
        return value

    @property
    def beta_mode(self) -> str:
        """``"free"`` (default) or ``"constrained"`` — how stage 2 treats β.

        Read from ``PHRIDGE_BETA_MODE``. Free by default: in normalized units β is the
        intercept of Z_o against E_C² and σ_A² the slope, so tying β = 1 − σ_A² asserts the
        Wilson normalization is exact and launders any error in it into σ_A.
        """
        cached = getattr(self, "_beta_mode", None)
        if cached is not None:
            return str(cached)
        from phridge.contrib.intensity_ll.free_beta import normalize_beta_mode

        value = normalize_beta_mode(os.environ.get("PHRIDGE_BETA_MODE"))
        self._beta_mode = value
        return value

    @property
    def sigma_a_shape(self) -> str:
        """``"free"`` (default) or ``"monotone"`` — whether σ_A(s) may rise.

        Read from ``PHRIDGE_SIGMA_A_SHAPE``. Free by default and regularized by smoothness:
        σ_A is commonly depressed at low resolution where the solvent model is poor, and a
        monotone profile can only push that structure into neighbouring shells.
        """
        cached = getattr(self, "_sigma_a_shape", None)
        if cached is not None:
            return str(cached)
        from phridge.contrib.intensity_ll.free_beta import normalize_sigma_a_shape

        value = normalize_sigma_a_shape(os.environ.get("PHRIDGE_SIGMA_A_SHAPE"))
        self._sigma_a_shape = value
        return value

    def _fix_scale_for_free_beta(self, fit_scale: bool, log: Any = None) -> bool:
        """Drop the stage-2 F_c scale when β is free, because the two are degenerate.

        Only ``σ_A²k²`` enters the slope of Z_o against E_C², so a free β leaves σ_A and an
        overall scale unseparable. The op raises on the combination; this keeps production
        out of that state instead of relying on the caller, in the same spirit as
        :meth:`_constrain_scaling_to_isotropic`. Bulk-solvent scaling already owns the
        overall scale in a normal phenix run, so there is usually nothing to give up.
        """
        if not fit_scale or self.beta_mode != "free":
            return fit_scale
        print(
            "[mli_quad] β is fitted per shell, so the stage-2 F_c scale is fixed (σ_A and "
            "an overall scale enter the Rice first moment only as σ_A²k² and cannot be "
            "separated). Set PHRIDGE_BETA_MODE=constrained to refine a joint scale instead.",
            file=log if log is not None else sys.stdout,
        )
        return False

    def _constrain_scaling_to_isotropic(self, params: Any, log: Any = None) -> Any:
        """Disable anisotropic scale-matrix refinement when Σ_W carries the anisotropy.

        An anisotropic Σ_W and ``k_anisotropic`` absorb the same directional falloff, so
        exactly one may be free (see :mod:`phridge.contrib.intensity_ll.wilson`). The
        normalization is the one that must carry it, because ``data%``, β and the
        posterior weights are only interpretable if the *prior* describes the
        observations' falloff.

        Setting the flag is best-effort across mmtbx versions; it is not the safety net.
        :meth:`_assert_single_anisotropy_carrier` measures what was actually applied
        afterwards, which is what a flag cannot tell us.
        """
        if not _wilson_carries_anisotropy(self.wilson_model):
            return params
        out = log if log is not None else sys.stdout
        # The real phenix.refine scope is bulk_solvent_and_scale, which carries both
        # anisotropic_scaling and minimization_b_cart -- turning off only the first still
        # leaves the b_cart minimizer free to fit the tensor. Both have to go.
        flags = ("anisotropic_scaling", "minimization_b_cart", "symmetry_constraints_on_b_cart")
        holders = [("", params), ("bulk_solvent_and_scale.", getattr(params, "bulk_solvent_and_scale", None))]
        touched, missing = [], []
        for attr in flags:
            if attr == "symmetry_constraints_on_b_cart":
                continue  # harmless either way once b_cart is not refined
            found = False
            for prefix, holder in holders:
                if holder is None or not hasattr(holder, attr):
                    continue
                try:
                    setattr(holder, attr, False)
                    touched.append(f"{prefix}{attr}")
                    found = True
                except Exception:
                    pass
            if not found:
                missing.append(attr)
        note = f"set {', '.join(touched)}" if touched else "no known flag found"
        if missing:
            note += f"; not present: {', '.join(missing)}"
        print(
            "[mli_quad] Wilson Σ_W is anisotropic, so the overall anisotropic scale is "
            "constrained to isotropic (the two are degenerate and the prior must carry "
            f"the anisotropy). Scaling params: {note}. The anisotropy actually applied to "
            "F_calc is verified after scaling, so a missing flag fails loudly rather than "
            "double-fitting.",
            file=out,
        )
        return params

    def _assert_single_anisotropy_carrier(self, log: Any = None) -> None:
        """Raise if Σ_W and ``k_anisotropic`` are both carrying anisotropy.

        Measures the anisotropy actually applied to F_calc rather than trusting the flag
        we set, because a flag records the intent and this records the outcome. Both
        carriers live is the one error mode that produces a well-fitting, uninterpretable
        pair of anisotropy parameters, so it raises instead of warning.
        """
        if not _wilson_carries_anisotropy(self.wilson_model):
            return
        from phridge.contrib.intensity_ll.wilson import ANISO_NEGLIGIBLE_DELTA_B

        applied = self.aniso_scale()
        if applied is None or not getattr(applied, "is_valid", False):
            return  # nothing measurable; the fit itself will still be reported
        if applied.anisotropy <= ANISO_NEGLIGIBLE_DELTA_B:
            return
        raise ValueError(
            "Wilson Σ_W and the overall scale both carry anisotropy: wilson_model="
            f"'{self.wilson_model}' but the applied k_anisotropic still spans "
            f"{applied.anisotropy:.3f} A**2 between principal values (limit "
            f"{ANISO_NEGLIGIBLE_DELTA_B} A**2). These two are degenerate — fitting both "
            "gives a good fit and two individually meaningless tensors.\n"
            "  Fix: set bulk_solvent_and_scale.anisotropic_scaling=False and "
            "bulk_solvent_and_scale.minimization_b_cart=False in the refinement "
            "parameters (phridge sets these automatically when it can reach the params "
            "object; this error means it could not).\n"
            "  Escape hatch: PHRIDGE_WILSON_MODEL=isotropic reverts to a scalar Σ_W, at "
            "the cost of directionally biased data%, beta and map coefficients."
        )

    def _report_aniso_scale(self, log: Any = None) -> None:
        """Print the overall anisotropic B right after the global scaling that set it."""
        from phridge.client.intensity.stats_report import format_aniso_scale_summary

        aniso = self.aniso_scale(refresh=True)
        if aniso is None:
            return
        out = log if log is not None else sys.stdout
        carried = _wilson_carries_anisotropy(self.wilson_model)
        label = "F_calc scale, held isotropic" if carried else "global scaling"
        print(f"[mli_quad] {label}: {format_aniso_scale_summary(aniso)}", file=out)
        if carried:
            # Zeros on the line above are the constraint, not a failed fit. Print the
            # tensor that does carry the anisotropy right beside it.
            p = getattr(self, "_sigma_wilson_params", None) or {}
            eig = p.get("b_eigenvalues") or []
            d_b = p.get("delta_b_aniso")
            if len(eig) == 3 and d_b is not None:
                print(
                    f"[mli_quad] data anisotropy is in Σ_W [{p.get('wilson_model', '?')}]: "
                    f"B eigenvalues {float(eig[0]):.2f}/{float(eig[1]):.2f}/"
                    f"{float(eig[2]):.2f} Å², ΔB_aniso={float(d_b):.3f} Å²",
                    file=out,
                )
        mismatch = self._aniso_scale_disagreement()
        if mismatch:
            print(
                f"[mli_quad] WARNING: stored b_cart disagrees with the applied "
                f"anisotropic scale ({mismatch}); the fitted values above are the ones "
                f"multiplying F_calc.",
                file=out,
            )

    def aniso_scale(self, refresh: bool = False) -> Any:
        """Overall anisotropic B of the global scale, recovered from ``k_anisotropic``.

        ``b_cart`` on this object is deliberately ``None`` -- it exists so mmtbx internals
        that touch the attribute do not trip -- so the anisotropic B has to come from the
        scale array that is actually applied to F_calc. That is the better source anyway:
        it reports the anisotropy the target sees rather than what a scaling object claims
        to have fitted. Cached; pass ``refresh=True`` after re-scaling.
        """
        from phridge.client.intensity.stats_report import fit_aniso_scale

        cached = getattr(self, "_aniso_scale", None)
        if cached is not None and not refresh:
            return cached
        result = None
        try:
            k_aniso = np.asarray(self.k_anisotropic(), dtype=np.float64)
            result = fit_aniso_scale(
                # flex.miller_index is a sequence of triples, not an array; list() is
                # what the rest of the client uses to get an (N, 3) block out of it.
                miller_indices=np.asarray(list(self._i_obs.indices()), dtype=np.float64),
                k_anisotropic=k_aniso,
                unit_cell=tuple(float(x) for x in self._i_obs.unit_cell().parameters()),
            )
        except Exception as exc:
            print(f"[mli_quad] anisotropic B unavailable: {exc}", file=sys.stderr, flush=True)
        self._aniso_scale = result
        return result

    def _aniso_scale_disagreement(self) -> str:
        """Describe any gap between a populated ``b_cart`` and the fitted tensor.

        Empty when nothing was stored to compare against, which is the normal case here.
        If some scaling path does write ``b_cart``, a mismatch means the deposited
        anisotropic B would not be the one applied to F_calc -- worth saying out loud,
        for the same reason the R fields are cross-checked.
        """
        stored = self.__dict__.get("b_cart")
        fitted = self.aniso_scale()
        if stored is None or fitted is None or not fitted.is_valid:
            return ""
        try:
            theirs = [float(x) for x in stored]
        except (TypeError, ValueError):
            return ""
        if len(theirs) != 6:
            return ""
        worst = max(abs(t - f) for t, f in zip(theirs, fitted.b_cart))
        if worst <= 0.05:  # well below anything that changes an interpretation
            return ""
        return f"max component difference {worst:.3f} A**2"

    def _attach_cc_isig_table(self, report: Any) -> None:
        """Build the CC_I (fixed I/sigma bin x resolution) table onto the report.

        ``I_calc = |F_model|^2`` on the same reflection list the target uses, matching
        the worker's ``cc_intensity_*`` definition so the table's marginals line up with
        the banner value. ``sigma_a`` is passed through so each resolution shell reports
        the sigma_A that goes with it -- reading down a column is then a sigma_A scan.
        """
        from phridge.client.intensity.stats_report import compute_cc_isig_table

        try:
            i_obs = self._i_obs
            if i_obs is None or i_obs.sigmas() is None:
                return
            data = self.f_model().data()
            # flex arrays expose as_numpy_array; plain arrays do not. Accept either
            # rather than hard-coding the cctbx spelling.
            arr = data.as_numpy_array() if hasattr(data, "as_numpy_array") else np.asarray(data)
            i_calc = np.abs(arr) ** 2
            if i_calc.size != i_obs.size():
                raise ValueError(
                    f"f_model has {i_calc.size} reflections, i_obs has {i_obs.size()}"
                )
            flags = self._r_free_flags
            report.cc_isig = compute_cc_isig_table(
                intensities=i_obs.data(),
                sigmas=i_obs.sigmas(),
                d_spacings=i_obs.d_spacings().data(),
                i_calc=i_calc,
                r_free=flags.data() if flags is not None else None,
                sigma_a=getattr(self, "sigma_a", None),
            )
        except Exception as exc:
            print(f"[mli_quad] CC_I table skipped: {exc}", file=sys.stderr, flush=True)

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
        nu_p = getattr(self, "_nu_params", None) or {}
        if nu_p.get("mode") in ("bins", "grid_bins") and nu_p.get("bin_nu"):
            bn = nu_p["bin_nu"]
            kind = "grid" if nu_p.get("mode") == "grid_bins" else "LBFGS"
            print(
                f"  ν(s) bins ({kind}): n={nu_p.get('n_bins')}  "
                f"values={' '.join('G' if float(v) >= 199 else f'{float(v):g}' for v in bn)}",
                file=target_out,
            )
        try:
            from phridge.client.intensity.stats_report import format_nu_grid_profiles

            for line in format_nu_grid_profiles(nu_p):
                print(line, file=target_out)
        except Exception:
            p_nu, p_nll = nu_p.get("profile_nu"), nu_p.get("profile_nll")
            if p_nu and p_nll and len(p_nu) == len(p_nll):
                g_nll = nu_p.get("profile_nll_gaussian")
                best = min(range(len(p_nll)), key=lambda i: p_nll[i])
                ref = g_nll if g_nll is not None else p_nll[best]
                kind = "σ_A,β refit" if nu_p.get("mode") == "grid" else "σ_A frozen"
                body = "  ".join(f"{n:g}:{v - ref:+.4f}" for n, v in zip(p_nu, p_nll))
                print(f"  ν profile ΔNLL vs Gaussian (per refl, {kind}): {body}", file=target_out)
                print(
                    f"    best ν={p_nu[best]:g} (ΔNLL={p_nll[best] - ref:+.4f}); "
                    f"Gaussian NLL={ref:.4f}",
                    file=target_out,
                )
        rv = self.inferred_r_values()
        cw = rv.get("cc_post_work", rv.get("cc_work", float("nan")))
        cf = rv.get("cc_post_free", rv.get("cc_free", float("nan")))
        print(f"  CC:  cc_work={cw:.4f} cc_free={cf:.4f}", file=target_out)
        rfw_w, rfw_f, rfw_a = self.r_work(), self.r_free(), self.r_all()
        if any(np.isfinite(float(x)) for x in (rfw_w, rfw_f, rfw_a)):
            print(
                f"  R (French-Wilson amplitudes):  r_work={rfw_w:.4f} "
                f"r_free={rfw_f:.4f} all={rfw_a:.4f}",
                file=target_out,
            )
        if "r_intensity_work" in rv and "r_intensity_free" in rv:
            print(f"  Direct Intensity R (I):  r_work={rv['r_intensity_work']:.4f} r_free={rv['r_intensity_free']:.4f} all={rv.get('r_intensity_all', float('nan')):.4f}", file=target_out)
        s_w = rv.get("s_post_work", float("nan"))
        s_f = rv.get("s_post_free", float("nan"))
        sp_w = rv.get("s_prior_work", float("nan"))
        sp_f = rv.get("s_prior_free", float("nan"))
        k_w = rv.get("k_s_work", float("nan"))
        if any(np.isfinite(float(x)) for x in (s_w, s_f, sp_w, sp_f, k_w)):
            print(
                f"  S_post: work={float(s_w):.4f} free={float(s_f):.4f}  "
                f"S_prior={float(sp_w):.4f}/{float(sp_f):.4f}  k_S={float(k_w):.4f}",
                file=target_out,
            )
        elif rv.get("s_report_error"):
            print(f"  S_post: unavailable ({rv['s_report_error']})", file=target_out)
        elif rv.get("s_report_version") != 9 and "r_intensity_work" in rv:
            print(
                "  S_post: stale worker — restart with --redis",
                file=target_out,
            )
        # Point estimates of the amplitude agreement. These shrink toward sigma_A E_C
        # as the data weaken, so they improve when the data get worse — kept for
        # diagnosis, never presented as an R factor.
        rw = rv.get("r_post_work", float("nan"))
        rf = rv.get("r_post_free", float("nan"))
        ra = rv.get("r_post_all", float("nan"))
        if any(np.isfinite(float(x)) for x in (rw, rf, ra)):
            print(
                "  shrunken-amplitude agreement (biased low — diagnostic, not an R factor):",
                file=target_out,
            )
            print(f"    posterior mean <F>: work={rw:.4f} free={rf:.4f} all={ra:.4f}", file=target_out)
            if "r_mode_work" in rv and "r_mode_free" in rv:
                print(
                    f"    posterior mode:     work={rv['r_mode_work']:.4f} free={rv['r_mode_free']:.4f}",
                    file=target_out,
                )
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
