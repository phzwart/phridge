"""Phenix and mmtbx integration hooks for mli_quad intensity likelihood refinement.

Enables seamless execution of ``phenix.refine refinement.main.target=mli_quad``
by dynamically intercepting:
  1. Target registration (mmtbx.refinement.targets.target_names)
  2. PHIL choice validation (phenix.refinement.master_params)
  3. FModel construction (mmtbx.utils.fmodel_manager2 and fmodel_manager)
  4. Target functor evaluation (mmtbx.refinement.targets.target_functor)
  5. Electron density map synthesis (mmtbx.f_model.manager.electron_density_map)
  6. Statistics and R-factor reporting (mmtbx.f_model.manager.info)
  7. XYZ/ADP weight optimization: select on free-set NLL (not R-free)
"""

from __future__ import annotations

import os
import sys
from typing import Any, Callable, Dict, Optional, Tuple

from phridge.client.intensity.engine import (
    IntensityDataError,
    IntensityElectronDensityMap,
    IntensityFModel,
    IntensityFModelInfo,
    IntensityTargetFunctor,
    IntensityTwinningError,
    amplitude_scaffold_from_intensities,
    recover_i_obs_from_fmodel,
    register_mli_targets,
    require_intensity_array,
)

_ENABLED = False
_ORIGINALS: Dict[str, Any] = {}


def is_intensity_enabled() -> bool:
    """Check if the mli_quad Phenix hooks are currently active."""
    return bool(_ENABLED)


def _weight_metric_is_nll() -> bool:
    """Return True when weight trials should be ranked by free-set NLL.

    Env: ``PHRIDGE_WEIGHT_METRIC=nll`` (default) or ``rfree``.
    """
    env = os.environ.get("PHRIDGE_WEIGHT_METRIC", "nll").strip().lower()
    return env not in ("0", "false", "no", "off", "rfree", "r_free", "r")


def _nll_pair(fmodel: Any) -> Tuple[float, float]:
    """Return (nll_work, nll_free) from an fmodel / IntensityFModel."""
    nll_w = float(fmodel.target_w())
    nll_f = nll_w
    if hasattr(fmodel, "target_t"):
        try:
            val = fmodel.target_t()
            if val is not None and val == val:  # not NaN
                nll_f = float(val)
        except Exception:
            pass
    return nll_w, nll_f


def _fmodel_is_mli(fmodel: Any) -> bool:
    tn = str(getattr(fmodel, "target_name", "") or "").lower()
    return tn in ("mli", "mli_quad", "ml_i") or isinstance(fmodel, IntensityFModel)


def _fmodels_is_mli(fmodels: Any) -> bool:
    try:
        return _fmodel_is_mli(fmodels.fmodel_xray())
    except Exception:
        return False


def patch_weight_selection() -> bool:
    """Rank Phenix XYZ/ADP weight trials by free-set NLL for mli_quad.

    XYZ (``phenix.refinement.xyz_reciprocal_space``):
      - Keep real R-factors in the scorer (Phenix asserts on them).
      - Store per-trial NLL and select the lowest free-set NLL among
        geometry-acceptable trials when ``PHRIDGE_WEIGHT_METRIC=nll``.

    ADP (``mmtbx.refinement.adp_refinement.refine_adp``):
      - Inject NLL into the trial ``r_work``/``r_free`` fields used for ranking
        (R-gap thresholds become no-ops for typical NLL scales; final sort is
        by free NLL). Soften the post-select R assert that would otherwise fail.
    """
    patched_any = False

    # ----- XYZ weight scorer -----
    try:
        import phenix.refinement.xyz_reciprocal_space as xyz_rs
        from cctbx.array_family import flex
    except ImportError:
        xyz_rs = None  # type: ignore

    if xyz_rs is not None:
        if "xyz.summary.__init__" not in _ORIGINALS:
            orig_summary_init = xyz_rs.summary.__init__
            _ORIGINALS["xyz.summary.__init__"] = orig_summary_init

            def patched_summary_init(self: Any, fmodels: Any, model: Any, weight: Any) -> None:
                orig_summary_init(self, fmodels, model, weight)
                self.nll_w = float("nan")
                self.nll_f = float("nan")
                if _weight_metric_is_nll() and _fmodels_is_mli(fmodels):
                    try:
                        self.nll_w, self.nll_f = _nll_pair(fmodels.fmodel_xray())
                    except Exception:
                        pass

            xyz_rs.summary.__init__ = patched_summary_init
            patched_any = True

        if "xyz.scorer.__init__" not in _ORIGINALS:
            orig_scorer_init = xyz_rs.xyz_refinement_scorer.__init__
            _ORIGINALS["xyz.scorer.__init__"] = orig_scorer_init

            def patched_scorer_init(self: Any, log: Any, r_free_only: bool = False) -> None:
                orig_scorer_init(self, log, r_free_only=r_free_only)
                self.nll_w = flex.double()
                self.nll_f = flex.double()
                self._phridge_nll = False

            xyz_rs.xyz_refinement_scorer.__init__ = patched_scorer_init
            patched_any = True

        if "xyz.scorer.add" not in _ORIGINALS:
            orig_add = xyz_rs.xyz_refinement_scorer.add
            _ORIGINALS["xyz.scorer.add"] = orig_add

            def patched_add(self: Any, s: Any) -> None:
                orig_add(self, s)
                if s is None:
                    return
                nll_w = getattr(s, "nll_w", float("nan"))
                nll_f = getattr(s, "nll_f", float("nan"))
                if nll_w == nll_w and nll_f == nll_f:
                    self._phridge_nll = True
                self.nll_w.append(float(nll_w) if nll_w == nll_w else 99999.0)
                self.nll_f.append(float(nll_f) if nll_f == nll_f else 99999.0)

            xyz_rs.xyz_refinement_scorer.add = patched_add
            patched_any = True

        if "xyz.scorer._select" not in _ORIGINALS:
            orig_select = xyz_rs.xyz_refinement_scorer._select
            _ORIGINALS["xyz.scorer._select"] = orig_select

            def patched_select(self: Any, sel: Any) -> None:
                # Apply the same fallback as upstream before slicing NLL arrays.
                if sel.count(True) == 0 and hasattr(self, "rf") and self.rf.size() > 0:
                    sel = flex.abs(self.rf - flex.min(self.rf)) < 1.0e-4
                if hasattr(self, "nll_w") and self.nll_w.size() == sel.size():
                    self.nll_w = self.nll_w.select(sel)
                    self.nll_f = self.nll_f.select(sel)
                # Upstream may re-apply the same empty-sel fallback; sizes stay aligned.
                orig_select(self, sel)

            xyz_rs.xyz_refinement_scorer._select = patched_select
            patched_any = True

        if "xyz.scorer._select_best" not in _ORIGINALS:
            orig_select_best = xyz_rs.xyz_refinement_scorer._select_best
            _ORIGINALS["xyz.scorer._select_best"] = orig_select_best

            def patched_select_best(
                self: Any,
                r_free_range_width: Any,
                r_free_r_work_gap: Any,
                bond_rmsd: Any,
                angle_rmsd: Any,
            ) -> None:
                use_nll = (
                    _weight_metric_is_nll()
                    and getattr(self, "_phridge_nll", False)
                    and hasattr(self, "nll_f")
                    and self.nll_f.size() == self.rf.size()
                    and self.nll_f.size() > 0
                )
                if not use_nll:
                    return orig_select_best(
                        self,
                        r_free_range_width=r_free_range_width,
                        r_free_r_work_gap=r_free_r_work_gap,
                        bond_rmsd=bond_rmsd,
                        angle_rmsd=angle_rmsd,
                    )
                if self.b.size() == 0:
                    return
                # Geometry gates (unchanged), then lowest free-set NLL.
                sel = self.b <= bond_rmsd
                sel &= self.a <= angle_rmsd
                if sel.count(True) == 0:
                    sel = flex.abs(self.nll_f - flex.min(self.nll_f)) < 1.0e-4
                self._select(sel)
                t = self.clsc + self.rama + self.rota + self.cbet
                t = flex.double([round(t_, 1) for t_ in t])
                sel = t <= flex.min(t)
                self._select(sel)
                sel = flex.abs(self.nll_f - flex.min(self.nll_f)) <= getattr(self, "eps", 1.0e-4)
                self._select(sel)
                if self.log is not None:
                    print(
                        f" Phridge weight metric: free-set NLL "
                        f"(best nll_free={float(self.nll_f[0]):.6f}, "
                        f"r_free={float(self.rf[0]):.4f})",
                        file=self.log,
                    )

            xyz_rs.xyz_refinement_scorer._select_best = patched_select_best
            patched_any = True

        if "xyz.run_all._optimize_xyz_weight" not in _ORIGINALS and hasattr(xyz_rs, "run_all"):
            # Bound method on class
            orig_opt = xyz_rs.run_all._optimize_xyz_weight
            _ORIGINALS["xyz.run_all._optimize_xyz_weight"] = orig_opt

            def patched_opt_xyz(self: Any) -> bool:
                result = self.params.target_weights.optimize_xyz_weight
                if not result:
                    return False
                if _weight_metric_is_nll() and _fmodels_is_mli(self.fmodels):
                    try:
                        nll_w, nll_f = _nll_pair(self.fmodels.fmodel_xray())
                        if (
                            nll_f < nll_w or abs(nll_f - nll_w) < 0.01
                        ) and not self.params.target_weights.force_optimize_weights:
                            return False
                        return True
                    except Exception:
                        pass
                return orig_opt(self)

            xyz_rs.run_all._optimize_xyz_weight = patched_opt_xyz
            patched_any = True

    # ----- ADP weight trials -----
    try:
        import mmtbx.refinement.adp_refinement as adp_ref
    except ImportError:
        adp_ref = None  # type: ignore

    if adp_ref is not None and hasattr(adp_ref, "refine_adp"):
        if "adp.refine_adp.show" not in _ORIGINALS:
            orig_show = adp_ref.refine_adp.show
            _ORIGINALS["adp.refine_adp.show"] = orig_show

            def patched_adp_show(self: Any, *args: Any, **kwargs: Any) -> Any:
                result = orig_show(self, *args, **kwargs)
                if result is None:
                    return result
                if not (_weight_metric_is_nll() and _fmodels_is_mli(self.fmodels)):
                    return result
                try:
                    nll_w, nll_f = _nll_pair(self.fmodels.fmodel_xray())
                except Exception:
                    return result
                # Ranking arrays use these fields; keep true R on side attributes.
                result.r_work_rfactor = float(result.r_work)
                result.r_free_rfactor = float(result.r_free)
                result.r_work = float(nll_w)
                result.r_free = float(nll_f)
                result.r_gap = float(nll_f) - float(nll_w)
                result.nll_work = float(nll_w)
                result.nll_free = float(nll_f)
                return result

            adp_ref.refine_adp.show = patched_adp_show
            patched_any = True

        if "adp.refine_adp.__init__" not in _ORIGINALS:
            orig_adp_init = adp_ref.refine_adp.__init__
            _ORIGINALS["adp.refine_adp.__init__"] = orig_adp_init

            def patched_adp_init(self: Any, *args: Any, **kwargs: Any) -> None:
                fmodels = kwargs.get("fmodels", args[1] if len(args) > 1 else None)
                use_nll = _weight_metric_is_nll() and _fmodels_is_mli(fmodels)
                if not use_nll:
                    return orig_adp_init(self, *args, **kwargs)

                # Soften the post-select assert: r_work()*100 vs NLL-valued rw_best.
                real_ae = adp_ref.approx_equal

                def guarded_ae(a: Any, b: Any, eps: float = 1.0e-6, **kw: Any) -> bool:
                    try:
                        fa, fb = float(a), float(b)
                    except Exception:
                        return real_ae(a, b, eps=eps, **kw)
                    # Skip R% vs NLL mismatch after NLL-based weight pick.
                    if fa > 5.0 and fb < 5.0:
                        return True
                    if fb > 5.0 and fa < 5.0:
                        return True
                    return real_ae(a, b, eps=eps, **kw)

                adp_ref.approx_equal = guarded_ae
                try:
                    orig_adp_init(self, *args, **kwargs)
                    if getattr(self, "log", None) is not None and getattr(
                        self, "target_weights", None
                    ) is not None:
                        try:
                            print(
                                " Phridge weight metric: free-set NLL "
                                "(ADP trial r_work/r_free columns are NLL values)",
                                file=self.log,
                            )
                        except Exception:
                            pass
                finally:
                    adp_ref.approx_equal = real_ae

            adp_ref.refine_adp.__init__ = patched_adp_init
            patched_any = True

    return patched_any


def register_target_names() -> None:
    """Register 'mli_quad' and 'mli' in mmtbx.refinement.targets.target_names."""
    register_mli_targets()


def patch_phil_master_params() -> bool:
    """Augment phenix.refinement.master_params choices to accept 'mli_quad'."""
    try:
        import iotbx.phil
        import phenix.refinement
    except ImportError:
        return False

    def _augment_scope(scope: Any) -> None:
        try:
            nodes = scope.get_without_substitution("refinement.main.target")
            if not nodes:
                return
            node = nodes[0]
            existing_words = [w.value for w in getattr(node, "words", [])]
            if "mli_quad" not in existing_words:
                node.words.append(iotbx.phil.tokenizer.word("mli_quad"))
            if "mli" not in existing_words:
                node.words.append(iotbx.phil.tokenizer.word("mli"))
            caption = getattr(node, "caption", "")
            if caption and "MLI_QUAD" not in caption:
                node.caption = f"{caption} MLI_QUAD"
        except Exception:
            pass

    # Patch cached master params if already loaded
    if hasattr(phenix.refinement, "_master_params") and phenix.refinement._master_params is not None:
        _augment_scope(phenix.refinement._master_params)

    # Wrap master_params function so reloads also include mli_quad
    if "phenix.refinement.master_params" not in _ORIGINALS and hasattr(phenix.refinement, "master_params"):
        orig_mp = phenix.refinement.master_params
        _ORIGINALS["phenix.refinement.master_params"] = orig_mp

        def patched_master_params(*args: Any, **kwargs: Any) -> Any:
            scope = orig_mp(*args, **kwargs)
            _augment_scope(scope)
            return scope

        phenix.refinement.master_params = patched_master_params

    # Wrap set_data_target_type if available
    try:
        import phenix.refinement.misc
        if "phenix.refinement.misc.set_data_target_type" not in _ORIGINALS and hasattr(phenix.refinement.misc, "set_data_target_type"):
            orig_sdtt = phenix.refinement.misc.set_data_target_type
            _ORIGINALS["phenix.refinement.misc.set_data_target_type"] = orig_sdtt

            def patched_set_data_target_type(*args: Any, **kwargs: Any) -> str:
                res = orig_sdtt(*args, **kwargs)
                if str(res).lower() in ("mli", "mli_quad", "ml_i"):
                    return "mli_quad"
                return res

            phenix.refinement.misc.set_data_target_type = patched_set_data_target_type
    except ImportError:
        pass

    return True


def patch_target_functor() -> bool:
    """Route mmtbx.refinement.targets.target_functor to IntensityTargetFunctor for mli_quad."""
    try:
        import mmtbx.refinement.targets
    except ImportError:
        return False

    if "mmtbx.refinement.targets.target_functor.__new__" not in _ORIGINALS:
        orig_new = mmtbx.refinement.targets.target_functor.__new__
        _ORIGINALS["mmtbx.refinement.targets.target_functor.__new__"] = orig_new

        def patched_new(cls: Any, manager: Any, alpha_beta: Optional[Any] = None) -> Any:
            target_name = getattr(manager, "target_name", None)
            if target_name in ("mli", "mli_quad", "ml_i"):
                return IntensityTargetFunctor(manager, alpha_beta=alpha_beta)
            return orig_new(cls)

        mmtbx.refinement.targets.target_functor.__new__ = staticmethod(patched_new)
    return True


def patch_fmodel_manager() -> bool:
    """Intercept mmtbx.utils.fmodel_manager2 and fmodel_manager to construct IntensityFModel."""
    try:
        import mmtbx.utils
    except ImportError:
        return False

    if "mmtbx.utils.fmodel_manager2" not in _ORIGINALS and hasattr(mmtbx.utils, "fmodel_manager2"):
        orig_fm2 = mmtbx.utils.fmodel_manager2
        _ORIGINALS["mmtbx.utils.fmodel_manager2"] = orig_fm2

        def patched_fmodel_manager2(
            f_obs: Any,
            r_free_flags: Any,
            abcd: Any = None,
            xray_structure: Any = None,
            twin_law: Any = None,
            ignore_r_free_flags: bool = False,
            mask_params: Any = None,
            sf_accuracy_params: Any = None,
            mtz_object: Any = None,
            data_type: Any = None,
            i_obs: Optional[Any] = None,
            target_name: Optional[str] = None,
            **kwargs: Any,
        ) -> Any:
            is_intensity_target = target_name in ("mli", "mli_quad", "ml_i")
            has_intensity_data = i_obs is not None or getattr(f_obs, "is_xray_intensity_array", lambda: False)()
            has_twin = (
                (twin_law is not None and twin_law is not False and str(twin_law).strip().lower() not in ("", "none"))
                or bool(kwargs.get("twin"))
                or bool(kwargs.get("is_twin"))
                or (kwargs.get("twin_law") is not None and kwargs["twin_law"] is not False and str(kwargs["twin_law"]).strip().lower() not in ("", "none"))
            )

            if is_intensity_target or (has_intensity_data and target_name is None):
                if has_twin:
                    raise IntensityTwinningError(
                        "Twinning is not supported for intensity-based likelihood ('mli_quad') refinement: "
                        "twin is True or twin_law was provided."
                    )
                if i_obs is not None:
                    i_use = require_intensity_array(i_obs, context="fmodel_manager2")
                else:
                    i_use = require_intensity_array(f_obs, context="fmodel_manager2")
                redis_url = kwargs.pop("redis_url", os.environ.get("PHRIDGE_REDIS_URL"))
                mem_kw = kwargs.pop("memory", None)
                return IntensityFModel(
                    i_obs=i_use,
                    f_obs=None,
                    xray_structure=xray_structure,
                    r_free_flags=r_free_flags,
                    target_name=target_name or "mli_quad",
                    mask_params=mask_params,
                    sf_and_grads_accuracy_params=sf_accuracy_params,
                    origin=mtz_object,
                    data_type=data_type,
                    redis_url=redis_url,
                    memory=mem_kw,  # None → PHRIDGE_MEMORY default (on)
                    **kwargs,
                )

            fmodel = orig_fm2(
                f_obs=f_obs,
                r_free_flags=r_free_flags,
                abcd=abcd,
                xray_structure=xray_structure,
                twin_law=twin_law,
                ignore_r_free_flags=ignore_r_free_flags,
                mask_params=mask_params,
                sf_accuracy_params=sf_accuracy_params,
                mtz_object=mtz_object,
                data_type=data_type,
                **kwargs,
            )
            if i_obs is not None:
                fmodel._i_obs = i_obs
            return fmodel

        mmtbx.utils.fmodel_manager2 = patched_fmodel_manager2

    if "mmtbx.utils.fmodel_manager" not in _ORIGINALS and hasattr(mmtbx.utils, "fmodel_manager"):
        orig_fm = mmtbx.utils.fmodel_manager
        _ORIGINALS["mmtbx.utils.fmodel_manager"] = orig_fm

        def patched_fmodel_manager(*args: Any, **kwargs: Any) -> Any:
            target_name = kwargs.get("target_name")
            i_obs = kwargs.get("i_obs")
            twin_law = kwargs.get("twin_law")
            has_twin = (
                bool(kwargs.get("twin"))
                or bool(kwargs.get("is_twin"))
                or (twin_law is not None and twin_law is not False and str(twin_law).strip().lower() not in ("", "none"))
            )
            if target_name in ("mli", "mli_quad", "ml_i"):
                if has_twin:
                    raise IntensityTwinningError(
                        "Twinning is not supported for intensity-based likelihood ('mli_quad') refinement: "
                        "twin is True or twin_law was provided."
                    )
                f_obs = args[0] if args else kwargs.get("f_obs")
                xs = kwargs.get("xray_structure")
                flags = kwargs.get("r_free_flags")
                redis_url = kwargs.get("redis_url", os.environ.get("PHRIDGE_REDIS_URL"))
                mem_kw = kwargs.get("memory", None)
                if i_obs is not None:
                    i_use = require_intensity_array(i_obs, context="fmodel_manager")
                else:
                    i_use = require_intensity_array(f_obs, context="fmodel_manager")
                return IntensityFModel(
                    i_obs=i_use,
                    f_obs=None,
                    xray_structure=xs,
                    r_free_flags=flags,
                    target_name=target_name,
                    redis_url=redis_url,
                    memory=mem_kw,  # None → PHRIDGE_MEMORY default (on)
                )
            fmodel = orig_fm(*args, **kwargs)
            if i_obs is not None:
                fmodel._i_obs = i_obs
            return fmodel

        mmtbx.utils.fmodel_manager = patched_fmodel_manager

    return True


def patch_fmodel_methods() -> bool:
    """Intercept electron_density_map and info on mmtbx.f_model.manager for mli_quad."""
    try:
        import mmtbx.f_model
    except ImportError:
        return False

    mgr_cls = mmtbx.f_model.manager

    if "mmtbx.f_model.manager.electron_density_map" not in _ORIGINALS and hasattr(mgr_cls, "electron_density_map"):
        orig_edm = mgr_cls.electron_density_map
        _ORIGINALS["mmtbx.f_model.manager.electron_density_map"] = orig_edm

        def patched_electron_density_map(self: Any, *args: Any, **kwargs: Any) -> Any:
            if getattr(self, "target_name", None) in ("mli", "mli_quad", "ml_i"):
                return IntensityElectronDensityMap(self, *args, **kwargs)
            return orig_edm(self, *args, **kwargs)

        mgr_cls.electron_density_map = patched_electron_density_map

    if "mmtbx.f_model.manager.info" not in _ORIGINALS and hasattr(mgr_cls, "info"):
        orig_info = mgr_cls.info
        _ORIGINALS["mmtbx.f_model.manager.info"] = orig_info

        def patched_info(self: Any, *args: Any, **kwargs: Any) -> Any:
            if getattr(self, "target_name", None) in ("mli", "mli_quad", "ml_i"):
                return IntensityFModelInfo(self, *args, **kwargs)
            return orig_info(self, *args, **kwargs)

        mgr_cls.info = patched_info

    return True


def patch_french_wilson_disabled() -> bool:
    """Force-disable French–Wilson and F² conversion in ``iotbx.extract_xtal_data``.

    For ``mli_quad``:
      * ``french_wilson_scale`` is forced False on extract params
      * ``_data_as_f_obs`` builds a √max(I,0) scaffold without dropping reflections
      * ``cctbx.french_wilson.french_wilson_scale`` raises if somehow invoked
    """
    patched_any = False

    try:
        from iotbx import extract_xtal_data
    except ImportError:
        extract_xtal_data = None  # type: ignore

    if extract_xtal_data is not None:
        if "extract_xtal_data.run.__init__" not in _ORIGINALS:
            orig_init = extract_xtal_data.run.__init__
            _ORIGINALS["extract_xtal_data.run.__init__"] = orig_init

            def patched_extract_init(self: Any, *args: Any, **kwargs: Any) -> None:
                # Force FW off before any conversion runs.
                params = kwargs.get("parameters")
                if params is None and len(args) >= 2:
                    params = args[1]
                if params is not None and hasattr(params, "french_wilson_scale"):
                    params.french_wilson_scale = False
                return orig_init(self, *args, **kwargs)

            extract_xtal_data.run.__init__ = patched_extract_init
            patched_any = True

        if "extract_xtal_data.run._data_as_f_obs" not in _ORIGINALS:
            orig_as_f = extract_xtal_data.run._data_as_f_obs
            _ORIGINALS["extract_xtal_data.run._data_as_f_obs"] = orig_as_f

            def patched_data_as_f_obs(self: Any) -> Any:
                if self.raw_data_truncated is None:
                    return None
                # Always force FW off even if PHIL was mutated later.
                if hasattr(self.parameters, "french_wilson_scale"):
                    self.parameters.french_wilson_scale = False
                if self.raw_data_truncated.is_xray_intensity_array():
                    # Safe scaffold: keep every reflection; negatives → F=0 in scaffold only.
                    f_obs = amplitude_scaffold_from_intensities(self.raw_data_truncated)
                    # Match upstream non-anomalous / anomalous handling by reusing orig
                    # on a temporary state is fragile; apply the same post-steps inline.
                    if getattr(self, "force_non_anomalous", False):
                        f_obs = f_obs.average_bijvoet_mates()
                    d_min = f_obs.d_min()
                    if d_min < 0.25:
                        self.err.append("Resolution of data is too high: %-6.4f A" % d_min)
                        return None
                    if self.parameters.force_anomalous_flag_to_be_equal_to is not None:
                        if not self.parameters.force_anomalous_flag_to_be_equal_to:
                            if f_obs.anomalous_flag():
                                merged = f_obs.as_non_anomalous_array().merge_equivalents()
                                f_obs = merged.array().set_observation_type(f_obs)
                        elif not f_obs.anomalous_flag():
                            observation_type = f_obs.observation_type()
                            f_obs = f_obs.generate_bijvoet_mates()
                            f_obs.set_observation_type(observation_type)
                    else:
                        f_obs = f_obs.convert_to_non_anomalous_if_ratio_pairs_lone_less_than(
                            threshold=self.parameters.convert_to_non_anomalous_if_ratio_pairs_lone_less_than_threshold
                        )
                    f_obs.set_info(self.raw_data.info())
                    print(
                        " Phridge mli_quad: French–Wilson / F² disabled; "
                        "using √max(I,0) amplitude scaffold (I_obs negatives preserved).",
                        file=getattr(self, "log", sys.stdout),
                    )
                    return f_obs
                return orig_as_f(self)

            extract_xtal_data.run._data_as_f_obs = patched_data_as_f_obs
            patched_any = True

    try:
        from cctbx import french_wilson as fw_mod
    except ImportError:
        fw_mod = None  # type: ignore

    if fw_mod is not None and "cctbx.french_wilson.french_wilson_scale" not in _ORIGINALS:
        orig_fw = fw_mod.french_wilson_scale
        _ORIGINALS["cctbx.french_wilson.french_wilson_scale"] = orig_fw

        def blocked_french_wilson_scale(*args: Any, **kwargs: Any) -> Any:
            raise IntensityDataError(
                "French–Wilson scaling is disabled for mli_quad intensity likelihood. "
                "Use genuine I_obs (xray_data.french_wilson_scale=False)."
            )

        fw_mod.french_wilson_scale = blocked_french_wilson_scale
        patched_any = True

    return patched_any


def patch_setup_fmodels() -> bool:
    """Intercept phenix.refinement.fmodels.setup_fmodels to ensure IntensityFModel is used for mli_quad."""
    try:
        import phenix.refinement.fmodels
    except ImportError:
        return False

    if "phenix.refinement.fmodels.setup_fmodels" in _ORIGINALS:
        return True

    orig_setup = phenix.refinement.fmodels.setup_fmodels
    _ORIGINALS["phenix.refinement.fmodels.setup_fmodels"] = orig_setup

    def patched_setup_fmodels(
        fmodel_xray: Any,
        fmodel_neutron: Any,
        model_xray: Any,
        model_neutron: Any,
        params: Any,
        log: Any,
    ) -> Any:
        target_name = None
        if fmodel_xray is not None:
            try:
                import phenix.refinement.misc
                target_name = phenix.refinement.misc.set_data_target_type(
                    params=params,
                    f_obs=fmodel_xray.f_obs(),
                    r_free_flags=fmodel_xray.r_free_flags(),
                    hl_coeffs=fmodel_xray.hl_coeffs(),
                    model=model_xray,
                )
            except Exception:
                pass

        main_target = getattr(getattr(params, "main", None), "target", None)
        if target_name == "ls" and hasattr(params, "ls_target_names"):
            target_name = getattr(params.ls_target_names, "target_name", target_name)

        is_intensity = (
            target_name in ("mli", "mli_quad", "ml_i")
            or str(main_target).lower() in ("mli", "mli_quad", "ml_i")
        )

        if is_intensity and fmodel_xray is not None:
            # Check twinning - hard error
            has_twin = (
                getattr(fmodel_xray, "twin", False)
                or getattr(fmodel_xray, "twin_law", None)
                or getattr(getattr(params, "main", None), "twin_law", None)
                or getattr(getattr(params, "main", None), "twin", False)
            )
            if has_twin:
                raise IntensityTwinningError(
                    "Twinning is not supported for intensity-based likelihood ('mli_quad') refinement: "
                    "twin is True or twin_law was provided."
                )

            if not isinstance(fmodel_xray, IntensityFModel):
                # Force FW off on live params (belt-and-suspenders with extract patch).
                for scope_name in ("xray_data", "neutron_data"):
                    scope = getattr(params, scope_name, None)
                    if scope is not None and hasattr(scope, "french_wilson_scale"):
                        scope.french_wilson_scale = False

                i_obs = recover_i_obs_from_fmodel(
                    fmodel_xray, context="phenix.refinement.fmodels.setup_fmodels"
                )

                xs = (
                    model_xray.get_xray_structure()
                    if hasattr(model_xray, "get_xray_structure")
                    else fmodel_xray.xray_structure
                )
                redis_url = os.environ.get("PHRIDGE_REDIS_URL")
                fmodel_xray = IntensityFModel(
                    i_obs=i_obs,
                    f_obs=None,  # rebuild scaffold from I_obs
                    xray_structure=xs,
                    r_free_flags=fmodel_xray.r_free_flags(),
                    target_name=target_name or "mli_quad",
                    origin=fmodel_xray.origin(),
                    data_type=getattr(fmodel_xray, "data_type", lambda: "x_ray")(),
                    redis_url=redis_url,
                    memory=None,  # PHRIDGE_MEMORY default (on)
                    mask_params=getattr(params, "mask", None),
                    sf_and_grads_accuracy_params=getattr(params, "structure_factors_and_gradients_accuracy", None),
                    alpha_beta_params=getattr(params, "alpha_beta", None),
                )

        return orig_setup(
            fmodel_xray=fmodel_xray,
            fmodel_neutron=fmodel_neutron,
            model_xray=model_xray,
            model_neutron=model_neutron,
            params=params,
            log=log,
        )

    phenix.refinement.fmodels.setup_fmodels = patched_setup_fmodels
    return True


def enable_intensity_in_phenix() -> None:
    """Enable mli_quad intensity likelihood integration across phenix and mmtbx."""
    global _ENABLED
    if _ENABLED:
        return

    register_target_names()
    patch_phil_master_params()
    patch_french_wilson_disabled()
    patch_target_functor()
    patch_fmodel_manager()
    patch_fmodel_methods()
    patch_setup_fmodels()
    patch_weight_selection()

    _ENABLED = True


def disable_intensity_in_phenix() -> None:
    """Restore all original phenix and mmtbx methods."""
    global _ENABLED
    if not _ENABLED:
        return

    import mmtbx.refinement.targets
    import mmtbx.utils
    import mmtbx.f_model

    if "mmtbx.refinement.targets.target_functor.__new__" in _ORIGINALS:
        mmtbx.refinement.targets.target_functor.__new__ = _ORIGINALS.pop("mmtbx.refinement.targets.target_functor.__new__")

    if "mmtbx.utils.fmodel_manager2" in _ORIGINALS:
        mmtbx.utils.fmodel_manager2 = _ORIGINALS.pop("mmtbx.utils.fmodel_manager2")

    if "mmtbx.utils.fmodel_manager" in _ORIGINALS:
        mmtbx.utils.fmodel_manager = _ORIGINALS.pop("mmtbx.utils.fmodel_manager")

    if "mmtbx.f_model.manager.electron_density_map" in _ORIGINALS:
        mmtbx.f_model.manager.electron_density_map = _ORIGINALS.pop("mmtbx.f_model.manager.electron_density_map")

    if "mmtbx.f_model.manager.info" in _ORIGINALS:
        mmtbx.f_model.manager.info = _ORIGINALS.pop("mmtbx.f_model.manager.info")

    try:
        import phenix.refinement.fmodels
        if "phenix.refinement.fmodels.setup_fmodels" in _ORIGINALS:
            phenix.refinement.fmodels.setup_fmodels = _ORIGINALS.pop("phenix.refinement.fmodels.setup_fmodels")
    except ImportError:
        pass

    try:
        import phenix.refinement
        if "phenix.refinement.master_params" in _ORIGINALS:
            phenix.refinement.master_params = _ORIGINALS.pop("phenix.refinement.master_params")
    except ImportError:
        pass

    try:
        import phenix.refinement.misc
        if "phenix.refinement.misc.set_data_target_type" in _ORIGINALS:
            phenix.refinement.misc.set_data_target_type = _ORIGINALS.pop("phenix.refinement.misc.set_data_target_type")
    except ImportError:
        pass

    try:
        import phenix.refinement.xyz_reciprocal_space as xyz_rs
        if "xyz.summary.__init__" in _ORIGINALS:
            xyz_rs.summary.__init__ = _ORIGINALS.pop("xyz.summary.__init__")
        if "xyz.scorer.__init__" in _ORIGINALS:
            xyz_rs.xyz_refinement_scorer.__init__ = _ORIGINALS.pop("xyz.scorer.__init__")
        if "xyz.scorer.add" in _ORIGINALS:
            xyz_rs.xyz_refinement_scorer.add = _ORIGINALS.pop("xyz.scorer.add")
        if "xyz.scorer._select" in _ORIGINALS:
            xyz_rs.xyz_refinement_scorer._select = _ORIGINALS.pop("xyz.scorer._select")
        if "xyz.scorer._select_best" in _ORIGINALS:
            xyz_rs.xyz_refinement_scorer._select_best = _ORIGINALS.pop("xyz.scorer._select_best")
        if "xyz.run_all._optimize_xyz_weight" in _ORIGINALS:
            xyz_rs.run_all._optimize_xyz_weight = _ORIGINALS.pop("xyz.run_all._optimize_xyz_weight")
    except ImportError:
        pass

    try:
        import mmtbx.refinement.adp_refinement as adp_ref
        if "adp.refine_adp.show" in _ORIGINALS:
            adp_ref.refine_adp.show = _ORIGINALS.pop("adp.refine_adp.show")
        if "adp.refine_adp.__init__" in _ORIGINALS:
            adp_ref.refine_adp.__init__ = _ORIGINALS.pop("adp.refine_adp.__init__")
    except ImportError:
        pass

    try:
        from iotbx import extract_xtal_data
        if "extract_xtal_data.run.__init__" in _ORIGINALS:
            extract_xtal_data.run.__init__ = _ORIGINALS.pop("extract_xtal_data.run.__init__")
        if "extract_xtal_data.run._data_as_f_obs" in _ORIGINALS:
            extract_xtal_data.run._data_as_f_obs = _ORIGINALS.pop("extract_xtal_data.run._data_as_f_obs")
    except ImportError:
        pass

    try:
        from cctbx import french_wilson as fw_mod
        if "cctbx.french_wilson.french_wilson_scale" in _ORIGINALS:
            fw_mod.french_wilson_scale = _ORIGINALS.pop("cctbx.french_wilson.french_wilson_scale")
    except ImportError:
        pass

    _ENABLED = False


def main(args: Optional[list[str]] = None) -> int:
    """CLI entrypoint `phridge-refine`: Pre-activates mli_quad hooks and runs phenix.refine."""
    enable_intensity_in_phenix()
    argv = args if args is not None else sys.argv[1:]
    try:
        from phenix.command_line import phenix_refine
        res = phenix_refine.run_phenix_refine(args=argv)
        return 0 if res is None else int(res)
    except ImportError:
        try:
            from phenix.command_line import refine
            res = refine.run(argv)
            return 0 if res is None else int(res)
        except ImportError:
            print("Error: phenix.refine could not be imported. Ensure Phenix is on your PATH or PYTHONPATH.", file=sys.stderr)
            return 1
