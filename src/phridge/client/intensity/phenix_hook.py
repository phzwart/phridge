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
    _resolve_intensity_engine,
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


def _adp_r_percent_vs_nll(a: Any, b: Any, eps: float = 1.0e-6) -> bool:
    """True when Phenix's post-select check is comparing R% to journal NLL.

    ``refine_adp`` asserts ``r_work()*100 == rw_best`` with ``eps=0.001``.
    Under ``PHRIDGE_WEIGHT_METRIC=nll`` we store NLL in ``r_work``, so
    ``rw_best`` is −log p(I) (around 6.5) and the left side is R as a percent
    (around 24). A cut of ``NLL < 5`` was written for −log p(Z) (around 0.8)
    and misses the journal scale: both numbers then sit above 5 and the
    assert kills the run after the winner is already installed. The
    ``target_w`` consistency assert does not pass ``eps``, so a real functor
    mismatch still fires.
    """
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return False
    if fa != fa or fb != fb:
        return False
    return float(eps) >= 0.001 and abs(fa - fb) > 1.0


def _nll_point(fmodel: Any) -> Any:
    """One exact evaluation as an :class:`~phridge.client.intensity.nll_log.NllPoint`.

    Prefers the engine's ``nll_point``, which produces the work and free means from a
    single functor call. ``target_w()`` followed by ``target_t()`` is two full
    quadratures for two numbers one call already computed, and a weight scan asks for
    the pair once per trial, so the difference is a scan's worth of evaluations.
    """
    from phridge.client.intensity.nll_log import NllPoint

    probe = getattr(fmodel, "nll_point", None)
    if callable(probe):
        try:
            point = probe()
            if point is not None and getattr(point, "ok", False):
                return point
        except Exception:
            pass
    try:
        nll_w = float(fmodel.target_w())
    except Exception:
        return NllPoint()
    nll_f = None
    getter = getattr(fmodel, "target_t", None)
    if callable(getter):
        try:
            val = getter()
            if val is not None and val == val:  # not NaN
                nll_f = float(val)
        except Exception:
            nll_f = None
    return NllPoint(work=nll_w, free=nll_f)


def _nll_pair(fmodel: Any) -> Tuple[float, float]:
    """``(nll_work, nll_free)``; free falls back to work when there is no free set."""
    point = _nll_point(fmodel)
    if not point.ok:
        raise ValueError("no NLL available from this fmodel")
    work = float(point.work)
    return work, float(point.free) if point.free is not None else work


def _fmodel_is_mli(fmodel: Any) -> bool:
    tn = str(getattr(fmodel, "target_name", "") or "").lower()
    return tn in ("mli", "mli_quad", "ml_i") or isinstance(fmodel, IntensityFModel)


def _fmodels_is_mli(fmodels: Any) -> bool:
    try:
        return _fmodel_is_mli(fmodels.fmodel_xray())
    except Exception:
        return False


# Sentinel the scorer arrays carry for a trial whose NLL could not be had. It has to be
# a large finite number rather than NaN because the selection takes flex.min over the
# array, and a NaN there would poison the comparison instead of losing it.
_NLL_SENTINEL = 99999.0


def _unsentinel(value: Any) -> Optional[float]:
    try:
        val = float(value)
    except Exception:
        return None
    return None if (val >= _NLL_SENTINEL or not (val == val)) else val


def _journal_from_fmodels(fmodels: Any, log: Any = None) -> Optional[Any]:
    """The run's NLL journal, or ``None`` when it is off or this is not an mli fmodel."""
    engine = _resolve_intensity_engine_from_fmodels(fmodels)
    if engine is None:
        return None
    try:
        journal = engine.nll_journal(log=log)
    except Exception:
        return None
    return journal if getattr(journal, "enabled", False) else None


def _xyz_trial_group(scorer: Any) -> Optional[Any]:
    """Snapshot the scorer's per-trial arrays into a scan, before selection prunes them.

    ``finalize`` selects in place, so after it runs the arrays hold only the winner and
    the scan is gone. This has to be taken on the way in.
    """
    from phridge.client.intensity.nll_log import NllPoint

    journal = getattr(scorer, "_phridge_journal", None)
    if journal is None or not getattr(scorer, "_phridge_nll", False):
        return None
    try:
        n = scorer.w.size()
        if n < 1 or scorer.nll_f.size() != n or scorer.nll_w.size() != n:
            return None
        group = journal.new_group(
            "XYZ target weight scan",
            "weight",
            extra_columns=("r_work", "r_free", "bonds", "angles"),
        )
        for i in range(n):
            group.add(
                float(scorer.w[i]),
                NllPoint(
                    work=_unsentinel(scorer.nll_w[i]), free=_unsentinel(scorer.nll_f[i])
                ),
                {
                    "r_work": float(scorer.rw[i]),
                    "r_free": float(scorer.rf[i]),
                    "bonds": float(scorer.b[i]),
                    "angles": float(scorer.a[i]),
                },
            )
        return group
    except Exception:
        return None


def _adp_trial_line(result: Any) -> str:
    """The NLL behind one ADP trial, at a precision that can separate two trials.

    Upstream prints ``xray_target`` at three decimals and the free value not at all,
    which is the one number the selection is made on.
    """
    from phridge.client.intensity.nll_log import fmt_value

    weight = getattr(result, "weight", None)
    return (
        f"  [nll] ADP trial weight={fmt_value(weight, '.4f')}: "
        f"work {fmt_value(getattr(result, 'nll_work', None), '.6f')} "
        f"free {fmt_value(getattr(result, 'nll_free', None), '.6f')}"
    )


def _log_adp_trial(refiner: Any, result: Any) -> None:
    """Print one ADP trial's NLL and record it in the scan, if a scan is open."""
    from phridge.client.intensity.nll_log import NllPoint

    try:
        if not getattr(result, "_phridge_printed", False):
            result._phridge_printed = True
            log = getattr(refiner, "log", None)
            if log is not None:
                print(_adp_trial_line(result), file=log)
        group = getattr(refiner, "_phridge_group", None)
        if group is None or getattr(refiner, "_phridge_trial_weight", None) is None:
            return
        group.add(
            float(result.weight),
            NllPoint(work=float(result.nll_work), free=float(result.nll_free)),
            {
                "r_work": float(result.r_work_rfactor),
                "r_free": float(result.r_free_rfactor),
                "delta_b": float(result.delta_b),
            },
        )
    except Exception:
        pass  # telemetry must never cost a trial


def _close_adp_scan(refiner: Any) -> None:
    """Emit the ADP scan, marking the weight Phenix ended up installing."""
    group = getattr(refiner, "_phridge_group", None)
    journal = getattr(refiner, "_phridge_journal", None)
    refiner._phridge_group = None
    if group is None or journal is None:
        return
    try:
        weights = getattr(refiner, "target_weights", None)
        chosen = getattr(getattr(weights, "adp_weights_result", None), "wx", None)
        if chosen is not None:
            group.selected_value = float(chosen)
        journal.add_group(group)
    except Exception:
        pass


def patch_weight_selection() -> bool:
    """Rank Phenix XYZ/ADP weight trials by free-set NLL for mli_quad, and log the scan.

    XYZ (``phenix.refinement.xyz_reciprocal_space``):
      - Keep real R-factors in the scorer (Phenix asserts on them).
      - Store per-trial NLL and select the lowest free-set NLL among
        geometry-acceptable trials when ``PHRIDGE_WEIGHT_METRIC=nll``.

    ADP (``mmtbx.refinement.adp_refinement.refine_adp``):
      - Inject NLL into the trial ``r_work``/``r_free`` fields used for ranking
        (R-gap thresholds become no-ops for typical NLL scales; final sort is
        by free NLL). Soften the post-select R assert that would otherwise fail.

    Both scans also record every trial's NLL into the run's journal. Upstream prints an
    R-factor table per trial, which under an NLL metric shows everything except the
    number the choice was made on; the scan tables make the decision auditable and, more
    to the point, show how far ahead the winner actually was.
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
                # The scorer is built from ``log`` alone and never sees an fmodel, so the
                # journal is carried to it on the summaries it accumulates.
                self._phridge_journal = None
                if _weight_metric_is_nll() and _fmodels_is_mli(fmodels):
                    try:
                        self.nll_w, self.nll_f = _nll_pair(fmodels.fmodel_xray())
                    except Exception:
                        pass
                    self._phridge_journal = _journal_from_fmodels(fmodels)

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
                self._phridge_journal = None

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
                self.nll_w.append(float(nll_w) if nll_w == nll_w else _NLL_SENTINEL)
                self.nll_f.append(float(nll_f) if nll_f == nll_f else _NLL_SENTINEL)
                if getattr(self, "_phridge_journal", None) is None:
                    self._phridge_journal = getattr(s, "_phridge_journal", None)

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
                # Geometry gates, then lowest free-set NLL. Phenix's min(clash+
                # rama+rota+cbet) filter is a relative ranking, not a cutoff:
                # applying it first lets a 0.4 clash difference (0.5 vs 0.9)
                # discard the whole scan and install the worst NLL.
                sel = self.b <= bond_rmsd
                sel &= self.a <= angle_rmsd
                if sel.count(True) == 0:
                    sel = flex.abs(self.nll_f - flex.min(self.nll_f)) < 1.0e-4
                self._select(sel)
                sel = flex.abs(self.nll_f - flex.min(self.nll_f)) <= getattr(self, "eps", 1.0e-4)
                self._select(sel)
                t = self.clsc + self.rama + self.rota + self.cbet
                t = flex.double([round(t_, 1) for t_ in t])
                sel = t <= flex.min(t)
                self._select(sel)
                if self.log is not None:
                    print(
                        f" Phridge weight metric: free-set NLL "
                        f"(best nll_free={float(self.nll_f[0]):.6f}, "
                        f"r_free={float(self.rf[0]):.4f} on French-Wilson amplitudes)",
                        file=self.log,
                    )

            xyz_rs.xyz_refinement_scorer._select_best = patched_select_best
            patched_any = True

        if "xyz.scorer.finalize" not in _ORIGINALS:
            orig_xyz_finalize = xyz_rs.xyz_refinement_scorer.finalize
            _ORIGINALS["xyz.scorer.finalize"] = orig_xyz_finalize

            def patched_xyz_finalize(self: Any, *a: Any, **kw: Any) -> Any:
                group = _xyz_trial_group(self)
                result = orig_xyz_finalize(self, *a, **kw)
                journal = getattr(self, "_phridge_journal", None)
                if group is not None and journal is not None:
                    try:
                        # finalize selects in place, so w[0] is now the winner.
                        if self.w.size() > 0:
                            group.selected_value = float(self.w[0])
                        journal.add_group(group)
                    except Exception:
                        pass
                return result

            xyz_rs.xyz_refinement_scorer.finalize = patched_xyz_finalize
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
                _log_adp_trial(self, result)
                return result

            adp_ref.refine_adp.show = patched_adp_show
            patched_any = True

        if "adp.refine_adp.try_weight" not in _ORIGINALS:
            orig_try_weight = adp_ref.refine_adp.try_weight
            _ORIGINALS["adp.refine_adp.try_weight"] = orig_try_weight

            def patched_try_weight(self: Any, weight: Any, print_stats: bool = False) -> Any:
                # ``show`` is also called for the pre-scan model and again for the winner,
                # so the scan only records what came from inside a trial.
                self._phridge_trial_weight = weight
                try:
                    return orig_try_weight(self, weight, print_stats=print_stats)
                finally:
                    self._phridge_trial_weight = None

            adp_ref.refine_adp.try_weight = patched_try_weight
            patched_any = True

        if "adp.weight_result.show" not in _ORIGINALS and hasattr(adp_ref, "weight_result"):
            orig_wr_show = adp_ref.weight_result.show
            _ORIGINALS["adp.weight_result.show"] = orig_wr_show

            def patched_wr_show(self: Any, out: Any, prefix: str = "") -> Any:
                """Carry the NLL onto the trial row printed by the parallel scan.

                With ``nproc>1`` the trials run in subprocesses whose stdout is discarded
                and the parent prints the pickled results, so the NLL attached in the child
                would otherwise never be seen.
                """
                result = orig_wr_show(self, out, prefix=prefix)
                if out is not None and not getattr(self, "_phridge_printed", False):
                    nll_w = getattr(self, "nll_work", None)
                    if nll_w is not None:
                        self._phridge_printed = True
                        try:
                            print(_adp_trial_line(self), file=out)
                        except Exception:
                            pass
                return result

            adp_ref.weight_result.show = patched_wr_show
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
                    if _adp_r_percent_vs_nll(a, b, eps=float(kw.get("eps", eps))):
                        return True
                    return real_ae(a, b, eps=eps, **kw)

                adp_ref.approx_equal = guarded_ae
                journal = _journal_from_fmodels(fmodels, log=kwargs.get("log"))
                self._phridge_trial_weight = None
                self._phridge_journal = journal
                self._phridge_group = (
                    journal.new_group(
                        "ADP target weight scan",
                        "weight",
                        extra_columns=("r_work", "r_free", "delta_b"),
                    )
                    if journal is not None
                    else None
                )
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
                    _close_adp_scan(self)

            adp_ref.refine_adp.__init__ = patched_adp_init
            patched_any = True

    return patched_any


def register_target_names() -> None:
    """Register 'mli_quad' and 'mli' in mmtbx.refinement.targets.target_names."""
    register_mli_targets()


_TARGET_MODE_PHIL = """\
target_mode = *exact interleaved
  .type = choice(multi=False)
  .short_caption = Intensity target evaluation mode
  .help = "exact evaluates the marginal intensity likelihood by quadrature at every \
target call. interleaved runs each macro cycle's inner machinery on a per-reflection \
Rice surrogate fitted to the exact score and curvature at a checkpoint, and spends the \
exact target only to adjudicate the block. The exact NLL remains the sole arbiter and \
the final macro cycle is always fully exact."
  .expert_level = 2
"""


def patch_phil_master_params() -> bool:
    """Augment phenix.refinement.master_params: accept 'mli_quad' and 'target_mode'."""
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
        # refinement.target_mode = exact | interleaved
        try:
            if scope.get_without_substitution("refinement.target_mode"):
                return
            refinement = scope.get_without_substitution("refinement")
            if not refinement:
                return
            addition = iotbx.phil.parse(_TARGET_MODE_PHIL)
            refinement[0].adopt_scope(addition)
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


class PhenixLbfgsBlockRunner:
    """:class:`~phridge.client.intensity.interleaved.BlockRunner` over Phenix's LBFGS.

    Phenix's ``mmtbx.refinement.minimization.lbfgs`` does the whole minimization in its
    constructor, so one construction is exactly one re-runnable inner block: the model
    on entry is the checkpoint, the model on exit is the block's proposal, and calling
    the original constructor again with a halved ``max_iterations`` is the halved block
    the rejection ladder asks for.

    Every interaction with Phenix internals is guarded. A version whose interface does
    not match degrades to exact mode rather than breaking a refinement.
    """

    def __init__(
        self,
        orig_init: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        fmodels: Any,
        model: Any = None,
    ) -> None:
        self._orig_init = orig_init
        self._instance = instance
        self._args = args
        self._kwargs = kwargs
        self._fmodels = fmodels
        self._model = model
        self._base_iterations = self._read_max_iterations()

    # -- budget -----------------------------------------------------------------
    def _termination(self) -> Any:
        return self._kwargs.get("lbfgs_termination_params")

    def _read_max_iterations(self) -> Optional[int]:
        term = self._termination()
        val = getattr(term, "max_iterations", None) if term is not None else None
        try:
            return int(val) if val is not None else None
        except Exception:
            return None

    def full_budget(self) -> Any:
        from phridge.client.intensity.interleaved import BlockBudget

        return BlockBudget(max_iterations=self._base_iterations, step_scale=1.0)

    # -- state ------------------------------------------------------------------
    def _xray_structure(self) -> Any:
        try:
            return self._fmodels.fmodel_xray().xray_structure
        except Exception:
            return getattr(self._fmodels, "xray_structure", None)

    def save_state(self) -> Any:
        xs = self._xray_structure()
        if xs is None:
            return None
        try:
            return xs.deep_copy_scatterers()
        except Exception:
            return None

    def restore_state(self, state: Any) -> None:
        if state is None:
            return
        try:
            self._fmodels.update_xray_structure(xray_structure=state, update_f_calc=True)
        except Exception:
            try:
                self._fmodels.fmodel_xray().update_xray_structure(
                    xray_structure=state, update_f_calc=True
                )
            except Exception:
                return
        if self._model is not None:
            try:
                self._model.set_sites_cart(state.sites_cart())
            except Exception:
                pass

    def interpolate_state(self, a: Any, b: Any, frac: float) -> Any:
        """Linear interpolation of the sites; ADPs stay at ``a``."""
        if a is None or b is None:
            return a
        try:
            mid = a.deep_copy_scatterers()
            sa, sb = a.sites_cart(), b.sites_cart()
            mid.set_sites_cart(sa + (sb - sa) * float(frac))
            return mid
        except Exception:
            return a

    # -- run --------------------------------------------------------------------
    def run(self, budget: Any) -> None:
        kwargs = dict(self._kwargs)
        term = self._termination()
        if term is not None and budget.max_iterations is not None:
            try:
                kwargs["lbfgs_termination_params"] = term.__class__(
                    max_iterations=int(budget.max_iterations)
                )
            except Exception:
                kwargs["lbfgs_termination_params"] = term
        self._orig_init(self._instance, *self._args, **kwargs)


def patch_minimization_blocks() -> bool:
    """Wrap Phenix's LBFGS: interleaved ladder when asked for, NLL journal always.

    Under ``refinement.target_mode=interleaved`` (or ``PHRIDGE_TARGET_MODE=interleaved``)
    the block runs through the accept/reject/halve ladder. In the default exact mode the
    constructor runs untouched but is bracketed by an NLL measurement, so the log says
    what each minimization stage bought on the target being minimized rather than only
    what it did to the R-factors. A non-intensity fmodel is left entirely alone.
    """
    try:
        import mmtbx.refinement.minimization as mini
    except ImportError:
        return False
    if not hasattr(mini, "lbfgs"):
        return False
    if "mmtbx.refinement.minimization.lbfgs.__init__" in _ORIGINALS:
        return True

    orig_init = mini.lbfgs.__init__
    _ORIGINALS["mmtbx.refinement.minimization.lbfgs.__init__"] = orig_init

    def patched_lbfgs_init(self: Any, *args: Any, **kwargs: Any) -> None:
        fmodels = kwargs.get("fmodels", args[0] if args else None)
        engine = _resolve_intensity_engine_from_fmodels(fmodels)
        controller = None
        if engine is not None:
            try:
                controller = engine.interleaved_controller(log=kwargs.get("log"))
            except Exception:
                controller = None
        if controller is None:
            if engine is None:
                return orig_init(self, *args, **kwargs)
            return _journalled_block(engine, orig_init, self, args, kwargs)

        runner = PhenixLbfgsBlockRunner(
            orig_init,
            self,
            args,
            kwargs,
            fmodels,
            model=kwargs.get("model"),
        )
        try:
            controller.run_macro_cycle(
                runner,
                final=_is_final_macro_cycle(engine),
                site=_block_site(engine, orig_init, self, args, kwargs),
            )
        except Exception as exc:
            # A controller failure must never cost the refinement: fall back to the
            # stock exact block.
            print(f"[interleaved] disabled for this block: {exc}", file=sys.stderr)
            engine.use_surrogate(False)
            return orig_init(self, *args, **kwargs)

    mini.lbfgs.__init__ = patched_lbfgs_init
    return True


def _journalled_block(
    engine: Any,
    orig_init: Callable[..., Any],
    instance: Any,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> None:
    """Run an exact-mode minimization stage inside an NLL journal stage.

    The stage label comes from the same ``refine_xyz`` / ``refine_adp`` flags the
    interleaved telemetry uses, so the two modes name the stages identically and their
    logs can be compared line for line. A journal failure must never cost the block.
    """
    from phridge.client.intensity.nll_log import StageKind

    try:
        site = _block_site(engine, orig_init, instance, args, kwargs)
        journal = engine.nll_journal(log=kwargs.get("log"))
        journal.set_prefix(site.when())
    except Exception:
        return orig_init(instance, *args, **kwargs)
    with journal.stage(site.where(), StageKind.model):
        return orig_init(instance, *args, **kwargs)


def _call_arguments(
    func: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """Resolve a call's arguments to names, whether they were passed positionally.

    Used for logging only, so a signature that will not bind yields the keywords alone
    rather than raising.
    """
    try:
        import inspect

        bound = inspect.signature(func).bind_partial(instance, *args, **kwargs)
        return {k: v for k, v in bound.arguments.items() if k != "self"}
    except Exception:
        return dict(kwargs)


def _block_site(
    engine: Any,
    orig_init: Callable[..., Any],
    instance: Any,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Any:
    """Name the refinement stage and macro cycle this LBFGS call belongs to.

    ``mmtbx.refinement.minimization.lbfgs`` takes ``refine_xyz`` / ``refine_adp`` /
    ``refine_occupancies`` flags and a ``macro_cycle`` number, which between them are
    exactly the "where" and "when" the interleaved telemetry needs. Anything the call
    does not supply falls back to the engine's own macro-cycle counter, and a signature
    that matches nothing still yields a usable label rather than raising.
    """
    from phridge.client.intensity.interleaved import BlockSite

    call = _call_arguments(orig_init, instance, args, kwargs)
    flags = [
        ("coordinates (xyz)", ("refine_xyz", "refine_sites")),
        ("B-factors (ADP)", ("refine_adp", "refine_u_iso")),
        ("occupancies", ("refine_occupancies", "refine_occ")),
    ]
    active = [name for name, keys in flags if any(bool(call.get(k)) for k in keys)]
    stage = " + ".join(active) + " minimization" if active else "LBFGS minimization"

    cycle = call.get("macro_cycle")
    try:
        cycle = int(cycle) if cycle is not None else None
    except Exception:
        cycle = None
    if cycle is None:
        cycle = getattr(engine, "macro_cycle_index", None) or None

    total = getattr(engine, "total_macro_cycles", None)
    try:
        total = int(total) if total else None
    except Exception:
        total = None

    return BlockSite(stage=stage, macro_cycle=cycle, total_macro_cycles=total)


def _resolve_intensity_engine_from_fmodels(fmodels: Any) -> Optional[Any]:
    if fmodels is None:
        return None
    try:
        fmodel = fmodels.fmodel_xray()
    except Exception:
        fmodel = fmodels
    if not _fmodel_is_mli(fmodel):
        return None
    return _resolve_intensity_engine(fmodel)


def _apply_target_mode(fmodel: Any, params: Any) -> None:
    """Copy ``refinement.target_mode`` and the macro-cycle count onto the fmodel.

    The PHIL parameter wins over ``PHRIDGE_TARGET_MODE`` when it is present, so a
    ``.eff`` file is reproducible without the environment.
    """
    from phridge.client.intensity.interleaved import TargetMode

    raw = getattr(params, "target_mode", None)
    if raw is not None:
        try:
            fmodel.target_mode = TargetMode(str(raw).strip().lower())
        except ValueError:
            pass
    try:
        n = getattr(getattr(params, "main", None), "number_of_macro_cycles", None)
        if n is not None:
            fmodel.total_macro_cycles = int(n)
    except Exception:
        pass
    if fmodel.target_mode is TargetMode.interleaved:
        print(
            "Phridge target_mode=interleaved: inner blocks run on a per-reflection "
            "surrogate; every block is adjudicated by an exact evaluation and the "
            "final macro cycle is fully exact.",
            flush=True,
        )


def _is_final_macro_cycle(engine: Any) -> bool:
    """True on the last macro cycle, which always runs fully exact."""
    total = getattr(engine, "total_macro_cycles", None)
    index = getattr(engine, "macro_cycle_index", None)
    if not total or not index:
        return False
    return int(index) >= int(total)


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

        if isinstance(fmodel_xray, IntensityFModel):
            _apply_target_mode(fmodel_xray, params)

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
    patch_minimization_blocks()

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
        if "xyz.scorer.finalize" in _ORIGINALS:
            xyz_rs.xyz_refinement_scorer.finalize = _ORIGINALS.pop("xyz.scorer.finalize")
        if "xyz.run_all._optimize_xyz_weight" in _ORIGINALS:
            xyz_rs.run_all._optimize_xyz_weight = _ORIGINALS.pop("xyz.run_all._optimize_xyz_weight")
    except ImportError:
        pass

    try:
        import mmtbx.refinement.adp_refinement as adp_ref
        if "adp.refine_adp.show" in _ORIGINALS:
            adp_ref.refine_adp.show = _ORIGINALS.pop("adp.refine_adp.show")
        if "adp.refine_adp.try_weight" in _ORIGINALS:
            adp_ref.refine_adp.try_weight = _ORIGINALS.pop("adp.refine_adp.try_weight")
        if "adp.weight_result.show" in _ORIGINALS:
            adp_ref.weight_result.show = _ORIGINALS.pop("adp.weight_result.show")
        if "adp.refine_adp.__init__" in _ORIGINALS:
            adp_ref.refine_adp.__init__ = _ORIGINALS.pop("adp.refine_adp.__init__")
    except ImportError:
        pass

    try:
        import mmtbx.refinement.minimization as mini
        if "mmtbx.refinement.minimization.lbfgs.__init__" in _ORIGINALS:
            mini.lbfgs.__init__ = _ORIGINALS.pop("mmtbx.refinement.minimization.lbfgs.__init__")
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
