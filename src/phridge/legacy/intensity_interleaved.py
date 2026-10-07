"""Trust-region variational-EM interleaving for intensity-likelihood refinement.

The exact marginal intensity target (``ml_i``) costs one adaptive quadrature per
evaluation, and a refinement macro cycle asks for hundreds: bulk-solvent scaling,
weight scans, minimizer line searches, simulated annealing. This module runs those
inner evaluations on a per-reflection Rice surrogate
(:mod:`phridge.contrib.intensity_ll.surrogate`) that speaks the stock ``ml_f``
interface, and spends the exact target only at block boundaries.

The loop, per macro cycle
-------------------------
1. **Checkpoint (exact).** One exact evaluation at the current model gives ``NLL_0``,
   the scores, the curvatures and the posterior moments. The surrogate arrays are
   fitted (or refreshed) here. Every statistic and every nuisance refit
   (``sigma_A(s)``, ``nu``) happens **here and only here**, from the exact posterior.
2. **Inner block (surrogate).** The macro cycle's inner machinery runs entirely on
   the ``ml_f`` path fed with the fitted ``(F_p, alpha_p(s), beta_p)``. No quadrature
   evaluation occurs during the block.
3. **Adjudication (exact).** One exact evaluation at the block's proposed model gives
   ``NLL_1``. Accept if ``NLL_1 <= NLL_0 + tol``. On rejection: refresh the surrogate
   at the midpoint model and re-run the block with the iteration and step budget
   halved; a second rejection falls back to running the macro cycle fully exact.

**The exact NLL is the sole arbiter.** No block is ever accepted, and no statistic is
ever computed, from a surrogate value. The final macro cycle always runs fully exact
regardless of mode, so deposited and published statistics never touch the surrogate.

Why the rejection ladder is the safety net rather than a bound
-------------------------------------------------------------
The surrogate matches the exact score *and* the exact curvature at the checkpoint,
which is deliberately not the EM lower bound (that one has the right gradient but
understates curvature by the missing information -- see the surrogate module
docstring). Giving up the bound buys Newton-quality inner convergence; the exact
accept/reject is what buys the safety back. With Student-t noise there is a second
reason to keep the ladder: the robust down-weighting is frozen into the surrogate at
the checkpoint, so a reflection that turns into an outlier mid-block is over-trusted
until the next refresh. A rising rejection rate is the signal to shorten blocks.

Optimizer independence
----------------------
The controller talks to a minimizer only through :class:`BlockRunner`
(``save_state`` / ``restore_state`` / ``interpolate_state`` / ``run(budget)``), so the
same ladder drives Phenix's LBFGS in host mode and any ``torch.optim`` optimizer --
LBFGS, Adam, AdamW, SGD -- through :class:`TorchOptimBlockRunner`.

Hard rule
---------
``F_p`` and ``beta_p`` are internal arrays. Nothing in this module's telemetry,
logging, or reporting may name or print them: ``F_p`` reads like an observed
amplitude and ``beta_p`` like an experimental variance.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Protocol, Sequence, runtime_checkable

import numpy as np
from pydantic import BaseModel, Field

# ---------------------------------------------------------------- modes and options
class TargetMode(str, Enum):
    """Value of ``refinement.target_mode``."""

    exact = "exact"
    interleaved = "interleaved"


class BlockOutcome(str, Enum):
    accepted = "accepted"
    rejected = "rejected"
    exact_fallback = "exact-fallback"
    exact = "exact"


def target_mode_from_env(default: TargetMode = TargetMode.exact) -> TargetMode:
    """``PHRIDGE_TARGET_MODE`` mirror of the ``refinement.target_mode`` PHIL parameter."""
    raw = os.environ.get("PHRIDGE_TARGET_MODE", "").strip().lower()
    if raw in ("interleaved", "interleave"):
        return TargetMode.interleaved
    if raw in ("exact", "0", "off", "false", "no"):
        return TargetMode.exact
    return default


class InterleavedOptions(BaseModel):
    """Controller options; PHIL / env exposed."""

    model_config = {"extra": "forbid"}

    tol: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "A block is accepted when NLL_1 <= NLL_0 + tol, in per-reflection-mean NLL "
            "units (the same units target_work() reports). Default 0.0: no uphill step "
            "is ever accepted on the exact target."
        ),
    )
    max_halvings: int = Field(
        default=1,
        ge=0,
        le=8,
        description="Rejected-block retries with the iteration and step budget halved before falling back to a fully exact macro cycle.",
    )
    refresh: str = Field(
        default="full",
        description=(
            "'full' refreshes every reflection's surrogate at each checkpoint (v1; it "
            "costs one exact evaluation you are already paying for). "
            "'visible_fraction' (v2, opt-in) refreshes only reflections whose rho2 is "
            "below rho2_threshold or whose |delta E_C| since the last refresh exceeds "
            "delta_ec_threshold: the EM contraction rate per reflection is the "
            "missing-information fraction 1 - rho2, so low-rho2 reflections are exactly "
            "the ones whose surrogates go stale fastest."
        ),
    )
    rho2_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    delta_ec_threshold: float = Field(default=1.0, gt=0.0)
    verbose: bool = Field(default=True, description="Emit one telemetry line per block.")

    def model_post_init(self, _context: Any) -> None:
        if self.refresh not in ("full", "visible_fraction"):
            raise ValueError("refresh must be 'full' or 'visible_fraction'")


def interleaved_options_from_env() -> InterleavedOptions:
    """Build options from the ``PHRIDGE_INTERLEAVED_*`` env mirrors."""

    def _f(name: str, default: float) -> float:
        raw = os.environ.get(name, "").strip()
        try:
            return float(raw) if raw else default
        except ValueError:
            return default

    def _i(name: str, default: int) -> int:
        raw = os.environ.get(name, "").strip()
        try:
            return int(raw) if raw else default
        except ValueError:
            return default

    refresh = os.environ.get("PHRIDGE_INTERLEAVED_REFRESH", "full").strip().lower() or "full"
    if refresh not in ("full", "visible_fraction"):
        refresh = "full"
    verbose = os.environ.get("PHRIDGE_INTERLEAVED_VERBOSE", "1").strip().lower()
    return InterleavedOptions(
        tol=_f("PHRIDGE_INTERLEAVED_TOL", 0.0),
        max_halvings=_i("PHRIDGE_INTERLEAVED_MAX_HALVINGS", 1),
        refresh=refresh,
        rho2_threshold=_f("PHRIDGE_INTERLEAVED_RHO2", 0.25),
        delta_ec_threshold=_f("PHRIDGE_INTERLEAVED_DELTA_EC", 1.0),
        verbose=verbose not in ("0", "false", "no", "off"),
    )


# ---------------------------------------------------------------- block runner contract
@dataclass(frozen=True)
class BlockBudget:
    """How much the inner minimizer is allowed to do in one block."""

    max_iterations: Optional[int] = None
    step_scale: float = 1.0

    def halved(self) -> "BlockBudget":
        """Halve both the iteration count and the maximum step."""
        n = None if self.max_iterations is None else max(1, self.max_iterations // 2)
        return BlockBudget(max_iterations=n, step_scale=self.step_scale * 0.5)


@runtime_checkable
class BlockRunner(Protocol):
    """One re-runnable inner block, whatever the minimizer underneath.

    ``run`` must leave the model in its proposed state; the controller then either
    keeps it or hands back a state from ``save_state``.
    """

    def full_budget(self) -> BlockBudget:
        """The block's natural (unhalved) budget."""

    def save_state(self) -> Any:
        """Snapshot everything ``run`` may change."""

    def restore_state(self, state: Any) -> None:
        """Put the model back to a snapshot."""

    def run(self, budget: BlockBudget) -> None:
        """Run the inner minimization under ``budget``."""


def interpolate_state(runner: Any, a: Any, b: Any, frac: float) -> Any:
    """Midpoint model between two snapshots, if the runner knows how.

    Runners that cannot interpolate fall back to ``a``, which makes the refresh happen
    at the checkpoint model instead of the midpoint -- correct, just a less useful
    linearization center for the shortened block.
    """
    fn = getattr(runner, "interpolate_state", None)
    if callable(fn):
        try:
            return fn(a, b, frac)
        except Exception:
            return a
    return a


# ---------------------------------------------------------------- host contract
@dataclass
class CheckpointResult:
    """What one exact checkpoint produced. ``f_p`` / ``beta_p`` are internal."""

    nll: float  # per-reflection-mean exact NLL on the work set
    nll_free: Optional[float]
    f_p: np.ndarray
    alpha_p: np.ndarray
    beta_p: np.ndarray
    mask: np.ndarray
    e_c: np.ndarray
    rho2: np.ndarray
    telemetry: dict[str, Any] = field(default_factory=dict)

    @property
    def n_exact_route(self) -> int:
        return int(self.telemetry.get("n_exact_route", 0))

    @property
    def n_fit_fallback(self) -> int:
        return int(self.telemetry.get("n_fallback_init", 0)) + self.n_exact_route


@runtime_checkable
class InterleavedHost(Protocol):
    """The model / target side of the loop.

    Implemented by :class:`~phridge.client.intensity.engine.IntensityFModel` in host
    mode, and by the test harness for the torch-only path.
    """

    def exact_checkpoint(self) -> CheckpointResult:
        """One exact evaluation at the current model; fit and install the surrogate."""

    def exact_nll(self) -> float:
        """One exact evaluation at the current model. No refit, no statistics."""

    def surrogate_nll(self) -> float:
        """The surrogate's own value at the current model (diagnostic only)."""

    def use_surrogate(self, enabled: bool) -> None:
        """Switch which target the inner machinery sees."""


# ---------------------------------------------------------------- telemetry
@dataclass(frozen=True)
class BlockSite:
    """Where in the refinement a block ran, and when.

    The controller cannot work this out for itself -- the caller that wrapped the
    minimizer is the only thing that knows which stage of which macro cycle it is
    sitting in -- so the site is passed to :meth:`InterleavedController.run_macro_cycle`
    and carried onto every log line. Without it the telemetry says "block 7" and the
    reader has no way to tell whether that was coordinate refinement in macro cycle 2
    or a B-factor weight trial in macro cycle 5.
    """

    stage: str = "unknown stage"
    macro_cycle: Optional[int] = None
    total_macro_cycles: Optional[int] = None
    detail: str = ""

    def when(self) -> str:
        if self.macro_cycle is None:
            return "macro cycle ?"
        if self.total_macro_cycles:
            return f"macro cycle {self.macro_cycle}/{self.total_macro_cycles}"
        return f"macro cycle {self.macro_cycle}"

    def where(self) -> str:
        return f"{self.stage} ({self.detail})" if self.detail else self.stage

    def label(self) -> str:
        return f"{self.when()} | {self.where()}"


@dataclass
class BlockRecord:
    """One block, as logged. Carries no surrogate parameter values."""

    index: int
    outcome: BlockOutcome
    nll_0: Optional[float]
    nll_1: Optional[float]
    exact_delta: Optional[float]
    predicted_delta: Optional[float]
    n_exact_evals: int
    n_inner_evals: int
    n_fit_fallback: int
    n_total: int
    budget: BlockBudget
    site: BlockSite = field(default_factory=BlockSite)

    def line(self) -> str:
        """One neutral log line. Never names or prints the surrogate parameters."""
        parts = [
            f"[interleaved] {self.site.label()} | block {self.index}: {self.outcome.value}"
        ]
        if self.nll_0 is not None and self.nll_1 is not None:
            parts.append(f"NLL {self.nll_0:.6f} -> {self.nll_1:.6f}")
        elif self.nll_1 is not None:
            parts.append(f"NLL {self.nll_1:.6f}")
        if self.exact_delta is not None:
            pred = (
                f", surrogate-predicted {self.predicted_delta:+.6f}"
                if self.predicted_delta is not None
                else ""
            )
            parts.append(f"delta exact {self.exact_delta:+.6f}{pred}")
        parts.append(f"exact evals {self.n_exact_evals}")
        parts.append(f"inner evals {self.n_inner_evals}")
        if self.n_total:
            parts.append(f"fit fallbacks {self.n_fit_fallback}/{self.n_total}")
        if self.budget.max_iterations is not None:
            parts.append(f"budget {self.budget.max_iterations} it x {self.budget.step_scale:.2f}")
        return " | ".join(parts)


@dataclass
class InterleavedTelemetry:
    blocks: list[BlockRecord] = field(default_factory=list)
    n_exact_evals: int = 0
    n_surrogate_evals: int = 0

    @property
    def n_accepted(self) -> int:
        return sum(1 for b in self.blocks if b.outcome is BlockOutcome.accepted)

    @property
    def n_rejected(self) -> int:
        return sum(1 for b in self.blocks if b.outcome is BlockOutcome.rejected)

    @property
    def n_exact_fallback(self) -> int:
        return sum(1 for b in self.blocks if b.outcome is BlockOutcome.exact_fallback)

    @property
    def rejection_rate(self) -> float:
        adjudicated = self.n_accepted + self.n_rejected
        return float(self.n_rejected) / adjudicated if adjudicated else 0.0

    def accepted_nll_trace(self) -> list[float]:
        """Exact NLL after each accepted block -- must be non-increasing."""
        out = []
        for b in self.blocks:
            if b.outcome is BlockOutcome.accepted and b.nll_1 is not None:
                out.append(float(b.nll_1))
        return out

    def stage_summary(self) -> list[str]:
        """One line per refinement stage that ran a block, in first-seen order.

        This is the "where did the interleaving actually happen" view: the per-block
        lines answer it one block at a time, but a run with dozens of blocks needs the
        roll-up to show, for instance, that every rejection came from the ADP stage.
        """
        order: list[str] = []
        # Outcome names are dict keys here, and one of them is "exact", so the eval
        # counters get their own namespace rather than colliding with it.
        counts: dict[str, dict[BlockOutcome, int]] = {}
        evals: dict[str, tuple[int, int]] = {}
        for b in self.blocks:
            key = b.site.where()
            if key not in counts:
                order.append(key)
                counts[key] = {o: 0 for o in BlockOutcome}
                evals[key] = (0, 0)
            counts[key][b.outcome] += 1
            n_exact, n_inner = evals[key]
            evals[key] = (n_exact + b.n_exact_evals, n_inner + b.n_inner_evals)
        lines = []
        for key in order:
            n_blocks = sum(counts[key].values())
            outcomes = ", ".join(
                f"{counts[key][o]} {o.value}" for o in BlockOutcome if counts[key][o]
            )
            n_exact, n_inner = evals[key]
            lines.append(
                f"[interleaved] where: {key} -- {n_blocks} block(s) "
                f"[{outcomes}] | exact evals {n_exact}, inner evals {n_inner}"
            )
        return lines

    def report(self) -> str:
        lines = [b.line() for b in self.blocks]
        lines.extend(self.stage_summary())
        lines.append(
            f"[interleaved] {self.n_accepted} accepted, {self.n_rejected} rejected, "
            f"{self.n_exact_fallback} exact fallback | exact evals {self.n_exact_evals}, "
            f"inner (surrogate) evals {self.n_surrogate_evals}"
        )
        if self.rejection_rate > 0.5:
            lines.append(
                "[interleaved] rejection rate above 50%: shorten the blocks "
                "(refinement.interleaved.max_halvings, or fewer inner iterations)."
            )
        return "\n".join(lines)


# ---------------------------------------------------------------- the controller
class InterleavedController:
    """Drives the checkpoint / block / adjudicate ladder for one refinement run."""

    def __init__(
        self,
        host: InterleavedHost,
        options: Optional[InterleavedOptions] = None,
        *,
        log: Any = None,
        mode: TargetMode = TargetMode.interleaved,
        journal: Any = None,
    ) -> None:
        self.host = host
        self.options = options or InterleavedOptions()
        self.log = log
        self.mode = mode
        # The run's NLL journal, if one is active. The ladder measures NLL_0 and NLL_1
        # exactly in order to adjudicate each block, so the journal is handed those
        # values rather than being left to re-measure the same two points.
        self.journal = journal
        self.telemetry = InterleavedTelemetry()
        self._last: Optional[CheckpointResult] = None
        self._block_index = 0

    # -- plumbing ---------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self.mode is TargetMode.interleaved

    def _emit(self, text: str) -> None:
        if not self.options.verbose:
            return
        from phridge.client.intensity.heartbeat import output_streams

        log = self.log if self.log is not None and hasattr(self.log, "write") else None
        for out in output_streams(log):
            try:
                print(text, file=out, flush=True)
            except Exception:
                pass

    def _checkpoint(self) -> CheckpointResult:
        cp = self.host.exact_checkpoint()
        self.telemetry.n_exact_evals += 1
        self._last = cp
        return cp

    def _exact_nll(self) -> float:
        val = float(self.host.exact_nll())
        self.telemetry.n_exact_evals += 1
        return val

    def _try_surrogate_nll(self) -> Optional[float]:
        """Surrogate value at the current model, for the trust diagnostic only."""
        try:
            val = float(self.host.surrogate_nll())
        except Exception:
            return None
        return val if val == val else None  # reject NaN

    def _record(self, record: BlockRecord) -> None:
        self.telemetry.blocks.append(record)
        self._emit(record.line())
        self._journal_block(record)

    def _journal_block(self, record: BlockRecord) -> None:
        """Mirror the block into the run-wide NLL journal, free of charge."""
        if self.journal is None:
            return
        from phridge.client.intensity.nll_log import NllPoint, StageKind

        try:
            # The ladder's NLL is -log p(Z). Σ_W does not move inside a block, so adding
            # the same log(εΣ) to both ends puts the line in -log p(I) units — the units
            # every other stage uses — without changing the delta the block is judged on.
            off_w = None
            getter = getattr(self.host, "_normalization_offset", None)
            if callable(getter):
                off_w = getter()[0]

            def _shift(value: Optional[float]) -> Optional[float]:
                if value is None or off_w is None:
                    return value
                return float(value) + float(off_w)

            self.journal.set_prefix(record.site.when())
            self.journal.record_stage(
                f"{record.site.where()} | block {record.index}",
                StageKind.model,
                NllPoint(work=_shift(record.nll_0)),
                NllPoint(work=_shift(record.nll_1)),
                detail=record.outcome.value,
            )
        except Exception:
            pass  # telemetry must never cost a block

    def _journal_entry_nll(self) -> Optional[float]:
        """Entry NLL for the fully-exact branch, which the ladder does not otherwise need.

        Only spent when a journal is active: without it the branch has no use for a
        before-value, and an exact evaluation is the single most expensive thing here.
        """
        if self.journal is None or not getattr(self.journal, "enabled", False):
            return None
        try:
            return self._exact_nll()
        except Exception:
            return None

    # -- the ladder -------------------------------------------------------------
    def run_macro_cycle(
        self,
        runner: BlockRunner,
        *,
        final: bool = False,
        site: Optional[BlockSite] = None,
    ) -> BlockOutcome:
        """Run one macro cycle's inner block under the controller.

        ``final=True`` forces a fully exact block: the last macro cycle never touches
        the surrogate, so deposited statistics are always exact. ``site`` says when and
        where this block sits in the refinement and appears on every line it logs.
        """
        self._block_index += 1
        site = site or BlockSite()
        full = runner.full_budget()

        if final or not self.enabled:
            reason = (
                "final macro cycle" if final and self.enabled else f"target_mode={self.mode.value}"
            )
            self._emit(
                f"[interleaved] {site.label()} | block {self._block_index}: "
                f"running fully exact ({reason}) -- no surrogate is used here"
            )
            self.host.use_surrogate(False)
            before = self.telemetry.n_surrogate_evals
            nll_before = self._journal_entry_nll()
            runner.run(full)
            nll = self._exact_nll()
            self._record(
                BlockRecord(
                    index=self._block_index,
                    outcome=BlockOutcome.exact,
                    nll_0=nll_before,
                    nll_1=nll,
                    exact_delta=(nll - nll_before) if nll_before is not None else None,
                    predicted_delta=None,
                    n_exact_evals=1,
                    n_inner_evals=self.telemetry.n_surrogate_evals - before,
                    n_fit_fallback=0,
                    n_total=0,
                    budget=full,
                    site=site,
                )
            )
            return BlockOutcome.exact

        budget_txt = (
            f"{full.max_iterations} it" if full.max_iterations is not None else "unbounded"
        )
        self._emit(
            f"[interleaved] {site.label()} | block {self._block_index}: "
            f"INTERLEAVING HERE -- exact checkpoint, then a {budget_txt} inner block on "
            f"the surrogate, then an exact adjudication"
        )

        # 1. checkpoint (exact): NLL_0, scores, curvatures, surrogate refresh
        cp = self._checkpoint()
        nll_0 = cp.nll
        self._emit(
            f"[interleaved] {site.label()} | block {self._block_index}: "
            f"checkpoint (exact) NLL_0 = {nll_0:.6f} over {int(cp.mask.size)} reflections "
            f"({cp.n_fit_fallback} fit fallback, {cp.n_exact_route} routed exact)"
        )
        checkpoint_state = runner.save_state()
        budget = full

        for attempt in range(self.options.max_halvings + 1):
            exact_evals = 1 if attempt == 0 else 2  # checkpoint (+ refresh on a retry)
            inner_before = self.telemetry.n_surrogate_evals

            # 2. inner block (surrogate): zero quadrature evaluations
            self.host.use_surrogate(True)
            try:
                surrogate_0 = self._try_surrogate_nll()
                runner.run(budget)
                surrogate_1 = self._try_surrogate_nll()
            finally:
                self.host.use_surrogate(False)
            proposal_state = runner.save_state()

            # The surrogate's own opinion of the block. The gap between this and the
            # exact delta below is the trust diagnostic; it never decides anything.
            predicted_delta: Optional[float] = (
                surrogate_1 - surrogate_0
                if surrogate_0 is not None and surrogate_1 is not None
                else None
            )

            # 3. adjudication (exact)
            nll_1 = self._exact_nll()
            exact_evals += 1
            n_inner = self.telemetry.n_surrogate_evals - inner_before

            if nll_1 <= nll_0 + self.options.tol:
                self._record(
                    BlockRecord(
                        index=self._block_index,
                        outcome=BlockOutcome.accepted,
                        nll_0=nll_0,
                        nll_1=nll_1,
                        exact_delta=nll_1 - nll_0,
                        predicted_delta=predicted_delta,
                        n_exact_evals=exact_evals,
                        n_inner_evals=n_inner,
                        n_fit_fallback=cp.n_fit_fallback,
                        n_total=int(cp.mask.size),
                        budget=budget,
                        site=site,
                    )
                )
                return BlockOutcome.accepted

            self._record(
                BlockRecord(
                    index=self._block_index,
                    outcome=BlockOutcome.rejected,
                    nll_0=nll_0,
                    nll_1=nll_1,
                    exact_delta=nll_1 - nll_0,
                    predicted_delta=predicted_delta,
                    n_exact_evals=exact_evals,
                    n_inner_evals=n_inner,
                    n_fit_fallback=cp.n_fit_fallback,
                    n_total=int(cp.mask.size),
                    budget=budget,
                    site=site,
                )
            )

            if attempt == self.options.max_halvings:
                break

            # halve the block after refreshing the surrogate at the midpoint model
            halved = budget.halved()
            self._emit(
                f"[interleaved] {site.label()} | block {self._block_index}: "
                f"refreshing at the midpoint model and retrying with the block halved "
                f"({halved.max_iterations} it x {halved.step_scale:.2f})"
            )
            midpoint = interpolate_state(runner, checkpoint_state, proposal_state, 0.5)
            runner.restore_state(midpoint)
            self._checkpoint()  # refresh only: NLL_0 stays the checkpoint value
            runner.restore_state(checkpoint_state)
            budget = halved

        # 4. never accept on the surrogate's word: run this macro cycle fully exact
        self._emit(
            f"[interleaved] {site.label()} | block {self._block_index}: "
            f"surrogate rejected twice -- rerunning this block fully exact"
        )
        runner.restore_state(checkpoint_state)
        self.host.use_surrogate(False)
        inner_before = self.telemetry.n_surrogate_evals
        runner.run(full)
        nll_exact = self._exact_nll()
        self._record(
            BlockRecord(
                index=self._block_index,
                outcome=BlockOutcome.exact_fallback,
                nll_0=nll_0,
                nll_1=nll_exact,
                exact_delta=nll_exact - nll_0,
                predicted_delta=None,
                n_exact_evals=1,
                n_inner_evals=self.telemetry.n_surrogate_evals - inner_before,
                n_fit_fallback=cp.n_fit_fallback,
                n_total=int(cp.mask.size),
                budget=full,
                site=site,
            )
        )
        return BlockOutcome.exact_fallback

    # -- v2 asymmetric refresh --------------------------------------------------
    def refresh_selection(self, cp: CheckpointResult, e_c_since: Optional[np.ndarray]) -> np.ndarray:
        """Which reflections need their surrogate refitted, under ``refresh``.

        v1 (``refresh='full'``) refreshes everything: it costs one exact evaluation
        that the checkpoint is already paying for. v2 refreshes only reflections whose
        visible fraction ``rho2`` is low or which have moved far since the last
        refresh. The theory note is that the EM contraction rate per reflection is the
        missing-information fraction ``1 - rho2``, so low-``rho2`` reflections are
        exactly the ones whose surrogates go stale fastest.
        """
        if self.options.refresh == "full":
            return np.ones(cp.rho2.shape, dtype=bool)
        stale = np.asarray(cp.rho2, dtype=np.float64) < self.options.rho2_threshold
        if e_c_since is not None and np.shape(e_c_since) == np.shape(cp.e_c):
            moved = np.abs(np.asarray(cp.e_c) - np.asarray(e_c_since)) > self.options.delta_ec_threshold
            stale = stale | moved
        return stale


# ---------------------------------------------------------------- torch adapter
class TorchOptimBlockRunner:
    """:class:`BlockRunner` over any ``torch.optim`` optimizer.

    Covers LBFGS, Adam, AdamW and SGD with one code path: LBFGS consumes the closure
    directly, the first-order optimizers get ``zero_grad`` / closure / ``step``. The
    budget's ``step_scale`` scales the learning rate (and the per-step parameter cap,
    when one is set), so a halved block really does take shorter steps as well as
    fewer of them.
    """

    def __init__(
        self,
        params: Sequence[Any],
        closure: Callable[[], Any],
        *,
        optimizer: str = "lbfgs",
        max_iterations: int = 20,
        lr: float = 1.0,
        step_cap: Optional[float] = None,
        optimizer_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.params = list(params)
        self.closure = closure
        self.optimizer = str(optimizer).lower()
        self.max_iterations = int(max_iterations)
        self.lr = float(lr)
        self.step_cap = step_cap
        self.optimizer_factory = optimizer_factory
        if self.optimizer not in ("lbfgs", "adam", "adamw", "sgd") and optimizer_factory is None:
            raise ValueError(
                "optimizer must be one of 'lbfgs', 'adam', 'adamw', 'sgd', "
                "or pass optimizer_factory=..."
            )

    # -- BlockRunner ------------------------------------------------------------
    def full_budget(self) -> BlockBudget:
        return BlockBudget(max_iterations=self.max_iterations, step_scale=1.0)

    def save_state(self) -> Any:
        return [p.detach().clone() for p in self.params]

    def restore_state(self, state: Any) -> None:
        import torch

        with torch.no_grad():
            for p, s in zip(self.params, state):
                p.copy_(s)

    def interpolate_state(self, a: Any, b: Any, frac: float) -> Any:
        return [(1.0 - frac) * x + frac * y for x, y in zip(a, b)]

    def _make_optimizer(self, budget: BlockBudget) -> Any:
        import torch

        lr = self.lr * budget.step_scale
        if self.optimizer_factory is not None:
            return self.optimizer_factory(self.params, lr=lr)
        if self.optimizer == "lbfgs":
            return torch.optim.LBFGS(
                self.params,
                lr=lr,
                max_iter=max(1, budget.max_iterations or self.max_iterations),
                line_search_fn="strong_wolfe",
            )
        if self.optimizer == "adam":
            return torch.optim.Adam(self.params, lr=lr)
        if self.optimizer == "adamw":
            return torch.optim.AdamW(self.params, lr=lr)
        return torch.optim.SGD(self.params, lr=lr)

    def run(self, budget: BlockBudget) -> None:
        import torch

        opt = self._make_optimizer(budget)
        n_iter = max(1, budget.max_iterations or self.max_iterations)
        if isinstance(opt, torch.optim.LBFGS):
            opt.step(self.closure)  # max_iter already carries the budget
            return
        cap = None if self.step_cap is None else self.step_cap * budget.step_scale
        for _ in range(n_iter):
            before = [p.detach().clone() for p in self.params] if cap is not None else None
            opt.zero_grad(set_to_none=False)
            self.closure()
            opt.step()
            if cap is not None and before is not None:
                with torch.no_grad():
                    for p, b in zip(self.params, before):
                        delta = p - b
                        norm = float(delta.abs().max())
                        if norm > cap:
                            p.copy_(b + delta * (cap / norm))
