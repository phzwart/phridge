"""Start/end NLL journal for one intensity refinement run.

Refinement alternates between moving atoms and refitting nuisance parameters, and the
only quantity that says whether either helped is the exact NLL. Phenix's log reports
R-factors per stage, but under an intensity target those are a derived statistic on
French-Wilson amplitudes rather than the thing being minimized: a stage can lower R
and raise the NLL, and nothing in the stock log would say so. This module records the
NLL at each stage boundary, so the log answers "what did this stage buy" in the units
the refinement is actually optimizing.

Model stages and target stages are not the same measurement
-----------------------------------------------------------
A **model** stage moves atoms with the target held fixed, so its NLL difference is the
minimizer's achievement and cannot be positive beyond the accept tolerance. A
**target** stage holds the atoms fixed and refits nuisances -- ``sigma_A(s)``,
``Sigma_W(s)``, ``beta``, the scales -- so its difference is the nuisance fit's
achievement. Both lower the same number and it is tempting to add them up
indiscriminately, but a run whose progress came entirely from target stages has not
improved the model at all, which is worth being able to see. They are labelled and
totalled separately for that reason.

Comparability
-------------
An NLL is comparable to another NLL only over the same reflections. Outlier rejection
or a change of the working set breaks that, so every measurement carries its
reflection counts and a difference taken across a count change is reported as
incomparable rather than as an improvement.

Cost
----
Each boundary measurement is one exact evaluation with no gradients, and a stage
brackets two of them. Against the hundreds of gradient evaluations a minimization
stage already spends this is noise, but it is not free, so ``PHRIDGE_NLL_LOG=0`` turns
the whole thing off and no probe is called at all.
"""

from __future__ import annotations

import atexit
import math
import os
import sys
import time

import numpy as np
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterator, Optional, Sequence, TextIO

__all__ = [
    "NllJournal",
    "NllPoint",
    "StageKind",
    "StageRecord",
    "TrialGroup",
    "TrialRecord",
    "fmt_value",
    "nll_logging_enabled",
    "normalization_offset",
]

_MISSING = "--"


def nll_logging_enabled(default: bool = True) -> bool:
    """``PHRIDGE_NLL_LOG``: journal the NLL at stage boundaries (default on)."""
    raw = os.environ.get("PHRIDGE_NLL_LOG")
    if raw is None or not str(raw).strip():
        return bool(default)
    return str(raw).strip().lower() not in ("0", "false", "no", "off")


def nll_trials_enabled(default: bool = True) -> bool:
    """``PHRIDGE_NLL_LOG_TRIALS``: also journal every weight trial (default on).

    Separate from :func:`nll_logging_enabled` because the trial tables are the verbose
    part: a weight scan is a dozen lines per stage per macro cycle.
    """
    raw = os.environ.get("PHRIDGE_NLL_LOG_TRIALS")
    if raw is None or not str(raw).strip():
        return bool(default)
    return str(raw).strip().lower() not in ("0", "false", "no", "off")


def normalization_offset(
    epsilon: Any,
    sigma_wilson: Any,
    free: Any = None,
) -> tuple[Optional[float], Optional[float]]:
    """Mean ``log(ε Σ_W)`` on the work set and, separately, the free set.

    The quadrature returns the log-density of the *normalized* intensity
    ``Z = I / (ε Σ_W)``. The density of the intensity that was actually measured is

        log p(I) = log p(Z) - log(ε Σ_W)

    so the per-reflection mean NLL of the data is the mean of ``-log p(Z)`` plus this
    offset. The two agree up to a constant whenever ``Σ_W`` is held fixed, which is
    every stage except the nuisance fit -- and that is the stage that replaces ``Σ_W``,
    swapping the per-bin moment estimate for a Wilson curve. Without the offset that
    swap moves the reported NLL by the change of units, and a fit that improved the
    likelihood of the data can print as one that nearly doubled it.
    """
    eps = np.asarray(epsilon, dtype=np.float64).reshape(-1)
    sw = np.asarray(sigma_wilson, dtype=np.float64).reshape(-1)
    if eps.shape != sw.shape or eps.size == 0:
        return None, None
    with np.errstate(all="ignore"):
        term = np.log(np.maximum(eps * sw, 1e-300))
    term = np.where(np.isfinite(term), term, np.nan)
    if free is None:
        free_mask = np.zeros(term.shape, dtype=bool)
    else:
        free_mask = np.asarray(free, dtype=bool).reshape(-1)
        if free_mask.shape != term.shape:
            return None, None
    return _masked_mean(term, ~free_mask), _masked_mean(term, free_mask)


def _masked_mean(values: Any, mask: Any) -> Optional[float]:
    sel = np.asarray(values, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    sel = sel[np.isfinite(sel)]
    if sel.size == 0:
        return None
    return float(sel.mean())


def fmt_value(value: Optional[float], spec: str = ".6f") -> str:
    """Format a float that may be absent or non-finite, right-aligned to ``spec``."""
    width = ""
    for ch in spec:
        if ch.isdigit():
            width += ch
        else:
            break
    pad = int(width) if width else 0
    if value is None or not math.isfinite(float(value)):
        return _MISSING.rjust(pad)
    return format(float(value), spec)


class StageKind(str, Enum):
    """What a stage was allowed to change, which is what its NLL delta measures."""

    model = "model"
    target = "target"
    mixed = "mixed"


@dataclass(frozen=True)
class NllPoint:
    """One exact NLL measurement: work and free means, with the counts behind them."""

    work: Optional[float] = None
    free: Optional[float] = None
    n_work: Optional[int] = None
    n_free: Optional[int] = None

    @property
    def ok(self) -> bool:
        return self.work is not None and math.isfinite(float(self.work))

    def counts_match(self, other: "NllPoint") -> bool:
        """True when a difference between the two is a like-for-like comparison.

        Unknown counts are treated as matching: the probe not reporting them is not
        evidence that the reflection set moved, and refusing to show a delta in that
        case would hide the common path behind a missing diagnostic.
        """
        for a, b in ((self.n_work, other.n_work), (self.n_free, other.n_free)):
            if a is not None and b is not None and int(a) != int(b):
                return False
        return True

    def text(self) -> str:
        parts = [f"work {fmt_value(self.work)}"]
        if self.free is not None:
            parts.append(f"free {fmt_value(self.free)}")
        return " | ".join(parts)


@dataclass
class StageRecord:
    """One journalled stage, from its entry NLL to its exit NLL."""

    label: str
    kind: StageKind
    start: NllPoint
    end: NllPoint
    seconds: float = 0.0
    detail: str = ""
    note: str = ""

    @property
    def comparable(self) -> bool:
        return self.start.counts_match(self.end)

    def _delta(self, a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is None or b is None or not self.comparable:
            return None
        if not (math.isfinite(float(a)) and math.isfinite(float(b))):
            return None
        return float(b) - float(a)

    @property
    def delta_work(self) -> Optional[float]:
        return self._delta(self.start.work, self.end.work)

    @property
    def delta_free(self) -> Optional[float]:
        return self._delta(self.start.free, self.end.free)

    def line(self) -> str:
        """One log line: where, what it was allowed to change, and what it bought."""
        head = f"[nll] {self.label} [{self.kind.value}]"
        body = []
        for name, start, end, delta in (
            ("work", self.start.work, self.end.work, self.delta_work),
            ("free", self.start.free, self.end.free, self.delta_free),
        ):
            if start is None and end is None:
                continue
            shown = f" ({delta:+.6f})" if delta is not None else ""
            body.append(f"{name} {fmt_value(start)} -> {fmt_value(end)}{shown}")
        if not body:
            # Neither end could be measured. Saying so beats a line of dashes that reads
            # like a stage which achieved nothing.
            body.append("NLL unavailable")
        if not self.comparable:
            body.append(
                f"reflections changed {self.start.n_work} -> {self.end.n_work}, "
                "NLL means are not comparable"
            )
        if self.seconds:
            body.append(f"{self.seconds:.1f}s")
        if self.detail:
            body.append(self.detail)
        if self.note:
            body.append(self.note)
        return " | ".join([head] + body)


@dataclass(frozen=True)
class TrialRecord:
    """One trial of a parameter scan: the parameter value and the NLL it reached."""

    value: Optional[float]
    point: NllPoint
    extra: dict = field(default_factory=dict)

    def sort_key(self) -> float:
        """Rank by free NLL, falling back to work when there is no free set."""
        for candidate in (self.point.free, self.point.work):
            if candidate is not None and math.isfinite(float(candidate)):
                return float(candidate)
        return math.inf


@dataclass
class TrialGroup:
    """A scan over one parameter, e.g. the XYZ or ADP target weight.

    The point of recording these is that the selection is made on free NLL but the
    stock Phenix trial table shows R-factors, so the log gives no way to check the
    decision or to see how close the runner-up was. ``margin`` is the part that
    matters in practice: a scan whose best two trials differ in the sixth decimal has
    not really chosen a weight, and that is worth knowing before trusting it.
    """

    name: str
    param_name: str
    extra_columns: Sequence[str] = ()
    records: list[TrialRecord] = field(default_factory=list)
    selected_value: Optional[float] = None

    def add(
        self,
        value: Optional[float],
        point: NllPoint,
        extra: Optional[dict] = None,
    ) -> None:
        self.records.append(TrialRecord(value=value, point=point, extra=dict(extra or {})))

    def ranked(self) -> list[TrialRecord]:
        return sorted(self.records, key=lambda r: r.sort_key())

    def best(self) -> Optional[TrialRecord]:
        ranked = self.ranked()
        return ranked[0] if ranked else None

    def margin(self) -> Optional[float]:
        """Free-NLL gap between the best trial and the runner-up."""
        ranked = self.ranked()
        if len(ranked) < 2:
            return None
        lo, next_ = ranked[0].sort_key(), ranked[1].sort_key()
        if not (math.isfinite(lo) and math.isfinite(next_)):
            return None
        return next_ - lo

    def _is_selected(self, record: TrialRecord) -> bool:
        if self.selected_value is None or record.value is None:
            return False
        return abs(float(record.value) - float(self.selected_value)) <= 1.0e-8

    def table(self) -> list[str]:
        """The scan as a table in trial order, marking the selected row with ``*``."""
        if not self.records:
            return []
        head = (
            f"[nll] {self.name}: {len(self.records)} trial(s) over {self.param_name}, "
            "ranked by free NLL"
        )
        cols = f"      {'':1} {self.param_name:>12} {'NLL work':>13} {'NLL free':>13}"
        for name in self.extra_columns:
            cols += f" {name:>9}"
        lines = [head, cols]
        for record in self.records:
            mark = "*" if self._is_selected(record) else " "
            row = (
                f"      {mark} {fmt_value(record.value, '12.4f')}"
                f" {fmt_value(record.point.work, '13.6f')} {fmt_value(record.point.free, '13.6f')}"
            )
            for name in self.extra_columns:
                row += f" {fmt_value(record.extra.get(name), '9.4f')}"
            lines.append(row)
        best = self.best()
        if best is not None:
            margin = self.margin()
            chosen = self.selected_value if self.selected_value is not None else best.value
            tail = (
                f"[nll] {self.name}: selected {self.param_name}={fmt_value(chosen, '.4f')} "
                f"at free NLL {fmt_value(best.point.free or best.point.work, '.6f')}"
            )
            if margin is not None:
                tail += f", {margin:+.6f} ahead of the runner-up"
                if margin < 1.0e-4:
                    tail += " -- the scan did not really separate them"
            lines.append(tail)
        return lines


class NllJournal:
    """Records the NLL at stage boundaries for one run and reports the trace.

    ``probe`` is the only way the journal reaches the target: a callable returning an
    :class:`NllPoint`. Injecting it keeps this module free of cctbx and the worker, and
    lets the tests drive the whole thing with a counter.
    """

    def __init__(
        self,
        probe: Callable[[], NllPoint],
        *,
        log: Any = None,
        enabled: Optional[bool] = None,
        trials: Optional[bool] = None,
        title: str = "mli_quad",
        register_atexit: bool = True,
    ) -> None:
        self._probe = probe
        self.log = log
        self.enabled = nll_logging_enabled() if enabled is None else bool(enabled)
        self.trials_enabled = nll_trials_enabled() if trials is None else bool(trials)
        self.title = title
        self.stages: list[StageRecord] = []
        self.groups: list[TrialGroup] = []
        self.run_start: Optional[NllPoint] = None
        self.run_end: Optional[NllPoint] = None
        self._reported = False
        self._prefix = ""
        if self.enabled and register_atexit:
            # A refinement that dies mid-run is exactly when the trace is most wanted,
            # and there is no end-of-refinement hook in Phenix to hang it on.
            atexit.register(self._atexit_report)

    # ------------------------------------------------------------------ plumbing
    def set_log(self, log: Any) -> None:
        """Adopt the current Phenix log; the journal outlives any single call's log."""
        if log is not None and hasattr(log, "write"):
            self.log = log

    def set_prefix(self, prefix: str) -> None:
        """Label every subsequent line with the macro cycle it belongs to."""
        self._prefix = str(prefix or "")

    def _label(self, label: str) -> str:
        return f"{self._prefix} | {label}" if self._prefix else label

    def _streams(self) -> list[TextIO]:
        from phridge.client.intensity.heartbeat import output_streams

        log = self.log if self.log is not None and hasattr(self.log, "write") else None
        return output_streams(log)

    def emit(self, lines: Sequence[str]) -> None:
        for out in self._streams():
            for line in lines:
                try:
                    print(line, file=out)
                except Exception:
                    pass
            try:
                if hasattr(out, "flush"):
                    out.flush()
            except Exception:
                pass

    def measure(self) -> NllPoint:
        """One probe call. Never raises: a failed measurement is an absent number."""
        if not self.enabled:
            return NllPoint()
        try:
            point = self._probe()
        except Exception:
            return NllPoint()
        return point if isinstance(point, NllPoint) else NllPoint()

    # ------------------------------------------------------------------ recording
    def mark_start(self, point: Optional[NllPoint] = None) -> Optional[NllPoint]:
        """The run's first NLL. Idempotent, so every entry point can call it.

        Takes an already-measured point when the caller has one, because the run's first
        NLL and the first stage's entry NLL are the same measurement and there is no
        reason to pay for it twice.
        """
        if not self.enabled or self.run_start is not None:
            return self.run_start
        if point is None:
            point = self.measure()
        if point.ok:
            self.run_start = point
            self.emit([f"[nll] run start: {point.text()}"])
        return self.run_start

    def mark_end(self, label: str = "run end") -> Optional[NllPoint]:
        """The run's latest NLL. Overwrites, so the last call wins."""
        if not self.enabled:
            return None
        point = self.measure()
        if point.ok:
            self.run_end = point
            self.emit([f"[nll] {label}: {point.text()}"])
        return self.run_end

    @contextmanager
    def stage(
        self,
        label: str,
        kind: StageKind = StageKind.model,
        *,
        detail: str = "",
    ) -> Iterator[StageRecord]:
        """Bracket a stage with an NLL measurement at entry and at exit.

        The record is appended and logged even when the body raises, because a stage
        that failed halfway is the one whose NLL you most want to see. The exception
        propagates untouched.
        """
        if not self.enabled:
            yield StageRecord(label=label, kind=kind, start=NllPoint(), end=NllPoint())
            return
        start = self.measure()
        if self.run_start is None and start.ok:
            self.run_start = start
        record = StageRecord(
            label=self._label(label), kind=kind, start=start, end=NllPoint(), detail=detail
        )
        t0 = time.monotonic()
        try:
            yield record
        finally:
            record.end = self.measure()
            record.seconds = time.monotonic() - t0
            if record.end.ok:
                self.run_end = record.end
            self.stages.append(record)
            self.emit([record.line()])

    def record_stage(
        self,
        label: str,
        kind: StageKind,
        start: NllPoint,
        end: NllPoint,
        *,
        seconds: float = 0.0,
        detail: str = "",
        note: str = "",
    ) -> StageRecord:
        """Journal a stage whose NLLs were already measured by someone else.

        The interleaved controller adjudicates every block with exact evaluations it
        has already paid for, so it feeds them here rather than making the journal
        measure the same two points again.
        """
        record = StageRecord(
            label=self._label(label),
            kind=kind,
            start=start,
            end=end,
            seconds=seconds,
            detail=detail,
            note=note,
        )
        if self.enabled:
            if self.run_start is None and start.ok:
                self.run_start = start
            if end.ok:
                self.run_end = end
            self.stages.append(record)
            self.emit([record.line()])
        return record

    def new_group(
        self,
        name: str,
        param_name: str,
        *,
        extra_columns: Sequence[str] = (),
    ) -> TrialGroup:
        """An empty scan, labelled with the current macro cycle. Not yet registered."""
        return TrialGroup(
            name=self._label(name), param_name=param_name, extra_columns=tuple(extra_columns)
        )

    def add_group(self, group: Optional[TrialGroup], *, min_trials: int = 2) -> None:
        """Register a finished scan and print its table.

        A scan with a single trial is not a scan -- Phenix runs one "trial" whenever
        weight optimization is off -- so by default it is dropped rather than printed as
        a one-row table.
        """
        if group is None or not self.enabled or not self.trials_enabled:
            return
        if len(group.records) < int(min_trials):
            return
        self.groups.append(group)
        self.emit(group.table())

    @contextmanager
    def trial_group(
        self,
        name: str,
        param_name: str,
        *,
        extra_columns: Sequence[str] = (),
        min_trials: int = 2,
    ) -> Iterator[TrialGroup]:
        """Collect a parameter scan and print its table when the scan closes."""
        group = self.new_group(name, param_name, extra_columns=extra_columns)
        try:
            yield group
        finally:
            self.add_group(group, min_trials=min_trials)

    # ------------------------------------------------------------------ reporting
    def totals(self) -> dict[str, Any]:
        """Per-kind sums of the comparable stage deltas, plus the overall change."""
        out: dict[str, Any] = {}
        for kind in StageKind:
            deltas = [
                s.delta_work
                for s in self.stages
                if s.kind is kind and s.delta_work is not None
            ]
            out[kind.value] = (sum(deltas) if deltas else None, len(deltas))
        overall_work = overall_free = None
        if self.run_start is not None and self.run_end is not None:
            if self.run_start.counts_match(self.run_end):
                if self.run_start.work is not None and self.run_end.work is not None:
                    overall_work = float(self.run_end.work) - float(self.run_start.work)
                if self.run_start.free is not None and self.run_end.free is not None:
                    overall_free = float(self.run_end.free) - float(self.run_start.free)
        out["overall_work"] = overall_work
        out["overall_free"] = overall_free
        return out

    def report(self) -> list[str]:
        """The whole run: start, every stage, end, and where the change came from."""
        if not self.stages and self.run_start is None:
            return []
        bar = "=" * 96
        rule = "-" * 96
        width = 54
        lines = [
            bar,
            f" {self.title} NLL journal -- exact intensity likelihood, "
            "per-reflection mean, lower is better",
            " [model] moved atoms at a fixed target; [target] refit nuisances at a fixed model",
            rule,
            f" {'stage':<{width}} {'kind':<7} {'NLL work':>12} {'change':>12} {'NLL free':>12}",
            rule,
        ]
        if self.run_start is not None:
            lines.append(
                f" {'run start':<{width}} {'':<7} {fmt_value(self.run_start.work, '12.6f')}"
                f" {_MISSING:>12} {fmt_value(self.run_start.free, '12.6f')}"
            )
        for record in self.stages:
            d = record.delta_work
            change = f"{d:+12.6f}" if d is not None else _MISSING.rjust(12)
            label = record.label if len(record.label) <= width else record.label[: width - 1] + "~"
            lines.append(
                f" {label:<{width}} {record.kind.value:<7}"
                f" {fmt_value(record.end.work, '12.6f')} {change}"
                f" {fmt_value(record.end.free, '12.6f')}"
            )
        if self.run_end is not None:
            lines.append(
                f" {'run end':<{width}} {'':<7} {fmt_value(self.run_end.work, '12.6f')}"
                f" {_MISSING:>12} {fmt_value(self.run_end.free, '12.6f')}"
            )
        lines.append(rule)
        totals = self.totals()
        if totals["overall_work"] is not None:
            free = (
                f", free {totals['overall_free']:+.6f}"
                if totals["overall_free"] is not None
                else ""
            )
            lines.append(f" overall change: work {totals['overall_work']:+.6f}{free}")
        for kind in StageKind:
            total, count = totals[kind.value]
            if total is not None:
                lines.append(
                    f"   from {kind.value} stages: work {total:+.6f} over {count} stage(s)"
                )
        unexplained = self._unexplained(totals)
        if unexplained is not None:
            lines.append(
                f"   unattributed: {unexplained:+.6f} -- stages the journal did not bracket"
            )
        for group in self.groups:
            margin = group.margin()
            if margin is not None:
                lines.append(
                    f"   {group.name}: selected {group.param_name}="
                    f"{fmt_value(group.selected_value or (group.best().value if group.best() else None), '.4f')}"
                    f", margin {margin:+.6f} over {len(group.records)} trial(s)"
                )
        lines.append(bar)
        return lines

    def _unexplained(self, totals: dict[str, Any]) -> Optional[float]:
        """Overall change minus the bracketed stages.

        A large residue means the NLL moved somewhere the journal is not watching, which
        is a gap in the instrumentation rather than in the refinement -- worth printing
        so the table is never mistaken for a closed account.
        """
        if totals["overall_work"] is None:
            return None
        attributed = 0.0
        for kind in StageKind:
            total, _ = totals[kind.value]
            if total is not None:
                attributed += float(total)
        residue = float(totals["overall_work"]) - attributed
        return residue if abs(residue) > 1.0e-6 else None

    def print_report(self) -> None:
        lines = self.report()
        if lines:
            self._reported = True
            self.emit(lines)

    def _atexit_report(self) -> None:
        """Backstop report at interpreter exit, only if nothing printed one already."""
        if self._reported:
            return
        try:
            lines = self.report()
            if not lines:
                return
            self._reported = True
            # The Phenix log may already be closed; stdout is the only safe stream here.
            for line in lines:
                print(line, file=sys.stdout)
        except Exception:
            pass
