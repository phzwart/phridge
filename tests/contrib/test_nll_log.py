"""The NLL journal: what each stage bought, and what each weight trial cost.

Two claims are under test. The first is that the journal reports the exact NLL at every
stage boundary and attributes the change to the right cause -- a stage that only refits
nuisances must not be credited with improving the model. The second is that it is honest
about what it does not know: a delta across a changed reflection set is not a delta, a
failed probe is not a zero, and change the journal did not bracket is called out rather
than silently folded into the totals.

Every measurement is one exact quadrature, so the cost tests are not incidental: they
pin the number of probe calls per stage, and pin that a disabled journal makes none.

The suite is deliberately free of cctbx and torch. The journal reaches the target only
through an injected probe, which is what makes that possible, and the Phenix-side
helpers are driven with fakes that mimic the flex arrays the scorer actually holds.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from phridge.client.intensity import nll_log as NL
from phridge.client.intensity.nll_log import NllJournal, NllPoint, StageKind


class _Probe:
    """A scripted target. Counts its calls, because each one is an exact evaluation."""

    def __init__(self, *values, n_work=1000, n_free=100):
        self.values = list(values)
        self.calls = 0
        self.n_work = n_work
        self.n_free = n_free
        self.fail = False

    def __call__(self) -> NllPoint:
        self.calls += 1
        if self.fail:
            raise RuntimeError("worker went away")
        value = self.values.pop(0) if self.values else float("nan")
        return NllPoint(work=value, free=value + 0.1, n_work=self.n_work, n_free=self.n_free)


def _journal(*values, **kw) -> tuple[NllJournal, _Probe]:
    probe = _Probe(*values, **kw)
    return NllJournal(probe, enabled=True, register_atexit=False), probe


# ------------------------------------------------------------------ cost and opting out
def test_a_disabled_journal_never_evaluates_the_target():
    """``PHRIDGE_NLL_LOG=0`` has to buy back the whole cost, not just the printing.

    An exact evaluation is the most expensive thing in the refinement, so a journal that
    still measured and merely stayed quiet would be a performance regression nobody asked
    for.
    """
    probe = _Probe(1.0, 0.9, 0.8)
    journal = NllJournal(probe, enabled=False, register_atexit=False)
    journal.mark_start()
    with journal.stage("coordinates (xyz) minimization", StageKind.model):
        pass
    journal.mark_end()
    journal.measure()
    assert probe.calls == 0
    assert journal.stages == []
    assert journal.report() == []


def test_a_stage_costs_exactly_two_evaluations():
    """One at entry, one at exit. A third would be a bug that doubles per-stage overhead."""
    journal, probe = _journal(2.0, 1.5)
    with journal.stage("coordinates (xyz) minimization", StageKind.model):
        pass
    assert probe.calls == 2


def test_the_run_start_reuses_a_measurement_the_caller_already_paid_for():
    """The run's first NLL and the first stage's entry NLL are the same point."""
    journal, probe = _journal(2.0, 1.5)
    entry = journal.measure()
    journal.mark_start(entry)
    assert probe.calls == 1
    assert journal.run_start is not None
    assert journal.run_start.work == pytest.approx(2.0)
    # and it is idempotent: a second call on a later cycle must not overwrite the start
    journal.mark_start(NllPoint(work=1.5, n_work=1000, n_free=100))
    assert journal.run_start.work == pytest.approx(2.0)
    assert probe.calls == 1


# ------------------------------------------------------------------ the delta itself
def test_a_stage_reports_the_change_it_caused():
    journal, _ = _journal(2.0, 1.4)
    with journal.stage("B-factors (ADP) minimization", StageKind.model) as record:
        pass
    assert record.delta_work == pytest.approx(-0.6)
    assert record.delta_free == pytest.approx(-0.6)
    line = record.line()
    assert "2.000000 -> 1.400000 (-0.600000)" in line
    assert "[model]" in line


def test_a_changed_reflection_set_makes_the_delta_incomparable():
    """The NLL is a per-reflection mean, so outlier rejection changes its denominator.

    Reporting the difference across that as an improvement would be reporting the effect
    of dropping reflections as the effect of refining, which is the one way this log could
    actively mislead.
    """
    probe = _Probe(2.0, 1.4)
    journal = NllJournal(probe, enabled=True, register_atexit=False)
    with journal.stage("bulk solvent + scaling", StageKind.target) as record:
        probe.n_work = 990  # an outlier was rejected mid-stage
    assert record.comparable is False
    assert record.delta_work is None
    assert "not comparable" in record.line()
    # and it must not be counted toward any total
    assert journal.totals()["target"] == (None, 0)


def test_unknown_counts_are_not_taken_as_evidence_that_the_set_moved():
    """A probe that does not report counts must still get its delta."""
    bare = NllPoint(work=2.0)
    assert bare.counts_match(NllPoint(work=1.0))
    record = NL.StageRecord("s", StageKind.model, bare, NllPoint(work=1.0))
    assert record.delta_work == pytest.approx(-1.0)


def test_a_failing_stage_is_recorded_and_its_error_still_propagates():
    """A stage that died halfway is the one whose NLL you most want to see."""
    journal, probe = _journal(2.0, 5.0)
    with pytest.raises(ValueError, match="lbfgs blew up"):
        with journal.stage("coordinates (xyz) minimization", StageKind.model):
            raise ValueError("lbfgs blew up")
    assert len(journal.stages) == 1
    assert journal.stages[0].delta_work == pytest.approx(+3.0)
    assert probe.calls == 2


def test_a_probe_failure_becomes_a_missing_number_not_a_zero():
    """The journal is instrumentation; it may not take the refinement down with it."""
    journal, probe = _journal(2.0)
    probe.fail = True
    point = journal.measure()
    assert point.ok is False and point.work is None
    with journal.stage("coordinates (xyz) minimization", StageKind.model) as record:
        pass
    assert record.delta_work is None
    # and it says the measurement is missing rather than printing a line of dashes that
    # would read as a stage which achieved nothing
    assert "NLL unavailable" in record.line()


# ------------------------------------------------------------------ attribution
def test_model_and_target_stages_are_totalled_separately():
    """A run whose progress came entirely from refitting nuisances has not moved the model.

    Both kinds lower the same number, so the only way to see that is to keep the two
    sums apart.
    """
    journal, _ = _journal(2.0, 1.8, 1.8, 1.5, 1.5, 1.45)
    start = journal.measure()
    journal.record_stage("bulk solvent + scaling", StageKind.target, start, journal.measure())
    with journal.stage("coordinates (xyz) minimization", StageKind.model):
        pass
    with journal.stage("B-factors (ADP) minimization", StageKind.model):
        pass
    totals = journal.totals()
    assert totals["target"][0] == pytest.approx(-0.2) and totals["target"][1] == 1
    assert totals["model"][0] == pytest.approx(-0.35) and totals["model"][1] == 2
    assert totals["overall_work"] == pytest.approx(-0.55)


def test_change_the_journal_did_not_bracket_is_called_out():
    """The stage table must never be mistaken for a closed account of the run.

    Here the NLL drops 2.0 -> 1.0 but the only bracketed stage accounts for 0.1 of it;
    the rest moved somewhere uninstrumented, which is a gap in the logging rather than in
    the refinement and has to be visible as one.
    """
    journal, _ = _journal(2.0, 1.9)
    with journal.stage("coordinates (xyz) minimization", StageKind.model):
        pass
    journal.run_end = NllPoint(work=1.0, free=1.1, n_work=1000, n_free=100)
    assert journal._unexplained(journal.totals()) == pytest.approx(-0.9)
    assert any("unattributed" in line for line in journal.report())


def test_a_run_with_no_overall_change_reports_no_unattributed_residue():
    journal, _ = _journal(2.0, 1.9)
    with journal.stage("coordinates (xyz) minimization", StageKind.model):
        pass
    assert journal._unexplained(journal.totals()) is None


# ------------------------------------------------------------------ weight scans
def _scan(journal, trials, selected=None):
    group = journal.new_group("XYZ target weight scan", "weight", extra_columns=("r_free",))
    for weight, work, free in trials:
        group.add(weight, NllPoint(work=work, free=free), {"r_free": 0.25})
    group.selected_value = selected
    return group


def test_the_scan_ranks_by_free_nll_and_marks_what_was_selected():
    """The choice is made on free NLL but upstream prints R-factors, so the scan table is
    the only place the decision can be checked."""
    journal, _ = _journal()
    group = _scan(
        journal,
        [(0.5, 1.51, 1.62), (1.0, 1.44, 1.55), (2.0, 1.41, 1.56), (4.0, 1.39, 1.61)],
        selected=1.0,
    )
    assert group.best().value == pytest.approx(1.0)
    assert group.margin() == pytest.approx(0.01)
    text = "\n".join(group.table())
    assert "ranked by free NLL" in text
    assert "selected weight=1.0000" in text
    # the selected row, and only it, carries the marker
    rows = [ln for ln in group.table() if ln.startswith("      ")]
    assert sum(1 for r in rows if r.lstrip().startswith("*")) == 1
    assert "1.440000" in [r for r in rows if r.lstrip().startswith("*")][0]


def test_a_scan_that_did_not_separate_its_best_two_trials_says_so():
    """A weight chosen by a gap in the sixth decimal has not really been chosen."""
    journal, _ = _journal()
    group = _scan(journal, [(1.0, 1.44, 1.550000), (2.0, 1.41, 1.550001)], selected=1.0)
    assert "did not really separate them" in "\n".join(group.table())


def test_a_single_trial_is_not_a_scan():
    """Phenix runs one 'trial' whenever weight optimization is off."""
    journal, _ = _journal()
    journal.add_group(_scan(journal, [(1.0, 1.44, 1.55)], selected=1.0))
    assert journal.groups == []
    journal.add_group(_scan(journal, [(1.0, 1.44, 1.55), (2.0, 1.4, 1.6)], selected=1.0))
    assert len(journal.groups) == 1


def test_a_trial_with_no_nll_cannot_win_the_scan():
    """A trial whose evaluation failed must rank last, not first."""
    journal, _ = _journal()
    group = _scan(journal, [(1.0, None, None), (2.0, 1.41, 1.56)], selected=2.0)
    assert group.best().value == pytest.approx(2.0)
    assert math.isinf(group.ranked()[-1].sort_key())
    assert NL._MISSING in "\n".join(group.table())


def test_scans_are_suppressed_on_their_own_switch():
    """``PHRIDGE_NLL_LOG_TRIALS=0`` drops the verbose part and keeps the stage lines."""
    probe = _Probe(2.0, 1.5)
    journal = NllJournal(probe, enabled=True, trials=False, register_atexit=False)
    journal.add_group(_scan(journal, [(1.0, 1.44, 1.55), (2.0, 1.4, 1.6)], selected=1.0))
    assert journal.groups == []
    with journal.stage("coordinates (xyz) minimization", StageKind.model):
        pass
    assert len(journal.stages) == 1


# ------------------------------------------------------------------ the report
def test_the_report_leads_with_the_run_start_and_ends_with_the_run_end():
    journal, _ = _journal(2.0, 1.8, 1.8, 1.5)
    start = journal.measure()
    journal.mark_start(start)
    journal.record_stage("bulk solvent + scaling", StageKind.target, start, journal.measure())
    with journal.stage("coordinates (xyz) minimization", StageKind.model):
        pass
    lines = journal.report()
    body = [ln for ln in lines if ln.startswith(" ") and not ln.startswith(" mli")]
    assert any(ln.strip().startswith("run start") for ln in body)
    assert any(ln.strip().startswith("run end") for ln in body)
    text = "\n".join(lines)
    assert "overall change: work -0.500000" in text
    # both kinds appear with their own sign, and the stage labels survive
    assert "bulk solvent + scaling" in text and "coordinates (xyz) minimization" in text


def test_the_report_is_printed_once_even_with_the_atexit_backstop():
    """A run that dies mid-refinement is when the trace matters most, but a run that
    ended cleanly must not print it twice."""
    journal, _ = _journal(2.0, 1.5)
    with journal.stage("coordinates (xyz) minimization", StageKind.model):
        pass
    emitted: list[str] = []
    journal.emit = lambda lines: emitted.extend(lines)  # type: ignore[assignment]
    journal.print_report()
    assert emitted
    before = len(emitted)
    journal._atexit_report()
    assert len(emitted) == before


def test_a_long_stage_label_is_truncated_rather_than_breaking_the_table():
    journal, _ = _journal(2.0, 1.5)
    with journal.stage("x" * 200, StageKind.model):
        pass
    for line in journal.report():
        assert len(line) < 140


def test_the_formatter_survives_absent_and_non_finite_values():
    """NaN reaches this from a failed quadrature; infinity from a sentinel."""
    assert NL.fmt_value(None) == NL._MISSING
    assert NL.fmt_value(float("nan")) == NL._MISSING
    assert NL.fmt_value(float("inf")) == NL._MISSING
    assert NL.fmt_value(1.5) == "1.500000"
    assert NL.fmt_value(None, "12.6f") == NL._MISSING.rjust(12)
    assert NL.fmt_value(1.5, "12.6f") == "    1.500000"


def test_the_env_switches_default_on_and_accept_the_usual_falsehoods(monkeypatch):
    monkeypatch.delenv("PHRIDGE_NLL_LOG", raising=False)
    monkeypatch.delenv("PHRIDGE_NLL_LOG_TRIALS", raising=False)
    assert NL.nll_logging_enabled() and NL.nll_trials_enabled()
    for value in ("0", "false", "no", "OFF", " off "):
        monkeypatch.setenv("PHRIDGE_NLL_LOG", value)
        assert NL.nll_logging_enabled() is False
    monkeypatch.setenv("PHRIDGE_NLL_LOG", "1")
    assert NL.nll_logging_enabled() is True
    monkeypatch.setenv("PHRIDGE_NLL_LOG_TRIALS", "0")
    assert NL.nll_trials_enabled() is False


# ================================================================ the Phenix-side wiring
# These drive the hook helpers with fakes standing in for the flex arrays the Phenix
# scorer holds, so the wiring is covered without cctbx.


class _Flex(list):
    """Just enough of ``flex.double`` for the helpers under test."""

    def size(self) -> int:
        return len(self)


def _fake_scorer(journal, weights, nll_w, nll_f):
    hook = pytest.importorskip("phridge.client.intensity.phenix_hook")
    scorer = type("S", (), {})()
    scorer.w = _Flex(weights)
    scorer.nll_w = _Flex(nll_w)
    scorer.nll_f = _Flex(nll_f)
    scorer.rw = _Flex([0.21] * len(weights))
    scorer.rf = _Flex([0.25] * len(weights))
    scorer.b = _Flex([0.013] * len(weights))
    scorer.a = _Flex([1.1] * len(weights))
    scorer._phridge_nll = True
    scorer._phridge_journal = journal
    return hook, scorer


def test_the_missing_nll_sentinel_never_reaches_the_scan_as_a_number():
    """99999 is the scorer's 'never pick this'; printed as a value it would read as an NLL."""
    hook = pytest.importorskip("phridge.client.intensity.phenix_hook")
    assert hook._unsentinel(1.25) == pytest.approx(1.25)
    assert hook._unsentinel(hook._NLL_SENTINEL) is None
    assert hook._unsentinel(float("nan")) is None
    assert hook._unsentinel("not a number") is None


def test_the_xyz_scan_is_snapshotted_before_selection_prunes_the_arrays():
    """``finalize`` selects in place, so after it the scan no longer exists to be logged."""
    journal, _ = _journal()
    hook, scorer = _fake_scorer(
        journal, [0.5, 1.0, 2.0], [1.51, 1.44, 1.41], [1.62, 1.55, 1.56]
    )
    group = hook._xyz_trial_group(scorer)
    assert group is not None and len(group.records) == 3

    # selection keeps only the winner; the snapshot taken above is unaffected
    scorer.w, scorer.nll_w, scorer.nll_f = _Flex([1.0]), _Flex([1.44]), _Flex([1.55])
    assert len(group.records) == 3
    group.selected_value = 1.0
    assert group.best().value == pytest.approx(1.0)


def test_the_xyz_scan_is_skipped_when_there_is_no_nll_to_report():
    """Under ``PHRIDGE_WEIGHT_METRIC=rfree`` no NLL was collected, so there is no scan."""
    journal, _ = _journal()
    hook, scorer = _fake_scorer(journal, [0.5, 1.0], [1.5, 1.4], [1.6, 1.5])
    scorer._phridge_nll = False
    assert hook._xyz_trial_group(scorer) is None
    scorer._phridge_nll = True
    scorer._phridge_journal = None
    assert hook._xyz_trial_group(scorer) is None
    # mismatched array lengths mean the patches are out of step; drop the scan rather
    # than pair a weight with another trial's NLL
    _, bad = _fake_scorer(journal, [0.5, 1.0], [1.5], [1.6])
    assert hook._xyz_trial_group(bad) is None


def test_the_adp_trial_line_carries_the_free_nll_at_a_useful_precision():
    """Upstream prints the work target at three decimals and the free value not at all.

    Three decimals cannot separate two ADP trials whose NLLs differ in the fifth, and the
    free value is the one the selection is actually made on.
    """
    hook = pytest.importorskip("phridge.client.intensity.phenix_hook")
    result = type("R", (), {})()
    result.weight = 12.5
    result.nll_work = 1.2012349
    result.nll_free = 1.2880002
    line = hook._adp_trial_line(result)
    assert "weight=12.5000" in line and "1.201235" in line and "1.288000" in line
    # an unmeasured trial prints as missing rather than as zero
    result.nll_free = None
    assert NL._MISSING in hook._adp_trial_line(result)


def test_the_adp_scan_records_only_what_came_from_a_trial():
    """``show`` also runs for the pre-scan model and again for the winner.

    Recording those would put the same weight in the table twice and let the baseline
    compete as if it were a trial.
    """
    journal, _ = _journal()
    hook = pytest.importorskip("phridge.client.intensity.phenix_hook")
    refiner = type("A", (), {})()
    refiner.log = None
    refiner._phridge_journal = journal
    refiner._phridge_group = journal.new_group("ADP target weight scan", "weight")
    refiner._phridge_trial_weight = None

    def result(weight, work, free):
        r = type("R", (), {})()
        r.weight, r.nll_work, r.nll_free = weight, work, free
        r.r_work_rfactor, r.r_free_rfactor, r.delta_b = 21.0, 25.0, 3.0
        return r

    hook._log_adp_trial(refiner, result(1.0, 1.5, 1.6))  # the pre-scan baseline
    assert refiner._phridge_group.records == []

    refiner._phridge_trial_weight = 2.0
    hook._log_adp_trial(refiner, result(2.0, 1.44, 1.55))
    hook._log_adp_trial(refiner, result(4.0, 1.41, 1.58))
    assert len(refiner._phridge_group.records) == 2

    refiner.target_weights = type(
        "W", (), {"adp_weights_result": type("R", (), {"wx": 2.0})()}
    )()
    hook._close_adp_scan(refiner)
    assert refiner._phridge_group is None
    assert len(journal.groups) == 1
    assert journal.groups[0].selected_value == pytest.approx(2.0)


def test_a_trial_row_is_printed_once_across_the_serial_and_parallel_paths():
    """With ``nproc>1`` the trial runs in a subprocess and the parent prints the result.

    Both paths can reach the same object, so the row carries its own printed flag.
    """
    journal, _ = _journal()
    hook = pytest.importorskip("phridge.client.intensity.phenix_hook")
    import io

    log = io.StringIO()
    refiner = type("A", (), {})()
    refiner.log = log
    refiner._phridge_group = None
    refiner._phridge_trial_weight = None
    r = type("R", (), {})()
    r.weight, r.nll_work, r.nll_free = 2.0, 1.44, 1.55
    hook._log_adp_trial(refiner, r)
    hook._log_adp_trial(refiner, r)
    assert log.getvalue().count("ADP trial") == 1
    assert r._phridge_printed is True


def test_the_nll_pair_prefers_the_single_pass_probe():
    """Two full quadratures for two numbers one call already produced, once per trial."""
    hook = pytest.importorskip("phridge.client.intensity.phenix_hook")

    class _Engine:
        def __init__(self):
            self.probe_calls = 0
            self.legacy_calls = 0

        def nll_point(self):
            self.probe_calls += 1
            return NllPoint(work=1.2, free=1.3, n_work=900, n_free=100)

        def target_w(self):
            self.legacy_calls += 1
            return 1.2

        def target_t(self):
            self.legacy_calls += 1
            return 1.3

    engine = _Engine()
    assert hook._nll_pair(engine) == (pytest.approx(1.2), pytest.approx(1.3))
    assert engine.probe_calls == 1 and engine.legacy_calls == 0

    # an fmodel without the probe still works, at the old cost
    class _Legacy(_Engine):
        nll_point = None

    legacy = _Legacy()
    assert hook._nll_pair(legacy) == (pytest.approx(1.2), pytest.approx(1.3))
    assert legacy.legacy_calls == 2

    # and one that cannot produce a target at all raises rather than inventing a value
    class _Broken:
        def target_w(self):
            raise RuntimeError("no data")

    with pytest.raises(ValueError, match="no NLL available"):
        hook._nll_pair(_Broken())


def test_a_free_set_that_is_absent_falls_back_to_the_work_value():
    """``_nll_pair`` promises two numbers; with no free set both are the work value."""
    hook = pytest.importorskip("phridge.client.intensity.phenix_hook")

    class _NoFree:
        def nll_point(self):
            return NllPoint(work=1.7)

    assert hook._nll_pair(_NoFree()) == (pytest.approx(1.7), pytest.approx(1.7))


# ================================================================ interleaved mirroring
def test_interleaved_blocks_are_mirrored_into_the_journal_for_free():
    """The ladder measures NLL_0 and NLL_1 exactly to adjudicate each block.

    The journal must take those values; re-measuring them would double the cost of the
    most expensive thing the controller does.
    """
    from phridge.client.intensity.interleaved import (
        BlockBudget,
        BlockOutcome,
        BlockRecord,
        BlockSite,
        InterleavedController,
        InterleavedOptions,
    )

    journal, probe = _journal()
    controller = InterleavedController(
        object(),  # type: ignore[arg-type]
        InterleavedOptions(verbose=False),
        journal=journal,
    )
    controller._record(
        BlockRecord(
            index=3,
            outcome=BlockOutcome.accepted,
            nll_0=1.50,
            nll_1=1.42,
            exact_delta=-0.08,
            predicted_delta=-0.09,
            n_exact_evals=2,
            n_inner_evals=25,
            n_fit_fallback=0,
            n_total=1000,
            budget=BlockBudget(max_iterations=25),
            site=BlockSite(stage="coordinates (xyz) minimization", macro_cycle=2,
                           total_macro_cycles=5),
        )
    )
    assert probe.calls == 0
    assert len(journal.stages) == 1
    stage = journal.stages[0]
    assert stage.delta_work == pytest.approx(-0.08)
    assert "macro cycle 2/5" in stage.label
    assert "block 3" in stage.label
    assert stage.detail == "accepted"


# ------------------------------------------------------- the normalization offset
def test_the_nll_of_the_data_gains_log_epsilon_sigma_when_sigma_changes():
    """-log p(I) = -log p(Z) + log(εΣ). Doubling Σ must add exactly log(2).

    This is the term the nuisance fit changes and the quadrature does not include. Leave
    it out and replacing the per-bin Σ with a Wilson curve reports the change of units
    as a change in the likelihood.
    """
    eps = np.array([1.0, 2.0, 1.0, 1.0])
    sw = np.array([1.0, 1.0, np.e, np.e])
    free = np.array([False, False, True, True])
    work, held = NL.normalization_offset(eps, sw, free)
    # work: log(1·1) = 0 and log(2·1) = log(2). free: both are log(e) = 1.
    assert work == pytest.approx(0.5 * math.log(2.0))
    assert held == pytest.approx(1.0) 

    work2, held2 = NL.normalization_offset(eps, sw * 2.0, free)
    assert work2 - work == pytest.approx(math.log(2.0))
    assert held2 - held == pytest.approx(math.log(2.0))


def test_the_offset_gives_up_rather_than_pairing_mismatched_arrays():
    """A wrong-length mask would add one reflection's normalization to another's NLL."""
    assert NL.normalization_offset([1.0, 1.0], [1.0]) == (None, None)
    assert NL.normalization_offset([1.0], [1.0], free=[True, False]) == (None, None)
    # no free set: the free mean is absent, the work mean covers everything
    work, held = NL.normalization_offset([1.0, np.e], [1.0, 1.0])
    assert work == pytest.approx(0.5)
    assert held is None


def test_a_log_that_already_writes_to_the_terminal_is_not_written_twice():
    """Phenix's log is a multi_out that already contains stdout.

    Writing the line to stdout as well is what pairs every line in the refinement log.
    """
    import io
    import sys

    from phridge.client.intensity.heartbeat import output_streams

    class _Multi:
        def __init__(self, *handles):
            self.file_objects = list(handles)

        def write(self, _text):
            pass

    tee = _Multi(sys.stdout)
    assert output_streams(tee) == [tee]
    assert output_streams(None) == [sys.stdout]
    assert output_streams(sys.stdout) == [sys.stdout]

    # a log that is only a file still needs the terminal copy
    only_file = io.StringIO()
    assert output_streams(only_file) == [sys.stdout, only_file]


def test_interleaved_blocks_are_shifted_into_the_same_units_without_changing_the_delta():
    """Inside a block Σ_W is fixed, so both ends take the same log(εΣ).

    The delta the block was accepted on must survive the shift, otherwise the journal
    would disagree with the decision it is reporting.
    """
    from phridge.client.intensity.interleaved import (
        BlockBudget,
        BlockOutcome,
        BlockRecord,
        BlockSite,
        InterleavedController,
        InterleavedOptions,
    )

    journal, _probe = _journal()

    class _Host:
        def _normalization_offset(self):
            return (4.0, 4.1)  # log(εΣ): large, and identical at both ends of a block

    controller = InterleavedController(
        _Host(),  # type: ignore[arg-type]
        InterleavedOptions(verbose=False),
        journal=journal,
    )
    controller._record(
        BlockRecord(
            index=1,
            outcome=BlockOutcome.accepted,
            nll_0=1.50,
            nll_1=1.42,
            exact_delta=-0.08,
            predicted_delta=None,
            n_exact_evals=2,
            n_inner_evals=0,
            n_fit_fallback=0,
            n_total=0,
            budget=BlockBudget(),
            site=BlockSite(stage="coordinates (xyz) minimization", macro_cycle=1,
                           total_macro_cycles=3),
        )
    )
    stage = journal.stages[0]
    assert stage.start.work == pytest.approx(5.50)
    assert stage.end.work == pytest.approx(5.42)
    assert stage.delta_work == pytest.approx(-0.08)


def test_the_nuisance_stage_says_how_far_the_normalization_moved():
    from phridge.client.intensity.engine import _nuisance_detail

    assert _nuisance_detail("free/1387", (1.2, 1.1), (2.1, 2.0)) == (
        "tune=free/1387, log(εΣ) work 1.200 -> 2.100, free 1.100 -> 2.000"
    )
    assert _nuisance_detail("free/1387", (1.2, None), (2.1, None)) == (
        "tune=free/1387, log(εΣ) work 1.200 -> 2.100"
    )
    # an unavailable offset must not invent a number
    assert _nuisance_detail("all/100", (None, None), (None, None)) == "tune=all/100"


def test_the_exact_branch_only_buys_an_entry_nll_when_a_journal_wants_one():
    """Without a journal the fully-exact branch has no use for a before-value."""
    from phridge.client.intensity.interleaved import InterleavedController, InterleavedOptions

    bare = InterleavedController(object(), InterleavedOptions(verbose=False))  # type: ignore[arg-type]
    assert bare._journal_entry_nll() is None

    journal, _ = _journal()
    disabled = NllJournal(_Probe(), enabled=False, register_atexit=False)
    off = InterleavedController(
        object(), InterleavedOptions(verbose=False), journal=disabled  # type: ignore[arg-type]
    )
    assert off._journal_entry_nll() is None
