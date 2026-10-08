"""Partition schemes, the calibration merge, and the audit rule."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from phridge.contracts.tensors import Observed
from phridge.core.graph import Graph, Node
from phridge.core.partitions import AuditRefused, PartitionRefused, Partitions, audit_rule
from phridge.core.registry import Registry


def _observed(n: int = 20) -> Observed:
    return Observed(
        index={"h": torch.arange(n)},
        data={"F": torch.ones(n, dtype=torch.float64)},
        partitions={},
    )


def _cover(parts: Partitions) -> None:
    union = torch.zeros(next(iter(parts.masks.values())).shape, dtype=torch.bool)
    for mask in parts.masks.values():
        assert mask.dtype == torch.bool
        assert not bool((union & mask).any())
        union = union | mask
    assert bool(union.all())


def test_schemes_cover_the_observed_rows() -> None:
    observed = _observed()
    two = Partitions("two", None, seed=1, unit="reflection", observed=observed)
    four = Partitions("four", None, seed=1, unit="reflection", observed=observed)
    halves = Partitions("halfmap", None, seed=1, unit="reflection", observed=observed)
    assert set(two.masks) == {"tune", "free"}
    assert set(four.masks) == {"calibration", "tune", "free", "audit"}
    assert set(halves.masks) == {"half1", "half2"}
    _cover(two)
    _cover(four)
    _cover(halves)
    other = Partitions("four", None, seed=2, unit="reflection", observed=observed)
    assert not torch.equal(four.masks["tune"], other.masks["tune"])


def test_calibration_may_merge_into_tune_and_audit_never_merges() -> None:
    parts = Partitions("four", None, seed=0, unit="reflection", observed=_observed())
    calibration = parts.masks["calibration"].clone()
    tune = parts.masks["tune"].clone()
    parts.merge("calibration", into="tune")
    assert torch.equal(parts.masks["tune"], tune | calibration)
    assert not bool(parts.masks["calibration"].any())
    with pytest.raises(PartitionRefused, match="audit"):
        parts.merge("audit", into="tune")
    with pytest.raises(PartitionRefused, match="calibration"):
        parts.merge("free", into="tune")


def test_audit_rule_refuses_a_fit_that_reads_audit() -> None:
    empty = Graph(Registry())
    audit_rule(empty)
    graph = Graph(Registry())
    graph.add(Node(name="partition.audit", kind="block"))
    graph.add(Node(name="statistics", kind="derived", reads=["partition.audit"]))
    audit_rule(graph)
    graph.add(Node(name="fit", kind="derived", reads=["partition.audit"]))
    with pytest.raises(AuditRefused, match="partition.audit") as caught:
        audit_rule(graph)
    assert caught.value.path == "partition.audit"
