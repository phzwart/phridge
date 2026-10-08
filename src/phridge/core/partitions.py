"""Observation partitions and the rule that keeps audit data out of a fit."""

from __future__ import annotations

from typing import Literal, Mapping, Optional

from phridge.contracts.tensors import Observed
from phridge.core.graph import Graph

Scheme = Literal["two", "four", "halfmap"]
Unit = Literal["reflection", "particle", "crystal_consistent"]

_NAMES: dict[str, tuple[str, ...]] = {
    "two": ("tune", "free"),
    "four": ("calibration", "tune", "free", "audit"),
    "halfmap": ("half1", "half2"),
}
SCHEME_NAMES = _NAMES
_GROUP_KEY = {"particle": "particle", "crystal_consistent": "crystal"}


class PartitionRefused(Exception):
    """A merge the scheme does not allow."""

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


class AuditRefused(Exception):
    """``partition.audit`` is read by something other than ``statistics``."""

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


class Partitions:
    """Boolean masks over an ``Observed``, one per name in the scheme."""

    def __init__(
        self,
        scheme: Scheme,
        fractions: Optional[Mapping[str, float]],
        seed: int,
        unit: Unit,
        observed: Observed,
    ) -> None:
        if scheme not in _NAMES:
            raise ValueError(f"unknown partition scheme {scheme!r}")
        self.scheme = scheme
        self.fractions = dict(fractions) if fractions is not None else None
        self.seed = seed
        self.unit = unit
        self.masks = _assign(_NAMES[scheme], self.fractions, seed, unit, observed)

    def merge(self, source: str, into: str) -> None:
        """Calibration may merge into tune. Audit never merges."""
        if source == "audit" or into == "audit":
            raise PartitionRefused("partition.audit", "audit never merges")
        if source != "calibration" or into != "tune":
            raise PartitionRefused(source, "only calibration may merge into tune")
        self.masks[into] = self.masks[into] | self.masks[source]
        self.masks[source] = self.masks[source] & False


def audit_rule(graph: Graph) -> None:
    """Pass when ``partition.audit`` is absent. Otherwise only ``statistics`` may read it."""
    present = "partition.audit" in graph.nodes or any("partition.audit" in node.reads for node in graph.nodes.values())
    if not present:
        return
    if graph.readers("partition.audit") != {"statistics"}:
        raise AuditRefused("partition.audit", "only statistics may read partition.audit")


def _assign(
    names: tuple[str, ...],
    fractions: Optional[dict[str, float]],
    seed: int,
    unit: Unit,
    observed: Observed,
) -> dict[str, object]:
    import torch

    length = _length(observed)
    weights = _fractions(names, fractions)
    labels = _labels(observed, unit, length)
    seen: list[object] = []
    for label in labels.tolist():
        if label not in seen:
            seen.append(label)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    order = torch.randperm(len(seen), generator=generator).tolist()
    shuffled = [seen[index] for index in order]
    sizes = _sizes(len(shuffled), weights)
    assigned: dict[object, str] = {}
    cursor = 0
    for name, size in zip(names, sizes):
        for label in shuffled[cursor : cursor + size]:
            assigned[label] = name
        cursor += size
    masks: dict[str, object] = {}
    for name in names:
        mask = torch.zeros(length, dtype=torch.bool)
        for index, label in enumerate(labels.tolist()):
            if assigned[label] == name:
                mask[index] = True
        masks[name] = mask
    return masks


def _length(observed: Observed) -> int:
    for tensor in (*observed.data.values(), *observed.index.values(), *observed.partitions.values()):
        return int(tensor.shape[0])
    raise ValueError("observed is empty")


def _fractions(names: tuple[str, ...], fractions: Optional[dict[str, float]]) -> list[float]:
    if fractions is None:
        return [1.0 / len(names)] * len(names)
    missing = [name for name in names if name not in fractions]
    if missing:
        raise ValueError(f"fractions missing {missing}")
    return [float(fractions[name]) for name in names]


def _labels(observed: Observed, unit: Unit, length: int) -> object:
    import torch

    key = _GROUP_KEY.get(unit)
    if key is not None and key in observed.index:
        return observed.index[key].reshape(length).detach().cpu()
    return torch.arange(length)


def _sizes(count: int, weights: list[float]) -> list[int]:
    sizes = [int(weight * count) for weight in weights]
    sizes[-1] += count - sum(sizes)
    if sizes[-1] < 0:
        raise ValueError("fractions assign more groups than exist")
    return sizes
