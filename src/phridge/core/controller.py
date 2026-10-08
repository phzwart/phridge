"""Controller. One state machine; policies decide which action comes next."""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Optional

from pydantic import BaseModel, ConfigDict

from phridge.core.graph import Graph
from phridge.core.moveset import MoveSet, MoveSetRefused, validate
from phridge.core.partitions import AuditRefused, audit_rule
from phridge.core.receipts import Receipt, ReceiptLog
from phridge.core.registry import Registry


class State(str, Enum):
    """Where the controller is. Actions below are the only way to move."""

    Load = "Load"
    Configure = "Configure"
    Calibrate = "Calibrate"
    Ready = "Ready"
    Step = "Step"
    Evaluate = "Evaluate"
    Report = "Report"
    Reconfigure = "Reconfigure"
    Finalize = "Finalize"
    Done = "Done"


class ControllerRefused(Exception):
    """A guard rejected an action."""

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


class ShellRow(BaseModel):
    """One shell in an observation. Scalars only."""

    model_config = ConfigDict(extra="forbid")

    shell: int
    value: float


class Observation(BaseModel):
    """What an action returns. Scalars, tables, counts, and versions. Never arrays."""

    model_config = ConfigDict(extra="forbid")

    action: str
    state: str
    count: int
    versions: dict[str, int]
    shells: list[ShellRow] = []
    scalars: dict[str, float] = {}


class Controller:
    """Guards the registry. Every action appends a receipt and returns an observation."""

    def __init__(
        self,
        registry: Registry,
        graph: Graph,
        move_set: Optional[MoveSet] = None,
        graph_capabilities: Optional[dict[str, set[str]]] = None,
    ) -> None:
        self.registry = registry
        self.graph = graph
        self.move_set = move_set
        self.graph_capabilities = graph_capabilities or {}
        self.state = State.Load
        self.receipts = ReceiptLog()
        self._accepted = False
        self._proposed = False

    def configure(self) -> Observation:
        self._from(State.Load, State.Reconfigure, State.Ready)
        return self._land("configure", State.Configure)

    def calibrate(self) -> Observation:
        self._from(State.Configure, State.Ready)
        return self._land("calibrate", State.Ready)

    def step(self, update: Optional[Callable[[Registry], None]] = None) -> Observation:
        if self.state == State.Report:
            raise ControllerRefused("step", "step after report until continue_")
        self._from(State.Ready)
        self._guard_move_set()
        self._guard_audit()
        if update is not None:
            update(self.registry)
        self._accepted = False
        return self._land("step", State.Step)

    def evaluate(self) -> Observation:
        self._from(State.Step)
        return self._land("evaluate", State.Evaluate)

    def accept(self) -> Observation:
        self._from(State.Evaluate)
        self._accepted = True
        return self._land("accept", State.Ready)

    def reject(self) -> Observation:
        self._from(State.Evaluate)
        return self._land("reject", State.Ready)

    def freeze(self, block_id: str) -> Observation:
        self._from(State.Ready, State.Configure)
        self.registry.freeze(block_id)
        return self._land("freeze", self.state)

    def unfreeze(self, block_id: str) -> Observation:
        self._from(State.Ready, State.Configure)
        self.registry.unfreeze(block_id)
        return self._land("unfreeze", self.state)

    def report(self) -> Observation:
        if not self._accepted:
            raise ControllerRefused("report", "report twice without an accepted step")
        self._from(State.Ready)
        self._accepted = False
        return self._land("report", State.Report)

    def continue_(self) -> Observation:
        self._from(State.Report)
        return self._land("continue_", State.Ready)

    def reconfigure(self) -> Observation:
        if self.state in (State.Step, State.Evaluate):
            raise ControllerRefused("reconfigure", "reconfigure refused from Step/Evaluate")
        return self._land("reconfigure", State.Reconfigure)

    def finalize(self) -> Observation:
        self._from(State.Ready, State.Report, State.Finalize)
        if self.state == State.Finalize:
            return self._land("finalize", State.Done)
        return self._land("finalize", State.Finalize)

    def query(self) -> Observation:
        return self._land("query", self.state)

    def propose(self) -> Observation:
        if self.state != State.Ready:
            raise ControllerRefused("propose", "propose only from Ready")
        self._proposed = True
        return self._land("propose", State.Ready)

    def apply(self, block_id: str, new_shape: tuple[int, ...], new_value: Any) -> Observation:
        """Reshape one block. Only after ``propose``. This is the structural event."""
        if not self._proposed:
            raise ControllerRefused("apply", "apply only after a propose")
        self.registry.reshape(block_id, new_shape, new_value)
        if block_id in self.graph.nodes:
            self.graph.structural_bump(block_id)
        self._proposed = False
        return self._land("apply", State.Ready)

    def replay(self, registry: Registry) -> None:
        """Write each receipt's payload, in order, so the block values match the live registry."""
        import torch

        for receipt in self.receipts:
            for block_id, shape, values in receipt.payload:
                block = registry.block(block_id)
                tensor = torch.tensor(values, dtype=torch.float64).reshape(shape)
                if tuple(block.shape) != tuple(shape):
                    registry.reshape(block_id, shape, tensor)
                elif not torch.equal(block.value.detach().cpu(), tensor):
                    registry.set_value(block_id, tensor)

    def _from(self, *allowed: State) -> None:
        if self.state not in allowed:
            raise ControllerRefused(self.state.value, f"not in {[item.value for item in allowed]}")

    def _guard_move_set(self) -> None:
        if self.move_set is None:
            return
        try:
            validate(self.move_set, self.registry, self.graph_capabilities)
        except MoveSetRefused as exc:
            raise ControllerRefused(exc.entry, exc.reason) from exc

    def _guard_audit(self) -> None:
        try:
            audit_rule(self.graph)
        except AuditRefused as exc:
            raise ControllerRefused(exc.path, exc.reason) from exc

    def _land(self, action: str, state: State) -> Observation:
        self.state = state
        payload = _snapshot(self.registry)
        self.receipts.append(
            Receipt(
                action=action,
                node=state.value,
                inputs_versions={},
                output_version=(self.registry.epoch(), len(self.receipts) + 1),
                wall_time=0.0,
                payload=payload,
            )
        )
        versions = {block.id: block.value_version for block in _blocks(self.registry)}
        return Observation(action=action, state=state.value, count=len(self.receipts), versions=versions)


def _blocks(registry: Registry) -> list[Any]:
    ids = getattr(registry, "_blocks", {})
    return [registry.block(name) for name in ids]


def _snapshot(registry: Registry) -> tuple[tuple[str, tuple[int, ...], tuple[float, ...]], ...]:
    rows = []
    for block in _blocks(registry):
        flat = tuple(float(value) for value in block.value.detach().cpu().reshape(-1).tolist())
        rows.append((block.id, tuple(int(dim) for dim in block.shape), flat))
    return tuple(rows)
