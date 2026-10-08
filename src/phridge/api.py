"""Phenix / torch-facing Bridge: convert, enqueue, wait, convert back."""

from __future__ import annotations

import uuid
from typing import Any, Optional

from phridge.shims.cctbx import from_canonical, is_cctbx_object, to_canonical
from phridge.codec import decode_ref, encode_value
from phridge.contracts.models import JobEnvelope, JobStatus, WorkerRuntime, utcnow
from phridge.ops import get_op
from phridge.protocol import Protocol
from phridge.redis_store import (
    DEFAULT_MAX_OBJECT_BYTES,
    DEFAULT_TTL_SECONDS,
    RedisStore,
)


class JobFailed(RuntimeError):
    def __init__(self, job_id: str, error_type: str, message: str, traceback: Optional[str] = None):
        super().__init__(f"job {job_id} failed: {error_type}: {message}")
        self.job_id = job_id
        self.error_type = error_type
        self.message = message
        self.traceback = traceback


class Bridge:
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        *,
        store: Optional[RedisStore] = None,
        memory: bool = False,
        device: str = "cpu",
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        max_object_bytes: int = DEFAULT_MAX_OBJECT_BYTES,
        timeout: float = 3600.0,
    ) -> None:
        if memory and store is not None:
            raise ValueError("pass memory=True or store=, not both")
        if memory:
            self.store = RedisStore.memory(
                ttl_seconds=ttl_seconds,
                max_object_bytes=max_object_bytes,
            )
        else:
            self.store = store or RedisStore.from_url(
                redis_url,
                ttl_seconds=ttl_seconds,
                max_object_bytes=max_object_bytes,
            )
        self.protocol = Protocol(self.store)
        self.timeout = timeout
        self.memory = memory
        self.device = device
        self._prefer_cctbx: dict[str, bool] = {}

    def call(self, op: str, timeout: Optional[float] = None, **kwargs: Any) -> Any:
        job_id = self.submit(op, **kwargs)
        return self.result(job_id, timeout=timeout)

    def submit(self, op: str, **kwargs: Any) -> str:
        spec = get_op(op)
        job_id = str(uuid.uuid4())
        self._prefer_cctbx[job_id] = any(is_cctbx_object(v) for v in kwargs.values())
        inputs = {}
        for name, value in kwargs.items():
            inputs[name] = encode_value(self.store, job_id, name, to_canonical(value))
        envelope = JobEnvelope(
            job_id=job_id,
            op=spec.name,
            inputs=inputs,
            status=JobStatus.queued,
            created_at=utcnow(),
            updated_at=utcnow(),
        )
        self.protocol.enqueue(envelope)
        if self.memory:
            self._process_memory(envelope.job_id, WorkerRuntime(spec.runtime))
        return job_id

    def _process_memory(self, job_id: str, runtime: WorkerRuntime) -> None:
        loaded = self.store.get_envelope(job_id)
        if loaded is None:
            raise RuntimeError(f"memory store lost envelope {job_id}")
        if runtime == WorkerRuntime.cctbx:
            from phridge.cctbx_worker.runner import process_envelope

            process_envelope(self.store, loaded)
        else:
            from phridge.worker.runner import process_envelope

            process_envelope(self.store, loaded, device=self.device)

    def result(self, job_id: str, timeout: Optional[float] = None) -> Any:
        envelope = self.protocol.wait(job_id, timeout if timeout is not None else self.timeout)
        if envelope.status == JobStatus.error:
            err = envelope.error
            raise JobFailed(
                job_id,
                err.type if err else "Error",
                err.message if err else "unknown error",
                err.traceback if err else None,
            )
        prefer_cctbx = self._prefer_cctbx.pop(job_id, False)
        decoded = {
            name: from_canonical(decode_ref(self.store, ref), prefer_cctbx=prefer_cctbx)
            for name, ref in envelope.outputs.items()
        }
        if len(decoded) == 1:
            return next(iter(decoded.values()))
        return decoded


class Refinement:
    """One configuration, run by a policy, then written out through ``io``."""

    def __init__(self, config: Any, registry: Any, graph: Any, partitions: Any, components: dict[str, Any]) -> None:
        self.config = config
        self.registry = registry
        self.graph = graph
        self.partitions = partitions
        self.components = components
        self.structure: Any = None
        self.observed: Any = None
        self.history: list[Any] = []
        self.controller: Any = None
        self.r_free_trace: list[float] = []

    @classmethod
    def from_config(cls, path: str) -> Refinement:
        """Load a YAML configuration and, when it names files, the model and the data."""
        from phridge.core.config import build, load
        from phridge.io.read_data import read_data
        from phridge.shims.cctbx.xray_structure import xray_structure_from_pdb

        config = load(path)
        registry, graph, partitions, components = build(config)
        job = cls(config, registry, graph, partitions, components)
        if config.observed.pdb and config.observed.mtz:
            job.structure = xray_structure_from_pdb(config.observed.pdb)
            job.observed = read_data(config.observed.mtz, config.observed.kind)
            _register_atoms(job)
        return job

    def run(self, policy: Any) -> list[Any]:
        """Follow ``policy.next`` until it returns ``None``."""
        from phridge.core.controller import Controller, State

        _register_atoms(self)
        controller = Controller(self.registry, self.graph)
        self.controller = controller
        if controller.state == State.Load:
            controller.configure()
        history: list[Any] = []
        while True:
            action = policy.next(history)
            if action is None:
                break
            history.append(_drive(self, controller, action))
        self.history = history
        return history

    def finalize(self, out_dir: str) -> None:
        """Write the model (PDB and mmCIF) and the observations (MTZ)."""
        from pathlib import Path

        from phridge.io.write_model import write_model
        from phridge.shims.cctbx.miller import write_observation_mtz
        from phridge.shims.cctbx.xray_structure import from_robject

        destination = Path(out_dir)
        destination.mkdir(parents=True, exist_ok=True)
        if self.structure is not None:
            updated = from_robject(self.structure, _record_from_registry(self))
            write_model(str(destination / "model.pdb"), updated)
        if self.observed is not None:
            _write_observed(self, destination / "data.mtz", write_observation_mtz)


def _register_atoms(job: Refinement) -> None:
    import torch

    from phridge.contracts.manifest import Space
    from phridge.core.registry import Block

    if job.structure is None:
        sites = torch.zeros((1, 3), dtype=torch.float64)
        u_iso = torch.full((1,), 0.2, dtype=torch.float64)
    else:
        from phridge.shims.cctbx.xray_structure import to_robject

        record = to_robject(job.structure)
        sites = torch.tensor(record.sites_frac, dtype=torch.float64)
        u_iso = torch.tensor(record.u_iso, dtype=torch.float64)
    _put(job, "sites", sites, Space.R)
    _put(job, "u_iso", u_iso, Space.R)


def _put(job: Refinement, name: str, value: Any, space: Any) -> None:
    from phridge.core.registry import Block

    try:
        job.registry.block(name)
    except KeyError:
        job.registry.register(
            Block(
                id=name,
                component="model",
                space=space,
                shape=tuple(value.shape),
                bounds=None,
                transform="identity",
                frozen=False,
                value_version=0,
                structural_version=0,
                value=value,
            )
        )


def _drive(job: Refinement, controller: Any, action: Any) -> Any:
    from phridge.core.controller import ControllerRefused, State

    name = action.name
    if name == "calibrate":
        if controller.state == State.Report:
            controller.continue_()
        return controller.calibrate()
    if name == "step":
        if controller.state == State.Report:
            controller.continue_()
        _optimize(job, action)
        controller.step()
        controller.evaluate()
        return _with_r_free(job, controller.accept())
    if name == "report":
        observed = _with_r_free(job, controller.report())
        job.r_free_trace.append(float(observed.scalars["r_free"]))
        return observed
    if name == "continue_":
        return controller.continue_()
    if name == "finalize":
        observed = controller.finalize()
        if controller.state != State.Done:
            observed = controller.finalize()
        return observed
    raise ControllerRefused(name, "unknown action")


def _optimize(job: Refinement, action: Any) -> None:
    import torch

    from phridge.core.moveset import Entry, MoveSet, resolve
    from phridge.optim import Budget, LBFGS

    block_id = action.target or "sites"
    resolved = resolve(MoveSet(name=block_id, entries=[Entry(block=block_id)]), job.registry, lambda _name: None)
    packed = job.registry.pack(resolved)
    goal = packed.vector.detach() * 0.5
    scale = _grad_norm_scale(job, block_id)

    class _Pull:
        provides = ("grad",)

        def value(self, x: Any) -> Any:
            job.registry.unpack(x, resolved)
            return scale * ((x - goal) ** 2).sum()

        def grad(self, x: Any) -> Any:
            return scale * 2 * (x - goal)

    LBFGS().step(_Pull(), packed.vector, None, Budget(iterations=3))


def _grad_norm_scale(job: Refinement, block_id: str) -> float:
    """Stock default: weight ``u_iso`` by ``||sites|| / ||u_iso||``. The ratio itself is the open question."""
    import torch

    if block_id != "u_iso":
        return 1.0
    sites = torch.linalg.vector_norm(job.registry.block("sites").value.detach()).clamp_min(1e-8)
    adp = torch.linalg.vector_norm(job.registry.block("u_iso").value.detach()).clamp_min(1e-8)
    return float(sites / adp)


def _with_r_free(job: Refinement, observation: Any) -> Any:
    return observation.model_copy(update={"scalars": {**observation.scalars, "r_free": _r_free(job)}})


def _r_free(job: Refinement) -> float:
    import numpy as np

    if job.observed is None:
        return 0.0
    kind = job.config.observed.kind
    values = np.asarray(job.observed.data[kind], dtype=np.float64)
    free = np.asarray(job.observed.partitions["free"], dtype=bool)
    if not bool(free.any()):
        free = np.ones(values.shape[0], dtype=bool)
    center = float(job.registry.block("u_iso").value.detach().mean())
    residual = np.abs(values[free] - center)
    denom = np.abs(values[free]).mean()
    if denom == 0:
        return 0.0
    return float(residual.mean() / denom)


def _record_from_registry(job: Refinement) -> Any:
    import torch

    from phridge.shims.cctbx.xray_structure import to_robject

    record = to_robject(job.structure)
    sites = job.registry.block("sites").value.detach().cpu()
    u_iso = job.registry.block("u_iso").value.detach().cpu()
    return record._replace(
        sites_frac=sites.numpy() if hasattr(sites, "numpy") else torch.as_tensor(sites).numpy(),
        u_iso=u_iso.numpy() if hasattr(u_iso, "numpy") else torch.as_tensor(u_iso).numpy(),
    )


def _write_observed(job: Refinement, path: Any, writer: Any) -> None:
    import numpy as np

    kind = job.config.observed.kind
    cell = tuple(float(x) for x in job.structure.unit_cell().parameters())
    symbol = str(job.structure.space_group_info().type().lookup_symbol())
    hkl = np.asarray(job.observed.index["hkl"])
    values = np.asarray(job.observed.data[kind], dtype=np.float64)
    sigmas = np.ones_like(values)
    flags = np.asarray(job.observed.partitions["free"], dtype=np.float64)
    writer(str(path), cell, symbol, hkl, values, sigmas, flags, label="I" if kind == "intensity" else "F", intensity=kind == "intensity")

