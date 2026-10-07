"""Torch-free client: common-set preparation on top of ``multixtal.client``."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from phridge.tools.ensemble.holdout import assign_roles
from phridge.tools.ensemble.options import EnsembleOptions
from phridge.redis_store import DEFAULT_MAX_OBJECT_BYTES


@dataclass
class EnsembleJob:
    """Arrays for ``ensemble_fit``, plus the multixtal job they were built from."""

    hkl: np.ndarray
    centric: np.ndarray
    epsilon: np.ndarray
    cells: np.ndarray
    i_obs: np.ndarray
    sig_i: np.ndarray
    mask: np.ndarray
    options: EnsembleOptions
    xray: Any = None
    table: Any = None
    params: Any = None
    sites_frac: Any = None
    f_mask: Optional[np.ndarray] = None
    f_var: Optional[np.ndarray] = None
    f_phase: Optional[np.ndarray] = None
    hl: Optional[np.ndarray] = None
    roles: Optional[np.ndarray] = None
    multixtal: Any = None
    logs: list[str] = field(default_factory=list)


def _merge_intensity(job: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    plus = np.asarray(job.i_plus, dtype=np.float64)
    minus = np.asarray(job.i_minus, dtype=np.float64)
    sig_p = np.asarray(job.sig_plus, dtype=np.float64)
    sig_m = np.asarray(job.sig_minus, dtype=np.float64)
    mask_p = np.asarray(job.mask_plus, dtype=bool)
    mask_m = np.asarray(job.mask_minus, dtype=bool)
    both = mask_p & mask_m
    i_obs = np.zeros_like(plus)
    sig = np.ones_like(plus)
    mask = mask_p | mask_m
    i_obs[mask_p] = plus[mask_p]
    sig[mask_p] = sig_p[mask_p]
    only_m = mask_m & ~mask_p
    i_obs[only_m] = minus[only_m]
    sig[only_m] = sig_m[only_m]
    i_obs[both] = 0.5 * (plus[both] + minus[both])
    sig[both] = 0.5 * np.hypot(sig_p[both], sig_m[both])
    return i_obs, sig, mask


def _f_calc_pdb(path: str, hkl: np.ndarray, cell: Sequence[float], space_group: Optional[str]) -> np.ndarray:
    from phridge.shims.cctbx.pdb import direct_f_calc_pdb

    return direct_f_calc_pdb(path, hkl, cell, space_group)


def prepare_ensemble_job(
    model_path: str | Path,
    data_paths: Sequence[str | Path],
    options: EnsembleOptions,
    manifest: Any = None,
    *,
    ligand_paths: Optional[Sequence[Optional[str]]] = None,
    phase_model_paths: Optional[Sequence[Optional[str]]] = None,
    free_flags: Optional[np.ndarray] = None,
) -> EnsembleJob:
    """Reindex, build the common ASU set and ``F_mask``, then attach ensemble arrays."""
    from phridge.legacy.multixtal.client import prepare_job
    from phridge.legacy.multixtal.options import MultixtalOptions

    mx = MultixtalOptions(
        d_min=options.d_min,
        labels=options.labels,
        placement=options.placement,
        space_group=options.space_group,
        cell=options.cell,
        anomalous=False,
    )
    job = prepare_job(str(model_path), list(data_paths), mx, manifest=manifest)
    i_obs, sig_i, mask = _merge_intensity(job)
    roles = assign_roles(
        mask,
        audit_fraction=options.roles.audit_fraction,
        calibration_fraction=options.roles.calibration_fraction,
        seed=options.roles.seed,
        free_flags=free_flags,
    )
    n_data, n_h = i_obs.shape
    f_var = np.zeros((n_data, n_h), dtype=np.complex128)
    f_phase = None
    logs = list(job.logs)
    if ligand_paths:
        for d, path in enumerate(ligand_paths):
            if path:
                f_var[d] = _f_calc_pdb(path, job.hkl, job.cells[d], options.space_group)
                logs.append(f"ligand model {path} -> F_var[{d}]")
    if phase_model_paths:
        f_phase = np.zeros((n_data, n_h), dtype=np.complex128)
        for d, path in enumerate(phase_model_paths):
            if path:
                f_phase[d] = _f_calc_pdb(path, job.hkl, job.cells[d], options.space_group)
                logs.append(f"phase model {path} -> f_phase[{d}]")
    return EnsembleJob(
        hkl=np.asarray(job.hkl, dtype=np.int32),
        centric=np.asarray(job.centric, dtype=bool),
        epsilon=np.asarray(job.epsilon, dtype=np.float64),
        cells=np.asarray(job.cells, dtype=np.float64),
        i_obs=i_obs,
        sig_i=sig_i,
        mask=mask,
        options=options,
        xray=job.xray,
        table=job.table,
        params=job.params,
        sites_frac=job.sites_frac,
        f_mask=np.asarray(job.f_mask, dtype=np.complex128),
        f_var=f_var,
        f_phase=f_phase,
        roles=roles,
        multixtal=job,
        logs=logs,
    )


def payload_nbytes(kwargs: dict[str, Any]) -> int:
    total = 0
    for value in kwargs.values():
        if isinstance(value, np.ndarray):
            total += int(value.nbytes)
    return total


def chunk_dataset_indices(n_data: int, row_nbytes: int, max_bytes: int = DEFAULT_MAX_OBJECT_BYTES) -> list[tuple[int, int]]:
    """Slices along the dataset axis so each slice stays under ``max_bytes``.

    The joint mode fit cannot be split this way. ``ensemble_scores`` and
    ``ensemble_maps`` can. A joint fit that would exceed the cap raises in
    :func:`job_to_op_kwargs`.
    """
    if row_nbytes <= 0 or n_data <= 0:
        return [(0, n_data)]
    rows = max(1, int(max_bytes) // int(row_nbytes))
    return [(start, min(n_data, start + rows)) for start in range(0, n_data, rows)]


def job_to_op_kwargs(job: EnsembleJob, *, max_bytes: int = DEFAULT_MAX_OBJECT_BYTES) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "hkl": job.hkl,
        "centric": job.centric,
        "epsilon": job.epsilon,
        "cells": job.cells,
        "i_obs": job.i_obs,
        "sig_i": job.sig_i,
        "mask": job.mask,
        "options": job.options.model_dump(),
    }
    if job.xray is not None:
        payload["xray"] = job.xray
        payload["table"] = job.table
        payload["params"] = job.params
        payload["sites_frac"] = job.sites_frac
    if job.f_mask is not None:
        payload["f_mask"] = job.f_mask
    if job.f_var is not None:
        payload["f_var"] = job.f_var
    if job.f_phase is not None:
        payload["f_phase"] = job.f_phase
    if job.hl is not None:
        payload["hl"] = job.hl
    if job.roles is not None:
        payload["roles"] = job.roles
    if payload_nbytes(payload) > int(max_bytes):
        raise ValueError(
            "ensemble_fit payload exceeds "
            f"{int(max_bytes)} bytes. The shared-mode fit is one job; "
            "score datasets afterwards with ensemble_scores, which chunks by dataset."
        )
    return payload


class EnsembleFitResult:
    """Thin view of an ``ensemble_fit`` result dictionary."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self.raw = raw

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]


class RemoteEnsembleFit:
    """Run ``ensemble_fit`` through a Bridge. The client module does not import torch."""

    def __init__(self, bridge: Any = None) -> None:
        self.bridge = bridge

    def fit(self, job: EnsembleJob) -> EnsembleFitResult:
        from phridge.api import Bridge
        from phridge.tools.ensemble.ops import OP_NAME_FIT, register_ops

        register_ops()
        bridge = self.bridge if self.bridge is not None else Bridge(memory=True)
        raw = bridge.call(OP_NAME_FIT, **job_to_op_kwargs(job))
        return EnsembleFitResult(raw)
