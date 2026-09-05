"""Worker op: torch geometry-restraint minimization (LBFGS / Adam / AdamW / SGD)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from phridge.packing_geometry import PackedRestraints
from phridge.packing_xtal import PackedCartesian
from phridge.sfcalc import ops as sfcalc_ops
from phridge.worker.geometry.energy import energy_and_sites

_DEVICE = sfcalc_ops._DEVICE


def geometry_minimize(
    sites: PackedCartesian,
    restraints: PackedRestraints,
    params: Optional[Any] = None,
) -> dict[str, Any]:
    """Minimize cartesian sites under packed bond/angle/dihedral restraints.

    ``params`` JSON:
      - ``max_iterations`` (default 100; 0 = energy only)
      - ``optimizer``: ``"lbfgs"`` | ``"adam"`` | ``"adamw"`` | ``"sgd"``
      - ``lr`` / ``lr_min``: peak and floor learning rates
      - ``schedule``: ``"none"`` | ``"cosine"`` | ``"triangular"``
      - ``momentum``: SGD momentum (default 0)
    """
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise TypeError("params must be a dict")
    max_iterations = int(params.get("max_iterations", 100))
    optimizer = str(params.get("optimizer", "lbfgs"))
    lr = params.get("lr", None)
    if lr is not None:
        lr = float(lr)
    lr_min = params.get("lr_min", None)
    if lr_min is not None:
        lr_min = float(lr_min)
    schedule = params.get("schedule", None)
    momentum = float(params.get("momentum", 0.0))
    xyz = sites.xyz
    if hasattr(xyz, "detach"):
        xyz = xyz.detach().cpu().numpy()
    xyz = np.asarray(xyz, dtype=np.float64)
    out_xyz, target = energy_and_sites(
        xyz,
        restraints,
        max_iterations=max_iterations,
        optimizer=optimizer,
        lr=lr,
        lr_min=lr_min,
        schedule=schedule,
        momentum=momentum,
        device=_DEVICE["device"],
    )
    return {
        "sites": PackedCartesian(out_xyz, crystal=sites.meta.crystal),
        "target": {
            "before": float(target["before"]),
            "after": float(target["after"]),
            "n_iter": int(target["n_calls"]),
            "n_steps": int(target["n_steps"]),
            "n_calls": int(target["n_calls"]),
            "optimizer": str(target["optimizer"]),
            "schedule": str(target.get("schedule") or "none"),
            "momentum": float(target.get("momentum") or 0.0),
            "lr": target.get("lr"),
            "lr_min": target.get("lr_min"),
            "rss_before_mb": float(target.get("rss_before_mb") or 0.0),
            "rss_after_mb": float(target.get("rss_after_mb") or 0.0),
            "rss_delta_mb": float(target.get("rss_delta_mb") or 0.0),
            "optimizer_state_mb": float(target.get("optimizer_state_mb") or 0.0),
            "cuda_peak_mb": float(target.get("cuda_peak_mb") or 0.0),
        },
    }


geometry_minimize.compute_dtype = "float64"
