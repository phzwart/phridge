"""Worker op: torch geometry-restraint minimization (LBFGS / Adam / SGD)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from phridge.packing_geometry import PackedRestraints
from phridge.packing_xtal import PackedCartesian
from phridge.worker.geometry.energy import energy_and_sites
from phridge.worker.ops import xtal_ops

_DEVICE = xtal_ops._DEVICE


def geometry_minimize(
    sites: PackedCartesian,
    restraints: PackedRestraints,
    params: Optional[Any] = None,
) -> dict[str, Any]:
    """Minimize cartesian sites under packed bond/angle/dihedral restraints.

    ``params`` JSON:
      - ``max_iterations`` (default 100; 0 = energy only)
      - ``optimizer``: ``"lbfgs"`` | ``"adam"`` | ``"sgd"`` (default ``lbfgs``)
      - ``lr``: optional learning rate for Adam/SGD (and LBFGS step size)
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
        device=_DEVICE["device"],
    )
    return {
        "sites": PackedCartesian(out_xyz, crystal=sites.meta.crystal),
        "target": {
            "before": float(target["before"]),
            "after": float(target["after"]),
            "n_iter": int(target["n_iter"]),
            "optimizer": str(target["optimizer"]),
        },
    }


geometry_minimize.compute_dtype = "float64"
