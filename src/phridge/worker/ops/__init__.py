"""Worker implementations keyed by op name from phridge.ops."""

from __future__ import annotations

from typing import Any, Callable

from phridge.worker.ops.geometry_ops import geometry_minimize
from phridge.worker.ops.scale_array import scale_array
from phridge.worker.ops.xtal_ops import gauss_newton_hvp, refine_gradients, sf_calc, sf_gradients, target_eval

IMPLEMENTATIONS: dict[str, Callable[..., Any]] = {
    "scale_array": scale_array,
    "sf_calc": sf_calc,
    "sf_gradients": sf_gradients,
    "target_eval": target_eval,
    "refine_gradients": refine_gradients,
    "gauss_newton_hvp": gauss_newton_hvp,
    "geometry_minimize": geometry_minimize,
}
