"""Worker implementations keyed by op name from phridge.ops.

Built-ins live in ``IMPLEMENTATIONS``. Third-party packages should prefer
``phridge.ops.register_op`` (see ``docs/extending.md``) rather than mutating
this dict.
"""

from __future__ import annotations

from typing import Any, Callable

from phridge.sfcalc.ops import (
    gauss_newton_blocks,
    gauss_newton_diagonal,
    gauss_newton_hvp,
    refine_gradients,
    sf_calc,
    sf_gradients,
    target_eval,
)
from phridge.worker.ops.geometry_ops import (
    adp_prior_eval,
    adp_prior_hvp,
    geometry_curvature,
    geometry_gn_solve,
    geometry_hvp,
    geometry_minimize,
)
from phridge.worker.ops.scale_array import scale_array

IMPLEMENTATIONS: dict[str, Callable[..., Any]] = {
    "scale_array": scale_array,
    "sf_calc": sf_calc,
    "sf_gradients": sf_gradients,
    "target_eval": target_eval,
    "refine_gradients": refine_gradients,
    "gauss_newton_hvp": gauss_newton_hvp,
    "gauss_newton_diagonal": gauss_newton_diagonal,
    "gauss_newton_blocks": gauss_newton_blocks,
    "geometry_minimize": geometry_minimize,
    "geometry_curvature": geometry_curvature,
    "geometry_hvp": geometry_hvp,
    "geometry_gn_solve": geometry_gn_solve,
    "adp_prior_eval": adp_prior_eval,
    "adp_prior_hvp": adp_prior_hvp,
}
