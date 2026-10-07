"""cctbx ADP conversions. The rest of phridge calls this, not adptbx."""

from __future__ import annotations

from typing import Any


def grad_u_star_as_u_cart(unit_cell: Any, u_star: Any) -> Any:
    from cctbx import adptbx

    return adptbx.grad_u_star_as_u_cart(unit_cell, u_star)
