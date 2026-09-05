"""External phridge plugin: agentsg Niggli cell reduction.

Not part of the phridge package. Install agentsg, then either::

    import plugins.agentsg_niggli  # with PYTHONPATH=examples

or::

    phridge-worker --preload plugins.agentsg_niggli

Install::

    pip install "git+https://github.com/phzwart/agentsg.git#subdirectory=agentsg"
    # or: pip install -e ".[agentsg]" from the phridge repo

Phenix-facing wrapper (cctbx in / cctbx out)::

    from plugins.agentsg_niggli import niggli_cell
    from phridge.client import Bridge

    bridge = Bridge(memory=True)  # or redis URL + worker --preload
    uc_red = niggli_cell(unit_cell, bridge=bridge)
    uc_red, cb_op = niggli_cell(unit_cell, bridge=bridge, return_change_of_basis=True)
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Union

from phridge.client.api import Bridge
from phridge.ops import register_op

CellLike = Union[Mapping[str, Any], Sequence[Any]]

_KEYS = ("a", "b", "c", "alpha", "beta", "gamma")
_OP = "agentsg_niggli_reduce"


def _as_params(cell: CellLike) -> tuple[float, float, float, float, float, float]:
    if isinstance(cell, Mapping):
        missing = [k for k in _KEYS if k not in cell]
        if missing:
            raise ValueError(f"cell mapping missing keys: {missing}")
        return tuple(float(cell[k]) for k in _KEYS)  # type: ignore[return-value]
    if len(cell) != 6:
        raise ValueError("cell sequence must be (a, b, c, alpha, beta, gamma)")
    return tuple(float(x) for x in cell)  # type: ignore[return-value]


def _params_from_cctbx(unit_cell: Any) -> tuple[float, float, float, float, float, float]:
    """Accept ``uctbx.unit_cell``, ``crystal.symmetry``, or a 6-sequence."""
    if hasattr(unit_cell, "unit_cell") and callable(unit_cell.unit_cell):
        # cctbx.crystal.symmetry
        unit_cell = unit_cell.unit_cell()
    if hasattr(unit_cell, "parameters") and callable(unit_cell.parameters):
        return tuple(float(x) for x in unit_cell.parameters())  # type: ignore[return-value]
    return _as_params(unit_cell)


def _cell_dict(params: Sequence[float]) -> dict[str, float]:
    return {k: float(v) for k, v in zip(_KEYS, params)}


def agentsg_niggli_reduce(cell: CellLike, params: Optional[Any] = None) -> dict[str, Any]:
    """Worker op: Niggli-reduce a unit cell via agentsg.

    ``params`` is reserved for future knobs (ignored for now).
    """
    try:
        from agentsg.cell import niggli_reduce
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "agentsg is required for agentsg_niggli_reduce. Install with:\n"
            '  pip install "git+https://github.com/phzwart/agentsg.git#subdirectory=agentsg"\n'
            "  # or: pip install -e \".[agentsg]\" from the phridge repo"
        ) from exc

    a, b, c, alpha, beta, gamma = _as_params(cell)
    reduced, cob = niggli_reduce(a, b, c, alpha, beta, gamma)
    return {
        "unit_cell": [float(x) for x in reduced],
        "change_of_basis": [[int(v) for v in row] for row in cob],
        "source": "agentsg",
        "input_cell": [a, b, c, alpha, beta, gamma],
    }


def _change_of_basis_op(matrix_3x3: Sequence[Sequence[int]]) -> Any:
    from cctbx import sgtbx

    flat = [int(matrix_3x3[i][j]) for i in range(3) for j in range(3)]
    return sgtbx.change_of_basis_op(sgtbx.rt_mx(sgtbx.rot_mx(flat, 1)))


def niggli_cell(
    unit_cell: Any,
    *,
    bridge: Bridge,
    return_change_of_basis: bool = False,
    timeout: Optional[float] = None,
) -> Any:
    """Reduce a cctbx unit cell through phridge → agentsg (Phenix-facing).

    Mirrors ``uctbx.unit_cell.niggli_cell()``: pass a ``uctbx.unit_cell``
    (or ``crystal.symmetry`` / 6-tuple) and get a ``uctbx.unit_cell`` back.
    The worker never sees cctbx — this wrapper packs JSON and unpacks.

    Parameters
    ----------
    unit_cell:
        ``cctbx.uctbx.unit_cell``, ``cctbx.crystal.symmetry``, or
        ``(a, b, c, alpha, beta, gamma)``.
    bridge:
        phridge ``Bridge`` (memory or Redis). The op must be registered
        (import this module; worker needs ``--preload plugins.agentsg_niggli``).
    return_change_of_basis:
        If true, return ``(unit_cell, sgtbx.change_of_basis_op)``.
    """
    from cctbx import uctbx

    params = _params_from_cctbx(unit_cell)
    raw = bridge.call(_OP, cell=_cell_dict(params), timeout=timeout)
    if not isinstance(raw, dict) or "unit_cell" not in raw:
        raise TypeError(f"unexpected op result: {type(raw)!r}")
    reduced = uctbx.unit_cell(tuple(float(x) for x in raw["unit_cell"]))
    if not return_change_of_basis:
        return reduced
    return reduced, _change_of_basis_op(raw["change_of_basis"])


def register() -> None:
    register_op(
        _OP,
        agentsg_niggli_reduce,
        inputs={"cell": "json", "params": "json"},
        outputs={"result": "json"},
    )


register()
