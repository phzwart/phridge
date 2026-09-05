"""External phridge plugin: agentsg Niggli cell reduction.

Not part of the phridge package. Install agentsg, then either::

    import plugins.agentsg_niggli  # with PYTHONPATH=examples

or::

    phridge-worker --preload plugins.agentsg_niggli

Install::

    pip install "git+https://github.com/phzwart/agentsg.git#subdirectory=agentsg"
    # or: pip install -e ".[agentsg]" from the phridge repo
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Union

from phridge.ops import register_op

CellLike = Union[Mapping[str, Any], Sequence[Any]]

_KEYS = ("a", "b", "c", "alpha", "beta", "gamma")


def _as_params(cell: CellLike) -> tuple[float, float, float, float, float, float]:
    if isinstance(cell, Mapping):
        missing = [k for k in _KEYS if k not in cell]
        if missing:
            raise ValueError(f"cell mapping missing keys: {missing}")
        return tuple(float(cell[k]) for k in _KEYS)  # type: ignore[return-value]
    if len(cell) != 6:
        raise ValueError("cell sequence must be (a, b, c, alpha, beta, gamma)")
    return tuple(float(x) for x in cell)  # type: ignore[return-value]


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


def register() -> None:
    register_op(
        "agentsg_niggli_reduce",
        agentsg_niggli_reduce,
        inputs={"cell": "json", "params": "json"},
        outputs={"result": "json"},
    )


register()
