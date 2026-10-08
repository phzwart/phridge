"""Stock macrocycle.

N cycles of: calibrate scales, L-BFGS on ``sites``, L-BFGS on ``u_iso``, report.
This is the stock default. Step weights are the ratio of the two block gradient
norms (an objective option: ``weight_by_grad_norm``). Whether that ratio should
stay the default is the open design question in the spec.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from phridge.policies.base import Action, Policy


class StockMacrocycle(Policy):
    """The default refinement loop. ``weight_by_grad_norm`` scales the ``u_iso`` step."""

    def __init__(self, cycles: int = 2, weight_by_grad_norm: bool = True) -> None:
        self.cycles = cycles
        self.weight_by_grad_norm = weight_by_grad_norm
        self._script = _script(cycles)

    def next(self, obs_history: Sequence[Any]) -> Optional[Action]:
        index = len(obs_history)
        if index >= len(self._script):
            return None
        return self._script[index]


def _script(cycles: int) -> list[Action]:
    actions: list[Action] = []
    for index in range(cycles):
        actions.append(Action(name="calibrate"))
        actions.append(Action(name="step", target="sites", optimizer="lbfgs"))
        actions.append(Action(name="step", target="u_iso", optimizer="lbfgs"))
        actions.append(Action(name="report"))
        if index + 1 < cycles:
            actions.append(Action(name="continue_"))
    actions.append(Action(name="finalize"))
    return actions
