"""Q-component protocol used by the conformance suite."""

from __future__ import annotations

from typing import Any

from phridge.contracts.tensors import Observed, Predictive


class FieldScale:
    """Linear scale. Adjoint is the same gain, so shapes match the input."""

    def __init__(self, gain: float, produces: list[str]) -> None:
        self.gain = float(gain)
        self.produces = list(produces)

    def forward(self, pred: Predictive) -> Predictive:
        location = pred.fields["location"]
        return Predictive({name: location * self.gain for name in self.produces})

    def adjoint(self, cotangent: Predictive) -> Predictive:
        total = None
        for name in self.produces:
            term = cotangent.fields[name] * self.gain
            total = term if total is None else total + term
        return Predictive({"location": total})


class QComponent:
    """One predictive block that can be fitted, applied, and differentiated."""

    def forward(self, pred: Predictive) -> Predictive:
        raise NotImplementedError

    def adjoint(self, cotangent: Predictive) -> Predictive:
        raise NotImplementedError

    def fit(self, obs: Observed, mask: Any) -> Predictive:
        raise NotImplementedError
