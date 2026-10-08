"""Policy protocol. A policy sequences components; it does not own their formulas."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict

from phridge.contracts.tensors import Observed


class Action(BaseModel):
    """One controller action a policy may request."""

    model_config = ConfigDict(extra="forbid")

    name: str
    target: str = ""
    optimizer: str = ""


class Policy:
    """Sequence component fits. ``next`` is the controller protocol; ``run`` remains for calibration."""

    def run(self, components: Mapping[str, Any], obs: Observed, mask: Any) -> Any:
        raise NotImplementedError

    def next(self, obs_history: Sequence[Any]) -> Optional[Action]:
        """Return the next action, or ``None`` when the policy is finished."""
        _ = obs_history
        return None
