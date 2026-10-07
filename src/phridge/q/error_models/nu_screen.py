"""ν screen: one σ_A/β refit per grid node, then the Gaussian node."""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Sequence


def parse_nu_grid(raw: Any, default: str = "5:50:5") -> list[float]:
    """ν values for ``nu_mode=grid``. ``lo:hi:step`` or an explicit list."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        raw = default
    if isinstance(raw, (list, tuple)):
        out = [float(x) for x in raw]
        return [v for v in out if v == v and v > 0.0]
    text = str(raw).strip()
    if ":" in text:
        parts = [p.strip() for p in text.split(":")]
        lo = float(parts[0])
        hi = float(parts[1]) if len(parts) > 1 else lo
        step = float(parts[2]) if len(parts) > 2 else 5.0
        if step <= 0.0:
            step = 5.0
        n = int(math.floor((hi - lo) / step + 1e-9)) + 1
        return [float(lo + i * step) for i in range(max(n, 1)) if lo + i * step <= hi + 0.5 * step]
    out = [float(x) for x in text.replace(",", " ").split()]
    return [v for v in out if v == v and v > 0.0]


class NuScreen:
    """Grid loop plus the ν=200 Gaussian node. The selected state is the best grid node.

    The Gaussian refit is run and then discarded, matching the legacy ``nu_mode='grid'``
    loop: that node is recorded, and the parameters restored are the lowest grid NLL.
    """

    def __init__(self) -> None:
        self.nu_shell: Optional[float] = None
        self.best: Any = None

    def forward(self, pred: Any) -> Any:
        return pred

    def adjoint(self, cotangent: Any) -> Any:
        return cotangent

    def fit(self, obs: Any, mask: Any) -> Any:
        _ = (obs, mask)
        raise RuntimeError("NuScreen.fit is the screen(); pass a refit callable")

    def screen(
        self,
        grid: Sequence[float],
        refit: Callable[[float], Any],
        criterion: Callable[[Any], float],
    ) -> Any:
        """Refit at each grid node, then at ν=200, and keep the lowest grid NLL."""
        nodes = [float(v) for v in grid] or parse_nu_grid(None)
        best: Any = None
        best_score = float("inf")
        best_nu = float(nodes[0])
        for nu in nodes:
            state = refit(float(nu))
            score = float(criterion(state))
            if score < best_score:
                best_score = score
                best = state
                best_nu = float(nu)
        refit(200.0)
        self.nu_shell = best_nu
        self.best = best
        return best
