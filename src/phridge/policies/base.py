"""Policy protocol. A policy sequences components; it does not own their formulas."""

from __future__ import annotations

from typing import Any, Mapping

from phridge.contracts.tensors import Observed


class Policy:
    """Run a fixed sequence of component fits."""

    def run(self, components: Mapping[str, Any], obs: Observed, mask: Any) -> Any:
        raise NotImplementedError
