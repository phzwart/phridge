"""L-BFGS-B. The packed vector is unconstrained; logit and log live in the registry."""

from __future__ import annotations

from typing import Any

from phridge.optim.base import Budget, StepResult, as_float, require_capabilities


class LBFGS:
    """``torch.optim.LBFGS`` on the packed vector. ``requires`` is gradient only."""

    requires = ("grad",)

    def step(self, objective: Any, x0: Any, bounds: Any, budget: Budget) -> StepResult:
        """Take up to ``budget.iterations`` L-BFGS iterations. ``bounds`` is unused."""
        import torch

        _ = bounds
        require_capabilities(self, objective)
        limit = budget.iterations if budget.iterations is not None else 8
        vector = x0.detach().clone().requires_grad_(True)
        opt = torch.optim.LBFGS([vector], max_iter=1, history_size=8, line_search_fn="strong_wolfe")
        trace: list[float] = []
        evals = 0

        def closure() -> Any:
            nonlocal evals
            opt.zero_grad()
            loss = objective.value(vector.detach())
            grad = objective.grad(vector.detach())
            vector.grad = grad.detach().reshape_as(vector)
            evals += 1
            trace.append(as_float(loss))
            return loss.detach().reshape(())

        for _ in range(limit):
            opt.step(closure)
        return StepResult(vector.detach(), tuple(trace), evals)
