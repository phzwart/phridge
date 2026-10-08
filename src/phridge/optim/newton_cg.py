"""Newton-CG. The preconditioner is the torch form of the Jacobi CG in the structure-factor client."""

from __future__ import annotations

from typing import Any, Callable

from phridge.optim.base import Budget, StepResult, require_capabilities
from phridge.optim._search import backtrack


class NewtonCG:
    """Solve ``H p = -g`` with preconditioned CG, then backtrack. Needs grad, hvp, and a preconditioner."""

    requires = ("grad", "hvp", "preconditioner")

    def step(self, objective: Any, x0: Any, bounds: Any, budget: Budget) -> StepResult:
        """One Newton step. ``bounds`` stay in the registry transforms."""
        _ = bounds
        require_capabilities(self, objective)
        limit = budget.iterations if budget.iterations is not None else 20
        vector = x0.detach().clone()
        value0 = objective.value(vector)
        grad = objective.grad(vector).detach().reshape(-1)
        diag = objective.preconditioner().detach().reshape(-1).abs().clamp_min(1e-8)

        def hvp(direction: Any) -> Any:
            return objective.hvp(vector, direction.reshape_as(vector)).detach().reshape(-1)

        direction = _preconditioned_cg(hvp, -grad, diag, max_iter=limit).reshape_as(vector)
        updated, _ = backtrack(objective, vector, direction, value0)
        return StepResult(updated.detach(), (float(value0.detach()),), 1 + 8)


def _preconditioned_cg(hvp: Callable[[Any], Any], b: Any, diag: Any, max_iter: int = 20, tol: float = 1e-6) -> Any:
    """Solve ``H p = b`` with Jacobi preconditioning. Same recurrence as ``sfcalc.client._preconditioned_cg``."""
    import torch

    x = torch.zeros_like(b)
    residual = b.clone()
    z = residual / diag
    direction = z.clone()
    rz = torch.dot(residual, z)
    scale = torch.linalg.vector_norm(b).clamp_min(1e-300)
    for _ in range(max_iter):
        hess_dir = hvp(direction)
        denom = torch.dot(direction, hess_dir)
        if abs(float(denom)) < 1e-300:
            break
        alpha = rz / denom
        x = x + alpha * direction
        residual = residual - alpha * hess_dir
        if float(torch.linalg.vector_norm(residual) / scale) < tol:
            break
        z = residual / diag
        rz_new = torch.dot(residual, z)
        direction = z + (rz_new / rz) * direction
        rz = rz_new
    return x
