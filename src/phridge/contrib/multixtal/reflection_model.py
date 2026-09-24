"""Phase-1 ``ReflectionModel``: Rice-moment Gaussian quasi-likelihood.

``ml_i_nuisance_fit`` is one dataset and the integrated intensity NLL. This module
implements the plug-in Gaussian quasi-likelihood specified for phase 1. Phase 2
swaps the body for ``mli.log_likelihood_*`` via ``rice_inputs``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class RiceMomentQuasiLikelihood:
    """Gaussian quasi-likelihood with Rice/Woolfson intensity moments.

    ``E[I] = A ε (α² |Fc|² + β)``
    acentric ``Var = A² ε² (β² + 2 α² |Fc|² β) + σ²``
    centric  ``Var = 2 A² ε² (β² + 2 α² |Fc|² β) + σ²``
    """

    def moments(
        self,
        i_bar: Any,
        sig2: Any,
        f_c_abs2: Any,
        epsilon: Any,
        centric: Any,
        scale_a: Any,
        alpha: Any,
        beta: Any,
        a_sys: Any = None,
    ) -> tuple[Any, Any]:
        scale = scale_a if a_sys is None else scale_a * a_sys
        mean, rice_var = _rice_moments(f_c_abs2, epsilon, scale, alpha, beta)
        if hasattr(centric, "to"):
            import torch

            two = torch.where(centric.to(dtype=torch.bool), 2.0, 1.0)
        else:
            two = np.where(np.asarray(centric, dtype=bool), 2.0, 1.0)
        var = two * rice_var + sig2
        return mean, var

    def nll(
        self,
        i_bar: Any,
        sig2: Any,
        f_c_abs2: Any,
        epsilon: Any,
        centric: Any,
        scale_a: Any,
        alpha: Any,
        beta: Any,
        a_sys: Any = None,
    ) -> Any:
        mean, var = self.moments(i_bar, sig2, f_c_abs2, epsilon, centric, scale_a, alpha, beta, a_sys=a_sys)
        if hasattr(var, "clamp"):
            var = var.clamp(min=1e-12)
            return 0.5 * ((i_bar - mean) ** 2 / var + var.log())
        var = np.maximum(np.asarray(var, dtype=np.float64), 1e-12)
        resid = np.asarray(i_bar, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
        return 0.5 * (resid**2 / var + np.log(var))


def _rice_moments(f_c_abs2: Any, epsilon: Any, scale_a: Any, alpha: Any, beta: Any) -> tuple[Any, Any]:
    """Rice first moment and the Rice part of the variance (no measurement, no centric 2)."""
    a2 = alpha * alpha
    mean_unit = a2 * f_c_abs2 + beta
    mean = scale_a * epsilon * mean_unit
    rice_unit = beta * beta + 2.0 * a2 * f_c_abs2 * beta
    rice_var = (scale_a * epsilon) ** 2 * rice_unit
    return mean, rice_var


def sigma_a_from_alpha_beta(alpha: Any, beta: Any, sigma_p: Any) -> Any:
    """``σ_A² = α² Σ_P / (α² Σ_P + β)``."""
    num = (alpha * alpha) * sigma_p
    den = num + beta
    if hasattr(den, "clamp"):
        return (num / den.clamp(min=1e-12)).clamp(0.0, 1.0 - 1e-8).sqrt()
    den = np.maximum(np.asarray(den, dtype=np.float64), 1e-12)
    sa2 = np.clip(np.asarray(num, dtype=np.float64) / den, 0.0, 1.0 - 1e-8)
    return np.sqrt(sa2)
