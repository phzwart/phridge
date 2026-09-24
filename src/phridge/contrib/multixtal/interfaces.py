"""Phase-2-swappable protocols. Phase 1 implements each; phase 2 swaps the body.

Torch-free at import: protocols are typing-only.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class ReflectionModel(Protocol):
    """Per-reflection / per-observation quasi-likelihood and moments.

    Phase 1: Rice-moment Gaussian quasi-likelihood.
    Phase 2: integrated intensity likelihood from ``mli.py`` with
    ``E_c`` replaced by ``|Fc + A z + sum_k g_k(D) A^dam_k|``.
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
        """``(expected_intensity, variance)``. ``a_sys`` is an optional observation scale."""

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
        """Per-reflection (or per-observation) negative log-likelihood."""


@runtime_checkable
class LatentPosterior(Protocol):
    """Per-dataset posterior over scores ``z_d``."""

    def estimate(self, residual: Any, weights: Any, loadings: Any, ridge: float = 1e-6) -> Any:
        """Return a point (or quadrature) representation of ``z_d``."""


@runtime_checkable
class LoadingParameterization(Protocol):
    """How factor loadings enter the mean."""

    def apply(self, loadings: Any, scores: Any) -> Any:
        """Contribution to the residual mean: ``scores @ loadings``."""


@runtime_checkable
class AnomalousTerm(Protocol):
    """Within-dataset Bijvoet-difference model."""

    def regress(
        self,
        d_e2: Any,
        scores: Any,
        weights: Any,
        observed: Any,
    ) -> tuple[Any, Any]:
        """Return ``(intercept, slopes)`` per reflection."""


@runtime_checkable
class ReflectionSelector(Protocol):
    """Which reflections enter the covariance model."""

    def select(
        self,
        i_bar: Any,
        sig_bar: Any,
        observed: Any,
        e_calc: Optional[Any] = None,
    ) -> Any:
        """Boolean mask over the common Miller set (same for every dataset)."""


@runtime_checkable
class InfluenceAnalyzer(Protocol):
    """Fisher information shares, Cook's distances, consistency.

    ``group_id`` (length O or N) aggregates shares over an arbitrary grouping
    (crystal, dose bin, pass). Phase 2 adds leave-one-group-out log predictive density.
    """

    def information_shares(
        self,
        weights: Any,
        design: Optional[Any] = None,
        group_id: Optional[Any] = None,
    ) -> Any:
        """Group shares of Fisher information; rows sum to 1."""

    def cooks_distance(self, theta_full: Any, theta_loo: Any, information: Any, n_params: int) -> Any:
        """Cook's distance with the full-data information."""

    def consistency(self, chi2_loo: Any, dof: Any) -> Any:
        """Leave-one-group-out chi-squared / dof."""


@runtime_checkable
class SystematicsModel(Protocol):
    """Per-batch scale/B, absorption surface, optional partiality.

    Smooth in reciprocal space and in batch: cannot fit a reflection-specific pattern.
    """

    def scale(self, s_sq: Any, s_inc: Any, s_dif: Any, batch: Any, wedge: Any = None) -> Any:
        """Observation-level multiplicative scale ``A_sys``."""


@runtime_checkable
class DoseModel(Protocol):
    """Dose profile ``g_k(ρ D)`` for damage modes."""

    def profiles(self, dose: Any, rho: Any = None) -> Any:
        """``(O, K)`` or ``(N, K)`` values of ``g_k``."""


@runtime_checkable
class DamageTerm(Protocol):
    """Damage loadings ``ℓ_{k,h}``.

    Phase 2: real-space modes entering ``E_c(z, D)``.
    """

    def apply(self, profiles: Any, loadings: Any) -> Any:
        """``profiles @ loadings`` contribution to the residual mean."""
