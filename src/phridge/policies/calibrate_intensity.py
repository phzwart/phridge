"""Calibrate the intensity likelihood.

Phases, in the order the legacy ``ml_i_nuisance_fit`` loop keeps them:

1. Wilson init and Σ_W. Moment Wilson plot for (Σ₀, B_W), then the model-free
   intensity fit: binned ``S_k exp(-½ sᵀ B s)`` with ``tr B = 0``, or the
   anisotropic / isotropic curve. No ``F_calc``.
2. Per-shell σ_A and β with Σ frozen. Logit shells, second-difference smoothing,
   optional β consistency and Rice-tensor terms.
3. ν, one of four modes. ``bins`` keeps ν logits in the same LBFGS as σ_A.
   ``grid`` refits σ_A/β at each ``parse_nu_grid`` node and at the Gaussian
   node ν=200, then restores the lowest grid NLL. ``grid_bins`` screens, then
   refits once with a ν picked per shell. ``global`` is a 1-D search after σ_A.

``run`` calls ``WilsonAniso.fit`` once, then ``NuScreen.screen``. The screen
calls ``SigmaABeta.fit`` at each node. There is no extra alternation.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np
from pydantic import BaseModel, Field

from phridge.contracts.manifest import BlockDecl, PluginManifest, Slot, Space
from phridge.contracts.tensors import Observed
from phridge.ops import register_manifest
from phridge.policies.base import Policy
from phridge.q.error_models.nu_screen import NuScreen, parse_nu_grid
from phridge.q.error_models.sigma_a_beta import ShellFitOptions, SigmaABeta
from phridge.q.error_models.wilson_aniso import WilsonAniso, WilsonOptions


class IntensityCalibrationOptions(BaseModel):
    """Legacy nuisance-fit arguments, one field per argument."""

    model_config = {"extra": "forbid"}

    wilson_model: str = "binned"
    sigma_a_mode: str = "bins"
    beta_mode: str = "free"
    sigma_a_shape: str = "free"
    nu_mode: str = "bins"
    nu_grid: Optional[str] = None
    fit_nu: bool = True
    fit_scale: bool = False
    fit_sigma_wilson: bool = True
    smooth_sigma_a: float = 1.0
    smooth_beta: float = 1.0
    beta_consistency_prior: float = 0.0
    n_sigma_a_bins: Optional[int] = None
    wilson_max_iter: int = 60
    sigma_a_tensor: bool = True
    sphericity: float = 1.0
    space_group_number: int = 1
    tv_norm: float = Field(default=0.0, ge=0.0)


class IntensityCalibration:
    """Shell σ_A, shell β, the selected ν, and the per-reflection Σ_W.

    ``sigma_a_refl`` and ``beta_residual`` are the interpolated, Rice-modulated
    arrays the scoring rule consumes. ``beta_residual`` is in normalized units.
    """

    def __init__(
        self,
        sigma_a: Any,
        beta: Any,
        nu: Any,
        sigma_w: Any,
        sigma_a_refl: Any = None,
        beta_residual: Any = None,
    ) -> None:
        self.sigma_a = sigma_a
        self.beta = beta
        self.nu = nu
        self.sigma_w = sigma_w
        self.sigma_a_refl = sigma_a_refl
        self.beta_residual = beta_residual


class CalibrateIntensity(Policy):
    """Wilson once, then the ν screen, which refits σ_A/β at every node."""

    def __init__(self, options: Optional[IntensityCalibrationOptions] = None) -> None:
        self.options = options or IntensityCalibrationOptions()

    def run(self, components: Mapping[str, Any], obs: Observed, mask: Any) -> IntensityCalibration:
        import torch

        opts = self.options
        if opts.fit_scale and opts.beta_mode == "free":
            raise ValueError("fit_scale=True with beta_mode='free' is not identifiable")
        if opts.nu_mode != "grid":
            raise ValueError(
                f"CalibrateIntensity.run currently follows the grid screen; got nu_mode={opts.nu_mode!r}"
            )
        wilson: WilsonAniso = components["wilson"]
        shells: SigmaABeta = components["sigma_a"]
        screen: NuScreen = components["nu"]
        sigma_w = wilson.fit(obs, mask).fields["sigma_w"]
        location = obs.data["f_calc"]

        def refit(nu: float) -> Any:
            shells.fit(location, obs, mask, float(nu), sigma_w)
            return shells.last

        def criterion(state: Any) -> float:
            return float(state.nll_sum)

        best = screen.screen(parse_nu_grid(opts.nu_grid), refit, criterion)
        s_sq = _numpy(obs.data["s_sq"], np.float64)
        hkl = _numpy(obs.index["hkl"], np.int32).reshape(-1, 3)
        cell = _numpy(obs.data["unit_cell"], np.float64)
        from phridge.q.error_models.wilson_aniso import reciprocal_cartesian

        s_cart = reciprocal_cartesian(tuple(float(x) for x in cell), hkl)
        sigma_a_refl, beta_residual = _expand_shells(best, s_sq, s_cart)
        return IntensityCalibration(
            sigma_a=torch.as_tensor(best.sigma_a, dtype=torch.float64),
            beta=torch.as_tensor(best.beta, dtype=torch.float64),
            nu=torch.as_tensor(screen.nu_shell, dtype=torch.float64),
            sigma_w=sigma_w,
            sigma_a_refl=torch.as_tensor(sigma_a_refl, dtype=torch.float64),
            beta_residual=torch.as_tensor(beta_residual, dtype=torch.float64),
        )

    def next(self, obs_history: Sequence[Any]) -> Optional[Any]:
        """One calibration, then stop. The controller protocol alongside ``run``."""
        from phridge.policies.base import Action

        for item in obs_history:
            name = getattr(item, "action", None) or getattr(item, "name", None)
            if name == "calibrate":
                return None
        return Action(name="calibrate")


def components_for(options: IntensityCalibrationOptions) -> dict[str, Any]:
    """The three fit components for one calibration."""
    from phridge.q.error_models.sigma_a_beta import normalize_beta_mode, normalize_sigma_a_shape

    shell = ShellFitOptions(
        beta_mode=normalize_beta_mode(options.beta_mode),
        sigma_a_shape=normalize_sigma_a_shape(options.sigma_a_shape),
        lambda_u=max(0.0, float(options.smooth_sigma_a)),
        lambda_v=max(0.0, float(options.smooth_beta)),
        lambda_consistency=max(0.0, float(options.beta_consistency_prior)),
    )
    return {
        "wilson": WilsonAniso(
            WilsonOptions(wilson_model=options.wilson_model),  # type: ignore[arg-type]
            max_iter=int(options.wilson_max_iter),
        ),
        "sigma_a": SigmaABeta(
            shell,
            n_bins=options.n_sigma_a_bins,
            sphericity=float(options.sphericity),
            sigma_a_tensor=bool(options.sigma_a_tensor),
            space_group_number=int(options.space_group_number),
        ),
        "nu": NuScreen(),
    }


def calibration_manifests() -> list[PluginManifest]:
    """Q-component and policy manifests. Not the FieldScale ``MANIFESTS`` dict."""
    q = dict(slot=Slot.q_component, space=Space.Q, fit_on="calibration", tier="native")
    return [
        PluginManifest(
            name="wilson_aniso_fit",
            produces=["sigma_w"],
            blocks=[BlockDecl(id="aniso_tensor", shape=[6])],
            **q,
        ),
        PluginManifest(
            name="sigma_a_beta_fit",
            produces=["sigma_a", "beta"],
            blocks=[
                BlockDecl(id="sigma_a_shell", shape=["n_shell"], bounds=(0.0, 1.0), transform="logit"),
                BlockDecl(id="beta_shell", shape=["n_shell"], bounds=(0.0, 1.0), transform="logit"),
            ],
            **q,
        ),
        PluginManifest(
            name="nu_screen",
            produces=["nu"],
            blocks=[BlockDecl(id="nu_shell", shape=["n_shell"])],
            **q,
        ),
        PluginManifest(
            name="calibrate_intensity",
            slot=Slot.policy,
            space=Space.Q,
            fit_on="calibration",
            reads=["i_obs", "f_calc"],
            produces=["sigma_w", "sigma_a", "beta", "nu"],
        ),
    ]


def register() -> None:
    """Register the calibration manifests outside the FieldScale catalog."""
    for manifest in calibration_manifests():
        register_manifest(manifest)


def _numpy(value: Any, dtype: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _expand_shells(result: Any, s_sq: np.ndarray, s_cart: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate shell curves onto reflections, then apply the Rice modulation."""
    from phridge.d.rules.mli_quad.aniso_rice import a_from_params, quadratic_form, unit_directions
    from phridge.q.error_models.sigma_a_beta import BETA_MAX, BETA_MIN

    edges = np.asarray(result.edges, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    sa_bins = np.asarray(result.sigma_a, dtype=np.float64)
    beta_bins = np.asarray(result.beta, dtype=np.float64)
    if sa_bins.size == 1:
        sigma_a = np.full(s_sq.shape[0], float(sa_bins[0]), dtype=np.float64)
        beta = np.full(s_sq.shape[0], float(beta_bins[0]), dtype=np.float64)
    else:
        sigma_a = np.interp(s_sq, centers, sa_bins, left=float(sa_bins[0]), right=float(sa_bins[-1]))
        beta = np.interp(s_sq, centers, beta_bins, left=float(beta_bins[0]), right=float(beta_bins[-1]))
    sigma_a = np.clip(sigma_a, 1e-4, 0.9999)
    beta = np.clip(beta, BETA_MIN, BETA_MAX)
    if result.u_m_a is not None and result.u_m_b is not None:
        s_hat = unit_directions(s_cart)
        q_a = np.clip(quadratic_form(s_hat, np.eye(3) + a_from_params(result.u_m_a, result.laue)), 1e-6, 1e6)
        q_b = np.clip(quadratic_form(s_hat, np.eye(3) + a_from_params(result.u_m_b, result.laue)), 1e-6, 1e6)
        sigma_a = np.clip(sigma_a * np.sqrt(q_a), 1e-4, 0.9999)
        beta = np.clip(beta * q_b, BETA_MIN, BETA_MAX)
    return sigma_a, beta
