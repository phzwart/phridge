"""Intensity Gaussian NLL: I_obs vs |F_calc|^2 (amplitude-only)."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from phridge.sfcalc.targets.base import Observations, Target, register_target


class IntensityLogLikelihoodOptions(BaseModel):
    """JSON options for ``{"name": "ml_i", ...}`` target specs."""

    model_config = {"extra": "forbid"}

    sigma: float = Field(
        default=1.0,
        gt=0.0,
        description="Fixed sigma when obs.sigmas is absent; ignored when sigmas are provided.",
    )
    use_sigmas: bool = Field(
        default=True,
        description="If True and obs.sigmas is set, use per-reflection sigmas.",
    )

    @field_validator("sigma")
    @classmethod
    def _finite_sigma(cls, v: float) -> float:
        if not (v == v) or v == float("inf"):  # NaN / inf
            raise ValueError("sigma must be a finite positive float")
        return float(v)


@register_target("ml_i")
class IntensityLogLikelihood(Target):
    """Gaussian NLL on intensities with I_calc = |F|^2.

    Per reflection: 0.5 * ((I_obs - I_calc) / sigma)^2.
    ``obs.data`` is I_obs (same convention as LS ``obs_type='I'``).
    """

    amplitude_only = True

    def __init__(self, **options: object) -> None:
        opts = IntensityLogLikelihoodOptions.model_validate(options)
        super().__init__(**opts.model_dump())
        self.sigma: float = opts.sigma
        self.use_sigmas: bool = opts.use_sigmas

    def _sigma(self, obs: Observations):
        import torch

        if self.use_sigmas and obs.sigmas is not None:
            return obs.sigmas.clamp(min=1e-300)
        return torch.full_like(obs.data, float(self.sigma))

    def per_reflection(self, f_calc, obs: Observations):
        i_calc = f_calc.real**2 + f_calc.imag**2
        resid = obs.data - i_calc
        sigma = self._sigma(obs)
        return 0.5 * (resid / sigma) ** 2
