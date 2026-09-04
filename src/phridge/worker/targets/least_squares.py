"""Least-squares residuals, matching cctbx xray::targets::least_squares_residual.

    Q = sum_h w_h (y_obs,h - k y_calc,h)^2 / sum_h w_h y_obs,h^2

with y = |F| (obs_type "F") or |F|^2 (obs_type "I"). The scale k is
either given or computed by the closed-form minimizer on the work set
(``compute_scale_using_all_data`` switches to all reflections); as in
cctbx, k is held fixed when differentiating.
"""

from __future__ import annotations

from typing import Optional

from phridge.worker.targets.base import Observations, Target, register_target


@register_target("ls")
class LeastSquares(Target):
    amplitude_only = True

    def __init__(
        self,
        obs_type: str = "F",
        scale_factor: Optional[float] = None,
        compute_scale_using_all_data: bool = False,
        use_sigmas_as_weights: bool = False,
    ) -> None:
        super().__init__(
            obs_type=obs_type,
            scale_factor=scale_factor,
            compute_scale_using_all_data=compute_scale_using_all_data,
            use_sigmas_as_weights=use_sigmas_as_weights,
        )
        if obs_type not in ("F", "I"):
            raise ValueError("obs_type must be 'F' or 'I'")
        self.obs_type = obs_type
        self.fixed_scale = scale_factor
        self.compute_scale_using_all_data = compute_scale_using_all_data
        self.use_sigmas_as_weights = use_sigmas_as_weights
        self.scale_factor_ = scale_factor

    def _ycalc(self, f_calc):
        return f_calc.abs() if self.obs_type == "F" else (f_calc.real**2 + f_calc.imag**2)

    def _weights(self, obs: Observations):
        import torch

        if obs.weights is not None:
            return obs.weights
        if self.use_sigmas_as_weights and obs.sigmas is not None:
            return 1.0 / obs.sigmas.clamp(min=1e-300) ** 2
        return torch.ones_like(obs.data)

    def prepare(self, f_calc, obs: Observations) -> None:
        import torch

        if self.fixed_scale is not None and self.fixed_scale != 0:
            self.scale_factor_ = float(self.fixed_scale)
            return
        with torch.no_grad():
            w = self._weights(obs)
            y = self._ycalc(f_calc)
            sel = torch.ones_like(w, dtype=torch.bool) if self.compute_scale_using_all_data else obs.work
            num = (w * obs.data * y * sel).sum()
            den = (w * y * y * sel).sum()
            if den == 0:
                raise ValueError("cannot compute LS scale: sum w ycalc^2 == 0")
            self.scale_factor_ = float((num / den).item())

    def per_reflection(self, f_calc, obs: Observations):
        w = self._weights(obs)
        y = self._ycalc(f_calc)
        k = self.scale_factor_ if self.scale_factor_ is not None else 1.0
        delta = obs.data - k * y
        return w * delta * delta

    def reduce(self, t, obs: Observations):
        w = self._weights(obs)
        norm = (w * obs.data * obs.data * obs.work).sum()
        return (t * obs.work).sum() / norm
