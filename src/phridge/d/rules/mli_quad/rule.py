"""ml_i as a scoring rule. The quadrature is unchanged; this only reads it."""

from __future__ import annotations

from typing import Any

from phridge.contracts.tensors import Observed, Predictive


class MliQuad:
    """Score predictive fields against intensity observations."""

    def score(
        self, pred: Predictive, obs: Observed, mask: Any
    ) -> tuple[float, dict[str, Any]]:
        import torch

        from phridge.d.rules.amplitude.base import Observations
        from phridge.d.rules.mli_quad.target import IntensityLogLikelihood

        location = _tensor(pred.fields["location"])
        if not location.is_complex():
            location = location.to(torch.complex128)
        n = int(location.shape[0])
        nu = _tensor(pred.fields["nu"]).to(torch.float64).reshape(-1)
        if nu.numel() == 1:
            nu = nu.expand(n)
        work = _tensor(mask).to(torch.bool).reshape(-1)
        observations = Observations(
            data=_tensor(obs.data["i_obs"]).to(torch.float64),
            sigmas=_tensor(obs.data["sig_i"]).to(torch.float64),
            r_free=~work,
            epsilon=_tensor(obs.data["epsilon"]).to(torch.float64),
            centric=_tensor(obs.data["centric"]).to(torch.bool),
            alpha=_tensor(pred.fields["sigma_a"]).to(torch.float64),
            beta=_tensor(pred.fields["sigma_w"]).to(torch.float64),
            nu=nu,
            beta_residual=_tensor(pred.fields["beta_residual"]).to(torch.float64),
        )
        ev = IntensityLogLikelihood().evaluate(location, observations, compute_curvature=False)
        d_location = torch.as_tensor(ev.d_target_d_f_calc)
        return float(ev.value), {"location": d_location}


def _tensor(value: Any) -> Any:
    import torch

    if hasattr(value, "detach"):
        return value.detach()
    return torch.as_tensor(value)
