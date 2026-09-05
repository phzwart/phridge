"""Amplitude-based maximum-likelihood target (cctbx xray::targets::mlf).

Per reflection, with a = k alpha, b = k^2 beta, eb = epsilon b,
fo = |F_obs|, fc = |F_calc|:

acentric:  -ln(2 fo/eb) + fo^2/eb + (a fc)^2/eb - ln I0(2 a fo fc / eb)
centric:   -1/2 ln(2/(pi eb)) + fo^2/(2 eb) + (a fc)^2/(2 eb) - ln cosh(a fo fc / eb)

(Lunin & Skovoroda 1995; Lunin, Afonine & Urzhumtsev 2002; Afonine's
cctbx implementation). Reflections with alpha <= 0, beta <= 1e-3,
fo <= 0 or fc <= 0 contribute 0, as in cctbx. Value = mean over the
work set.
"""

from __future__ import annotations

import math

from phridge.sfcalc.targets.base import Observations, Target, register_target


def ln_i0(x):
    """log I0(x), stable for large x."""
    import torch

    return torch.log(torch.special.i0e(x)) + x.abs()


def ln_cosh(x):
    import torch

    ax = x.abs()
    return ax + torch.log1p(torch.exp(-2.0 * ax)) - math.log(2.0)


@register_target("ml_f")
class MaximumLikelihoodAmplitude(Target):
    amplitude_only = True

    def __init__(self, scale_factor: float = 1.0) -> None:
        super().__init__(scale_factor=scale_factor)
        self.scale_factor_ = float(scale_factor) if scale_factor > 0 else 1.0

    def per_reflection(self, f_calc, obs: Observations):
        import torch

        if obs.alpha is None or obs.beta is None:
            raise ValueError("ml_f needs alpha and beta per reflection")
        fo = obs.data
        fc = f_calc.abs()
        k = self.scale_factor_
        a = obs.alpha * k
        b = obs.beta * (k * k)
        eps = obs.epsilon if obs.epsilon is not None else torch.ones_like(fo)
        centric = obs.centric if obs.centric is not None else torch.zeros_like(fo, dtype=torch.bool)
        eb = eps * b
        ok = (obs.alpha > 0) & (obs.beta > 1e-3) & (fo > 0) & (fc > 0)
        # safe values where ok is False (result is masked to 0 anyway)
        eb_s = torch.where(ok, eb, torch.ones_like(eb))
        fo_s = torch.where(ok, fo, torch.ones_like(fo))
        fc_s = torch.where(ok, fc, torch.ones_like(fc))
        afc = a * fc_s
        x = 2.0 * a * fo_s * fc_s / eb_s
        acen = -torch.log(2.0 * fo_s / eb_s) + fo_s * fo_s / eb_s + afc * afc / eb_s - ln_i0(x)
        cen = (
            -0.5 * torch.log(2.0 / (math.pi * eb_s))
            + fo_s * fo_s / (2.0 * eb_s)
            + afc * afc / (2.0 * eb_s)
            - ln_cosh(0.5 * x)
        )
        t = torch.where(centric, cen, acen)
        return torch.where(ok, t, torch.zeros_like(t))
