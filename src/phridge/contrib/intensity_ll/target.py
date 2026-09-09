"""Crystallographic intensity likelihood target (``ml_i``).

Negative log-likelihood via adaptive Rice/Woolfson × intensity-noise quadrature
(:mod:`phridge.contrib.intensity_ll.mli`).

Observation mapping (same ``alpha`` / ``beta`` slots as ``ml_f``, different meaning):

* ``obs.data`` — I_obs
* ``obs.sigmas`` — sigma(I); else fixed ``sigma`` option
* ``obs.epsilon`` — multiplicity epsilon (default 1)
* ``obs.centric`` — centric flags (default False)
* ``obs.alpha`` — sigma_A in (0, 1); else scalar ``sigma_a`` option
* ``obs.beta`` — Wilson scale Sigma = <|F|^2>/epsilon; else ``sigma_wilson`` option
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

from phridge.sfcalc.targets.base import Observations, Target, TargetEval, register_target


class IntensityLogLikelihoodOptions(BaseModel):
    """JSON options for ``{"name": "ml_i", ...}`` target specs."""

    model_config = {"extra": "forbid"}

    sigma_a: Optional[float] = Field(
        default=None,
        gt=0.0,
        lt=1.0,
        description="Scalar sigma_A when obs.alpha is absent.",
    )
    sigma_wilson: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Scalar Wilson scale Sigma when obs.beta is absent.",
    )
    sigma: float = Field(
        default=1.0,
        gt=0.0,
        description="Fixed sigma(I) when obs.sigmas is absent; ignored when sigmas are provided.",
    )
    use_sigmas: bool = Field(
        default=True,
        description="If True and obs.sigmas is set, use per-reflection sigmas.",
    )
    nu: Optional[float] = Field(
        default=None,
        gt=2.0,
        description="Student-t degrees of freedom; None uses normal intensity noise.",
    )
    snr_strong: float = Field(default=5.0, gt=0.0)
    n_hermite: int = Field(default=7, ge=3)
    n_legendre: int = Field(default=24, ge=4)
    k_window: float = Field(default=8.0, gt=0.0)
    n_u: int = Field(default=12, ge=4, description="Gauss nodes for Student-t mixture in log(lambda).")
    differentiate_window: bool = Field(
        default=False,
        description="If True, backprop through Newton window (roughly 2x cost).",
    )

    @field_validator("sigma", "snr_strong", "k_window")
    @classmethod
    def _finite_positive(cls, v: float) -> float:
        if not (v == v) or v == float("inf"):
            raise ValueError("must be a finite positive float")
        return float(v)

    @field_validator("sigma_a", "sigma_wilson", "nu")
    @classmethod
    def _finite_optional(cls, v: Optional[float]) -> Optional[float]:
        if v is None:
            return None
        if not (v == v) or v == float("inf"):
            raise ValueError("must be a finite float")
        return float(v)


@register_target("ml_i")
class IntensityLogLikelihood(Target):
    """Intensity NLL: -log ∫ Rice/Woolfson(E)×noise(Z_o|E²) dE (amplitude-only)."""

    amplitude_only = True

    def __init__(self, **options: object) -> None:
        opts = IntensityLogLikelihoodOptions.model_validate(options)
        super().__init__(**opts.model_dump())
        self.sigma_a: Optional[float] = opts.sigma_a
        self.sigma_wilson: Optional[float] = opts.sigma_wilson
        self.sigma: float = opts.sigma
        self.use_sigmas: bool = opts.use_sigmas
        self.nu: Optional[float] = opts.nu
        self.snr_strong: float = opts.snr_strong
        self.n_hermite: int = opts.n_hermite
        self.n_legendre: int = opts.n_legendre
        self.k_window: float = opts.k_window
        self.n_u: int = opts.n_u
        self.differentiate_window: bool = opts.differentiate_window

    def _sigma(self, obs: Observations) -> Any:
        import torch

        if self.use_sigmas and obs.sigmas is not None:
            return obs.sigmas.clamp(min=1e-300)
        return torch.full_like(obs.data, float(self.sigma))

    def _sigma_a(self, obs: Observations) -> Any:
        import torch

        if obs.alpha is not None:
            return obs.alpha
        if self.sigma_a is not None:
            return torch.full_like(obs.data, float(self.sigma_a))
        raise ValueError("ml_i needs obs.alpha (sigma_A) or options.sigma_a")

    def _sigma_wilson(self, obs: Observations) -> Any:
        import torch

        if obs.beta is not None:
            return obs.beta
        if self.sigma_wilson is not None:
            return torch.full_like(obs.data, float(self.sigma_wilson))
        raise ValueError("ml_i needs obs.beta (sigma_wilson) or options.sigma_wilson")

    def _nu(self, obs: Observations) -> Any:
        if getattr(obs, "nu", None) is not None:
            return obs.nu
        if self.nu is not None:
            return self.nu
        return None

    def per_reflection(self, f_calc, obs: Observations):
        import torch

        from phridge.contrib.intensity_ll.mli import (
            log_likelihood_normal,
            log_likelihood_t,
            normalize,
        )

        fo = obs.data
        fc = f_calc.abs()
        eps = obs.epsilon if obs.epsilon is not None else torch.ones_like(fo)
        centric = obs.centric if obs.centric is not None else torch.zeros_like(fo, dtype=torch.bool)
        sA = self._sigma_a(obs)
        sW = self._sigma_wilson(obs)
        sig = self._sigma(obs)

        ok = (
            (sA > 0)
            & (sA < 1.0 - 1e-6)
            & (sW > 0)
            & (sig > 0)
            & (eps > 0)
            & (fc > 0)
        )
        # Safe stand-ins where ok is False (masked to 0 below).
        sA_s = torch.where(ok, sA, torch.full_like(sA, 0.5))
        sW_s = torch.where(ok, sW, torch.ones_like(sW))
        sig_s = torch.where(ok, sig, torch.ones_like(sig))
        eps_s = torch.where(ok, eps, torch.ones_like(eps))
        fc_s = torch.where(ok, fc, torch.ones_like(fc))
        fo_s = torch.where(ok, fo, torch.zeros_like(fo))

        Ec, sA_n, Zo, sZ = normalize(fc_s, fo_s, sig_s, eps_s, sW_s, sA_s)
        q_kw = dict(
            snr_strong=self.snr_strong,
            n_hermite=self.n_hermite,
            n_legendre=self.n_legendre,
            k_window=self.k_window,
            differentiate_window=self.differentiate_window,
        )
        nu = self._nu(obs)
        if nu is not None:
            ll = log_likelihood_t(Ec, sA_n, Zo, sZ, centric, nu, n_u=self.n_u, **q_kw)
        else:
            ll = log_likelihood_normal(Ec, sA_n, Zo, sZ, centric, **q_kw)
            assert isinstance(ll, torch.Tensor)
        return torch.where(ok, -ll, torch.zeros_like(ll))

    def evaluate(self, f_calc, obs: Observations, compute_curvature: bool = True) -> TargetEval:
        import time

        t0 = time.perf_counter()
        res = super().evaluate(f_calc, obs, compute_curvature=compute_curvature)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        import os

        env_verb = os.environ.get("PHRIDGE_VERBOSE_TARGET", "0").strip().lower()
        if env_verb in ("1", "true", "yes", "on"):
            nu_val = self._nu(obs)
            if nu_val is not None:
                if hasattr(nu_val, "numel") and nu_val.numel() > 1:
                    nu_str = f"nu={float(nu_val.mean().item()):.1f} (mean)"
                else:
                    nu_f = float(nu_val.item()) if hasattr(nu_val, "item") else float(nu_val)
                    nu_str = f"nu={nu_f:.1f}"
            else:
                nu_str = "Gaussian"
            test_val = f"{res.value_test:.6f}" if res.value_test is not None else "N/A"
            curv_str = f"curv={'yes' if compute_curvature else 'no'}"
            print(
                f">>> [mli_quad worker] Target (work) = {res.value:.6f} | Free = {test_val} | "
                f"PyTorch eval: {elapsed_ms:.2f} ms | {nu_str} | {curv_str}",
                flush=True,
            )
        return res
