"""Worker op ``ml_i_surrogate_fit``: the interleaved-refinement checkpoint.

One round trip does everything a checkpoint needs:

1. the **exact** ``ml_i`` evaluation at the current model (NLL work / free, per
   reflection, ``dQ/dF_calc``, curvature) -- the number that adjudicates blocks;
2. the exact score and curvature in ``t = E_C``, and the Rice surrogate fitted to
   match them (:mod:`phridge.contrib.intensity_ll.surrogate`);
3. the posterior quantities the controller's refresh heuristic reads (``<E>``,
   ``rho2``).

Inputs mirror ``target_eval`` so the caller passes the same per-reflection arrays it
already assembles for the exact target: the surrogate and the exact NLL therefore
share one posterior, one ``sigma_A``, one ``Sigma_W``.

The returned ``f_p`` / ``beta_p`` arrays are **internal**. They are in ``ml_f``
(amplitude) units, ready to be fed straight back as ``f_obs`` / ``beta``, and they
must never be printed, labelled, or written to an output file: ``f_p`` reads like an
observed amplitude and ``beta_p`` like an experimental variance.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from phridge.packing import PackedMiller

SURROGATE_FIT_OP_NAME = "ml_i_surrogate_fit"

_SURROGATE_INPUTS = {
    "f_calc": "MillerArray",
    "f_obs": "MillerArray",
    "target": "json",
    "weights": "array",
    "r_free": "array",
    "alpha": "array",
    "beta": "array",
    "epsilon": "array",
    "centric": "array",
    "nu": "array",
    "shell_index": "array",
    "surrogate": "json",
}

_SURROGATE_OUTPUTS = {
    # exact side -- the sole arbiter
    "target": "TargetResult",
    "nll": "json",
    # surrogate side -- internal arrays, never user-visible
    "f_p": "array",
    "alpha_p": "array",
    "beta_p": "array",
    "mask": "array",
    # diagnostics / refresh heuristic
    "score": "array",
    "curvature": "array",
    "e_c": "array",
    "e_mean": "array",
    "rho2": "array",
    "telemetry": "json",
}


def ml_i_surrogate_fit(
    f_calc: PackedMiller,
    f_obs: PackedMiller,
    target: dict,
    weights: Optional[Any] = None,
    r_free: Optional[Any] = None,
    alpha: Optional[Any] = None,
    beta: Optional[Any] = None,
    epsilon: Optional[Any] = None,
    centric: Optional[Any] = None,
    nu: Optional[Any] = None,
    shell_index: Optional[Any] = None,
    surrogate: Optional[dict] = None,
) -> dict[str, Any]:
    """Exact ``ml_i`` evaluation plus the Rice surrogate fitted at the same model."""
    import torch

    from phridge.contrib.intensity_ll.surrogate import (
        SurrogateFitOptions,
        fit_surrogate_for_observations,
    )
    from phridge.sfcalc.ops import _DEVICE, _np, _observations, _target_result
    from phridge.sfcalc.targets import build_target

    if _np(f_calc.hkl).shape != _np(f_obs.hkl).shape or not np.array_equal(
        _np(f_calc.hkl), _np(f_obs.hkl)
    ):
        raise ValueError("f_calc and f_obs must be on identical hkl lists")

    opts = SurrogateFitOptions.model_validate(surrogate or {})
    spec = dict(target)
    if spec.get("name") not in ("ml_i", None):
        raise ValueError(f"ml_i_surrogate_fit needs the ml_i target, got {spec.get('name')!r}")
    spec["name"] = "ml_i"
    tgt = build_target(spec)
    obs = _observations(f_obs, weights, r_free, alpha, beta, epsilon, centric, nu=nu)

    dev = _DEVICE["device"]
    is_mps = dev.startswith("mps")
    cdtype = torch.complex64 if is_mps else torch.complex128
    np_cdtype = np.complex64 if is_mps else np.complex128
    fc = torch.as_tensor(_np(f_calc.data, np_cdtype), dtype=cdtype, device=dev)

    # 1. the exact evaluation -- this is what accepts or rejects a block
    ev = tgt.evaluate(fc, obs, compute_curvature=True)

    # 2. the surrogate, fitted on the same posterior
    shell_t = None
    if shell_index is not None:
        shell_np = np.asarray(_np(shell_index), dtype=np.int64)
        if shell_np.size:
            shell_t = torch.as_tensor(shell_np, device=dev)

    fit = fit_surrogate_for_observations(tgt, fc, obs, opts, shell_index=shell_t)

    n_work = int(obs.n_work)
    per = np.asarray(ev.per_reflection, dtype=np.float64)
    nll = {
        "work": float(ev.value),
        "free": None if ev.value_test is None else float(ev.value_test),
        "n_work": n_work,
        "sum_work": float(per[np.asarray(obs.work.cpu().numpy(), dtype=bool)].sum()),
    }

    return {
        "target": _target_result(tgt.name, ev),
        "nll": nll,
        "f_p": fit.f_p,
        "alpha_p": fit.alpha_p,
        "beta_p": fit.beta_p,
        "mask": fit.mask.astype(np.float64),
        "score": fit.score,
        "curvature": fit.curvature,
        "e_c": fit.e_c,
        "e_mean": fit.e_mean,
        "rho2": fit.rho2,
        "telemetry": fit.telemetry,
    }


ml_i_surrogate_fit.compute_dtype = "float64"  # type: ignore[attr-defined]
