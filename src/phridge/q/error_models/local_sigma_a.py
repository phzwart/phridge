"""v1 spatial σ_A: inverse-mask two-channel mix and inspectable φ coefficients.

Torch-free. The worker may pass torch tensors into :func:`mix_f_eff` / 
:func:`channel_weights`; those helpers stay on whatever array type they are handed
so LBFGS can backprop through ``u``. See ``local_sigma_a.md``.
"""

from __future__ import annotations

import math
from typing import Any, Literal, Optional, Union

import numpy as np
from pydantic import BaseModel, Field, field_validator

DEFAULT_E_BAR = 0.5
DEFAULT_D_MIN = 15.0
DEFAULT_LAMBDA_U = 1.0

RmsGauge = Literal["shell", "global"]


class LocalSigmaAOptions(BaseModel):
    """JSON options for the inverse-mask spatial σ_A field (v1)."""

    model_config = {"extra": "forbid"}

    enabled: bool = False
    d_min: float = Field(default=DEFAULT_D_MIN, gt=0.0, description="Envelope cutoff in Å.")
    blur_b: float = Field(default=0.0, ge=0.0, description="Optional extra B on φ coefficients.")
    lambda_u: float = Field(
        default=DEFAULT_LAMBDA_U,
        ge=0.0,
        description="u² prior; tune-set regularization only.",
    )
    rms_gauge: RmsGauge = Field(
        default="shell",
        description="Per-shell (default) or global rms renormalization of F_mix.",
    )
    e_bar: float = Field(
        default=DEFAULT_E_BAR,
        gt=0.0,
        lt=1.0,
        description="Molecular volume fraction of the low-passed envelope.",
    )
    residual: bool = Field(
        default=False,
        description="v2 intra-molecular ψ; not implemented.",
    )

    @field_validator("residual")
    @classmethod
    def _no_v2(cls, v: bool) -> bool:
        if v:
            raise ValueError("intra-molecular residual ψ (v2) is not implemented")
        return False


def channel_weights(u: Any, e_bar: float) -> tuple[Any, Any]:
    """``w_mol = exp(u (1 - ē))``, ``w_sol = exp(-u ē)``. At ``u=0`` both are 1."""
    e = float(e_bar)
    try:
        import torch

        if isinstance(u, torch.Tensor):
            return torch.exp(u * (1.0 - e)), torch.exp(u * (-e))
    except ImportError:
        pass
    u_f = float(np.asarray(u, dtype=np.float64))
    return math.exp(u_f * (1.0 - e)), math.exp(u_f * (-e))


def _is_torch(x: Any) -> bool:
    try:
        import torch

        return isinstance(x, torch.Tensor)
    except ImportError:
        return False


def _abs2(x: Any) -> Any:
    if _is_torch(x):
        return x.real * x.real + x.imag * x.imag
    z = np.asarray(x)
    return np.real(z) * np.real(z) + np.imag(z) * np.imag(z)


def _rms(x: Any) -> Any:
    a2 = _abs2(x)
    if _is_torch(x):
        import torch

        n = max(int(a2.numel()), 1)
        return torch.sqrt(a2.sum() / n)
    a2 = np.asarray(a2, dtype=np.float64)
    n = max(int(a2.size), 1)
    return float(np.sqrt(np.sum(a2) / n))


def _ones_like_shells(n_shells: int, like: Any) -> Any:
    if _is_torch(like):
        import torch

        return torch.ones(n_shells, dtype=like.real.dtype, device=like.device)
    return np.ones(n_shells, dtype=np.float64)


def mix_f_eff(
    f_atoms: Any,
    f_bulk: Any,
    w_mol: Any,
    w_sol: Any,
    shell_id: Optional[Any] = None,
    gauge: RmsGauge = "shell",
) -> tuple[Any, Any]:
    """Two-channel mix plus rms gauge. At ``w_mol = w_sol = 1`` this is the identity.

    Returns ``(F_eff, n_k)``. ``n_k`` is one value per shell (``gauge='shell'``) or a
    length-1 array (``gauge='global'``).
    """
    f_model = f_atoms + f_bulk
    w_m_is_one = (not _is_torch(w_mol)) and float(w_mol) == 1.0
    w_s_is_one = (not _is_torch(w_sol)) and float(w_sol) == 1.0
    if w_m_is_one and w_s_is_one:
        if gauge == "global":
            return f_model, _ones_like_shells(1, f_model)
        n_shells = 1
        if shell_id is not None:
            if _is_torch(shell_id):
                n_shells = int(shell_id.max().item()) + 1 if int(shell_id.numel()) else 1
            else:
                sid = np.asarray(shell_id)
                n_shells = int(sid.max()) + 1 if sid.size else 1
        return f_model, _ones_like_shells(max(n_shells, 1), f_model)

    f_mix = w_mol * f_atoms + w_sol * f_bulk
    if gauge == "global" or shell_id is None:
        denom = _rms(f_model)
        n = _rms(f_mix) / _clamp_pos(denom)
        return f_mix / n, _as_len1(n, f_mix)

    if _is_torch(shell_id):
        return _mix_shell_torch(f_mix, f_model, shell_id)
    return _mix_shell_numpy(f_mix, f_model, np.asarray(shell_id, dtype=np.int64))


def _clamp_pos(x: Any, lo: float = 1e-30) -> Any:
    if _is_torch(x):
        return x.clamp(min=lo)
    return max(float(x), lo)


def _as_len1(n: Any, like: Any) -> Any:
    if _is_torch(n):
        return n.reshape(1)
    if _is_torch(like):
        import torch

        return torch.as_tensor([float(n)], dtype=like.real.dtype, device=like.device)
    return np.asarray([float(n)], dtype=np.float64)


def _mix_shell_numpy(
    f_mix: np.ndarray, f_model: np.ndarray, shell_id: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    n_shells = int(shell_id.max()) + 1 if shell_id.size else 1
    n_k = np.ones(n_shells, dtype=np.float64)
    for k in range(n_shells):
        sel = shell_id == k
        if not np.any(sel):
            continue
        denom = _rms(f_model[sel])
        n_k[k] = float(_rms(f_mix[sel]) / _clamp_pos(denom))
    scale = n_k[shell_id]
    scale = np.where(scale > 1e-30, scale, 1.0)
    return f_mix / scale, n_k


def _mix_shell_torch(f_mix: Any, f_model: Any, shell_id: Any) -> tuple[Any, Any]:
    import torch

    n_shells = int(shell_id.max().item()) + 1 if int(shell_id.numel()) else 1
    n_k = torch.ones(n_shells, dtype=f_mix.real.dtype, device=f_mix.device)
    for k in range(n_shells):
        sel = shell_id == k
        if not bool(sel.any()):
            continue
        denom = _rms(f_model[sel])
        n_k[k] = _rms(f_mix[sel]) / denom.clamp(min=1e-30)
    scale = n_k[shell_id].clamp(min=1e-30)
    return f_mix / scale, n_k


def apply_frozen_mix(
    f_atoms: Any,
    f_bulk: Any,
    w_mol: float,
    w_sol: float,
    n_k: Any,
    shell_id: Optional[Any] = None,
) -> Any:
    """Remix with frozen channel weights and frozen ``n_k`` (no re-gauge)."""
    f_mix = float(w_mol) * f_atoms + float(w_sol) * f_bulk
    n = np.asarray(n_k, dtype=np.float64).reshape(-1)
    if n.size == 1:
        scale = float(n[0]) if float(n[0]) > 1e-30 else 1.0
        return f_mix / scale
    if shell_id is None:
        raise ValueError("shell_id is required when n_k is per-shell")
    sid = np.asarray(shell_id, dtype=np.int64)
    scale = n[np.clip(sid, 0, n.size - 1)]
    scale = np.where(scale > 1e-30, scale, 1.0)
    return f_mix / scale


def envelope_coefficients(
    hkl: Any,
    f_mask: Any,
    s_sq: Any,
    d_min: float,
    blur_b: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverse-mask coefficients: ``c_h = -F_mask(h) exp(-B s²/4)`` for ``h≠0``, ``d≥d_min``.

    Returns ``(hkl_keep, c_h, keep_mask)``. Origin is dropped (``c_0`` stays 0).
    ``s_sq`` is ``1/d²``.
    """
    h = np.asarray(hkl, dtype=np.int32)
    fm = np.asarray(f_mask, dtype=np.complex128).reshape(-1)
    ss = np.asarray(s_sq, dtype=np.float64).reshape(-1)
    if h.ndim != 2 or h.shape[1] != 3:
        raise ValueError(f"hkl must be (N, 3), got {h.shape}")
    if fm.shape[0] != h.shape[0] or ss.shape[0] != h.shape[0]:
        raise ValueError("hkl, f_mask and s_sq must have the same length")
    d = 1.0 / np.sqrt(np.maximum(ss, 1e-30))
    nonzero = np.any(h != 0, axis=1)
    keep = nonzero & (d >= float(d_min)) & np.isfinite(fm) & np.isfinite(ss)
    damp = np.exp(-float(blur_b) * ss / 4.0)
    c = np.where(keep, -fm * damp, 0.0 + 0.0j)
    return h[keep], c[keep], keep


def phi_coefficients(
    hkl: Any,
    f_mask: Any,
    s_sq: Any,
    u: float,
    d_min: float,
    blur_b: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """φ map coefficients: ``c_0 = 0``, ``c_h = u ×`` inverse-mask envelope."""
    h_keep, c_env, keep = envelope_coefficients(hkl, f_mask, s_sq, d_min, blur_b)
    return h_keep, float(u) * c_env, keep


def phi_on_full_list(
    f_mask: Any,
    s_sq: Any,
    u: float,
    d_min: float,
    blur_b: float = 0.0,
    hkl: Optional[Any] = None,
) -> np.ndarray:
    """φ coefficients on the full ``F_mask`` list (zeros outside the cutoff)."""
    fm = np.asarray(f_mask, dtype=np.complex128).reshape(-1)
    if hkl is None:
        hkl = np.zeros((fm.shape[0], 3), dtype=np.int32)
        hkl[:, 0] = 1  # treat every slot as nonzero if hkl is unknown
    _h, _c, keep = envelope_coefficients(hkl, fm, s_sq, d_min, blur_b)
    out = np.zeros(fm.shape[0], dtype=np.complex128)
    out[keep] = float(u) * (-fm[keep] * np.exp(-float(blur_b) * np.asarray(s_sq)[keep] / 4.0))
    return out


def describe_local_sigma_a(
    *,
    enabled: bool,
    u: Optional[float] = None,
    blur_b: float = 0.0,
    d_min: float = DEFAULT_D_MIN,
    e_bar: float = DEFAULT_E_BAR,
    w_mol: Optional[float] = None,
    w_sol: Optional[float] = None,
    n_shell_rms: Optional[Any] = None,
    n_envelope_hkl: int = 0,
    lambda_u: float = DEFAULT_LAMBDA_U,
    rms_gauge: RmsGauge = "shell",
    fallback: Optional[str] = None,
) -> dict[str, Any]:
    """JSON fragment for ``sigma_a_params['local_sigma_a']``."""
    out: dict[str, Any] = {
        "enabled": bool(enabled),
        "d_min": float(d_min),
        "b_blur": float(blur_b),
        "e_bar": float(e_bar),
        "lambda_u": float(lambda_u),
        "rms_gauge": str(rms_gauge),
        "n_envelope_hkl": int(n_envelope_hkl),
    }
    if u is not None:
        out["u"] = float(u)
    if w_mol is not None:
        out["w_mol"] = float(w_mol)
    if w_sol is not None:
        out["w_sol"] = float(w_sol)
    if n_shell_rms is not None:
        out["n_shell_rms"] = [float(x) for x in np.asarray(n_shell_rms, dtype=np.float64).ravel()]
    if fallback:
        out["fallback"] = str(fallback)
    return out


def resolve_options(raw: Optional[Union[dict[str, Any], LocalSigmaAOptions, Any]]) -> LocalSigmaAOptions:
    """Validate a JSON / dict / model as :class:`LocalSigmaAOptions`."""
    if raw is None:
        return LocalSigmaAOptions()
    if isinstance(raw, LocalSigmaAOptions):
        return raw
    if isinstance(raw, dict):
        return LocalSigmaAOptions.model_validate(raw)
    if isinstance(raw, bool):
        return LocalSigmaAOptions(enabled=bool(raw))
    return LocalSigmaAOptions.model_validate(raw)
