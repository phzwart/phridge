"""Laue-class modulation tensors for per-shell σ_A and β.

The nuisance fit keeps one scalar σ_A and β per resolution shell. Residual
*directional* model quality — after Σ_W has taken the data anisotropy — is a single
pair of dimensionless 3×3 tensors:

    σ_A(h) = σ_A_k · √(ŝᵀ M_A ŝ)
    β(h)   = β_k   · (ŝᵀ M_β ŝ)

ŝ is the unit Cartesian reciprocal direction, so the tensors do not mix with
resolution (the shells own that). ``M = I + A`` with ``A`` traceless and restricted
to the Laue-allowed slots of a 2nd-rank symmetric tensor. Cubic crystals have no
free slots. Free components are ``A_MAX · tanh(u)`` so ``M`` stays SPD.

Σ_W remains the only carrier of *data* anisotropy. These tensors are model-quality
anisotropy after normalization. A sphericity prior ``λ ‖A‖_F²`` starts the fit at
``A = 0`` and charges any departure from a sphere.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from phridge.contrib.intensity_ll.wilson import (
    MIN_TENSOR_IDENTIFIABILITY,
    describe_wilson_tensor,
    reciprocal_cartesian,
    tensor_identifiability,
)

__all__ = [
    "A_MAX",
    "LaueClass",
    "ModulationTensor",
    "a_from_params",
    "a_from_params_torch",
    "describe_modulation",
    "frobenius_sq",
    "laue_from_crystal",
    "modulation_wanted",
    "n_free",
    "quadratic_form",
    "unit_directions",
]

A_MAX = 0.5


@dataclass(frozen=True)
class LaueClass:
    """Allowed independent components of a traceless symmetric 3×3 in this Laue class.

    ``n_free`` is the number of unconstrained logits. Cubic is empty — the tensor is
    identically ``I`` and the fit must not grow extra parameters.
    """

    name: str
    n_free: int
    # How the free vector ``u`` (after tanh) fills a traceless A. Implemented as a
    # named recipe rather than a matrix so torch and numpy share one definition.

    def __post_init__(self) -> None:
        if self.n_free < 0:
            raise ValueError("n_free must be >= 0")


LAUE_CUBIC = LaueClass("m-3m", 0)
LAUE_TETRAGONAL = LaueClass("4/mmm", 1)  # also 4/m, 6/mmm, -3m (hexagonal axes)
LAUE_ORTHORHOMBIC = LaueClass("mmm", 2)
LAUE_MONOCLINIC = LaueClass("2/m", 3)  # unique b: A13 free
LAUE_TRICLINIC = LaueClass("-1", 5)


def n_free(laue: LaueClass) -> int:
    return int(laue.n_free)


def _space_group_number(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if 1 <= n <= 230 else None


def _number_from_hall(hall: str) -> Optional[int]:
    """Best-effort IT number from a Hall string; most crystals already send the number."""
    text = str(hall or "").strip()
    # Occasional "P 21 21 21 (19)" or similar.
    m = re.search(r"\((\d{1,3})\)", text)
    if m:
        return _space_group_number(m.group(1))
    return None


def laue_from_number(number: int) -> LaueClass:
    """International Tables ranges → Laue class for a 2nd-rank symmetric tensor."""
    n = int(number)
    if n <= 2:
        return LAUE_TRICLINIC
    if n <= 15:
        return LAUE_MONOCLINIC
    if n <= 74:
        return LAUE_ORTHORHOMBIC
    if n <= 142:
        return LAUE_TETRAGONAL
    if n <= 194:
        return LAUE_TETRAGONAL  # trigonal / hexagonal, hexagonal axes
    return LAUE_CUBIC


def laue_from_hall(hall: str) -> LaueClass:
    """Fallback when ``space_group_number`` is missing. Conservative: unknown → triclinic."""
    n = _number_from_hall(hall)
    if n is not None:
        return laue_from_number(n)
    t = re.sub(r"\s+", " ", str(hall or "").strip().lower())
    if t in ("", "p 1", "-p 1", "p -1", "-p -1"):
        return LAUE_TRICLINIC
    if any(tok in t for tok in ("4 3", "2 3", "-3 2", "m-3", "m 3")):
        return LAUE_CUBIC
    if "6" in t or re.search(r"\b3\b", t) or "4" in t:
        return LAUE_TETRAGONAL
    n2 = len(re.findall(r"\b2", t))
    if n2 >= 2:
        return LAUE_ORTHORHOMBIC
    if n2 == 1:
        return LAUE_MONOCLINIC
    return LAUE_TRICLINIC


def laue_from_crystal(crystal: Any) -> LaueClass:
    """Read ``space_group_number`` or Hall off a :class:`CrystalSymmetry` (or similar)."""
    number = _space_group_number(getattr(crystal, "space_group_number", None))
    if number is None and isinstance(crystal, dict):
        number = _space_group_number(crystal.get("space_group_number"))
    if number is not None:
        return laue_from_number(number)
    hall = getattr(crystal, "space_group_hall", None)
    if hall is None and isinstance(crystal, dict):
        hall = crystal.get("space_group_hall")
    return laue_from_hall(str(hall or ""))


def _fill_a(values: np.ndarray, laue: LaueClass) -> np.ndarray:
    """Traceless A (3×3) from the bounded free components (already tanh-scaled)."""
    a = np.zeros((3, 3), dtype=np.float64)
    v = np.asarray(values, dtype=np.float64).ravel()
    if laue.n_free == 0 or v.size == 0:
        return a
    if laue.name == "4/mmm":
        x = float(v[0])
        a[0, 0] = a[1, 1] = x
        a[2, 2] = -2.0 * x
        return a
    if laue.name == "mmm":
        x, y = float(v[0]), float(v[1])
        a[0, 0], a[1, 1], a[2, 2] = x, y, -(x + y)
        return a
    if laue.name == "2/m":
        x, y, z = float(v[0]), float(v[1]), float(v[2])
        a[0, 0], a[1, 1], a[2, 2] = x, y, -(x + y)
        a[0, 2] = a[2, 0] = z
        return a
    # triclinic: 11, 22, 12, 13, 23; 33 = -(11+22)
    x, y = float(v[0]), float(v[1])
    a[0, 0], a[1, 1], a[2, 2] = x, y, -(x + y)
    a[0, 1] = a[1, 0] = float(v[2]) if v.size > 2 else 0.0
    a[0, 2] = a[2, 0] = float(v[3]) if v.size > 3 else 0.0
    a[1, 2] = a[2, 1] = float(v[4]) if v.size > 4 else 0.0
    return a


def a_from_params(u: Any, laue: LaueClass, a_max: float = A_MAX) -> np.ndarray:
    """``A = A_MAX · tanh(u)`` assembled into a traceless Laue-allowed 3×3."""
    raw = np.asarray(u, dtype=np.float64).ravel()
    if laue.n_free == 0:
        return np.zeros((3, 3), dtype=np.float64)
    if raw.size != laue.n_free:
        raise ValueError(f"expected {laue.n_free} logits for {laue.name}, got {raw.size}")
    return _fill_a(float(a_max) * np.tanh(raw), laue)


def a_from_params_torch(u: Any, laue: LaueClass, a_max: float = A_MAX) -> Any:
    """Torch twin of :func:`a_from_params`; ``u`` is a 1-D parameter tensor."""
    import torch

    z = u.sum() * 0.0
    if laue.n_free == 0:
        return torch.zeros((3, 3), dtype=u.dtype, device=u.device)
    v = float(a_max) * torch.tanh(u)

    def _diag(x, y, zz) -> Any:
        return torch.stack(
            [torch.stack([x, z, z]), torch.stack([z, y, z]), torch.stack([z, z, zz])]
        )

    if laue.name == "4/mmm":
        x = v[0]
        return _diag(x, x, -2.0 * x)
    if laue.name == "mmm":
        x, y = v[0], v[1]
        return _diag(x, y, -(x + y))
    if laue.name == "2/m":
        x, y, c = v[0], v[1], v[2]
        return torch.stack(
            [
                torch.stack([x, z, c]),
                torch.stack([z, y, z]),
                torch.stack([c, z, -(x + y)]),
            ]
        )
    x, y = v[0], v[1]
    return torch.stack(
        [
            torch.stack([x, v[2], v[3]]),
            torch.stack([v[2], y, v[4]]),
            torch.stack([v[3], v[4], -(x + y)]),
        ]
    )


def frobenius_sq(a: Any) -> Any:
    """‖A‖_F², numpy or torch."""
    if hasattr(a, "pow"):
        return (a * a).sum()
    arr = np.asarray(a, dtype=np.float64)
    return float(np.sum(arr * arr))


def unit_directions(s_cart: Any) -> np.ndarray:
    """``ŝ = s / |s|``, zeros left as zeros (those reflections do not constrain M)."""
    s = np.asarray(s_cart, dtype=np.float64)
    nrm = np.linalg.norm(s, axis=1)
    out = np.zeros_like(s)
    ok = nrm > 0
    out[ok] = s[ok] / nrm[ok, None]
    return out


def quadratic_form(s_hat: Any, m: Any) -> np.ndarray:
    """``ŝᵀ M ŝ`` for a stack of unit directions, shape (N,)."""
    u = np.asarray(s_hat, dtype=np.float64)
    mat = np.asarray(m, dtype=np.float64)
    return np.einsum("ni,ij,nj->n", u, mat, u)


def quadratic_form_torch(s_hat: Any, m: Any) -> Any:
    return (s_hat * (s_hat @ m.T)).sum(dim=-1)


def modulation_wanted(
    *,
    enabled: bool,
    beta_free: bool,
    sigma_a_free: bool,
    bins_mode: bool,
    laue: LaueClass,
    s_cart: Optional[np.ndarray],
    identifiability: Optional[float] = None,
) -> tuple[bool, Optional[str]]:
    """Whether to put M_A / M_β on the LBFGS, and why not if not.

    Constrained β or monotone σ_A stay bit-identical: no extra parameters. Cubic has
    nothing to fit. Unidentifiable directions fall back rather than invent a tensor.
    """
    if not enabled:
        return False, "disabled"
    if not bins_mode:
        return False, "no shells (read-style σ_A)"
    if not beta_free or not sigma_a_free:
        return False, None  # silent: constrained+monotone must not mention a fallback
    if laue.n_free == 0:
        return False, None  # cubic: isotropic by symmetry, not a failure
    if s_cart is None:
        return False, "no usable unit cell for the tensor"
    ident = (
        float(identifiability)
        if identifiability is not None
        else float(tensor_identifiability(s_cart))
    )
    if ident < MIN_TENSOR_IDENTIFIABILITY:
        return False, (
            f"reflection directions cannot determine a tensor "
            f"(identifiability {ident:.2e} < {MIN_TENSOR_IDENTIFIABILITY:g})"
        )
    return True, None


@dataclass
class ModulationTensor:
    """Inspectable eigen-form of ``M = I + A`` (dimensionless)."""

    m_cart: tuple[float, ...] = (1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    eigenvalues: tuple[float, ...] = (1.0, 1.0, 1.0)
    direction_cosines: tuple[tuple[float, ...], ...] = ()
    delta_aniso: float = 0.0
    n_params: int = 0
    laue: str = "m-3m"

    def as_json(self) -> dict[str, Any]:
        return {
            "m_cart": [float(x) for x in self.m_cart],
            "eigenvalues": [float(x) for x in self.eigenvalues],
            "direction_cosines": [[float(x) for x in row] for row in self.direction_cosines],
            "delta_aniso": float(self.delta_aniso),
            "n_params": int(self.n_params),
            "laue": str(self.laue),
        }


def describe_modulation(
    a: np.ndarray,
    *,
    unit_cell: Sequence[float],
    laue: LaueClass,
) -> ModulationTensor:
    """Eigen-form of ``M = I + A``, same reporting convention as the Wilson ``B``."""
    m = np.eye(3) + np.asarray(a, dtype=np.float64)
    # Reuse the Wilson reporter for axes / cosines; strip the Σ_W-specific fields.
    w = describe_wilson_tensor(
        sigma_0=1.0, b_cart=m, unit_cell=unit_cell, model="anisotropic", n_params=laue.n_free
    )
    return ModulationTensor(
        m_cart=w.b_cart,
        eigenvalues=w.eigenvalues,
        direction_cosines=w.direction_cosines,
        delta_aniso=w.delta_b_aniso,
        n_params=laue.n_free,
        laue=laue.name,
    )


def s_hat_from_crystal(crystal: Any, hkl: Any) -> np.ndarray:
    """Unit reciprocal directions from a crystal + Miller indices."""
    uc = crystal.unit_cell if hasattr(crystal, "unit_cell") else crystal["unit_cell"]
    return unit_directions(reciprocal_cartesian(uc, hkl))
