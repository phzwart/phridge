"""ASU density → crystal F via agentsg (h' = h W, phase = exp(2πi h·w)).

Same split as cctbx: splat the asymmetric unit only, then expand in
reciprocal space. agentsg provides the space group and the closed Seitz
list (point-group rotations plus translations and centering). The sum
over those operators matches ``maptbx.structure_factors.from_map``.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Optional, Sequence

import numpy as np


def agentsg_available() -> bool:
    try:
        import agentsg  # noqa: F401

        return True
    except Exception:
        return False


def resolve_space_group(key):
    """Look up a space group from number, Hall, HM, or IUCr ``R 3 :H`` / cctbx `` R 3``."""
    from agentsg import space_group

    if key is None:
        raise KeyError("empty space-group key")
    if isinstance(key, int):
        return space_group(key)
    raw = " ".join(str(key).strip().split())
    for cand in (raw, raw.replace(" ", "")):
        try:
            return space_group(cand)
        except KeyError:
            pass
    if ":" in raw:
        base, setting = [p.strip() for p in raw.split(":", 1)]
        try:
            sg = space_group(base)
        except KeyError:
            sg = None
        if sg is not None:
            if setting.upper() in ("H", "HEX", "HEXAGONAL"):
                return sg
            if setting.upper() in ("R", "RHOMB", "RHOMBOHEDRAL") and sg.number == 146:
                return sg
    compact = raw.lower().replace(" ", "").replace(":", "")
    aliases = {"r3h": 146, "r3": 146, "r32h": 155, "r3m": 160, "r3c": 161}
    if compact in aliases:
        return space_group(aliases[compact])
    raise KeyError(f"unknown space-group symbol {key!r}")


def ops_from_rt(rot: np.ndarray, trans: np.ndarray):
    """Exact Seitz ops from the numeric list already on the model, then closed."""
    from agentsg.group import centering_translations, close_group
    from agentsg.linalg import ZERO3, Matrix3, Vector3
    from agentsg.symmetry_op import SymmetryOp

    ops = []
    for R, t in zip(np.asarray(rot, dtype=np.float64), np.asarray(trans, dtype=np.float64)):
        W = Matrix3([[Fraction(int(round(float(R[i, j])))) for j in range(3)] for i in range(3)])
        w = Vector3(Fraction(float(t[k])).limit_denominator(24) for k in range(3))
        ops.append(SymmetryOp(W, w))
    cent = list(centering_translations(ops)) or [ZERO3]
    return list(close_group(ops, cent))


def mate_tables(
    hkl: np.ndarray,
    rot: np.ndarray,
    trans: np.ndarray,
    n_real: Sequence[int],
    hall: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-reflection gather of symmetry mates.

    Returns
    -------
    index : (n_refl, n_sym) int64
        Flat FFT index of ``-h W`` (engine convention: ``F_h`` from ``FFT[-h]``).
    phase : (n_refl, n_sym) complex128
        ``exp(2πi h·w)`` from agentsg ``phase_shift``.
    """
    from agentsg.group import phase_shift, transform_hkl
    from agentsg.linalg import Vector3

    ops = ops_from_rt(rot, trans)
    if hall:
        try:
            hall_ops = list(resolve_space_group(hall).operations())
            if len(hall_ops) == len(ops):
                ops = hall_ops
        except Exception:
            pass
    n0, n1, n2 = (int(v) for v in n_real)
    hkl = np.asarray(hkl, dtype=np.int64).reshape(-1, 3)
    n_refl = int(hkl.shape[0])
    n_sym = len(ops)
    index = np.zeros((n_refl, n_sym), dtype=np.int64)
    phase = np.ones((n_refl, n_sym), dtype=np.complex128)
    two_pi_i = 2j * np.pi
    for i, h in enumerate(hkl):
        hv = Vector3((int(h[0]), int(h[1]), int(h[2])))
        for s, op in enumerate(ops):
            hp = transform_hkl(hv, op.W)
            hr = tuple(int(x) for x in hp.v)
            i0 = (-hr[0]) % n0
            i1 = (-hr[1]) % n1
            i2 = (-hr[2]) % n2
            index[i, s] = (i0 * n1 + i1) * n2 + i2
            phase[i, s] = np.exp(two_pi_i * float(phase_shift(hv, op)))
    return index, phase
