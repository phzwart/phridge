"""Symmetry-adapted, band-limited Fourier bases for λ(x) and κ(x).

Evaluated at atom positions only — never as a real-space product with the
density (spec §6: e^λ is not band-limited even when λ is).

The field is real: Hermitian pairing c_{-k}=conj(c_k), c_0 excluded, cutoff
in Å. Unique real coefficients under the space group (reciprocal ASU).
Centric indices carry one real (phase restriction); acentric carry A and B.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from phridge.sfcalc.engine.cell import reciprocal_cartesian

_TWO_PI = 2.0 * np.pi


def _as_ops(rot: Any, trans: Any) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(rot, dtype=np.float64).reshape(-1, 3, 3)
    t = np.asarray(trans, dtype=np.float64).reshape(-1, 3)
    if r.shape[0] != t.shape[0]:
        raise ValueError("rot / trans length mismatch")
    return r, t


def _round_h(h: np.ndarray) -> np.ndarray:
    return np.rint(h).astype(np.int64)


def _hkl_box(unit_cell: tuple, cutoff: float) -> list[tuple[int, int, int]]:
    """All integer hkl with |d*| < 1/cutoff, excluding 000."""
    dstar_max = 1.0 / float(cutoff)
    # Generous index box from the three reciprocal basis lengths.
    dummy = np.eye(3, dtype=np.float64)
    step = np.linalg.norm(reciprocal_cartesian(unit_cell, dummy), axis=1)
    nmax = np.maximum(np.ceil(dstar_max / np.maximum(step, 1e-12)).astype(int) + 1, 1)
    out: list[tuple[int, int, int]] = []
    for h in range(-nmax[0], nmax[0] + 1):
        for k in range(-nmax[1], nmax[1] + 1):
            for l in range(-nmax[2], nmax[2] + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                out.append((h, k, l))
    if not out:
        return []
    hkl = np.asarray(out, dtype=np.int64)
    dstar = np.linalg.norm(reciprocal_cartesian(unit_cell, hkl), axis=1)
    keep = dstar < dstar_max - 1e-12
    return [out[i] for i in np.nonzero(keep)[0]]


def _representative(h: tuple[int, int, int], rot: np.ndarray) -> tuple[int, int, int]:
    hv = np.asarray(h, dtype=np.float64)
    images = _round_h(np.einsum("i,sij->sj", hv, rot))
    cands = [tuple(int(x) for x in row) for row in images]
    cands.extend([(-a, -b, -c) for a, b, c in cands])
    return min(cands)


def _is_absent(h: tuple[int, int, int], rot: np.ndarray, trans: np.ndarray) -> bool:
    hv = np.asarray(h, dtype=np.float64)
    for R, t in zip(rot, trans):
        hp = _round_h(hv @ R)
        if np.array_equal(hp, hv.astype(np.int64)):
            phase = np.exp(1j * _TWO_PI * float(np.dot(hv, t)))
            if abs(phase - 1.0) > 1e-8:
                return True
    return False


def _centric_phase(h: tuple[int, int, int], rot: np.ndarray, trans: np.ndarray) -> Optional[complex]:
    """If h is centric, return exp(2πi h·t) for an op with hR = −h; else None.

    Hermitian + this relation forces the phase of c_h (spec §6).
    """
    hv = np.asarray(h, dtype=np.float64)
    target = -hv.astype(np.int64)
    for R, t in zip(rot, trans):
        hp = _round_h(hv @ R)
        if np.array_equal(hp, target):
            return complex(np.exp(1j * _TWO_PI * float(np.dot(hv, t))))
    return None


@dataclass
class FieldBasis:
    """Unique real Fourier coefficients for a mean-zero, SG-adapted field.

    ``hkl[k]`` is the ASU representative for real parameter ``k``. Acentric
    indices appear twice (A then B). Centric indices appear once; ``is_sine``
    says whether the allowed component is the sine (imaginary) term.
    """

    hkl: np.ndarray  # (K, 3) int
    is_sine: np.ndarray  # (K,) bool
    unit_cell: tuple
    rot: np.ndarray
    trans: np.ndarray
    cutoff: float

    @property
    def n_coeff(self) -> int:
        return int(self.hkl.shape[0])

    @classmethod
    def build(
        cls,
        unit_cell: tuple,
        rot: Any,
        trans: Any,
        cutoff: float,
    ) -> FieldBasis:
        if cutoff <= 0.0:
            raise ValueError("field cutoff must be positive Å")
        rot, trans = _as_ops(rot, trans)
        seen: set[tuple[int, int, int]] = set()
        rows: list[tuple[int, int, int]] = []
        sine: list[bool] = []
        for h in sorted(_hkl_box(unit_cell, cutoff)):
            if _is_absent(h, rot, trans):
                continue
            rep = _representative(h, rot)
            if rep in seen:
                continue
            seen.add(rep)
            # c_0 excluded by skipping 000 in the box.
            phase = _centric_phase(rep, rot, trans)
            if phase is None:
                rows.append(rep)
                sine.append(False)
                rows.append(rep)
                sine.append(True)
            else:
                # conj(c) = c * phase ⇒ allowed direction. phase ≈ ±1 on standard SGs.
                rows.append(rep)
                sine.append(abs(phase + 1.0) < 1e-6)
        hkl = np.asarray(rows, dtype=np.int64).reshape(-1, 3)
        return cls(
            hkl=hkl,
            is_sine=np.asarray(sine, dtype=bool),
            unit_cell=tuple(float(x) for x in unit_cell),
            rot=rot,
            trans=trans,
            cutoff=float(cutoff),
        )

    def design_matrix(self, sites_frac: np.ndarray) -> np.ndarray:
        """(N, K) matrix M with field_j = (M @ c)_j. Atoms only (spec §6)."""
        x = np.asarray(sites_frac, dtype=np.float64).reshape(-1, 3)
        n, k = x.shape[0], self.n_coeff
        m = np.zeros((n, k), dtype=np.float64)
        if k == 0:
            return m
        # Expand each unique representative over the orbit, Hermitian-paired.
        # Group A/B pairs that share an hkl.
        i = 0
        while i < k:
            h = self.hkl[i]
            if i + 1 < k and tuple(self.hkl[i + 1]) == tuple(h) and (not self.is_sine[i]) and self.is_sine[i + 1]:
                m[:, i] = self._orbit_trig(x, h, sine=False)
                m[:, i + 1] = self._orbit_trig(x, h, sine=True)
                i += 2
            else:
                m[:, i] = self._orbit_trig(x, h, sine=bool(self.is_sine[i]))
                i += 1
        return m

    def _orbit_trig(self, x: np.ndarray, h: np.ndarray, *, sine: bool) -> np.ndarray:
        """SG-adapted 2 cos or −2 sin over the Friedel-unique orbit of h."""
        hv = h.astype(np.float64)
        acc = np.zeros(x.shape[0], dtype=np.float64)
        seen: set[tuple[int, int, int]] = set()
        for R, t in zip(self.rot, self.trans):
            hp = _round_h(hv @ R)
            key = tuple(int(v) for v in hp)
            key_m = (-key[0], -key[1], -key[2])
            # Keep one of {h', −h'} (Hermitian pairing).
            rep = min(key, key_m)
            if rep in seen:
                continue
            seen.add(rep)
            # Use the encountered h' (not the lex min). The Friedel mate is skipped
            # via ``seen``; flipping the index would require a non-constant phase.
            use = hp.astype(np.float64)
            # c_{h'} = c_h exp(2πi h·t) when h' = h R. Absorb the phase into the trig.
            phase = np.exp(1j * _TWO_PI * float(np.dot(hv, t)))
            ang = _TWO_PI * (x @ use)
            if sine:
                # −2 Im(phase) contribution on the sine/cosine mix.
                acc += -2.0 * (phase.real * np.sin(ang) + phase.imag * np.cos(ang))
            else:
                acc += 2.0 * (phase.real * np.cos(ang) - phase.imag * np.sin(ang))
        return acc

    def evaluate(self, sites_frac: np.ndarray, coeffs: Any) -> Any:
        """λ(x_j) or κ(x_j). Numpy or torch, whichever ``coeffs`` is."""
        m = self.design_matrix(sites_frac)
        try:
            import torch

            if isinstance(coeffs, torch.Tensor):
                dt = coeffs.dtype
                if str(coeffs.device).startswith("mps") and dt == torch.float64:
                    dt = torch.float32
                mt = torch.as_tensor(np.ascontiguousarray(m), dtype=dt)
                if mt.device != coeffs.device:
                    mt = mt.to(device=coeffs.device)
                return mt @ coeffs.reshape(-1)
        except ImportError:
            pass
        c = np.asarray(coeffs, dtype=np.float64).reshape(-1)
        if c.shape[0] != self.n_coeff:
            raise ValueError(f"expected {self.n_coeff} coefficients, got {c.shape[0]}")
        return m @ c

    def pack_hkl(self) -> np.ndarray:
        return np.asarray(self.hkl, dtype=np.int32)


def evaluate_fields(
    sites_frac: np.ndarray,
    lambda_c: Any,
    kappa_c: Any,
    basis: FieldBasis,
) -> tuple[Any, Any]:
    """(λ_j, κ_j) at atom positions. Both fields share the same basis."""
    return basis.evaluate(sites_frac, lambda_c), basis.evaluate(sites_frac, kappa_c)


def grid_n_real(
    unit_cell: tuple,
    cutoff: float,
    *,
    samples_per_cutoff: float = 4.0,
    n_min: int = 8,
    n_max: int = 48,
) -> tuple[int, int, int]:
    """Coarse grid: the field is band-limited at ``cutoff`` Å."""
    spacing = max(float(cutoff) / float(samples_per_cutoff), 0.5)
    out = []
    for i in range(3):
        n = int(round(float(unit_cell[i]) / spacing))
        out.append(int(np.clip(n, n_min, n_max)))
    return int(out[0]), int(out[1]), int(out[2])


def grid_sites_frac(n_real: tuple[int, int, int]) -> np.ndarray:
    """Fractional centres of a ``(nx, ny, nz)`` map, C-order ``i + nx*(j + ny*k)``."""
    nx, ny, nz = (int(n_real[0]), int(n_real[1]), int(n_real[2]))
    ii, jj, kk = np.meshgrid(
        np.arange(nx, dtype=np.float64) / nx,
        np.arange(ny, dtype=np.float64) / ny,
        np.arange(nz, dtype=np.float64) / nz,
        indexing="ij",
    )
    return np.stack([ii, jj, kk], axis=-1).reshape(-1, 3)


def sample_on_grid(
    basis: FieldBasis,
    coeffs: Any,
    n_real: tuple[int, int, int],
) -> np.ndarray:
    """Evaluate the band-limited field on a real-space grid. Same basis as atoms.

    This is not an FFT of ``exp(λ)`` (spec §6). Sample λ or κ, then ``exp``
    pointwise if you want a weight volume.
    """
    nx, ny, nz = (int(n_real[0]), int(n_real[1]), int(n_real[2]))
    c = np.asarray(coeffs, dtype=np.float64).reshape(-1)
    if basis.n_coeff == 0 or c.size == 0:
        return np.zeros((nx, ny, nz), dtype=np.float64)
    vals = np.asarray(basis.evaluate(grid_sites_frac((nx, ny, nz)), c), dtype=np.float64)
    return vals.reshape(nx, ny, nz)
