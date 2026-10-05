"""Real-space proximal maps on a P1 FFT grid.

Support, F(000) removal, the band-limit selected by a Gaussian weight, and
the positivity clip are projections: each is idempotent. A Gaussian
multiplication itself is not idempotent, so ``project_smooth`` zeros
reflections whose Gaussian weight is below one half. That cutoff depends
only on ``s²`` and is centrosymmetric, so Friedel symmetry is preserved.

Gauge (orthonormality) is not applied here.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def _grid_shape(grid: Sequence[int]) -> tuple[int, int, int]:
    if len(grid) != 3:
        raise ValueError("grid must have length 3")
    return int(grid[0]), int(grid[1]), int(grid[2])


def _wrap(h: int, n: int) -> int:
    return int(h) % int(n)


def s_squared(hkl: np.ndarray, cell: Sequence[float]) -> np.ndarray:
    """Orthogonal-cell ``s²`` plus a general triclinic metric when angles are not 90°."""
    h = np.asarray(hkl, dtype=np.float64)
    a, b, c, alpha, beta, gamma = (float(x) for x in cell)
    ar, br, gr = np.radians([alpha, beta, gamma])
    # Reciprocal metric for a general cell.
    cos_a, cos_b, cos_g = np.cos(ar), np.cos(br), np.cos(gr)
    sin_g = np.sin(gr)
    volume = a * b * c * np.sqrt(max(1.0 - cos_a**2 - cos_b**2 - cos_g**2 + 2 * cos_a * cos_b * cos_g, 1e-12))
    astar = b * c * np.sin(ar) / volume
    bstar = a * c * np.sin(br) / volume
    cstar = a * b * sin_g / volume
    cos_as = (cos_b * cos_g - cos_a) / (np.sin(br) * sin_g)
    cos_bs = (cos_a * cos_g - cos_b) / (np.sin(ar) * sin_g)
    cos_gs = (cos_a * cos_b - cos_g) / (np.sin(ar) * np.sin(br))
    hh, kk, ll = h[:, 0], h[:, 1], h[:, 2]
    return (
        (hh * astar) ** 2
        + (kk * bstar) ** 2
        + (ll * cstar) ** 2
        + 2 * hh * kk * astar * bstar * cos_gs
        + 2 * hh * ll * astar * cstar * cos_bs
        + 2 * kk * ll * bstar * cstar * cos_as
    )


def complete_friedel(hkl: np.ndarray, coeff: np.ndarray, grid: Sequence[int]) -> np.ndarray:
    """Place a half-set on the grid and set ``F(-h) = conj F(h)``. F(000) stays real."""
    n0, n1, n2 = _grid_shape(grid)
    volume = np.zeros((n0, n1, n2), dtype=np.complex128)
    hkl_i = np.asarray(hkl, dtype=np.int32)
    data = np.asarray(coeff, dtype=np.complex128).reshape(-1)
    for h, value in zip(hkl_i, data):
        ih = (_wrap(int(h[0]), n0), _wrap(int(h[1]), n1), _wrap(int(h[2]), n2))
        volume[ih] = value
        mate = (_wrap(-int(h[0]), n0), _wrap(-int(h[1]), n1), _wrap(-int(h[2]), n2))
        if mate != ih:
            volume[mate] = np.conj(value)
        else:
            volume[ih] = complex(value.real, 0.0)
    volume[0, 0, 0] = complex(volume[0, 0, 0].real, 0.0)
    return volume


def coefficients_from_grid(volume: np.ndarray, hkl: np.ndarray) -> np.ndarray:
    n0, n1, n2 = volume.shape
    out = np.zeros(len(hkl), dtype=np.complex128)
    for i, h in enumerate(np.asarray(hkl, dtype=np.int32)):
        out[i] = volume[_wrap(int(h[0]), n0), _wrap(int(h[1]), n1), _wrap(int(h[2]), n2)]
    return out


def grid_to_density(volume: np.ndarray) -> np.ndarray:
    """Inverse FFT. A Friedel-complete Hermitian grid yields a real map."""
    return np.fft.ifftn(volume)


def density_to_grid(density: np.ndarray) -> np.ndarray:
    return np.fft.fftn(np.asarray(density, dtype=np.float64))


def project_support(density: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero density outside ``mask``. Idempotent."""
    out = np.array(density, dtype=np.float64, copy=True)
    out[~np.asarray(mask, dtype=bool)] = 0.0
    return out


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary dilation by ``radius`` voxels (Chebyshev). ``radius=0`` is a copy."""
    src = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return src.copy()
    out = src.copy()
    for _step in range(int(radius)):
        padded = np.pad(out, 1, mode="wrap")
        grown = np.zeros_like(out)
        for ia in range(3):
            for ib in range(3):
                for ic in range(3):
                    grown |= padded[ia : ia + out.shape[0], ib : ib + out.shape[1], ic : ic + out.shape[2]]
        out = grown
    return out


def project_f000(volume: np.ndarray) -> np.ndarray:
    """Set the ``(0,0,0)`` coefficient to 0. Idempotent and Friedel-preserving."""
    out = np.array(volume, copy=True)
    out[0, 0, 0] = 0.0
    return out


def project_smooth(volume: np.ndarray, hkl: np.ndarray, cell: Sequence[float], sigma: float) -> np.ndarray:
    """Band-limit projection selected by a Gaussian weight.

    Reflections with ``exp(-2 π² σ² s²) < 1/2`` are zeroed, together with
    their Friedel mates. ``sigma <= 0`` returns the grid unchanged. The map
    is idempotent.
    """
    if sigma <= 0.0:
        return np.array(volume, copy=True)
    weight = np.exp(-2.0 * np.pi**2 * float(sigma) ** 2 * s_squared(hkl, cell))
    out = np.array(volume, copy=True)
    n0, n1, n2 = out.shape
    for h, w in zip(np.asarray(hkl, dtype=np.int32), weight):
        if w >= 0.5:
            continue
        ih = (_wrap(int(h[0]), n0), _wrap(int(h[1]), n1), _wrap(int(h[2]), n2))
        mate = (_wrap(-int(h[0]), n0), _wrap(-int(h[1]), n1), _wrap(-int(h[2]), n2))
        out[ih] = 0.0
        out[mate] = 0.0
    return out


def project_positivity(
    density: np.ndarray,
    mode_maps: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    """Shift ``density`` so ``ρ + Σ z_k u_k`` stays non-negative on the observed z box.

    The minimum of the linear function over the coordinate-wise box of
    ``scores`` is lifted to 0. A second call sees a zero deficit.
    """
    rho = np.asarray(density, dtype=np.float64)
    modes = np.asarray(mode_maps, dtype=np.float64)
    z = np.asarray(scores, dtype=np.float64)
    if modes.size == 0 or z.size == 0:
        return np.maximum(rho, 0.0)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    zmin = z.min(axis=0)
    zmax = z.max(axis=0)
    lowest = rho.copy()
    for k in range(modes.shape[0]):
        lowest = lowest + np.minimum(zmin[k] * modes[k], zmax[k] * modes[k])
    deficit = np.minimum(lowest, 0.0)
    return rho - deficit


def support_mask_from_density(density: np.ndarray, level: float = 0.0) -> np.ndarray:
    """Boolean mask where ``density > level``. The reference solvent mask can replace this."""
    return np.asarray(density, dtype=np.float64) > float(level)


def project_modes(
    hkl: np.ndarray,
    cell: Sequence[float],
    delta0: np.ndarray,
    modes: np.ndarray,
    scores: np.ndarray,
    *,
    grid: Sequence[int] = (16, 16, 16),
    support: bool = False,
    support_mask: Optional[np.ndarray] = None,
    support_dilate: int = 0,
    smooth_sigma: float = 0.0,
    zero_f000: bool = True,
    positivity: bool = False,
    rho_mu: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the selected projections and return ``(delta0, modes)`` on ``hkl``."""
    hkl = np.asarray(hkl, dtype=np.int32)
    cell_t = tuple(float(x) for x in cell)
    delta = np.asarray(delta0, dtype=np.complex128).reshape(-1).copy()
    loads = np.asarray(modes, dtype=np.complex128)
    if loads.ndim == 1:
        loads = loads.reshape(1, -1)

    def _one(coeff: np.ndarray) -> np.ndarray:
        volume = complete_friedel(hkl, coeff, grid)
        if zero_f000:
            volume = project_f000(volume)
        if smooth_sigma > 0.0:
            volume = project_smooth(volume, hkl, cell_t, smooth_sigma)
        density = np.real(grid_to_density(volume))
        if support and support_mask is not None:
            mask = dilate_mask(support_mask, support_dilate)
            density = project_support(density, mask)
            volume = density_to_grid(density)
        return volume, density

    vol0, rho0 = _one(delta)
    mode_vols = []
    mode_dens = []
    for k in range(loads.shape[0]):
        vol_k, rho_k = _one(loads[k])
        mode_vols.append(vol_k)
        mode_dens.append(rho_k)
    if positivity and mode_dens:
        base = rho0 if rho_mu is None else np.asarray(rho_mu, dtype=np.float64)
        lifted = project_positivity(base, np.stack(mode_dens, axis=0), scores)
        # The lift is a real shift of the mean map; push it into Δ0.
        rho0 = rho0 + (lifted - base)
        vol0 = density_to_grid(rho0)
        if zero_f000:
            vol0 = project_f000(vol0)
    delta_out = coefficients_from_grid(vol0, hkl)
    if loads.shape[0] == 0:
        return delta_out, loads
    modes_out = np.stack([coefficients_from_grid(vol, hkl) for vol in mode_vols], axis=0)
    return delta_out, modes_out
