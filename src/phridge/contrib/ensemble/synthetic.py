"""Helix campaign used by the numeric tests.

Parameters (also recorded in ``ensemble_modes.md``):

- orthogonal cell ``(30, 30, 40, 90, 90, 90)`` Å
- 12 carbon-like sites on a helix, radius 0.12, pitch 0.45, fractional
- ``K_true = 3`` cartesian displacement modes (tangent, radial, axis), step 0.008
- datasets draw ``z ~ N(0, diag(1.0, 0.6, 0.3))`` then centred
- Friedel half-set with ``hmax = 2`` (``hmax = 3`` on the slow campaign)
- intensity noise ``σ_I = noise * |F|² + 0.5``, growing with ``s²`` when
  ``resolution_noise`` is set
- seed 0

The note's N=128 campaign is not in the tree. ``reference/results*.json``,
when added, is the oracle for those stored angles and likelihoods. This
generator is the always-on campaign.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

CELL = (30.0, 30.0, 40.0, 90.0, 90.0, 90.0)
K_TRUE = 3
N_NOTE = 128
SEED = 0


@dataclass
class CampaignDifferenceMaps:
    """Real-space maps of the campaign differences.

    ``datasets[n]`` is the inverse FFT of ``F_n - F_μ``, which is the planted
    signal plus the complex residual. ``modes[k]`` is the inverse FFT of
    planted mode ``k``. Both are real: the half-set is Friedel-completed
    before the transform, and F(000) is absent from the half-set.
    """

    grid: tuple[int, int, int]
    datasets: np.ndarray
    modes: np.ndarray


@dataclass
class Campaign:
    hkl: np.ndarray
    centric: np.ndarray
    epsilon: np.ndarray
    cell: tuple[float, ...]
    f_mu: np.ndarray
    modes_true: np.ndarray
    z_true: np.ndarray
    f_true: np.ndarray
    i_obs: np.ndarray
    sig_i: np.ndarray
    mask: np.ndarray
    shell_index: np.ndarray
    d_spacing: np.ndarray
    sites: np.ndarray


def helix_sites(n_atoms: int = 12, radius: float = 0.12, pitch: float = 0.45) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, n_atoms, endpoint=False)
    along = np.linspace(0.2, 0.2 + pitch, n_atoms, endpoint=False)
    return np.stack(
        [0.5 + radius * np.cos(t), 0.5 + radius * np.sin(t), along],
        axis=1,
    ).astype(np.float64)


def half_set(hmax: int) -> np.ndarray:
    rows = []
    for h in range(-hmax, hmax + 1):
        for k in range(-hmax, hmax + 1):
            for l in range(-hmax, hmax + 1):
                if (h, k, l) == (0, 0, 0):
                    continue
                if (h, k, l) < (-h, -k, -l):
                    continue
                rows.append((h, k, l))
    return np.asarray(rows, dtype=np.int32)


def structure_factors(
    sites: np.ndarray,
    hkl: np.ndarray,
    cell: tuple[float, ...] = CELL,
    f_atom: float = 6.0,
    b_iso: float = 8.0,
) -> np.ndarray:
    """Direct summation, orthogonal or triclinic via the fractional phase only.

    The atomic factor is ``f exp(-B s² / 4)`` with ``s²`` from the orthogonal
    formula when the cell angles are 90°.
    """
    h = np.asarray(hkl, dtype=np.float64)
    xyz = np.asarray(sites, dtype=np.float64)
    phase = 2.0 * np.pi * (xyz @ h.T)
    a, b, c = cell[0], cell[1], cell[2]
    s2 = (h[:, 0] / a) ** 2 + (h[:, 1] / b) ** 2 + (h[:, 2] / c) ** 2
    factor = f_atom * np.exp(-b_iso * s2 / 4.0)
    return (factor[None, :] * np.exp(1j * phase)).sum(axis=0).astype(np.complex128)


def _mode_directions(sites: np.ndarray) -> list[np.ndarray]:
    centred = sites - sites.mean(axis=0)
    tangent = np.stack([-centred[:, 1], centred[:, 0], np.zeros(sites.shape[0])], axis=1)
    radial = centred.copy()
    radial[:, 2] = 0.0
    axial = np.zeros_like(sites)
    axial[:, 2] = 1.0
    directions = []
    for vec in (tangent, radial, axial):
        norm = np.linalg.norm(vec, axis=1, keepdims=True)
        directions.append(vec / np.maximum(norm, 1e-8))
    return directions


def displacement_modes(sites: np.ndarray, hkl: np.ndarray, step: float = 0.008) -> np.ndarray:
    """Finite-difference ``ΔF`` for three helix motions, then orthonormalised."""
    base = structure_factors(sites, hkl)
    modes = []
    for direction in _mode_directions(sites):
        moved = sites + step * direction
        modes.append((structure_factors(moved, hkl) - base) / step)
    raw = np.stack(modes, axis=0)
    embedded = np.concatenate([raw.real, raw.imag], axis=1)
    _u, _s, vh = np.linalg.svd(embedded, full_matrices=False)
    n_h = hkl.shape[0]
    ortho = vh[: raw.shape[0], :n_h] + 1j * vh[: raw.shape[0], n_h:]
    return ortho.astype(np.complex128)


def equal_count_shells(d_spacing: np.ndarray, bin_size: int) -> np.ndarray:
    d = np.asarray(d_spacing, dtype=np.float64)
    order = np.argsort(-d)
    shells = np.empty(d.shape[0], dtype=np.int32)
    shells[order] = np.arange(d.shape[0]) // max(int(bin_size), 1)
    return shells


def d_spacing(hkl: np.ndarray, cell: tuple[float, ...] = CELL) -> np.ndarray:
    h = np.asarray(hkl, dtype=np.float64)
    s2 = (h[:, 0] / cell[0]) ** 2 + (h[:, 1] / cell[1]) ** 2 + (h[:, 2] / cell[2]) ** 2
    return 1.0 / np.sqrt(np.maximum(s2, 1e-12))


def simulate_campaign(
    n_datasets: int = 16,
    k_true: int = K_TRUE,
    hmax: int = 2,
    seed: int = SEED,
    noise: float = 0.05,
    resolution_noise: bool = False,
    cell: tuple[float, ...] = CELL,
) -> Campaign:
    """Intensities of a helix plus ``k_true`` linearised displacement modes."""
    rng = np.random.default_rng(int(seed))
    sites = helix_sites()
    hkl = half_set(hmax)
    f_mu = structure_factors(sites, hkl, cell)
    modes_all = displacement_modes(sites, hkl)
    k_true = int(min(k_true, modes_all.shape[0]))
    modes = modes_all[:k_true]
    scales = np.array([1.0, 0.6, 0.3], dtype=np.float64)[:k_true]
    z = rng.normal(size=(n_datasets, k_true)) * scales
    z = z - z.mean(axis=0, keepdims=True)
    signal = z.astype(np.complex128) @ modes
    # Complex normal residual, variance growing with resolution when requested.
    d = d_spacing(hkl, cell)
    s2 = 1.0 / np.maximum(d**2, 1e-6)
    sig_f = 0.15 * (1.0 + (4.0 * s2 if resolution_noise else 0.0))
    eps = rng.normal(size=(n_datasets, hkl.shape[0])) + 1j * rng.normal(size=(n_datasets, hkl.shape[0]))
    eps = eps * (sig_f / np.sqrt(2.0))
    f_true = f_mu + signal + eps
    intensity = np.abs(f_true) ** 2
    sig_i = noise * intensity + 0.5
    if resolution_noise:
        sig_i = sig_i * (1.0 + 3.0 * s2)
    i_obs = intensity + rng.normal(scale=sig_i)
    shells = equal_count_shells(d, bin_size=max(4, hkl.shape[0] // 4))
    return Campaign(
        hkl=hkl,
        centric=np.zeros(hkl.shape[0], dtype=bool),
        epsilon=np.ones(hkl.shape[0], dtype=np.float64),
        cell=tuple(float(x) for x in cell),
        f_mu=f_mu,
        modes_true=modes,
        z_true=z,
        f_true=f_true.astype(np.complex128),
        i_obs=i_obs.astype(np.float64),
        sig_i=sig_i.astype(np.float64),
        mask=np.ones(i_obs.shape, dtype=bool),
        shell_index=shells,
        d_spacing=d,
        sites=sites,
    )


def _map_grid(hkl: np.ndarray, grid: Optional[tuple[int, int, int]]) -> tuple[int, int, int]:
    """A grid on which every campaign index is distinct from its Friedel mate."""
    if grid is not None:
        shape = tuple(int(n) for n in grid)
        if len(shape) != 3 or any(n < 4 for n in shape):
            raise ValueError("grid must be three integers >= 4")
        return shape
    hmax = int(np.max(np.abs(hkl))) if len(hkl) else 1
    n = max(2 * hmax + 2, 4)
    return (n, n, n)


def _real_map(hkl: np.ndarray, coeff: np.ndarray, grid: tuple[int, int, int]) -> np.ndarray:
    from phridge.contrib.ensemble.projection import complete_friedel, grid_to_density

    return np.real(grid_to_density(complete_friedel(hkl, coeff, grid)))


def campaign_difference_maps(
    campaign: Campaign,
    grid: Optional[tuple[int, int, int]] = None,
) -> CampaignDifferenceMaps:
    """Inverse-FFT the campaign differences onto a P1 grid.

    One map per dataset (``F_n - F_μ``) and one map per planted mode.
    Linearity of the transform gives ``datasets ≈ z @ modes`` plus the map
    of the complex residual.
    """
    shape = _map_grid(campaign.hkl, grid)
    diff = np.asarray(campaign.f_true - campaign.f_mu, dtype=np.complex128)
    datasets = np.stack([_real_map(campaign.hkl, diff[i], shape) for i in range(diff.shape[0])])
    modes = np.stack(
        [_real_map(campaign.hkl, campaign.modes_true[k], shape) for k in range(campaign.modes_true.shape[0])]
    )
    return CampaignDifferenceMaps(grid=shape, datasets=datasets.astype(np.float64), modes=modes.astype(np.float64))


def write_difference_maps(path: str, maps: CampaignDifferenceMaps) -> None:
    """Write the real-space maps as ``datasets``, ``modes``, and ``grid``."""
    np.savez(
        path,
        datasets=maps.datasets,
        modes=maps.modes,
        grid=np.asarray(maps.grid, dtype=np.int32),
    )


def campaign_with_engine(hmax: int = 1, n_atoms: int = 4) -> np.ndarray:
    """Structure factors through ``multixtal.structure.f_calc_s_g``.

    Requires the torch structure-factor engine. The scattering model is a
    short helix of carbon scatterers, the same construction
    ``multixtal.synthetic`` uses for its benchmark atoms.
    """
    from phridge.legacy.multixtal.structure import f_calc_s_g
    from phridge.sfcalc.engine.engine import ScatteringModel
    from phridge.sfcalc.engine.symmetry import identity_ops

    sites = helix_sites(n_atoms)
    hkl = half_set(hmax)
    rot, trans = identity_ops()
    n = sites.shape[0]
    model = ScatteringModel(
        unit_cell=CELL,
        sites_frac=sites,
        occupancy=np.ones(n),
        u_iso=np.full(n, 0.05),
        u_star=np.zeros((n, 6)),
        anisotropic=np.zeros(n, dtype=bool),
        fp=np.zeros(n),
        fdp=np.zeros(n),
        type_index=np.zeros(n, dtype=np.int64),
        gauss_a=np.array([[2.31, 1.02, 1.59, 0.87]]),
        gauss_b=np.array([[20.84, 10.21, 0.57, 51.65]]),
        gauss_c=np.array([0.22]),
        rot=rot,
        trans=trans,
    )
    f_full, _s, _g = f_calc_s_g(model, hkl)
    return np.asarray(f_full, dtype=np.complex128)
