"""Two-domain synthetic crystal (spec §11.6) and null calibration (§11.8).

P2_12_12_1 so the set includes centric reflections. Domain B can be
displaced, down-weighted, or error-inflated; the uniform-error control
applies the same perturbation to every atom.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from phridge.contrib.spatial_sigmaa_v2.alternate import field_step, initialise_block
from phridge.contrib.spatial_sigmaa_v2.fields import FieldBasis, evaluate_fields
from phridge.contrib.spatial_sigmaa_v2.moments import f_eff_direct, moments, resolution_s2
from phridge.contrib.spatial_sigmaa_v2.packing import PackedSpatialSigmaAV2
from phridge.contrib.spatial_sigmaa_v2.per_atom import EIGHT_PI2
from phridge.contrib.spatial_sigmaa_v2.target import draw_circular_normal, rice_nll_intensity
from phridge.sfcalc.engine.engine import ScatteringModel
from phridge.sfcalc.engine.symmetry import identity_ops

P212121_ROT = np.array(
    [
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
    ]
)
P212121_TRANS = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [0.5, 0.5, 0.0],
    ]
)

CELL_P212121 = (48.0, 40.0, 56.0, 90.0, 90.0, 90.0)


def two_domain_sites(n_each: int = 6, *, rng: np.random.Generator, jitter: float = 0.035) -> tuple[np.ndarray, np.ndarray]:
    """Two compact blobs. Returns (sites, domain_b_mask)."""
    a = np.array([0.18, 0.20, 0.22]) + rng.uniform(-jitter, jitter, size=(n_each, 3))
    b = np.array([0.72, 0.70, 0.68]) + rng.uniform(-jitter, jitter, size=(n_each, 3))
    sites = np.clip(np.vstack([a, b]), 0.04, 0.96)
    mask_b = np.zeros(2 * n_each, dtype=bool)
    mask_b[n_each:] = True
    return sites, mask_b


def two_domain_model(
    sites: np.ndarray,
    *,
    occ: Optional[np.ndarray] = None,
    u_iso: Optional[np.ndarray] = None,
    cell: tuple = CELL_P212121,
    f: float = 6.0,
    space_group: str = "P212121",
) -> ScatteringModel:
    n = sites.shape[0]
    if space_group == "P212121":
        rot, trans = P212121_ROT, P212121_TRANS
    else:
        rot, trans = identity_ops()
    return ScatteringModel(
        unit_cell=tuple(float(x) for x in cell),
        sites_frac=np.asarray(sites, dtype=np.float64),
        occupancy=np.ones(n) if occ is None else np.asarray(occ, dtype=np.float64),
        u_iso=np.full(n, 0.02) if u_iso is None else np.asarray(u_iso, dtype=np.float64),
        u_star=np.zeros((n, 6)),
        anisotropic=np.zeros(n, dtype=bool),
        fp=np.zeros(n),
        fdp=np.zeros(n),
        type_index=np.zeros(n, dtype=int),
        gauss_a=np.zeros((1, 1)),
        gauss_b=np.zeros((1, 1)),
        gauss_c=np.array([float(f)]),
        rot=np.asarray(rot, dtype=np.float64),
        trans=np.asarray(trans, dtype=np.float64),
    )


def perturb_domain(
    model: ScatteringModel,
    mask: np.ndarray,
    *,
    occ_scale: float = 0.35,
    u_extra: float = 0.05,
    shift: tuple[float, float, float] = (0.025, 0.0, 0.0),
    uniform: bool = False,
) -> ScatteringModel:
    """Displaced / partly deleted / error-inflated. ``uniform`` applies to every atom."""
    sel = np.ones(model.n_scatterers, dtype=bool) if uniform else np.asarray(mask, dtype=bool)
    sites = np.asarray(model.sites_frac, dtype=np.float64).copy()
    occ = np.asarray(model.occupancy, dtype=np.float64).copy()
    u_iso = np.asarray(model.u_iso, dtype=np.float64).copy()
    sites[sel] = (sites[sel] + np.asarray(shift, dtype=np.float64)) % 1.0
    occ[sel] = occ[sel] * float(occ_scale)
    u_iso[sel] = u_iso[sel] + float(u_extra)
    return ScatteringModel(
        unit_cell=model.unit_cell,
        sites_frac=sites,
        occupancy=occ,
        u_iso=u_iso,
        u_star=model.u_star.copy(),
        anisotropic=model.anisotropic.copy(),
        fp=model.fp.copy(),
        fdp=model.fdp.copy(),
        type_index=model.type_index.copy(),
        gauss_a=model.gauss_a,
        gauss_b=model.gauss_b,
        gauss_c=model.gauss_c,
        rot=model.rot,
        trans=model.trans,
        multiplicity=None if model.multiplicity is None else model.multiplicity.copy(),
    )


def miller_box(n: int = 3) -> np.ndarray:
    out = []
    for h in range(-n, n + 1):
        for k in range(-n, n + 1):
            for l in range(-n, n + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                out.append((h, k, l))
    return np.asarray(out, dtype=np.int64)


def intensity_from_model(model: ScatteringModel, hkl: np.ndarray) -> np.ndarray:
    """Noise-free I = |F|² of the (possibly perturbed) structure."""
    d_j = np.ones((model.n_scatterers, hkl.shape[0]))
    return np.abs(f_eff_direct(model, hkl, d_j)) ** 2


def fit_uniform_shells(
    model: ScatteringModel,
    hkl: np.ndarray,
    intensity: np.ndarray,
    block: PackedSpatialSigmaAV2,
) -> PackedSpatialSigmaAV2:
    """Absorb the mean scale into D_0 / Σ_miss (λ, κ stay mean-zero; spec eq. 13)."""
    d_j = np.ones((model.n_scatterers, hkl.shape[0]))
    fc = f_eff_direct(model, hkl, d_j)
    i_c = np.abs(fc) ** 2
    s2 = resolution_s2(model.unit_cell, hkl)
    edges = np.asarray(block.shell_s2_edges, dtype=np.float64)
    n_shells = int(block.D0.shape[0])
    if n_shells == 0 or edges.size != n_shells + 1:
        return block
    d0 = np.ones(n_shells, dtype=np.float64)
    miss = np.ones(n_shells, dtype=np.float64)
    i_obs = np.asarray(intensity, dtype=np.float64).reshape(-1)
    for s in range(n_shells):
        sel = (s2 >= edges[s]) & (s2 < edges[s + 1])
        if s == n_shells - 1:
            sel = (s2 >= edges[s]) & (s2 <= edges[s + 1] + 1e-15)
        if not np.any(sel):
            continue
        denom = float(np.mean(i_c[sel]))
        num = float(np.mean(i_obs[sel]))
        scale2 = num / denom if denom > 1e-12 else 1.0
        d0[s] = float(np.clip(np.sqrt(max(scale2, 0.0)), 0.15, 2.5))
        resid = i_obs[sel] - (d0[s] ** 2) * i_c[sel]
        miss[s] = float(max(np.mean(np.maximum(resid, 0.0)), 0.05 * max(num, 1.0)))
    return PackedSpatialSigmaAV2(
        lambda_c=block.lambda_c,
        kappa_c=block.kappa_c,
        coeff_hkl=block.coeff_hkl,
        D0=d0,
        Sigma_miss=miss,
        shell_s2_edges=block.shell_s2_edges,
        v_k=block.v_k,
        enabled=True,
        field_cutoff=block.meta.field_cutoff,
        entropy_weight=block.meta.entropy_weight,
        spectral_taper_scale=block.meta.spectral_taper_scale,
        fisher=block.meta.fisher,
        n_scatterers=model.n_scatterers,
    )


def rice_at_block(
    model: ScatteringModel,
    hkl: np.ndarray,
    intensity: np.ndarray,
    block: PackedSpatialSigmaAV2,
    basis: FieldBasis,
) -> float:
    from phridge.contrib.spatial_sigmaa_v2.per_atom import interpolate_shell

    s2 = resolution_s2(model.unit_cell, hkl)
    lam, kap = evaluate_fields(model.sites_frac, block.lambda_c, block.kappa_c, basis)
    w = np.exp(np.clip(np.asarray(lam, dtype=np.float64), -3.0, 3.0))
    u = np.maximum(np.asarray(kap, dtype=np.float64) / EIGHT_PI2, 0.0)
    d0 = interpolate_shell(s2, block.shell_s2_edges, block.D0, 1.0)
    miss = np.maximum(interpolate_shell(s2, block.shell_s2_edges, block.Sigma_miss, 1.0), 1e-8)
    fe, sig = moments(model, hkl, w, u, d0, miss)
    return rice_nll_intensity(intensity, fe, sig).value


def likelihood_gain_free_field(
    model: ScatteringModel,
    hkl: np.ndarray,
    intensity: np.ndarray,
    *,
    field_cutoff: float = 16.0,
    n_steps: int = 8,
) -> tuple[float, PackedSpatialSigmaAV2, FieldBasis]:
    """L(uniform) − L(free field). Positive when the free field improves the fit."""
    seed = PackedSpatialSigmaAV2.empty(field_cutoff=field_cutoff)
    block0, basis = initialise_block(model, hkl, seed)
    block0 = fit_uniform_shells(model, hkl, intensity, block0)
    l_uni = rice_at_block(model, hkl, intensity, block0, basis)
    fitted, _ = field_step(model, hkl, intensity, block0, n_steps=n_steps)
    l_free = rice_at_block(model, hkl, intensity, fitted, basis)
    return l_uni - l_free, fitted, basis


def null_calibration(
    model: ScatteringModel,
    hkl: np.ndarray,
    *,
    n_boot: int = 8,
    quantile: float = 0.9,
    field_cutoff: float = 16.0,
    n_steps: int = 6,
    rng: Optional[np.random.Generator] = None,
) -> tuple[float, np.ndarray]:
    """Parametric bootstrap of the free-field gain under a uniform error model (spec §11.8)."""
    rng = np.random.default_rng(0) if rng is None else rng
    w = np.ones(model.n_scatterers)
    u = np.zeros(model.n_scatterers)
    d0 = np.ones(hkl.shape[0])
    miss = np.full(hkl.shape[0], 20.0)
    fe, sig = moments(model, hkl, w, u, d0, miss)
    gains = np.empty(n_boot)
    for i in range(n_boot):
        f = draw_circular_normal(fe, sig, rng)
        intensity = np.abs(f) ** 2
        gain, _fit, _b = likelihood_gain_free_field(
            model, hkl, intensity, field_cutoff=field_cutoff, n_steps=n_steps
        )
        gains[i] = gain
    return float(np.quantile(gains, quantile)), gains
