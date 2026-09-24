"""Macrocycle schedule (spec §10 Alternation) and diagnostics.

Atoms and ADPs are refined with (w_j, U_j) frozen. Field coefficients
(or the Fisher U_j) are refined with the atoms frozen. Evaluating λ(x_j)
live during the coordinate step would add a spurious force along ∇λ and
is avoided.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from phridge.contrib.spatial_sigmaa_v2.fields import FieldBasis, evaluate_fields
from phridge.contrib.spatial_sigmaa_v2.fisher import iterate_fisher_closure
from phridge.contrib.spatial_sigmaa_v2.gradients import (
    add_atom_grads,
    atom_to_field_params,
    d_d0_and_sigma_miss,
    field_gradients,
    mean_gradients_direct,
    variance_gradients_iso,
)
from phridge.contrib.spatial_sigmaa_v2.moments import moments, resolution_s2
from phridge.contrib.spatial_sigmaa_v2.options import SpatialSigmaAV2Options
from phridge.contrib.spatial_sigmaa_v2.packing import PackedSpatialSigmaAV2, PackedSpatialSigmaAV2Result
from phridge.contrib.spatial_sigmaa_v2.per_atom import EIGHT_PI2, iso_u_star, luzzati_d_iso
from phridge.contrib.spatial_sigmaa_v2.regulariser import regulariser
from phridge.contrib.spatial_sigmaa_v2.target import rice_nll_intensity
from phridge.sfcalc.engine.cell import reciprocal_cartesian
from phridge.sfcalc.engine.engine import ScatteringModel


@dataclass
class FrozenAtomErrors:
    """(w_j, U_j) captured at the field/Fisher step. Do not re-evaluate λ(x)."""

    w: np.ndarray
    u_err_iso: np.ndarray
    sites_frac: np.ndarray

    def matches_sites(self, sites_frac: np.ndarray, *, atol: float = 1e-12) -> bool:
        return np.allclose(self.sites_frac, np.asarray(sites_frac, dtype=np.float64), atol=atol)


def freeze_atom_errors(
    basis: FieldBasis,
    sites_frac: np.ndarray,
    lambda_c: np.ndarray,
    kappa_c: np.ndarray,
) -> FrozenAtomErrors:
    """w_j = exp(λ_j), U_j = κ_j/(8π²) at the current (frozen) sites."""
    sites = np.asarray(sites_frac, dtype=np.float64).reshape(-1, 3)
    lam, kap = evaluate_fields(sites, lambda_c, kappa_c, basis)
    return FrozenAtomErrors(
        w=np.exp(np.asarray(lam, dtype=np.float64)),
        u_err_iso=np.asarray(kap, dtype=np.float64) / EIGHT_PI2,
        sites_frac=sites.copy(),
    )


def _shell_edges(s2: np.ndarray, n_shells: int) -> np.ndarray:
    ss = np.sort(np.asarray(s2, dtype=np.float64).reshape(-1))
    if ss.size == 0 or n_shells <= 0:
        return np.zeros((0,), dtype=np.float64)
    qs = np.linspace(0.0, 1.0, n_shells + 1)
    edges = np.quantile(ss, qs)
    edges[0] = min(float(edges[0]), float(ss[0]))
    edges[-1] = max(float(edges[-1]), float(ss[-1]) + 1e-12)
    return edges


def _v_k(basis: FieldBasis, scale: float) -> np.ndarray:
    if basis.n_coeff == 0:
        return np.zeros((0,), dtype=np.float64)
    dstar = np.linalg.norm(reciprocal_cartesian(basis.unit_cell, basis.hkl), axis=1)
    return np.asarray(scale, dtype=np.float64) / (1.0 + (dstar * float(basis.cutoff)) ** 2)


def initialise_block(
    model: ScatteringModel,
    hkl: np.ndarray,
    block: PackedSpatialSigmaAV2,
    *,
    n_shells: int = 4,
) -> tuple[PackedSpatialSigmaAV2, FieldBasis]:
    """Expand an empty Phase-1 payload to a usable basis and shell table."""
    basis = FieldBasis.build(model.unit_cell, model.rot, model.trans, block.meta.field_cutoff)
    s2 = resolution_s2(model.unit_cell, hkl)
    if block.meta.n_coeff == basis.n_coeff and block.meta.n_shells > 0:
        return block, basis
    edges = _shell_edges(s2, n_shells)
    d0 = np.ones(n_shells, dtype=np.float64)
    # Floor so Rice is non-singular even with w=1, U=0.
    miss = np.full(n_shells, 1.0, dtype=np.float64)
    v = _v_k(basis, block.meta.spectral_taper_scale)
    packed = PackedSpatialSigmaAV2(
        lambda_c=np.zeros(basis.n_coeff),
        kappa_c=np.zeros(basis.n_coeff),
        coeff_hkl=basis.pack_hkl(),
        D0=d0,
        Sigma_miss=miss,
        shell_s2_edges=edges,
        v_k=v,
        enabled=True,
        field_cutoff=block.meta.field_cutoff,
        entropy_weight=block.meta.entropy_weight,
        spectral_taper_scale=block.meta.spectral_taper_scale,
        fisher=block.meta.fisher,
        n_scatterers=model.n_scatterers,
    )
    return packed, basis


def _d0_miss_per_h(block: PackedSpatialSigmaAV2, s2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from phridge.contrib.spatial_sigmaa_v2.per_atom import interpolate_shell

    d0 = interpolate_shell(s2, block.shell_s2_edges, block.D0, 1.0)
    miss = interpolate_shell(s2, block.shell_s2_edges, block.Sigma_miss, 1.0)
    return d0, np.maximum(miss, 1e-8)


def _objective(
    model: ScatteringModel,
    hkl: np.ndarray,
    intensity: np.ndarray,
    w: np.ndarray,
    u_err: np.ndarray,
    d0: np.ndarray,
    miss: np.ndarray,
    block: PackedSpatialSigmaAV2,
    basis: FieldBasis,
    *,
    epsilon: Optional[np.ndarray],
    centric: Optional[np.ndarray],
) -> tuple[float, object, object, object]:
    fe, sig = moments(model, hkl, w, u_err, d0, miss)
    rice = rice_nll_intensity(intensity, fe, sig, epsilon=epsilon, centric=centric)
    d_j = d0.reshape(1, -1) * w.reshape(-1, 1) * luzzati_d_iso(u_err, resolution_s2(model.unit_cell, hkl))
    reg = regulariser(
        model,
        hkl,
        d_j,
        block.lambda_c,
        block.kappa_c,
        block.v_k,
        basis=basis,
        sites_frac=model.sites_frac,
        shell_s2_edges=block.shell_s2_edges,
        entropy_weight=block.meta.entropy_weight,
        spectral_taper_scale=block.meta.spectral_taper_scale,
    )
    return rice.value + reg.value, rice, reg, d_j


def field_step(
    model: ScatteringModel,
    hkl: np.ndarray,
    intensity: np.ndarray,
    block: PackedSpatialSigmaAV2,
    *,
    epsilon: Optional[np.ndarray] = None,
    centric: Optional[np.ndarray] = None,
    n_steps: int = 10,
    step0: float = 0.05,
) -> tuple[PackedSpatialSigmaAV2, FrozenAtomErrors]:
    """Refine (λ, κ) with atoms frozen. Spec §10 alternation, field half."""
    block, basis = initialise_block(model, hkl, block)
    s2 = resolution_s2(model.unit_cell, hkl)
    sites = np.asarray(model.sites_frac, dtype=np.float64)
    for _ in range(int(n_steps)):
        lam, kap = evaluate_fields(sites, block.lambda_c, block.kappa_c, basis)
        lam = np.clip(np.asarray(lam, dtype=np.float64), -3.0, 3.0)
        kap = np.clip(np.asarray(kap, dtype=np.float64), -8.0, 40.0)
        w = np.exp(lam)
        u_err = np.maximum(kap / EIGHT_PI2, 0.0)
        d0, miss = _d0_miss_per_h(block, s2)
        L0, rice, reg, d_j = _objective(
            model, hkl, intensity, w, u_err, d0, miss, block, basis, epsilon=epsilon, centric=centric
        )
        tot = add_atom_grads(
            mean_gradients_direct(model, hkl, d_j, rice.d_f_eff_star, w),
            variance_gradients_iso(model, hkl, w, u_err, rice.d_sigma),
        )
        d_lam, d_kap = atom_to_field_params(tot, w)
        g_l, g_k = field_gradients(basis, sites, d_lam, d_kap)
        g_l = g_l + reg.d_lambda_c
        g_k = g_k + reg.d_kappa_c
        gn = max(float(np.linalg.norm(np.concatenate([g_l, g_k]))), 1.0)
        g_l = g_l / gn
        g_k = g_k / gn
        accepted = False
        for shrink in (1.0, 0.4, 0.15, 0.05, 0.01):
            trial = PackedSpatialSigmaAV2(
                lambda_c=np.clip(block.lambda_c - shrink * step0 * g_l, -1.5, 1.5),
                kappa_c=np.clip(block.kappa_c - shrink * step0 * g_k, -8.0, 20.0),
                coeff_hkl=block.coeff_hkl,
                D0=block.D0,
                Sigma_miss=block.Sigma_miss,
                shell_s2_edges=block.shell_s2_edges,
                v_k=block.v_k,
                enabled=True,
                field_cutoff=block.meta.field_cutoff,
                entropy_weight=block.meta.entropy_weight,
                spectral_taper_scale=block.meta.spectral_taper_scale,
                fisher=False,
                n_scatterers=model.n_scatterers,
            )
            lj, kj = evaluate_fields(sites, trial.lambda_c, trial.kappa_c, basis)
            lj = np.clip(np.asarray(lj, dtype=np.float64), -3.0, 3.0)
            kj = np.clip(np.asarray(kj, dtype=np.float64), -8.0, 40.0)
            wt = np.exp(lj)
            ut = np.maximum(kj / EIGHT_PI2, 0.0)
            Lt, _, _, _ = _objective(
                model, hkl, intensity, wt, ut, d0, miss, trial, basis, epsilon=epsilon, centric=centric
            )
            if Lt < L0:
                block = trial
                accepted = True
                break
        if not accepted:
            break
    frozen = freeze_atom_errors(basis, sites, block.lambda_c, block.kappa_c)
    return block, frozen


def fisher_step(
    model: ScatteringModel,
    hkl: np.ndarray,
    block: PackedSpatialSigmaAV2,
    *,
    epsilon: Optional[np.ndarray] = None,
    u_init: Optional[np.ndarray] = None,
    n_iter: int = 6,
) -> tuple[PackedSpatialSigmaAV2, FrozenAtomErrors]:
    """Per-atom U_j from the Fisher closure; fields stay at zero (spec §9)."""
    n = model.n_scatterers
    u0 = np.full(n, 0.05) if u_init is None else np.asarray(u_init, dtype=np.float64).reshape(-1)
    s2 = resolution_s2(model.unit_cell, hkl)
    d0, miss = _d0_miss_per_h(block, s2) if block.meta.n_shells > 0 else (np.ones(hkl.shape[0]), np.full(hkl.shape[0], 1.0))
    u, _ = iterate_fisher_closure(model, hkl, u0, w=np.ones(n), d0=d0, sigma_miss=miss, epsilon=epsilon, n_iter=n_iter)
    # Zero fields: the Fisher U is the default; λ, κ carry nothing beyond it.
    packed = PackedSpatialSigmaAV2(
        lambda_c=np.zeros(block.meta.n_coeff),
        kappa_c=np.zeros(block.meta.n_coeff),
        coeff_hkl=block.coeff_hkl,
        D0=block.D0 if block.meta.n_shells else np.ones(1),
        Sigma_miss=block.Sigma_miss if block.meta.n_shells else np.ones(1),
        shell_s2_edges=block.shell_s2_edges if block.meta.n_shells else np.array([0.0, 1.0]),
        v_k=block.v_k if block.meta.n_coeff else np.zeros(0),
        enabled=True,
        field_cutoff=block.meta.field_cutoff,
        entropy_weight=block.meta.entropy_weight,
        spectral_taper_scale=block.meta.spectral_taper_scale,
        fisher=True,
        n_scatterers=n,
    )
    if packed.meta.n_shells == 1 and block.meta.n_shells == 0:
        pass
    frozen = FrozenAtomErrors(w=np.ones(n), u_err_iso=u, sites_frac=np.asarray(model.sites_frac, dtype=np.float64).copy())
    return packed, frozen


def diagnostics(
    model: ScatteringModel,
    hkl: np.ndarray,
    intensity: np.ndarray,
    block: PackedSpatialSigmaAV2,
    frozen: FrozenAtomErrors,
    *,
    basis: Optional[FieldBasis] = None,
    epsilon: Optional[np.ndarray] = None,
    centric: Optional[np.ndarray] = None,
) -> PackedSpatialSigmaAV2Result:
    """Field-coefficient grads plus per-atom w / U for the wire result."""
    s2 = resolution_s2(model.unit_cell, hkl)
    d0, miss = _d0_miss_per_h(block, s2)
    w, u_err = frozen.w, frozen.u_err_iso
    fe, sig = moments(model, hkl, w, u_err, d0, miss)
    rice = rice_nll_intensity(intensity, fe, sig, epsilon=epsilon, centric=centric)
    d_j = d0.reshape(1, -1) * w.reshape(-1, 1) * luzzati_d_iso(u_err, s2)
    tot = add_atom_grads(
        mean_gradients_direct(model, hkl, d_j, rice.d_f_eff_star, w),
        variance_gradients_iso(model, hkl, w, u_err, rice.d_sigma),
    )
    d_lam, d_kap = atom_to_field_params(tot, w)
    if basis is None:
        basis = FieldBasis.build(model.unit_cell, model.rot, model.trans, block.meta.field_cutoff)
    if basis.n_coeff == block.meta.n_coeff and basis.n_coeff > 0:
        d_lc, d_kc = field_gradients(basis, model.sites_frac, d_lam, d_kap)
    else:
        d_lc = np.zeros(block.meta.n_coeff)
        d_kc = np.zeros(block.meta.n_coeff)
    d_d0_h, d_miss_h = d_d0_and_sigma_miss(model, hkl, w, u_err, d0, rice.d_f_eff_star, rice.d_sigma, fe)
    # Bin per-reflection grads back to the shell table.
    n_s = int(block.meta.n_shells)
    d_d0 = np.zeros(n_s)
    d_miss = np.zeros(n_s)
    if n_s > 0:
        idx = np.clip(np.searchsorted(block.shell_s2_edges, s2, side="right") - 1, 0, n_s - 1)
        for s in range(n_s):
            sel = idx == s
            if np.any(sel):
                d_d0[s] = float(np.sum(d_d0_h[sel]))
                d_miss[s] = float(np.sum(d_miss_h[sel]))
    u_star = iso_u_star(model.unit_cell, u_err)
    return PackedSpatialSigmaAV2Result(
        d_lambda_c=d_lc,
        d_kappa_c=d_kc,
        d_D0=d_d0 if d_d0.size else np.zeros(max(block.meta.n_shells, 0)),
        d_Sigma_miss=d_miss if d_miss.size else np.zeros(max(block.meta.n_shells, 0)),
        w=w,
        u_err_star=u_star,
        tr_U=3.0 * u_err,
    )


def run_macrocycle_step(
    model: ScatteringModel,
    hkl: np.ndarray,
    intensity: np.ndarray,
    block: PackedSpatialSigmaAV2,
    opts: SpatialSigmaAV2Options,
    *,
    epsilon: Optional[np.ndarray] = None,
    centric: Optional[np.ndarray] = None,
) -> tuple[PackedSpatialSigmaAV2, PackedSpatialSigmaAV2Result, FrozenAtomErrors]:
    """One field or Fisher half-step. Atoms stay where they are."""
    if opts.fisher or block.meta.fisher:
        packed, frozen = fisher_step(model, hkl, block, epsilon=epsilon)
    else:
        packed, frozen = field_step(model, hkl, intensity, block, epsilon=epsilon, centric=centric)
    result = diagnostics(model, hkl, intensity, packed, frozen, epsilon=epsilon, centric=centric)
    return packed, result, frozen
