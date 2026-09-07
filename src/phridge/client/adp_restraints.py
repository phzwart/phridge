"""Client-side ADP restraint pair generation and empirical Bayes hyperparameter fitting.

Constructs 1-2, 1-3, and sphere classes from Cartesian coordinates and bond/angle tables
using scipy.spatial.cKDTree (no cctbx needed).
Fits prior hyperparameters by maximizing the Laplace approximation to the marginal
likelihood of the working set via sparse Cholesky / LU determinants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, Field, model_validator
from scipy.spatial import cKDTree

from phridge.packing_geometry import PackedRestraints


class ADPPriorOptions(BaseModel):
    """Configuration options for the hierarchical scale-invariant ADP prior."""

    w_12: float = 1.0
    tau_12: float = 0.15
    w_13: float = 0.5
    tau_13: float = 0.20
    w_sphere: float = 0.1
    tau_sphere: float = 0.35
    r_sphere: float = 4.5
    w_hirshfeld: float = 10000.0  # 1 / (0.01 A^2)^2
    w_iso: float = 0.05
    tau_iso: float = 0.5
    nu: float = 4.0
    level_weight: float = 1.0
    wilson_b: Optional[float] = None
    allow_large_sphere_weight: bool = False

    @model_validator(mode="after")
    def check_sphere_weight(self) -> "ADPPriorOptions":
        if not self.allow_large_sphere_weight and self.w_sphere > 0.5 * self.w_12:
            raise ValueError(
                f"w_sphere ({self.w_sphere}) must stay <= 0.5 * w_12 ({self.w_12}) "
                "unless allow_large_sphere_weight=True"
            )
        return self


def build_adp_restraints(
    hierarchy: Any,
    restraints: PackedRestraints,
    *,
    options: Optional[ADPPriorOptions] = None,
    wilson_b: Optional[float] = None,
    is_aniso: Optional[np.ndarray] = None,
) -> PackedRestraints:
    """Build ADP prior pair tables (1-2, 1-3, sphere, rigid-bond) without cctbx.

    Parameters
    ----------
    hierarchy : PackedHierarchy or object with xyz or atom coordinates
    restraints : PackedRestraints with bond_i_seqs and angle_i_seqs
    options : ADPPriorOptions
    wilson_b : Wilson B value (overrides options.wilson_b if provided)
    is_aniso : boolean array of shape (N,) indicating anisotropic atoms
    """
    opts = options or ADPPriorOptions()
    if wilson_b is not None:
        opts.wilson_b = float(wilson_b)

    # Extract coordinates
    if hasattr(hierarchy, "xyz"):
        xyz = np.asarray(hierarchy.xyz, dtype=np.float64)
    elif hasattr(hierarchy, "atoms"):
        xyz = np.array([a.xyz for a in hierarchy.atoms], dtype=np.float64)
    elif hasattr(hierarchy, "sites_cart"):
        xyz = np.asarray(hierarchy.sites_cart(), dtype=np.float64)
    else:
        xyz = np.asarray(hierarchy, dtype=np.float64)

    n_sites = int(restraints.n_sites)
    if xyz.shape[0] != n_sites:
        raise ValueError(f"xyz atom count ({xyz.shape[0]}) does not match restraints.n_sites ({n_sites})")

    # Anisotropic flags
    if is_aniso is not None:
        aniso_flags = np.asarray(is_aniso, dtype=bool).reshape(-1)
    elif restraints.adp_is_aniso is not None and restraints.adp_is_aniso.size == n_sites:
        aniso_flags = np.asarray(restraints.adp_is_aniso, dtype=bool)
    else:
        aniso_flags = np.zeros(n_sites, dtype=bool)

    # 1. Bonded 1-2 pairs
    pairs_12: set[tuple[int, int]] = set()
    b_seqs = restraints.bond_i_seqs
    if b_seqs is not None and b_seqs.size > 0:
        for p in b_seqs:
            i, j = int(p[0]), int(p[1])
            if i != j:
                pairs_12.add((min(i, j), max(i, j)))

    # 2. Bonded 1-3 pairs (outer atoms of angles)
    pairs_13: set[tuple[int, int]] = set()
    a_seqs = restraints.angle_i_seqs
    if a_seqs is not None and a_seqs.size > 0:
        for ang in a_seqs:
            i, k = int(ang[0]), int(ang[2])
            if i != k:
                pair = (min(i, k), max(i, k))
                if pair not in pairs_12:
                    pairs_13.add(pair)

    # 3. Sphere class pairs (non-bonded within r_sphere)
    tree = cKDTree(xyz)
    cand_pairs = tree.query_pairs(opts.r_sphere)
    pairs_sphere: list[tuple[int, int, float]] = []  # (i, j, weight)
    for i, j in cand_pairs:
        pair = (min(i, j), max(i, j))
        if pair in pairs_12 or pair in pairs_13:
            continue
        dist = float(np.linalg.norm(xyz[i] - xyz[j]))
        if dist < opts.r_sphere:
            # Kernel w = w_s * (1 - d / r_s)^2
            w = opts.w_sphere * ((1.0 - dist / opts.r_sphere) ** 2)
            pairs_sphere.append((pair[0], pair[1], w))

    # Assemble pair arrays
    pair_list: list[tuple[int, int]] = []
    weight_list: list[float] = []
    tau_list: list[float] = []
    class_list: list[int] = []

    for i, j in sorted(pairs_12):
        pair_list.append((i, j))
        weight_list.append(opts.w_12)
        tau_list.append(opts.tau_12)
        class_list.append(0)

    for i, j in sorted(pairs_13):
        pair_list.append((i, j))
        weight_list.append(opts.w_13)
        tau_list.append(opts.tau_13)
        class_list.append(1)

    for i, j, w in sorted(pairs_sphere, key=lambda x: (x[0], x[1])):
        pair_list.append((i, j))
        weight_list.append(w)
        tau_list.append(opts.tau_sphere)
        class_list.append(2)

    p_arr = np.array(pair_list, dtype=np.int32) if pair_list else np.zeros((0, 2), dtype=np.int32)
    w_arr = np.array(weight_list, dtype=np.float64) if weight_list else np.zeros(0, dtype=np.float64)
    tau_arr = np.array(tau_list, dtype=np.float64) if tau_list else np.zeros(0, dtype=np.float64)
    cls_arr = np.array(class_list, dtype=np.int32) if class_list else np.zeros(0, dtype=np.int32)

    # 4. Hirshfeld rigid bond pairs (bonded pairs with at least one anisotropic atom)
    rigid_list: list[tuple[int, int]] = []
    if opts.w_hirshfeld > 0.0:
        for i, j in sorted(pairs_12):
            if aniso_flags[i] or aniso_flags[j]:
                rigid_list.append((i, j))

    r_arr = np.array(rigid_list, dtype=np.int32) if rigid_list else np.zeros((0, 2), dtype=np.int32)
    rw_arr = np.full(len(rigid_list), opts.w_hirshfeld, dtype=np.float64) if rigid_list else np.zeros(0, dtype=np.float64)

    # Return new PackedRestraints with ADP prior arrays populated
    return PackedRestraints(
        n_sites=n_sites,
        crystal=restraints.meta.crystal,
        bond_i_seqs=restraints.bond_i_seqs,
        bond_distance_ideal=restraints.bond_distance_ideal,
        bond_weight=restraints.bond_weight,
        bond_slack=restraints.bond_slack,
        bond_origin_id=restraints.bond_origin_id,
        bond_asu_i_seqs=restraints.bond_asu_i_seqs,
        bond_asu_distance_ideal=restraints.bond_asu_distance_ideal,
        bond_asu_weight=restraints.bond_asu_weight,
        bond_asu_rt_mx=restraints.bond_asu_rt_mx,
        angle_i_seqs=restraints.angle_i_seqs,
        angle_ideal=restraints.angle_ideal,
        angle_weight=restraints.angle_weight,
        angle_origin_id=restraints.angle_origin_id,
        dihedral_i_seqs=restraints.dihedral_i_seqs,
        dihedral_angle_ideal=restraints.dihedral_angle_ideal,
        dihedral_weight=restraints.dihedral_weight,
        dihedral_periodicity=restraints.dihedral_periodicity,
        dihedral_origin_id=restraints.dihedral_origin_id,
        chirality_i_seqs=restraints.chirality_i_seqs,
        chirality_volume_ideal=restraints.chirality_volume_ideal,
        chirality_weight=restraints.chirality_weight,
        chirality_both_signs=restraints.chirality_both_signs,
        chirality_origin_id=restraints.chirality_origin_id,
        planarity_i_seqs=restraints.planarity_i_seqs,
        planarity_offsets=restraints.planarity_offsets,
        planarity_weight=restraints.planarity_weight,
        planarity_origin_id=restraints.planarity_origin_id,
        parallelity_i_offsets=restraints.parallelity_i_offsets,
        parallelity_i_seqs=restraints.parallelity_i_seqs,
        parallelity_j_offsets=restraints.parallelity_j_offsets,
        parallelity_j_seqs=restraints.parallelity_j_seqs,
        parallelity_weight=restraints.parallelity_weight,
        parallelity_target_angle=restraints.parallelity_target_angle,
        parallelity_origin_id=restraints.parallelity_origin_id,
        nonbonded_i_seqs=restraints.nonbonded_i_seqs,
        nonbonded_vdw=restraints.nonbonded_vdw,
        refcoord_i_seq=restraints.refcoord_i_seq,
        refcoord_xyz=restraints.refcoord_xyz,
        refcoord_weight=restraints.refcoord_weight,
        bondsim_pair_i_seqs=restraints.bondsim_pair_i_seqs,
        bondsim_offsets=restraints.bondsim_offsets,
        bondsim_weight=restraints.bondsim_weight,
        adp_pair_i_seqs=p_arr,
        adp_pair_weight=w_arr,
        adp_pair_tau=tau_arr,
        adp_pair_class=cls_arr,
        rigid_bond_i_seqs=r_arr,
        rigid_bond_weight=rw_arr,
        wilson_b=opts.wilson_b,
        adp_nu=opts.nu,
        adp_level_weight=opts.level_weight,
        adp_is_aniso=aniso_flags.astype(np.uint8),
        model_indices=restraints.model_indices,
        conformer_indices=restraints.conformer_indices,
    )


def fit_adp_hyperparameters(
    prior: Any,
    adp_vec: Any,
    h_xray_diag: Optional[np.ndarray] = None,
    *,
    sites: Optional[Any] = None,
    method: str = "Nelder-Mead",
    maxiter: int = 50,
) -> dict[str, Any]:
    """Fit ADP hyperparameters by maximizing Laplace marginal likelihood approximation.

    Delegates to ``phridge.worker.geometry.adp.fit_adp_hyperparameters`` so that
    importing ``phridge.client.adp_restraints`` remains strictly torch-free.
    """
    from phridge.worker.geometry.adp import fit_adp_hyperparameters as _worker_fit

    return _worker_fit(
        prior,
        adp_vec,
        h_xray_diag=h_xray_diag,
        sites=sites,
        method=method,
        maxiter=maxiter,
    )
