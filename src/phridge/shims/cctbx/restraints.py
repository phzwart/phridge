"""Geometry and ADP restraint adapter."""

from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np
from pydantic import BaseModel, model_validator
from scipy.spatial import cKDTree

from phridge.contracts.packing import PackedRestraints
from phridge.shims.cctbx._flex import double
from phridge.shims.cctbx.symmetry import crystal_from_cctbx


from phridge.contracts.manifest import PluginManifest, Slot, Space, Tier

MANIFEST = PluginManifest(
    name='cctbx_restraints',
    slot=Slot.io_adapter,
    space=Space.D,
    tier=Tier.hybrid,
)

def _i_seqs_array(proxies: Iterable[Any], width: int) -> tuple[np.ndarray, list[Any]]:
    rows = []
    items = list(proxies) if proxies is not None else []
    for proxy in items:
        seq = tuple(int(i) for i in proxy.i_seqs)
        if len(seq) != width:
            raise ValueError(f"expected i_seqs width {width}, got {len(seq)}")
        rows.append(seq)
    if not rows:
        return np.zeros((0, width), dtype=np.int32), items
    return np.asarray(rows, dtype=np.int32), items


def _csr(seq_lists: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    offsets = [0]
    flat: list[int] = []
    for seq in seq_lists:
        flat.extend(int(i) for i in seq)
        offsets.append(len(flat))
    return np.asarray(flat, dtype=np.int32), np.asarray(offsets, dtype=np.int32)


def restraints_from_cctbx(manager: Any, n_sites: Optional[int] = None) -> PackedRestraints:
    """Pack cctbx.geometry_restraints.manager (or an object with the same proxy attrs)."""
    if n_sites is None:
        n_sites = None
        if hasattr(manager, "bond_params_table") and manager.bond_params_table is not None:
            n_sites = int(manager.bond_params_table.size())
        elif getattr(manager, "sites_cart_used_for_pair_proxies", None) is not None:
            n_sites = int(manager.sites_cart_used_for_pair_proxies.size())
    crystal = None
    if getattr(manager, "crystal_symmetry", None) is not None:
        crystal = crystal_from_cctbx(manager.crystal_symmetry)

    simple, asu = None, None
    if hasattr(manager, "get_all_bond_proxies"):
        pair = manager.get_all_bond_proxies()
        if pair is not None:
            simple, asu = pair
    bond_i, bond_d, bond_w, bond_s, bond_o = [], [], [], [], []
    if simple:
        for proxy in simple:
            bond_i.append(tuple(int(i) for i in proxy.i_seqs))
            bond_d.append(float(proxy.distance_ideal))
            bond_w.append(float(proxy.weight))
            bond_s.append(float(getattr(proxy, "slack", 0.0) or 0.0))
            bond_o.append(int(getattr(proxy, "origin_id", 0) or 0))
    asu_i, asu_d, asu_w, asu_rt = [], [], [], []
    if asu:
        for proxy in asu:
            seq = tuple(int(i) for i in proxy.i_seqs)
            asu_i.append(seq)
            asu_d.append(float(proxy.distance_ideal))
            asu_w.append(float(proxy.weight))
            rt = getattr(proxy, "rt_mx_ji", None)
            asu_rt.append(str(rt) if rt is not None else "x,y,z")

    angles = getattr(manager, "angle_proxies", None) or []
    ang_i, items = _i_seqs_array(angles, 3)
    ang_ideal = np.array([float(p.angle_ideal) for p in items], dtype=np.float64) if items else None
    ang_w = np.array([float(p.weight) for p in items], dtype=np.float64) if items else None
    ang_o = np.array([int(getattr(p, "origin_id", 0) or 0) for p in items], dtype=np.int32) if items else None

    dihedrals = getattr(manager, "dihedral_proxies", None) or []
    dih_i, ditems = _i_seqs_array(dihedrals, 4)
    dih_ideal = np.array([float(p.angle_ideal) for p in ditems], dtype=np.float64) if ditems else None
    dih_w = np.array([float(p.weight) for p in ditems], dtype=np.float64) if ditems else None
    dih_per = np.array([int(getattr(p, "periodicity", 0) or 0) for p in ditems], dtype=np.int32) if ditems else None
    dih_o = np.array([int(getattr(p, "origin_id", 0) or 0) for p in ditems], dtype=np.int32) if ditems else None

    chirs = getattr(manager, "chirality_proxies", None) or []
    ch_i, citems = _i_seqs_array(chirs, 4)
    ch_v = np.array([float(p.volume_ideal) for p in citems], dtype=np.float64) if citems else None
    ch_w = np.array([float(p.weight) for p in citems], dtype=np.float64) if citems else None
    ch_b = np.array([1 if getattr(p, "both_signs", False) else 0 for p in citems], dtype=np.uint8) if citems else None
    ch_o = np.array([int(getattr(p, "origin_id", 0) or 0) for p in citems], dtype=np.int32) if citems else None

    planes = list(getattr(manager, "planarity_proxies", None) or [])
    plane_seqs = [list(p.i_seqs) for p in planes]
    p_flat, p_off = _csr(plane_seqs)
    p_w = np.array([float(np.mean(list(p.weights))) if getattr(p, "weights", None) is not None else 1.0 for p in planes], dtype=np.float64) if planes else None
    p_o = np.array([int(getattr(p, "origin_id", 0) or 0) for p in planes], dtype=np.int32) if planes else None

    parallels = list(getattr(manager, "parallelity_proxies", None) or [])
    pi_seqs = [list(p.i_seqs) for p in parallels]
    pj_seqs = [list(p.j_seqs) for p in parallels]
    pi_flat, pi_off = _csr(pi_seqs)
    pj_flat, pj_off = _csr(pj_seqs)
    par_w = np.array([float(p.weight) for p in parallels], dtype=np.float64) if parallels else None
    par_ang = np.array([float(getattr(p, "target_angle_deg", 0.0) or 0.0) for p in parallels], dtype=np.float64) if parallels else None
    par_o = np.array([int(getattr(p, "origin_id", 0) or 0) for p in parallels], dtype=np.int32) if parallels else None

    nb_i, nb_vdw = [], []
    pair_proxies = None
    if hasattr(manager, "pair_proxies"):
        try:
            pair_proxies = manager.pair_proxies()
        except Exception:
            pair_proxies = None
    if pair_proxies is not None and getattr(pair_proxies, "nonbonded_proxies", None) is not None:
        simple_nb = getattr(pair_proxies.nonbonded_proxies, "simple", None)
        if simple_nb:
            for proxy in simple_nb:
                nb_i.append(tuple(int(i) for i in proxy.i_seqs))
                nb_vdw.append(float(getattr(proxy, "vdw_distance", getattr(proxy, "distance", 0.0)) or 0.0))

    ref_i, ref_xyz, ref_w = [], [], []
    refs = getattr(manager, "reference_coordinate_proxies", None) or []
    for proxy in refs:
        if hasattr(proxy, "i_seqs"):
            ref_i.append(int(proxy.i_seqs[0]))
        else:
            ref_i.append(int(getattr(proxy, "i_seq")))
        site = getattr(proxy, "ref_sites", None) or getattr(proxy, "sites", None) or getattr(proxy, "site", (0, 0, 0))
        if hasattr(site, "__len__") and len(site) and hasattr(site[0], "__len__"):
            site = site[0]
        ref_xyz.append(tuple(float(x) for x in site))
        ref_w.append(float(getattr(proxy, "weight", 1.0) or 1.0))

    sim_pairs: list[tuple[int, int]] = []
    sim_off = [0]
    sim_w: list[float] = []
    sims = getattr(manager, "bond_similarity_proxies", None) or []
    for proxy in sims:
        pairs = list(proxy.i_seqs)
        weights = list(getattr(proxy, "weights", []))
        for k, pair in enumerate(pairs):
            sim_pairs.append((int(pair[0]), int(pair[1])))
            sim_w.append(float(weights[k]) if k < len(weights) else 1.0)
        sim_off.append(len(sim_pairs))

    model_indices = None
    if getattr(manager, "model_indices", None) is not None:
        model_indices = np.asarray(list(manager.model_indices), dtype=np.int32)
    conformer_indices = None
    if getattr(manager, "conformer_indices", None) is not None:
        conformer_indices = np.asarray(list(manager.conformer_indices), dtype=np.int32)

    if n_sites is None:
        n_sites = 0
        for arr in (bond_i, asu_i, ang_i.tolist(), dih_i.tolist(), ch_i.tolist(), plane_seqs, ref_i):
            if arr is None or len(arr) == 0:
                continue
            flat = np.asarray(arr).reshape(-1)
            if flat.size:
                n_sites = max(n_sites, int(flat.max()) + 1)

    return PackedRestraints(
        n_sites=n_sites or 0,
        crystal=crystal,
        bond_i_seqs=np.asarray(bond_i, dtype=np.int32) if bond_i else None,
        bond_distance_ideal=np.asarray(bond_d, dtype=np.float64) if bond_d else None,
        bond_weight=np.asarray(bond_w, dtype=np.float64) if bond_w else None,
        bond_slack=np.asarray(bond_s, dtype=np.float64) if bond_s else None,
        bond_origin_id=np.asarray(bond_o, dtype=np.int32) if bond_o else None,
        bond_asu_i_seqs=np.asarray(asu_i, dtype=np.int32) if asu_i else None,
        bond_asu_distance_ideal=np.asarray(asu_d, dtype=np.float64) if asu_d else None,
        bond_asu_weight=np.asarray(asu_w, dtype=np.float64) if asu_w else None,
        bond_asu_rt_mx=asu_rt,
        angle_i_seqs=ang_i if ang_i.size else None,
        angle_ideal=ang_ideal,
        angle_weight=ang_w,
        angle_origin_id=ang_o,
        dihedral_i_seqs=dih_i if dih_i.size else None,
        dihedral_angle_ideal=dih_ideal,
        dihedral_weight=dih_w,
        dihedral_periodicity=dih_per,
        dihedral_origin_id=dih_o,
        chirality_i_seqs=ch_i if ch_i.size else None,
        chirality_volume_ideal=ch_v,
        chirality_weight=ch_w,
        chirality_both_signs=ch_b,
        chirality_origin_id=ch_o,
        planarity_i_seqs=p_flat if planes else None,
        planarity_offsets=p_off if planes else None,
        planarity_weight=p_w,
        planarity_origin_id=p_o,
        parallelity_i_seqs=pi_flat if parallels else None,
        parallelity_i_offsets=pi_off if parallels else None,
        parallelity_j_seqs=pj_flat if parallels else None,
        parallelity_j_offsets=pj_off if parallels else None,
        parallelity_weight=par_w,
        parallelity_target_angle=par_ang,
        parallelity_origin_id=par_o,
        nonbonded_i_seqs=np.asarray(nb_i, dtype=np.int32) if nb_i else None,
        nonbonded_vdw=np.asarray(nb_vdw, dtype=np.float64) if nb_vdw else None,
        refcoord_i_seq=np.asarray(ref_i, dtype=np.int32) if ref_i else None,
        refcoord_xyz=np.asarray(ref_xyz, dtype=np.float64) if ref_xyz else None,
        refcoord_weight=np.asarray(ref_w, dtype=np.float64) if ref_w else None,
        bondsim_pair_i_seqs=np.asarray(sim_pairs, dtype=np.int32) if sim_pairs else None,
        bondsim_offsets=np.asarray(sim_off, dtype=np.int32) if sim_pairs else None,
        bondsim_weight=np.asarray(sim_w, dtype=np.float64) if sim_w else None,
        model_indices=model_indices,
        conformer_indices=conformer_indices,
    )


def restraints_to_proxies(packed: PackedRestraints) -> dict[str, list[Any]]:
    """Rebuild cctbx proxy objects from packed tables (not a full manager)."""
    from cctbx.geometry_restraints import (
        angle_proxy,
        bond_simple_proxy,
        chirality_proxy,
        dihedral_proxy,
        parallelity_proxy,
        planarity_proxy,
    )
    
    bonds = []
    for i, seq in enumerate(packed.bond_i_seqs):
        bonds.append(
            bond_simple_proxy(
                i_seqs=tuple(int(x) for x in seq),
                distance_ideal=float(packed.bond_distance_ideal[i]),
                weight=float(packed.bond_weight[i]),
                slack=float(packed.bond_slack[i]) if packed.bond_slack.size else 0,
                origin_id=int(packed.bond_origin_id[i]) if packed.bond_origin_id.size else 0,
            )
        )
    angles = []
    for i, seq in enumerate(packed.angle_i_seqs):
        angles.append(
            angle_proxy(
                i_seqs=tuple(int(x) for x in seq),
                angle_ideal=float(packed.angle_ideal[i]),
                weight=float(packed.angle_weight[i]),
                origin_id=int(packed.angle_origin_id[i]) if packed.angle_origin_id.size else 0,
            )
        )
    dihedrals = []
    for i, seq in enumerate(packed.dihedral_i_seqs):
        dihedrals.append(
            dihedral_proxy(
                i_seqs=tuple(int(x) for x in seq),
                angle_ideal=float(packed.dihedral_angle_ideal[i]),
                weight=float(packed.dihedral_weight[i]),
                periodicity=int(packed.dihedral_periodicity[i]) if packed.dihedral_periodicity.size else 0,
                origin_id=int(packed.dihedral_origin_id[i]) if packed.dihedral_origin_id.size else 0,
            )
        )
    chirs = []
    for i, seq in enumerate(packed.chirality_i_seqs):
        chirs.append(
            chirality_proxy(
                i_seqs=tuple(int(x) for x in seq),
                volume_ideal=float(packed.chirality_volume_ideal[i]),
                weight=float(packed.chirality_weight[i]),
                both_signs=bool(packed.chirality_both_signs[i]) if packed.chirality_both_signs.size else False,
                origin_id=int(packed.chirality_origin_id[i]) if packed.chirality_origin_id.size else 0,
            )
        )
    planes = []
    off = packed.planarity_offsets
    for p in range(len(off) - 1):
        seq = packed.planarity_i_seqs[off[p] : off[p + 1]]
        w = float(packed.planarity_weight[p]) if packed.planarity_weight.size else 1.0
        weights = double([w] * len(seq))
        planes.append(
            planarity_proxy(
                i_seqs=tuple(int(x) for x in seq),
                weights=weights,
                origin_id=int(packed.planarity_origin_id[p]) if packed.planarity_origin_id.size else 0,
            )
        )
    parallels = []
    ioff, joff = packed.parallelity_i_offsets, packed.parallelity_j_offsets
    for p in range(len(ioff) - 1):
        iseq = packed.parallelity_i_seqs[ioff[p] : ioff[p + 1]]
        jseq = packed.parallelity_j_seqs[joff[p] : joff[p + 1]]
        parallels.append(
            parallelity_proxy(
                i_seqs=tuple(int(x) for x in iseq),
                j_seqs=tuple(int(x) for x in jseq),
                weight=float(packed.parallelity_weight[p]) if packed.parallelity_weight.size else 1.0,
                target_angle_deg=float(packed.parallelity_target_angle[p]) if packed.parallelity_target_angle.size else 0.0,
                origin_id=int(packed.parallelity_origin_id[p]) if packed.parallelity_origin_id.size else 0,
            )
        )
    return {
        "bonds": bonds,
        "angles": angles,
        "dihedrals": dihedrals,
        "chiralities": chirs,
        "planarities": planes,
        "parallelities": parallels,
    }


def _check_i_seq_labels(labels: list[Any], n: int, kind: str) -> None:
    if len(labels) != n:
        raise ValueError(f"{kind}: expected {n} labels, got {len(labels)}")
    for i, item in enumerate(labels):
        if int(item.i) != i:
            raise ValueError(f"{kind}: {kind}[{i}].i is {item.i}, expected {i} (i_seq identity)")


def _restraint_i_seqs(packed: PackedRestraints) -> np.ndarray:
    chunks = [
        packed.bond_i_seqs.reshape(-1),
        packed.bond_asu_i_seqs.reshape(-1),
        packed.angle_i_seqs.reshape(-1),
        packed.dihedral_i_seqs.reshape(-1),
        packed.chirality_i_seqs.reshape(-1),
        packed.planarity_i_seqs.reshape(-1),
        packed.parallelity_i_seqs.reshape(-1),
        packed.parallelity_j_seqs.reshape(-1),
        packed.nonbonded_i_seqs.reshape(-1),
        packed.refcoord_i_seq.reshape(-1),
        packed.bondsim_pair_i_seqs.reshape(-1),
    ]
    nonempty = [c for c in chunks if c.size]
    if not nonempty:
        return np.zeros((0,), dtype=np.int32)
    return np.concatenate(nonempty)


def model_geometry(
    *,
    hierarchy: Any = None,
    xray: Any = None,
    restraints: Any = None,
) -> Any:
    """Assert one i_seq space across hierarchy, xray scatterers, and restraints."""
    from phridge.contracts.models import ModelGeometry
    from phridge.contracts.packing import PackedRestraints
    from phridge.contracts.packing import PackedHierarchy, PackedXray

    sizes: list[int] = []
    if hierarchy is not None:
        if not isinstance(hierarchy, PackedHierarchy):
            raise TypeError("hierarchy must be PackedHierarchy")
        _check_i_seq_labels(hierarchy.meta.atoms, hierarchy.meta.n_atoms, "Hierarchy.atoms")
        sizes.append(hierarchy.meta.n_atoms)
    if xray is not None:
        if not isinstance(xray, PackedXray):
            raise TypeError("xray must be PackedXray")
        _check_i_seq_labels(xray.meta.scatterers, xray.meta.n_scatterers, "XrayStructure.scatterers")
        sizes.append(xray.meta.n_scatterers)
    if restraints is not None:
        if not isinstance(restraints, PackedRestraints):
            raise TypeError("restraints must be PackedRestraints")
        sizes.append(restraints.n_sites)
        seqs = _restraint_i_seqs(restraints)
        if seqs.size and (seqs.min() < 0 or seqs.max() >= restraints.n_sites):
            raise ValueError("restraint i_seq out of range for n_sites")
    if not sizes:
        raise ValueError("model_geometry needs at least one of hierarchy, xray, restraints")
    if len(set(sizes)) != 1:
        raise ValueError(f"model–geometry correspondence: n_sites mismatch {sizes}")
    return ModelGeometry(
        n_sites=sizes[0],
        has_hierarchy=hierarchy is not None,
        has_xray=xray is not None,
        has_restraints=restraints is not None,
        i_seq_identity=True,
    )

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
    importing this module remains strictly torch-free.
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
