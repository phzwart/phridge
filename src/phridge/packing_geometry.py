"""Packed geometry-restraint tables keyed by i_seq."""

from __future__ import annotations

import io
from typing import Optional

import numpy as np

from phridge.models import CrystalSymmetry, GeometryRestraints
from phridge.packing import as_canonical_float, as_canonical_int


def _savez(**arrays: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def _empty_int(cols: int) -> np.ndarray:
    return np.zeros((0, cols), dtype=np.int32)


def _empty_float() -> np.ndarray:
    return np.zeros((0,), dtype=np.float64)


class PackedRestraints:
    def __init__(
        self,
        n_sites: int,
        *,
        crystal: Optional[CrystalSymmetry] = None,
        bond_i_seqs: Optional[np.ndarray] = None,
        bond_distance_ideal: Optional[np.ndarray] = None,
        bond_weight: Optional[np.ndarray] = None,
        bond_slack: Optional[np.ndarray] = None,
        bond_origin_id: Optional[np.ndarray] = None,
        bond_asu_i_seqs: Optional[np.ndarray] = None,
        bond_asu_distance_ideal: Optional[np.ndarray] = None,
        bond_asu_weight: Optional[np.ndarray] = None,
        bond_asu_rt_mx: Optional[list[str]] = None,
        angle_i_seqs: Optional[np.ndarray] = None,
        angle_ideal: Optional[np.ndarray] = None,
        angle_weight: Optional[np.ndarray] = None,
        angle_origin_id: Optional[np.ndarray] = None,
        dihedral_i_seqs: Optional[np.ndarray] = None,
        dihedral_angle_ideal: Optional[np.ndarray] = None,
        dihedral_weight: Optional[np.ndarray] = None,
        dihedral_periodicity: Optional[np.ndarray] = None,
        dihedral_origin_id: Optional[np.ndarray] = None,
        chirality_i_seqs: Optional[np.ndarray] = None,
        chirality_volume_ideal: Optional[np.ndarray] = None,
        chirality_weight: Optional[np.ndarray] = None,
        chirality_both_signs: Optional[np.ndarray] = None,
        chirality_origin_id: Optional[np.ndarray] = None,
        planarity_i_seqs: Optional[np.ndarray] = None,
        planarity_offsets: Optional[np.ndarray] = None,
        planarity_weight: Optional[np.ndarray] = None,
        planarity_origin_id: Optional[np.ndarray] = None,
        parallelity_i_offsets: Optional[np.ndarray] = None,
        parallelity_i_seqs: Optional[np.ndarray] = None,
        parallelity_j_offsets: Optional[np.ndarray] = None,
        parallelity_j_seqs: Optional[np.ndarray] = None,
        parallelity_weight: Optional[np.ndarray] = None,
        parallelity_target_angle: Optional[np.ndarray] = None,
        parallelity_origin_id: Optional[np.ndarray] = None,
        nonbonded_i_seqs: Optional[np.ndarray] = None,
        nonbonded_vdw: Optional[np.ndarray] = None,
        refcoord_i_seq: Optional[np.ndarray] = None,
        refcoord_xyz: Optional[np.ndarray] = None,
        refcoord_weight: Optional[np.ndarray] = None,
        bondsim_pair_i_seqs: Optional[np.ndarray] = None,
        bondsim_offsets: Optional[np.ndarray] = None,
        bondsim_weight: Optional[np.ndarray] = None,
        adp_pair_i_seqs: Optional[np.ndarray] = None,
        adp_pair_weight: Optional[np.ndarray] = None,
        adp_pair_tau: Optional[np.ndarray] = None,
        adp_pair_class: Optional[np.ndarray] = None,
        rigid_bond_i_seqs: Optional[np.ndarray] = None,
        rigid_bond_weight: Optional[np.ndarray] = None,
        wilson_b: Optional[float] = None,
        adp_nu: Optional[float] = None,
        adp_level_weight: Optional[float] = None,
        adp_is_aniso: Optional[np.ndarray] = None,
        model_indices: Optional[np.ndarray] = None,
        conformer_indices: Optional[np.ndarray] = None,
    ) -> None:
        self.n_sites = int(n_sites)
        self.bond_i_seqs = as_canonical_int(bond_i_seqs) if bond_i_seqs is not None else _empty_int(2)
        self.bond_distance_ideal = (
            as_canonical_float(bond_distance_ideal) if bond_distance_ideal is not None else _empty_float()
        )
        self.bond_weight = as_canonical_float(bond_weight) if bond_weight is not None else _empty_float()
        self.bond_slack = as_canonical_float(bond_slack) if bond_slack is not None else _empty_float()
        self.bond_origin_id = as_canonical_int(bond_origin_id.reshape(-1)) if bond_origin_id is not None else np.zeros((0,), dtype=np.int32)
        self.bond_asu_i_seqs = as_canonical_int(bond_asu_i_seqs) if bond_asu_i_seqs is not None else _empty_int(2)
        self.bond_asu_distance_ideal = (
            as_canonical_float(bond_asu_distance_ideal) if bond_asu_distance_ideal is not None else _empty_float()
        )
        self.bond_asu_weight = as_canonical_float(bond_asu_weight) if bond_asu_weight is not None else _empty_float()
        self.bond_asu_rt_mx = list(bond_asu_rt_mx or [])
        self.angle_i_seqs = as_canonical_int(angle_i_seqs) if angle_i_seqs is not None else _empty_int(3)
        self.angle_ideal = as_canonical_float(angle_ideal) if angle_ideal is not None else _empty_float()
        self.angle_weight = as_canonical_float(angle_weight) if angle_weight is not None else _empty_float()
        self.angle_origin_id = as_canonical_int(angle_origin_id.reshape(-1)) if angle_origin_id is not None else np.zeros((0,), dtype=np.int32)
        self.dihedral_i_seqs = as_canonical_int(dihedral_i_seqs) if dihedral_i_seqs is not None else _empty_int(4)
        self.dihedral_angle_ideal = (
            as_canonical_float(dihedral_angle_ideal) if dihedral_angle_ideal is not None else _empty_float()
        )
        self.dihedral_weight = as_canonical_float(dihedral_weight) if dihedral_weight is not None else _empty_float()
        self.dihedral_periodicity = (
            as_canonical_int(dihedral_periodicity.reshape(-1)) if dihedral_periodicity is not None else np.zeros((0,), dtype=np.int32)
        )
        self.dihedral_origin_id = (
            as_canonical_int(dihedral_origin_id.reshape(-1)) if dihedral_origin_id is not None else np.zeros((0,), dtype=np.int32)
        )
        self.chirality_i_seqs = as_canonical_int(chirality_i_seqs) if chirality_i_seqs is not None else _empty_int(4)
        self.chirality_volume_ideal = (
            as_canonical_float(chirality_volume_ideal) if chirality_volume_ideal is not None else _empty_float()
        )
        self.chirality_weight = as_canonical_float(chirality_weight) if chirality_weight is not None else _empty_float()
        self.chirality_both_signs = (
            np.ascontiguousarray(chirality_both_signs, dtype=np.uint8) if chirality_both_signs is not None else np.zeros((0,), dtype=np.uint8)
        )
        self.chirality_origin_id = (
            as_canonical_int(chirality_origin_id.reshape(-1)) if chirality_origin_id is not None else np.zeros((0,), dtype=np.int32)
        )
        self.planarity_i_seqs = as_canonical_int(planarity_i_seqs.reshape(-1)) if planarity_i_seqs is not None else np.zeros((0,), dtype=np.int32)
        self.planarity_offsets = as_canonical_int(planarity_offsets.reshape(-1)) if planarity_offsets is not None else np.array([0], dtype=np.int32)
        self.planarity_weight = as_canonical_float(planarity_weight) if planarity_weight is not None else _empty_float()
        self.planarity_origin_id = (
            as_canonical_int(planarity_origin_id.reshape(-1)) if planarity_origin_id is not None else np.zeros((0,), dtype=np.int32)
        )
        self.parallelity_i_seqs = as_canonical_int(parallelity_i_seqs.reshape(-1)) if parallelity_i_seqs is not None else np.zeros((0,), dtype=np.int32)
        self.parallelity_i_offsets = (
            as_canonical_int(parallelity_i_offsets.reshape(-1)) if parallelity_i_offsets is not None else np.array([0], dtype=np.int32)
        )
        self.parallelity_j_seqs = as_canonical_int(parallelity_j_seqs.reshape(-1)) if parallelity_j_seqs is not None else np.zeros((0,), dtype=np.int32)
        self.parallelity_j_offsets = (
            as_canonical_int(parallelity_j_offsets.reshape(-1)) if parallelity_j_offsets is not None else np.array([0], dtype=np.int32)
        )
        self.parallelity_weight = as_canonical_float(parallelity_weight) if parallelity_weight is not None else _empty_float()
        self.parallelity_target_angle = (
            as_canonical_float(parallelity_target_angle) if parallelity_target_angle is not None else _empty_float()
        )
        self.parallelity_origin_id = (
            as_canonical_int(parallelity_origin_id.reshape(-1)) if parallelity_origin_id is not None else np.zeros((0,), dtype=np.int32)
        )
        self.nonbonded_i_seqs = as_canonical_int(nonbonded_i_seqs) if nonbonded_i_seqs is not None else _empty_int(2)
        self.nonbonded_vdw = as_canonical_float(nonbonded_vdw) if nonbonded_vdw is not None else _empty_float()
        self.refcoord_i_seq = (
            as_canonical_int(refcoord_i_seq.reshape(-1)) if refcoord_i_seq is not None else np.zeros((0,), dtype=np.int32)
        )
        self.refcoord_xyz = as_canonical_float(refcoord_xyz) if refcoord_xyz is not None else np.zeros((0, 3), dtype=np.float64)
        self.refcoord_weight = as_canonical_float(refcoord_weight) if refcoord_weight is not None else _empty_float()
        self.bondsim_pair_i_seqs = (
            as_canonical_int(bondsim_pair_i_seqs) if bondsim_pair_i_seqs is not None else _empty_int(2)
        )
        self.bondsim_offsets = (
            as_canonical_int(bondsim_offsets.reshape(-1)) if bondsim_offsets is not None else np.array([0], dtype=np.int32)
        )
        self.bondsim_weight = as_canonical_float(bondsim_weight) if bondsim_weight is not None else _empty_float()
        self.adp_pair_i_seqs = as_canonical_int(adp_pair_i_seqs) if adp_pair_i_seqs is not None else _empty_int(2)
        self.adp_pair_weight = as_canonical_float(adp_pair_weight) if adp_pair_weight is not None else _empty_float()
        self.adp_pair_tau = as_canonical_float(adp_pair_tau) if adp_pair_tau is not None else _empty_float()
        self.adp_pair_class = (
            as_canonical_int(adp_pair_class.reshape(-1)) if adp_pair_class is not None else np.zeros((0,), dtype=np.int32)
        )
        self.rigid_bond_i_seqs = (
            as_canonical_int(rigid_bond_i_seqs) if rigid_bond_i_seqs is not None else _empty_int(2)
        )
        self.rigid_bond_weight = (
            as_canonical_float(rigid_bond_weight) if rigid_bond_weight is not None else _empty_float()
        )
        self.wilson_b = float(wilson_b) if wilson_b is not None else None
        self.adp_nu = float(adp_nu) if adp_nu is not None else None
        self.adp_level_weight = float(adp_level_weight) if adp_level_weight is not None else None
        self.adp_is_aniso = (
            np.ascontiguousarray(adp_is_aniso, dtype=np.uint8).reshape(-1)
            if adp_is_aniso is not None
            else np.zeros((self.n_sites,), dtype=np.uint8)
        )
        self.model_indices = as_canonical_int(model_indices.reshape(-1)) if model_indices is not None else None
        self.conformer_indices = (
            as_canonical_int(conformer_indices.reshape(-1)) if conformer_indices is not None else None
        )
        self.meta = GeometryRestraints(
            crystal=crystal,
            n_sites=self.n_sites,
            n_bonds=int(self.bond_i_seqs.shape[0]),
            n_bond_asu=int(self.bond_asu_i_seqs.shape[0]),
            n_angles=int(self.angle_i_seqs.shape[0]),
            n_dihedrals=int(self.dihedral_i_seqs.shape[0]),
            n_chiralities=int(self.chirality_i_seqs.shape[0]),
            n_planarities=int(len(self.planarity_offsets) - 1),
            n_parallelities=int(len(self.parallelity_i_offsets) - 1),
            n_nonbonded=int(self.nonbonded_i_seqs.shape[0]),
            n_reference_coords=int(self.refcoord_i_seq.shape[0]),
            n_bond_similarities=int(len(self.bondsim_offsets) - 1),
            bond_asu_rt_mx=self.bond_asu_rt_mx,
            n_adp_pairs=int(self.adp_pair_i_seqs.shape[0]),
            n_rigid_bonds=int(self.rigid_bond_i_seqs.shape[0]),
            wilson_b=self.wilson_b,
            adp_nu=self.adp_nu,
            adp_level_weight=self.adp_level_weight,
        )

    def pack(self) -> bytes:
        return _savez(
            bond_i_seqs=self.bond_i_seqs,
            bond_distance_ideal=self.bond_distance_ideal,
            bond_weight=self.bond_weight,
            bond_slack=self.bond_slack,
            bond_origin_id=self.bond_origin_id,
            bond_asu_i_seqs=self.bond_asu_i_seqs,
            bond_asu_distance_ideal=self.bond_asu_distance_ideal,
            bond_asu_weight=self.bond_asu_weight,
            angle_i_seqs=self.angle_i_seqs,
            angle_ideal=self.angle_ideal,
            angle_weight=self.angle_weight,
            angle_origin_id=self.angle_origin_id,
            dihedral_i_seqs=self.dihedral_i_seqs,
            dihedral_angle_ideal=self.dihedral_angle_ideal,
            dihedral_weight=self.dihedral_weight,
            dihedral_periodicity=self.dihedral_periodicity,
            dihedral_origin_id=self.dihedral_origin_id,
            chirality_i_seqs=self.chirality_i_seqs,
            chirality_volume_ideal=self.chirality_volume_ideal,
            chirality_weight=self.chirality_weight,
            chirality_both_signs=self.chirality_both_signs,
            chirality_origin_id=self.chirality_origin_id,
            planarity_i_seqs=self.planarity_i_seqs,
            planarity_offsets=self.planarity_offsets,
            planarity_weight=self.planarity_weight,
            planarity_origin_id=self.planarity_origin_id,
            parallelity_i_seqs=self.parallelity_i_seqs,
            parallelity_i_offsets=self.parallelity_i_offsets,
            parallelity_j_seqs=self.parallelity_j_seqs,
            parallelity_j_offsets=self.parallelity_j_offsets,
            parallelity_weight=self.parallelity_weight,
            parallelity_target_angle=self.parallelity_target_angle,
            parallelity_origin_id=self.parallelity_origin_id,
            nonbonded_i_seqs=self.nonbonded_i_seqs,
            nonbonded_vdw=self.nonbonded_vdw,
            refcoord_i_seq=self.refcoord_i_seq,
            refcoord_xyz=self.refcoord_xyz,
            refcoord_weight=self.refcoord_weight,
            bondsim_pair_i_seqs=self.bondsim_pair_i_seqs,
            bondsim_offsets=self.bondsim_offsets,
            bondsim_weight=self.bondsim_weight,
            **(
                {
                    "adp_pair_i_seqs": self.adp_pair_i_seqs,
                    "adp_pair_weight": self.adp_pair_weight,
                    "adp_pair_tau": self.adp_pair_tau,
                    "adp_pair_class": self.adp_pair_class,
                }
                if self.adp_pair_i_seqs.size
                else {}
            ),
            **(
                {
                    "rigid_bond_i_seqs": self.rigid_bond_i_seqs,
                    "rigid_bond_weight": self.rigid_bond_weight,
                }
                if self.rigid_bond_i_seqs.size
                else {}
            ),
            **(
                {"adp_is_aniso": self.adp_is_aniso}
                if self.adp_is_aniso.any()
                else {}
            ),
            **(
                {"wilson_b": np.array([self.wilson_b], dtype=np.float64)}
                if self.wilson_b is not None
                else {}
            ),
            **(
                {"adp_nu": np.array([self.adp_nu], dtype=np.float64)}
                if self.adp_nu is not None
                else {}
            ),
            **(
                {"adp_level_weight": np.array([self.adp_level_weight], dtype=np.float64)}
                if self.adp_level_weight is not None
                else {}
            ),
            **(
                {"model_indices": self.model_indices}
                if self.model_indices is not None
                else {}
            ),
            **(
                {"conformer_indices": self.conformer_indices}
                if self.conformer_indices is not None
                else {}
            ),
        )


def unpack_restraints(blob: bytes, meta: GeometryRestraints) -> PackedRestraints:
    with np.load(io.BytesIO(blob), allow_pickle=False) as zf:
        def get(name, default):
            return zf[name] if name in zf.files else default

        def get_scalar(name, default):
            if name in zf.files:
                val = zf[name]
                return float(val.reshape(-1)[0])
            return default

        return PackedRestraints(
            n_sites=meta.n_sites,
            crystal=meta.crystal,
            bond_i_seqs=get("bond_i_seqs", None),
            bond_distance_ideal=get("bond_distance_ideal", None),
            bond_weight=get("bond_weight", None),
            bond_slack=get("bond_slack", None),
            bond_origin_id=get("bond_origin_id", None),
            bond_asu_i_seqs=get("bond_asu_i_seqs", None),
            bond_asu_distance_ideal=get("bond_asu_distance_ideal", None),
            bond_asu_weight=get("bond_asu_weight", None),
            bond_asu_rt_mx=list(meta.bond_asu_rt_mx),
            angle_i_seqs=get("angle_i_seqs", None),
            angle_ideal=get("angle_ideal", None),
            angle_weight=get("angle_weight", None),
            angle_origin_id=get("angle_origin_id", None),
            dihedral_i_seqs=get("dihedral_i_seqs", None),
            dihedral_angle_ideal=get("dihedral_angle_ideal", None),
            dihedral_weight=get("dihedral_weight", None),
            dihedral_periodicity=get("dihedral_periodicity", None),
            dihedral_origin_id=get("dihedral_origin_id", None),
            chirality_i_seqs=get("chirality_i_seqs", None),
            chirality_volume_ideal=get("chirality_volume_ideal", None),
            chirality_weight=get("chirality_weight", None),
            chirality_both_signs=get("chirality_both_signs", None),
            chirality_origin_id=get("chirality_origin_id", None),
            planarity_i_seqs=get("planarity_i_seqs", None),
            planarity_offsets=get("planarity_offsets", None),
            planarity_weight=get("planarity_weight", None),
            planarity_origin_id=get("planarity_origin_id", None),
            parallelity_i_seqs=get("parallelity_i_seqs", None),
            parallelity_i_offsets=get("parallelity_i_offsets", None),
            parallelity_j_seqs=get("parallelity_j_seqs", None),
            parallelity_j_offsets=get("parallelity_j_offsets", None),
            parallelity_weight=get("parallelity_weight", None),
            parallelity_target_angle=get("parallelity_target_angle", None),
            parallelity_origin_id=get("parallelity_origin_id", None),
            nonbonded_i_seqs=get("nonbonded_i_seqs", None),
            nonbonded_vdw=get("nonbonded_vdw", None),
            refcoord_i_seq=get("refcoord_i_seq", None),
            refcoord_xyz=get("refcoord_xyz", None),
            refcoord_weight=get("refcoord_weight", None),
            bondsim_pair_i_seqs=get("bondsim_pair_i_seqs", None),
            bondsim_offsets=get("bondsim_offsets", None),
            bondsim_weight=get("bondsim_weight", None),
            adp_pair_i_seqs=get("adp_pair_i_seqs", None),
            adp_pair_weight=get("adp_pair_weight", None),
            adp_pair_tau=get("adp_pair_tau", None),
            adp_pair_class=get("adp_pair_class", None),
            rigid_bond_i_seqs=get("rigid_bond_i_seqs", None),
            rigid_bond_weight=get("rigid_bond_weight", None),
            wilson_b=get_scalar("wilson_b", meta.wilson_b),
            adp_nu=get_scalar("adp_nu", meta.adp_nu),
            adp_level_weight=get_scalar("adp_level_weight", meta.adp_level_weight),
            adp_is_aniso=get("adp_is_aniso", None),
            model_indices=zf["model_indices"] if "model_indices" in zf.files else None,
            conformer_indices=zf["conformer_indices"] if "conformer_indices" in zf.files else None,
        )
