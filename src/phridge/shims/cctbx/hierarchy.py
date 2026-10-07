"""Hierarchy and site adapters."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from phridge.contracts.models import Atom, CoordinateFrame
from phridge.contracts.packing import PackedCartesian, PackedFractional, PackedHierarchy
from phridge.shims.cctbx._flex import vec3_double
from phridge.shims.cctbx.symmetry import crystal_from_cctbx


from phridge.contracts.manifest import PluginManifest, Slot, Space, Tier

MANIFEST = PluginManifest(
    name='cctbx_hierarchy',
    slot=Slot.io_adapter,
    space=Space.D,
    tier=Tier.shim,
)

def sites_cart_from_cctbx(sites: Any, crystal: Optional[Any] = None) -> PackedCartesian:
    xyz = np.asarray(list(sites), dtype=np.float64)
    cs = crystal_from_cctbx(crystal) if crystal is not None and hasattr(crystal, "unit_cell") else crystal
    return PackedCartesian(xyz, crystal=cs)


def sites_frac_from_cctbx(sites: Any, crystal: Any) -> PackedFractional:
    xyz = np.asarray(list(sites), dtype=np.float64)
    cs = crystal_from_cctbx(crystal) if hasattr(crystal, "unit_cell") else crystal
    return PackedFractional(xyz, crystal=cs)


def sites_cart_to_cctbx(packed: PackedCartesian) -> Any:
    
    return vec3_double([tuple(row) for row in packed.xyz])


def sites_frac_to_cctbx(packed: PackedFractional) -> Any:
    
    return vec3_double([tuple(row) for row in packed.xyz])


def hierarchy_from_cctbx(root: Any, crystal: Optional[Any] = None) -> PackedHierarchy:
    atoms_flex = root.atoms()
    n = atoms_flex.size()
    xyz = np.zeros((n, 3), dtype=np.float64)
    occ = np.zeros(n, dtype=np.float64)
    b_iso = np.zeros(n, dtype=np.float64)
    u_cart = None
    uij_defined = None
    labels: list[Atom] = []
    for i, atom in enumerate(atoms_flex):
        xyz[i] = atom.xyz
        occ[i] = atom.occ
        b_iso[i] = atom.b
        if atom.uij_is_defined():
            if u_cart is None:
                u_cart = np.zeros((n, 6), dtype=np.float64)
                uij_defined = np.zeros(n, dtype=np.uint8)
            u_cart[i] = atom.uij
            uij_defined[i] = 1
        ag = atom.parent()
        rg = ag.parent()
        chain = rg.parent()
        model = chain.parent()
        icode = str(rg.icode).strip() or None
        altloc = str(ag.altloc).strip() or None
        labels.append(
            Atom(
                i=i,
                name=str(atom.name),
                element=str(atom.element).strip(),
                model_id=str(model.id) or None,
                chain_id=str(chain.id),
                resseq=int(rg.resseq_as_int()),
                icode=icode,
                resname=str(ag.resname).strip(),
                altloc=altloc,
                hetero=bool(atom.hetero),
            )
        )
    cs = None
    if crystal is not None:
        cs = crystal_from_cctbx(crystal) if hasattr(crystal, "unit_cell") else crystal
    return PackedHierarchy(
        xyz=xyz,
        occupancy=occ,
        b_iso=b_iso,
        atoms=labels,
        crystal=cs,
        frame=CoordinateFrame.cartesian,
        u_cart=u_cart,
        uij_defined=uij_defined,
    )


def hierarchy_to_cctbx(packed: PackedHierarchy) -> Any:
    from iotbx.pdb import hierarchy

    root = hierarchy.root()
    models: dict[str, Any] = {}
    chains: dict[tuple, Any] = {}
    groups: dict[tuple, Any] = {}
    atom_groups: dict[tuple, Any] = {}
    for atom_meta, xyz, occ, b_iso in zip(
        packed.meta.atoms, packed.xyz, packed.occupancy, packed.b_iso
    ):
        model_id = atom_meta.model_id or ""
        if model_id not in models:
            model = hierarchy.model(id=model_id)
            models[model_id] = model
            root.append_model(model)
        model = models[model_id]
        chain_key = (model_id, atom_meta.chain_id)
        if chain_key not in chains:
            chain = hierarchy.chain(id=atom_meta.chain_id)
            chains[chain_key] = chain
            model.append_chain(chain)
        chain = chains[chain_key]
        icode = atom_meta.icode or " "
        rg_key = (model_id, atom_meta.chain_id, atom_meta.resseq, icode)
        if rg_key not in groups:
            rg = hierarchy.residue_group(resseq="%4d" % atom_meta.resseq, icode=icode)
            groups[rg_key] = rg
            chain.append_residue_group(rg)
        rg = groups[rg_key]
        altloc = atom_meta.altloc or ""
        ag_key = rg_key + (altloc, atom_meta.resname)
        if ag_key not in atom_groups:
            ag = hierarchy.atom_group(altloc=altloc, resname=atom_meta.resname)
            atom_groups[ag_key] = ag
            rg.append_atom_group(ag)
        ag = atom_groups[ag_key]
        atom = hierarchy.atom()
        name = atom_meta.name if len(atom_meta.name) >= 4 else atom_meta.name.center(4)
        atom.set_name(name)
        atom.set_element(atom_meta.element)
        atom.set_xyz(tuple(float(x) for x in xyz))
        atom.set_occ(float(occ))
        atom.set_b(float(b_iso))
        atom.set_hetero(bool(atom_meta.hetero))
        if packed.uij_defined is not None and packed.u_cart is not None and packed.uij_defined[atom_meta.i]:
            atom.set_uij(tuple(float(x) for x in packed.u_cart[atom_meta.i]))
        ag.append_atom(atom)
    return root
