#!/usr/bin/env python3
"""cctbx → phridge → cctbx geometry-restraint minimization.

Starts from live cctbx objects (hierarchy + geometry_restraints.manager),
packs them through phridge, reconstitutes cctbx objects, and runs LBFGS
on the reconstituted manager.

Requires chem_data in the active env::

    conda install -n phridge-cctbx -c chem_data chem_data
    python examples/restraint_minimization.py
"""

from __future__ import annotations

from cctbx import geometry_restraints
from cctbx.geometry_restraints import bond_params, bond_params_table
from cctbx.geometry_restraints.lbfgs import lbfgs as geo_lbfgs
from cctbx.geometry_restraints.manager import manager
from libtbx.utils import null_out
from scitbx import lbfgs as scitbx_lbfgs
from scitbx.array_family import flex
import iotbx.pdb
import mmtbx.model

from phridge.client.convert import from_canonical, to_canonical
from phridge.client.convert_xtal import hierarchy_from_cctbx
from phridge.client.geometry import model_geometry, restraints_from_cctbx, restraints_to_proxies
from phridge.packing_geometry import PackedRestraints, unpack_restraints
from phridge.packing_xtal import PackedHierarchy

PDB = """\
CRYST1   21.937    4.866   23.477  90.00 107.08  90.00 P 1 21 1      2
ATOM      1  N   GLY A   1      -9.009   4.612   6.102  1.00 16.77           N
ATOM      2  CA  GLY A   1      -9.052   4.207   4.651  1.00 16.57           C
ATOM      3  C   GLY A   1      -8.015   3.140   4.419  1.00 16.16           C
ATOM      4  O   GLY A   1      -7.523   2.521   5.381  1.00 16.78           O
ATOM      5  N   ASN A   2      -7.656   2.923   3.155  1.00 15.02           N
ATOM      6  CA  ASN A   2      -6.522   2.038   2.831  1.00 14.10           C
ATOM      7  C   ASN A   2      -5.241   2.537   3.427  1.00 13.13           C
ATOM      8  O   ASN A   2      -4.978   3.742   3.426  1.00 11.91           O
ATOM      9  CB  ASN A   2      -6.346   1.881   1.341  1.00 15.38           C
ATOM     10  CG  ASN A   2      -7.584   1.342   0.692  1.00 14.08           C
ATOM     11  OD1 ASN A   2      -8.025   0.227   1.016  1.00 17.46           O
ATOM     12  ND2 ASN A   2      -8.204   2.155  -0.169  1.00 11.72           N
ATOM     13  N   ASN A   3      -4.438   1.590   3.905  1.00 12.26           N
ATOM     14  CA  ASN A   3      -3.193   1.904   4.589  1.00 11.74           C
ATOM     15  C   ASN A   3      -1.955   1.332   3.895  1.00 11.10           C
ATOM     16  O   ASN A   3      -1.872   0.119   3.648  1.00 10.42           O
ATOM     17  CB  ASN A   3      -3.259   1.378   6.042  1.00 12.15           C
END
"""


def require_monomer_library() -> str:
    from mmtbx.monomer_library import server as mon_server

    path = mon_server.find_mon_lib_file(relative_path_components=["list", "mon_lib_list.cif"])
    if path is None:
        raise SystemExit(
            "chem_data / monomer library not found. Install with:\n"
            "  conda install -n phridge-cctbx -c chem_data chem_data\n"
            "  # or: make chem-data"
        )
    return path


def build_cctbx_model():
    """1. Start with cctbx: hierarchy + geometry_restraints.manager."""
    model = mmtbx.model.manager(
        model_input=iotbx.pdb.input(source_info=None, lines=PDB.split("\n")),
        log=null_out(),
    )
    model.process(make_restraints=True)
    hierarchy = model.get_hierarchy()
    grm = model.get_restraints_manager().geometry
    return hierarchy, grm


def through_phridge(hierarchy, grm) -> tuple[PackedHierarchy, PackedRestraints]:
    """2. Pack live cctbx objects into phridge canonical types (+ wire bytes)."""
    # Explicit converters (same path Bridge.call uses via to_canonical).
    packed_hier = hierarchy_from_cctbx(hierarchy)
    packed_restr = restraints_from_cctbx(grm, n_sites=hierarchy.atoms().size())
    assert isinstance(to_canonical(hierarchy), PackedHierarchy)
    assert isinstance(to_canonical(grm), PackedRestraints)

    # Wire format: one npz blob + LinkML metadata (what Redis would store).
    blob = packed_restr.pack()
    restored = unpack_restraints(blob, packed_restr.meta)
    assert restored.meta.n_bonds == packed_restr.meta.n_bonds

    header = model_geometry(hierarchy=packed_hier, restraints=packed_restr)
    print(
        f"phridge: n_sites={header.n_sites} i_seq_identity={header.i_seq_identity} "
        f"bonds={packed_restr.meta.n_bonds} angles={packed_restr.meta.n_angles} "
        f"dihedrals={packed_restr.meta.n_dihedrals}  wire_bytes={len(blob)}"
    )
    return packed_hier, packed_restr


def manager_from_proxies(packed: PackedRestraints) -> manager:
    """Rebuild a bond/angle manager from phridge proxy tables.

    ``restraints_to_proxies`` returns cctbx proxy objects, not a full manager
    (bond_params_table / nonbonded / ASU shells are not reconstituted).
    """
    proxies = restraints_to_proxies(packed)
    bpt = bond_params_table(packed.n_sites)
    for bond in proxies["bonds"]:
        i, j = bond.i_seqs
        bpt.update(
            i,
            j,
            bond_params(distance_ideal=bond.distance_ideal, weight=bond.weight, slack=bond.slack),
        )
    return manager(
        bond_params_table=bpt,
        angle_proxies=geometry_restraints.shared_angle_proxy(proxies["angles"]),
        dihedral_proxies=geometry_restraints.shared_dihedral_proxy(proxies["dihedrals"]),
    )


def back_to_cctbx(packed_hier: PackedHierarchy, packed_restr: PackedRestraints):
    """3. Receive cctbx objects again from phridge packed types."""
    hierarchy = from_canonical(packed_hier, prefer_cctbx=True)
    proxies = from_canonical(packed_restr, prefer_cctbx=True)  # dict of cctbx proxies
    grm = manager_from_proxies(packed_restr)
    assert hierarchy.atoms().size() == packed_hier.meta.n_atoms
    assert len(proxies["bonds"]) == packed_restr.meta.n_bonds
    return hierarchy, grm


def minimize(hierarchy, grm) -> None:
    """4. Run restraint minimization on the reconstituted cctbx objects."""
    sites0 = hierarchy.atoms().extract_xyz()
    sites = sites0.deep_copy()
    sites += flex.vec3_double([(0.25, -0.15, 0.1)] * sites.size())

    e0 = grm.energies_sites(sites_cart=sites, compute_gradients=False)
    print(f"cctbx energy before: {e0.target:.6g}")

    minimizer = geo_lbfgs(
        sites_cart=sites,
        geometry_restraints_manager=grm,
        lbfgs_termination_params=scitbx_lbfgs.termination_parameters(max_iterations=100),
    )
    hierarchy.atoms().set_xyz(sites)
    e1 = grm.energies_sites(sites_cart=sites, compute_gradients=False)
    print(f"cctbx energy after:  {e1.target:.6g}")
    print(f"LBFGS: {minimizer.first_target_value:.6g} -> {minimizer.final_target_value:.6g}")
    if not (e1.target < 0.2 * e0.target):
        raise SystemExit("geometry minimization did not drop the target enough")


def main() -> None:
    mon_lib = require_monomer_library()
    print(f"monomer library: {mon_lib}")
    print("--- 1. cctbx objects ---")
    hierarchy, grm = build_cctbx_model()
    print(f"hierarchy atoms={hierarchy.atoms().size()}  grm={type(grm).__module__}.{type(grm).__name__}")

    print("--- 2. through phridge ---")
    packed_hier, packed_restr = through_phridge(hierarchy, grm)

    print("--- 3. back to cctbx ---")
    hierarchy2, grm2 = back_to_cctbx(packed_hier, packed_restr)
    print(
        f"hierarchy atoms={hierarchy2.atoms().size()}  "
        f"grm={type(grm2).__module__}.{type(grm2).__name__}"
    )

    print("--- 4. minimize with reconstituted cctbx ---")
    minimize(hierarchy2, grm2)
    print("OK")


if __name__ == "__main__":
    main()
