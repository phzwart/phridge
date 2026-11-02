#!/usr/bin/env python3
"""Write one PDB per TDP1 dimer in the frame of the SVD maps.

The maps use the median cell. 6DHU chain A is the pose that sits on those
maps. The dimer is moved as one rigid body, so chains A and B keep their
crystal arrangement. Each structure is its own file.

Usage::

    python examples/case_study/svd_frame.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import iotbx.ccp4_map
import iotbx.pdb
import numpy as np
from cctbx import crystal

_ROOT = Path(__file__).resolve().parent
_STRUCTURES = _ROOT / "structures"
_SKIP = {"8uvz"}
_POSE = "6dhu"
_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def map_cell() -> tuple[float, ...]:
    """Unit cell of the SVD maps. Falls back to the median cell if no map is on disk."""
    candidates = (
        _ROOT / "gradient_svd" / "component_01.ccp4",
        _ROOT / "svd_with_water" / "component_01.ccp4",
        _ROOT / "fwt_with_water" / "component_01.ccp4",
    )
    for path in candidates:
        if not path.is_file():
            continue
        reader = iotbx.ccp4_map.map_reader(file_name=str(path))
        return tuple(float(value) for value in reader.unit_cell_parameters)
    return (49.99700164794922, 105.12300109863281, 193.7659912109375, 90.0, 90.0, 90.0)


def model_path(folder: Path) -> Path | None:
    pdb = folder / f"{folder.name}.pdb"
    if pdb.is_file():
        return pdb
    cif = folder / f"{folder.name}.cif"
    return cif if cif.is_file() else None


def trace(chain, unit_cell) -> dict[int, np.ndarray]:
    """Residue number to Cartesian Cα."""
    points: dict[int, np.ndarray] = {}
    for group in chain.residue_groups():
        if list(group.atom_groups())[0].resname.strip() not in _ONE:
            continue
        alpha = next((atom for atom in group.atoms() if atom.name.strip() == "CA"), None)
        if alpha is None:
            continue
        points[int(group.resseq)] = np.array(unit_cell.orthogonalize(unit_cell.fractionalize(alpha.xyz)), dtype=np.float64)
    return points


def map_trace(chain, source_cell, map_unit_cell) -> dict[int, np.ndarray]:
    """Reference Cα in the SVD map cell. Fractional coordinates are unchanged."""
    points: dict[int, np.ndarray] = {}
    for group in chain.residue_groups():
        if list(group.atom_groups())[0].resname.strip() not in _ONE:
            continue
        alpha = next((atom for atom in group.atoms() if atom.name.strip() == "CA"), None)
        if alpha is None:
            continue
        fractional = source_cell.fractionalize(alpha.xyz)
        points[int(group.resseq)] = np.array(map_unit_cell.orthogonalize(fractional), dtype=np.float64)
    return points


def kabsch(mobile: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation and translation taking ``mobile`` onto ``reference``."""
    mobile_center = mobile.mean(axis=0)
    reference_center = reference.mean(axis=0)
    covariance = (mobile - mobile_center).T @ (reference - reference_center)
    left, _values, right = np.linalg.svd(covariance)
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right[-1] *= -1.0
        rotation = right.T @ left.T
    translation = reference_center - rotation @ mobile_center
    return rotation, translation


def place(mobile: dict[int, np.ndarray], reference: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray, float]:
    """Rigid fit of this chain onto the reference pose."""
    common = sorted(set(mobile) & set(reference))
    if len(common) < 80:
        raise ValueError(f"only {len(common)} residues in common")
    source = np.vstack([mobile[number] for number in common])
    target = np.vstack([reference[number] for number in common])
    rotation, translation = kabsch(source, target)
    moved = (rotation @ source.T).T + translation
    residual = float(np.sqrt(np.mean(np.sum((moved - target) ** 2, axis=1))))
    return rotation, translation, residual


def anchor_chain(hierarchy):
    """Chain A when it is protein; otherwise the longest protein chain."""
    chains = [chain for chain in hierarchy.chains() if _protein_count(chain) >= 80]
    for chain in chains:
        if chain.id.strip() == "A":
            return chain
    return max(chains, key=_protein_count, default=None)


def _protein_count(chain) -> int:
    return sum(1 for group in chain.residue_groups() if list(group.atom_groups())[0].resname.strip() in _ONE)


def write_dimer(hierarchy, rotation: np.ndarray, translation: np.ndarray, destination: Path, header: str) -> None:
    """Write every chain with one transform. Chain IDs are kept. Water is omitted."""
    symmetry = crystal.symmetry(unit_cell=map_cell(), space_group_symbol="P 21 21 21")
    lines = [header.ljust(80)[:80], _cryst1(symmetry)]
    serial = 1
    for chain in hierarchy.chains():
        chain_id = (chain.id.strip() or "A")[:1]
        for atom in chain.atoms():
            group = atom.parent()
            if group.resname.strip() in {"HOH", "WAT", "DOD"}:
                continue
            cartesian = rotation @ np.array(atom.xyz, dtype=np.float64) + translation
            record = "HETATM" if atom.hetero else "ATOM"
            residue = group.parent()
            lines.append(
                f"{record:<6}{serial:5d} {atom.name:>4}{group.altloc:1}{group.resname:>3} {chain_id}"
                f"{residue.resseq:>4}{residue.icode:1}   {cartesian[0]:8.3f}{cartesian[1]:8.3f}{cartesian[2]:8.3f}"
                f"{atom.occ:6.2f}{atom.b:6.2f}"
            )
            serial += 1
    lines.append("END")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _cryst1(symmetry) -> str:
    parameters = symmetry.unit_cell().parameters()
    return (
        f"CRYST1{parameters[0]:9.3f}{parameters[1]:9.3f}{parameters[2]:9.3f}"
        f"{parameters[3]:7.2f}{parameters[4]:7.2f}{parameters[5]:7.2f} P 21 21 21"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PDBs in the SVD map frame.")
    parser.add_argument("--structures", type=Path, default=_STRUCTURES)
    parser.add_argument("--out", type=Path, default=_ROOT / "svd_frame")
    args = parser.parse_args(argv)
    pose_path = model_path(args.structures / _POSE)
    pose_input = iotbx.pdb.input(file_name=str(pose_path))
    pose_chain = next(chain for chain in pose_input.construct_hierarchy().chains() if chain.id.strip() == "A")
    output_cell = crystal.symmetry(unit_cell=map_cell(), space_group_symbol="P 21 21 21").unit_cell()
    reference = map_trace(pose_chain, pose_input.crystal_symmetry().unit_cell(), output_cell)
    args.out.mkdir(parents=True, exist_ok=True)
    for stale in args.out.glob("*.pdb"):
        stale.unlink()
    written = 0
    for folder in sorted(path for path in args.structures.iterdir() if path.is_dir()):
        if folder.name in _SKIP:
            continue
        path = model_path(folder)
        if path is None:
            continue
        loaded = iotbx.pdb.input(file_name=str(path))
        unit_cell = loaded.crystal_symmetry().unit_cell()
        hierarchy = loaded.construct_hierarchy()
        anchor = anchor_chain(hierarchy)
        if anchor is None:
            print(f"skip {folder.name}: no protein chain")
            continue
        try:
            rotation, translation, residual = place(trace(anchor, unit_cell), reference)
        except ValueError as exc:
            print(f"skip {folder.name}: {exc}")
            continue
        destination = args.out / f"{folder.name}.pdb"
        write_dimer(
            hierarchy,
            rotation,
            translation,
            destination,
            f"REMARK dimer fit on chain {anchor.id.strip()} to 6DHU A in the SVD map cell; Cα rmsd {residual:.3f} A",
        )
        written += 1
        print(f"{folder.name} chain {anchor.id.strip()} Cα rmsd {residual:.3f} A")
    print(f"wrote {written} dimers to {args.out}")
    return 0


def model_path(folder: Path) -> Path | None:
    pdb = folder / f"{folder.name}.pdb"
    if pdb.is_file():
        return pdb
    cif = folder / f"{folder.name}.cif"
    return cif if cif.is_file() else None


if __name__ == "__main__":
    raise SystemExit(main())
