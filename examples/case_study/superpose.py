#!/usr/bin/env python3
"""Superimpose every deposited TDP1 chain onto 8V0C.

8UVZ is Bacillus subtilis DHFR, so it is not the structural reference.
8V0C is the TDP1 crystal closest to the median cell of the median set.
Each protein chain is written as its own PDB under ``superposed/``.

Usage::

    python examples/case_study/superpose.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import iotbx.pdb
import numpy as np

_ROOT = Path(__file__).resolve().parent
_STRUCTURES = _ROOT / "structures"
_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
_SKIP = {"8uvz"}
_REFERENCE = "8v0c"


def model_path(folder: Path) -> Path | None:
    pdb = folder / f"{folder.name}.pdb"
    if pdb.is_file():
        return pdb
    cif = folder / f"{folder.name}.cif"
    return cif if cif.is_file() else None


def protein_chains(hierarchy):
    """Chains that contain a protein Cα trace."""
    found = []
    for chain in hierarchy.chains():
        residues = _residues(chain)
        if len(residues) >= 80:
            found.append((chain, residues))
    return found


def _residues(chain) -> list[tuple[str, int, np.ndarray]]:
    residues = []
    for group in chain.residue_groups():
        name = list(group.atom_groups())[0].resname.strip()
        one = _ONE.get(name)
        if one is None:
            continue
        alpha = next((atom for atom in group.atoms() if atom.name.strip() == "CA"), None)
        if alpha is None:
            continue
        residues.append((one, int(group.resseq), np.array(alpha.xyz, dtype=np.float64)))
    return residues


def align(reference: str, mobile: str) -> list[tuple[int, int]]:
    """Needleman–Wunsch. Returns pairs of indexes into the two sequences."""
    gap = -2
    rows, cols = len(reference) + 1, len(mobile) + 1
    score = np.zeros((rows, cols))
    pointer = np.zeros((rows, cols), dtype=np.int8)
    for i in range(1, rows):
        score[i, 0] = i * gap
        pointer[i, 0] = 1
    for j in range(1, cols):
        score[0, j] = j * gap
        pointer[0, j] = 2
    for i in range(1, rows):
        for j in range(1, cols):
            match = score[i - 1, j - 1] + (1 if reference[i - 1] == mobile[j - 1] else -1)
            delete = score[i - 1, j] + gap
            insert = score[i, j - 1] + gap
            choice = max(match, delete, insert)
            score[i, j] = choice
            pointer[i, j] = 0 if choice == match else 1 if choice == delete else 2
    pairs: list[tuple[int, int]] = []
    i, j = len(reference), len(mobile)
    while i > 0 or j > 0:
        step = int(pointer[i, j])
        if step == 0:
            i -= 1
            j -= 1
            if reference[i] == mobile[j]:
                pairs.append((i, j))
        elif step == 1:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def kabsch(mobile: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation and translation that map ``mobile`` onto ``reference``."""
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


def chain_text(chain, rotation: np.ndarray, translation: np.ndarray, serial: int) -> tuple[str, int]:
    """PDB ATOM/HETATM records for one chain, excluding water."""
    lines = []
    for atom in chain.atoms():
        parent = atom.parent()
        if parent.resname.strip() in {"HOH", "WAT", "DOD"}:
            continue
        xyz = rotation @ np.array(atom.xyz, dtype=np.float64) + translation
        record = "HETATM" if atom.hetero else "ATOM"
        serial = serial % 99999 + 1
        lines.append(
            f"{record:<6}{serial:5d} {atom.name:>4}{parent.altloc:1}"
            f"{parent.resname:>3} {chain.id:>1}{parent.parent().resseq:>4}{parent.parent().icode:1}   "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{atom.occ:6.2f}{atom.b:6.2f}"
        )
    return "\n".join(lines), serial


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Superimpose deposited TDP1 chains onto 8V0C.")
    parser.add_argument("--structures", type=Path, default=_STRUCTURES)
    parser.add_argument("--out", type=Path, default=_ROOT / "superposed")
    args = parser.parse_args(argv)
    reference_hierarchy = iotbx.pdb.input(file_name=str(model_path(args.structures / _REFERENCE))).construct_hierarchy()
    reference_chain, reference_residues = protein_chains(reference_hierarchy)[0]
    reference_sequence = "".join(one for one, _number, _xyz in reference_residues)
    args.out.mkdir(parents=True, exist_ok=True)
    deviations: dict[int, list[float]] = {}
    model_number = 0
    for folder in sorted(path for path in args.structures.iterdir() if path.is_dir()):
        if folder.name in _SKIP:
            continue
        path = model_path(folder)
        if path is None:
            continue
        hierarchy = iotbx.pdb.input(file_name=str(path)).construct_hierarchy()
        for chain, residues in protein_chains(hierarchy):
            pairs = align(reference_sequence, "".join(one for one, _number, _xyz in residues))
            if len(pairs) < 80:
                print(f"skip {folder.name} chain {chain.id}: only {len(pairs)} matched residues")
                continue
            mobile = np.vstack([residues[j][2] for _i, j in pairs])
            target = np.vstack([reference_residues[i][2] for i, _j in pairs])
            rotation, translation = kabsch(mobile, target)
            moved = (rotation @ mobile.T).T + translation
            rmsd = float(np.sqrt(np.mean(np.sum((moved - target) ** 2, axis=1))))
            for (i, j), xyz in zip(pairs, moved):
                deviations.setdefault(reference_residues[i][1], []).append(float(np.linalg.norm(xyz - reference_residues[i][2])))
            body, _serial = chain_text(chain, rotation, translation, 0)
            chain_id = chain.id.strip() or "A"
            destination = args.out / f"{folder.name}_{chain_id}.pdb"
            destination.write_text(f"REMARK fit to 8V0C; Cα rmsd {rmsd:.3f} A\n{body}\nEND\n", encoding="utf-8")
            model_number += 1
            print(f"{folder.name} {chain_id} matched {len(pairs)} Cα rmsd {rmsd:.3f}")
    _write_profile(args.out / "ca_deviation.txt", deviations)
    print(f"wrote {model_number} PDB files to {args.out}")
    return 0


def _write_profile(path: Path, deviations: dict[int, list[float]]) -> None:
    lines = ["resseq n mean_A std_A"]
    for number in sorted(deviations):
        values = np.array(deviations[number], dtype=np.float64)
        lines.append(f"{number:4d} {values.size:3d} {values.mean():7.3f} {values.std():7.3f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
