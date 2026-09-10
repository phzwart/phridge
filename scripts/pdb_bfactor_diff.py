#!/usr/bin/env python3
"""Write a PDB whose B-factors are the per-atom difference between two models.

Coordinates come from the minuend model (first file) by default; each atom's
B is ``B(minuend) - B(subtrahend)``.  Useful for comparing two refinements
that differ mainly in ADPs.

Example:
  python scripts/pdb_bfactor_diff.py refined_a.pdb refined_b.pdb -o b_diff.pdb
  python scripts/pdb_bfactor_diff.py a.pdb b.pdb --subtrahend b.pdb --minuend a.pdb
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional, Tuple

import numpy as np

AtomKey = Tuple[str, int, str, str, str, str]


@dataclass(frozen=True)
class AtomRecord:
    key: AtomKey
    atom: Any
    chain: Any
    residue_group: Any
    atom_group: Any


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Subtract B-factors of two PDB models and write a difference PDB.",
    )
    p.add_argument("minuend", type=Path, help="PDB A (B values minus subtrahend)")
    p.add_argument("subtrahend", type=Path, help="PDB B (subtracted from minuend)")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output PDB path (B column = B_A - B_B)",
    )
    p.add_argument(
        "--coords-from",
        choices=("minuend", "subtrahend"),
        default="minuend",
        help="Which model supplies xyz/occ for matched atoms (default: minuend)",
    )
    p.add_argument(
        "--swap",
        action="store_true",
        help="Equivalent to swapping minuend/subtrahend (B = subtrahend - minuend)",
    )
    p.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Multiply difference B by this factor before writing (default: 1)",
    )
    p.add_argument(
        "--aniso",
        choices=("equiv", "ignore"),
        default="equiv",
        help=(
            "How to treat anisotropic atoms: "
            "'equiv' uses equivalent isotropic B from U (default); "
            "'ignore' uses the isotropic B column only"
        ),
    )
    p.add_argument(
        "--require-match",
        action="store_true",
        help="Exit with error if any atom in minuend coords model is unmatched",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress summary statistics on stdout",
    )
    return p.parse_args(argv)


def _load_hierarchy(path: Path) -> tuple[Any, Any]:
    from iotbx import pdb

    inp = pdb.input(file_name=str(path))
    return inp.construct_hierarchy(), inp.crystal_symmetry()


def _iter_atoms(hierarchy: Any) -> Iterator[AtomRecord]:
    for model in hierarchy.models():
        for chain in model.chains():
            for rg in chain.residue_groups():
                for ag in rg.atom_groups():
                    for atom in ag.atoms():
                        key = (
                            chain.id.strip(),
                            rg.resseq_as_int(),
                            (rg.icode.strip() or " "),
                            ag.resname.strip(),
                            atom.name.strip(),
                            (ag.altloc.strip() or " "),
                        )
                        yield AtomRecord(key=key, atom=atom, chain=chain, residue_group=rg, atom_group=ag)


def _index_atoms(hierarchy: Any) -> dict[AtomKey, AtomRecord]:
    return {rec.key: rec for rec in _iter_atoms(hierarchy)}


def _effective_b_iso(atom: Any, unit_cell: Any, *, aniso_mode: str) -> float:
    if aniso_mode == "ignore" or not atom.uij_is_defined():
        return float(atom.b)
    from cctbx import adptbx

    u_cart = adptbx.u_cif_as_u_cart(unit_cell, atom.uij)
    u_iso = adptbx.u_cart_as_u_iso(u_cart)
    return float(adptbx.u_iso_as_b_iso(u_iso))


def _format_key(key: AtomKey) -> str:
    chain, resseq, icode, resname, name, altloc = key
    alt = f" altloc={altloc!r}" if altloc.strip() else ""
    return f"{chain} {resname} {resseq}{icode} {name}{alt}"


def _clone_hierarchy(hierarchy: Any) -> Any:
    from iotbx import pdb

    lines = hierarchy.as_pdb_string().splitlines()
    return pdb.input(source_info=None, lines=lines).construct_hierarchy()


def _apply_diff(
    *,
    out_hierarchy: Any,
    lookup_sub: dict[AtomKey, AtomRecord],
    unit_cell: Any,
    aniso_mode: str,
    scale: float,
    swapped: bool,
) -> tuple[np.ndarray, list[AtomKey], list[AtomKey]]:
    diffs: list[float] = []
    missing_in_sub: list[AtomKey] = []
    missing_in_out: list[AtomKey] = []

    out_index = _index_atoms(out_hierarchy)
    for key, out_rec in out_index.items():
        sub_rec = lookup_sub.get(key)
        if sub_rec is None:
            missing_in_sub.append(key)
            continue
        b_out = _effective_b_iso(out_rec.atom, unit_cell, aniso_mode=aniso_mode)
        b_sub = _effective_b_iso(sub_rec.atom, unit_cell, aniso_mode=aniso_mode)
        delta = (b_sub - b_out) if swapped else (b_out - b_sub)
        delta *= scale
        out_rec.atom.set_b(float(delta))
        if out_rec.atom.uij_is_defined():
            out_rec.atom.set_uij((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        diffs.append(float(delta))

    minu_index = set(out_index)
    for key in lookup_sub:
        if key not in minu_index:
            missing_in_out.append(key)

    return np.asarray(diffs, dtype=np.float64), missing_in_sub, missing_in_out


def _write_remarks(
    path: Path,
    *,
    minuend: Path,
    subtrahend: Path,
    coords_from: str,
    swapped: bool,
    scale: float,
    aniso_mode: str,
    n_matched: int,
    stats: Optional[dict[str, float]],
) -> None:
    sign = "B(subtrahend) - B(minuend)" if swapped else "B(minuend) - B(subtrahend)"
    lines = [
        f"REMARK   3 B-FACTOR DIFFERENCE PDB",
        f"REMARK   3   MINUEND     : {minuend}",
        f"REMARK   3   SUBTRAHEND  : {subtrahend}",
        f"REMARK   3   FORMULA     : {sign}",
        f"REMARK   3   COORDS FROM : {coords_from}",
        f"REMARK   3   ANISO MODE  : {aniso_mode}",
        f"REMARK   3   SCALE       : {scale:g}",
        f"REMARK   3   MATCHED     : {n_matched}",
    ]
    if stats is not None:
        lines.extend(
            [
                f"REMARK   3   MEAN DB     : {stats['mean']:+.3f}",
                f"REMARK   3   RMS DB      : {stats['rms']:.3f}",
                f"REMARK   3   MIN DB      : {stats['min']:+.3f}",
                f"REMARK   3   MAX DB      : {stats['max']:+.3f}",
                f"REMARK   3   MEAN |DB|   : {stats['mean_abs']:.3f}",
            ]
        )
    body = path.read_text()
    path.write_text("\n".join(lines) + "\n" + body)


def _summarize(diffs: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(diffs)),
        "rms": float(np.sqrt(np.mean(diffs**2))),
        "min": float(np.min(diffs)),
        "max": float(np.max(diffs)),
        "mean_abs": float(np.mean(np.abs(diffs))),
    }


def run(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    minuend_path = args.minuend.resolve()
    subtrahend_path = args.subtrahend.resolve()
    if not minuend_path.is_file():
        raise SystemExit(f"Minuend PDB not found: {minuend_path}")
    if not subtrahend_path.is_file():
        raise SystemExit(f"Subtrahend PDB not found: {subtrahend_path}")

    hier_a, cs_a = _load_hierarchy(minuend_path)
    hier_b, cs_b = _load_hierarchy(subtrahend_path)
    unit_cell = cs_a.unit_cell() if cs_a.unit_cell() is not None else cs_b.unit_cell()
    if unit_cell is None:
        raise SystemExit("Neither PDB has CRYST1 / unit cell information.")

    lookup_a = _index_atoms(hier_a)
    lookup_b = _index_atoms(hier_b)

    coords_hier = hier_b if args.coords_from == "subtrahend" else hier_a
    coords_lookup = lookup_b if args.coords_from == "subtrahend" else lookup_a
    other_lookup = lookup_a if args.coords_from == "subtrahend" else lookup_b

    out_hierarchy = _clone_hierarchy(coords_hier)
    diffs, missing_other, missing_coords = _apply_diff(
        out_hierarchy=out_hierarchy,
        lookup_sub=other_lookup,
        unit_cell=unit_cell,
        aniso_mode=args.aniso,
        scale=args.scale,
        swapped=bool(args.swap),
    )

    if diffs.size == 0:
        raise SystemExit("No matched atoms between the two models.")

    if args.require_match and missing_other:
        sample = ", ".join(_format_key(k) for k in missing_other[:5])
        more = f" (+{len(missing_other) - 5} more)" if len(missing_other) > 5 else ""
        raise SystemExit(f"Unmatched atoms in coords model: {sample}{more}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_hierarchy.write_pdb_file(file_name=str(args.output))

    stats = _summarize(diffs)
    _write_remarks(
        args.output,
        minuend=minuend_path,
        subtrahend=subtrahend_path,
        coords_from=args.coords_from,
        swapped=bool(args.swap),
        scale=args.scale,
        aniso_mode=args.aniso,
        n_matched=int(diffs.size),
        stats=stats,
    )

    if not args.quiet:
        sign = "B_B - B_A" if args.swap else "B_A - B_B"
        print(f"Wrote {args.output}  ({diffs.size} atoms, {sign})")
        print(
            f"  ΔB: mean={stats['mean']:+.3f}  rms={stats['rms']:.3f}  "
            f"min={stats['min']:+.3f}  max={stats['max']:+.3f}  "
            f"mean|ΔB|={stats['mean_abs']:.3f}"
        )
        if missing_other:
            print(f"  warning: {len(missing_other)} coords-model atoms missing in other PDB")
            for key in missing_other[:3]:
                print(f"    - {_format_key(key)}")
            if len(missing_other) > 3:
                print(f"    ... and {len(missing_other) - 3} more")
        if missing_coords:
            print(f"  note: {len(missing_coords)} atoms only in other PDB (omitted from output)")

    return 0


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
