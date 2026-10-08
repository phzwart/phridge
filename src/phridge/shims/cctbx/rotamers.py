"""Van der Waals radii and the rotamer library. Loaded once; scoring stays in torch."""

from __future__ import annotations

from typing import Any

import numpy as np

from phridge.contracts.manifest import PluginManifest, Slot, Space, Tier

MANIFEST = PluginManifest(
    name="cctbx_rotamers",
    slot=Slot.io_adapter,
    space=Space.R,
    tier=Tier.hybrid,
)

_RADII: dict[str, float] | None = None
_ROTAMERS: dict[str, dict[str, np.ndarray]] | None = None


def vdw_radius(element: str) -> float:
    """One radius from the cctbx van der Waals table."""
    table = _vdw_table()
    return float(table.get(element.strip().title(), table.get(element.strip().upper(), 1.7)))


def rotamer_library() -> dict[str, dict[str, np.ndarray]]:
    """Chi targets and frequencies, keyed by lowercase residue name. Parsed once."""
    global _ROTAMERS
    if _ROTAMERS is None:
        _ROTAMERS = _parse_rotamers()
    return _ROTAMERS


def _vdw_table() -> dict[str, float]:
    global _RADII
    if _RADII is None:
        from cctbx.eltbx import van_der_waals_radii

        _RADII = {str(key): float(value) for key, value in van_der_waals_radii.vdw.table.items()}
    return _RADII


def _parse_rotamers() -> dict[str, dict[str, np.ndarray]]:
    import mmtbx.rotamer as rotamer

    path = __import__("pathlib").Path(rotamer.__file__).resolve().parent / "sidechain_angles.props"
    text = path.read_text(encoding="utf-8")
    names: dict[str, list[str]] = {}
    chis: dict[str, dict[str, list[float]]] = {}
    freq: dict[str, list[float]] = {}
    for raw in text.splitlines():
        line = raw.strip().strip('"')
        if ".rotamers" in line and "=" in line:
            residue, _, value = line.partition("=")
            residue = residue.split(".", 1)[0].strip()
            names[residue] = [item.strip() for item in value.replace('"', "").split(",") if item.strip()]
        elif ".frequencies" in line and "=" in line:
            residue, _, value = line.partition("=")
            residue = residue.split(".", 1)[0].strip()
            freq[residue] = [float(item.strip().rstrip("%")) / 100.0 for item in value.replace('"', "").split(",") if item.strip()]
        left, sep, value = line.partition("=")
        if sep and left.count(".") == 1 and ".chi" not in left and ".angles" not in left:
            name, key = left.split(".", 1)
            name = name.strip()
            key = key.strip()
            if key in {"rotamers", "frequencies", "angles", "chis"} or key.endswith("_atoms") or key.startswith("chi"):
                continue
            try:
                angles = [float(item) for item in value.replace('"', "").split()]
            except ValueError:
                continue
            chis.setdefault(name, {})[key] = angles
    library: dict[str, dict[str, np.ndarray]] = {}
    for residue, rotamers in names.items():
        rows = [chis.get(residue, {}).get(item, []) for item in rotamers]
        if not rows or not rows[0]:
            continue
        width = len(rows[0])
        rows = [row for row in rows if len(row) == width]
        if len(rows) != len(rotamers):
            continue
        library[residue] = {
            "chi": np.asarray(rows, dtype=np.float64),
            "frequency": np.asarray(freq.get(residue, [1.0] * len(rows)), dtype=np.float64),
        }
    return library
