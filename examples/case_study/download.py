#!/usr/bin/env python3
"""Download deposited models and structure-factor mmCIFs for the case-study PDB IDs.

Each ID becomes one directory under ``structures/``:

- ``{id}.pdb`` — deposited model (``{id}.cif`` when RCSB has no PDB-format file)
- ``{id}-sf.cif`` — deposited structure factors; intensities, when present, are in this file

Usage::

    python examples/case_study/download.py
    python examples/case_study/download.py --force
"""

from __future__ import annotations

import argparse
import shutil
import urllib.error
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_RCSB = "https://files.rcsb.org/download"
_USER_AGENT = "phridge-case-study"


def read_ids(path: Path) -> list[str]:
    """One PDB ID per line. Blank lines and ``#`` comments are skipped."""
    ids: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        pdb_id = line.lower()
        if len(pdb_id) != 4 or not pdb_id[0].isdigit() or not pdb_id.isalnum():
            raise SystemExit(f"not a PDB ID: {line}")
        ids.append(pdb_id)
    if not ids:
        raise SystemExit(f"no PDB IDs in {path}")
    return ids


def download_file(url: str, dest: Path, *, force: bool) -> None:
    """Fetch ``url`` to ``dest``. An existing non-empty file is kept unless ``force``."""
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"  using {dest.relative_to(_ROOT)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    print(f"  downloading {url}")
    try:
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(dest)
    print(f"  wrote {dest.name} ({dest.stat().st_size / 1024:.1f} KB)")


def fetch_entry(pdb_id: str, out: Path, *, force: bool) -> str | None:
    """Download one model and its structure factors. Return an error string on failure."""
    folder = out / pdb_id
    print(pdb_id)
    try:
        _fetch_model(pdb_id, folder, force=force)
        download_file(f"{_RCSB}/{pdb_id}-sf.cif", folder / f"{pdb_id}-sf.cif", force=force)
    except urllib.error.HTTPError as exc:
        return f"{pdb_id}: HTTP {exc.code} for {exc.url}"
    except urllib.error.URLError as exc:
        return f"{pdb_id}: {exc.reason}"
    return None


def _fetch_model(pdb_id: str, folder: Path, *, force: bool) -> None:
    pdb = folder / f"{pdb_id}.pdb"
    try:
        download_file(f"{_RCSB}/{pdb_id}.pdb", pdb, force=force)
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        print(f"  no PDB-format model for {pdb_id}; fetching mmCIF")
        download_file(f"{_RCSB}/{pdb_id}.cif", folder / f"{pdb_id}.cif", force=force)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download case-study models and structure factors from the PDB.")
    parser.add_argument("--ids", type=Path, default=_ROOT / "pdb_ids.txt", help="PDB ID list (default: pdb_ids.txt)")
    parser.add_argument("--out", type=Path, default=_ROOT / "structures", help="destination directory")
    parser.add_argument("--force", action="store_true", help="re-download files that are already present")
    args = parser.parse_args(argv)
    failures = [err for pdb_id in read_ids(args.ids) if (err := fetch_entry(pdb_id, args.out, force=args.force))]
    print(f"done: {args.out}")
    for err in failures:
        print(f"failed: {err}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
