"""Import-boundary rules for src/phridge. Expected to fail until later stages."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.boundaries

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PHRIDGE = REPO_ROOT / "src" / "phridge"

LEGACY_EXEMPT = ("src/phridge/legacy/",)

_CCTBX_PACKAGES = frozenset({"cctbx", "mmtbx", "iotbx", "scitbx", "libtbx"})
_CCTBX_DOORS = (
    "src/phridge/io/",
    "src/phridge/shims/",
    "src/phridge/worker/cctbx_runner.py",
)
_TORCH_FORBIDDEN = (
    "src/phridge/io/",
    "src/phridge/shims/cctbx/",
)
_CORE_SPACES = (
    "src/phridge/core/",
    "src/phridge/r/",
    "src/phridge/q/",
    "src/phridge/d/",
    "src/phridge/transforms/",
    "src/phridge/optim/",
    "src/phridge/policies/",
    "src/phridge/report/",
)
_CORE_ALLOWED_THIRD_PARTY = frozenset({"torch", "numpy", "pydantic", "redis", "phridge"})


def iter_modules(root: Path) -> list[tuple[Path, set[str]]]:
    """Return each module path and the top-level packages it imports."""
    modules: list[tuple[Path, set[str]]] = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    names.add(alias.name.split(".", 1)[0])
            elif isinstance(node, ast.ImportFrom):
                if node.level > 0 or not node.module:
                    continue
                names.add(node.module.split(".", 1)[0])
        modules.append((path, names))
    return modules


def _rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _under(rel: str, prefixes: tuple[str, ...]) -> bool:
    return any(rel == prefix or rel.startswith(prefix) for prefix in prefixes)


def _format(offenders: list[str]) -> str:
    return "\n".join(offenders)


def test_no_phenix_anywhere() -> None:
    offenders = [
        _rel(path)
        for path, names in iter_modules(SRC_PHRIDGE)
        if "phenix" in names
    ]
    assert not offenders, "phenix imports:\n" + _format(offenders)


def test_cctbx_only_behind_doors() -> None:
    offenders: list[str] = []
    for path, names in iter_modules(SRC_PHRIDGE):
        if not (names & _CCTBX_PACKAGES):
            continue
        rel = _rel(path)
        if _under(rel, LEGACY_EXEMPT) or _under(rel, _CCTBX_DOORS):
            continue
        offenders.append(rel)
    assert not offenders, "cctbx-family imports outside the doors:\n" + _format(offenders)


def test_no_torch_in_io_or_cctbx_shims() -> None:
    offenders = [
        _rel(path)
        for path, names in iter_modules(SRC_PHRIDGE)
        if "torch" in names and _under(_rel(path), _TORCH_FORBIDDEN)
    ]
    assert not offenders, "torch imports in io/ or shims/cctbx/:\n" + _format(offenders)


def test_core_tier_only_in_spaces() -> None:
    allowed = set(sys.stdlib_module_names) | _CORE_ALLOWED_THIRD_PARTY
    offenders: list[str] = []
    for path, names in iter_modules(SRC_PHRIDGE):
        rel = _rel(path)
        if not _under(rel, _CORE_SPACES):
            continue
        extra = sorted(names - allowed)
        if extra:
            offenders.append(f"{rel}: {', '.join(extra)}")
    assert not offenders, "core-tier imports outside the allow-list:\n" + _format(offenders)
