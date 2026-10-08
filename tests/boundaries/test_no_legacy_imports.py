"""Nothing outside ``legacy/`` imports it, except tests marked ``legacy``."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.boundaries

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOTS = (REPO_ROOT / "src", REPO_ROOT / "tests")


def _imports_legacy(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "phridge.legacy" or alias.name.startswith("phridge.legacy."):
                    return True
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "phridge.legacy" or node.module.startswith("phridge.legacy."):
                return True
            if node.module == "phridge" and any(alias.name == "legacy" or alias.name.startswith("legacy.") for alias in node.names):
                return True
    return False


def _marked_legacy(tree: ast.AST) -> bool:
    for node in tree.body if isinstance(tree, ast.Module) else []:
        value = None
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "pytestmark" for target in node.targets):
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "pytestmark":
            value = node.value
        if value is not None and _expr_is_legacy(value):
            return True
    return False


def _expr_is_legacy(node: ast.AST) -> bool:
    if isinstance(node, ast.Attribute) and node.attr == "legacy":
        return True
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return any(_expr_is_legacy(item) for item in node.elts)
    if isinstance(node, ast.Call):
        return _expr_is_legacy(node.func)
    return False


def test_no_legacy_imports_outside_legacy() -> None:
    offenders: list[str] = []
    for root in ROOTS:
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts or "legacy" in path.parts and "src" in path.parts:
                if "src" in path.parts and "legacy" in path.parts:
                    continue
            rel = path.relative_to(REPO_ROOT).as_posix()
            if rel.startswith("src/phridge/legacy/"):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            if not _imports_legacy(tree):
                continue
            if rel.startswith("tests/") and _marked_legacy(tree):
                continue
            offenders.append(rel)
    assert offenders == []
