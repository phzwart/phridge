"""No module over 800 lines and no function over 150, outside legacy."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "src" / "phridge"
MODULE_CAP = 800
FUNCTION_CAP = 150


def test_module_and_function_caps() -> None:
    too_long: list[str] = []
    for path in sorted(ROOT.rglob("*.py")):
        if "legacy" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        nlines = text.count("\n") + (0 if text.endswith("\n") or text == "" else 1)
        if nlines > MODULE_CAP:
            too_long.append(f"{path}:{nlines} lines")
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                span = (node.end_lineno or node.lineno) - node.lineno + 1
                if span > FUNCTION_CAP:
                    too_long.append(f"{path}:{node.lineno}:{node.name} {span} lines")
    assert too_long == []
