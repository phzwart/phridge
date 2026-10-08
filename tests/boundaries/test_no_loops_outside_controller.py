"""Optimizer steps stay in the controller, policies, and the conformance harness."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "src" / "phridge"
PATTERNS = ("optimizer.step(", ".step(objective")
ALLOWED = {"legacy", "core", "policies", "testing"}


def test_no_optimizer_loops_outside_controller() -> None:
    hits: list[str] = []
    for path in sorted(ROOT.rglob("*.py")):
        if ALLOWED.intersection(path.parts):
            continue
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if any(pattern in line for pattern in PATTERNS):
                hits.append(f"{path}:{lineno}:{line.strip()}")
    assert hits == []
