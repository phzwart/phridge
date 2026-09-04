#!/usr/bin/env python3
"""Deprecated path — use examples/restraint_minimization.py."""

from __future__ import annotations

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parents[1] / "examples" / "restraint_minimization.py"), run_name="__main__")
