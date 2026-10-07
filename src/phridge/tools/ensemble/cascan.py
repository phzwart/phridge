"""Optional gemmi cross-check. Kept outside the core import allow-list."""

from __future__ import annotations


def module_file() -> str:
    import cascan_pipeline

    return getattr(cascan_pipeline, "__file__", "cascan_pipeline")
