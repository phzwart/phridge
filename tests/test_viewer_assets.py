"""Tests for viewer asset extraction and HTML template generation."""

from __future__ import annotations

import tempfile
from pathlib import Path

from phridge.client.viewer import HTML_TEMPLATE, STATIC_HTML_TEMPLATE, open_static_viewer


def test_viewer_assets_loaded():
    assert "<!DOCTYPE html>" in STATIC_HTML_TEMPLATE
    assert "%%EMBEDDED_PDB%%" in STATIC_HTML_TEMPLATE
    assert "%%FILE_GUIDE_HTML%%" in STATIC_HTML_TEMPLATE

    assert "<!DOCTYPE html>" in HTML_TEMPLATE
    assert "%%CONFIG_JSON%%" in HTML_TEMPLATE


def test_open_static_viewer():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "test_viewer.html"
        html_path = open_static_viewer(
            output_html=out_file,
            open_browser=False,
        )
        assert Path(html_path).is_file()
        content = Path(html_path).read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "phridge Zero-Server 3D" in content
