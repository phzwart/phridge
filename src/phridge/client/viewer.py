"""Browser-based 3D crystallographic structure and map viewer using Mol*.

Serves a lightweight local web interface embedding the Mol* viewer
(from https://molstar.org / CDN) to inspect PDB atomic models and CCP4/MRC
electron density / gradient difference maps without external server dependencies.
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import socketserver
import sys
import threading
import urllib.parse
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional
import importlib.resources


def _load_template(name: str) -> str:
    """Load an HTML template asset from the package resources."""
    try:
        ref = importlib.resources.files("phridge.client.assets").joinpath(name)
        return ref.read_text(encoding="utf-8")
    except Exception:
        fallback_path = Path(__file__).resolve().parent / "assets" / name
        return fallback_path.read_text(encoding="utf-8")


STATIC_HTML_TEMPLATE: str = _load_template("viewer_standalone.html")
HTML_TEMPLATE: str = _load_template("viewer.html")


class ViewerHTTPHandler(http.server.SimpleHTTPRequestHandler):
    """Serves the Mol* viewer and local crystallographic files with proper headers."""

    config: Dict[str, Any] = {}
    base_dir: Path = Path(".")

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ["/", "/index.html"]:
            html = HTML_TEMPLATE.replace("%%CONFIG_JSON%%", json.dumps(self.config))
            content = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(content)
            return

        # Serve static file requested by relative path
        rel_path = path.lstrip("/")
        file_path = (self.base_dir / rel_path).resolve()

        if not file_path.is_file():
            self.send_error(404, f"File not found: {rel_path}")
            return

        # Content types for crystallographic files
        ext = file_path.suffix.lower()
        content_type = {
            ".pdb": "chemical/x-pdb",
            ".ent": "chemical/x-pdb",
            ".cif": "chemical/x-mmcif",
            ".ccp4": "application/octet-stream",
            ".map": "application/octet-stream",
            ".mrc": "application/octet-stream",
            ".mtz": "application/octet-stream",
        }.get(ext, "application/octet-stream")

        try:
            with open(file_path, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
        except (ConnectionResetError, BrokenPipeError):
            pass
        except Exception as e:
            try:
                self.send_error(500, f"Error reading file: {e}")
            except Exception:
                pass

    def log_message(self, format: str, *args: Any) -> None:
        # Quiet log unless requested
        if os.environ.get("PHRIDGE_VIEWER_DEBUG"):
            super().log_message(format, *args)


def launch_viewer(
    pdb_path: Optional[str | Path] = None,
    gradient_map: Optional[str | Path] = None,
    map_2fofc: Optional[str | Path] = None,
    map_fofc: Optional[str | Path] = None,
    port: int = 8899,
    open_browser: bool = True,
    block: bool = True,
) -> tuple[socketserver.TCPServer, str]:
    """Launch a local Mol* web viewer server for the specified model and maps."""
    # Determine common root directory to serve files safely
    all_paths = [Path(p).resolve() for p in [pdb_path, gradient_map, map_2fofc, map_fofc] if p]
    if not all_paths:
        base_dir = Path.cwd()
    else:
        # Common ancestor or parent of first file
        common = Path(os.path.commonpath([str(p.parent) for p in all_paths]))
        base_dir = common

    def to_url_path(p: Optional[str | Path]) -> Optional[str]:
        if not p:
            return None
        res = Path(p).resolve()
        try:
            rel = res.relative_to(base_dir).as_posix()
            return f"/{rel}"
        except ValueError:
            return f"/{res.name}"

    config = {
        "pdb": to_url_path(pdb_path),
        "gradient_map": to_url_path(gradient_map),
        "map_2fofc": to_url_path(map_2fofc),
        "map_fofc": to_url_path(map_fofc),
    }

    # Bind handler settings
    handler_cls = type(
        "ConfiguredViewerHandler",
        (ViewerHTTPHandler,),
        {"config": config, "base_dir": base_dir},
    )

    # Find free port starting at `port`
    server = None
    actual_port = port
    for p in range(port, port + 50):
        try:
            server = socketserver.TCPServer(("127.0.0.1", p), handler_cls)
            actual_port = p
            break
        except OSError:
            continue

    if server is None:
        raise RuntimeError(f"Could not bind viewer to any port between {port} and {port+50}")

    url = f"http://127.0.0.1:{actual_port}/"
    print("\n" + "=" * 65)
    print(" phridge 3D Crystallographic Map & Model Viewer (Mol*)")
    print("=" * 65)
    print(f" URL:        {url}")
    if pdb_path:
        print(f" Model:      {pdb_path}")
    if gradient_map:
        print(f" Grad Map:   {gradient_map} (Green: +3σ, Red: -3σ)")
    if map_2fofc:
        print(f" 2mFo-DFc:   {map_2fofc} (Blue: 1.5σ)")
    print(" Press Ctrl+C in terminal to stop server.")
    print("=" * 65 + "\n")

    if open_browser:
        webbrowser.open(url)

    if block:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nViewer server stopped.")
            server.server_close()
    else:
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()

    return server, url


def open_static_viewer(
    pdb_path: Optional[str | Path] = None,
    gradient_map: Optional[str | Path] = None,
    map_2fofc: Optional[str | Path] = None,
    map_fofc: Optional[str | Path] = None,
    output_html: Optional[str | Path] = None,
    open_browser: bool = True,
) -> Path:
    """Generate and open a standalone, zero-server HTML viewer for model and maps.

    No local HTTP server, port binding, or background process is used.
    The viewer runs completely client-side in the web browser using WebGL and WebAssembly.
    """
    # 1. Read PDB content if available
    embedded_pdb = ""
    if pdb_path and Path(pdb_path).is_file():
        try:
            embedded_pdb = Path(pdb_path).read_text(encoding="utf-8")
        except Exception as e:
            print(f"Warning: Could not read PDB file to embed: {e}", file=sys.stderr)

    # 2. Build file guide HTML
    guide_items = []
    if pdb_path and Path(pdb_path).is_file():
        guide_items.append(f"<div><b>Model:</b> <code>{Path(pdb_path).name}</code> (Embedded &amp; pre-loaded)</div>")
    elif pdb_path:
        guide_items.append(f"<div><b>Model:</b> <code>{pdb_path}</code> (Drop file here)</div>")

    if gradient_map:
        guide_items.append(f"<div><b>Gradient Difference Map:</b> <code>{Path(gradient_map).name}</code> &rarr; <i>Drop into viewer for &plusmn;3&sigma; green/red contours</i></div>")
    if map_2fofc:
        guide_items.append(f"<div><b>2mFo-DFc Density Map:</b> <code>{Path(map_2fofc).name}</code> &rarr; <i>Drop into viewer for 1.5&sigma; blue mesh</i></div>")
    if map_fofc:
        guide_items.append(f"<div><b>mFo-DFc Difference Map:</b> <code>{Path(map_fofc).name}</code> &rarr; <i>Drop into viewer for &plusmn;3&sigma; contours</i></div>")

    if not guide_items:
        guide_items.append("<div>Drop any <code>.pdb</code> / <code>.cif</code> model or <code>.ccp4</code> / <code>.map</code> file here.</div>")

    file_guide_html = "\n".join(guide_items)

    html_content = (
        STATIC_HTML_TEMPLATE
        .replace("%%EMBEDDED_PDB%%", embedded_pdb)
        .replace("%%FILE_GUIDE_HTML%%", file_guide_html)
    )

    # 3. Determine output path
    if output_html is not None:
        out_path = Path(output_html).resolve()
    elif pdb_path:
        p = Path(pdb_path).resolve()
        out_path = p.parent / f"{p.stem}_viewer.html"
    elif gradient_map:
        p = Path(gradient_map).resolve()
        out_path = p.parent / f"{p.stem}_viewer.html"
    else:
        out_path = Path("viewer.html").resolve()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_content, encoding="utf-8")

    file_url = out_path.as_uri()
    print("\n" + "=" * 65)
    print(" phridge Zero-Server 3D Viewer (Mol*)")
    print("=" * 65)
    print(f" Standalone HTML: {out_path}")
    print(f" Browser URL:     {file_url}")
    print(" No local server, open ports, or background processes required.")
    if pdb_path:
        print(f" Model:           {pdb_path} ({'Embedded & Pre-loaded' if embedded_pdb else 'Ready to drop'})")
    if gradient_map:
        print(f" Gradient Map:    {gradient_map} (Ready to drop: +3σ Green, -3σ Red)")
    if map_2fofc:
        print(f" 2mFo-DFc Map:    {map_2fofc} (Ready to drop: 1.5σ Blue)")
    print("\n Alternative Public Web Viewers (drag-and-drop your files):")
    print("   • Mol* Web App:    https://molstar.org/viewer/")
    print("   • UglyMol Viewer:  https://uglymol.github.io/")
    print("   • Moorhen WebCoot: https://moorhen.org/")
    print("=" * 65 + "\n")

    if open_browser:
        webbrowser.open(file_url)

    return out_path


def main(args: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Launch browser-based 3D crystallographic viewer (Mol*) for PDB model and CCP4 maps (Zero-Server by default)",
    )
    parser.add_argument("pdb", nargs="?", default=None, help="Path to PDB atomic model")
    parser.add_argument("--grad", "--gradient", default=None, dest="gradient", help="Path to target gradient map (.ccp4)")
    parser.add_argument("--2fofc", default=None, dest="map_2fofc", help="Path to 2mFo-DFc map (.ccp4)")
    parser.add_argument("--fofc", default=None, dest="map_fofc", help="Path to mFo-DFc difference map (.ccp4)")
    parser.add_argument("--prefix", default=None, help="Prefix used by phridge-intensity to auto-detect files")
    parser.add_argument("--server", action="store_true", help="Launch local HTTP server instead of zero-server static HTML")
    parser.add_argument("--no-server", action="store_true", default=True, help="Open as standalone zero-server HTML (default)")
    parser.add_argument("--port", type=int, default=8899, help="Local HTTP server port when --server is enabled (default: 8899)")
    parser.add_argument("--output-html", default=None, help="Output path for standalone HTML viewer file")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")

    opts = parser.parse_args(args)

    pdb = opts.pdb
    grad = opts.gradient
    map_2fofc = opts.map_2fofc
    map_fofc = opts.map_fofc

    if opts.prefix:
        prefix = opts.prefix
        if not pdb and Path(f"{prefix}_refined.pdb").is_file():
            pdb = f"{prefix}_refined.pdb"
        elif not pdb and Path(f"{prefix}.pdb").is_file():
            pdb = f"{prefix}.pdb"
        if not grad and Path(f"{prefix}_gradient.ccp4").is_file():
            grad = f"{prefix}_gradient.ccp4"
        if not map_2fofc and Path(f"{prefix}_2fofc.ccp4").is_file():
            map_2fofc = f"{prefix}_2fofc.ccp4"
        if not map_fofc and Path(f"{prefix}_fofc.ccp4").is_file():
            map_fofc = f"{prefix}_fofc.ccp4"

    if not pdb and not grad and not map_2fofc:
        parser.print_help()
        print("\nError: Please provide at least a PDB file or a map file to visualize.", file=sys.stderr)
        return 1

    try:
        if opts.server:
            launch_viewer(
                pdb_path=pdb,
                gradient_map=grad,
                map_2fofc=map_2fofc,
                map_fofc=map_fofc,
                port=opts.port,
                open_browser=not opts.no_browser,
                block=True,
            )
        else:
            open_static_viewer(
                pdb_path=pdb,
                gradient_map=grad,
                map_2fofc=map_2fofc,
                map_fofc=map_fofc,
                output_html=opts.output_html,
                open_browser=not opts.no_browser,
            )
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
