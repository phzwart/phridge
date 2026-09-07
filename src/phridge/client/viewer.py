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

STATIC_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>phridge Zero-Server 3D Map & Model Viewer</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/molstar@4.17.0/build/viewer/molstar.css" />
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    html, body {
      width: 100%;
      height: 100%;
      overflow: hidden;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: #0e0e11;
      color: #f1f5f9;
    }
    #top-bar {
      height: 52px;
      background: #18181b;
      border-bottom: 1px solid #27272a;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 16px;
      z-index: 20;
      position: relative;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      font-size: 14px;
      font-weight: 600;
    }
    .brand .logo {
      color: #38bdf8;
      font-weight: 700;
      font-size: 16px;
      letter-spacing: 0.5px;
    }
    .file-actions {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .file-input-label {
      background: #2563eb;
      color: #fff;
      padding: 6px 14px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 500;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: background 0.15s;
    }
    .file-input-label:hover { background: #1d4ed8; }
    #file-picker { display: none; }
    .badge-list {
      display: flex;
      gap: 8px;
    }
    .badge {
      background: #27272a;
      border: 1px solid #3f3f46;
      border-radius: 4px;
      padding: 3px 8px;
      font-size: 11px;
      color: #94a3b8;
    }
    .badge.active {
      background: #0f172a;
      border-color: #38bdf8;
      color: #38bdf8;
    }
    #viewer-container {
      width: 100%;
      height: calc(100% - 52px);
      position: relative;
    }
    #drop-overlay {
      position: absolute;
      top: 0; left: 0; width: 100%; height: 100%;
      background: rgba(14, 14, 17, 0.88);
      backdrop-filter: blur(4px);
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 16px;
      z-index: 10;
      pointer-events: all;
      transition: opacity 0.2s;
    }
    #drop-overlay.hidden {
      opacity: 0;
      pointer-events: none;
    }
    .drop-box {
      border: 2px dashed #475569;
      border-radius: 12px;
      padding: 36px 50px;
      text-align: center;
      background: rgba(30, 41, 59, 0.45);
      cursor: pointer;
      max-width: 540px;
      transition: all 0.2s;
    }
    .drop-box:hover, .drop-box.drag-over {
      border-color: #38bdf8;
      background: rgba(56, 189, 248, 0.08);
    }
    .drop-box h3 {
      font-size: 16px;
      margin-bottom: 8px;
      color: #f8fafc;
    }
    .drop-box p {
      font-size: 12px;
      color: #94a3b8;
      line-height: 1.5;
    }
    .drop-box .hint {
      margin-top: 14px;
      font-size: 11px;
      color: #64748b;
    }
    .file-guide {
      margin-top: 16px;
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid #334155;
      border-radius: 8px;
      padding: 12px 16px;
      font-size: 11px;
      text-align: left;
      width: 100%;
      color: #cbd5e1;
    }
    .file-guide div {
      margin-bottom: 4px;
    }
    .file-guide div:last-child {
      margin-bottom: 0;
    }
    .file-guide code {
      color: #38bdf8;
      font-family: monospace;
      background: rgba(56, 189, 248, 0.1);
      padding: 1px 4px;
      border-radius: 3px;
    }
    #status-toast {
      position: absolute;
      bottom: 16px;
      left: 16px;
      background: rgba(15, 23, 42, 0.95);
      border: 1px solid #334155;
      padding: 8px 16px;
      border-radius: 6px;
      font-size: 12px;
      color: #e2e8f0;
      z-index: 30;
      display: none;
      align-items: center;
      gap: 8px;
    }
  </style>
  <script src="https://cdn.jsdelivr.net/npm/molstar@4.17.0/build/viewer/molstar.js"></script>
</head>
<body>
  <div id="top-bar">
    <div class="brand">
      <span class="logo">phridge</span>
      <span>Zero-Server 3D Molecular & Map Viewer</span>
    </div>
    <div class="badge-list" id="loaded-badges"></div>
    <div class="file-actions">
      <label class="file-input-label" for="file-picker">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
        Open PDB / CCP4 Files
      </label>
      <input type="file" id="file-picker" multiple accept=".pdb,.ent,.cif,.ccp4,.map,.mrc,.dsn6" />
    </div>
  </div>

  <div id="viewer-container">
    <div id="app" style="width: 100%; height: 100%;"></div>

    <div id="drop-overlay">
      <div class="drop-box" id="drop-zone">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="1.5" style="margin-bottom: 12px;"><path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242M12 12v9m-4-4 4-4 4 4"/></svg>
        <h3>Drop PDB and CCP4 Map Files Here</h3>
        <p>Drop your <b>.pdb</b> atomic model and <b>.ccp4 / .map</b> density or gradient difference maps directly into this window.</p>
        <div class="file-guide" id="file-guide">
          %%FILE_GUIDE_HTML%%
        </div>
        <p class="hint">Zero local server required — everything runs 100% client-side via WebGL inside your browser.</p>
      </div>
    </div>

    <div id="status-toast">
      <span id="toast-msg">Loading...</span>
    </div>
  </div>

  <script type="text/plain" id="embedded-pdb">%%EMBEDDED_PDB%%</script>

  <script>
    let viewerInstance = null;
    const dropOverlay = document.getElementById("drop-overlay");
    const dropZone = document.getElementById("drop-zone");
    const filePicker = document.getElementById("file-picker");
    const badgesContainer = document.getElementById("loaded-badges");
    const toast = document.getElementById("status-toast");
    const toastMsg = document.getElementById("toast-msg");

    function showToast(msg, duration = 3000) {
      toastMsg.textContent = msg;
      toast.style.display = "flex";
      if (duration > 0) {
        setTimeout(() => { toast.style.display = "none"; }, duration);
      }
    }

    async function initViewer() {
      viewerInstance = await molstar.Viewer.create(document.getElementById("app"), {
        layoutIsExpanded: false,
        layoutShowControls: true,
        layoutShowRemoteState: false,
        layoutShowSequence: true,
        layoutShowLog: false,
        layoutShowLeftPanel: true,
        viewportShowExpand: true,
        viewportShowSelectionMode: true,
        viewportShowAnimation: false,
        pdbProvider: "rcsb",
        emdbProvider: "rcsb",
      });

      // Check for embedded PDB
      const embeddedPdbEl = document.getElementById("embedded-pdb");
      if (embeddedPdbEl && embeddedPdbEl.textContent.trim().length > 0) {
        try {
          const blob = new Blob([embeddedPdbEl.textContent], { type: "text/plain" });
          const url = URL.createObjectURL(blob);
          await viewerInstance.loadStructureFromUrl(url, "pdb", false);
          dropOverlay.classList.add("hidden");

          const badge = document.createElement("span");
          badge.className = "badge active";
          badge.textContent = "Embedded Model (Active)";
          badgesContainer.appendChild(badge);
          showToast("Atomic model loaded successfully");
        } catch (e) {
          console.error("Error loading embedded PDB:", e);
        }
      }
    }

    async function loadFiles(files) {
      if (!viewerInstance) await initViewer();
      dropOverlay.classList.add("hidden");

      for (const file of files) {
        const name = file.name.toLowerCase();
        const url = URL.createObjectURL(file);
        showToast("Loading " + file.name + "...", 0);

        const badge = document.createElement("span");
        badge.className = "badge active";
        badge.textContent = file.name;
        badgesContainer.appendChild(badge);

        if (name.endsWith(".pdb") || name.endsWith(".ent") || name.endsWith(".cif")) {
          const fmt = name.endsWith(".cif") ? "mmcif" : "pdb";
          try {
            await viewerInstance.loadStructureFromUrl(url, fmt, false);
            showToast("Loaded model: " + file.name);
          } catch (e) {
            console.error(e);
            showToast("Failed to load model: " + e.message, 4000);
          }
        } else if (name.endsWith(".ccp4") || name.endsWith(".map") || name.endsWith(".mrc")) {
          const isGrad = name.includes("grad");
          const isFofc = name.includes("fofc") && !name.includes("2fofc");

          try {
            if (isGrad || isFofc) {
              await viewerInstance.loadVolumeFromUrl(
                { url, format: "ccp4", isBinary: true },
                [
                  { type: "relative", value: 3.0, color: 0x22c55e, alpha: 0.55 },
                  { type: "relative", value: -3.0, color: 0xef4444, alpha: 0.55 },
                ]
              );
              showToast("Loaded difference map (+3σ Green, -3σ Red): " + file.name);
            } else {
              await viewerInstance.loadVolumeFromUrl(
                { url, format: "ccp4", isBinary: true },
                [
                  { type: "relative", value: 1.5, color: 0x3362b2, alpha: 0.45 },
                ]
              );
              showToast("Loaded electron density (1.5σ Blue): " + file.name);
            }
          } catch (e) {
            console.error(e);
            showToast("Failed to load map: " + e.message, 4000);
          }
        }
      }
    }

    window.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropZone.classList.add("drag-over");
    });
    window.addEventListener("dragleave", (e) => {
      if (e.clientX <= 0 || e.clientY <= 0) {
        dropZone.classList.remove("drag-over");
      }
    });
    window.addEventListener("drop", (e) => {
      e.preventDefault();
      dropZone.classList.remove("drag-over");
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        loadFiles(e.dataTransfer.files);
      }
    });

    filePicker.addEventListener("change", (e) => {
      if (e.target.files && e.target.files.length > 0) {
        loadFiles(e.target.files);
      }
    });

    dropZone.addEventListener("click", () => filePicker.click());
    window.addEventListener("DOMContentLoaded", initViewer);
  </script>
</body>
</html>
"""

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>phridge Molecular & Map Viewer</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/molstar@4.17.0/build/viewer/molstar.css" />
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    html, body { width: 100%; height: 100%; overflow: hidden; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #0e0e10; color: #f0f0f2; }
    #header {
      height: 48px;
      background: #17171a;
      border-bottom: 1px solid #2d2d33;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 16px;
      font-size: 13px;
      z-index: 10;
      position: relative;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 10px;
      font-weight: 600;
      letter-spacing: 0.5px;
    }
    .brand span.phridge {
      color: #60a5fa;
      font-weight: 700;
    }
    .file-badges {
      display: flex;
      gap: 8px;
      align-items: center;
    }
    .badge {
      background: #27272a;
      padding: 4px 10px;
      border-radius: 4px;
      font-size: 11px;
      color: #a1a1aa;
      border: 1px solid #3f3f46;
    }
    .badge.active {
      color: #93c5fd;
      border-color: #3b82f6;
      background: #1e293b;
    }
    .controls {
      display: flex;
      gap: 8px;
      align-items: center;
    }
    .btn {
      background: #2563eb;
      color: #fff;
      border: none;
      padding: 5px 12px;
      border-radius: 4px;
      cursor: pointer;
      font-size: 12px;
      font-weight: 500;
      transition: background 0.15s;
    }
    .btn:hover { background: #1d4ed8; }
    .btn.secondary {
      background: #3f3f46;
      color: #e4e4e7;
    }
    .btn.secondary:hover { background: #52525b; }
    #app {
      width: 100%;
      height: calc(100% - 48px);
      position: relative;
    }
    #status-bar {
      position: absolute;
      bottom: 12px;
      left: 12px;
      background: rgba(24, 24, 27, 0.85);
      backdrop-filter: blur(8px);
      border: 1px solid #3f3f46;
      padding: 6px 14px;
      border-radius: 6px;
      font-size: 12px;
      color: #cbd5e1;
      z-index: 100;
      pointer-events: none;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .spinner {
      width: 12px;
      height: 12px;
      border: 2px solid #60a5fa;
      border-top-color: transparent;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
  <script src="https://cdn.jsdelivr.net/npm/molstar@4.17.0/build/viewer/molstar.js"></script>
</head>
<body>
  <div id="header">
    <div class="brand">
      <span class="phridge">phridge</span>
      <span>Crystallographic Map & Model Viewer</span>
    </div>
    <div class="file-badges" id="badges"></div>
    <div class="controls">
      <button class="btn secondary" id="btn-toggle-grad">Toggle Gradient Map</button>
      <button class="btn secondary" id="btn-toggle-2fofc">Toggle 2mFo-DFc</button>
      <button class="btn" id="btn-recenter">Recenter</button>
    </div>
  </div>

  <div id="app"></div>
  <div id="status-bar">
    <div class="spinner" id="spinner"></div>
    <span id="status-text">Initializing Mol* 3D Canvas...</span>
  </div>

  <script>
    const CONFIG = %%CONFIG_JSON%%;

    const statusEl = document.getElementById("status-text");
    const spinnerEl = document.getElementById("spinner");
    const badgesEl = document.getElementById("badges");

    function setStatus(text, loading = false) {
      statusEl.textContent = text;
      spinnerEl.style.display = loading ? "inline-block" : "none";
    }

    // Populate badges
    if (CONFIG.pdb) {
      const b = document.createElement("span");
      b.className = "badge active";
      b.textContent = "Model: " + CONFIG.pdb.split("/").pop();
      badgesEl.appendChild(b);
    }
    if (CONFIG.gradient_map) {
      const b = document.createElement("span");
      b.className = "badge";
      b.textContent = "Grad: " + CONFIG.gradient_map.split("/").pop();
      badgesEl.appendChild(b);
    }
    if (CONFIG.map_2fofc) {
      const b = document.createElement("span");
      b.className = "badge";
      b.textContent = "2mFo-DFc: " + CONFIG.map_2fofc.split("/").pop();
      badgesEl.appendChild(b);
    }

    async function init() {
      try {
        setStatus("Creating Mol* Viewer...", true);
        const viewer = await molstar.Viewer.create(document.getElementById("app"), {
          layoutIsExpanded: false,
          layoutShowControls: true,
          layoutShowRemoteState: false,
          layoutShowSequence: true,
          layoutShowLog: false,
          layoutShowLeftPanel: true,
          viewportShowExpand: true,
          viewportShowSelectionMode: true,
          viewportShowAnimation: false,
          pdbProvider: "rcsb",
          emdbProvider: "rcsb",
        });

        // 1. Load PDB Structure
        if (CONFIG.pdb) {
          setStatus(`Loading model ${CONFIG.pdb.split('/').pop()}...`, true);
          await viewer.loadStructureFromUrl(CONFIG.pdb, "pdb", false);
        }

        // 2. Load 2mFo-DFc Density Map (Blue wireframe at 1.5 sigma)
        if (CONFIG.map_2fofc) {
          setStatus(`Loading 2mFo-DFc map...`, true);
          try {
            await viewer.loadVolumeFromUrl(
              { url: CONFIG.map_2fofc, format: "ccp4", isBinary: true },
              [
                { type: "relative", value: 1.5, color: 0x3362b2, alpha: 0.45 },
              ]
            );
          } catch (e) {
            console.warn("Could not load 2mFo-DFc map:", e);
          }
        }

        // 3. Load Target Gradient Map (+3.0 sigma Green, -3.0 sigma Red)
        if (CONFIG.gradient_map) {
          setStatus(`Loading target gradient map...`, true);
          try {
            await viewer.loadVolumeFromUrl(
              { url: CONFIG.gradient_map, format: "ccp4", isBinary: true },
              [
                { type: "relative", value: 3.0, color: 0x22c55e, alpha: 0.55 },
                { type: "relative", value: -3.0, color: 0xef4444, alpha: 0.55 },
              ]
            );
          } catch (e) {
            console.warn("Could not load gradient map:", e);
          }
        }

        // 4. Load mFo-DFc if gradient not present
        if (!CONFIG.gradient_map && CONFIG.map_fofc) {
          setStatus(`Loading mFo-DFc difference map...`, true);
          try {
            await viewer.loadVolumeFromUrl(
              { url: CONFIG.map_fofc, format: "ccp4", isBinary: true },
              [
                { type: "relative", value: 3.0, color: 0x22c55e, alpha: 0.55 },
                { type: "relative", value: -3.0, color: 0xef4444, alpha: 0.55 },
              ]
            );
          } catch (e) {
            console.warn("Could not load mFo-DFc map:", e);
          }
        }

        setStatus("Ready. Inspect model & map in 3D.");

        document.getElementById("btn-recenter").addEventListener("click", () => {
          viewer.plugin?.canvas3d?.requestCameraReset();
        });

      } catch (err) {
        console.error("Mol* initialization failed:", err);
        setStatus("Error: " + err.message, false);
      }
    }

    window.addEventListener("DOMContentLoaded", init);
  </script>
</body>
</html>
"""


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
