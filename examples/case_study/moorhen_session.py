#!/usr/bin/env python3
"""Write a Moorhen loader script for one result directory (box/ or cell/) and serve the files.

The 2mFo−DFc maps, one per structure plus the average, are loaded as density
maps in blue at ``--density-level`` σ. Residuals, SVD components, and power maps
are loaded as difference maps, positive green and negative red, contoured at
±``--sigma`` times their standard deviation inside the mask. MTZ files are
read from ``FWT`` / ``PHWT``; maps without an MTZ are read from CCP4. Models are
drawn as ball and stick, without ribbons. Only the average and component 1 are
shown at load; the others, power maps included, load hidden.

The loader is written as ``moorhen_load.js`` in the directory. While this script
is serving, paste the ``fetch(...)`` line it prints into the browser console
(F12) on moorhen.org; the Interactive scripting box on moorhen.org fails before
running any script.

Usage::

    source .venv/bin/activate
    python examples/case_study/moorhen_session.py cell/clusters/01_6w7j
    python examples/case_study/moorhen_session.py cell/fwt_with_water --no-serve
    python examples/case_study/moorhen_session.py box/clusters/01_6w7j --no-maps
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SHOWN = {"average", "component_01"}

_TEMPLATE = """// Moorhen loader for {title}
// Open moorhen.org, press F12, paste this into the Console, press Enter.
// The file server must be running.
(async () => {{
  const BASE = {base};
  const MODELS = {models};
  const MAPS = {maps};
  const SIGMA = {sigma};
  const GREEN = {{ r: 0, g: 204, b: 0 }};
  const RED = {{ r: 230, g: 0, b: 0 }};
  const BLUE = {{ r: 51, g: 102, b: 230 }};
  const DENSITY_LEVEL = {density_level};
  const root = document.querySelector("canvas");
  let fiber = root && root[Object.keys(root).find(key => key.startsWith("__reactFiber"))];
  while (fiber && !(fiber.memoizedProps && fiber.memoizedProps.value && fiber.memoizedProps.value.instance)) {{
    fiber = fiber.return;
  }}
  if (!fiber) throw new Error("Moorhen instance not found; wait for Moorhen to finish loading");
  const moorhen = fiber.memoizedProps.value.instance;
  const style = {{ positiveColour: GREEN, negativeColour: RED }};
  moorhen._defaultStyle = "CBs";
  await moorhen.files.loadFiles(MODELS.map(([path, name]) => ({{ url: BASE + path, filename: name + path.slice(path.lastIndexOf(".")) }})));
  for (const molecule of moorhen.getMoleculeList()) {{
    for (const representation of molecule.representations.filter(item => item.style === "CBs")) {{
      representation.setBondOptions({{ ...molecule.defaultBondOptions, width: 0.12, atomRadiusBondRatio: 2.5 }});
      await representation.redraw();
    }}
  }}
  console.log("models", MODELS.length);
  for (const [path, name, kind, show, density, sigma] of MAPS) {{
    let loaded;
    if (kind === "mtz") {{
      loaded = await moorhen.files.loadMtz(BASE + path, [{{
        name, columns: {{ F: "FWT", PHI: "PHWT" }}, isDifference: !density, calcStructFact: false,
        style: density ? {{ colour: BLUE }} : style, visible: show,
      }}]);
    }} else {{
      loaded = await moorhen.files.loadFiles([{{ url: BASE + path, filename: name.replace(/ /g, "_") + "_diff.ccp4" }}]);
    }}
    const map = moorhen.getMap(loaded.find(item => item.type === "map").uniqueID);
    const level = density ? DENSITY_LEVEL : SIGMA * (sigma || await map.fetchMapRmsd());
    if (!density) {{
      moorhen.store.dispatch({{ type: "mapContourSettings/setPositiveMapColours", payload: {{ molNo: map.molNo, rgb: GREEN }} }});
      moorhen.store.dispatch({{ type: "mapContourSettings/setNegativeMapColours", payload: {{ molNo: map.molNo, rgb: RED }} }});
    }}
    moorhen.store.dispatch({{ type: "mapContourSettings/setContourLevel", payload: {{ molNo: map.molNo, contourLevel: level }} }});
    if (show && !density) {{
      await map.fetchDiffMapColourAndRedraw("positiveDiffColour");
      await map.fetchDiffMapColourAndRedraw("negativeDiffColour");
    }} else if (show) {{
      await map.fetchColourAndRedraw();
    }} else {{
      moorhen.store.dispatch({{ type: "mapContourSettings/hideMap", payload: {{ molNo: map.molNo }} }});
    }}
    console.log("map", name, "level", level.toFixed(3));
  }}
}})();
"""


class _CorsHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Access-Control-Allow-Headers", "*")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()


def _relative(path: Path) -> str:
    return path.relative_to(_ROOT).as_posix()


def _masked_sigma(path: Path) -> float:
    """Standard deviation of a CCP4 map over its nonzero voxels, which are the mask."""
    import numpy as np

    with path.open("rb") as handle:
        header = np.frombuffer(handle.read(1024), dtype="<i4")
        handle.seek(1024 + int(header[23]))
        values = np.frombuffer(handle.read(), dtype="<f4")
    inside = values[values != 0.0]
    return float(inside.std()) if inside.size else 0.0


def _map_entry(path: Path, name: str, density: bool = False) -> list:
    """MTZ when there is one, else CCP4. MTZ maps without a CCP4 beside them are already at unit σ in the mask."""
    mtz, ccp4 = path.with_suffix(".mtz"), path.with_suffix(".ccp4")
    chosen, kind = (mtz, "mtz") if mtz.is_file() else (ccp4, "map")
    density = density and kind == "mtz"
    sigma = None if density else (_masked_sigma(ccp4) if ccp4.is_file() else 1.0)
    return [_relative(chosen), name, kind, path.stem in _SHOWN, density, sigma]


def _maps(folder: Path, pattern: str) -> list[Path]:
    """Maps matching ``pattern`` (no suffix), one per stem."""
    stems = {path.with_suffix("") for suffix in (".mtz", ".ccp4") for path in folder.glob(pattern + suffix)}
    return sorted(stems)


def collect(directory: Path) -> tuple[list[list], list[list]]:
    """Models and maps under ``directory``, in loading order."""
    maps = [_map_entry(path, f"{directory.name} {path.name}", density=True) for path in _maps(directory / "maps", "*")]
    svd = directory / "residual_svd" if (directory / "residual_svd").is_dir() else directory
    maps += [_map_entry(path, f"{directory.name} average", density=True) for path in _maps(svd, "average")]
    for pattern in ("component_*", "power_*"):
        maps += [_map_entry(path, f"{directory.name} {path.name}") for path in _maps(svd, pattern)]
    maps += [_map_entry(path, f"{directory.name} residual {path.name}") for path in _maps(svd / "residuals", "*")]
    structures = directory / "structures"
    if not structures.is_dir():
        wanted = {path.name for path in _maps(directory / "maps", "*")}
        top = next(folder for folder in (directory, *directory.parents) if (folder / "models").is_dir())
        candidates = [path for path in (top / "models").iterdir() if path.stem in wanted]
    else:
        candidates = list(structures.iterdir())
    models = [[_relative(path), path.stem] for path in sorted(candidates) if path.suffix in {".pdb", ".cif"}]
    return models, maps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Moorhen loader for one result directory.")
    parser.add_argument("directory", type=Path, help="e.g. cell/clusters/01_6w7j, cell/fwt_with_water, box/fwt_with_water")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--sigma", type=float, default=4.0, help="difference maps, in σ inside the mask")
    parser.add_argument("--density-level", type=float, default=1.25, help="2mFo-DFc maps, in σ inside the mask")
    parser.add_argument("--no-serve", action="store_true")
    parser.add_argument("--no-maps", action="store_true", help="load the structures only")
    args = parser.parse_args(argv)
    directory = args.directory.resolve()
    if not directory.is_dir():
        directory = (_ROOT / args.directory).resolve()
    if not directory.is_dir() or not directory.is_relative_to(_ROOT):
        parser.error(f"{args.directory} is not a directory under {_ROOT}")
    models, maps = collect(directory)
    if args.no_maps:
        maps = []
    script = _TEMPLATE.format(
        title=_relative(directory),
        base=json.dumps(f"http://localhost:{args.port}/"),
        models=json.dumps(models, indent=2),
        maps=json.dumps(maps, indent=2),
        sigma=args.sigma,
        density_level=args.density_level,
    )
    destination = directory / "moorhen_load.js"
    destination.write_text(script, encoding="utf-8")
    print(f"wrote {destination}: {len(models)} models, {len(maps)} maps")
    if args.no_serve:
        return 0
    handler = functools.partial(_CorsHandler, directory=str(_ROOT))
    url = f"http://localhost:{args.port}/{_relative(destination)}"
    print(f"serving {_ROOT} on http://localhost:{args.port}/ (Ctrl-C to stop)")
    print("paste into the moorhen.org browser console:")
    print(f'fetch("{url}").then(r => r.text()).then(eval)')
    http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
