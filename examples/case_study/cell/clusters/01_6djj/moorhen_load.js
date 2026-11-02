// Moorhen loader for cell/clusters/01_6djj
// Open moorhen.org, press F12, paste this into the Console, press Enter.
// The file server must be running.
(async () => {
  const BASE = "http://localhost:8765/";
  const MODELS = [
  [
    "cell/clusters/01_6djj/structures/6dhu.pdb",
    "6dhu"
  ],
  [
    "cell/clusters/01_6djj/structures/6die.pdb",
    "6die"
  ],
  [
    "cell/clusters/01_6djj/structures/6djf.pdb",
    "6djf"
  ],
  [
    "cell/clusters/01_6djj/structures/6djj.pdb",
    "6djj"
  ],
  [
    "cell/clusters/01_6djj/structures/6mz0.pdb",
    "6mz0"
  ],
  [
    "cell/clusters/01_6djj/structures/6n0d.pdb",
    "6n0d"
  ],
  [
    "cell/clusters/01_6djj/structures/6n0n.pdb",
    "6n0n"
  ],
  [
    "cell/clusters/01_6djj/structures/6n0r.pdb",
    "6n0r"
  ],
  [
    "cell/clusters/01_6djj/structures/6n19.pdb",
    "6n19"
  ],
  [
    "cell/clusters/01_6djj/structures/6w7k.pdb",
    "6w7k"
  ],
  [
    "cell/clusters/01_6djj/structures/7ufy.pdb",
    "7ufy"
  ],
  [
    "cell/clusters/01_6djj/structures/8cw2.pdb",
    "8cw2"
  ],
  [
    "cell/clusters/01_6djj/structures/8v0b.pdb",
    "8v0b"
  ],
  [
    "cell/clusters/01_6djj/structures/9b3b.pdb",
    "9b3b"
  ]
];
  const MAPS = [
  [
    "cell/clusters/01_6djj/maps/6dhu.mtz",
    "01_6djj 6dhu",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/clusters/01_6djj/maps/6die.mtz",
    "01_6djj 6die",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/clusters/01_6djj/maps/6djf.mtz",
    "01_6djj 6djf",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/clusters/01_6djj/maps/6djj.mtz",
    "01_6djj 6djj",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/clusters/01_6djj/maps/6mz0.mtz",
    "01_6djj 6mz0",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/clusters/01_6djj/maps/6n0d.mtz",
    "01_6djj 6n0d",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/clusters/01_6djj/maps/6n0n.mtz",
    "01_6djj 6n0n",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/clusters/01_6djj/maps/6n0r.mtz",
    "01_6djj 6n0r",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/clusters/01_6djj/maps/6n19.mtz",
    "01_6djj 6n19",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/clusters/01_6djj/maps/6w7k.mtz",
    "01_6djj 6w7k",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/clusters/01_6djj/maps/7ufy.mtz",
    "01_6djj 7ufy",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/clusters/01_6djj/maps/8cw2.mtz",
    "01_6djj 8cw2",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/clusters/01_6djj/maps/8v0b.mtz",
    "01_6djj 8v0b",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/clusters/01_6djj/maps/9b3b.mtz",
    "01_6djj 9b3b",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/clusters/01_6djj/residual_svd/average.mtz",
    "01_6djj average",
    "mtz",
    true,
    true,
    null
  ],
  [
    "cell/clusters/01_6djj/residual_svd/component_01.mtz",
    "01_6djj component_01",
    "mtz",
    true,
    false,
    1.0
  ],
  [
    "cell/clusters/01_6djj/residual_svd/component_02.mtz",
    "01_6djj component_02",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/01_6djj/residual_svd/component_03.mtz",
    "01_6djj component_03",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/01_6djj/residual_svd/component_04.mtz",
    "01_6djj component_04",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/01_6djj/residual_svd/component_05.mtz",
    "01_6djj component_05",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/01_6djj/residual_svd/component_06.mtz",
    "01_6djj component_06",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/01_6djj/residual_svd/component_07.mtz",
    "01_6djj component_07",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/01_6djj/residual_svd/component_08.mtz",
    "01_6djj component_08",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/01_6djj/residual_svd/component_09.mtz",
    "01_6djj component_09",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/01_6djj/residual_svd/component_10.mtz",
    "01_6djj component_10",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/01_6djj/residual_svd/component_11.mtz",
    "01_6djj component_11",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/01_6djj/residual_svd/component_12.mtz",
    "01_6djj component_12",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/01_6djj/residual_svd/component_13.mtz",
    "01_6djj component_13",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/01_6djj/residual_svd/power_1-4.ccp4",
    "01_6djj power_1-4",
    "map",
    false,
    false,
    1.3920331001281738
  ],
  [
    "cell/clusters/01_6djj/residual_svd/power_all.ccp4",
    "01_6djj power_all",
    "map",
    false,
    false,
    1.0239802598953247
  ],
  [
    "cell/clusters/01_6djj/residual_svd/residuals/6dhu.mtz",
    "01_6djj residual 6dhu",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/01_6djj/residual_svd/residuals/6die.mtz",
    "01_6djj residual 6die",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/01_6djj/residual_svd/residuals/6djf.mtz",
    "01_6djj residual 6djf",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/01_6djj/residual_svd/residuals/6djj.mtz",
    "01_6djj residual 6djj",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/01_6djj/residual_svd/residuals/6mz0.mtz",
    "01_6djj residual 6mz0",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/01_6djj/residual_svd/residuals/6n0d.mtz",
    "01_6djj residual 6n0d",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/01_6djj/residual_svd/residuals/6n0n.mtz",
    "01_6djj residual 6n0n",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/01_6djj/residual_svd/residuals/6n0r.mtz",
    "01_6djj residual 6n0r",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/01_6djj/residual_svd/residuals/6n19.mtz",
    "01_6djj residual 6n19",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/01_6djj/residual_svd/residuals/6w7k.mtz",
    "01_6djj residual 6w7k",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/01_6djj/residual_svd/residuals/7ufy.mtz",
    "01_6djj residual 7ufy",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/01_6djj/residual_svd/residuals/8cw2.mtz",
    "01_6djj residual 8cw2",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/01_6djj/residual_svd/residuals/8v0b.mtz",
    "01_6djj residual 8v0b",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/01_6djj/residual_svd/residuals/9b3b.mtz",
    "01_6djj residual 9b3b",
    "mtz",
    false,
    false,
    1.0
  ]
];
  const SIGMA = 4.0;
  const GREEN = { r: 0, g: 204, b: 0 };
  const RED = { r: 230, g: 0, b: 0 };
  const BLUE = { r: 51, g: 102, b: 230 };
  const DENSITY_LEVEL = 1.25;
  const root = document.querySelector("canvas");
  let fiber = root && root[Object.keys(root).find(key => key.startsWith("__reactFiber"))];
  while (fiber && !(fiber.memoizedProps && fiber.memoizedProps.value && fiber.memoizedProps.value.instance)) {
    fiber = fiber.return;
  }
  if (!fiber) throw new Error("Moorhen instance not found; wait for Moorhen to finish loading");
  const moorhen = fiber.memoizedProps.value.instance;
  const style = { positiveColour: GREEN, negativeColour: RED };
  moorhen._defaultStyle = "CBs";
  await moorhen.files.loadFiles(MODELS.map(([path, name]) => ({ url: BASE + path, filename: name + path.slice(path.lastIndexOf(".")) })));
  for (const molecule of moorhen.getMoleculeList()) {
    for (const representation of molecule.representations.filter(item => item.style === "CBs")) {
      representation.setBondOptions({ ...molecule.defaultBondOptions, width: 0.12, atomRadiusBondRatio: 2.5 });
      await representation.redraw();
    }
  }
  console.log("models", MODELS.length);
  for (const [path, name, kind, show, density, sigma] of MAPS) {
    let loaded;
    if (kind === "mtz") {
      loaded = await moorhen.files.loadMtz(BASE + path, [{
        name, columns: { F: "FWT", PHI: "PHWT" }, isDifference: !density, calcStructFact: false,
        style: density ? { colour: BLUE } : style, visible: show,
      }]);
    } else {
      loaded = await moorhen.files.loadFiles([{ url: BASE + path, filename: name.replace(/ /g, "_") + "_diff.ccp4" }]);
    }
    const map = moorhen.getMap(loaded.find(item => item.type === "map").uniqueID);
    const level = density ? DENSITY_LEVEL : SIGMA * (sigma || await map.fetchMapRmsd());
    if (!density) {
      moorhen.store.dispatch({ type: "mapContourSettings/setPositiveMapColours", payload: { molNo: map.molNo, rgb: GREEN } });
      moorhen.store.dispatch({ type: "mapContourSettings/setNegativeMapColours", payload: { molNo: map.molNo, rgb: RED } });
    }
    moorhen.store.dispatch({ type: "mapContourSettings/setContourLevel", payload: { molNo: map.molNo, contourLevel: level } });
    if (show && !density) {
      await map.fetchDiffMapColourAndRedraw("positiveDiffColour");
      await map.fetchDiffMapColourAndRedraw("negativeDiffColour");
    } else if (show) {
      await map.fetchColourAndRedraw();
    } else {
      moorhen.store.dispatch({ type: "mapContourSettings/hideMap", payload: { molNo: map.molNo } });
    }
    console.log("map", name, "level", level.toFixed(3));
  }
})();
