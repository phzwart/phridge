// Moorhen loader for cell/clusters/04_6djd
// Open moorhen.org, press F12, paste this into the Console, press Enter.
// The file server must be running.
(async () => {
  const BASE = "http://localhost:8765/";
  const MODELS = [
  [
    "cell/clusters/04_6djd/structures/6dih.pdb",
    "6dih"
  ],
  [
    "cell/clusters/04_6djd/structures/6dim.pdb",
    "6dim"
  ],
  [
    "cell/clusters/04_6djd/structures/6djd.pdb",
    "6djd"
  ],
  [
    "cell/clusters/04_6djd/structures/6dje.pdb",
    "6dje"
  ],
  [
    "cell/clusters/04_6djd/structures/6djh.pdb",
    "6djh"
  ],
  [
    "cell/clusters/04_6djd/structures/6dji.pdb",
    "6dji"
  ],
  [
    "cell/clusters/04_6djd/structures/6myz.pdb",
    "6myz"
  ],
  [
    "cell/clusters/04_6djd/structures/6n0o.pdb",
    "6n0o"
  ],
  [
    "cell/clusters/04_6djd/structures/6n17.pdb",
    "6n17"
  ]
];
  const MAPS = [
  [
    "cell/clusters/04_6djd/maps/6dih.mtz",
    "04_6djd 6dih",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/clusters/04_6djd/maps/6dim.mtz",
    "04_6djd 6dim",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/clusters/04_6djd/maps/6djd.mtz",
    "04_6djd 6djd",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/clusters/04_6djd/maps/6dje.mtz",
    "04_6djd 6dje",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/clusters/04_6djd/maps/6djh.mtz",
    "04_6djd 6djh",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/clusters/04_6djd/maps/6dji.mtz",
    "04_6djd 6dji",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/clusters/04_6djd/maps/6myz.mtz",
    "04_6djd 6myz",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/clusters/04_6djd/maps/6n0o.mtz",
    "04_6djd 6n0o",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/clusters/04_6djd/maps/6n17.mtz",
    "04_6djd 6n17",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/clusters/04_6djd/residual_svd/average.mtz",
    "04_6djd average",
    "mtz",
    true,
    true,
    null
  ],
  [
    "cell/clusters/04_6djd/residual_svd/component_01.mtz",
    "04_6djd component_01",
    "mtz",
    true,
    false,
    1.0
  ],
  [
    "cell/clusters/04_6djd/residual_svd/component_02.mtz",
    "04_6djd component_02",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/04_6djd/residual_svd/component_03.mtz",
    "04_6djd component_03",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/04_6djd/residual_svd/component_04.mtz",
    "04_6djd component_04",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/04_6djd/residual_svd/component_05.mtz",
    "04_6djd component_05",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/04_6djd/residual_svd/component_06.mtz",
    "04_6djd component_06",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/04_6djd/residual_svd/component_07.mtz",
    "04_6djd component_07",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/04_6djd/residual_svd/component_08.mtz",
    "04_6djd component_08",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/04_6djd/residual_svd/power_1-4.ccp4",
    "04_6djd power_1-4",
    "map",
    false,
    false,
    1.1481807231903076
  ],
  [
    "cell/clusters/04_6djd/residual_svd/power_all.ccp4",
    "04_6djd power_all",
    "map",
    false,
    false,
    0.9411011934280396
  ],
  [
    "cell/clusters/04_6djd/residual_svd/residuals/6dih.mtz",
    "04_6djd residual 6dih",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/04_6djd/residual_svd/residuals/6dim.mtz",
    "04_6djd residual 6dim",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/04_6djd/residual_svd/residuals/6djd.mtz",
    "04_6djd residual 6djd",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/04_6djd/residual_svd/residuals/6dje.mtz",
    "04_6djd residual 6dje",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/04_6djd/residual_svd/residuals/6djh.mtz",
    "04_6djd residual 6djh",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/04_6djd/residual_svd/residuals/6dji.mtz",
    "04_6djd residual 6dji",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/04_6djd/residual_svd/residuals/6myz.mtz",
    "04_6djd residual 6myz",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/04_6djd/residual_svd/residuals/6n0o.mtz",
    "04_6djd residual 6n0o",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/04_6djd/residual_svd/residuals/6n17.mtz",
    "04_6djd residual 6n17",
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
