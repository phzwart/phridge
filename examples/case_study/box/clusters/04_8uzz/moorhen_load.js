// Moorhen loader for box/clusters/04_8uzz
// Open moorhen.org, press F12, paste this into the Console, press Enter.
// The file server must be running.
(async () => {
  const BASE = "http://localhost:8765/";
  const MODELS = [
  [
    "box/clusters/04_8uzz/structures/6djg.pdb",
    "6djg"
  ],
  [
    "box/clusters/04_8uzz/structures/6mz0.pdb",
    "6mz0"
  ],
  [
    "box/clusters/04_8uzz/structures/6n0o.pdb",
    "6n0o"
  ],
  [
    "box/clusters/04_8uzz/structures/6w4r.pdb",
    "6w4r"
  ],
  [
    "box/clusters/04_8uzz/structures/6w7l.pdb",
    "6w7l"
  ],
  [
    "box/clusters/04_8uzz/structures/8cw2.pdb",
    "8cw2"
  ],
  [
    "box/clusters/04_8uzz/structures/8uv1.pdb",
    "8uv1"
  ],
  [
    "box/clusters/04_8uzz/structures/8uzv.pdb",
    "8uzv"
  ],
  [
    "box/clusters/04_8uzz/structures/8uzz.pdb",
    "8uzz"
  ]
];
  const MAPS = [
  [
    "box/clusters/04_8uzz/maps/6djg.mtz",
    "04_8uzz 6djg",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/clusters/04_8uzz/maps/6mz0.mtz",
    "04_8uzz 6mz0",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/clusters/04_8uzz/maps/6n0o.mtz",
    "04_8uzz 6n0o",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/clusters/04_8uzz/maps/6w4r.mtz",
    "04_8uzz 6w4r",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/clusters/04_8uzz/maps/6w7l.mtz",
    "04_8uzz 6w7l",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/clusters/04_8uzz/maps/8cw2.mtz",
    "04_8uzz 8cw2",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/clusters/04_8uzz/maps/8uv1.mtz",
    "04_8uzz 8uv1",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/clusters/04_8uzz/maps/8uzv.mtz",
    "04_8uzz 8uzv",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/clusters/04_8uzz/maps/8uzz.mtz",
    "04_8uzz 8uzz",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/clusters/04_8uzz/residual_svd/average.mtz",
    "04_8uzz average",
    "mtz",
    true,
    true,
    null
  ],
  [
    "box/clusters/04_8uzz/residual_svd/component_01.mtz",
    "04_8uzz component_01",
    "mtz",
    true,
    false,
    0.000536038598511368
  ],
  [
    "box/clusters/04_8uzz/residual_svd/component_02.mtz",
    "04_8uzz component_02",
    "mtz",
    false,
    false,
    0.0005360381910577416
  ],
  [
    "box/clusters/04_8uzz/residual_svd/component_03.mtz",
    "04_8uzz component_03",
    "mtz",
    false,
    false,
    0.0005360385403037071
  ],
  [
    "box/clusters/04_8uzz/residual_svd/component_04.mtz",
    "04_8uzz component_04",
    "mtz",
    false,
    false,
    0.0005360383656807244
  ],
  [
    "box/clusters/04_8uzz/residual_svd/component_05.mtz",
    "04_8uzz component_05",
    "mtz",
    false,
    false,
    0.0005360383074730635
  ],
  [
    "box/clusters/04_8uzz/residual_svd/component_06.mtz",
    "04_8uzz component_06",
    "mtz",
    false,
    false,
    0.0005360384238883853
  ],
  [
    "box/clusters/04_8uzz/residual_svd/component_07.mtz",
    "04_8uzz component_07",
    "mtz",
    false,
    false,
    0.0005360381910577416
  ],
  [
    "box/clusters/04_8uzz/residual_svd/component_08.mtz",
    "04_8uzz component_08",
    "mtz",
    false,
    false,
    0.0005360389477573335
  ],
  [
    "box/clusters/04_8uzz/residual_svd/power_1-4.ccp4",
    "04_8uzz power_1-4",
    "map",
    false,
    false,
    1.1354683637619019
  ],
  [
    "box/clusters/04_8uzz/residual_svd/power_all.ccp4",
    "04_8uzz power_all",
    "map",
    false,
    false,
    0.966887891292572
  ],
  [
    "box/clusters/04_8uzz/residual_svd/residuals/6djg.mtz",
    "04_8uzz residual 6djg",
    "mtz",
    false,
    false,
    0.300598680973053
  ],
  [
    "box/clusters/04_8uzz/residual_svd/residuals/6mz0.mtz",
    "04_8uzz residual 6mz0",
    "mtz",
    false,
    false,
    0.2935745418071747
  ],
  [
    "box/clusters/04_8uzz/residual_svd/residuals/6n0o.mtz",
    "04_8uzz residual 6n0o",
    "mtz",
    false,
    false,
    0.2937604486942291
  ],
  [
    "box/clusters/04_8uzz/residual_svd/residuals/6w4r.mtz",
    "04_8uzz residual 6w4r",
    "mtz",
    false,
    false,
    0.2539510130882263
  ],
  [
    "box/clusters/04_8uzz/residual_svd/residuals/6w7l.mtz",
    "04_8uzz residual 6w7l",
    "mtz",
    false,
    false,
    0.44193175435066223
  ],
  [
    "box/clusters/04_8uzz/residual_svd/residuals/8cw2.mtz",
    "04_8uzz residual 8cw2",
    "mtz",
    false,
    false,
    0.27156174182891846
  ],
  [
    "box/clusters/04_8uzz/residual_svd/residuals/8uv1.mtz",
    "04_8uzz residual 8uv1",
    "mtz",
    false,
    false,
    0.3550698757171631
  ],
  [
    "box/clusters/04_8uzz/residual_svd/residuals/8uzv.mtz",
    "04_8uzz residual 8uzv",
    "mtz",
    false,
    false,
    0.21802693605422974
  ],
  [
    "box/clusters/04_8uzz/residual_svd/residuals/8uzz.mtz",
    "04_8uzz residual 8uzz",
    "mtz",
    false,
    false,
    0.2217375487089157
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
