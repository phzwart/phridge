// Moorhen loader for cell/clusters/02_8v0c
// Open moorhen.org, press F12, paste this into the Console, press Enter.
// The file server must be running.
(async () => {
  const BASE = "http://localhost:8765/";
  const MODELS = [
  [
    "cell/clusters/02_8v0c/structures/6djg.pdb",
    "6djg"
  ],
  [
    "cell/clusters/02_8v0c/structures/6mj5.pdb",
    "6mj5"
  ],
  [
    "cell/clusters/02_8v0c/structures/6w4r.pdb",
    "6w4r"
  ],
  [
    "cell/clusters/02_8v0c/structures/6w7j.pdb",
    "6w7j"
  ],
  [
    "cell/clusters/02_8v0c/structures/6w7l.pdb",
    "6w7l"
  ],
  [
    "cell/clusters/02_8v0c/structures/7ufz.pdb",
    "7ufz"
  ],
  [
    "cell/clusters/02_8v0c/structures/8cvq.pdb",
    "8cvq"
  ],
  [
    "cell/clusters/02_8v0c/structures/8uv1.pdb",
    "8uv1"
  ],
  [
    "cell/clusters/02_8v0c/structures/8uzv.pdb",
    "8uzv"
  ],
  [
    "cell/clusters/02_8v0c/structures/8uzz.pdb",
    "8uzz"
  ],
  [
    "cell/clusters/02_8v0c/structures/8v0c.pdb",
    "8v0c"
  ]
];
  const MAPS = [
  [
    "cell/clusters/02_8v0c/maps/6djg.mtz",
    "02_8v0c 6djg",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/clusters/02_8v0c/maps/6mj5.mtz",
    "02_8v0c 6mj5",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/clusters/02_8v0c/maps/6w4r.mtz",
    "02_8v0c 6w4r",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/clusters/02_8v0c/maps/6w7j.mtz",
    "02_8v0c 6w7j",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/clusters/02_8v0c/maps/6w7l.mtz",
    "02_8v0c 6w7l",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/clusters/02_8v0c/maps/7ufz.mtz",
    "02_8v0c 7ufz",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/clusters/02_8v0c/maps/8cvq.mtz",
    "02_8v0c 8cvq",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/clusters/02_8v0c/maps/8uv1.mtz",
    "02_8v0c 8uv1",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/clusters/02_8v0c/maps/8uzv.mtz",
    "02_8v0c 8uzv",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/clusters/02_8v0c/maps/8uzz.mtz",
    "02_8v0c 8uzz",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/clusters/02_8v0c/maps/8v0c.mtz",
    "02_8v0c 8v0c",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/clusters/02_8v0c/residual_svd/average.mtz",
    "02_8v0c average",
    "mtz",
    true,
    true,
    null
  ],
  [
    "cell/clusters/02_8v0c/residual_svd/component_01.mtz",
    "02_8v0c component_01",
    "mtz",
    true,
    false,
    1.0
  ],
  [
    "cell/clusters/02_8v0c/residual_svd/component_02.mtz",
    "02_8v0c component_02",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/02_8v0c/residual_svd/component_03.mtz",
    "02_8v0c component_03",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/02_8v0c/residual_svd/component_04.mtz",
    "02_8v0c component_04",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/02_8v0c/residual_svd/component_05.mtz",
    "02_8v0c component_05",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/02_8v0c/residual_svd/component_06.mtz",
    "02_8v0c component_06",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/02_8v0c/residual_svd/component_07.mtz",
    "02_8v0c component_07",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/02_8v0c/residual_svd/component_08.mtz",
    "02_8v0c component_08",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/02_8v0c/residual_svd/component_09.mtz",
    "02_8v0c component_09",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/02_8v0c/residual_svd/component_10.mtz",
    "02_8v0c component_10",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/02_8v0c/residual_svd/power_1-4.ccp4",
    "02_8v0c power_1-4",
    "map",
    false,
    false,
    1.3334718942642212
  ],
  [
    "cell/clusters/02_8v0c/residual_svd/power_all.ccp4",
    "02_8v0c power_all",
    "map",
    false,
    false,
    1.0475268363952637
  ],
  [
    "cell/clusters/02_8v0c/residual_svd/residuals/6djg.mtz",
    "02_8v0c residual 6djg",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/02_8v0c/residual_svd/residuals/6mj5.mtz",
    "02_8v0c residual 6mj5",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/02_8v0c/residual_svd/residuals/6w4r.mtz",
    "02_8v0c residual 6w4r",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/02_8v0c/residual_svd/residuals/6w7j.mtz",
    "02_8v0c residual 6w7j",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/02_8v0c/residual_svd/residuals/6w7l.mtz",
    "02_8v0c residual 6w7l",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/02_8v0c/residual_svd/residuals/7ufz.mtz",
    "02_8v0c residual 7ufz",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/02_8v0c/residual_svd/residuals/8cvq.mtz",
    "02_8v0c residual 8cvq",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/02_8v0c/residual_svd/residuals/8uv1.mtz",
    "02_8v0c residual 8uv1",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/02_8v0c/residual_svd/residuals/8uzv.mtz",
    "02_8v0c residual 8uzv",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/02_8v0c/residual_svd/residuals/8uzz.mtz",
    "02_8v0c residual 8uzz",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/clusters/02_8v0c/residual_svd/residuals/8v0c.mtz",
    "02_8v0c residual 8v0c",
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
