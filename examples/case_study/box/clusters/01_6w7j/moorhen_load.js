// Moorhen loader for box/clusters/01_6w7j
// Open moorhen.org, press F12, paste this into the Console, press Enter.
// The file server must be running.
(async () => {
  const BASE = "http://localhost:8765/";
  const MODELS = [
  [
    "box/clusters/01_6w7j/structures/6die.pdb",
    "6die"
  ],
  [
    "box/clusters/01_6w7j/structures/6djj.pdb",
    "6djj"
  ],
  [
    "box/clusters/01_6w7j/structures/6n0d.pdb",
    "6n0d"
  ],
  [
    "box/clusters/01_6w7j/structures/6n0n.pdb",
    "6n0n"
  ],
  [
    "box/clusters/01_6w7j/structures/6n0r.pdb",
    "6n0r"
  ],
  [
    "box/clusters/01_6w7j/structures/6n19.pdb",
    "6n19"
  ],
  [
    "box/clusters/01_6w7j/structures/6w7j.pdb",
    "6w7j"
  ],
  [
    "box/clusters/01_6w7j/structures/7ufy.pdb",
    "7ufy"
  ],
  [
    "box/clusters/01_6w7j/structures/7ufz.pdb",
    "7ufz"
  ],
  [
    "box/clusters/01_6w7j/structures/8cvq.pdb",
    "8cvq"
  ],
  [
    "box/clusters/01_6w7j/structures/8v0b.pdb",
    "8v0b"
  ],
  [
    "box/clusters/01_6w7j/structures/8v0c.pdb",
    "8v0c"
  ],
  [
    "box/clusters/01_6w7j/structures/9b3b.pdb",
    "9b3b"
  ]
];
  const MAPS = [
  [
    "box/clusters/01_6w7j/maps/6die.mtz",
    "01_6w7j 6die",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/clusters/01_6w7j/maps/6djj.mtz",
    "01_6w7j 6djj",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/clusters/01_6w7j/maps/6n0d.mtz",
    "01_6w7j 6n0d",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/clusters/01_6w7j/maps/6n0n.mtz",
    "01_6w7j 6n0n",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/clusters/01_6w7j/maps/6n0r.mtz",
    "01_6w7j 6n0r",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/clusters/01_6w7j/maps/6n19.mtz",
    "01_6w7j 6n19",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/clusters/01_6w7j/maps/6w7j.mtz",
    "01_6w7j 6w7j",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/clusters/01_6w7j/maps/7ufy.mtz",
    "01_6w7j 7ufy",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/clusters/01_6w7j/maps/7ufz.mtz",
    "01_6w7j 7ufz",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/clusters/01_6w7j/maps/8cvq.mtz",
    "01_6w7j 8cvq",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/clusters/01_6w7j/maps/8v0b.mtz",
    "01_6w7j 8v0b",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/clusters/01_6w7j/maps/8v0c.mtz",
    "01_6w7j 8v0c",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/clusters/01_6w7j/maps/9b3b.mtz",
    "01_6w7j 9b3b",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/clusters/01_6w7j/residual_svd/average.mtz",
    "01_6w7j average",
    "mtz",
    true,
    true,
    null
  ],
  [
    "box/clusters/01_6w7j/residual_svd/component_01.mtz",
    "01_6w7j component_01",
    "mtz",
    true,
    false,
    0.0005360384238883853
  ],
  [
    "box/clusters/01_6w7j/residual_svd/component_02.mtz",
    "01_6w7j component_02",
    "mtz",
    false,
    false,
    0.0005360385403037071
  ],
  [
    "box/clusters/01_6w7j/residual_svd/component_03.mtz",
    "01_6w7j component_03",
    "mtz",
    false,
    false,
    0.000536038656719029
  ],
  [
    "box/clusters/01_6w7j/residual_svd/component_04.mtz",
    "01_6w7j component_04",
    "mtz",
    false,
    false,
    0.0005360383074730635
  ],
  [
    "box/clusters/01_6w7j/residual_svd/component_05.mtz",
    "01_6w7j component_05",
    "mtz",
    false,
    false,
    0.0005360383656807244
  ],
  [
    "box/clusters/01_6w7j/residual_svd/component_06.mtz",
    "01_6w7j component_06",
    "mtz",
    false,
    false,
    0.0005360383074730635
  ],
  [
    "box/clusters/01_6w7j/residual_svd/component_07.mtz",
    "01_6w7j component_07",
    "mtz",
    false,
    false,
    0.0005360382492654026
  ],
  [
    "box/clusters/01_6w7j/residual_svd/component_08.mtz",
    "01_6w7j component_08",
    "mtz",
    false,
    false,
    0.0005360384238883853
  ],
  [
    "box/clusters/01_6w7j/residual_svd/component_09.mtz",
    "01_6w7j component_09",
    "mtz",
    false,
    false,
    0.0005360387731343508
  ],
  [
    "box/clusters/01_6w7j/residual_svd/component_10.mtz",
    "01_6w7j component_10",
    "mtz",
    false,
    false,
    0.0005360383074730635
  ],
  [
    "box/clusters/01_6w7j/residual_svd/component_11.mtz",
    "01_6w7j component_11",
    "mtz",
    false,
    false,
    0.0005360381328500807
  ],
  [
    "box/clusters/01_6w7j/residual_svd/component_12.mtz",
    "01_6w7j component_12",
    "mtz",
    false,
    false,
    0.0005360382492654026
  ],
  [
    "box/clusters/01_6w7j/residual_svd/power_1-4.ccp4",
    "01_6w7j power_1-4",
    "map",
    false,
    false,
    1.330572485923767
  ],
  [
    "box/clusters/01_6w7j/residual_svd/power_all.ccp4",
    "01_6w7j power_all",
    "map",
    false,
    false,
    1.0129653215408325
  ],
  [
    "box/clusters/01_6w7j/residual_svd/residuals/6die.mtz",
    "01_6w7j residual 6die",
    "mtz",
    false,
    false,
    0.2932097613811493
  ],
  [
    "box/clusters/01_6w7j/residual_svd/residuals/6djj.mtz",
    "01_6w7j residual 6djj",
    "mtz",
    false,
    false,
    0.22922483086585999
  ],
  [
    "box/clusters/01_6w7j/residual_svd/residuals/6n0d.mtz",
    "01_6w7j residual 6n0d",
    "mtz",
    false,
    false,
    0.2477228343486786
  ],
  [
    "box/clusters/01_6w7j/residual_svd/residuals/6n0n.mtz",
    "01_6w7j residual 6n0n",
    "mtz",
    false,
    false,
    0.44354334473609924
  ],
  [
    "box/clusters/01_6w7j/residual_svd/residuals/6n0r.mtz",
    "01_6w7j residual 6n0r",
    "mtz",
    false,
    false,
    0.26758474111557007
  ],
  [
    "box/clusters/01_6w7j/residual_svd/residuals/6n19.mtz",
    "01_6w7j residual 6n19",
    "mtz",
    false,
    false,
    0.2748319208621979
  ],
  [
    "box/clusters/01_6w7j/residual_svd/residuals/6w7j.mtz",
    "01_6w7j residual 6w7j",
    "mtz",
    false,
    false,
    0.27298077940940857
  ],
  [
    "box/clusters/01_6w7j/residual_svd/residuals/7ufy.mtz",
    "01_6w7j residual 7ufy",
    "mtz",
    false,
    false,
    0.2851027548313141
  ],
  [
    "box/clusters/01_6w7j/residual_svd/residuals/7ufz.mtz",
    "01_6w7j residual 7ufz",
    "mtz",
    false,
    false,
    0.28125590085983276
  ],
  [
    "box/clusters/01_6w7j/residual_svd/residuals/8cvq.mtz",
    "01_6w7j residual 8cvq",
    "mtz",
    false,
    false,
    0.3132787048816681
  ],
  [
    "box/clusters/01_6w7j/residual_svd/residuals/8v0b.mtz",
    "01_6w7j residual 8v0b",
    "mtz",
    false,
    false,
    0.23407255113124847
  ],
  [
    "box/clusters/01_6w7j/residual_svd/residuals/8v0c.mtz",
    "01_6w7j residual 8v0c",
    "mtz",
    false,
    false,
    0.24539916217327118
  ],
  [
    "box/clusters/01_6w7j/residual_svd/residuals/9b3b.mtz",
    "01_6w7j residual 9b3b",
    "mtz",
    false,
    false,
    0.28340163826942444
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
