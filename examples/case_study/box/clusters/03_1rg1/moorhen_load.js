// Moorhen loader for box/clusters/03_1rg1
// Open moorhen.org, press F12, paste this into the Console, press Enter.
// The file server must be running.
(async () => {
  const BASE = "http://localhost:8765/";
  const MODELS = [
  [
    "box/clusters/03_1rg1/structures/1mu7.pdb",
    "1mu7"
  ],
  [
    "box/clusters/03_1rg1/structures/1mu9.pdb",
    "1mu9"
  ],
  [
    "box/clusters/03_1rg1/structures/1nop.pdb",
    "1nop"
  ],
  [
    "box/clusters/03_1rg1/structures/1rff.pdb",
    "1rff"
  ],
  [
    "box/clusters/03_1rg1/structures/1rfi.pdb",
    "1rfi"
  ],
  [
    "box/clusters/03_1rg1/structures/1rg1.pdb",
    "1rg1"
  ],
  [
    "box/clusters/03_1rg1/structures/1rg2.pdb",
    "1rg2"
  ],
  [
    "box/clusters/03_1rg1/structures/1rgt.pdb",
    "1rgt"
  ],
  [
    "box/clusters/03_1rg1/structures/1rgu.pdb",
    "1rgu"
  ],
  [
    "box/clusters/03_1rg1/structures/1rh0.pdb",
    "1rh0"
  ]
];
  const MAPS = [
  [
    "box/clusters/03_1rg1/maps/1mu7.mtz",
    "03_1rg1 1mu7",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/clusters/03_1rg1/maps/1mu9.mtz",
    "03_1rg1 1mu9",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/clusters/03_1rg1/maps/1nop.mtz",
    "03_1rg1 1nop",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/clusters/03_1rg1/maps/1rff.mtz",
    "03_1rg1 1rff",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/clusters/03_1rg1/maps/1rfi.mtz",
    "03_1rg1 1rfi",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/clusters/03_1rg1/maps/1rg1.mtz",
    "03_1rg1 1rg1",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/clusters/03_1rg1/maps/1rg2.mtz",
    "03_1rg1 1rg2",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/clusters/03_1rg1/maps/1rgt.mtz",
    "03_1rg1 1rgt",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/clusters/03_1rg1/maps/1rgu.mtz",
    "03_1rg1 1rgu",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/clusters/03_1rg1/maps/1rh0.mtz",
    "03_1rg1 1rh0",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/clusters/03_1rg1/residual_svd/average.mtz",
    "03_1rg1 average",
    "mtz",
    true,
    true,
    null
  ],
  [
    "box/clusters/03_1rg1/residual_svd/component_01.mtz",
    "03_1rg1 component_01",
    "mtz",
    true,
    false,
    0.0005360385403037071
  ],
  [
    "box/clusters/03_1rg1/residual_svd/component_02.mtz",
    "03_1rg1 component_02",
    "mtz",
    false,
    false,
    0.0005360383074730635
  ],
  [
    "box/clusters/03_1rg1/residual_svd/component_03.mtz",
    "03_1rg1 component_03",
    "mtz",
    false,
    false,
    0.0005360385403037071
  ],
  [
    "box/clusters/03_1rg1/residual_svd/component_04.mtz",
    "03_1rg1 component_04",
    "mtz",
    false,
    false,
    0.000536038598511368
  ],
  [
    "box/clusters/03_1rg1/residual_svd/component_05.mtz",
    "03_1rg1 component_05",
    "mtz",
    false,
    false,
    0.0005360383656807244
  ],
  [
    "box/clusters/03_1rg1/residual_svd/component_06.mtz",
    "03_1rg1 component_06",
    "mtz",
    false,
    false,
    0.0005360381328500807
  ],
  [
    "box/clusters/03_1rg1/residual_svd/component_07.mtz",
    "03_1rg1 component_07",
    "mtz",
    false,
    false,
    0.0005360380164347589
  ],
  [
    "box/clusters/03_1rg1/residual_svd/component_08.mtz",
    "03_1rg1 component_08",
    "mtz",
    false,
    false,
    0.0005360383656807244
  ],
  [
    "box/clusters/03_1rg1/residual_svd/component_09.mtz",
    "03_1rg1 component_09",
    "mtz",
    false,
    false,
    0.0005360382492654026
  ],
  [
    "box/clusters/03_1rg1/residual_svd/power_1-4.ccp4",
    "03_1rg1 power_1-4",
    "map",
    false,
    false,
    3.6657755374908447
  ],
  [
    "box/clusters/03_1rg1/residual_svd/power_all.ccp4",
    "03_1rg1 power_all",
    "map",
    false,
    false,
    3.3823649883270264
  ],
  [
    "box/clusters/03_1rg1/residual_svd/residuals/1mu7.mtz",
    "03_1rg1 residual 1mu7",
    "mtz",
    false,
    false,
    0.36409103870391846
  ],
  [
    "box/clusters/03_1rg1/residual_svd/residuals/1mu9.mtz",
    "03_1rg1 residual 1mu9",
    "mtz",
    false,
    false,
    0.2829892039299011
  ],
  [
    "box/clusters/03_1rg1/residual_svd/residuals/1nop.mtz",
    "03_1rg1 residual 1nop",
    "mtz",
    false,
    false,
    0.28816336393356323
  ],
  [
    "box/clusters/03_1rg1/residual_svd/residuals/1rff.mtz",
    "03_1rg1 residual 1rff",
    "mtz",
    false,
    false,
    0.2520790994167328
  ],
  [
    "box/clusters/03_1rg1/residual_svd/residuals/1rfi.mtz",
    "03_1rg1 residual 1rfi",
    "mtz",
    false,
    false,
    0.21979326009750366
  ],
  [
    "box/clusters/03_1rg1/residual_svd/residuals/1rg1.mtz",
    "03_1rg1 residual 1rg1",
    "mtz",
    false,
    false,
    0.29715001583099365
  ],
  [
    "box/clusters/03_1rg1/residual_svd/residuals/1rg2.mtz",
    "03_1rg1 residual 1rg2",
    "mtz",
    false,
    false,
    0.20129063725471497
  ],
  [
    "box/clusters/03_1rg1/residual_svd/residuals/1rgt.mtz",
    "03_1rg1 residual 1rgt",
    "mtz",
    false,
    false,
    0.2640228271484375
  ],
  [
    "box/clusters/03_1rg1/residual_svd/residuals/1rgu.mtz",
    "03_1rg1 residual 1rgu",
    "mtz",
    false,
    false,
    0.3058016300201416
  ],
  [
    "box/clusters/03_1rg1/residual_svd/residuals/1rh0.mtz",
    "03_1rg1 residual 1rh0",
    "mtz",
    false,
    false,
    0.25354301929473877
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
