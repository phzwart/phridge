// Moorhen loader for box/clusters/02_6dih
// Open moorhen.org, press F12, paste this into the Console, press Enter.
// The file server must be running.
(async () => {
  const BASE = "http://localhost:8765/";
  const MODELS = [
  [
    "box/clusters/02_6dih/structures/6dhu.pdb",
    "6dhu"
  ],
  [
    "box/clusters/02_6dih/structures/6dih.pdb",
    "6dih"
  ],
  [
    "box/clusters/02_6dih/structures/6dim.pdb",
    "6dim"
  ],
  [
    "box/clusters/02_6dih/structures/6djd.pdb",
    "6djd"
  ],
  [
    "box/clusters/02_6dih/structures/6dje.pdb",
    "6dje"
  ],
  [
    "box/clusters/02_6dih/structures/6djf.pdb",
    "6djf"
  ],
  [
    "box/clusters/02_6dih/structures/6djh.pdb",
    "6djh"
  ],
  [
    "box/clusters/02_6dih/structures/6dji.pdb",
    "6dji"
  ],
  [
    "box/clusters/02_6dih/structures/6mj5.pdb",
    "6mj5"
  ],
  [
    "box/clusters/02_6dih/structures/6myz.pdb",
    "6myz"
  ],
  [
    "box/clusters/02_6dih/structures/6n17.pdb",
    "6n17"
  ],
  [
    "box/clusters/02_6dih/structures/6w7k.pdb",
    "6w7k"
  ]
];
  const MAPS = [
  [
    "box/clusters/02_6dih/maps/6dhu.mtz",
    "02_6dih 6dhu",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/clusters/02_6dih/maps/6dih.mtz",
    "02_6dih 6dih",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/clusters/02_6dih/maps/6dim.mtz",
    "02_6dih 6dim",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/clusters/02_6dih/maps/6djd.mtz",
    "02_6dih 6djd",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/clusters/02_6dih/maps/6dje.mtz",
    "02_6dih 6dje",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/clusters/02_6dih/maps/6djf.mtz",
    "02_6dih 6djf",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/clusters/02_6dih/maps/6djh.mtz",
    "02_6dih 6djh",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/clusters/02_6dih/maps/6dji.mtz",
    "02_6dih 6dji",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/clusters/02_6dih/maps/6mj5.mtz",
    "02_6dih 6mj5",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/clusters/02_6dih/maps/6myz.mtz",
    "02_6dih 6myz",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/clusters/02_6dih/maps/6n17.mtz",
    "02_6dih 6n17",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/clusters/02_6dih/maps/6w7k.mtz",
    "02_6dih 6w7k",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/clusters/02_6dih/residual_svd/average.mtz",
    "02_6dih average",
    "mtz",
    true,
    true,
    null
  ],
  [
    "box/clusters/02_6dih/residual_svd/component_01.mtz",
    "02_6dih component_01",
    "mtz",
    true,
    false,
    0.0005360383074730635
  ],
  [
    "box/clusters/02_6dih/residual_svd/component_02.mtz",
    "02_6dih component_02",
    "mtz",
    false,
    false,
    0.0005360383656807244
  ],
  [
    "box/clusters/02_6dih/residual_svd/component_03.mtz",
    "02_6dih component_03",
    "mtz",
    false,
    false,
    0.000536038656719029
  ],
  [
    "box/clusters/02_6dih/residual_svd/component_04.mtz",
    "02_6dih component_04",
    "mtz",
    false,
    false,
    0.0005360383074730635
  ],
  [
    "box/clusters/02_6dih/residual_svd/component_05.mtz",
    "02_6dih component_05",
    "mtz",
    false,
    false,
    0.0005360385403037071
  ],
  [
    "box/clusters/02_6dih/residual_svd/component_06.mtz",
    "02_6dih component_06",
    "mtz",
    false,
    false,
    0.0005360384820960462
  ],
  [
    "box/clusters/02_6dih/residual_svd/component_07.mtz",
    "02_6dih component_07",
    "mtz",
    false,
    false,
    0.000536038598511368
  ],
  [
    "box/clusters/02_6dih/residual_svd/component_08.mtz",
    "02_6dih component_08",
    "mtz",
    false,
    false,
    0.0005360377836041152
  ],
  [
    "box/clusters/02_6dih/residual_svd/component_09.mtz",
    "02_6dih component_09",
    "mtz",
    false,
    false,
    0.0005360387731343508
  ],
  [
    "box/clusters/02_6dih/residual_svd/component_10.mtz",
    "02_6dih component_10",
    "mtz",
    false,
    false,
    0.0005360385403037071
  ],
  [
    "box/clusters/02_6dih/residual_svd/component_11.mtz",
    "02_6dih component_11",
    "mtz",
    false,
    false,
    0.0005360396462492645
  ],
  [
    "box/clusters/02_6dih/residual_svd/power_1-4.ccp4",
    "02_6dih power_1-4",
    "map",
    false,
    false,
    0.9970472455024719
  ],
  [
    "box/clusters/02_6dih/residual_svd/power_all.ccp4",
    "02_6dih power_all",
    "map",
    false,
    false,
    0.7125768661499023
  ],
  [
    "box/clusters/02_6dih/residual_svd/residuals/6dhu.mtz",
    "02_6dih residual 6dhu",
    "mtz",
    false,
    false,
    0.21010883152484894
  ],
  [
    "box/clusters/02_6dih/residual_svd/residuals/6dih.mtz",
    "02_6dih residual 6dih",
    "mtz",
    false,
    false,
    0.2366483062505722
  ],
  [
    "box/clusters/02_6dih/residual_svd/residuals/6dim.mtz",
    "02_6dih residual 6dim",
    "mtz",
    false,
    false,
    0.27844738960266113
  ],
  [
    "box/clusters/02_6dih/residual_svd/residuals/6djd.mtz",
    "02_6dih residual 6djd",
    "mtz",
    false,
    false,
    0.20459912717342377
  ],
  [
    "box/clusters/02_6dih/residual_svd/residuals/6dje.mtz",
    "02_6dih residual 6dje",
    "mtz",
    false,
    false,
    0.26239463686943054
  ],
  [
    "box/clusters/02_6dih/residual_svd/residuals/6djf.mtz",
    "02_6dih residual 6djf",
    "mtz",
    false,
    false,
    0.242808997631073
  ],
  [
    "box/clusters/02_6dih/residual_svd/residuals/6djh.mtz",
    "02_6dih residual 6djh",
    "mtz",
    false,
    false,
    0.27743393182754517
  ],
  [
    "box/clusters/02_6dih/residual_svd/residuals/6dji.mtz",
    "02_6dih residual 6dji",
    "mtz",
    false,
    false,
    0.2559637129306793
  ],
  [
    "box/clusters/02_6dih/residual_svd/residuals/6mj5.mtz",
    "02_6dih residual 6mj5",
    "mtz",
    false,
    false,
    0.28268349170684814
  ],
  [
    "box/clusters/02_6dih/residual_svd/residuals/6myz.mtz",
    "02_6dih residual 6myz",
    "mtz",
    false,
    false,
    0.26801955699920654
  ],
  [
    "box/clusters/02_6dih/residual_svd/residuals/6n17.mtz",
    "02_6dih residual 6n17",
    "mtz",
    false,
    false,
    0.2191886454820633
  ],
  [
    "box/clusters/02_6dih/residual_svd/residuals/6w7k.mtz",
    "02_6dih residual 6w7k",
    "mtz",
    false,
    false,
    0.2507254183292389
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
