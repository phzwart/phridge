// Moorhen loader for cell/fwt_no_water
// Open moorhen.org, press F12, paste this into the Console, press Enter.
// The file server must be running.
(async () => {
  const BASE = "http://localhost:8765/";
  const MODELS = [
  [
    "cell/models/1mu7.pdb",
    "1mu7"
  ],
  [
    "cell/models/1mu9.pdb",
    "1mu9"
  ],
  [
    "cell/models/1nop.pdb",
    "1nop"
  ],
  [
    "cell/models/1rff.pdb",
    "1rff"
  ],
  [
    "cell/models/1rfi.pdb",
    "1rfi"
  ],
  [
    "cell/models/1rg1.pdb",
    "1rg1"
  ],
  [
    "cell/models/1rg2.pdb",
    "1rg2"
  ],
  [
    "cell/models/1rgt.pdb",
    "1rgt"
  ],
  [
    "cell/models/1rgu.pdb",
    "1rgu"
  ],
  [
    "cell/models/1rh0.pdb",
    "1rh0"
  ],
  [
    "cell/models/6dhu.pdb",
    "6dhu"
  ],
  [
    "cell/models/6die.pdb",
    "6die"
  ],
  [
    "cell/models/6dih.pdb",
    "6dih"
  ],
  [
    "cell/models/6dim.pdb",
    "6dim"
  ],
  [
    "cell/models/6djd.pdb",
    "6djd"
  ],
  [
    "cell/models/6dje.pdb",
    "6dje"
  ],
  [
    "cell/models/6djf.pdb",
    "6djf"
  ],
  [
    "cell/models/6djg.pdb",
    "6djg"
  ],
  [
    "cell/models/6djh.pdb",
    "6djh"
  ],
  [
    "cell/models/6dji.pdb",
    "6dji"
  ],
  [
    "cell/models/6djj.pdb",
    "6djj"
  ],
  [
    "cell/models/6mj5.pdb",
    "6mj5"
  ],
  [
    "cell/models/6myz.pdb",
    "6myz"
  ],
  [
    "cell/models/6mz0.pdb",
    "6mz0"
  ],
  [
    "cell/models/6n0d.pdb",
    "6n0d"
  ],
  [
    "cell/models/6n0n.pdb",
    "6n0n"
  ],
  [
    "cell/models/6n0o.pdb",
    "6n0o"
  ],
  [
    "cell/models/6n0r.pdb",
    "6n0r"
  ],
  [
    "cell/models/6n17.pdb",
    "6n17"
  ],
  [
    "cell/models/6n19.pdb",
    "6n19"
  ],
  [
    "cell/models/6w4r.pdb",
    "6w4r"
  ],
  [
    "cell/models/6w7j.pdb",
    "6w7j"
  ],
  [
    "cell/models/6w7k.pdb",
    "6w7k"
  ],
  [
    "cell/models/6w7l.pdb",
    "6w7l"
  ],
  [
    "cell/models/7ufy.pdb",
    "7ufy"
  ],
  [
    "cell/models/7ufz.pdb",
    "7ufz"
  ],
  [
    "cell/models/8cvq.pdb",
    "8cvq"
  ],
  [
    "cell/models/8cw2.pdb",
    "8cw2"
  ],
  [
    "cell/models/8uv1.pdb",
    "8uv1"
  ],
  [
    "cell/models/8uzv.pdb",
    "8uzv"
  ],
  [
    "cell/models/8uzz.pdb",
    "8uzz"
  ],
  [
    "cell/models/8v0b.pdb",
    "8v0b"
  ],
  [
    "cell/models/8v0c.pdb",
    "8v0c"
  ],
  [
    "cell/models/9b3b.pdb",
    "9b3b"
  ]
];
  const MAPS = [
  [
    "cell/fwt_no_water/maps/1mu7.mtz",
    "fwt_no_water 1mu7",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/fwt_no_water/maps/1mu9.mtz",
    "fwt_no_water 1mu9",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/fwt_no_water/maps/1nop.mtz",
    "fwt_no_water 1nop",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/fwt_no_water/maps/1rff.mtz",
    "fwt_no_water 1rff",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/fwt_no_water/maps/1rfi.mtz",
    "fwt_no_water 1rfi",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/fwt_no_water/maps/1rg1.mtz",
    "fwt_no_water 1rg1",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/fwt_no_water/maps/1rg2.mtz",
    "fwt_no_water 1rg2",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/fwt_no_water/maps/1rgt.mtz",
    "fwt_no_water 1rgt",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/fwt_no_water/maps/1rgu.mtz",
    "fwt_no_water 1rgu",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/fwt_no_water/maps/1rh0.mtz",
    "fwt_no_water 1rh0",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/fwt_no_water/maps/6dhu.mtz",
    "fwt_no_water 6dhu",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/fwt_no_water/maps/6die.mtz",
    "fwt_no_water 6die",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/fwt_no_water/maps/6dih.mtz",
    "fwt_no_water 6dih",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/fwt_no_water/maps/6dim.mtz",
    "fwt_no_water 6dim",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/fwt_no_water/maps/6djd.mtz",
    "fwt_no_water 6djd",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/fwt_no_water/maps/6dje.mtz",
    "fwt_no_water 6dje",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/fwt_no_water/maps/6djf.mtz",
    "fwt_no_water 6djf",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/fwt_no_water/maps/6djg.mtz",
    "fwt_no_water 6djg",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/fwt_no_water/maps/6djh.mtz",
    "fwt_no_water 6djh",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/fwt_no_water/maps/6dji.mtz",
    "fwt_no_water 6dji",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/fwt_no_water/maps/6djj.mtz",
    "fwt_no_water 6djj",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/fwt_no_water/maps/6mj5.mtz",
    "fwt_no_water 6mj5",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/fwt_no_water/maps/6myz.mtz",
    "fwt_no_water 6myz",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/fwt_no_water/maps/6mz0.mtz",
    "fwt_no_water 6mz0",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/fwt_no_water/maps/6n0d.mtz",
    "fwt_no_water 6n0d",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/fwt_no_water/maps/6n0n.mtz",
    "fwt_no_water 6n0n",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/fwt_no_water/maps/6n0o.mtz",
    "fwt_no_water 6n0o",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/fwt_no_water/maps/6n0r.mtz",
    "fwt_no_water 6n0r",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/fwt_no_water/maps/6n17.mtz",
    "fwt_no_water 6n17",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/fwt_no_water/maps/6n19.mtz",
    "fwt_no_water 6n19",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/fwt_no_water/maps/6w4r.mtz",
    "fwt_no_water 6w4r",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/fwt_no_water/maps/6w7j.mtz",
    "fwt_no_water 6w7j",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/fwt_no_water/maps/6w7k.mtz",
    "fwt_no_water 6w7k",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/fwt_no_water/maps/6w7l.mtz",
    "fwt_no_water 6w7l",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/fwt_no_water/maps/7ufy.mtz",
    "fwt_no_water 7ufy",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/fwt_no_water/maps/7ufz.mtz",
    "fwt_no_water 7ufz",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/fwt_no_water/maps/8cvq.mtz",
    "fwt_no_water 8cvq",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/fwt_no_water/maps/8cw2.mtz",
    "fwt_no_water 8cw2",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/fwt_no_water/maps/8uv1.mtz",
    "fwt_no_water 8uv1",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/fwt_no_water/maps/8uzv.mtz",
    "fwt_no_water 8uzv",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/fwt_no_water/maps/8uzz.mtz",
    "fwt_no_water 8uzz",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/fwt_no_water/maps/8v0b.mtz",
    "fwt_no_water 8v0b",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/fwt_no_water/maps/8v0c.mtz",
    "fwt_no_water 8v0c",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/fwt_no_water/maps/9b3b.mtz",
    "fwt_no_water 9b3b",
    "mtz",
    false,
    true,
    null
  ],
  [
    "cell/fwt_no_water/residual_svd/average.mtz",
    "fwt_no_water average",
    "mtz",
    true,
    true,
    null
  ],
  [
    "cell/fwt_no_water/residual_svd/component_01.mtz",
    "fwt_no_water component_01",
    "mtz",
    true,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/component_02.mtz",
    "fwt_no_water component_02",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/component_03.mtz",
    "fwt_no_water component_03",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/component_04.mtz",
    "fwt_no_water component_04",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/component_05.mtz",
    "fwt_no_water component_05",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/component_06.mtz",
    "fwt_no_water component_06",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/component_07.mtz",
    "fwt_no_water component_07",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/component_08.mtz",
    "fwt_no_water component_08",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/component_09.mtz",
    "fwt_no_water component_09",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/component_10.mtz",
    "fwt_no_water component_10",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/component_11.mtz",
    "fwt_no_water component_11",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/component_12.mtz",
    "fwt_no_water component_12",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/component_13.mtz",
    "fwt_no_water component_13",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/component_14.mtz",
    "fwt_no_water component_14",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/component_15.mtz",
    "fwt_no_water component_15",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/component_16.mtz",
    "fwt_no_water component_16",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/component_17.mtz",
    "fwt_no_water component_17",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/component_18.mtz",
    "fwt_no_water component_18",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/component_19.mtz",
    "fwt_no_water component_19",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/component_20.mtz",
    "fwt_no_water component_20",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/component_21.mtz",
    "fwt_no_water component_21",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/component_22.mtz",
    "fwt_no_water component_22",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/component_23.mtz",
    "fwt_no_water component_23",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/component_24.mtz",
    "fwt_no_water component_24",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/component_25.mtz",
    "fwt_no_water component_25",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/component_26.mtz",
    "fwt_no_water component_26",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/component_27.mtz",
    "fwt_no_water component_27",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/component_28.mtz",
    "fwt_no_water component_28",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/component_29.mtz",
    "fwt_no_water component_29",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/component_30.mtz",
    "fwt_no_water component_30",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/component_31.mtz",
    "fwt_no_water component_31",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/component_32.mtz",
    "fwt_no_water component_32",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/component_33.mtz",
    "fwt_no_water component_33",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/component_34.mtz",
    "fwt_no_water component_34",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/component_35.mtz",
    "fwt_no_water component_35",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/component_36.mtz",
    "fwt_no_water component_36",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/component_37.mtz",
    "fwt_no_water component_37",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/component_38.mtz",
    "fwt_no_water component_38",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/component_39.mtz",
    "fwt_no_water component_39",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/component_40.mtz",
    "fwt_no_water component_40",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/component_41.mtz",
    "fwt_no_water component_41",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/component_42.mtz",
    "fwt_no_water component_42",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/component_43.mtz",
    "fwt_no_water component_43",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/power_1-4.ccp4",
    "fwt_no_water power_1-4",
    "map",
    false,
    false,
    1.8432574272155762
  ],
  [
    "cell/fwt_no_water/residual_svd/power_all.ccp4",
    "fwt_no_water power_all",
    "map",
    false,
    false,
    1.4668941497802734
  ],
  [
    "cell/fwt_no_water/residual_svd/residuals/1mu7.mtz",
    "fwt_no_water residual 1mu7",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/residuals/1mu9.mtz",
    "fwt_no_water residual 1mu9",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/residuals/1nop.mtz",
    "fwt_no_water residual 1nop",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/residuals/1rff.mtz",
    "fwt_no_water residual 1rff",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/residuals/1rfi.mtz",
    "fwt_no_water residual 1rfi",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/residuals/1rg1.mtz",
    "fwt_no_water residual 1rg1",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/residuals/1rg2.mtz",
    "fwt_no_water residual 1rg2",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/residuals/1rgt.mtz",
    "fwt_no_water residual 1rgt",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/residuals/1rgu.mtz",
    "fwt_no_water residual 1rgu",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/residuals/1rh0.mtz",
    "fwt_no_water residual 1rh0",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/residuals/6dhu.mtz",
    "fwt_no_water residual 6dhu",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/residuals/6die.mtz",
    "fwt_no_water residual 6die",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/residuals/6dih.mtz",
    "fwt_no_water residual 6dih",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/residuals/6dim.mtz",
    "fwt_no_water residual 6dim",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/residuals/6djd.mtz",
    "fwt_no_water residual 6djd",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/residuals/6dje.mtz",
    "fwt_no_water residual 6dje",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/residuals/6djf.mtz",
    "fwt_no_water residual 6djf",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/residuals/6djg.mtz",
    "fwt_no_water residual 6djg",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/residuals/6djh.mtz",
    "fwt_no_water residual 6djh",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/residuals/6dji.mtz",
    "fwt_no_water residual 6dji",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/residuals/6djj.mtz",
    "fwt_no_water residual 6djj",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/residuals/6mj5.mtz",
    "fwt_no_water residual 6mj5",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/residuals/6myz.mtz",
    "fwt_no_water residual 6myz",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/residuals/6mz0.mtz",
    "fwt_no_water residual 6mz0",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/residuals/6n0d.mtz",
    "fwt_no_water residual 6n0d",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/residuals/6n0n.mtz",
    "fwt_no_water residual 6n0n",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/residuals/6n0o.mtz",
    "fwt_no_water residual 6n0o",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/residuals/6n0r.mtz",
    "fwt_no_water residual 6n0r",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/residuals/6n17.mtz",
    "fwt_no_water residual 6n17",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/residuals/6n19.mtz",
    "fwt_no_water residual 6n19",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/residuals/6w4r.mtz",
    "fwt_no_water residual 6w4r",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/residuals/6w7j.mtz",
    "fwt_no_water residual 6w7j",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/residuals/6w7k.mtz",
    "fwt_no_water residual 6w7k",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/residuals/6w7l.mtz",
    "fwt_no_water residual 6w7l",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/residuals/7ufy.mtz",
    "fwt_no_water residual 7ufy",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/residuals/7ufz.mtz",
    "fwt_no_water residual 7ufz",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/residuals/8cvq.mtz",
    "fwt_no_water residual 8cvq",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/residuals/8cw2.mtz",
    "fwt_no_water residual 8cw2",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/residuals/8uv1.mtz",
    "fwt_no_water residual 8uv1",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/residuals/8uzv.mtz",
    "fwt_no_water residual 8uzv",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/residuals/8uzz.mtz",
    "fwt_no_water residual 8uzz",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/residuals/8v0b.mtz",
    "fwt_no_water residual 8v0b",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/residuals/8v0c.mtz",
    "fwt_no_water residual 8v0c",
    "mtz",
    false,
    false,
    1.0
  ],
  [
    "cell/fwt_no_water/residual_svd/residuals/9b3b.mtz",
    "fwt_no_water residual 9b3b",
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
