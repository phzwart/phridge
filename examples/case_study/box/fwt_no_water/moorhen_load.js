// Moorhen loader for box/fwt_no_water
// Open moorhen.org, press F12, paste this into the Console, press Enter.
// The file server must be running.
(async () => {
  const BASE = "http://localhost:8765/";
  const MODELS = [
  [
    "box/models/1mu7.pdb",
    "1mu7"
  ],
  [
    "box/models/1mu9.pdb",
    "1mu9"
  ],
  [
    "box/models/1nop.pdb",
    "1nop"
  ],
  [
    "box/models/1rff.pdb",
    "1rff"
  ],
  [
    "box/models/1rfi.pdb",
    "1rfi"
  ],
  [
    "box/models/1rg1.pdb",
    "1rg1"
  ],
  [
    "box/models/1rg2.pdb",
    "1rg2"
  ],
  [
    "box/models/1rgt.pdb",
    "1rgt"
  ],
  [
    "box/models/1rgu.pdb",
    "1rgu"
  ],
  [
    "box/models/1rh0.pdb",
    "1rh0"
  ],
  [
    "box/models/6dhu.pdb",
    "6dhu"
  ],
  [
    "box/models/6die.pdb",
    "6die"
  ],
  [
    "box/models/6dih.pdb",
    "6dih"
  ],
  [
    "box/models/6dim.pdb",
    "6dim"
  ],
  [
    "box/models/6djd.pdb",
    "6djd"
  ],
  [
    "box/models/6dje.pdb",
    "6dje"
  ],
  [
    "box/models/6djf.pdb",
    "6djf"
  ],
  [
    "box/models/6djg.pdb",
    "6djg"
  ],
  [
    "box/models/6djh.pdb",
    "6djh"
  ],
  [
    "box/models/6dji.pdb",
    "6dji"
  ],
  [
    "box/models/6djj.pdb",
    "6djj"
  ],
  [
    "box/models/6mj5.pdb",
    "6mj5"
  ],
  [
    "box/models/6myz.pdb",
    "6myz"
  ],
  [
    "box/models/6mz0.pdb",
    "6mz0"
  ],
  [
    "box/models/6n0d.pdb",
    "6n0d"
  ],
  [
    "box/models/6n0n.pdb",
    "6n0n"
  ],
  [
    "box/models/6n0o.pdb",
    "6n0o"
  ],
  [
    "box/models/6n0r.pdb",
    "6n0r"
  ],
  [
    "box/models/6n17.pdb",
    "6n17"
  ],
  [
    "box/models/6n19.pdb",
    "6n19"
  ],
  [
    "box/models/6w4r.pdb",
    "6w4r"
  ],
  [
    "box/models/6w7j.pdb",
    "6w7j"
  ],
  [
    "box/models/6w7k.pdb",
    "6w7k"
  ],
  [
    "box/models/6w7l.pdb",
    "6w7l"
  ],
  [
    "box/models/7ufy.pdb",
    "7ufy"
  ],
  [
    "box/models/7ufz.pdb",
    "7ufz"
  ],
  [
    "box/models/8cvq.pdb",
    "8cvq"
  ],
  [
    "box/models/8cw2.pdb",
    "8cw2"
  ],
  [
    "box/models/8uv1.pdb",
    "8uv1"
  ],
  [
    "box/models/8uzv.pdb",
    "8uzv"
  ],
  [
    "box/models/8uzz.pdb",
    "8uzz"
  ],
  [
    "box/models/8v0b.pdb",
    "8v0b"
  ],
  [
    "box/models/8v0c.pdb",
    "8v0c"
  ],
  [
    "box/models/9b3b.pdb",
    "9b3b"
  ]
];
  const MAPS = [
  [
    "box/fwt_no_water/maps/1mu7.mtz",
    "fwt_no_water 1mu7",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/fwt_no_water/maps/1mu9.mtz",
    "fwt_no_water 1mu9",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/fwt_no_water/maps/1nop.mtz",
    "fwt_no_water 1nop",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/fwt_no_water/maps/1rff.mtz",
    "fwt_no_water 1rff",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/fwt_no_water/maps/1rfi.mtz",
    "fwt_no_water 1rfi",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/fwt_no_water/maps/1rg1.mtz",
    "fwt_no_water 1rg1",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/fwt_no_water/maps/1rg2.mtz",
    "fwt_no_water 1rg2",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/fwt_no_water/maps/1rgt.mtz",
    "fwt_no_water 1rgt",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/fwt_no_water/maps/1rgu.mtz",
    "fwt_no_water 1rgu",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/fwt_no_water/maps/1rh0.mtz",
    "fwt_no_water 1rh0",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/fwt_no_water/maps/6dhu.mtz",
    "fwt_no_water 6dhu",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/fwt_no_water/maps/6die.mtz",
    "fwt_no_water 6die",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/fwt_no_water/maps/6dih.mtz",
    "fwt_no_water 6dih",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/fwt_no_water/maps/6dim.mtz",
    "fwt_no_water 6dim",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/fwt_no_water/maps/6djd.mtz",
    "fwt_no_water 6djd",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/fwt_no_water/maps/6dje.mtz",
    "fwt_no_water 6dje",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/fwt_no_water/maps/6djf.mtz",
    "fwt_no_water 6djf",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/fwt_no_water/maps/6djg.mtz",
    "fwt_no_water 6djg",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/fwt_no_water/maps/6djh.mtz",
    "fwt_no_water 6djh",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/fwt_no_water/maps/6dji.mtz",
    "fwt_no_water 6dji",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/fwt_no_water/maps/6djj.mtz",
    "fwt_no_water 6djj",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/fwt_no_water/maps/6mj5.mtz",
    "fwt_no_water 6mj5",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/fwt_no_water/maps/6myz.mtz",
    "fwt_no_water 6myz",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/fwt_no_water/maps/6mz0.mtz",
    "fwt_no_water 6mz0",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/fwt_no_water/maps/6n0d.mtz",
    "fwt_no_water 6n0d",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/fwt_no_water/maps/6n0n.mtz",
    "fwt_no_water 6n0n",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/fwt_no_water/maps/6n0o.mtz",
    "fwt_no_water 6n0o",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/fwt_no_water/maps/6n0r.mtz",
    "fwt_no_water 6n0r",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/fwt_no_water/maps/6n17.mtz",
    "fwt_no_water 6n17",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/fwt_no_water/maps/6n19.mtz",
    "fwt_no_water 6n19",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/fwt_no_water/maps/6w4r.mtz",
    "fwt_no_water 6w4r",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/fwt_no_water/maps/6w7j.mtz",
    "fwt_no_water 6w7j",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/fwt_no_water/maps/6w7k.mtz",
    "fwt_no_water 6w7k",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/fwt_no_water/maps/6w7l.mtz",
    "fwt_no_water 6w7l",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/fwt_no_water/maps/7ufy.mtz",
    "fwt_no_water 7ufy",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/fwt_no_water/maps/7ufz.mtz",
    "fwt_no_water 7ufz",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/fwt_no_water/maps/8cvq.mtz",
    "fwt_no_water 8cvq",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/fwt_no_water/maps/8cw2.mtz",
    "fwt_no_water 8cw2",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/fwt_no_water/maps/8uv1.mtz",
    "fwt_no_water 8uv1",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/fwt_no_water/maps/8uzv.mtz",
    "fwt_no_water 8uzv",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/fwt_no_water/maps/8uzz.mtz",
    "fwt_no_water 8uzz",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/fwt_no_water/maps/8v0b.mtz",
    "fwt_no_water 8v0b",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/fwt_no_water/maps/8v0c.mtz",
    "fwt_no_water 8v0c",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/fwt_no_water/maps/9b3b.mtz",
    "fwt_no_water 9b3b",
    "mtz",
    false,
    true,
    null
  ],
  [
    "box/fwt_no_water/residual_svd/average.mtz",
    "fwt_no_water average",
    "mtz",
    true,
    true,
    null
  ],
  [
    "box/fwt_no_water/residual_svd/component_01.mtz",
    "fwt_no_water component_01",
    "mtz",
    true,
    false,
    0.0005360384238883853
  ],
  [
    "box/fwt_no_water/residual_svd/component_02.mtz",
    "fwt_no_water component_02",
    "mtz",
    false,
    false,
    0.0005360385403037071
  ],
  [
    "box/fwt_no_water/residual_svd/component_03.mtz",
    "fwt_no_water component_03",
    "mtz",
    false,
    false,
    0.0005360384238883853
  ],
  [
    "box/fwt_no_water/residual_svd/component_04.mtz",
    "fwt_no_water component_04",
    "mtz",
    false,
    false,
    0.0005360385403037071
  ],
  [
    "box/fwt_no_water/residual_svd/component_05.mtz",
    "fwt_no_water component_05",
    "mtz",
    false,
    false,
    0.000536038656719029
  ],
  [
    "box/fwt_no_water/residual_svd/component_06.mtz",
    "fwt_no_water component_06",
    "mtz",
    false,
    false,
    0.0005360385403037071
  ],
  [
    "box/fwt_no_water/residual_svd/component_07.mtz",
    "fwt_no_water component_07",
    "mtz",
    false,
    false,
    0.0005360382492654026
  ],
  [
    "box/fwt_no_water/residual_svd/component_08.mtz",
    "fwt_no_water component_08",
    "mtz",
    false,
    false,
    0.0005360384238883853
  ],
  [
    "box/fwt_no_water/residual_svd/component_09.mtz",
    "fwt_no_water component_09",
    "mtz",
    false,
    false,
    0.0005360388313420117
  ],
  [
    "box/fwt_no_water/residual_svd/component_10.mtz",
    "fwt_no_water component_10",
    "mtz",
    false,
    false,
    0.0005360384238883853
  ],
  [
    "box/fwt_no_water/residual_svd/component_11.mtz",
    "fwt_no_water component_11",
    "mtz",
    false,
    false,
    0.0005360382492654026
  ],
  [
    "box/fwt_no_water/residual_svd/component_12.mtz",
    "fwt_no_water component_12",
    "mtz",
    false,
    false,
    0.0005360387149266899
  ],
  [
    "box/fwt_no_water/residual_svd/component_13.mtz",
    "fwt_no_water component_13",
    "mtz",
    false,
    false,
    0.0005360381328500807
  ],
  [
    "box/fwt_no_water/residual_svd/component_14.mtz",
    "fwt_no_water component_14",
    "mtz",
    false,
    false,
    0.0005360382492654026
  ],
  [
    "box/fwt_no_water/residual_svd/component_15.mtz",
    "fwt_no_water component_15",
    "mtz",
    false,
    false,
    0.0005360380164347589
  ],
  [
    "box/fwt_no_water/residual_svd/component_16.mtz",
    "fwt_no_water component_16",
    "mtz",
    false,
    false,
    0.0005360383074730635
  ],
  [
    "box/fwt_no_water/residual_svd/component_17.mtz",
    "fwt_no_water component_17",
    "mtz",
    false,
    false,
    0.0005360387149266899
  ],
  [
    "box/fwt_no_water/residual_svd/component_18.mtz",
    "fwt_no_water component_18",
    "mtz",
    false,
    false,
    0.000536037958227098
  ],
  [
    "box/fwt_no_water/residual_svd/component_19.mtz",
    "fwt_no_water component_19",
    "mtz",
    false,
    false,
    0.0005360397626645863
  ],
  [
    "box/fwt_no_water/residual_svd/component_20.mtz",
    "fwt_no_water component_20",
    "mtz",
    false,
    false,
    0.0005360385403037071
  ],
  [
    "box/fwt_no_water/residual_svd/component_21.mtz",
    "fwt_no_water component_21",
    "mtz",
    false,
    false,
    0.0005360367358662188
  ],
  [
    "box/fwt_no_water/residual_svd/component_22.mtz",
    "fwt_no_water component_22",
    "mtz",
    false,
    false,
    0.0005360383074730635
  ],
  [
    "box/fwt_no_water/residual_svd/component_23.mtz",
    "fwt_no_water component_23",
    "mtz",
    false,
    false,
    0.0005360381328500807
  ],
  [
    "box/fwt_no_water/residual_svd/component_24.mtz",
    "fwt_no_water component_24",
    "mtz",
    false,
    false,
    0.0005360377253964543
  ],
  [
    "box/fwt_no_water/residual_svd/component_25.mtz",
    "fwt_no_water component_25",
    "mtz",
    false,
    false,
    0.0005360391805879772
  ],
  [
    "box/fwt_no_water/residual_svd/component_26.mtz",
    "fwt_no_water component_26",
    "mtz",
    false,
    false,
    0.0005360389477573335
  ],
  [
    "box/fwt_no_water/residual_svd/component_27.mtz",
    "fwt_no_water component_27",
    "mtz",
    false,
    false,
    0.0005360380746424198
  ],
  [
    "box/fwt_no_water/residual_svd/component_28.mtz",
    "fwt_no_water component_28",
    "mtz",
    false,
    false,
    0.0005360397044569254
  ],
  [
    "box/fwt_no_water/residual_svd/component_29.mtz",
    "fwt_no_water component_29",
    "mtz",
    false,
    false,
    0.0005360383656807244
  ],
  [
    "box/fwt_no_water/residual_svd/component_30.mtz",
    "fwt_no_water component_30",
    "mtz",
    false,
    false,
    0.0005360382492654026
  ],
  [
    "box/fwt_no_water/residual_svd/component_31.mtz",
    "fwt_no_water component_31",
    "mtz",
    false,
    false,
    0.0005360372015275061
  ],
  [
    "box/fwt_no_water/residual_svd/component_32.mtz",
    "fwt_no_water component_32",
    "mtz",
    false,
    false,
    0.0005360384820960462
  ],
  [
    "box/fwt_no_water/residual_svd/component_33.mtz",
    "fwt_no_water component_33",
    "mtz",
    false,
    false,
    0.0005360414506867528
  ],
  [
    "box/fwt_no_water/residual_svd/component_34.mtz",
    "fwt_no_water component_34",
    "mtz",
    false,
    false,
    0.0005360406939871609
  ],
  [
    "box/fwt_no_water/residual_svd/component_35.mtz",
    "fwt_no_water component_35",
    "mtz",
    false,
    false,
    0.0005360385403037071
  ],
  [
    "box/fwt_no_water/residual_svd/component_36.mtz",
    "fwt_no_water component_36",
    "mtz",
    false,
    false,
    0.0005360380746424198
  ],
  [
    "box/fwt_no_water/residual_svd/component_37.mtz",
    "fwt_no_water component_37",
    "mtz",
    false,
    false,
    0.0005360374925658107
  ],
  [
    "box/fwt_no_water/residual_svd/component_38.mtz",
    "fwt_no_water component_38",
    "mtz",
    false,
    false,
    0.0005360391223803163
  ],
  [
    "box/fwt_no_water/residual_svd/component_39.mtz",
    "fwt_no_water component_39",
    "mtz",
    false,
    false,
    0.0005360402283258736
  ],
  [
    "box/fwt_no_water/residual_svd/component_40.mtz",
    "fwt_no_water component_40",
    "mtz",
    false,
    false,
    0.0005360384820960462
  ],
  [
    "box/fwt_no_water/residual_svd/component_41.mtz",
    "fwt_no_water component_41",
    "mtz",
    false,
    false,
    0.0005360391223803163
  ],
  [
    "box/fwt_no_water/residual_svd/component_42.mtz",
    "fwt_no_water component_42",
    "mtz",
    false,
    false,
    0.0005360382492654026
  ],
  [
    "box/fwt_no_water/residual_svd/component_43.mtz",
    "fwt_no_water component_43",
    "mtz",
    false,
    false,
    0.0005360374343581498
  ],
  [
    "box/fwt_no_water/residual_svd/power_1-4.ccp4",
    "fwt_no_water power_1-4",
    "map",
    false,
    false,
    1.8637781143188477
  ],
  [
    "box/fwt_no_water/residual_svd/power_all.ccp4",
    "fwt_no_water power_all",
    "map",
    false,
    false,
    1.4055256843566895
  ],
  [
    "box/fwt_no_water/residual_svd/residuals/1mu7.mtz",
    "fwt_no_water residual 1mu7",
    "mtz",
    false,
    false,
    0.5990024209022522
  ],
  [
    "box/fwt_no_water/residual_svd/residuals/1mu9.mtz",
    "fwt_no_water residual 1mu9",
    "mtz",
    false,
    false,
    0.5847886204719543
  ],
  [
    "box/fwt_no_water/residual_svd/residuals/1nop.mtz",
    "fwt_no_water residual 1nop",
    "mtz",
    false,
    false,
    0.6061779856681824
  ],
  [
    "box/fwt_no_water/residual_svd/residuals/1rff.mtz",
    "fwt_no_water residual 1rff",
    "mtz",
    false,
    false,
    0.5698493123054504
  ],
  [
    "box/fwt_no_water/residual_svd/residuals/1rfi.mtz",
    "fwt_no_water residual 1rfi",
    "mtz",
    false,
    false,
    0.5715328454971313
  ],
  [
    "box/fwt_no_water/residual_svd/residuals/1rg1.mtz",
    "fwt_no_water residual 1rg1",
    "mtz",
    false,
    false,
    0.6068955063819885
  ],
  [
    "box/fwt_no_water/residual_svd/residuals/1rg2.mtz",
    "fwt_no_water residual 1rg2",
    "mtz",
    false,
    false,
    0.5790103077888489
  ],
  [
    "box/fwt_no_water/residual_svd/residuals/1rgt.mtz",
    "fwt_no_water residual 1rgt",
    "mtz",
    false,
    false,
    0.6022881865501404
  ],
  [
    "box/fwt_no_water/residual_svd/residuals/1rgu.mtz",
    "fwt_no_water residual 1rgu",
    "mtz",
    false,
    false,
    0.6193135976791382
  ],
  [
    "box/fwt_no_water/residual_svd/residuals/1rh0.mtz",
    "fwt_no_water residual 1rh0",
    "mtz",
    false,
    false,
    0.6036424040794373
  ],
  [
    "box/fwt_no_water/residual_svd/residuals/6dhu.mtz",
    "fwt_no_water residual 6dhu",
    "mtz",
    false,
    false,
    0.34115681052207947
  ],
  [
    "box/fwt_no_water/residual_svd/residuals/6die.mtz",
    "fwt_no_water residual 6die",
    "mtz",
    false,
    false,
    0.33344703912734985
  ],
  [
    "box/fwt_no_water/residual_svd/residuals/6dih.mtz",
    "fwt_no_water residual 6dih",
    "mtz",
    false,
    false,
    0.33294782042503357
  ],
  [
    "box/fwt_no_water/residual_svd/residuals/6dim.mtz",
    "fwt_no_water residual 6dim",
    "mtz",
    false,
    false,
    0.4012220501899719
  ],
  [
    "box/fwt_no_water/residual_svd/residuals/6djd.mtz",
    "fwt_no_water residual 6djd",
    "mtz",
    false,
    false,
    0.3335382640361786
  ],
  [
    "box/fwt_no_water/residual_svd/residuals/6dje.mtz",
    "fwt_no_water residual 6dje",
    "mtz",
    false,
    false,
    0.431988924741745
  ],
  [
    "box/fwt_no_water/residual_svd/residuals/6djf.mtz",
    "fwt_no_water residual 6djf",
    "mtz",
    false,
    false,
    0.3745335638523102
  ],
  [
    "box/fwt_no_water/residual_svd/residuals/6djg.mtz",
    "fwt_no_water residual 6djg",
    "mtz",
    false,
    false,
    0.37733879685401917
  ],
  [
    "box/fwt_no_water/residual_svd/residuals/6djh.mtz",
    "fwt_no_water residual 6djh",
    "mtz",
    false,
    false,
    0.39085593819618225
  ],
  [
    "box/fwt_no_water/residual_svd/residuals/6dji.mtz",
    "fwt_no_water residual 6dji",
    "mtz",
    false,
    false,
    0.35291653871536255
  ],
  [
    "box/fwt_no_water/residual_svd/residuals/6djj.mtz",
    "fwt_no_water residual 6djj",
    "mtz",
    false,
    false,
    0.28636959195137024
  ],
  [
    "box/fwt_no_water/residual_svd/residuals/6mj5.mtz",
    "fwt_no_water residual 6mj5",
    "mtz",
    false,
    false,
    0.3821825683116913
  ],
  [
    "box/fwt_no_water/residual_svd/residuals/6myz.mtz",
    "fwt_no_water residual 6myz",
    "mtz",
    false,
    false,
    0.42206084728240967
  ],
  [
    "box/fwt_no_water/residual_svd/residuals/6mz0.mtz",
    "fwt_no_water residual 6mz0",
    "mtz",
    false,
    false,
    0.3844374716281891
  ],
  [
    "box/fwt_no_water/residual_svd/residuals/6n0d.mtz",
    "fwt_no_water residual 6n0d",
    "mtz",
    false,
    false,
    0.34455606341362
  ],
  [
    "box/fwt_no_water/residual_svd/residuals/6n0n.mtz",
    "fwt_no_water residual 6n0n",
    "mtz",
    false,
    false,
    0.5137532353401184
  ],
  [
    "box/fwt_no_water/residual_svd/residuals/6n0o.mtz",
    "fwt_no_water residual 6n0o",
    "mtz",
    false,
    false,
    0.3673481047153473
  ],
  [
    "box/fwt_no_water/residual_svd/residuals/6n0r.mtz",
    "fwt_no_water residual 6n0r",
    "mtz",
    false,
    false,
    0.3736630380153656
  ],
  [
    "box/fwt_no_water/residual_svd/residuals/6n17.mtz",
    "fwt_no_water residual 6n17",
    "mtz",
    false,
    false,
    0.3243839144706726
  ],
  [
    "box/fwt_no_water/residual_svd/residuals/6n19.mtz",
    "fwt_no_water residual 6n19",
    "mtz",
    false,
    false,
    0.392852246761322
  ],
  [
    "box/fwt_no_water/residual_svd/residuals/6w4r.mtz",
    "fwt_no_water residual 6w4r",
    "mtz",
    false,
    false,
    0.34124764800071716
  ],
  [
    "box/fwt_no_water/residual_svd/residuals/6w7j.mtz",
    "fwt_no_water residual 6w7j",
    "mtz",
    false,
    false,
    0.35026344656944275
  ],
  [
    "box/fwt_no_water/residual_svd/residuals/6w7k.mtz",
    "fwt_no_water residual 6w7k",
    "mtz",
    false,
    false,
    0.3304501473903656
  ],
  [
    "box/fwt_no_water/residual_svd/residuals/6w7l.mtz",
    "fwt_no_water residual 6w7l",
    "mtz",
    false,
    false,
    0.4615253508090973
  ],
  [
    "box/fwt_no_water/residual_svd/residuals/7ufy.mtz",
    "fwt_no_water residual 7ufy",
    "mtz",
    false,
    false,
    0.40256744623184204
  ],
  [
    "box/fwt_no_water/residual_svd/residuals/7ufz.mtz",
    "fwt_no_water residual 7ufz",
    "mtz",
    false,
    false,
    0.4075832664966583
  ],
  [
    "box/fwt_no_water/residual_svd/residuals/8cvq.mtz",
    "fwt_no_water residual 8cvq",
    "mtz",
    false,
    false,
    0.39352715015411377
  ],
  [
    "box/fwt_no_water/residual_svd/residuals/8cw2.mtz",
    "fwt_no_water residual 8cw2",
    "mtz",
    false,
    false,
    0.356454461812973
  ],
  [
    "box/fwt_no_water/residual_svd/residuals/8uv1.mtz",
    "fwt_no_water residual 8uv1",
    "mtz",
    false,
    false,
    0.436944842338562
  ],
  [
    "box/fwt_no_water/residual_svd/residuals/8uzv.mtz",
    "fwt_no_water residual 8uzv",
    "mtz",
    false,
    false,
    0.3328295052051544
  ],
  [
    "box/fwt_no_water/residual_svd/residuals/8uzz.mtz",
    "fwt_no_water residual 8uzz",
    "mtz",
    false,
    false,
    0.32580098509788513
  ],
  [
    "box/fwt_no_water/residual_svd/residuals/8v0b.mtz",
    "fwt_no_water residual 8v0b",
    "mtz",
    false,
    false,
    0.3338784873485565
  ],
  [
    "box/fwt_no_water/residual_svd/residuals/8v0c.mtz",
    "fwt_no_water residual 8v0c",
    "mtz",
    false,
    false,
    0.3174494206905365
  ],
  [
    "box/fwt_no_water/residual_svd/residuals/9b3b.mtz",
    "fwt_no_water residual 9b3b",
    "mtz",
    false,
    false,
    0.3311415910720825
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
