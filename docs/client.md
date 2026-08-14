# Client helpers

`Bridge` (`phridge.client`) submits jobs. Converters run on the Phenix
side only. Passing a cctbx object into `call`/`submit` goes through
`to_canonical`; results can be converted back with `prefer_cctbx=True`.

## Coordinates and xray

`phridge.client.convert` and `phridge.client.convert_xtal`:

- miller arrays, HL coefficients, MTZ `ReflectionFile`
- real/complex maps, map coefficients, crystal gridding
- Cartesian/fractional sites
- `hierarchy_from_cctbx` / `hierarchy_to_cctbx` (including ANISOU)
- `xray_from_cctbx` / `xray_to_cctbx` (including anisotropic `u_star`)

## Geometry

```python
from phridge.client.geometry import model_geometry, restraints_from_cctbx, restraints_to_proxies

packed = restraints_from_cctbx(manager)   # cctbx.geometry_restraints.manager
proxies = restraints_to_proxies(packed)   # dict of proxy lists, not a full manager
header = model_geometry(hierarchy=hier, xray=xrs, restraints=packed)
```

`restraints_to_proxies` rebuilds bond/angle/dihedral/chirality/planarity/parallelity
proxies. It does not reconstruct `bond_params_table` or a live manager.

## EM maps

`iotbx.map_manager` round-trips as `EmMap`. Helpers in `phridge.client.em`:

| Function | Role |
|----------|------|
| `em_map_from_map_manager` / `em_map_to_map_manager` | cctbx round-trip |
| `em_map_from_numpy` | build `EmMap` from a 3-D array + crystal |
| `em_map_as_real_map` | drop EM metadata for a `RealMap` grid |
| `density_at_sites` | sample at Cartesian Å coordinates |
| `density_at_hierarchy` | sample at packed hierarchy xyz (i_seq order) |

Cryo-EM maps are typically `wrapping=False`. Crystal maps may wrap.
