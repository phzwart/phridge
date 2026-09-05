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

Converters:

```python
from phridge.client.geometry import model_geometry, restraints_from_cctbx, restraints_to_proxies

packed = restraints_from_cctbx(manager)   # cctbx.geometry_restraints.manager
proxies = restraints_to_proxies(packed)   # dict of proxy lists, not a full manager
header = model_geometry(hierarchy=hier, xray=xrs, restraints=packed)
```

`restraints_to_proxies` rebuilds bond/angle/dihedral/chirality/planarity/parallelity
proxies. It does not reconstruct `bond_params_table` or a live manager.

### RemoteGeometry.minimize

Phenix-facing manager: pack hierarchy + restraints to Redis, block until the
worker finishes a **torch** optimizer (`lbfgs`, `adam`, `adamw`, or `sgd`), convert
sites back to a cctbx hierarchy.

```python
from phridge.client import Bridge, RemoteGeometry

bridge = Bridge("redis://gpu-box:6379/0")
# bridge = Bridge(memory=True)  # no redis-server; runs the op in-process
geo = RemoteGeometry(bridge, hierarchy, restraints_manager)
hierarchy_out = geo.minimize(max_iterations=100, optimizer="lbfgs")
# geo.energy()              # remote eval, no steps
# geo.last_target           # before/after, n_steps, n_calls, optimizer_state_mb, rss_*_mb, ...
```

First-order methods accept `schedule` (`none` / `cosine` / `triangular`),
`lr` / `lr_min`, and SGD `momentum`. Memory fields on `last_target`:

| key | meaning |
|-----|---------|
| `optimizer_state_mb` | torch optimizer state tensors (LBFGS history vs Adam moments vs SGD velocity) |
| `rss_before_mb` / `rss_after_mb` / `rss_delta_mb` | process VmRSS around the minimize |
| `cuda_peak_mb` | CUDA peak allocation delta (0 on CPU) |

v1 worker energy uses packed bonds / angles / dihedrals only (no nonbonded /
ASU / planarity). See `examples/restraint_minimization.py`.

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
