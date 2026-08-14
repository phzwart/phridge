# Types and packing

Canonical objects are LinkML classes. JSON metadata sits on
`ObjectRef.meta`. Numeric buffers are a single **packed npz** at
`ObjectRef.key` (`kind=cctbx`). Store dtypes are little-endian float64
and int32.

## Index identity (model–geometry–data)

One atom/scatterer row index `i` is used everywhere:

| Object | Index field |
|--------|-------------|
| `Hierarchy` | `atoms[i].i == i` (npz row `i`) |
| `XrayStructure` | `scatterers[i].i == i` (npz row `i`) |
| `GeometryRestraints` | every proxy `i_seq` into that same `0 .. n_sites-1` |

`ModelGeometry` is a JSON header that records `n_sites` and which of
hierarchy / xray / restraints are present. It does not duplicate
payloads. `phridge.client.geometry.model_geometry(...)` verifies the
identity and writes the header.

Ramachandran, NCS, and DEN managers are not packed; they are not flat
`i_seq` proxy tables.

## npz layouts

**MillerArray:** `hkl` int32[N,3], `data` float64 or complex128[N],
optional `sigmas`.

**HendricksonLattman:** `hkl`, `A`, `B`, `C`, `D`.

**ReflectionFile:** `hkl` plus one array per column (`npz_name`),
optional `{npz_name}_sigmas`.

**RealMap / ComplexMap / EmMap:** `data` with shape `(nx, ny, nz)`.
`EmMap` metadata adds `wrapping`, `experiment_type`, `is_mask`,
`pixel_sizes`, `origin_cart`, `resolution`.

**Hierarchy:** `xyz`, `occupancy`, `b_iso`; optional `u_cart` [N,6]
(U in Å²) and `uij_defined` uint8. `has_uij` is true when `u_cart` is
present.

**XrayStructure:** `sites_frac`, `occupancy`, `u_iso`, `u_star` [N,6]
(cctbx `u_star`, fractional). Mixed iso/aniso: `Scatterer.anisotropic`
marks which `u_star` rows are live.

**GeometryRestraints** (missing arrays mean zero proxies of that kind):

- bonds: `bond_i_seqs` [M,2], distance, weight, slack, origin_id
- ASU bonds: `bond_asu_i_seqs`, plus `bond_asu_rt_mx` strings in JSON meta
- angles, dihedrals, chirality
- planarity / parallelity as CSR (`*_offsets` + flat `*_i_seqs`)
- nonbonded: `nonbonded_i_seqs`, `nonbonded_vdw`
- reference coordinates: `refcoord_i_seq`, `refcoord_xyz`, `refcoord_weight`
- bond similarity: `bondsim_pair_i_seqs`, `bondsim_offsets`, `bondsim_weight`
- optional `model_indices`, `conformer_indices` length `n_sites`
