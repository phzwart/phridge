# Plan C — One cctbx adapter layer

Branch: `refactor/C`. Two stages. Collapses `packing*.py` (top level) and
`client/convert*.py` into `shims/cctbx/`, one module per cctbx type, tensor-in
or tensor-out, tagged by tier.

---

## Stage C1 — inventory and move

**Goal.** Every function that converts between a cctbx object and a packed /
tensor representation lives under `src/phridge/shims/cctbx/`, exactly once.

**Do.**

1. Inventory first. Run and paste the output into your notes:
   ```
   grep -n "^def \|^class " src/phridge/packing.py src/phridge/packing_xtal.py src/phridge/packing_geometry.py src/phridge/packing_scattering.py src/phridge/client/convert.py src/phridge/client/convert_xtal.py src/phridge/client/convert_geometry.py src/phridge/client/geometry.py src/phridge/client/adp_restraints.py src/phridge/client/em.py
   ```
   For each function, classify: (a) pure numpy packing with no cctbx import
   → stays importable without cctbx; (b) touches a cctbx/flex object → cctbx side.
2. Create `src/phridge/shims/cctbx/` with modules, moving functions by the
   cctbx type they handle (use `git mv` for whole files where a file maps to
   one type; otherwise create the new module and move function bodies verbatim):
   - `symmetry.py` — `CrystalSymmetry` ↔ cell/Hall/IT; sym ops as `Tensor[nsym,3,4]`
   - `miller.py` — `miller.array` ↔ `ComplexArray` / packed hkl + columns
   - `xray_structure.py` — `xray_structure` ↔ `RObject`; MUST include
     `to_robject(xs) -> RObject` and `from_robject(xs_template, r: RObject) -> xray_structure`
     and `packed_order(xs) -> dict[str, slice]` (cctbx packed-gradient order)
   - `hierarchy.py` — `pdb_hierarchy` ↔ packed atoms (xyz, occ, b, u_cart)
   - `restraints.py` — geometry restraints manager → proxy tables as index
     tensors + parameters (from `client/geometry.py`, `convert_geometry.py`,
     `packing_geometry.py`); ADP restraint tables (from `client/adp_restraints.py`)
   - `scattering.py` — `ScatteringTable` ↔ Gaussian coefficient tensors
   - `maps.py` — real/complex maps, `EmMap` (from `client/em.py`)
   - `_flex.py` — the one place `scitbx.array_family.flex` is imported;
     `flex_to_numpy`, `numpy_to_flex` helpers.
   Pure-numpy packing helpers that need no cctbx go to
   `src/phridge/contracts/packing.py` (new) so the worker can import them
   without cctbx.
3. Each `shims/cctbx/*.py` module declares `MANIFEST = PluginManifest(name=..., slot=Slot.io_adapter, space=Space.D, tier=Tier.shim, ...)`
   except `restraints.py` and `symmetry.py` which are `Tier.hybrid`
   (built once, evaluated by torch).
4. Delete the now-empty originals. Fix every importer
   (`grep -rn "phridge.packing\|client.convert\|client.geometry\|client.adp_restraints\|client.em"`).
   `src/phridge/sfcalc/client.py` and `src/phridge/client/api.py` are the main
   consumers; update them to import from `phridge.shims.cctbx.*`.
5. `tests/`: move `test_converters_cctbx.py`, `test_converters_numpy.py`,
   `test_bidirectional_cctbx.py`, `test_geometry_curvature.py` (only the
   conversion parts) under `tests/shims/cctbx/` with imports fixed.

**Don't.** Do not change any function body. Do not merge two functions into one.

**Verify.**
```
python -m pytest -q tests/shims tests/test_codec.py tests/boundaries
grep -rn "scitbx.array_family" src/phridge --include=*.py | grep -v "shims/cctbx/_flex.py\|legacy/\|io/\|worker/cctbx_runner.py" ; echo "exit=$?"   # expect exit=1
```

**Commit.** `refactor: single cctbx adapter layer under shims/cctbx`

---

## Stage C2 — round-trip contract test

**Goal.** Prove the model-state contract: `xray_structure → RObject → xray_structure`
is bitwise identical, on the six space groups the repo already uses.

**Do.**

1. Find the space-group fixtures in `tests/test_agarwal_gradients.py`
   (P1, P21, P212121, C2, P4132, R3:H) and factor the random-structure builder
   into `tests/shims/cctbx/conftest.py` as a fixture `random_xs(space_group, n_atoms)`.
2. `tests/shims/cctbx/test_roundtrip.py`: for each space group, with and
   without anisotropic U, with fp/fdp set:
   `xs2 = from_robject(xs, to_robject(xs))`; assert `sites_frac`, `u_iso`,
   `u_star`, `occupancy`, `fp`, `fdp`, scattering types and `uij_defined`
   are exactly equal (numpy `array_equal`, no tolerance).
3. `test_packed_order_matches_cctbx`: build the packed gradient vector via
   cctbx (`xs.pack_gradients` or the existing helper in `sfcalc/client.py`)
   and via `packed_order` slices; assert identical layout for a structure with
   mixed iso/aniso atoms and refinement flags set in each combination.
4. Move `conftest.py`'s "import cctbx before torch" comment so it also lives
   in `tests/shims/cctbx/conftest.py`.

**Verify.**
```
python -m pytest -q tests/shims/cctbx/test_roundtrip.py tests/test_agarwal_gradients.py
```

**Commit.** `tests: bitwise xray_structure round-trip and packed-order contract`
