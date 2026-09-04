# Examples

Runnable scripts that start from **live cctbx objects**, pack them through
**phridge** (LinkML metadata + npz tables), and convert back to **cctbx**.

## Environment

Use the conda env with `cctbx-base` and the monomer library (`chem_data`):

```bash
conda activate phridge-cctbx          # or: ~/miniforge3/envs/phridge-cctbx/bin/python
cd /path/to/phridge
pip install -e ".[dev]"               # once
```

If `chem_data` is missing:

```bash
make chem-data
# or: conda install -n phridge-cctbx -c chem_data chem_data
```

## Restraint minimization (cctbx → phridge → cctbx)

```bash
make example-restraints
# or:
python examples/restraint_minimization.py
```

Flow:

1. Build an `mmtbx.model` (cctbx hierarchy + `geometry_restraints.manager`)
2. Pack with `to_canonical` / `restraints_from_cctbx` / `hierarchy_from_cctbx`
3. Optional wire bytes via `packed.pack()` / `unpack_restraints`
4. Restore cctbx with `from_canonical(..., prefer_cctbx=True)` and
   `restraints_to_proxies` → rebuilt manager
5. Run `cctbx.geometry_restraints.lbfgs` on the restored objects
