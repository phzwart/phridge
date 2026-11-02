# Case study

Four scatterers in P1, six measured intensities, and the intensity MLI
configuration that refines them. This directory holds inputs only. Refinement
writes its model and MTZ into the `--out` directory you pass.

| File | Contents |
|------|----------|
| `model.pdb` | C, N, O, C in P1. Cell 4.986, 6.482, 8.476 Å, 83°, 109°, 129°. |
| `data.mtz` | Intensities `I` with sigmas, plus `FreeR_flag` (`0` is free). Six reflections: 100, 010, 001, 110, 101, 011. |
| `config.yaml` | Same component graph as [`../configs/intensity_mli.yaml`](../configs/intensity_mli.yaml), with `observed` pointed at these files. |

Run from the repository root so the paths in `config.yaml` resolve:

```bash
phridge validate examples/case_study/config.yaml
phridge refine examples/case_study/config.yaml --policy stock_macrocycle --out out/case_study
```

## Deposited structures

[`pdb_ids.txt`](pdb_ids.txt) lists the PDB entries for this case study.
[`download.py`](download.py) fetches each deposited model and its structure-factor
mmCIF into `structures/{id}/`. Intensities, when the entry deposited them, are
columns in `{id}-sf.cif`.

```bash
python examples/case_study/download.py
```
