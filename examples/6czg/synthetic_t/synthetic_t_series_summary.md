# 6CZG Synthetic Intensity Series: Student-t Noise Model

## 1. Experimental Setup & Model Ground Truth

- **Reference Model PDB**: `/Users/phzwart/Projects/phridge/examples/6czg/6czg.pdb`
- **Reference MTZ**: `/Users/phzwart/Projects/phridge/examples/6czg/6czg.mtz`
- **Student-t Degrees of Freedom ($\nu$)**: `7.0`
- **Error Scaling Mode**: `direct`
- **Refined Bulk Solvent Scale ($k_\text{total}$)**: `0.3248`
- **Refined Solvent Parameter ($k_\text{sol}$)**: `0.3424`
- **Refined Solvent $B$-factor ($B_\text{sol}$)**: `27.52` Å²
- **Random Seed**: `42`

## 2. Multiplier Series Summary

| Multiplier | MTZ File | $\langle I/\sigma \rangle$ | Median $I/\sigma$ | Outer Shell $\langle I/\sigma \rangle$ | $\langle \sigma \rangle$ | Neg Count | Neg % | $R_\text{noise}$ (%) | True/Synth Corr |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `100x` | `6czg_synthetic_t_mult_100.mtz` | 0.55 | 0.52 | 0.29 | 13050.4 | 2957 | 32.61% | 119.74% | 0.6338 |

## 3. Resolution Shell Breakdown Across Multipliers

### Multiplier 100x (`0.55` mean $I/\sigma$, $R_\text{noise} = 119.74\%$)

| Shell | $d_\text{max}$ (Å) | $d_\text{min}$ (Å) | $N_\text{refl}$ | $\langle I/\sigma \rangle$ | Median $I/\sigma$ | Expected SNR | $\langle \sigma \rangle$ | Neg Count | Neg % | $R_\text{noise}$ (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 29.77 | 6.78 | 333 | 0.95 | 0.71 | 0.98 | 32260.8 | 88 | 26.43% | 98.60% |
| 1 | 6.78 | 4.86 | 561 | 0.98 | 0.91 | 0.94 | 21544.7 | 123 | 21.93% | 84.83% |
| 2 | 4.86 | 3.98 | 703 | 0.87 | 0.83 | 0.78 | 32782.4 | 158 | 22.48% | 84.14% |
| 3 | 3.98 | 3.46 | 820 | 0.70 | 0.66 | 0.64 | 22237.3 | 232 | 28.29% | 125.20% |
| 4 | 3.46 | 3.10 | 917 | 0.63 | 0.61 | 0.59 | 17679.1 | 270 | 29.44% | 139.75% |
| 5 | 3.10 | 2.83 | 1008 | 0.53 | 0.54 | 0.53 | 10481.8 | 326 | 32.34% | 134.67% |
| 6 | 2.83 | 2.62 | 1097 | 0.45 | 0.42 | 0.46 | 7599.8 | 380 | 34.64% | 166.06% |
| 7 | 2.62 | 2.45 | 1142 | 0.42 | 0.40 | 0.39 | 5728.3 | 415 | 36.34% | 193.94% |
| 8 | 2.45 | 2.31 | 1227 | 0.38 | 0.33 | 0.35 | 5192.4 | 460 | 37.49% | 214.61% |
| 9 | 2.31 | 2.20 | 1261 | 0.29 | 0.29 | 0.29 | 4930.4 | 505 | 40.05% | 267.38% |
