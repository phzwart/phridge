# Intensity Refinement & Map Generation Demo

Generated: 2026-09-05 07:09:52 UTC

## Executive Summary
This demo exercises the end-to-end integration between `cctbx` and `phridge` for intensity-based crystallographic refinement.

- **Data Alignment**: Ingested PDB coordinates and MTZ observed intensities.
- **Bulk Solvent via Map Gridding**: Real-space mask computed via `mmtbx.masks.manager` and FFT transformed to $F_{mask}$.
- **Direct Parameter Estimation**: Scaled model ($k_{total} = 2.2465, k_{sol} = 0.352, B_{sol} = 45.0\text{ Å}^2$) directly minimizing the phridge `ml_i` intensity negative log-likelihood.
- **Direct $\sigma_A$ Estimation**: Estimated $\sigma_A$ across 8 resolution shells by 1D bounded minimization of intensity NLL on the test set.
- **Gradient Maps**: Computed $F_{grad} = -\frac{1}{2} \frac{dQ}{dF_{model}}$ and Fourier transformed into real-space CCP4 format.
- **Weighted Difference Maps**: Synthesized $2mF_o - DF_c$ and $mF_o - DF_c$ maps with figure of merit $m$ from the Rice/Woolfson conditional distribution.

## Resolution Binning & $\sigma_A$ Distribution

| Bin | Resolution Range (Å) | $\sigma_A$ | $\langle \Sigma \rangle$ |
| --- | ------------------- | ---------- | -------------------------- |
| 1 | 31.82 - 4.00 | 0.9883 | 18417.2 |
| 2 | 4.00 - 3.17 | 0.9844 | 21956.9 |
| 3 | 3.17 - 2.77 | 0.9865 | 18931.3 |
| 4 | 2.77 - 2.52 | 0.9912 | 14254.5 |
| 5 | 2.52 - 2.34 | 0.9773 | 11723.8 |
| 6 | 2.34 - 2.20 | 0.9893 | 9160.4 |
| 7 | 2.20 - 2.09 | 0.9811 | 7547.9 |
| 8 | 2.09 - 2.00 | 0.9869 | 7152.3 |

## Refinement Progression

| Iteration | Target NLL |
| --------- | ---------- |
| 0 | -0.3618 |
| 1 | -0.3831 |
| 2 | -0.3932 |
| 3 | -0.3998 |

## Generated Artifacts

| Type | Path |
| ---- | ---- |
| MTZ Fourier Coefficients | `/Users/phzwart/Projects/phridge/examples/intensity_demo_output/phridge_demo_maps.mtz` |
| Gradient CCP4 Map | `/Users/phzwart/Projects/phridge/examples/intensity_demo_output/phridge_demo_gradient.ccp4` |
| $2mF_o - DF_c$ CCP4 Map | `/Users/phzwart/Projects/phridge/examples/intensity_demo_output/phridge_demo_2fofc.ccp4` |
| $mF_o - DF_c$ CCP4 Map | `/Users/phzwart/Projects/phridge/examples/intensity_demo_output/phridge_demo_fofc.ccp4` |
| Refined PDB Model | `/Users/phzwart/Projects/phridge/examples/intensity_demo_output/phridge_demo_refined.pdb` |

All checks passed successfully!
