# Held-Out Log-Likelihood Comparison: `recovered_i` vs `shaken`

## 1. Executive Summary

- **Scored Audit Reflections (|A|)**: 900 (audit set)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `0.5008` nats/refl
- **Model B NLL**: `0.0782` nats/refl
- **Difference (Gain $\Delta$)**: `+0.4225` nats/refl (`-0.4225` nats NLL reduction)
- **Estimated Log Bayes Factor**: `+380.29` nats
- **Uncertainty**: Bootstrap SE = `0.0329` (95% CI: `[+0.3522, +0.4818]`) | Naive SE = `0.0297` (Ratio: `1.11`x)
- **Win Fraction $P(d_h > 0)$**: `79.1%` (712 wins, 188 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `2.15e-72`
- **Wilcoxon Signed-Rank $p$-value**: `2.66e-61`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `-0.2994` | `-0.1231` | `-0.4225` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `-0.6375` | `+0.2150` | `-0.4225` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`shaken`) | Model B (`recovered_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `+0.0019` | `-0.0132` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `0.6082` | `0.0842` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.1074` | `-0.0060` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -0.9646 | -1.3493 | +0.3847 | 0.2540 | 46.5% | 20/23 |
| 1 | 6.10 | 4.86 | 46 | 0.2139 | 0.1040 | +0.1099 | 0.1289 | 69.6% | 32/14 |
| 2 | 4.83 | 4.21 | 45 | -0.1279 | -0.3639 | +0.2360 | 0.0953 | 75.6% | 34/11 |
| 3 | 4.20 | 3.81 | 46 | 0.5317 | 0.0335 | +0.4981 | 0.1144 | 93.5% | 43/3 |
| 4 | 3.80 | 3.53 | 46 | 0.4041 | 0.2538 | +0.1502 | 0.1306 | 76.1% | 35/11 |
| 5 | 3.53 | 3.32 | 46 | 0.9915 | 0.3239 | +0.6676 | 0.1974 | 73.9% | 34/12 |
| 6 | 3.31 | 3.16 | 44 | 0.1961 | -0.0279 | +0.2240 | 0.0819 | 75.0% | 33/11 |
| 7 | 3.14 | 3.02 | 45 | 0.8541 | 0.4715 | +0.3827 | 0.1388 | 77.8% | 35/10 |
| 8 | 3.01 | 2.89 | 46 | 0.4128 | -0.0402 | +0.4530 | 0.1052 | 84.8% | 39/7 |
| 9 | 2.88 | 2.79 | 43 | 0.5190 | 0.1390 | +0.3800 | 0.1159 | 74.4% | 32/11 |
| 10 | 2.79 | 2.70 | 45 | 0.2947 | -0.1671 | +0.4618 | 0.0843 | 91.1% | 41/4 |
| 11 | 2.70 | 2.62 | 45 | 0.6344 | 0.0815 | +0.5528 | 0.1150 | 88.9% | 40/5 |
| 12 | 2.62 | 2.55 | 44 | 0.4994 | -0.0682 | +0.5676 | 0.0881 | 84.1% | 37/7 |
| 13 | 2.55 | 2.49 | 47 | 0.6294 | 0.1692 | +0.4602 | 0.1066 | 78.7% | 37/10 |
| 14 | 2.49 | 2.43 | 43 | 0.4747 | -0.0419 | +0.5166 | 0.1188 | 79.1% | 34/9 |
| 15 | 2.43 | 2.38 | 43 | 0.8787 | 0.4615 | +0.4173 | 0.1448 | 88.4% | 38/5 |
| 16 | 2.38 | 2.33 | 44 | 0.8894 | 0.2519 | +0.6375 | 0.1563 | 81.8% | 36/8 |
| 17 | 2.32 | 2.28 | 47 | 0.8515 | 0.3551 | +0.4965 | 0.1006 | 76.6% | 36/11 |
| 18 | 2.28 | 2.24 | 48 | 0.8068 | 0.4142 | +0.3926 | 0.0941 | 77.1% | 37/11 |
| 19 | 2.24 | 2.20 | 44 | 0.9454 | 0.4765 | +0.4689 | 0.1294 | 88.6% | 39/5 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (3, 1, 0) | 22.09 | 21112.0 | 272.2 | 77.5 | 204.5 | 199.3 | 10.709 | 0.596 | +10.114 |
| (20, 4, 7) | 3.47 | 531.1 | 53.2 | 10.0 | 127.8 | 35.1 | 5.198 | -1.075 | +6.273 |
| (-16, 4, 12) | 3.38 | 80054.6 | 917.4 | 87.3 | 224.8 | 308.6 | 5.703 | 1.423 | +4.280 |
| (-1, 1, 3) | 17.00 | 198.2 | 44.8 | 4.4 | 57.6 | 29.6 | 1.144 | -2.810 | +3.954 |
| (-30, 2, 14) | 2.35 | 1122.6 | 63.0 | 17.8 | 95.6 | 41.4 | 3.859 | 0.058 | +3.800 |
| (-20, 2, 14) | 2.98 | 15598.6 | 134.2 | 116.2 | 51.6 | 117.0 | 4.380 | 0.642 | +3.738 |
| (-30, 4, 4) | 2.66 | 15866.4 | 265.2 | 59.8 | 79.2 | 130.7 | 4.471 | 0.969 | +3.501 |
| (-29, 9, 3) | 2.33 | -64.8 | 41.8 | -1.5 | 63.6 | 20.8 | 5.662 | 2.177 | +3.486 |
| (-12, 8, 19) | 2.41 | 6035.3 | 117.3 | 51.5 | 31.4 | 84.3 | 4.204 | 0.755 | +3.449 |
| (2, 2, 14) | 4.11 | 124622.5 | 1711.6 | 72.8 | 423.9 | 437.9 | 2.396 | 5.825 | -3.429 |
| (27, 7, 12) | 2.34 | 11715.6 | 148.6 | 78.8 | 75.4 | 118.9 | 4.487 | 1.137 | +3.349 |
| (-24, 2, 2) | 3.45 | 55504.6 | 1155.1 | 48.0 | 182.7 | 246.5 | 4.351 | 1.036 | +3.315 |
| (-13, 1, 8) | 4.90 | 286163.9 | 2820.0 | 101.5 | 553.9 | 619.7 | 1.727 | 4.887 | -3.160 |
| (16, 10, 13) | 2.46 | 34.6 | 40.8 | 0.8 | 62.6 | 19.8 | 2.193 | -0.889 | +3.082 |
| (27, 1, 2) | 3.11 | 63525.2 | 1281.0 | 49.6 | 213.4 | 272.1 | 4.388 | 1.361 | +3.026 |
| (9, 3, 14) | 3.66 | 132472.5 | 1664.9 | 79.6 | 363.5 | 441.6 | 1.724 | 4.723 | -2.999 |
| (-25, 11, 8) | 2.23 | 6973.5 | 122.4 | 57.0 | 56.1 | 92.2 | 3.934 | 1.103 | +2.831 |
| (15, 7, 4) | 3.64 | 72767.8 | 1165.5 | 62.4 | 212.6 | 287.4 | 3.508 | 0.827 | +2.681 |
| (16, 8, 17) | 2.43 | 150.4 | 72.6 | 2.1 | 4.9 | 41.8 | -0.970 | 1.667 | -2.637 |
| (-19, 1, 20) | 2.49 | -39.2 | 56.9 | -0.7 | 49.8 | 5.9 | 2.138 | -0.492 | +2.631 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `shaken` | 35.13 | 0.9592 | 3.1024938534637583 | 1.791144890696521 | 1.000 |
| `recovered_i` | 15.79 | 0.9921 | 4.709337646368276 | 3.471521180753639 | 1.000 |