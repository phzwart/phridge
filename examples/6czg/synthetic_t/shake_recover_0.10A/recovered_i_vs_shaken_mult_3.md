# Held-Out Log-Likelihood Comparison: `recovered_i` vs `shaken`

## 1. Executive Summary

- **Scored Test Reflections (|T|)**: 900 (held-out)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `0.0977` nats/refl
- **Model B NLL**: `-0.0498` nats/refl
- **Difference (Gain $\Delta$)**: `+0.1475` nats/refl (`-0.1475` nats NLL reduction)
- **Estimated Log Bayes Factor**: `+132.78` nats
- **Uncertainty**: Bootstrap SE = `0.0681` (95% CI: `[+0.0024, +0.2666]`) | Naive SE = `0.0271` (Ratio: `2.51`x)
- **Win Fraction $P(d_h > 0)$**: `63.6%` (572 wins, 328 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `3.65e-16`
- **Wilcoxon Signed-Rank $p$-value**: `1.97e-13`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `-0.0598` | `-0.0877` | `-0.1475` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `-0.2020` | `+0.0544` | `-0.1475` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`shaken`) | Model B (`recovered_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0061` | `-0.0091` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `0.1984` | `-0.0479` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.1006` | `-0.0019` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -1.3545 | -1.1286 | -0.2259 | 0.2072 | 25.6% | 11/32 |
| 1 | 6.10 | 4.86 | 46 | -0.1511 | 0.4733 | -0.6244 | 0.1531 | 19.6% | 9/37 |
| 2 | 4.83 | 4.21 | 45 | -0.4764 | -0.2386 | -0.2378 | 0.0822 | 22.2% | 10/35 |
| 3 | 4.20 | 3.81 | 46 | 0.2116 | 0.1296 | +0.0820 | 0.1155 | 50.0% | 23/23 |
| 4 | 3.80 | 3.53 | 46 | 0.1191 | 0.3703 | -0.2513 | 0.1549 | 30.4% | 14/32 |
| 5 | 3.53 | 3.32 | 46 | 0.3606 | 0.3700 | -0.0093 | 0.1596 | 47.8% | 22/24 |
| 6 | 3.31 | 3.16 | 44 | -0.1138 | -0.1093 | -0.0045 | 0.0762 | 54.5% | 24/20 |
| 7 | 3.14 | 3.02 | 45 | 0.3802 | 0.4878 | -0.1076 | 0.1074 | 51.1% | 23/22 |
| 8 | 3.01 | 2.89 | 46 | 0.0306 | -0.1530 | +0.1836 | 0.0766 | 71.7% | 33/13 |
| 9 | 2.88 | 2.79 | 43 | 0.1042 | -0.1452 | +0.2494 | 0.0966 | 67.4% | 29/14 |
| 10 | 2.79 | 2.70 | 45 | -0.1824 | -0.4650 | +0.2826 | 0.0685 | 80.0% | 36/9 |
| 11 | 2.70 | 2.62 | 45 | 0.2920 | -0.0797 | +0.3716 | 0.1108 | 73.3% | 33/12 |
| 12 | 2.62 | 2.55 | 44 | -0.0315 | -0.2554 | +0.2239 | 0.0820 | 75.0% | 33/11 |
| 13 | 2.55 | 2.49 | 47 | 0.2530 | -0.0968 | +0.3498 | 0.1003 | 80.9% | 38/9 |
| 14 | 2.49 | 2.43 | 43 | 0.1547 | -0.3928 | +0.5475 | 0.0907 | 90.7% | 39/4 |
| 15 | 2.43 | 2.38 | 43 | 0.6445 | 0.2425 | +0.4020 | 0.1179 | 83.7% | 36/7 |
| 16 | 2.38 | 2.33 | 44 | 0.2629 | -0.1394 | +0.4023 | 0.0924 | 81.8% | 36/8 |
| 17 | 2.32 | 2.28 | 47 | 0.4897 | 0.0676 | +0.4221 | 0.0746 | 87.2% | 41/6 |
| 18 | 2.28 | 2.24 | 48 | 0.3945 | -0.0451 | +0.4395 | 0.0724 | 89.6% | 43/5 |
| 19 | 2.24 | 2.20 | 44 | 0.4882 | 0.0155 | +0.4727 | 0.0675 | 88.6% | 39/5 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (3, 1, 0) | 22.09 | 21820.7 | 544.5 | 40.1 | 193.0 | 175.6 | 7.497 | -0.342 | +7.839 |
| (9, 3, 14) | 3.66 | 120355.5 | 3329.7 | 36.1 | 401.0 | 445.5 | 2.867 | 7.970 | -5.103 |
| (20, 4, 7) | 3.47 | 514.5 | 106.5 | 4.8 | 80.4 | 33.5 | 3.705 | -1.183 | +4.888 |
| (14, 2, 4) | 5.33 | 173387.8 | 4572.0 | 37.9 | 471.7 | 532.1 | 3.148 | 8.019 | -4.871 |
| (-18, 8, 15) | 2.52 | 1764.1 | 173.1 | 10.2 | 80.9 | 48.6 | 3.603 | -0.096 | +3.699 |
| (-13, 1, 8) | 4.90 | 286196.6 | 5640.0 | 50.7 | 598.6 | 646.3 | 3.941 | 7.639 | -3.698 |
| (-30, 2, 14) | 2.35 | 1239.0 | 126.0 | 9.8 | 69.7 | 42.8 | 3.180 | -0.005 | +3.185 |
| (17, 5, 0) | 4.08 | 53884.0 | 2291.1 | 23.5 | 233.8 | 293.4 | 0.703 | 3.520 | -2.817 |
| (-7, 1, 22) | 2.64 | 4340.6 | 233.7 | 18.6 | 110.8 | 87.8 | 4.251 | 1.467 | +2.784 |
| (-5, 9, 1) | 3.82 | 3986.0 | 274.5 | 14.5 | 112.1 | 82.9 | 2.520 | -0.048 | +2.568 |
| (14, 2, 12) | 3.73 | 9072.7 | 892.2 | 10.2 | 155.3 | 128.8 | 3.120 | 0.620 | +2.500 |
| (-18, 4, 10) | 3.43 | 49670.1 | 1929.9 | 25.7 | 261.8 | 283.0 | 2.180 | 4.673 | -2.493 |
| (-12, 8, 6) | 3.51 | 50136.9 | 1274.7 | 39.3 | 249.5 | 277.5 | 1.274 | 3.672 | -2.398 |
| (16, 8, 11) | 2.86 | 214.2 | 89.1 | 2.4 | 48.0 | 23.1 | 1.064 | -1.300 | +2.364 |
| (0, 4, 12) | 4.33 | 81200.9 | 1243.2 | 65.3 | 295.5 | 351.4 | 0.410 | 2.760 | -2.349 |
| (16, 10, 13) | 2.46 | 107.3 | 81.6 | 1.3 | 38.4 | 15.9 | 0.947 | -1.386 | +2.333 |
| (-15, 3, 23) | 2.32 | 2412.1 | 244.2 | 9.9 | 23.7 | 43.2 | 2.691 | 0.401 | +2.290 |
| (1, 7, 2) | 4.96 | 78832.9 | 4110.3 | 19.2 | 314.2 | 353.6 | 1.388 | 3.644 | -2.256 |
| (35, 1, 9) | 2.26 | 236.7 | 107.1 | 2.2 | 47.3 | 29.6 | 2.969 | 0.754 | +2.215 |
| (16, 8, 17) | 2.43 | -222.1 | 145.2 | -1.5 | 5.5 | 26.5 | 0.961 | 3.127 | -2.167 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `shaken` | 18.03 | 0.9895 | 199.17921810494738 | 903.8683114097493 | 1.000 |
| `recovered_i` | 10.19 | 0.9963 | 199.19342066648593 | 792.7415793823787 | 1.000 |