# Held-Out Log-Likelihood Comparison: `recovered_i` vs `recovered_f`

## 1. Executive Summary

- **Scored Test Reflections (|T|)**: 900 (held-out)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `-0.0512` nats/refl
- **Model B NLL**: `-0.0498` nats/refl
- **Difference (Gain $\Delta$)**: `-0.0013` nats/refl (`+0.0013` nats NLL reduction)
- **Estimated Log Bayes Factor**: `-1.21` nats
- **Uncertainty**: Bootstrap SE = `0.0006` (95% CI: `[-0.0027, -0.0001]`) | Naive SE = `0.0004` (Ratio: `1.44`x)
- **Win Fraction $P(d_h > 0)$**: `40.8%` (367 wins, 533 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `3.49e-08`
- **Wilcoxon Signed-Rank $p$-value**: `7.38e-04`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.0014` | `-0.0001` | `+0.0013` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.0014` | `-0.0001` | `+0.0013` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`recovered_f`) | Model B (`recovered_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0091` | `-0.0091` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `-0.0493` | `-0.0479` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.0019` | `-0.0019` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -1.1295 | -1.1286 | -0.0008 | 0.0014 | 34.9% | 15/28 |
| 1 | 6.10 | 4.86 | 46 | 0.4672 | 0.4733 | -0.0061 | 0.0028 | 39.1% | 18/28 |
| 2 | 4.83 | 4.21 | 45 | -0.2400 | -0.2386 | -0.0014 | 0.0018 | 37.8% | 17/28 |
| 3 | 4.20 | 3.81 | 46 | 0.1237 | 0.1296 | -0.0058 | 0.0027 | 34.8% | 16/30 |
| 4 | 3.80 | 3.53 | 46 | 0.3648 | 0.3703 | -0.0055 | 0.0024 | 39.1% | 18/28 |
| 5 | 3.53 | 3.32 | 46 | 0.3678 | 0.3700 | -0.0022 | 0.0029 | 50.0% | 23/23 |
| 6 | 3.31 | 3.16 | 44 | -0.1147 | -0.1093 | -0.0054 | 0.0025 | 31.8% | 14/30 |
| 7 | 3.14 | 3.02 | 45 | 0.4882 | 0.4878 | +0.0004 | 0.0025 | 55.6% | 25/20 |
| 8 | 3.01 | 2.89 | 46 | -0.1564 | -0.1530 | -0.0034 | 0.0016 | 37.0% | 17/29 |
| 9 | 2.88 | 2.79 | 43 | -0.1468 | -0.1452 | -0.0016 | 0.0013 | 34.9% | 15/28 |
| 10 | 2.79 | 2.70 | 45 | -0.4672 | -0.4650 | -0.0022 | 0.0012 | 31.1% | 14/31 |
| 11 | 2.70 | 2.62 | 45 | -0.0772 | -0.0797 | +0.0025 | 0.0013 | 53.3% | 24/21 |
| 12 | 2.62 | 2.55 | 44 | -0.2564 | -0.2554 | -0.0010 | 0.0013 | 47.7% | 21/23 |
| 13 | 2.55 | 2.49 | 47 | -0.0941 | -0.0968 | +0.0028 | 0.0014 | 51.1% | 24/23 |
| 14 | 2.49 | 2.43 | 43 | -0.3923 | -0.3928 | +0.0005 | 0.0017 | 39.5% | 17/26 |
| 15 | 2.43 | 2.38 | 43 | 0.2453 | 0.2425 | +0.0029 | 0.0021 | 44.2% | 19/24 |
| 16 | 2.38 | 2.33 | 44 | -0.1398 | -0.1394 | -0.0004 | 0.0014 | 40.9% | 18/26 |
| 17 | 2.32 | 2.28 | 47 | 0.0658 | 0.0676 | -0.0018 | 0.0016 | 36.2% | 17/30 |
| 18 | 2.28 | 2.24 | 48 | -0.0428 | -0.0451 | +0.0022 | 0.0015 | 43.8% | 21/27 |
| 19 | 2.24 | 2.20 | 44 | 0.0153 | 0.0155 | -0.0002 | 0.0014 | 31.8% | 14/30 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (17, 5, 0) | 4.08 | 53884.0 | 2291.1 | 23.5 | 292.3 | 293.4 | 3.419 | 3.520 | -0.101 |
| (-16, 4, 12) | 3.38 | 80259.0 | 1834.8 | 43.7 | 328.1 | 328.8 | 2.886 | 2.956 | -0.070 |
| (32, 6, 2) | 2.41 | 13800.6 | 602.7 | 22.9 | 139.9 | 139.6 | 2.535 | 2.470 | +0.064 |
| (-18, 4, 10) | 3.43 | 49670.1 | 1929.9 | 25.7 | 282.4 | 283.0 | 4.618 | 4.673 | -0.055 |
| (19, 7, 4) | 3.26 | 3905.2 | 277.8 | 14.1 | 81.4 | 82.4 | -0.070 | -0.016 | -0.054 |
| (14, 2, 4) | 5.33 | 173387.8 | 4572.0 | 37.9 | 531.4 | 532.1 | 7.965 | 8.019 | -0.054 |
| (-2, 10, 8) | 3.18 | 38800.2 | 1075.2 | 36.1 | 233.8 | 234.4 | 2.007 | 2.058 | -0.051 |
| (-12, 10, 2) | 3.14 | 17903.0 | 1116.6 | 16.0 | 166.9 | 166.4 | 1.970 | 1.920 | +0.050 |
| (-12, 8, 6) | 3.51 | 50136.9 | 1274.7 | 39.3 | 277.7 | 277.5 | 3.721 | 3.672 | +0.050 |
| (1, 7, 2) | 4.96 | 78832.9 | 4110.3 | 19.2 | 352.9 | 353.6 | 3.596 | 3.644 | -0.048 |
| (-9, 5, 4) | 5.30 | 77244.7 | 3140.1 | 24.6 | 332.9 | 333.8 | 2.166 | 2.213 | -0.047 |
| (-14, 4, 13) | 3.40 | 9574.6 | 132.3 | 72.4 | 124.1 | 124.7 | 0.914 | 0.959 | -0.045 |
| (-1, 11, 4) | 3.14 | 11533.4 | 507.9 | 22.7 | 132.3 | 132.9 | 1.148 | 1.192 | -0.044 |
| (9, 5, 4) | 5.27 | 125566.9 | 4317.0 | 29.1 | 423.4 | 424.1 | 3.163 | 3.206 | -0.043 |
| (8, 2, 18) | 3.10 | 2691.9 | 320.1 | 8.4 | 71.7 | 72.2 | 0.313 | 0.355 | -0.041 |
| (-7, 1, 22) | 2.64 | 4340.6 | 233.7 | 18.6 | 88.0 | 87.8 | 1.508 | 1.467 | +0.041 |
| (0, 2, 19) | 3.09 | 5730.4 | 329.1 | 17.4 | 89.9 | 90.7 | 0.257 | 0.297 | -0.041 |
| (-11, 7, 14) | 3.01 | 11494.9 | 490.2 | 23.4 | 127.4 | 128.0 | 0.683 | 0.723 | -0.040 |
| (-9, 1, 17) | 3.29 | 40585.2 | 795.6 | 51.0 | 243.9 | 243.5 | 2.328 | 2.288 | +0.040 |
| (-28, 10, 1) | 2.30 | 1350.8 | 116.4 | 11.6 | 25.0 | 24.8 | 0.743 | 0.783 | -0.040 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `recovered_f` | 10.17 | 0.9963 | 199.19333434868048 | 792.8839187958348 | 1.000 |
| `recovered_i` | 10.19 | 0.9963 | 199.19342066648593 | 792.7415793823787 | 1.000 |