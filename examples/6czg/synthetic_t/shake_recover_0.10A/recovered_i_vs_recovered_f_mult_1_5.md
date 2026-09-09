# Held-Out Log-Likelihood Comparison: `recovered_i` vs `recovered_f`

## 1. Executive Summary

- **Scored Audit Reflections (|A|)**: 900 (audit set)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `-0.0906` nats/refl
- **Model B NLL**: `-0.0900` nats/refl
- **Difference (Gain $\Delta$)**: `-0.0006` nats/refl (`+0.0006` nats NLL reduction)
- **Estimated Log Bayes Factor**: `-0.52` nats
- **Uncertainty**: Bootstrap SE = `0.0003` (95% CI: `[-0.0011, -0.0000]`) | Naive SE = `0.0002` (Ratio: `1.36`x)
- **Win Fraction $P(d_h > 0)$**: `42.0%` (378 wins, 522 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `1.79e-06`
- **Wilcoxon Signed-Rank $p$-value**: `2.03e-05`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.0006` | `-0.0001` | `+0.0006` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.0006` | `-0.0001` | `+0.0006` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`recovered_f`) | Model B (`recovered_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0069` | `-0.0070` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `-0.0903` | `-0.0895` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.0003` | `-0.0005` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -1.1499 | -1.1491 | -0.0007 | 0.0009 | 51.2% | 22/21 |
| 1 | 6.10 | 4.86 | 46 | 0.4732 | 0.4751 | -0.0019 | 0.0012 | 39.1% | 18/28 |
| 2 | 4.83 | 4.21 | 45 | -0.2710 | -0.2684 | -0.0027 | 0.0009 | 31.1% | 14/31 |
| 3 | 4.20 | 3.81 | 46 | 0.0829 | 0.0853 | -0.0023 | 0.0012 | 30.4% | 14/32 |
| 4 | 3.80 | 3.53 | 46 | 0.3225 | 0.3228 | -0.0003 | 0.0011 | 41.3% | 19/27 |
| 5 | 3.53 | 3.32 | 46 | 0.3667 | 0.3676 | -0.0009 | 0.0011 | 39.1% | 18/28 |
| 6 | 3.31 | 3.16 | 44 | -0.0852 | -0.0825 | -0.0027 | 0.0010 | 31.8% | 14/30 |
| 7 | 3.14 | 3.02 | 45 | 0.4448 | 0.4434 | +0.0014 | 0.0013 | 53.3% | 24/21 |
| 8 | 3.01 | 2.89 | 46 | -0.1578 | -0.1577 | -0.0001 | 0.0008 | 45.7% | 21/25 |
| 9 | 2.88 | 2.79 | 43 | -0.1419 | -0.1416 | -0.0003 | 0.0007 | 46.5% | 20/23 |
| 10 | 2.79 | 2.70 | 45 | -0.4332 | -0.4319 | -0.0012 | 0.0006 | 37.8% | 17/28 |
| 11 | 2.70 | 2.62 | 45 | -0.1669 | -0.1663 | -0.0006 | 0.0008 | 33.3% | 15/30 |
| 12 | 2.62 | 2.55 | 44 | -0.2991 | -0.2991 | -0.0000 | 0.0009 | 43.2% | 19/25 |
| 13 | 2.55 | 2.49 | 47 | -0.1596 | -0.1600 | +0.0004 | 0.0007 | 51.1% | 24/23 |
| 14 | 2.49 | 2.43 | 43 | -0.5171 | -0.5165 | -0.0006 | 0.0006 | 44.2% | 19/24 |
| 15 | 2.43 | 2.38 | 43 | 0.1270 | 0.1251 | +0.0019 | 0.0007 | 53.5% | 23/20 |
| 16 | 2.38 | 2.33 | 44 | -0.1734 | -0.1727 | -0.0007 | 0.0007 | 43.2% | 19/25 |
| 17 | 2.32 | 2.28 | 47 | -0.0667 | -0.0663 | -0.0004 | 0.0008 | 36.2% | 17/30 |
| 18 | 2.28 | 2.24 | 48 | -0.0619 | -0.0634 | +0.0014 | 0.0010 | 54.2% | 26/22 |
| 19 | 2.24 | 2.20 | 44 | -0.0430 | -0.0420 | -0.0010 | 0.0005 | 34.1% | 15/29 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (17, 5, 0) | 4.08 | 57159.6 | 1145.6 | 49.9 | 292.6 | 293.1 | 2.869 | 2.911 | -0.042 |
| (23, 3, 8) | 3.17 | 4655.4 | 96.6 | 48.2 | 95.4 | 95.8 | 0.663 | 0.695 | -0.031 |
| (-16, 4, 12) | 3.38 | 80054.6 | 917.4 | 87.3 | 328.6 | 329.0 | 3.066 | 3.095 | -0.030 |
| (32, 4, 12) | 2.25 | 3048.8 | 165.5 | 18.4 | 69.8 | 69.6 | 1.761 | 1.732 | +0.029 |
| (-9, 5, 4) | 5.30 | 73539.8 | 1570.0 | 46.8 | 332.1 | 332.5 | 2.624 | 2.652 | -0.029 |
| (1, 7, 2) | 4.96 | 79170.5 | 2055.1 | 38.5 | 354.0 | 353.8 | 3.807 | 3.780 | +0.027 |
| (11, 3, 0) | 6.46 | 59866.9 | 532.3 | 112.5 | 306.3 | 306.9 | 1.378 | 1.405 | -0.027 |
| (-1, 11, 4) | 3.14 | 11504.0 | 254.0 | 45.3 | 134.1 | 133.9 | 1.344 | 1.320 | +0.024 |
| (5, 9, 4) | 3.70 | 52203.8 | 1116.3 | 46.8 | 273.8 | 273.5 | 1.952 | 1.928 | +0.024 |
| (2, 10, 4) | 3.42 | 30854.4 | 510.8 | 60.4 | 212.4 | 212.2 | 2.064 | 2.041 | +0.023 |
| (-7, 1, 22) | 2.64 | 4794.6 | 116.9 | 41.0 | 88.7 | 88.5 | 1.222 | 1.201 | +0.021 |
| (12, 4, 8) | 4.41 | 20128.2 | 313.8 | 64.1 | 174.5 | 175.1 | 0.441 | 0.462 | -0.020 |
| (-9, 3, 14) | 3.70 | 51594.7 | 639.6 | 80.7 | 268.4 | 268.1 | 1.664 | 1.643 | +0.020 |
| (27, 1, 2) | 3.11 | 63525.2 | 1281.0 | 49.6 | 290.5 | 290.4 | 2.769 | 2.749 | +0.020 |
| (-18, 4, 10) | 3.43 | 53950.7 | 964.9 | 55.9 | 286.0 | 285.9 | 3.970 | 3.952 | +0.018 |
| (-14, 4, 16) | 3.00 | 7579.8 | 68.4 | 110.8 | 105.1 | 104.9 | 0.408 | 0.390 | +0.018 |
| (-22, 2, 17) | 2.59 | 2241.4 | 67.2 | 33.4 | 62.5 | 62.4 | 0.463 | 0.445 | +0.018 |
| (8, 2, 18) | 3.10 | 2419.8 | 160.0 | 15.1 | 71.8 | 72.0 | 0.520 | 0.538 | -0.018 |
| (-14, 6, 7) | 3.80 | 10915.8 | 461.4 | 23.7 | 145.7 | 145.5 | 1.133 | 1.116 | +0.018 |
| (32, 6, 2) | 2.41 | 13980.3 | 301.3 | 46.4 | 141.5 | 141.4 | 2.815 | 2.798 | +0.018 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `recovered_f` | 8.91 | 0.9977 | 7.157700865338023 | 7.086207309349849 | 1.000 |
| `recovered_i` | 8.94 | 0.9976 | 7.144230624326415 | 7.062454964122224 | 1.000 |