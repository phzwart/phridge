# Held-Out Log-Likelihood Comparison: `recovered_f` vs `deposited`

## 1. Executive Summary

- **Scored Test Reflections (|T|)**: 900 (held-out)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `1.1555` nats/refl
- **Model B NLL**: `1.1822` nats/refl
- **Difference (Gain $\Delta$)**: `-0.0267` nats/refl (`+0.0267` nats NLL reduction)
- **Estimated Log Bayes Factor**: `-24.06` nats
- **Uncertainty**: Bootstrap SE = `0.0080` (95% CI: `[-0.0423, -0.0108]`) | Naive SE = `0.0087` (Ratio: `0.92`x)
- **Win Fraction $P(d_h > 0)$**: `40.8%` (367 wins, 533 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `3.49e-08`
- **Wilcoxon Signed-Rank $p$-value**: `3.86e-06`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.0347` | `-0.0080` | `+0.0267` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.0259` | `+0.0008` | `+0.0267` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`deposited`) | Model B (`recovered_f`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0069` | `-0.0080` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `1.3184` | `1.3369` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.1629` | `-0.1547` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -0.8849 | -0.8686 | -0.0163 | 0.0616 | 37.2% | 16/27 |
| 1 | 6.10 | 4.86 | 46 | 0.6455 | 0.6768 | -0.0313 | 0.0346 | 45.7% | 21/25 |
| 2 | 4.83 | 4.21 | 45 | 0.4586 | 0.5536 | -0.0950 | 0.0539 | 28.9% | 13/32 |
| 3 | 4.20 | 3.81 | 46 | 0.9819 | 0.9499 | +0.0320 | 0.0354 | 41.3% | 19/27 |
| 4 | 3.80 | 3.53 | 46 | 1.2523 | 1.3087 | -0.0564 | 0.0398 | 32.6% | 15/31 |
| 5 | 3.53 | 3.32 | 46 | 1.3688 | 1.3653 | +0.0035 | 0.0356 | 39.1% | 18/28 |
| 6 | 3.31 | 3.16 | 44 | 1.1069 | 1.1754 | -0.0685 | 0.0338 | 43.2% | 19/25 |
| 7 | 3.14 | 3.02 | 45 | 1.3792 | 1.4418 | -0.0626 | 0.0545 | 48.9% | 22/23 |
| 8 | 3.01 | 2.89 | 46 | 0.7427 | 0.8193 | -0.0766 | 0.0511 | 39.1% | 18/28 |
| 9 | 2.88 | 2.79 | 43 | 0.9986 | 1.0478 | -0.0493 | 0.0353 | 41.9% | 18/25 |
| 10 | 2.79 | 2.70 | 45 | 0.8386 | 0.8279 | +0.0107 | 0.0423 | 40.0% | 18/27 |
| 11 | 2.70 | 2.62 | 45 | 1.4649 | 1.4665 | -0.0016 | 0.0331 | 46.7% | 21/24 |
| 12 | 2.62 | 2.55 | 44 | 1.3107 | 1.3176 | -0.0068 | 0.0311 | 50.0% | 22/22 |
| 13 | 2.55 | 2.49 | 47 | 1.4068 | 1.4114 | -0.0046 | 0.0359 | 36.2% | 17/30 |
| 14 | 2.49 | 2.43 | 43 | 1.1857 | 1.1733 | +0.0124 | 0.0311 | 55.8% | 24/19 |
| 15 | 2.43 | 2.38 | 43 | 1.7405 | 1.8072 | -0.0667 | 0.0326 | 32.6% | 14/29 |
| 16 | 2.38 | 2.33 | 44 | 1.4157 | 1.4651 | -0.0494 | 0.0367 | 22.7% | 10/34 |
| 17 | 2.32 | 2.28 | 47 | 1.8088 | 1.8080 | +0.0007 | 0.0280 | 48.9% | 23/24 |
| 18 | 2.28 | 2.24 | 48 | 1.8633 | 1.8716 | -0.0083 | 0.0330 | 45.8% | 22/26 |
| 19 | 2.24 | 2.20 | 44 | 1.9094 | 1.9145 | -0.0050 | 0.0179 | 38.6% | 17/27 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (24, 0, 10) | 3.02 | 49740.7 | 6740.0 | 7.4 | 192.8 | 167.9 | 2.407 | 4.210 | -1.803 |
| (-20, 2, 14) | 2.98 | 21187.8 | 4475.0 | 4.7 | 129.8 | 108.3 | 1.005 | 2.356 | -1.351 |
| (-3, 5, 11) | 4.26 | 10860.2 | 1340.0 | 8.1 | 59.3 | 55.7 | 5.953 | 4.653 | +1.300 |
| (8, 0, 12) | 4.47 | 9414.3 | 3940.0 | 2.4 | 102.6 | 132.3 | -0.490 | 0.673 | -1.162 |
| (-1, 5, 4) | 6.37 | 22235.6 | 3590.0 | 6.2 | 182.7 | 179.4 | 2.165 | 1.017 | +1.147 |
| (13, 5, 7) | 4.16 | 14029.5 | 2660.0 | 5.3 | 80.5 | 90.0 | 2.178 | 1.035 | +1.143 |
| (17, 3, 3) | 4.47 | -400.1 | 1665.0 | -0.2 | 44.3 | 60.2 | -0.478 | 0.626 | -1.104 |
| (6, 2, 2) | 10.31 | 11568.0 | 515.0 | 22.5 | 111.1 | 118.6 | -2.249 | -1.167 | -1.082 |
| (6, 0, 7) | 7.24 | 3275.8 | 1265.0 | 2.6 | 22.1 | 34.1 | -0.236 | -1.215 | +0.979 |
| (-30, 2, 4) | 2.75 | 33.8 | 1710.0 | 0.0 | 49.7 | 14.3 | 0.843 | -0.116 | +0.959 |
| (-2, 2, 5) | 9.64 | 5606.6 | 1410.0 | 4.0 | 51.5 | 59.2 | -0.467 | -1.401 | +0.935 |
| (26, 4, 2) | 3.04 | 6607.7 | 2215.0 | 3.0 | 87.0 | 107.0 | 0.277 | 1.199 | -0.921 |
| (20, 2, 14) | 2.94 | 21827.8 | 7405.0 | 2.9 | 91.0 | 116.0 | 2.641 | 1.724 | +0.917 |
| (-3, 5, 24) | 2.34 | -578.6 | 2075.0 | -0.3 | 39.1 | 56.9 | 1.400 | 2.311 | -0.910 |
| (5, 7, 16) | 2.94 | 1440.0 | 11475.0 | 0.1 | 128.2 | 156.4 | 2.340 | 3.222 | -0.882 |
| (8, 12, 16) | 2.25 | -1066.5 | 1855.0 | -0.6 | 64.9 | 59.2 | 4.214 | 3.337 | +0.876 |
| (17, 3, 14) | 3.10 | 11907.1 | 2670.0 | 4.5 | 108.7 | 86.7 | 0.382 | 1.252 | -0.870 |
| (-7, 11, 10) | 2.76 | 2373.7 | 1395.0 | 1.7 | 60.7 | 72.1 | 0.110 | 0.970 | -0.860 |
| (18, 12, 2) | 2.49 | 5451.0 | 1965.0 | 2.8 | 42.8 | 55.3 | 2.038 | 1.184 | +0.854 |
| (2, 14, 12) | 2.24 | 70.6 | 1250.0 | 0.1 | 34.8 | 47.8 | 1.038 | 1.889 | -0.851 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `deposited` | 73.06 | 0.7118 | 8.316639414194936 | 2.2151127829374033 | 1.000 |
| `recovered_f` | 73.98 | 0.7108 | 9.387353003246394 | 2.9075257851968495 | 1.000 |