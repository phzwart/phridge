# Held-Out Log-Likelihood Comparison: `drifted_i` vs `drifted_f`

## 1. Executive Summary

- **Scored Test Reflections (|T|)**: 900 (held-out)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `-0.1002` nats/refl
- **Model B NLL**: `-0.0993` nats/refl
- **Difference (Gain $\Delta$)**: `-0.0009` nats/refl (`+0.0009` nats NLL reduction)
- **Estimated Log Bayes Factor**: `-0.79` nats
- **Uncertainty**: Bootstrap SE = `0.0012` (95% CI: `[-0.0032, +0.0013]`) | Naive SE = `0.0011` (Ratio: `1.06`x)
- **Win Fraction $P(d_h > 0)$**: `48.2%` (434 wins, 466 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `3.01e-01`
- **Wilcoxon Signed-Rank $p$-value**: `8.98e-01`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.0010` | `-0.0001` | `+0.0009` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.0010` | `-0.0001` | `+0.0009` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`drifted_f`) | Model B (`drifted_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0030` | `-0.0034` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `-0.0777` | `-0.0749` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.0224` | `-0.0244` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -1.1585 | -1.1502 | -0.0083 | 0.0038 | 32.6% | 14/29 |
| 1 | 6.10 | 4.86 | 46 | 0.6892 | 0.6928 | -0.0037 | 0.0086 | 45.7% | 21/25 |
| 2 | 4.83 | 4.21 | 45 | -0.2707 | -0.2566 | -0.0141 | 0.0049 | 37.8% | 17/28 |
| 3 | 4.20 | 3.81 | 46 | 0.1728 | 0.1741 | -0.0014 | 0.0071 | 56.5% | 26/20 |
| 4 | 3.80 | 3.53 | 46 | 0.3794 | 0.3880 | -0.0086 | 0.0068 | 41.3% | 19/27 |
| 5 | 3.53 | 3.32 | 46 | 0.4422 | 0.4467 | -0.0045 | 0.0103 | 45.7% | 21/25 |
| 6 | 3.31 | 3.16 | 44 | -0.1860 | -0.1850 | -0.0010 | 0.0038 | 43.2% | 19/25 |
| 7 | 3.14 | 3.02 | 45 | 0.4757 | 0.4783 | -0.0026 | 0.0057 | 42.2% | 19/26 |
| 8 | 3.01 | 2.89 | 46 | -0.2495 | -0.2503 | +0.0008 | 0.0030 | 50.0% | 23/23 |
| 9 | 2.88 | 2.79 | 43 | -0.2250 | -0.2231 | -0.0018 | 0.0023 | 37.2% | 16/27 |
| 10 | 2.79 | 2.70 | 45 | -0.4966 | -0.4945 | -0.0021 | 0.0021 | 44.4% | 20/25 |
| 11 | 2.70 | 2.62 | 45 | -0.1144 | -0.1184 | +0.0040 | 0.0030 | 55.6% | 25/20 |
| 12 | 2.62 | 2.55 | 44 | -0.2854 | -0.2936 | +0.0082 | 0.0021 | 61.4% | 27/17 |
| 13 | 2.55 | 2.49 | 47 | -0.1953 | -0.1977 | +0.0024 | 0.0029 | 59.6% | 28/19 |
| 14 | 2.49 | 2.43 | 43 | -0.4651 | -0.4711 | +0.0059 | 0.0032 | 67.4% | 29/14 |
| 15 | 2.43 | 2.38 | 43 | 0.0716 | 0.0682 | +0.0034 | 0.0033 | 44.2% | 19/24 |
| 16 | 2.38 | 2.33 | 44 | -0.2950 | -0.2950 | -0.0000 | 0.0025 | 52.3% | 23/21 |
| 17 | 2.32 | 2.28 | 47 | -0.0686 | -0.0693 | +0.0006 | 0.0029 | 42.6% | 20/27 |
| 18 | 2.28 | 2.24 | 48 | -0.2188 | -0.2217 | +0.0029 | 0.0024 | 50.0% | 24/24 |
| 19 | 2.24 | 2.20 | 44 | -0.1096 | -0.1120 | +0.0025 | 0.0024 | 54.5% | 24/20 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (-12, 8, 6) | 3.51 | 50136.9 | 1274.7 | 39.3 | 269.0 | 271.8 | 3.082 | 3.416 | -0.334 |
| (14, 2, 4) | 5.33 | 173387.8 | 4572.0 | 37.9 | 532.3 | 533.6 | 9.388 | 9.553 | -0.165 |
| (-8, 8, 4) | 3.94 | 47882.6 | 988.5 | 48.4 | 259.8 | 261.6 | 2.124 | 2.278 | -0.155 |
| (-9, 5, 11) | 3.93 | 21919.7 | 1134.3 | 19.3 | 180.6 | 182.7 | 1.283 | 1.435 | -0.151 |
| (-12, 4, 8) | 4.46 | 77065.1 | 3085.8 | 25.0 | 333.0 | 335.3 | 2.033 | 2.183 | -0.150 |
| (-9, 3, 14) | 3.70 | 51904.5 | 1279.2 | 40.6 | 267.3 | 269.5 | 1.638 | 1.787 | -0.149 |
| (-18, 4, 10) | 3.43 | 49670.1 | 1929.9 | 25.7 | 277.3 | 278.3 | 4.453 | 4.598 | -0.145 |
| (2, 10, 4) | 3.42 | 31102.7 | 1021.5 | 30.4 | 215.1 | 216.3 | 2.466 | 2.594 | -0.128 |
| (-10, 4, 2) | 6.01 | 18917.4 | 752.7 | 25.1 | 177.8 | 175.5 | 0.951 | 0.828 | +0.123 |
| (-9, 5, 4) | 5.30 | 77244.7 | 3140.1 | 24.6 | 334.5 | 333.1 | 2.529 | 2.411 | +0.117 |
| (24, 2, 2) | 3.44 | 30941.8 | 1048.5 | 29.5 | 213.9 | 212.9 | 2.354 | 2.237 | +0.117 |
| (27, 1, 2) | 3.11 | 60265.8 | 2562.0 | 23.5 | 304.3 | 305.0 | 6.257 | 6.371 | -0.114 |
| (-13, 1, 8) | 4.90 | 286196.6 | 5640.0 | 50.7 | 647.9 | 648.8 | 9.235 | 9.348 | -0.113 |
| (-5, 1, 15) | 3.86 | 7406.9 | 493.8 | 15.0 | 114.6 | 113.0 | 0.710 | 0.598 | +0.112 |
| (14, 4, 0) | 5.00 | 55468.7 | 1602.0 | 34.6 | 296.0 | 294.9 | 3.042 | 2.931 | +0.110 |
| (-6, 10, 1) | 3.42 | 7236.0 | 534.6 | 13.5 | 106.5 | 104.8 | 0.530 | 0.420 | +0.109 |
| (12, 0, 12) | 4.03 | 58936.2 | 4967.7 | 11.9 | 288.1 | 290.8 | 1.885 | 1.993 | -0.108 |
| (10, 4, 2) | 5.98 | 69335.0 | 1240.5 | 55.9 | 316.7 | 318.4 | 2.096 | 2.204 | -0.108 |
| (19, 1, 10) | 3.53 | 15251.7 | 498.0 | 30.6 | 148.4 | 149.9 | 0.934 | 1.037 | -0.102 |
| (-9, 5, 5) | 5.13 | 9320.4 | 490.5 | 19.0 | 120.9 | 118.3 | 0.103 | 0.002 | +0.101 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `drifted_f` | 7.92 | 0.9978 | 199.20328969208404 | 653.8373836511294 | 1.000 |
| `drifted_i` | 7.95 | 0.9978 | 199.20381643503458 | 654.2130007284266 | 1.000 |