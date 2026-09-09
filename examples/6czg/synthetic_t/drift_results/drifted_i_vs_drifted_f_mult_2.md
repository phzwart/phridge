# Held-Out Log-Likelihood Comparison: `drifted_i` vs `drifted_f`

## 1. Executive Summary

- **Scored Audit Reflections (|A|)**: 900 (audit set)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `-0.1294` nats/refl
- **Model B NLL**: `-0.1293` nats/refl
- **Difference (Gain $\Delta$)**: `-0.0001` nats/refl (`+0.0001` nats NLL reduction)
- **Estimated Log Bayes Factor**: `-0.08` nats
- **Uncertainty**: Bootstrap SE = `0.0010` (95% CI: `[-0.0022, +0.0017]`) | Naive SE = `0.0006` (Ratio: `1.60`x)
- **Win Fraction $P(d_h > 0)$**: `51.6%` (464 wins, 436 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `3.68e-01`
- **Wilcoxon Signed-Rank $p$-value**: `2.25e-01`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.0003` | `-0.0002` | `+0.0001` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.0003` | `-0.0002` | `+0.0001` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`drifted_f`) | Model B (`drifted_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `+0.0007` | `+0.0005` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `-0.0997` | `-0.0979` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.0297` | `-0.0314` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -1.1584 | -1.1585 | +0.0001 | 0.0020 | 41.9% | 18/25 |
| 1 | 6.10 | 4.86 | 46 | 0.6962 | 0.7058 | -0.0097 | 0.0041 | 41.3% | 19/27 |
| 2 | 4.83 | 4.21 | 45 | -0.2682 | -0.2616 | -0.0066 | 0.0042 | 48.9% | 22/23 |
| 3 | 4.20 | 3.81 | 46 | 0.1748 | 0.1797 | -0.0049 | 0.0040 | 41.3% | 19/27 |
| 4 | 3.80 | 3.53 | 46 | 0.3123 | 0.3215 | -0.0092 | 0.0040 | 43.5% | 20/26 |
| 5 | 3.53 | 3.32 | 46 | 0.5946 | 0.5932 | +0.0014 | 0.0046 | 43.5% | 20/26 |
| 6 | 3.31 | 3.16 | 44 | -0.1828 | -0.1819 | -0.0009 | 0.0027 | 52.3% | 23/21 |
| 7 | 3.14 | 3.02 | 45 | 0.3767 | 0.3756 | +0.0010 | 0.0031 | 46.7% | 21/24 |
| 8 | 3.01 | 2.89 | 46 | -0.2621 | -0.2611 | -0.0010 | 0.0017 | 54.3% | 25/21 |
| 9 | 2.88 | 2.79 | 43 | -0.2578 | -0.2580 | +0.0001 | 0.0017 | 37.2% | 16/27 |
| 10 | 2.79 | 2.70 | 45 | -0.5336 | -0.5360 | +0.0023 | 0.0013 | 68.9% | 31/14 |
| 11 | 2.70 | 2.62 | 45 | -0.1768 | -0.1791 | +0.0023 | 0.0021 | 46.7% | 21/24 |
| 12 | 2.62 | 2.55 | 44 | -0.2600 | -0.2650 | +0.0050 | 0.0016 | 65.9% | 29/15 |
| 13 | 2.55 | 2.49 | 47 | -0.2384 | -0.2401 | +0.0017 | 0.0013 | 57.4% | 27/20 |
| 14 | 2.49 | 2.43 | 43 | -0.5746 | -0.5769 | +0.0023 | 0.0013 | 65.1% | 28/15 |
| 15 | 2.43 | 2.38 | 43 | -0.0412 | -0.0449 | +0.0037 | 0.0021 | 48.8% | 21/22 |
| 16 | 2.38 | 2.33 | 44 | -0.2759 | -0.2768 | +0.0009 | 0.0018 | 54.5% | 24/20 |
| 17 | 2.32 | 2.28 | 47 | -0.1468 | -0.1511 | +0.0043 | 0.0017 | 51.1% | 24/23 |
| 18 | 2.28 | 2.24 | 48 | -0.2906 | -0.2946 | +0.0039 | 0.0014 | 70.8% | 34/14 |
| 19 | 2.24 | 2.20 | 44 | -0.1830 | -0.1847 | +0.0017 | 0.0012 | 50.0% | 22/22 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (0, 4, 12) | 4.33 | 80210.9 | 828.8 | 96.8 | 345.9 | 347.3 | 2.742 | 2.858 | -0.116 |
| (17, 5, 0) | 4.08 | 57697.2 | 1527.4 | 37.8 | 292.9 | 294.0 | 3.184 | 3.299 | -0.116 |
| (-19, 1, 10) | 3.58 | 56227.7 | 921.8 | 61.0 | 285.2 | 286.3 | 2.527 | 2.640 | -0.113 |
| (7, 7, 11) | 3.52 | 16059.1 | 462.6 | 34.7 | 160.1 | 159.1 | 1.701 | 1.602 | +0.098 |
| (-12, 8, 6) | 3.51 | 49710.6 | 849.8 | 58.5 | 270.8 | 270.1 | 3.544 | 3.449 | +0.095 |
| (10, 4, 2) | 5.98 | 68525.0 | 827.0 | 82.9 | 319.9 | 321.1 | 2.493 | 2.585 | -0.092 |
| (-12, 4, 8) | 4.46 | 74425.5 | 2057.2 | 36.2 | 334.0 | 335.2 | 2.449 | 2.539 | -0.091 |
| (9, 5, 4) | 5.27 | 130392.0 | 2878.0 | 45.3 | 439.7 | 440.6 | 4.753 | 4.841 | -0.088 |
| (-12, 10, 2) | 3.14 | 17847.4 | 744.4 | 24.0 | 168.8 | 168.2 | 2.566 | 2.482 | +0.084 |
| (2, 10, 4) | 3.42 | 30373.2 | 681.0 | 44.6 | 216.5 | 217.2 | 2.980 | 3.061 | -0.080 |
| (11, 5, 4) | 4.90 | 39956.7 | 804.2 | 49.7 | 250.9 | 251.8 | 2.303 | 2.381 | -0.078 |
| (-8, 8, 6) | 3.78 | 25365.8 | 1180.4 | 21.5 | 208.7 | 209.4 | 2.146 | 2.214 | -0.068 |
| (2, 2, 14) | 4.11 | 118570.2 | 2282.2 | 52.0 | 428.4 | 428.9 | 7.395 | 7.459 | -0.064 |
| (-9, 5, 4) | 5.30 | 74702.7 | 2093.4 | 35.7 | 334.8 | 335.5 | 3.042 | 3.102 | -0.059 |
| (-8, 8, 4) | 3.94 | 47152.2 | 659.0 | 71.6 | 260.6 | 261.3 | 2.411 | 2.467 | -0.056 |
| (-21, 3, 2) | 3.80 | 15630.3 | 1258.2 | 12.4 | 151.3 | 152.4 | 0.421 | 0.473 | -0.053 |
| (19, 1, 10) | 3.53 | 15221.7 | 332.0 | 45.8 | 150.4 | 151.0 | 1.108 | 1.160 | -0.052 |
| (-17, 3, 14) | 3.14 | 2730.7 | 121.2 | 22.5 | 64.1 | 65.0 | -0.237 | -0.185 | -0.051 |
| (-19, 1, 8) | 3.83 | 16452.7 | 494.0 | 33.3 | 154.0 | 154.9 | 0.825 | 0.876 | -0.051 |
| (32, 6, 2) | 2.41 | 13874.4 | 401.8 | 34.5 | 131.1 | 130.7 | 1.388 | 1.337 | +0.051 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `drifted_f` | 6.37 | 0.9988 | 199.182903763072 | 827.8462302632518 | 1.000 |
| `drifted_i` | 6.47 | 0.9988 | 199.18270108859792 | 830.6985045818757 | 1.000 |