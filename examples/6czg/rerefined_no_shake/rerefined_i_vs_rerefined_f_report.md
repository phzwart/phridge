# Held-Out Log-Likelihood Comparison: `rerefined_i` vs `rerefined_f`

## 1. Executive Summary

- **Scored Test Reflections (|T|)**: 900 (held-out)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `0.5159` nats/refl
- **Model B NLL**: `0.5156` nats/refl
- **Difference (Gain $\Delta$)**: `+0.0003` nats/refl (`-0.0003` nats NLL reduction)
- **Estimated Log Bayes Factor**: `+0.26` nats
- **Uncertainty**: Bootstrap SE = `0.0003` (95% CI: `[-0.0003, +0.0008]`) | Naive SE = `0.0003` (Ratio: `1.14`x)
- **Win Fraction $P(d_h > 0)$**: `55.0%` (495 wins, 405 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `2.99e-03`
- **Wilcoxon Signed-Rank $p$-value**: `9.83e-02`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `-0.0003` | `+0.0000` | `-0.0003` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `-0.0003` | `+0.0000` | `-0.0003` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`rerefined_f`) | Model B (`rerefined_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0069` | `-0.0069` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `0.5174` | `0.5173` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.0015` | `-0.0017` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -0.7294 | -0.7315 | +0.0021 | 0.0014 | 51.2% | 22/21 |
| 1 | 6.10 | 4.86 | 46 | 0.4784 | 0.4823 | -0.0039 | 0.0018 | 47.8% | 22/24 |
| 2 | 4.83 | 4.21 | 45 | 0.0971 | 0.0954 | +0.0017 | 0.0014 | 46.7% | 21/24 |
| 3 | 4.20 | 3.81 | 46 | 0.4451 | 0.4430 | +0.0022 | 0.0009 | 71.7% | 33/13 |
| 4 | 3.80 | 3.53 | 46 | 0.7284 | 0.7284 | +0.0000 | 0.0012 | 54.3% | 25/21 |
| 5 | 3.53 | 3.32 | 46 | 0.9699 | 0.9680 | +0.0019 | 0.0026 | 54.3% | 25/21 |
| 6 | 3.31 | 3.16 | 44 | 0.4882 | 0.4875 | +0.0007 | 0.0016 | 54.5% | 24/20 |
| 7 | 3.14 | 3.02 | 45 | 1.3287 | 1.3288 | -0.0001 | 0.0022 | 48.9% | 22/23 |
| 8 | 3.01 | 2.89 | 46 | 0.3663 | 0.3650 | +0.0013 | 0.0007 | 60.9% | 28/18 |
| 9 | 2.88 | 2.79 | 43 | 0.6730 | 0.6727 | +0.0003 | 0.0004 | 55.8% | 24/19 |
| 10 | 2.79 | 2.70 | 45 | 0.2513 | 0.2516 | -0.0003 | 0.0006 | 37.8% | 17/28 |
| 11 | 2.70 | 2.62 | 45 | 0.5400 | 0.5407 | -0.0006 | 0.0005 | 64.4% | 29/16 |
| 12 | 2.62 | 2.55 | 44 | 0.4329 | 0.4324 | +0.0006 | 0.0008 | 63.6% | 28/16 |
| 13 | 2.55 | 2.49 | 47 | 0.5331 | 0.5337 | -0.0006 | 0.0004 | 48.9% | 23/24 |
| 14 | 2.49 | 2.43 | 43 | 0.2249 | 0.2248 | +0.0000 | 0.0005 | 60.5% | 26/17 |
| 15 | 2.43 | 2.38 | 43 | 0.7589 | 0.7584 | +0.0005 | 0.0004 | 65.1% | 28/15 |
| 16 | 2.38 | 2.33 | 44 | 0.6760 | 0.6764 | -0.0003 | 0.0006 | 56.8% | 25/19 |
| 17 | 2.32 | 2.28 | 47 | 0.9238 | 0.9235 | +0.0003 | 0.0004 | 61.7% | 29/18 |
| 18 | 2.28 | 2.24 | 48 | 0.4717 | 0.4717 | +0.0000 | 0.0004 | 60.4% | 29/19 |
| 19 | 2.24 | 2.20 | 44 | 0.5867 | 0.5866 | +0.0001 | 0.0004 | 34.1% | 15/29 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (20, 6, 0) | 3.44 | 65812.5 | 807.1 | 81.5 | 209.2 | 210.5 | 4.769 | 4.681 | +0.089 |
| (-11, 9, 4) | 3.41 | 35272.8 | 436.3 | 80.8 | 122.0 | 121.4 | 4.936 | 4.984 | -0.048 |
| (28, 0, 0) | 3.03 | 82866.5 | 1460.6 | 56.7 | 187.4 | 186.8 | 8.338 | 8.382 | -0.044 |
| (24, 0, 10) | 3.02 | 5069.1 | 134.8 | 37.6 | 176.3 | 175.7 | 3.566 | 3.525 | +0.041 |
| (8, 0, 9) | 5.58 | 141.6 | 32.4 | 4.4 | 91.1 | 93.1 | -1.260 | -1.219 | -0.041 |
| (-13, 1, 8) | 4.90 | 159005.0 | 1880.0 | 84.6 | 563.4 | 563.8 | 4.478 | 4.518 | -0.040 |
| (-21, 3, 2) | 3.80 | 50411.9 | 629.1 | 80.1 | 131.7 | 132.2 | 5.251 | 5.215 | +0.036 |
| (17, 3, 14) | 3.10 | 2133.4 | 53.4 | 40.0 | 97.2 | 96.4 | 1.332 | 1.298 | +0.034 |
| (-24, 2, 2) | 3.45 | 60731.8 | 770.1 | 78.9 | 233.5 | 234.2 | 2.753 | 2.721 | +0.032 |
| (12, 10, 2) | 3.14 | 37671.5 | 459.8 | 81.9 | 165.9 | 166.5 | 3.282 | 3.250 | +0.031 |
| (20, 8, 0) | 3.06 | 77200.7 | 929.6 | 83.0 | 190.3 | 190.1 | 11.186 | 11.218 | -0.031 |
| (-3, 3, 13) | 4.23 | 13572.3 | 197.8 | 68.6 | 192.1 | 191.3 | 1.612 | 1.581 | +0.031 |
| (-21, 5, 6) | 3.32 | 20106.8 | 269.3 | 74.7 | 86.3 | 86.8 | 2.810 | 2.780 | +0.030 |
| (-9, 5, 4) | 5.30 | 88155.3 | 1046.7 | 84.2 | 280.5 | 279.2 | 2.231 | 2.261 | -0.029 |
| (-3, 5, 11) | 4.26 | 176.2 | 26.8 | 6.6 | 55.4 | 54.4 | -0.906 | -0.935 | +0.029 |
| (14, 8, 7) | 3.28 | 334.2 | 30.2 | 11.1 | 45.9 | 45.1 | -0.665 | -0.693 | +0.028 |
| (-19, 7, 4) | 3.27 | 65816.4 | 797.1 | 82.6 | 188.8 | 189.1 | 5.582 | 5.554 | +0.028 |
| (10, 0, 0) | 8.49 | 179683.0 | 2994.0 | 60.0 | 332.9 | 334.3 | 2.716 | 2.688 | +0.028 |
| (26, 4, 2) | 3.04 | 1242.4 | 44.3 | 28.0 | 94.4 | 94.8 | 1.739 | 1.766 | -0.027 |
| (-23, 3, 8) | 3.20 | 22889.6 | 300.5 | 76.2 | 104.6 | 105.1 | 2.734 | 2.708 | +0.026 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `rerefined_f` | 42.76 | 0.8855 | 5.10113697321753 | 7.104841332042472 | 1.000 |
| `rerefined_i` | 42.74 | 0.8858 | 5.101871723204086 | 7.102378100414612 | 1.000 |