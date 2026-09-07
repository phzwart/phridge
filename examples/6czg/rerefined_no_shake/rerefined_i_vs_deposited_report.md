# Held-Out Log-Likelihood Comparison: `rerefined_i` vs `deposited_6czg`

## 1. Executive Summary

- **Scored Test Reflections (|T|)**: 900 (held-out)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `0.5269` nats/refl
- **Model B NLL**: `0.5156` nats/refl
- **Difference (Gain $\Delta$)**: `+0.0113` nats/refl (`-0.0113` nats NLL reduction)
- **Estimated Log Bayes Factor**: `+10.13` nats
- **Uncertainty**: Bootstrap SE = `0.0107` (95% CI: `[-0.0110, +0.0315]`) | Naive SE = `0.0080` (Ratio: `1.34`x)
- **Win Fraction $P(d_h > 0)$**: `51.0%` (459 wins, 441 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `5.71e-01`
- **Wilcoxon Signed-Rank $p$-value**: `5.48e-01`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `-0.0076` | `-0.0036` | `-0.0113` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `-0.0118` | `+0.0006` | `-0.0113` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`deposited_6czg`) | Model B (`rerefined_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0014` | `-0.0069` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `0.5231` | `0.5173` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `+0.0038` | `-0.0017` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -0.8390 | -0.7315 | -0.1075 | 0.0334 | 14.0% | 6/37 |
| 1 | 6.10 | 4.86 | 46 | 0.4423 | 0.4823 | -0.0400 | 0.0426 | 32.6% | 15/31 |
| 2 | 4.83 | 4.21 | 45 | 0.0393 | 0.0954 | -0.0561 | 0.0297 | 17.8% | 8/37 |
| 3 | 4.20 | 3.81 | 46 | 0.4607 | 0.4430 | +0.0177 | 0.0154 | 39.1% | 18/28 |
| 4 | 3.80 | 3.53 | 46 | 0.7685 | 0.7284 | +0.0401 | 0.0499 | 28.3% | 13/33 |
| 5 | 3.53 | 3.32 | 46 | 0.9284 | 0.9680 | -0.0396 | 0.0311 | 34.8% | 16/30 |
| 6 | 3.31 | 3.16 | 44 | 0.5496 | 0.4875 | +0.0621 | 0.0370 | 52.3% | 23/21 |
| 7 | 3.14 | 3.02 | 45 | 1.4329 | 1.3288 | +0.1041 | 0.0436 | 64.4% | 29/16 |
| 8 | 3.01 | 2.89 | 46 | 0.4290 | 0.3650 | +0.0641 | 0.0379 | 43.5% | 20/26 |
| 9 | 2.88 | 2.79 | 43 | 0.6684 | 0.6727 | -0.0043 | 0.0299 | 72.1% | 31/12 |
| 10 | 2.79 | 2.70 | 45 | 0.2783 | 0.2516 | +0.0267 | 0.0331 | 40.0% | 18/27 |
| 11 | 2.70 | 2.62 | 45 | 0.5405 | 0.5407 | -0.0002 | 0.0351 | 68.9% | 31/14 |
| 12 | 2.62 | 2.55 | 44 | 0.4655 | 0.4324 | +0.0332 | 0.0461 | 61.4% | 27/17 |
| 13 | 2.55 | 2.49 | 47 | 0.5095 | 0.5337 | -0.0242 | 0.0250 | 63.8% | 30/17 |
| 14 | 2.49 | 2.43 | 43 | 0.2470 | 0.2248 | +0.0222 | 0.0261 | 60.5% | 26/17 |
| 15 | 2.43 | 2.38 | 43 | 0.7808 | 0.7584 | +0.0224 | 0.0227 | 62.8% | 27/16 |
| 16 | 2.38 | 2.33 | 44 | 0.7066 | 0.6764 | +0.0302 | 0.0459 | 70.5% | 31/13 |
| 17 | 2.32 | 2.28 | 47 | 0.9569 | 0.9235 | +0.0335 | 0.0297 | 57.4% | 27/20 |
| 18 | 2.28 | 2.24 | 48 | 0.4733 | 0.4717 | +0.0017 | 0.0402 | 62.5% | 30/18 |
| 19 | 2.24 | 2.20 | 44 | 0.6249 | 0.5866 | +0.0383 | 0.0247 | 75.0% | 33/11 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (-21, 3, 2) | 3.80 | 50411.9 | 629.1 | 80.1 | 123.8 | 132.2 | 6.345 | 5.215 | +1.130 |
| (-19, 7, 4) | 3.27 | 65816.4 | 797.1 | 82.6 | 179.2 | 189.1 | 6.660 | 5.554 | +1.106 |
| (17, 9, 12) | 2.61 | 6081.7 | 94.3 | 64.5 | 58.9 | 43.4 | 1.411 | 2.465 | -1.054 |
| (-8, 8, 6) | 3.78 | 48404.5 | 590.2 | 82.0 | 164.6 | 185.8 | 3.003 | 1.990 | +1.014 |
| (2, 6, 0) | 5.83 | 973.3 | 27.6 | 35.3 | 140.9 | 128.8 | 2.529 | 1.543 | +0.987 |
| (30, 4, 2) | 2.68 | 7198.0 | 118.0 | 61.0 | 57.6 | 42.6 | 2.144 | 3.130 | -0.986 |
| (16, 14, 2) | 2.27 | 8267.3 | 169.8 | 48.7 | 75.6 | 68.0 | 2.844 | 3.815 | -0.971 |
| (-23, 1, 4) | 3.58 | 21340.5 | 304.5 | 70.1 | 91.8 | 111.9 | 2.374 | 1.451 | +0.923 |
| (-19, 9, 8) | 2.75 | 14480.0 | 198.7 | 72.9 | 51.1 | 58.3 | 6.495 | 5.615 | +0.880 |
| (-3, 5, 24) | 2.34 | 425.9 | 41.5 | 10.3 | 37.3 | 47.4 | 0.299 | 1.167 | -0.868 |
| (11, 7, 14) | 2.98 | 2502.8 | 55.5 | 45.1 | 114.9 | 104.5 | 2.664 | 1.802 | +0.862 |
| (27, 7, 12) | 2.34 | 5812.1 | 99.1 | 58.6 | 111.8 | 98.7 | 2.244 | 1.420 | +0.823 |
| (14, 10, 12) | 2.59 | 140.1 | 27.1 | 5.2 | 62.8 | 54.5 | 2.200 | 1.379 | +0.821 |
| (28, 0, 0) | 3.03 | 82866.5 | 1460.6 | 56.7 | 175.4 | 186.8 | 9.200 | 8.382 | +0.818 |
| (-13, 1, 8) | 4.90 | 159005.0 | 1880.0 | 84.6 | 534.8 | 563.8 | 3.701 | 4.518 | -0.817 |
| (14, 2, 4) | 5.33 | 126810.0 | 1524.0 | 83.2 | 436.4 | 476.3 | 1.948 | 2.731 | -0.783 |
| (17, 13, 6) | 2.32 | 6425.1 | 97.8 | 65.7 | 51.2 | 60.3 | 3.650 | 2.867 | +0.783 |
| (30, 4, 4) | 2.65 | 37.6 | 31.3 | 1.2 | 43.0 | 53.1 | 0.281 | 1.060 | -0.779 |
| (21, 3, 2) | 3.79 | 77579.9 | 956.4 | 81.1 | 238.5 | 259.3 | 2.724 | 1.946 | +0.778 |
| (-25, 5, 6) | 2.94 | 1091.6 | 39.7 | 27.5 | 77.7 | 64.0 | 1.090 | 0.316 | +0.775 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `deposited_6czg` | 42.89 | 0.8845 | 4.829139997587939 | 6.78336178322578 | 1.000 |
| `rerefined_i` | 42.74 | 0.8858 | 5.101871723204086 | 7.102378100414612 | 1.000 |