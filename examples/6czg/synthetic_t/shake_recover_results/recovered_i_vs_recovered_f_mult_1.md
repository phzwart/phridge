# Held-Out Log-Likelihood Comparison: `recovered_i` vs `recovered_f`

## 1. Executive Summary

- **Scored Test Reflections (|T|)**: 900 (held-out)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `0.0568` nats/refl
- **Model B NLL**: `0.0568` nats/refl
- **Difference (Gain $\Delta$)**: `-0.0000` nats/refl (`+0.0000` nats NLL reduction)
- **Estimated Log Bayes Factor**: `-0.02` nats
- **Uncertainty**: Bootstrap SE = `0.0001` (95% CI: `[-0.0002, +0.0001]`) | Naive SE = `0.0001` (Ratio: `1.00`x)
- **Win Fraction $P(d_h > 0)$**: `49.7%` (447 wins, 453 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `8.68e-01`
- **Wilcoxon Signed-Rank $p$-value**: `9.75e-01`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.0000` | `+0.0000` | `+0.0000` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.0000` | `+0.0000` | `+0.0000` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`recovered_f`) | Model B (`recovered_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0169` | `-0.0169` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `0.0635` | `0.0636` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.0068` | `-0.0069` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -1.3429 | -1.3422 | -0.0007 | 0.0002 | 25.6% | 11/32 |
| 1 | 6.10 | 4.86 | 46 | 0.0894 | 0.0899 | -0.0006 | 0.0004 | 32.6% | 15/31 |
| 2 | 4.83 | 4.21 | 45 | -0.3724 | -0.3721 | -0.0003 | 0.0003 | 28.9% | 13/32 |
| 3 | 4.20 | 3.81 | 46 | 0.0441 | 0.0444 | -0.0003 | 0.0003 | 32.6% | 15/31 |
| 4 | 3.80 | 3.53 | 46 | 0.2441 | 0.2443 | -0.0003 | 0.0005 | 52.2% | 24/22 |
| 5 | 3.53 | 3.32 | 46 | 0.3241 | 0.3240 | +0.0001 | 0.0003 | 45.7% | 21/25 |
| 6 | 3.31 | 3.16 | 44 | -0.0336 | -0.0339 | +0.0003 | 0.0003 | 45.5% | 20/24 |
| 7 | 3.14 | 3.02 | 45 | 0.4455 | 0.4456 | -0.0001 | 0.0003 | 40.0% | 18/27 |
| 8 | 3.01 | 2.89 | 46 | -0.0506 | -0.0507 | +0.0001 | 0.0002 | 58.7% | 27/19 |
| 9 | 2.88 | 2.79 | 43 | 0.1094 | 0.1093 | +0.0001 | 0.0004 | 51.2% | 22/21 |
| 10 | 2.79 | 2.70 | 45 | -0.2713 | -0.2713 | -0.0001 | 0.0002 | 62.2% | 28/17 |
| 11 | 2.70 | 2.62 | 45 | 0.0827 | 0.0827 | +0.0000 | 0.0002 | 46.7% | 21/24 |
| 12 | 2.62 | 2.55 | 44 | -0.0990 | -0.0992 | +0.0002 | 0.0002 | 63.6% | 28/16 |
| 13 | 2.55 | 2.49 | 47 | 0.1611 | 0.1606 | +0.0005 | 0.0002 | 59.6% | 28/19 |
| 14 | 2.49 | 2.43 | 43 | -0.0371 | -0.0374 | +0.0003 | 0.0003 | 69.8% | 30/13 |
| 15 | 2.43 | 2.38 | 43 | 0.4457 | 0.4452 | +0.0005 | 0.0003 | 58.1% | 25/18 |
| 16 | 2.38 | 2.33 | 44 | 0.2388 | 0.2390 | -0.0002 | 0.0003 | 61.4% | 27/17 |
| 17 | 2.32 | 2.28 | 47 | 0.3323 | 0.3321 | +0.0002 | 0.0003 | 51.1% | 24/23 |
| 18 | 2.28 | 2.24 | 48 | 0.3568 | 0.3570 | -0.0002 | 0.0004 | 52.1% | 25/23 |
| 19 | 2.24 | 2.20 | 44 | 0.3821 | 0.3823 | -0.0002 | 0.0004 | 56.8% | 25/19 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (21, 3, 2) | 3.79 | 56287.7 | 956.4 | 58.9 | 306.0 | 306.1 | 3.743 | 3.754 | -0.012 |
| (2, 2, 14) | 4.11 | 122539.2 | 1141.1 | 107.4 | 437.9 | 438.0 | 6.254 | 6.264 | -0.010 |
| (-8, 8, 6) | 3.78 | 27261.2 | 590.2 | 46.2 | 209.9 | 210.1 | 1.586 | 1.596 | -0.010 |
| (32, 6, 2) | 2.41 | 13761.9 | 200.9 | 68.5 | 155.4 | 155.3 | 3.746 | 3.736 | +0.010 |
| (-12, 4, 8) | 4.46 | 74634.5 | 1028.6 | 72.6 | 331.6 | 331.5 | 2.298 | 2.288 | +0.009 |
| (-13, 15, 1) | 2.21 | 123.0 | 28.3 | 4.3 | 34.8 | 34.8 | 1.662 | 1.670 | -0.009 |
| (-13, 11, 2) | 2.87 | 1897.8 | 55.9 | 33.9 | 74.7 | 74.6 | 1.295 | 1.287 | +0.009 |
| (19, 1, 8) | 3.78 | 40912.8 | 518.4 | 78.9 | 242.0 | 241.8 | 1.364 | 1.355 | +0.008 |
| (8, 2, 18) | 3.10 | 2749.6 | 106.7 | 25.8 | 89.6 | 89.5 | 1.620 | 1.612 | +0.008 |
| (17, 5, 22) | 2.24 | 1530.6 | 37.0 | 41.4 | 27.9 | 27.8 | 0.874 | 0.881 | -0.008 |
| (-29, 9, 1) | 2.35 | 640.3 | 35.1 | 18.2 | 40.6 | 40.7 | 0.432 | 0.440 | -0.008 |
| (14, 2, 4) | 5.33 | 191022.5 | 1524.0 | 125.3 | 509.1 | 509.1 | 3.638 | 3.631 | +0.008 |
| (-12, 8, 22) | 2.20 | 42.8 | 35.9 | 1.2 | 24.8 | 24.8 | 0.459 | 0.467 | -0.007 |
| (-6, 6, 15) | 3.21 | 1453.9 | 41.1 | 35.4 | 68.6 | 68.5 | 0.261 | 0.254 | +0.007 |
| (11, 5, 4) | 4.90 | 41410.7 | 402.1 | 103.0 | 243.0 | 243.1 | 1.339 | 1.346 | -0.007 |
| (-14, 4, 2) | 4.94 | 126721.0 | 1288.0 | 98.4 | 401.1 | 401.2 | 1.847 | 1.854 | -0.007 |
| (17, 11, 12) | 2.36 | 2897.1 | 50.8 | 57.0 | 79.3 | 79.4 | 2.056 | 2.063 | -0.007 |
| (9, 13, 10) | 2.39 | 1804.3 | 37.7 | 47.9 | 67.1 | 67.1 | 1.712 | 1.719 | -0.007 |
| (-9, 5, 4) | 5.30 | 73520.8 | 1046.7 | 70.2 | 319.4 | 319.5 | 1.875 | 1.882 | -0.007 |
| (-13, 1, 8) | 4.90 | 286007.4 | 1880.0 | 152.1 | 620.7 | 620.6 | 4.945 | 4.939 | +0.007 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `recovered_f` | 15.85 | 0.9914 | 199.09691978653197 | 1000.0 | 1.000 |
| `recovered_i` | 15.85 | 0.9914 | 199.09692149524685 | 1000.0 | 1.000 |