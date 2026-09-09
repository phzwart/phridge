# Held-Out Log-Likelihood Comparison: `rerefined_f` vs `deposited_6czg`

## 1. Executive Summary

- **Scored Audit Reflections (|A|)**: 900 (audit set)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `0.5269` nats/refl
- **Model B NLL**: `0.5159` nats/refl
- **Difference (Gain $\Delta$)**: `+0.0110` nats/refl (`-0.0110` nats NLL reduction)
- **Estimated Log Bayes Factor**: `+9.87` nats
- **Uncertainty**: Bootstrap SE = `0.0108` (95% CI: `[-0.0115, +0.0311]`) | Naive SE = `0.0080` (Ratio: `1.35`x)
- **Win Fraction $P(d_h > 0)$**: `51.4%` (463 wins, 437 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `4.05e-01`
- **Wilcoxon Signed-Rank $p$-value**: `5.82e-01`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `-0.0073` | `-0.0037` | `-0.0110` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `-0.0116` | `+0.0006` | `-0.0110` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`deposited_6czg`) | Model B (`rerefined_f`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0014` | `-0.0069` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `0.5231` | `0.5174` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `+0.0038` | `-0.0015` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -0.8390 | -0.7294 | -0.1096 | 0.0330 | 14.0% | 6/37 |
| 1 | 6.10 | 4.86 | 46 | 0.4423 | 0.4784 | -0.0361 | 0.0422 | 32.6% | 15/31 |
| 2 | 4.83 | 4.21 | 45 | 0.0393 | 0.0971 | -0.0578 | 0.0295 | 17.8% | 8/37 |
| 3 | 4.20 | 3.81 | 46 | 0.4607 | 0.4451 | +0.0156 | 0.0152 | 41.3% | 19/27 |
| 4 | 3.80 | 3.53 | 46 | 0.7685 | 0.7284 | +0.0401 | 0.0494 | 28.3% | 13/33 |
| 5 | 3.53 | 3.32 | 46 | 0.9284 | 0.9699 | -0.0415 | 0.0298 | 34.8% | 16/30 |
| 6 | 3.31 | 3.16 | 44 | 0.5496 | 0.4882 | +0.0614 | 0.0361 | 52.3% | 23/21 |
| 7 | 3.14 | 3.02 | 45 | 1.4329 | 1.3287 | +0.1042 | 0.0431 | 64.4% | 29/16 |
| 8 | 3.01 | 2.89 | 46 | 0.4290 | 0.3663 | +0.0628 | 0.0377 | 43.5% | 20/26 |
| 9 | 2.88 | 2.79 | 43 | 0.6684 | 0.6730 | -0.0046 | 0.0298 | 72.1% | 31/12 |
| 10 | 2.79 | 2.70 | 45 | 0.2783 | 0.2513 | +0.0270 | 0.0328 | 42.2% | 19/26 |
| 11 | 2.70 | 2.62 | 45 | 0.5405 | 0.5400 | +0.0005 | 0.0347 | 68.9% | 31/14 |
| 12 | 2.62 | 2.55 | 44 | 0.4655 | 0.4329 | +0.0326 | 0.0456 | 61.4% | 27/17 |
| 13 | 2.55 | 2.49 | 47 | 0.5095 | 0.5331 | -0.0236 | 0.0249 | 63.8% | 30/17 |
| 14 | 2.49 | 2.43 | 43 | 0.2470 | 0.2249 | +0.0221 | 0.0258 | 60.5% | 26/17 |
| 15 | 2.43 | 2.38 | 43 | 0.7808 | 0.7589 | +0.0219 | 0.0226 | 62.8% | 27/16 |
| 16 | 2.38 | 2.33 | 44 | 0.7066 | 0.6760 | +0.0306 | 0.0457 | 70.5% | 31/13 |
| 17 | 2.32 | 2.28 | 47 | 0.9569 | 0.9238 | +0.0331 | 0.0295 | 59.6% | 28/19 |
| 18 | 2.28 | 2.24 | 48 | 0.4733 | 0.4717 | +0.0017 | 0.0401 | 64.6% | 31/17 |
| 19 | 2.24 | 2.20 | 44 | 0.6249 | 0.5867 | +0.0382 | 0.0244 | 75.0% | 33/11 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (-21, 3, 2) | 3.80 | 50411.9 | 629.1 | 80.1 | 123.8 | 131.7 | 6.345 | 5.251 | +1.094 |
| (-19, 7, 4) | 3.27 | 65816.4 | 797.1 | 82.6 | 179.2 | 188.8 | 6.660 | 5.582 | +1.078 |
| (17, 9, 12) | 2.61 | 6081.7 | 94.3 | 64.5 | 58.9 | 43.7 | 1.411 | 2.442 | -1.031 |
| (-8, 8, 6) | 3.78 | 48404.5 | 590.2 | 82.0 | 164.6 | 185.7 | 3.003 | 1.993 | +1.010 |
| (2, 6, 0) | 5.83 | 973.3 | 27.6 | 35.3 | 140.9 | 128.5 | 2.529 | 1.522 | +1.007 |
| (30, 4, 2) | 2.68 | 7198.0 | 118.0 | 61.0 | 57.6 | 42.7 | 2.144 | 3.119 | -0.975 |
| (16, 14, 2) | 2.27 | 8267.3 | 169.8 | 48.7 | 75.6 | 68.0 | 2.844 | 3.817 | -0.972 |
| (-23, 1, 4) | 3.58 | 21340.5 | 304.5 | 70.1 | 91.8 | 112.0 | 2.374 | 1.448 | +0.926 |
| (-19, 9, 8) | 2.75 | 14480.0 | 198.7 | 72.9 | 51.1 | 58.1 | 6.495 | 5.630 | +0.864 |
| (28, 0, 0) | 3.03 | 82866.5 | 1460.6 | 56.7 | 175.4 | 187.4 | 9.200 | 8.338 | +0.862 |
| (11, 7, 14) | 2.98 | 2502.8 | 55.5 | 45.1 | 114.9 | 104.5 | 2.664 | 1.803 | +0.861 |
| (-3, 5, 24) | 2.34 | 425.9 | 41.5 | 10.3 | 37.3 | 47.3 | 0.299 | 1.159 | -0.860 |
| (27, 7, 12) | 2.34 | 5812.1 | 99.1 | 58.6 | 111.8 | 98.7 | 2.244 | 1.421 | +0.823 |
| (14, 10, 12) | 2.59 | 140.1 | 27.1 | 5.2 | 62.8 | 54.7 | 2.200 | 1.395 | +0.805 |
| (17, 13, 6) | 2.32 | 6425.1 | 97.8 | 65.7 | 51.2 | 60.2 | 3.650 | 2.872 | +0.778 |
| (-13, 1, 8) | 4.90 | 159005.0 | 1880.0 | 84.6 | 534.8 | 563.4 | 3.701 | 4.478 | -0.777 |
| (14, 2, 4) | 5.33 | 126810.0 | 1524.0 | 83.2 | 436.4 | 476.3 | 1.948 | 2.722 | -0.774 |
| (-16, 12, 11) | 2.33 | 2270.1 | 47.4 | 47.9 | 9.4 | 18.8 | 3.138 | 2.370 | +0.768 |
| (-25, 5, 6) | 2.94 | 1091.6 | 39.7 | 27.5 | 77.7 | 64.1 | 1.090 | 0.323 | +0.768 |
| (30, 4, 4) | 2.65 | 37.6 | 31.3 | 1.2 | 43.0 | 53.0 | 0.281 | 1.046 | -0.765 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `deposited_6czg` | 42.89 | 0.8845 | 4.829139997587939 | 6.78336178322578 | 1.000 |
| `rerefined_f` | 42.76 | 0.8855 | 5.10113697321753 | 7.104841332042472 | 1.000 |