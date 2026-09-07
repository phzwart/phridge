# Held-Out Log-Likelihood Comparison: `rerefined_deposited_f` vs `deposited_6czg`

## 1. Executive Summary

- **Scored Test Reflections (|T|)**: 900 (held-out)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `0.5269` nats/refl
- **Model B NLL**: `0.5155` nats/refl
- **Difference (Gain $\Delta$)**: `+0.0114` nats/refl (`-0.0114` nats NLL reduction)
- **Estimated Log Bayes Factor**: `+10.27` nats
- **Uncertainty**: Bootstrap SE = `0.0184` (95% CI: `[-0.0273, +0.0455]`) | Naive SE = `0.0112` (Ratio: `1.65`x)
- **Win Fraction $P(d_h > 0)$**: `51.0%` (459 wins, 441 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `5.71e-01`
- **Wilcoxon Signed-Rank $p$-value**: `7.77e-01`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `-0.0046` | `-0.0068` | `-0.0114` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `-0.0170` | `+0.0056` | `-0.0114` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`deposited_6czg`) | Model B (`rerefined_deposited_f`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0014` | `-0.0057` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `0.5231` | `0.4771` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `+0.0038` | `+0.0383` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -0.8390 | -0.6523 | -0.1867 | 0.0491 | 14.0% | 6/37 |
| 1 | 6.10 | 4.86 | 46 | 0.4423 | 0.5556 | -0.1133 | 0.0559 | 28.3% | 13/33 |
| 2 | 4.83 | 4.21 | 45 | 0.0393 | 0.1317 | -0.0924 | 0.0487 | 15.6% | 7/38 |
| 3 | 4.20 | 3.81 | 46 | 0.4607 | 0.4899 | -0.0292 | 0.0244 | 32.6% | 15/31 |
| 4 | 3.80 | 3.53 | 46 | 0.7685 | 0.7420 | +0.0265 | 0.0680 | 26.1% | 12/34 |
| 5 | 3.53 | 3.32 | 46 | 0.9284 | 0.9457 | -0.0174 | 0.0457 | 32.6% | 15/31 |
| 6 | 3.31 | 3.16 | 44 | 0.5496 | 0.4563 | +0.0932 | 0.0697 | 54.5% | 24/20 |
| 7 | 3.14 | 3.02 | 45 | 1.4329 | 1.2442 | +0.1887 | 0.0625 | 71.1% | 32/13 |
| 8 | 3.01 | 2.89 | 46 | 0.4290 | 0.3849 | +0.0441 | 0.0487 | 47.8% | 22/24 |
| 9 | 2.88 | 2.79 | 43 | 0.6684 | 0.6415 | +0.0269 | 0.0355 | 67.4% | 29/14 |
| 10 | 2.79 | 2.70 | 45 | 0.2783 | 0.2172 | +0.0612 | 0.0488 | 55.6% | 25/20 |
| 11 | 2.70 | 2.62 | 45 | 0.5405 | 0.5425 | -0.0020 | 0.0250 | 55.6% | 25/20 |
| 12 | 2.62 | 2.55 | 44 | 0.4655 | 0.3504 | +0.1152 | 0.0634 | 65.9% | 29/15 |
| 13 | 2.55 | 2.49 | 47 | 0.5095 | 0.5131 | -0.0036 | 0.0249 | 57.4% | 27/20 |
| 14 | 2.49 | 2.43 | 43 | 0.2470 | 0.2441 | +0.0028 | 0.0375 | 58.1% | 25/18 |
| 15 | 2.43 | 2.38 | 43 | 0.7808 | 0.7122 | +0.0686 | 0.0274 | 81.4% | 35/8 |
| 16 | 2.38 | 2.33 | 44 | 0.7066 | 0.7995 | -0.0929 | 0.0520 | 54.5% | 24/20 |
| 17 | 2.32 | 2.28 | 47 | 0.9569 | 0.9155 | +0.0414 | 0.0444 | 66.0% | 31/16 |
| 18 | 2.28 | 2.24 | 48 | 0.4733 | 0.4109 | +0.0624 | 0.0489 | 70.8% | 34/14 |
| 19 | 2.24 | 2.20 | 44 | 0.6249 | 0.5948 | +0.0301 | 0.0360 | 65.9% | 29/15 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (-19, 7, 4) | 3.27 | 65816.4 | 797.1 | 82.6 | 179.2 | 205.3 | 6.660 | 4.198 | +2.462 |
| (28, 6, 6) | 2.59 | 1729.0 | 46.5 | 37.2 | 100.1 | 78.7 | 3.555 | 1.547 | +2.008 |
| (-21, 3, 2) | 3.80 | 50411.9 | 629.1 | 80.1 | 123.8 | 138.8 | 6.345 | 4.499 | +1.846 |
| (-19, 9, 8) | 2.75 | 14480.0 | 198.7 | 72.9 | 51.1 | 64.9 | 6.495 | 4.815 | +1.680 |
| (28, 0, 0) | 3.03 | 82866.5 | 1460.6 | 56.7 | 175.4 | 197.8 | 9.200 | 7.525 | +1.674 |
| (-8, 12, 14) | 2.36 | 338.5 | 29.4 | 11.5 | 48.2 | 58.6 | 1.189 | 2.788 | -1.599 |
| (14, 2, 4) | 5.33 | 126810.0 | 1524.0 | 83.2 | 436.4 | 508.0 | 1.948 | 3.526 | -1.578 |
| (37, 1, 0) | 2.29 | 6425.9 | 107.4 | 59.8 | 55.9 | 43.8 | 3.383 | 4.938 | -1.556 |
| (24, 0, 10) | 3.02 | 5069.1 | 134.8 | 37.6 | 182.5 | 160.9 | 4.100 | 2.612 | +1.488 |
| (14, 10, 12) | 2.59 | 140.1 | 27.1 | 5.2 | 62.8 | 48.3 | 2.200 | 0.806 | +1.393 |
| (-7, 7, 2) | 4.61 | 66075.7 | 781.0 | 84.6 | 175.6 | 196.6 | 3.794 | 2.489 | +1.305 |
| (12, 10, 2) | 3.14 | 37671.5 | 459.8 | 81.9 | 156.3 | 177.9 | 3.868 | 2.694 | +1.174 |
| (-23, 3, 8) | 3.20 | 22889.6 | 300.5 | 76.2 | 100.9 | 121.5 | 3.029 | 1.857 | +1.172 |
| (-10, 0, 12) | 4.31 | 4234.8 | 122.6 | 34.5 | 213.2 | 207.0 | 3.337 | 2.167 | +1.170 |
| (-23, 1, 4) | 3.58 | 21340.5 | 304.5 | 70.1 | 91.8 | 119.6 | 2.374 | 1.230 | +1.143 |
| (20, 8, 0) | 3.06 | 77200.7 | 929.6 | 83.0 | 185.5 | 194.3 | 11.719 | 10.597 | +1.122 |
| (-13, 1, 8) | 4.90 | 159005.0 | 1880.0 | 84.6 | 534.8 | 579.8 | 3.701 | 4.802 | -1.101 |
| (2, 14, 12) | 2.24 | 160.0 | 25.0 | 6.4 | 33.2 | 18.7 | 0.519 | -0.554 | +1.073 |
| (4, 12, 13) | 2.46 | 6027.1 | 91.4 | 65.9 | 53.1 | 43.7 | 2.319 | 3.364 | -1.044 |
| (10, 0, 0) | 8.49 | 179683.0 | 2994.0 | 60.0 | 312.2 | 354.9 | 3.386 | 2.372 | +1.014 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `deposited_6czg` | 42.89 | 0.8845 | 4.829139997587939 | 6.78336178322578 | 1.000 |
| `rerefined_deposited_f` | 42.98 | 0.8875 | 5.035406244011455 | 6.798548242855606 | 1.000 |