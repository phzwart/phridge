# Held-Out Log-Likelihood Comparison: `recovered_i` vs `recovered_f`

## 1. Executive Summary

- **Scored Audit Reflections (|A|)**: 900 (audit set)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `0.3143` nats/refl
- **Model B NLL**: `0.3142` nats/refl
- **Difference (Gain $\Delta$)**: `+0.0001` nats/refl (`-0.0001` nats NLL reduction)
- **Estimated Log Bayes Factor**: `+0.09` nats
- **Uncertainty**: Bootstrap SE = `0.0016` (95% CI: `[-0.0031, +0.0031]`) | Naive SE = `0.0012` (Ratio: `1.34`x)
- **Win Fraction $P(d_h > 0)$**: `50.0%` (450 wins, 450 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `1.00e+00`
- **Wilcoxon Signed-Rank $p$-value**: `5.28e-01`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `-0.0001` | `-0.0000` | `-0.0001` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `-0.0001` | `-0.0000` | `-0.0001` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`recovered_f`) | Model B (`recovered_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0097` | `-0.0098` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `0.3412` | `0.3396` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.0269` | `-0.0254` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -1.2594 | -1.2593 | -0.0001 | 0.0114 | 39.5% | 17/26 |
| 1 | 6.10 | 4.86 | 46 | 0.0412 | 0.0608 | -0.0196 | 0.0063 | 23.9% | 11/35 |
| 2 | 4.83 | 4.21 | 45 | -0.3526 | -0.3442 | -0.0084 | 0.0058 | 40.0% | 18/27 |
| 3 | 4.20 | 3.81 | 46 | 0.0417 | 0.0396 | +0.0021 | 0.0039 | 43.5% | 20/26 |
| 4 | 3.80 | 3.53 | 46 | 0.3388 | 0.3496 | -0.0109 | 0.0055 | 32.6% | 15/31 |
| 5 | 3.53 | 3.32 | 46 | 0.5481 | 0.5553 | -0.0072 | 0.0069 | 43.5% | 20/26 |
| 6 | 3.31 | 3.16 | 44 | 0.0914 | 0.0887 | +0.0028 | 0.0043 | 54.5% | 24/20 |
| 7 | 3.14 | 3.02 | 45 | 0.7296 | 0.7364 | -0.0068 | 0.0040 | 40.0% | 18/27 |
| 8 | 3.01 | 2.89 | 46 | 0.1582 | 0.1528 | +0.0054 | 0.0025 | 54.3% | 25/21 |
| 9 | 2.88 | 2.79 | 43 | 0.4974 | 0.4901 | +0.0073 | 0.0053 | 65.1% | 28/15 |
| 10 | 2.79 | 2.70 | 45 | 0.0352 | 0.0373 | -0.0021 | 0.0044 | 46.7% | 21/24 |
| 11 | 2.70 | 2.62 | 45 | 0.4575 | 0.4580 | -0.0004 | 0.0051 | 53.3% | 24/21 |
| 12 | 2.62 | 2.55 | 44 | 0.3418 | 0.3388 | +0.0031 | 0.0027 | 56.8% | 25/19 |
| 13 | 2.55 | 2.49 | 47 | 0.4640 | 0.4591 | +0.0048 | 0.0030 | 68.1% | 32/15 |
| 14 | 2.49 | 2.43 | 43 | 0.2781 | 0.2746 | +0.0035 | 0.0042 | 53.5% | 23/20 |
| 15 | 2.43 | 2.38 | 43 | 0.6142 | 0.6016 | +0.0127 | 0.0051 | 69.8% | 30/13 |
| 16 | 2.38 | 2.33 | 44 | 0.6812 | 0.6753 | +0.0059 | 0.0039 | 56.8% | 25/19 |
| 17 | 2.32 | 2.28 | 47 | 0.9164 | 0.9192 | -0.0028 | 0.0030 | 51.1% | 24/23 |
| 18 | 2.28 | 2.24 | 48 | 0.7542 | 0.7473 | +0.0069 | 0.0055 | 52.1% | 25/23 |
| 19 | 2.24 | 2.20 | 44 | 0.8206 | 0.8132 | +0.0074 | 0.0044 | 56.8% | 25/19 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (-1, 1, 3) | 17.00 | -15.7 | 299.0 | -0.1 | 30.2 | 28.4 | -2.100 | -2.332 | +0.232 |
| (0, 4, 12) | 4.33 | 75649.3 | 4144.0 | 18.3 | 342.2 | 344.0 | 3.820 | 4.018 | -0.198 |
| (11, 3, 0) | 6.46 | 53805.3 | 3549.0 | 15.2 | 277.3 | 279.2 | 1.734 | 1.917 | -0.183 |
| (8, 2, 0) | 9.10 | 80729.7 | 3744.0 | 21.6 | 314.9 | 316.6 | 0.859 | 1.035 | -0.175 |
| (14, 2, 4) | 5.33 | 169128.4 | 15240.0 | 11.1 | 498.3 | 500.7 | 4.531 | 4.702 | -0.171 |
| (35, 1, 9) | 2.26 | -181.5 | 357.0 | -0.5 | 40.3 | 39.1 | 2.768 | 2.599 | +0.170 |
| (-5, 3, 4) | 8.14 | 27472.1 | 5651.0 | 4.9 | 241.2 | 242.7 | 3.986 | 4.152 | -0.166 |
| (-18, 4, 10) | 3.43 | 49119.0 | 6433.0 | 7.6 | 280.9 | 283.0 | 3.077 | 3.242 | -0.165 |
| (-23, 3, 2) | 3.50 | 1109.6 | 448.0 | 2.5 | 63.2 | 65.5 | 0.248 | 0.409 | -0.161 |
| (3, 1, 0) | 22.09 | 19917.5 | 1815.0 | 11.0 | 194.5 | 194.1 | 8.069 | 7.915 | +0.155 |
| (-7, 11, 10) | 2.76 | 3264.3 | 279.0 | 11.7 | 98.4 | 99.5 | 2.568 | 2.720 | -0.152 |
| (26, 4, 8) | 2.82 | 1150.5 | 482.0 | 2.4 | 71.5 | 70.1 | 1.677 | 1.542 | +0.134 |
| (-2, 2, 5) | 9.64 | 2616.1 | 282.0 | 9.3 | 63.5 | 61.1 | -2.070 | -2.204 | +0.133 |
| (11, 9, 4) | 3.40 | 11179.2 | 3979.0 | 2.8 | 179.9 | 178.6 | 4.083 | 3.953 | +0.130 |
| (16, 8, 17) | 2.43 | 78.9 | 484.0 | 0.2 | 41.7 | 40.3 | 1.261 | 1.136 | +0.125 |
| (7, 1, 22) | 2.63 | 3357.2 | 671.0 | 5.0 | 89.3 | 90.5 | 1.684 | 1.806 | -0.122 |
| (15, 9, 12) | 2.69 | 245.4 | 318.0 | 0.8 | 41.5 | 40.0 | 0.418 | 0.296 | +0.121 |
| (-21, 5, 21) | 2.22 | -247.5 | 396.0 | -0.6 | 30.5 | 29.2 | 1.695 | 1.573 | +0.121 |
| (-9, 5, 4) | 5.30 | 59768.6 | 10467.0 | 5.7 | 309.7 | 311.9 | 2.666 | 2.786 | -0.120 |
| (11, 3, 2) | 6.29 | 12462.4 | 485.0 | 25.7 | 131.0 | 128.5 | -0.695 | -0.815 | +0.120 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `recovered_f` | 21.34 | 0.9840 | 199.05123663228872 | 685.766229474762 | 1.000 |
| `recovered_i` | 21.30 | 0.9838 | 199.04809889248483 | 688.2128744615293 | 1.000 |