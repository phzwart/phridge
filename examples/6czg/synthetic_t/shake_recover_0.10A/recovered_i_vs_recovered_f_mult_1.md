# Held-Out Log-Likelihood Comparison: `recovered_i` vs `recovered_f`

## 1. Executive Summary

- **Scored Audit Reflections (|A|)**: 900 (audit set)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `-0.1152` nats/refl
- **Model B NLL**: `-0.1147` nats/refl
- **Difference (Gain $\Delta$)**: `-0.0006` nats/refl (`+0.0006` nats NLL reduction)
- **Estimated Log Bayes Factor**: `-0.50` nats
- **Uncertainty**: Bootstrap SE = `0.0003` (95% CI: `[-0.0013, +0.0000]`) | Naive SE = `0.0003` (Ratio: `1.01`x)
- **Win Fraction $P(d_h > 0)$**: `41.2%` (371 wins, 529 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `1.55e-07`
- **Wilcoxon Signed-Rank $p$-value**: `1.59e-03`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.0006` | `-0.0000` | `+0.0006` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.0006` | `-0.0000` | `+0.0006` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`recovered_f`) | Model B (`recovered_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0086` | `-0.0087` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `-0.1113` | `-0.1107` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.0039` | `-0.0039` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -1.1361 | -1.1354 | -0.0007 | 0.0012 | 44.2% | 19/24 |
| 1 | 6.10 | 4.86 | 46 | 0.4904 | 0.4908 | -0.0004 | 0.0045 | 28.3% | 13/33 |
| 2 | 4.83 | 4.21 | 45 | -0.2718 | -0.2704 | -0.0014 | 0.0014 | 53.3% | 24/21 |
| 3 | 4.20 | 3.81 | 46 | 0.0778 | 0.0821 | -0.0043 | 0.0016 | 41.3% | 19/27 |
| 4 | 3.80 | 3.53 | 46 | 0.3022 | 0.3070 | -0.0048 | 0.0024 | 41.3% | 19/27 |
| 5 | 3.53 | 3.32 | 46 | 0.3819 | 0.3815 | +0.0005 | 0.0019 | 45.7% | 21/25 |
| 6 | 3.31 | 3.16 | 44 | -0.1149 | -0.1166 | +0.0017 | 0.0013 | 56.8% | 25/19 |
| 7 | 3.14 | 3.02 | 45 | 0.4019 | 0.4014 | +0.0005 | 0.0021 | 51.1% | 23/22 |
| 8 | 3.01 | 2.89 | 46 | -0.1634 | -0.1631 | -0.0003 | 0.0005 | 45.7% | 21/25 |
| 9 | 2.88 | 2.79 | 43 | -0.1657 | -0.1650 | -0.0007 | 0.0005 | 32.6% | 14/29 |
| 10 | 2.79 | 2.70 | 45 | -0.5322 | -0.5319 | -0.0003 | 0.0004 | 42.2% | 19/26 |
| 11 | 2.70 | 2.62 | 45 | -0.1535 | -0.1542 | +0.0006 | 0.0005 | 35.6% | 16/29 |
| 12 | 2.62 | 2.55 | 44 | -0.3432 | -0.3430 | -0.0002 | 0.0003 | 29.5% | 13/31 |
| 13 | 2.55 | 2.49 | 47 | -0.1688 | -0.1687 | -0.0000 | 0.0006 | 42.6% | 20/27 |
| 14 | 2.49 | 2.43 | 43 | -0.5181 | -0.5181 | +0.0000 | 0.0004 | 41.9% | 18/25 |
| 15 | 2.43 | 2.38 | 43 | 0.1050 | 0.1051 | -0.0001 | 0.0005 | 44.2% | 19/24 |
| 16 | 2.38 | 2.33 | 44 | -0.2061 | -0.2058 | -0.0003 | 0.0005 | 43.2% | 19/25 |
| 17 | 2.32 | 2.28 | 47 | -0.0985 | -0.0982 | -0.0002 | 0.0006 | 25.5% | 12/35 |
| 18 | 2.28 | 2.24 | 48 | -0.1482 | -0.1486 | +0.0004 | 0.0006 | 50.0% | 24/24 |
| 19 | 2.24 | 2.20 | 44 | -0.1397 | -0.1389 | -0.0008 | 0.0004 | 29.5% | 13/31 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (-13, 1, 8) | 4.90 | 286007.4 | 1880.0 | 152.1 | 649.8 | 649.2 | 8.911 | 8.788 | +0.123 |
| (14, 2, 4) | 5.33 | 191022.5 | 1524.0 | 125.3 | 535.5 | 534.9 | 6.297 | 6.200 | +0.097 |
| (-9, 5, 4) | 5.30 | 73520.8 | 1046.7 | 70.2 | 333.4 | 334.2 | 2.815 | 2.876 | -0.061 |
| (21, 3, 2) | 3.79 | 56287.7 | 956.4 | 58.9 | 297.2 | 297.9 | 3.088 | 3.147 | -0.060 |
| (-14, 4, 2) | 4.94 | 126721.0 | 1288.0 | 98.4 | 427.2 | 426.8 | 3.801 | 3.748 | +0.053 |
| (10, 4, 2) | 5.98 | 67898.4 | 413.5 | 164.2 | 311.1 | 312.0 | 1.785 | 1.832 | -0.046 |
| (9, 5, 4) | 5.27 | 126633.4 | 1439.0 | 88.0 | 423.5 | 424.0 | 3.258 | 3.300 | -0.042 |
| (8, 2, 18) | 3.10 | 2749.6 | 106.7 | 25.8 | 73.0 | 72.6 | 0.402 | 0.361 | +0.041 |
| (9, 3, 14) | 3.66 | 132444.0 | 1109.9 | 119.3 | 445.8 | 446.1 | 5.906 | 5.945 | -0.039 |
| (24, 2, 2) | 3.44 | 31486.3 | 349.5 | 90.1 | 217.6 | 217.2 | 2.428 | 2.388 | +0.039 |
| (-14, 2, 12) | 3.78 | 72853.3 | 380.7 | 191.4 | 322.3 | 322.8 | 2.457 | 2.495 | -0.038 |
| (-12, 10, 2) | 3.14 | 19020.1 | 372.2 | 51.1 | 166.3 | 165.9 | 1.620 | 1.582 | +0.038 |
| (-4, 6, 4) | 5.31 | 21428.0 | 514.3 | 41.7 | 181.8 | 182.6 | 0.835 | 0.872 | -0.037 |
| (2, 2, 14) | 4.11 | 122539.2 | 1141.1 | 107.4 | 434.5 | 434.9 | 6.611 | 6.648 | -0.036 |
| (18, 8, 4) | 3.14 | 20760.3 | 140.0 | 148.3 | 175.8 | 175.5 | 1.973 | 1.939 | +0.034 |
| (-9, 3, 14) | 3.70 | 50414.8 | 426.4 | 118.2 | 267.3 | 267.8 | 1.770 | 1.803 | -0.033 |
| (-1, 11, 4) | 3.14 | 11639.7 | 169.3 | 68.8 | 133.9 | 134.2 | 1.289 | 1.319 | -0.030 |
| (7, 5, 4) | 5.63 | 50478.1 | 634.4 | 79.6 | 260.2 | 260.9 | 1.046 | 1.075 | -0.029 |
| (0, 8, 5) | 4.14 | 443.2 | 36.0 | 12.3 | 34.8 | 35.4 | -1.405 | -1.377 | -0.028 |
| (-12, 6, 14) | 3.11 | 10869.7 | 108.5 | 100.2 | 138.1 | 138.3 | 2.157 | 2.185 | -0.028 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `recovered_f` | 9.01 | 0.9975 | 199.16575536516672 | 1000.0 | 1.000 |
| `recovered_i` | 9.02 | 0.9975 | 199.16555110781067 | 1000.0 | 1.000 |