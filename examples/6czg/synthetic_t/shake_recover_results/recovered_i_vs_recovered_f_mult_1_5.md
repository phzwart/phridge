# Held-Out Log-Likelihood Comparison: `recovered_i` vs `recovered_f`

## 1. Executive Summary

- **Scored Test Reflections (|T|)**: 900 (held-out)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `0.0782` nats/refl
- **Model B NLL**: `0.0782` nats/refl
- **Difference (Gain $\Delta$)**: `-0.0001` nats/refl (`+0.0001` nats NLL reduction)
- **Estimated Log Bayes Factor**: `-0.05` nats
- **Uncertainty**: Bootstrap SE = `0.0002` (95% CI: `[-0.0004, +0.0003]`) | Naive SE = `0.0001` (Ratio: `1.32`x)
- **Win Fraction $P(d_h > 0)$**: `49.9%` (449 wins, 451 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `9.73e-01`
- **Wilcoxon Signed-Rank $p$-value**: `9.85e-01`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.0001` | `-0.0000` | `+0.0001` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.0001` | `-0.0000` | `+0.0001` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`recovered_f`) | Model B (`recovered_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0132` | `-0.0132` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `0.0841` | `0.0842` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.0059` | `-0.0060` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -1.3488 | -1.3493 | +0.0005 | 0.0008 | 53.5% | 23/20 |
| 1 | 6.10 | 4.86 | 46 | 0.1057 | 0.1040 | +0.0016 | 0.0011 | 50.0% | 23/23 |
| 2 | 4.83 | 4.21 | 45 | -0.3643 | -0.3639 | -0.0005 | 0.0006 | 48.9% | 22/23 |
| 3 | 4.20 | 3.81 | 46 | 0.0322 | 0.0335 | -0.0013 | 0.0007 | 37.0% | 17/29 |
| 4 | 3.80 | 3.53 | 46 | 0.2538 | 0.2538 | +0.0000 | 0.0007 | 45.7% | 21/25 |
| 5 | 3.53 | 3.32 | 46 | 0.3236 | 0.3239 | -0.0003 | 0.0006 | 41.3% | 19/27 |
| 6 | 3.31 | 3.16 | 44 | -0.0296 | -0.0279 | -0.0017 | 0.0006 | 36.4% | 16/28 |
| 7 | 3.14 | 3.02 | 45 | 0.4722 | 0.4715 | +0.0007 | 0.0008 | 53.3% | 24/21 |
| 8 | 3.01 | 2.89 | 46 | -0.0395 | -0.0402 | +0.0007 | 0.0004 | 65.2% | 30/16 |
| 9 | 2.88 | 2.79 | 43 | 0.1385 | 0.1390 | -0.0005 | 0.0007 | 48.8% | 21/22 |
| 10 | 2.79 | 2.70 | 45 | -0.1685 | -0.1671 | -0.0014 | 0.0005 | 40.0% | 18/27 |
| 11 | 2.70 | 2.62 | 45 | 0.0818 | 0.0815 | +0.0003 | 0.0006 | 53.3% | 24/21 |
| 12 | 2.62 | 2.55 | 44 | -0.0677 | -0.0682 | +0.0005 | 0.0004 | 61.4% | 27/17 |
| 13 | 2.55 | 2.49 | 47 | 0.1690 | 0.1692 | -0.0003 | 0.0005 | 46.8% | 22/25 |
| 14 | 2.49 | 2.43 | 43 | -0.0418 | -0.0419 | +0.0000 | 0.0003 | 55.8% | 24/19 |
| 15 | 2.43 | 2.38 | 43 | 0.4620 | 0.4615 | +0.0005 | 0.0004 | 48.8% | 21/22 |
| 16 | 2.38 | 2.33 | 44 | 0.2526 | 0.2519 | +0.0007 | 0.0004 | 65.9% | 29/15 |
| 17 | 2.32 | 2.28 | 47 | 0.3545 | 0.3551 | -0.0005 | 0.0005 | 42.6% | 20/27 |
| 18 | 2.28 | 2.24 | 48 | 0.4141 | 0.4142 | -0.0001 | 0.0005 | 58.3% | 28/20 |
| 19 | 2.24 | 2.20 | 44 | 0.4764 | 0.4765 | -0.0001 | 0.0004 | 45.5% | 20/24 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (-13, 1, 8) | 4.90 | 286163.9 | 2820.0 | 101.5 | 620.1 | 619.7 | 4.925 | 4.887 | +0.038 |
| (3, 1, 0) | 22.09 | 21112.0 | 272.2 | 77.5 | 198.9 | 199.3 | 0.574 | 0.596 | -0.022 |
| (-7, 1, 22) | 2.64 | 4794.6 | 116.9 | 41.0 | 103.8 | 103.7 | 2.386 | 2.366 | +0.021 |
| (-7, 11, 10) | 2.76 | 3344.6 | 41.8 | 79.9 | 99.9 | 100.0 | 2.776 | 2.795 | -0.019 |
| (-12, 6, 14) | 3.11 | 10697.2 | 162.8 | 65.7 | 151.9 | 151.7 | 3.098 | 3.079 | +0.019 |
| (17, 5, 0) | 4.08 | 57159.6 | 1145.6 | 49.9 | 287.3 | 287.6 | 2.236 | 2.255 | -0.019 |
| (-18, 0, 14) | 3.18 | 5539.7 | 347.2 | 16.0 | 130.8 | 131.1 | 1.745 | 1.763 | -0.018 |
| (-5, 3, 4) | 8.14 | 42332.2 | 847.6 | 49.9 | 245.5 | 245.1 | 0.418 | 0.402 | +0.016 |
| (-11, 1, 12) | 4.17 | 38948.5 | 809.4 | 48.1 | 235.5 | 235.8 | 1.477 | 1.493 | -0.016 |
| (21, 3, 14) | 2.82 | 3820.6 | 128.1 | 29.8 | 104.9 | 105.1 | 2.983 | 2.998 | -0.015 |
| (22, 6, 1) | 3.22 | 730.3 | 83.2 | 8.8 | 50.2 | 50.4 | -0.494 | -0.479 | -0.015 |
| (0, 14, 11) | 2.29 | 1048.1 | 30.2 | 34.8 | 51.6 | 51.7 | 1.177 | 1.192 | -0.015 |
| (-12, 8, 6) | 3.51 | 50749.5 | 637.3 | 79.6 | 278.9 | 278.7 | 3.092 | 3.078 | +0.014 |
| (17, 5, 22) | 2.24 | 1604.3 | 55.5 | 28.9 | 27.6 | 27.4 | 1.043 | 1.056 | -0.013 |
| (16, 12, 4) | 2.53 | 372.6 | 30.3 | 12.3 | 38.6 | 38.7 | 0.241 | 0.254 | -0.013 |
| (-24, 0, 14) | 2.74 | -39.2 | 70.9 | -0.6 | 26.9 | 27.1 | -1.201 | -1.189 | -0.012 |
| (18, 8, 4) | 3.14 | 20682.9 | 210.0 | 98.5 | 183.4 | 183.3 | 2.177 | 2.165 | +0.012 |
| (-3, 3, 11) | 4.86 | 24311.1 | 293.4 | 82.9 | 195.5 | 195.3 | 1.225 | 1.213 | +0.012 |
| (-9, 13, 12) | 2.31 | 186.9 | 41.8 | 4.5 | 24.4 | 24.5 | -0.358 | -0.346 | -0.012 |
| (8, 6, 11) | 3.72 | 782.6 | 57.1 | 13.7 | 72.3 | 72.1 | 0.499 | 0.487 | +0.012 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `recovered_f` | 15.78 | 0.9921 | 4.7118256357385295 | 3.474313210162355 | 1.000 |
| `recovered_i` | 15.79 | 0.9921 | 4.709337646368276 | 3.471521180753639 | 1.000 |