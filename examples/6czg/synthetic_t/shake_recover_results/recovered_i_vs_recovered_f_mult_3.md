# Held-Out Log-Likelihood Comparison: `recovered_i` vs `recovered_f`

## 1. Executive Summary

- **Scored Test Reflections (|T|)**: 900 (held-out)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `0.1085` nats/refl
- **Model B NLL**: `0.1092` nats/refl
- **Difference (Gain $\Delta$)**: `-0.0007` nats/refl (`+0.0007` nats NLL reduction)
- **Estimated Log Bayes Factor**: `-0.64` nats
- **Uncertainty**: Bootstrap SE = `0.0005` (95% CI: `[-0.0017, +0.0002]`) | Naive SE = `0.0003` (Ratio: `1.57`x)
- **Win Fraction $P(d_h > 0)$**: `46.1%` (415 wins, 485 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `2.14e-02`
- **Wilcoxon Signed-Rank $p$-value**: `6.61e-03`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.0007` | `+0.0000` | `+0.0007` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.0007` | `+0.0000` | `+0.0007` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`recovered_f`) | Model B (`recovered_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0151` | `-0.0150` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `0.1170` | `0.1175` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.0085` | `-0.0082` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -1.3514 | -1.3448 | -0.0067 | 0.0018 | 18.6% | 8/35 |
| 1 | 6.10 | 4.86 | 46 | 0.1233 | 0.1245 | -0.0012 | 0.0028 | 34.8% | 16/30 |
| 2 | 4.83 | 4.21 | 45 | -0.3322 | -0.3320 | -0.0002 | 0.0013 | 37.8% | 17/28 |
| 3 | 4.20 | 3.81 | 46 | 0.0997 | 0.1042 | -0.0045 | 0.0018 | 26.1% | 12/34 |
| 4 | 3.80 | 3.53 | 46 | 0.2856 | 0.2894 | -0.0038 | 0.0020 | 37.0% | 17/29 |
| 5 | 3.53 | 3.32 | 46 | 0.3746 | 0.3773 | -0.0027 | 0.0018 | 32.6% | 15/31 |
| 6 | 3.31 | 3.16 | 44 | -0.0315 | -0.0311 | -0.0004 | 0.0014 | 38.6% | 17/27 |
| 7 | 3.14 | 3.02 | 45 | 0.4886 | 0.4884 | +0.0002 | 0.0021 | 53.3% | 24/21 |
| 8 | 3.01 | 2.89 | 46 | -0.0329 | -0.0336 | +0.0008 | 0.0007 | 56.5% | 26/20 |
| 9 | 2.88 | 2.79 | 43 | 0.1145 | 0.1163 | -0.0018 | 0.0009 | 34.9% | 15/28 |
| 10 | 2.79 | 2.70 | 45 | -0.2183 | -0.2168 | -0.0014 | 0.0009 | 31.1% | 14/31 |
| 11 | 2.70 | 2.62 | 45 | 0.1596 | 0.1585 | +0.0011 | 0.0013 | 55.6% | 25/20 |
| 12 | 2.62 | 2.55 | 44 | -0.0503 | -0.0506 | +0.0004 | 0.0006 | 45.5% | 20/24 |
| 13 | 2.55 | 2.49 | 47 | 0.1982 | 0.1964 | +0.0018 | 0.0009 | 63.8% | 30/17 |
| 14 | 2.49 | 2.43 | 43 | 0.0477 | 0.0484 | -0.0008 | 0.0009 | 44.2% | 19/24 |
| 15 | 2.43 | 2.38 | 43 | 0.5574 | 0.5557 | +0.0017 | 0.0006 | 60.5% | 26/17 |
| 16 | 2.38 | 2.33 | 44 | 0.2720 | 0.2707 | +0.0013 | 0.0007 | 65.9% | 29/15 |
| 17 | 2.32 | 2.28 | 47 | 0.4872 | 0.4876 | -0.0004 | 0.0008 | 57.4% | 27/20 |
| 18 | 2.28 | 2.24 | 48 | 0.4121 | 0.4108 | +0.0013 | 0.0009 | 62.5% | 30/18 |
| 19 | 2.24 | 2.20 | 44 | 0.4764 | 0.4757 | +0.0007 | 0.0007 | 63.6% | 28/16 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (-13, 1, 8) | 4.90 | 286196.6 | 5640.0 | 50.7 | 619.7 | 619.2 | 4.928 | 4.843 | +0.085 |
| (17, 5, 0) | 4.08 | 53884.0 | 2291.1 | 23.5 | 287.0 | 287.8 | 2.754 | 2.813 | -0.059 |
| (26, 4, 2) | 3.04 | 6977.4 | 132.9 | 52.5 | 120.1 | 120.7 | 1.830 | 1.888 | -0.057 |
| (1, 7, 2) | 4.96 | 78832.9 | 4110.3 | 19.2 | 340.2 | 341.1 | 2.752 | 2.806 | -0.055 |
| (-7, 1, 22) | 2.64 | 4340.6 | 233.7 | 18.6 | 103.1 | 102.8 | 2.661 | 2.614 | +0.046 |
| (8, 6, 11) | 3.72 | 605.6 | 114.3 | 5.3 | 71.1 | 71.7 | 0.580 | 0.626 | -0.046 |
| (19, 1, 8) | 3.78 | 40268.7 | 1555.2 | 25.9 | 241.2 | 242.0 | 1.384 | 1.426 | -0.042 |
| (14, 2, 4) | 5.33 | 173387.8 | 4572.0 | 37.9 | 507.5 | 507.6 | 5.790 | 5.749 | +0.041 |
| (-18, 4, 10) | 3.43 | 49670.1 | 1929.9 | 25.7 | 287.7 | 288.1 | 4.366 | 4.406 | -0.040 |
| (-5, 3, 4) | 8.14 | 42932.4 | 1695.3 | 25.3 | 244.2 | 245.4 | 0.332 | 0.371 | -0.039 |
| (-6, 10, 1) | 3.42 | 7236.0 | 534.6 | 13.5 | 117.1 | 116.6 | 1.061 | 1.022 | +0.039 |
| (0, 8, 5) | 4.14 | 523.6 | 108.0 | 4.8 | 45.2 | 45.9 | -0.900 | -0.862 | -0.038 |
| (10, 0, 0) | 8.49 | 106409.9 | 8982.0 | 11.8 | 377.1 | 378.8 | 1.006 | 1.041 | -0.036 |
| (-8, 8, 6) | 3.78 | 27640.2 | 1770.6 | 15.6 | 210.4 | 211.0 | 1.501 | 1.535 | -0.033 |
| (21, 3, 14) | 2.82 | 3545.7 | 256.2 | 13.8 | 104.6 | 104.9 | 3.176 | 3.208 | -0.032 |
| (12, 8, 6) | 3.49 | 34723.0 | 1263.3 | 27.5 | 220.2 | 220.8 | 1.526 | 1.558 | -0.031 |
| (-7, 11, 10) | 2.76 | 3429.2 | 83.7 | 41.0 | 100.4 | 100.7 | 2.673 | 2.705 | -0.031 |
| (-5, 9, 1) | 3.82 | 3986.0 | 274.5 | 14.5 | 94.1 | 93.7 | 0.539 | 0.508 | +0.031 |
| (8, 2, 18) | 3.10 | 2691.9 | 320.1 | 8.4 | 88.2 | 88.0 | 1.487 | 1.460 | +0.027 |
| (-28, 10, 1) | 2.30 | 1350.8 | 116.4 | 11.6 | 17.6 | 17.4 | 1.268 | 1.294 | -0.026 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `recovered_f` | 16.71 | 0.9900 | 199.14759571182614 | 1000.0 | 1.000 |
| `recovered_i` | 16.73 | 0.9900 | 199.1474825067724 | 1000.0 | 1.000 |