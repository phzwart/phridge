# Held-Out Log-Likelihood Comparison: `recovered_i` vs `recovered_f`

## 1. Executive Summary

- **Scored Test Reflections (|T|)**: 900 (held-out)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `0.0833` nats/refl
- **Model B NLL**: `0.0837` nats/refl
- **Difference (Gain $\Delta$)**: `-0.0004` nats/refl (`+0.0004` nats NLL reduction)
- **Estimated Log Bayes Factor**: `-0.34` nats
- **Uncertainty**: Bootstrap SE = `0.0003` (95% CI: `[-0.0010, +0.0002]`) | Naive SE = `0.0002` (Ratio: `1.27`x)
- **Win Fraction $P(d_h > 0)$**: `48.6%` (437 wins, 463 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `4.05e-01`
- **Wilcoxon Signed-Rank $p$-value**: `3.50e-02`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.0005` | `-0.0001` | `+0.0004` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.0005` | `-0.0001` | `+0.0004` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`recovered_f`) | Model B (`recovered_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0152` | `-0.0154` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `0.0982` | `0.0983` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.0149` | `-0.0146` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -1.3363 | -1.3324 | -0.0039 | 0.0012 | 32.6% | 14/29 |
| 1 | 6.10 | 4.86 | 46 | 0.0903 | 0.0881 | +0.0022 | 0.0026 | 39.1% | 18/28 |
| 2 | 4.83 | 4.21 | 45 | -0.3622 | -0.3618 | -0.0004 | 0.0011 | 46.7% | 21/24 |
| 3 | 4.20 | 3.81 | 46 | 0.0558 | 0.0576 | -0.0019 | 0.0011 | 28.3% | 13/33 |
| 4 | 3.80 | 3.53 | 46 | 0.2362 | 0.2384 | -0.0022 | 0.0014 | 34.8% | 16/30 |
| 5 | 3.53 | 3.32 | 46 | 0.4064 | 0.4083 | -0.0019 | 0.0016 | 30.4% | 14/32 |
| 6 | 3.31 | 3.16 | 44 | -0.0285 | -0.0283 | -0.0002 | 0.0010 | 43.2% | 19/25 |
| 7 | 3.14 | 3.02 | 45 | 0.4593 | 0.4595 | -0.0002 | 0.0008 | 48.9% | 22/23 |
| 8 | 3.01 | 2.89 | 46 | -0.0506 | -0.0506 | +0.0000 | 0.0004 | 45.7% | 21/25 |
| 9 | 2.88 | 2.79 | 43 | 0.1065 | 0.1073 | -0.0008 | 0.0005 | 30.2% | 13/30 |
| 10 | 2.79 | 2.70 | 45 | -0.2621 | -0.2618 | -0.0003 | 0.0005 | 44.4% | 20/25 |
| 11 | 2.70 | 2.62 | 45 | 0.0850 | 0.0849 | +0.0001 | 0.0004 | 60.0% | 27/18 |
| 12 | 2.62 | 2.55 | 44 | -0.0018 | -0.0010 | -0.0007 | 0.0004 | 45.5% | 20/24 |
| 13 | 2.55 | 2.49 | 47 | 0.1523 | 0.1524 | -0.0001 | 0.0005 | 63.8% | 30/17 |
| 14 | 2.49 | 2.43 | 43 | -0.0666 | -0.0663 | -0.0003 | 0.0005 | 48.8% | 21/22 |
| 15 | 2.43 | 2.38 | 43 | 0.4675 | 0.4675 | +0.0000 | 0.0006 | 69.8% | 30/13 |
| 16 | 2.38 | 2.33 | 44 | 0.3398 | 0.3384 | +0.0014 | 0.0009 | 59.1% | 26/18 |
| 17 | 2.32 | 2.28 | 47 | 0.3796 | 0.3790 | +0.0005 | 0.0006 | 68.1% | 32/15 |
| 18 | 2.28 | 2.24 | 48 | 0.4105 | 0.4095 | +0.0010 | 0.0006 | 62.5% | 30/18 |
| 19 | 2.24 | 2.20 | 44 | 0.4988 | 0.4989 | -0.0002 | 0.0008 | 68.2% | 30/14 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (-13, 1, 8) | 4.90 | 285707.9 | 3760.0 | 76.0 | 621.1 | 621.0 | 5.192 | 5.132 | +0.059 |
| (-18, 4, 10) | 3.43 | 54968.0 | 1286.6 | 42.7 | 287.9 | 288.5 | 3.148 | 3.199 | -0.051 |
| (14, 2, 4) | 5.33 | 190733.7 | 3048.0 | 62.6 | 509.4 | 509.2 | 3.830 | 3.780 | +0.050 |
| (14, 4, 0) | 5.00 | 55363.9 | 1068.0 | 51.8 | 274.5 | 273.9 | 1.404 | 1.365 | +0.040 |
| (-5, 3, 4) | 8.14 | 40050.2 | 1130.2 | 35.4 | 244.8 | 245.6 | 0.622 | 0.661 | -0.039 |
| (14, 4, 2) | 4.92 | 22133.3 | 226.8 | 97.6 | 184.0 | 183.3 | 0.950 | 0.912 | +0.039 |
| (9, 3, 14) | 3.66 | 133223.3 | 2219.8 | 60.0 | 441.8 | 442.2 | 4.584 | 4.622 | -0.037 |
| (11, 9, 4) | 3.40 | 17263.6 | 795.8 | 21.7 | 181.0 | 180.8 | 2.791 | 2.759 | +0.031 |
| (5, 5, 4) | 5.96 | 16950.9 | 97.2 | 174.4 | 165.6 | 165.1 | 0.690 | 0.661 | +0.030 |
| (-14, 6, 11) | 3.35 | 98.4 | 210.2 | 0.5 | 33.8 | 34.2 | -0.634 | -0.607 | -0.027 |
| (10, 2, 6) | 6.02 | 22869.1 | 594.6 | 38.5 | 190.3 | 190.9 | 0.379 | 0.405 | -0.026 |
| (-12, 4, 8) | 4.46 | 74425.5 | 2057.2 | 36.2 | 332.1 | 331.8 | 2.323 | 2.298 | +0.025 |
| (-19, 1, 10) | 3.58 | 56227.7 | 921.8 | 61.0 | 278.4 | 278.8 | 1.667 | 1.691 | -0.024 |
| (19, 7, 4) | 3.26 | 4023.2 | 185.2 | 21.7 | 94.8 | 95.2 | 0.621 | 0.645 | -0.024 |
| (2, 6, 3) | 5.59 | 885.8 | 30.0 | 29.5 | 47.3 | 47.9 | -1.241 | -1.217 | -0.024 |
| (-12, 8, 6) | 3.51 | 49710.6 | 849.8 | 58.5 | 279.2 | 279.1 | 3.385 | 3.362 | +0.023 |
| (0, 4, 12) | 4.33 | 80210.9 | 828.8 | 96.8 | 350.4 | 350.7 | 3.053 | 3.076 | -0.023 |
| (-11, 1, 12) | 4.17 | 37957.8 | 1079.2 | 35.2 | 235.7 | 235.4 | 1.644 | 1.621 | +0.023 |
| (-8, 4, 14) | 3.62 | 12510.0 | 525.2 | 23.8 | 151.2 | 150.9 | 1.128 | 1.106 | +0.022 |
| (-8, 8, 6) | 3.78 | 25365.8 | 1180.4 | 21.5 | 209.8 | 210.1 | 1.955 | 1.977 | -0.022 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `recovered_f` | 16.08 | 0.9909 | 199.07228966685224 | 1000.0 | 1.000 |
| `recovered_i` | 16.09 | 0.9909 | 199.072829761161 | 1000.0 | 1.000 |