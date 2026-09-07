# Held-Out Log-Likelihood Comparison: `recovered_i` vs `deposited`

## 1. Executive Summary

- **Scored Test Reflections (|T|)**: 900 (held-out)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `0.0322` nats/refl
- **Model B NLL**: `0.1974` nats/refl
- **Difference (Gain $\Delta$)**: `-0.1653` nats/refl (`+0.1653` nats NLL reduction)
- **Estimated Log Bayes Factor**: `-148.75` nats
- **Uncertainty**: Bootstrap SE = `0.0306` (95% CI: `[-0.2273, -0.1073]`) | Naive SE = `0.0161` (Ratio: `1.91`x)
- **Win Fraction $P(d_h > 0)$**: `26.4%` (238 wins, 662 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `6.77e-47`
- **Wilcoxon Signed-Rank $p$-value**: `1.39e-44`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.2633` | `-0.0980` | `+0.1653` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.1216` | `+0.0436` | `+0.1653` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`deposited`) | Model B (`recovered_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0071` | `-0.0162` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `0.1228` | `0.2092` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.0907` | `-0.0117` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -1.5441 | -1.1466 | -0.3975 | 0.1675 | 18.6% | 8/35 |
| 1 | 6.10 | 4.86 | 46 | -0.1528 | 0.2603 | -0.4131 | 0.0357 | 2.2% | 1/45 |
| 2 | 4.83 | 4.21 | 45 | -0.6179 | -0.2553 | -0.3626 | 0.0487 | 13.3% | 6/39 |
| 3 | 4.20 | 3.81 | 46 | -0.1780 | 0.0645 | -0.2425 | 0.0491 | 19.6% | 9/37 |
| 4 | 3.80 | 3.53 | 46 | 0.0820 | 0.3664 | -0.2843 | 0.0663 | 13.0% | 6/40 |
| 5 | 3.53 | 3.32 | 46 | 0.3107 | 0.5054 | -0.1947 | 0.0542 | 19.6% | 9/37 |
| 6 | 3.31 | 3.16 | 44 | -0.1737 | 0.0335 | -0.2072 | 0.0466 | 20.5% | 9/35 |
| 7 | 3.14 | 3.02 | 45 | 0.3932 | 0.6434 | -0.2503 | 0.0732 | 17.8% | 8/37 |
| 8 | 3.01 | 2.89 | 46 | -0.2065 | 0.0432 | -0.2498 | 0.0454 | 15.2% | 7/39 |
| 9 | 2.88 | 2.79 | 43 | 0.1173 | 0.2830 | -0.1658 | 0.0878 | 23.3% | 10/33 |
| 10 | 2.79 | 2.70 | 45 | -0.3108 | -0.1728 | -0.1380 | 0.0459 | 26.7% | 12/33 |
| 11 | 2.70 | 2.62 | 45 | 0.1873 | 0.2877 | -0.1004 | 0.0463 | 40.0% | 18/27 |
| 12 | 2.62 | 2.55 | 44 | 0.1010 | 0.1893 | -0.0883 | 0.0473 | 36.4% | 16/28 |
| 13 | 2.55 | 2.49 | 47 | 0.0881 | 0.2053 | -0.1172 | 0.0461 | 25.5% | 12/35 |
| 14 | 2.49 | 2.43 | 43 | -0.1044 | -0.0512 | -0.0532 | 0.0401 | 37.2% | 16/27 |
| 15 | 2.43 | 2.38 | 43 | 0.4917 | 0.4116 | +0.0801 | 0.0836 | 32.6% | 14/29 |
| 16 | 2.38 | 2.33 | 44 | 0.3501 | 0.4230 | -0.0729 | 0.0553 | 38.6% | 17/27 |
| 17 | 2.32 | 2.28 | 47 | 0.7179 | 0.6953 | +0.0226 | 0.0816 | 42.6% | 20/27 |
| 18 | 2.28 | 2.24 | 48 | 0.3901 | 0.4931 | -0.1030 | 0.0506 | 41.7% | 20/28 |
| 19 | 2.24 | 2.20 | 44 | 0.6173 | 0.5741 | +0.0432 | 0.0935 | 45.5% | 20/24 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (3, 1, 0) | 22.09 | 19917.5 | 1815.0 | 11.0 | 176.3 | 176.4 | 3.700 | -0.291 | +3.990 |
| (-5, 13, 13) | 2.32 | -560.0 | 338.0 | -1.7 | 38.6 | 28.8 | 7.011 | 3.737 | +3.275 |
| (-27, 11, 4) | 2.22 | 1793.3 | 497.0 | 3.6 | 68.9 | 59.8 | 4.779 | 1.728 | +3.051 |
| (-5, 3, 4) | 8.14 | 27472.1 | 5651.0 | 4.9 | 232.4 | 243.3 | 3.792 | 1.576 | +2.217 |
| (-21, 3, 14) | 2.86 | 291.3 | 1066.0 | 0.3 | 88.0 | 90.0 | 8.008 | 5.947 | +2.062 |
| (0, 4, 1) | 8.73 | 4.5 | 131.0 | 0.0 | 6.5 | 15.6 | -4.092 | -2.091 | -2.001 |
| (-1, 1, 3) | 17.00 | -15.7 | 299.0 | -0.1 | 9.8 | 10.9 | -3.940 | -2.050 | -1.890 |
| (-12, 6, 14) | 3.11 | 11109.8 | 1085.0 | 10.2 | 115.2 | 140.3 | 0.264 | 2.042 | -1.778 |
| (-19, 1, 21) | 2.41 | 6678.8 | 1013.0 | 6.6 | 104.2 | 93.9 | 2.759 | 1.034 | +1.725 |
| (-10, 2, 2) | 7.44 | 22735.9 | 2999.0 | 7.6 | 192.1 | 201.3 | 2.218 | 0.512 | +1.706 |
| (14, 8, 18) | 2.42 | 4573.6 | 736.0 | 6.2 | 87.7 | 74.8 | 2.248 | 0.564 | +1.684 |
| (-9, 13, 10) | 2.39 | -189.1 | 223.0 | -0.8 | 28.9 | 21.7 | 2.668 | 1.001 | +1.667 |
| (27, 11, 4) | 2.22 | 3271.2 | 607.0 | 5.4 | 74.2 | 61.0 | 2.248 | 0.597 | +1.652 |
| (6, 2, 0) | 11.04 | 59213.2 | 8776.0 | 6.7 | 293.5 | 289.9 | 2.189 | 0.552 | +1.637 |
| (-16, 4, 12) | 3.38 | 66572.1 | 6116.0 | 10.9 | 315.7 | 322.1 | 5.947 | 4.359 | +1.587 |
| (1, 1, 3) | 16.91 | 238.8 | 82.0 | 2.9 | 18.8 | 13.8 | -4.304 | -2.758 | -1.545 |
| (26, 4, 2) | 3.04 | 6535.1 | 443.0 | 14.8 | 92.7 | 111.7 | 0.203 | 1.681 | -1.477 |
| (8, 6, 11) | 3.72 | 592.2 | 381.0 | 1.6 | 29.9 | 53.7 | -1.918 | -0.538 | -1.379 |
| (2, 10, 12) | 2.87 | 887.2 | 458.0 | 1.9 | 38.2 | 55.3 | -0.604 | 0.768 | -1.372 |
| (2, 2, 5) | 9.59 | 1147.6 | 133.0 | 8.6 | 35.3 | 49.1 | -3.293 | -1.931 | -1.362 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `deposited` | 12.69 | 0.9925 | 199.00140959097453 | 495.1435794009256 | 1.000 |
| `recovered_i` | 15.89 | 0.9902 | 199.12194708666098 | 503.87368945984736 | 1.000 |