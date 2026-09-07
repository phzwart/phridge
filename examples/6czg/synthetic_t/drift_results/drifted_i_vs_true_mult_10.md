# Held-Out Log-Likelihood Comparison: `drifted_i` vs `true_model`

## 1. Executive Summary

- **Scored Test Reflections (|T|)**: 900 (held-out)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `0.0322` nats/refl
- **Model B NLL**: `0.1584` nats/refl
- **Difference (Gain $\Delta$)**: `-0.1262` nats/refl (`+0.1262` nats NLL reduction)
- **Estimated Log Bayes Factor**: `-113.60` nats
- **Uncertainty**: Bootstrap SE = `0.0419` (95% CI: `[-0.2166, -0.0506]`) | Naive SE = `0.0141` (Ratio: `2.96`x)
- **Win Fraction $P(d_h > 0)$**: `27.8%` (250 wins, 650 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `9.90e-42`
- **Wilcoxon Signed-Rank $p$-value**: `1.00e-39`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.2158` | `-0.0896` | `+0.1262` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.0936` | `+0.0327` | `+0.1262` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`true_model`) | Model B (`drifted_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0071` | `-0.0117` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `0.1228` | `0.1764` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.0907` | `-0.0180` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -1.5441 | -1.1387 | -0.4054 | 0.1688 | 16.3% | 7/36 |
| 1 | 6.10 | 4.86 | 46 | -0.1528 | 0.4033 | -0.5560 | 0.0569 | 0.0% | 0/46 |
| 2 | 4.83 | 4.21 | 45 | -0.6179 | -0.2334 | -0.3845 | 0.0411 | 8.9% | 4/41 |
| 3 | 4.20 | 3.81 | 46 | -0.1780 | 0.1093 | -0.2873 | 0.0266 | 2.2% | 1/45 |
| 4 | 3.80 | 3.53 | 46 | 0.0820 | 0.3978 | -0.3158 | 0.0453 | 15.2% | 7/39 |
| 5 | 3.53 | 3.32 | 46 | 0.3107 | 0.5181 | -0.2074 | 0.0338 | 13.0% | 6/40 |
| 6 | 3.31 | 3.16 | 44 | -0.1737 | -0.0331 | -0.1407 | 0.0281 | 18.2% | 8/36 |
| 7 | 3.14 | 3.02 | 45 | 0.3932 | 0.5463 | -0.1531 | 0.0482 | 15.6% | 7/38 |
| 8 | 3.01 | 2.89 | 46 | -0.2065 | -0.0765 | -0.1300 | 0.0307 | 17.4% | 8/38 |
| 9 | 2.88 | 2.79 | 43 | 0.1173 | 0.2297 | -0.1125 | 0.0510 | 25.6% | 11/32 |
| 10 | 2.79 | 2.70 | 45 | -0.3108 | -0.2312 | -0.0795 | 0.0263 | 22.2% | 10/35 |
| 11 | 2.70 | 2.62 | 45 | 0.1873 | 0.2989 | -0.1116 | 0.0380 | 28.9% | 13/32 |
| 12 | 2.62 | 2.55 | 44 | 0.1010 | 0.0993 | +0.0017 | 0.0387 | 43.2% | 19/25 |
| 13 | 2.55 | 2.49 | 47 | 0.0881 | 0.1349 | -0.0468 | 0.0332 | 29.8% | 14/33 |
| 14 | 2.49 | 2.43 | 43 | -0.1044 | -0.1172 | +0.0129 | 0.0250 | 41.9% | 18/25 |
| 15 | 2.43 | 2.38 | 43 | 0.4917 | 0.3187 | +0.1730 | 0.0727 | 44.2% | 19/24 |
| 16 | 2.38 | 2.33 | 44 | 0.3501 | 0.3127 | +0.0374 | 0.0385 | 50.0% | 22/22 |
| 17 | 2.32 | 2.28 | 47 | 0.7179 | 0.5938 | +0.1241 | 0.0666 | 48.9% | 23/24 |
| 18 | 2.28 | 2.24 | 48 | 0.3901 | 0.3749 | +0.0152 | 0.0345 | 66.7% | 32/16 |
| 19 | 2.24 | 2.20 | 44 | 0.6173 | 0.5623 | +0.0550 | 0.0675 | 47.7% | 21/23 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (3, 1, 0) | 22.09 | 19917.5 | 1815.0 | 11.0 | 176.3 | 166.6 | 3.700 | -0.476 | +4.176 |
| (-5, 13, 13) | 2.32 | -560.0 | 338.0 | -1.7 | 38.6 | 30.2 | 7.011 | 4.203 | +2.808 |
| (-5, 3, 4) | 8.14 | 27472.1 | 5651.0 | 4.9 | 232.4 | 240.6 | 3.792 | 1.455 | +2.337 |
| (-27, 11, 4) | 2.22 | 1793.3 | 497.0 | 3.6 | 68.9 | 62.8 | 4.779 | 2.537 | +2.241 |
| (-13, 1, 8) | 4.90 | 305862.6 | 18800.0 | 16.3 | 601.8 | 647.0 | 2.986 | 5.019 | -2.033 |
| (0, 4, 1) | 8.73 | 4.5 | 131.0 | 0.0 | 6.5 | 13.1 | -4.092 | -2.124 | -1.969 |
| (-1, 1, 3) | 17.00 | -15.7 | 299.0 | -0.1 | 9.8 | 11.6 | -3.940 | -2.041 | -1.899 |
| (-19, 1, 21) | 2.41 | 6678.8 | 1013.0 | 6.6 | 104.2 | 93.8 | 2.759 | 1.051 | +1.709 |
| (-10, 2, 2) | 7.44 | 22735.9 | 2999.0 | 7.6 | 192.1 | 201.5 | 2.218 | 0.527 | +1.691 |
| (14, 8, 18) | 2.42 | 4573.6 | 736.0 | 6.2 | 87.7 | 76.9 | 2.248 | 0.663 | +1.585 |
| (-21, 3, 14) | 2.86 | 291.3 | 1066.0 | 0.3 | 88.0 | 89.5 | 8.008 | 6.463 | +1.546 |
| (1, 1, 3) | 16.91 | 238.8 | 82.0 | 2.9 | 18.8 | 12.2 | -4.304 | -2.766 | -1.538 |
| (27, 11, 4) | 2.22 | 3271.2 | 607.0 | 5.4 | 74.2 | 64.1 | 2.248 | 0.711 | +1.537 |
| (14, 2, 4) | 5.33 | 169128.4 | 15240.0 | 11.1 | 488.8 | 528.5 | 5.578 | 7.105 | -1.527 |
| (-9, 13, 10) | 2.39 | -189.1 | 223.0 | -0.8 | 28.9 | 22.8 | 2.668 | 1.223 | +1.445 |
| (3, 1, 3) | 14.68 | 2784.1 | 147.0 | 18.9 | 57.9 | 62.7 | -2.971 | -1.638 | -1.334 |
| (11, 3, 0) | 6.46 | 53805.3 | 3549.0 | 15.2 | 273.3 | 296.2 | 2.922 | 1.591 | +1.331 |
| (2, 8, 14) | 3.05 | 202.7 | 1701.0 | 0.1 | 75.7 | 66.8 | 2.898 | 1.585 | +1.313 |
| (9, 5, 4) | 5.27 | 139864.1 | 14390.0 | 9.7 | 397.9 | 437.4 | 1.338 | 2.631 | -1.293 |
| (3, 5, 1) | 6.80 | 165.3 | 128.0 | 1.3 | 12.7 | 12.2 | -3.902 | -2.626 | -1.276 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `true_model` | 12.69 | 0.9925 | 199.00140959097453 | 495.1435794009256 | 1.000 |
| `drifted_i` | 14.52 | 0.9916 | 199.15699829036046 | 430.76400237912554 | 1.000 |