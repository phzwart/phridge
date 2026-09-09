# Held-Out Log-Likelihood Comparison: `drifted_f` vs `true_model`

## 1. Executive Summary

- **Scored Audit Reflections (|A|)**: 900 (audit set)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `0.0322` nats/refl
- **Model B NLL**: `0.1586` nats/refl
- **Difference (Gain $\Delta$)**: `-0.1264` nats/refl (`+0.1264` nats NLL reduction)
- **Estimated Log Bayes Factor**: `-113.75` nats
- **Uncertainty**: Bootstrap SE = `0.0403` (95% CI: `[-0.2124, -0.0534]`) | Naive SE = `0.0143` (Ratio: `2.83`x)
- **Win Fraction $P(d_h > 0)$**: `28.0%` (252 wins, 648 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `6.65e-41`
- **Wilcoxon Signed-Rank $p$-value**: `5.67e-38`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.2127` | `-0.0863` | `+0.1264` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.0938` | `+0.0326` | `+0.1264` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`true_model`) | Model B (`drifted_f`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0071` | `-0.0110` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `0.1228` | `0.1789` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.0907` | `-0.0203` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -1.5441 | -1.1287 | -0.4154 | 0.1690 | 16.3% | 7/36 |
| 1 | 6.10 | 4.86 | 46 | -0.1528 | 0.3694 | -0.5222 | 0.0562 | 0.0% | 0/46 |
| 2 | 4.83 | 4.21 | 45 | -0.6179 | -0.2652 | -0.3527 | 0.0462 | 8.9% | 4/41 |
| 3 | 4.20 | 3.81 | 46 | -0.1780 | 0.1064 | -0.2844 | 0.0283 | 4.3% | 2/44 |
| 4 | 3.80 | 3.53 | 46 | 0.0820 | 0.3735 | -0.2915 | 0.0449 | 13.0% | 6/40 |
| 5 | 3.53 | 3.32 | 46 | 0.3107 | 0.5448 | -0.2341 | 0.0379 | 6.5% | 3/43 |
| 6 | 3.31 | 3.16 | 44 | -0.1737 | -0.0268 | -0.1470 | 0.0301 | 22.7% | 10/34 |
| 7 | 3.14 | 3.02 | 45 | 0.3932 | 0.5434 | -0.1503 | 0.0510 | 20.0% | 9/36 |
| 8 | 3.01 | 2.89 | 46 | -0.2065 | -0.0830 | -0.1235 | 0.0341 | 21.7% | 10/36 |
| 9 | 2.88 | 2.79 | 43 | 0.1173 | 0.2528 | -0.1355 | 0.0538 | 20.9% | 9/34 |
| 10 | 2.79 | 2.70 | 45 | -0.3108 | -0.2448 | -0.0660 | 0.0289 | 33.3% | 15/30 |
| 11 | 2.70 | 2.62 | 45 | 0.1873 | 0.2999 | -0.1126 | 0.0409 | 28.9% | 13/32 |
| 12 | 2.62 | 2.55 | 44 | 0.1010 | 0.1182 | -0.0172 | 0.0417 | 38.6% | 17/27 |
| 13 | 2.55 | 2.49 | 47 | 0.0881 | 0.1506 | -0.0625 | 0.0336 | 29.8% | 14/33 |
| 14 | 2.49 | 2.43 | 43 | -0.1044 | -0.1014 | -0.0030 | 0.0284 | 37.2% | 16/27 |
| 15 | 2.43 | 2.38 | 43 | 0.4917 | 0.3157 | +0.1759 | 0.0751 | 41.9% | 18/25 |
| 16 | 2.38 | 2.33 | 44 | 0.3501 | 0.2916 | +0.0584 | 0.0398 | 50.0% | 22/22 |
| 17 | 2.32 | 2.28 | 47 | 0.7179 | 0.5840 | +0.1339 | 0.0644 | 57.4% | 27/20 |
| 18 | 2.28 | 2.24 | 48 | 0.3901 | 0.3936 | -0.0036 | 0.0413 | 60.4% | 29/19 |
| 19 | 2.24 | 2.20 | 44 | 0.6173 | 0.5809 | +0.0364 | 0.0642 | 47.7% | 21/23 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (3, 1, 0) | 22.09 | 19917.5 | 1815.0 | 11.0 | 176.3 | 167.0 | 3.700 | -0.468 | +4.167 |
| (-5, 13, 13) | 2.32 | -560.0 | 338.0 | -1.7 | 38.6 | 30.7 | 7.011 | 4.332 | +2.679 |
| (-5, 3, 4) | 8.14 | 27472.1 | 5651.0 | 4.9 | 232.4 | 242.4 | 3.792 | 1.537 | +2.255 |
| (-13, 1, 8) | 4.90 | 305862.6 | 18800.0 | 16.3 | 601.8 | 647.7 | 2.986 | 5.148 | -2.162 |
| (0, 4, 1) | 8.73 | 4.5 | 131.0 | 0.0 | 6.5 | 10.7 | -4.092 | -2.141 | -1.951 |
| (-1, 1, 3) | 17.00 | -15.7 | 299.0 | -0.1 | 9.8 | 12.2 | -3.940 | -2.033 | -1.907 |
| (-27, 11, 4) | 2.22 | 1793.3 | 497.0 | 3.6 | 68.9 | 64.4 | 4.779 | 2.950 | +1.829 |
| (-19, 1, 21) | 2.41 | 6678.8 | 1013.0 | 6.6 | 104.2 | 93.0 | 2.759 | 0.990 | +1.770 |
| (-10, 2, 2) | 7.44 | 22735.9 | 2999.0 | 7.6 | 192.1 | 199.7 | 2.218 | 0.458 | +1.760 |
| (14, 8, 18) | 2.42 | 4573.6 | 736.0 | 6.2 | 87.7 | 76.1 | 2.248 | 0.609 | +1.640 |
| (27, 11, 4) | 2.22 | 3271.2 | 607.0 | 5.4 | 74.2 | 62.7 | 2.248 | 0.629 | +1.619 |
| (1, 1, 3) | 16.91 | 238.8 | 82.0 | 2.9 | 18.8 | 14.9 | -4.304 | -2.741 | -1.563 |
| (-9, 13, 10) | 2.39 | -189.1 | 223.0 | -0.8 | 28.9 | 22.4 | 2.668 | 1.169 | +1.499 |
| (-21, 3, 14) | 2.86 | 291.3 | 1066.0 | 0.3 | 88.0 | 89.8 | 8.008 | 6.601 | +1.408 |
| (3, 1, 3) | 14.68 | 2784.1 | 147.0 | 18.9 | 57.9 | 66.4 | -2.971 | -1.578 | -1.393 |
| (2, 8, 14) | 3.05 | 202.7 | 1701.0 | 0.1 | 75.7 | 65.7 | 2.898 | 1.506 | +1.392 |
| (14, 2, 4) | 5.33 | 169128.4 | 15240.0 | 11.1 | 488.8 | 526.4 | 5.578 | 6.955 | -1.377 |
| (3, 5, 1) | 6.80 | 165.3 | 128.0 | 1.3 | 12.7 | 16.0 | -3.902 | -2.578 | -1.324 |
| (2, 2, 5) | 9.59 | 1147.6 | 133.0 | 8.6 | 35.3 | 47.3 | -3.293 | -1.971 | -1.322 |
| (11, 3, 0) | 6.46 | 53805.3 | 3549.0 | 15.2 | 273.3 | 297.2 | 2.922 | 1.638 | +1.284 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `true_model` | 12.69 | 0.9925 | 199.00140959097453 | 495.1435794009256 | 1.000 |
| `drifted_f` | 14.56 | 0.9917 | 199.15490824739788 | 432.57052806047244 | 1.000 |