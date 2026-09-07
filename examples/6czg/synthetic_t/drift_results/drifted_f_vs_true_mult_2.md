# Held-Out Log-Likelihood Comparison: `drifted_f` vs `true_model`

## 1. Executive Summary

- **Scored Test Reflections (|T|)**: 900 (held-out)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `-0.2994` nats/refl
- **Model B NLL**: `-0.1294` nats/refl
- **Difference (Gain $\Delta$)**: `-0.1700` nats/refl (`+0.1700` nats NLL reduction)
- **Estimated Log Bayes Factor**: `-153.02` nats
- **Uncertainty**: Bootstrap SE = `0.0434` (95% CI: `[-0.2624, -0.0951]`) | Naive SE = `0.0124` (Ratio: `3.51`x)
- **Win Fraction $P(d_h > 0)$**: `20.0%` (180 wins, 720 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `4.07e-77`
- **Wilcoxon Signed-Rank $p$-value**: `1.03e-64`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.2315` | `-0.0614` | `+0.1700` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.1412` | `+0.0288` | `+0.1700` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`true_model`) | Model B (`drifted_f`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0016` | `+0.0007` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `-0.1959` | `-0.0997` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.1036` | `-0.0297` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -1.4326 | -1.1584 | -0.2742 | 0.0255 | 2.3% | 1/42 |
| 1 | 6.10 | 4.86 | 46 | 0.0249 | 0.6962 | -0.6713 | 0.1108 | 0.0% | 0/46 |
| 2 | 4.83 | 4.21 | 45 | -0.6048 | -0.2682 | -0.3366 | 0.0332 | 0.0% | 0/45 |
| 3 | 4.20 | 3.81 | 46 | -0.1605 | 0.1748 | -0.3353 | 0.0503 | 2.2% | 1/45 |
| 4 | 3.80 | 3.53 | 46 | -0.1082 | 0.3123 | -0.4204 | 0.0426 | 0.0% | 0/46 |
| 5 | 3.53 | 3.32 | 46 | 0.2507 | 0.5946 | -0.3439 | 0.0472 | 2.2% | 1/45 |
| 6 | 3.31 | 3.16 | 44 | -0.4064 | -0.1828 | -0.2236 | 0.0259 | 6.8% | 3/41 |
| 7 | 3.14 | 3.02 | 45 | 0.1832 | 0.3767 | -0.1934 | 0.0425 | 15.6% | 7/38 |
| 8 | 3.01 | 2.89 | 46 | -0.4520 | -0.2621 | -0.1899 | 0.0292 | 17.4% | 8/38 |
| 9 | 2.88 | 2.79 | 43 | -0.4831 | -0.2578 | -0.2253 | 0.0414 | 11.6% | 5/38 |
| 10 | 2.79 | 2.70 | 45 | -0.6960 | -0.5336 | -0.1624 | 0.0238 | 13.3% | 6/39 |
| 11 | 2.70 | 2.62 | 45 | -0.3233 | -0.1768 | -0.1465 | 0.0300 | 15.6% | 7/38 |
| 12 | 2.62 | 2.55 | 44 | -0.3336 | -0.2600 | -0.0736 | 0.0282 | 31.8% | 14/30 |
| 13 | 2.55 | 2.49 | 47 | -0.2988 | -0.2384 | -0.0603 | 0.0385 | 29.8% | 14/33 |
| 14 | 2.49 | 2.43 | 43 | -0.6052 | -0.5746 | -0.0307 | 0.0267 | 34.9% | 15/28 |
| 15 | 2.43 | 2.38 | 43 | 0.0209 | -0.0412 | +0.0621 | 0.0472 | 46.5% | 20/23 |
| 16 | 2.38 | 2.33 | 44 | -0.2560 | -0.2759 | +0.0199 | 0.0540 | 52.3% | 23/21 |
| 17 | 2.32 | 2.28 | 47 | -0.0633 | -0.1468 | +0.0835 | 0.0672 | 36.2% | 17/30 |
| 18 | 2.28 | 2.24 | 48 | -0.2267 | -0.2906 | +0.0640 | 0.0366 | 41.7% | 20/28 |
| 19 | 2.24 | 2.20 | 44 | -0.1127 | -0.1830 | +0.0702 | 0.0569 | 40.9% | 18/26 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (-13, 1, 8) | 4.90 | 285707.9 | 3760.0 | 76.0 | 610.9 | 654.1 | 7.362 | 10.980 | -3.619 |
| (14, 2, 4) | 5.33 | 190733.7 | 3048.0 | 62.6 | 496.7 | 533.1 | 4.363 | 6.818 | -2.455 |
| (9, 5, 4) | 5.27 | 130392.0 | 2878.0 | 45.3 | 404.4 | 439.7 | 2.454 | 4.753 | -2.299 |
| (1, 7, 2) | 4.96 | 81943.9 | 2740.2 | 29.9 | 319.1 | 353.4 | 1.573 | 3.781 | -2.208 |
| (30, 2, 14) | 2.32 | 10398.0 | 346.0 | 30.1 | 119.2 | 111.5 | 3.233 | 1.068 | +2.165 |
| (2, 2, 14) | 4.11 | 118570.2 | 2282.2 | 52.0 | 400.5 | 428.4 | 5.355 | 7.395 | -2.040 |
| (27, 7, 12) | 2.34 | 12356.0 | 198.2 | 62.3 | 127.1 | 115.6 | 2.612 | 0.769 | +1.843 |
| (14, 4, 0) | 5.00 | 55363.9 | 1068.0 | 51.8 | 266.1 | 295.5 | 1.323 | 3.119 | -1.796 |
| (-16, 4, 12) | 3.38 | 79184.0 | 1223.2 | 64.7 | 321.2 | 341.3 | 4.184 | 5.769 | -1.585 |
| (-14, 2, 12) | 3.78 | 72282.0 | 761.4 | 94.9 | 304.7 | 328.3 | 1.856 | 3.292 | -1.437 |
| (6, 14, 10) | 2.29 | 7872.6 | 293.0 | 26.9 | 101.6 | 92.6 | 2.010 | 0.673 | +1.337 |
| (17, 5, 0) | 4.08 | 57697.2 | 1527.4 | 37.8 | 271.7 | 292.9 | 1.861 | 3.184 | -1.323 |
| (-8, 8, 6) | 3.78 | 25365.8 | 1180.4 | 21.5 | 187.2 | 208.7 | 0.872 | 2.146 | -1.274 |
| (-9, 5, 4) | 5.30 | 74702.7 | 2093.4 | 35.7 | 310.3 | 334.8 | 1.777 | 3.042 | -1.265 |
| (5, 9, 4) | 3.70 | 52438.1 | 1488.4 | 35.2 | 258.4 | 279.8 | 1.298 | 2.541 | -1.242 |
| (-9, 11, 8) | 2.82 | 16574.2 | 342.4 | 48.4 | 143.0 | 155.9 | 1.048 | 2.283 | -1.235 |
| (-14, 4, 2) | 4.94 | 125171.6 | 2576.0 | 48.6 | 405.2 | 429.8 | 3.556 | 4.781 | -1.225 |
| (10, 12, 14) | 2.32 | 8878.8 | 170.4 | 52.1 | 108.8 | 104.5 | 2.264 | 1.080 | +1.184 |
| (0, 4, 12) | 4.33 | 80210.9 | 828.8 | 96.8 | 322.1 | 345.9 | 1.614 | 2.742 | -1.128 |
| (27, 1, 2) | 3.11 | 66291.8 | 1708.0 | 38.8 | 291.3 | 306.7 | 3.697 | 4.806 | -1.109 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `true_model` | 2.62 | 0.9996 | 199.18869153077665 | 744.5611865270802 | 1.000 |
| `drifted_f` | 6.37 | 0.9988 | 199.182903763072 | 827.8462302632518 | 1.000 |