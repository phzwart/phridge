# Held-Out Log-Likelihood Comparison: `drifted_i` vs `true_model`

## 1. Executive Summary

- **Scored Test Reflections (|T|)**: 900 (held-out)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `-0.2994` nats/refl
- **Model B NLL**: `-0.1293` nats/refl
- **Difference (Gain $\Delta$)**: `-0.1701` nats/refl (`+0.1701` nats NLL reduction)
- **Estimated Log Bayes Factor**: `-153.11` nats
- **Uncertainty**: Bootstrap SE = `0.0442` (95% CI: `[-0.2646, -0.0941]`) | Naive SE = `0.0125` (Ratio: `3.54`x)
- **Win Fraction $P(d_h > 0)$**: `20.3%` (183 wins, 717 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `2.53e-75`
- **Wilcoxon Signed-Rank $p$-value**: `1.06e-63`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.2319` | `-0.0618` | `+0.1701` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.1413` | `+0.0288` | `+0.1701` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`true_model`) | Model B (`drifted_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0016` | `+0.0005` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `-0.1959` | `-0.0979` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.1036` | `-0.0314` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -1.4326 | -1.1585 | -0.2740 | 0.0252 | 2.3% | 1/42 |
| 1 | 6.10 | 4.86 | 46 | 0.0249 | 0.7058 | -0.6809 | 0.1114 | 2.2% | 1/45 |
| 2 | 4.83 | 4.21 | 45 | -0.6048 | -0.2616 | -0.3432 | 0.0359 | 0.0% | 0/45 |
| 3 | 4.20 | 3.81 | 46 | -0.1605 | 0.1797 | -0.3402 | 0.0524 | 2.2% | 1/45 |
| 4 | 3.80 | 3.53 | 46 | -0.1082 | 0.3215 | -0.4296 | 0.0443 | 0.0% | 0/46 |
| 5 | 3.53 | 3.32 | 46 | 0.2507 | 0.5932 | -0.3425 | 0.0461 | 4.3% | 2/44 |
| 6 | 3.31 | 3.16 | 44 | -0.4064 | -0.1819 | -0.2245 | 0.0261 | 6.8% | 3/41 |
| 7 | 3.14 | 3.02 | 45 | 0.1832 | 0.3756 | -0.1924 | 0.0423 | 15.6% | 7/38 |
| 8 | 3.01 | 2.89 | 46 | -0.4520 | -0.2611 | -0.1909 | 0.0297 | 17.4% | 8/38 |
| 9 | 2.88 | 2.79 | 43 | -0.4831 | -0.2580 | -0.2252 | 0.0412 | 11.6% | 5/38 |
| 10 | 2.79 | 2.70 | 45 | -0.6960 | -0.5360 | -0.1601 | 0.0240 | 13.3% | 6/39 |
| 11 | 2.70 | 2.62 | 45 | -0.3233 | -0.1791 | -0.1442 | 0.0297 | 15.6% | 7/38 |
| 12 | 2.62 | 2.55 | 44 | -0.3336 | -0.2650 | -0.0686 | 0.0286 | 31.8% | 14/30 |
| 13 | 2.55 | 2.49 | 47 | -0.2988 | -0.2401 | -0.0586 | 0.0383 | 29.8% | 14/33 |
| 14 | 2.49 | 2.43 | 43 | -0.6052 | -0.5769 | -0.0283 | 0.0265 | 34.9% | 15/28 |
| 15 | 2.43 | 2.38 | 43 | 0.0209 | -0.0449 | +0.0658 | 0.0475 | 44.2% | 19/24 |
| 16 | 2.38 | 2.33 | 44 | -0.2560 | -0.2768 | +0.0208 | 0.0541 | 52.3% | 23/21 |
| 17 | 2.32 | 2.28 | 47 | -0.0633 | -0.1511 | +0.0878 | 0.0671 | 38.3% | 18/29 |
| 18 | 2.28 | 2.24 | 48 | -0.2267 | -0.2946 | +0.0679 | 0.0369 | 43.8% | 21/27 |
| 19 | 2.24 | 2.20 | 44 | -0.1127 | -0.1847 | +0.0719 | 0.0570 | 40.9% | 18/26 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (-13, 1, 8) | 4.90 | 285707.9 | 3760.0 | 76.0 | 610.9 | 654.0 | 7.362 | 10.946 | -3.584 |
| (14, 2, 4) | 5.33 | 190733.7 | 3048.0 | 62.6 | 496.7 | 533.3 | 4.363 | 6.837 | -2.474 |
| (9, 5, 4) | 5.27 | 130392.0 | 2878.0 | 45.3 | 404.4 | 440.6 | 2.454 | 4.841 | -2.388 |
| (1, 7, 2) | 4.96 | 81943.9 | 2740.2 | 29.9 | 319.1 | 353.6 | 1.573 | 3.793 | -2.220 |
| (30, 2, 14) | 2.32 | 10398.0 | 346.0 | 30.1 | 119.2 | 111.5 | 3.233 | 1.071 | +2.162 |
| (2, 2, 14) | 4.11 | 118570.2 | 2282.2 | 52.0 | 400.5 | 428.9 | 5.355 | 7.459 | -2.104 |
| (27, 7, 12) | 2.34 | 12356.0 | 198.2 | 62.3 | 127.1 | 115.6 | 2.612 | 0.768 | +1.844 |
| (14, 4, 0) | 5.00 | 55363.9 | 1068.0 | 51.8 | 266.1 | 295.2 | 1.323 | 3.087 | -1.763 |
| (-16, 4, 12) | 3.38 | 79184.0 | 1223.2 | 64.7 | 321.2 | 341.2 | 4.184 | 5.742 | -1.558 |
| (-14, 2, 12) | 3.78 | 72282.0 | 761.4 | 94.9 | 304.7 | 328.8 | 1.856 | 3.343 | -1.487 |
| (17, 5, 0) | 4.08 | 57697.2 | 1527.4 | 37.8 | 271.7 | 294.0 | 1.861 | 3.299 | -1.438 |
| (-8, 8, 6) | 3.78 | 25365.8 | 1180.4 | 21.5 | 187.2 | 209.4 | 0.872 | 2.214 | -1.342 |
| (6, 14, 10) | 2.29 | 7872.6 | 293.0 | 26.9 | 101.6 | 92.4 | 2.010 | 0.670 | +1.340 |
| (-9, 5, 4) | 5.30 | 74702.7 | 2093.4 | 35.7 | 310.3 | 335.5 | 1.777 | 3.102 | -1.324 |
| (-14, 4, 2) | 4.94 | 125171.6 | 2576.0 | 48.6 | 405.2 | 430.3 | 3.556 | 4.829 | -1.273 |
| (5, 9, 4) | 3.70 | 52438.1 | 1488.4 | 35.2 | 258.4 | 279.8 | 1.298 | 2.550 | -1.252 |
| (0, 4, 12) | 4.33 | 80210.9 | 828.8 | 96.8 | 322.1 | 347.3 | 1.614 | 2.858 | -1.244 |
| (10, 12, 14) | 2.32 | 8878.8 | 170.4 | 52.1 | 108.8 | 104.3 | 2.264 | 1.056 | +1.208 |
| (-9, 11, 8) | 2.82 | 16574.2 | 342.4 | 48.4 | 143.0 | 155.7 | 1.048 | 2.249 | -1.201 |
| (2, 10, 4) | 3.42 | 30373.2 | 681.0 | 44.6 | 201.7 | 217.2 | 1.968 | 3.061 | -1.092 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `true_model` | 2.62 | 0.9996 | 199.18869153077665 | 744.5611865270802 | 1.000 |
| `drifted_i` | 6.47 | 0.9988 | 199.18270108859792 | 830.6985045818757 | 1.000 |