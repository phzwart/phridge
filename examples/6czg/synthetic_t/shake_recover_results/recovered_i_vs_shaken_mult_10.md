# Held-Out Log-Likelihood Comparison: `recovered_i` vs `shaken`

## 1. Executive Summary

- **Scored Test Reflections (|T|)**: 900 (held-out)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `0.6422` nats/refl
- **Model B NLL**: `0.3142` nats/refl
- **Difference (Gain $\Delta$)**: `+0.3280` nats/refl (`-0.3280` nats NLL reduction)
- **Estimated Log Bayes Factor**: `+295.18` nats
- **Uncertainty**: Bootstrap SE = `0.0207` (95% CI: `[+0.2884, +0.3698]`) | Naive SE = `0.0263` (Ratio: `0.79`x)
- **Win Fraction $P(d_h > 0)$**: `76.1%` (685 wins, 215 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `8.44e-58`
- **Wilcoxon Signed-Rank $p$-value**: `2.20e-49`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `-0.2504` | `-0.0776` | `-0.3280` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `-0.4610` | `+0.1330` | `-0.3280` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`shaken`) | Model B (`recovered_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0065` | `-0.0098` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `0.7667` | `0.3396` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.1245` | `-0.0254` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -1.0854 | -1.2593 | +0.1739 | 0.1634 | 67.4% | 29/14 |
| 1 | 6.10 | 4.86 | 46 | 0.3888 | 0.0608 | +0.3280 | 0.1335 | 76.1% | 35/11 |
| 2 | 4.83 | 4.21 | 45 | -0.0259 | -0.3442 | +0.3183 | 0.1098 | 82.2% | 37/8 |
| 3 | 4.20 | 3.81 | 46 | 0.5451 | 0.0396 | +0.5055 | 0.0903 | 89.1% | 41/5 |
| 4 | 3.80 | 3.53 | 46 | 0.5356 | 0.3496 | +0.1860 | 0.1115 | 60.9% | 28/18 |
| 5 | 3.53 | 3.32 | 46 | 1.0393 | 0.5553 | +0.4840 | 0.1830 | 69.6% | 32/14 |
| 6 | 3.31 | 3.16 | 44 | 0.3115 | 0.0887 | +0.2228 | 0.0650 | 72.7% | 32/12 |
| 7 | 3.14 | 3.02 | 45 | 0.9821 | 0.7364 | +0.2458 | 0.1379 | 73.3% | 33/12 |
| 8 | 3.01 | 2.89 | 46 | 0.5123 | 0.1528 | +0.3595 | 0.1197 | 82.6% | 38/8 |
| 9 | 2.88 | 2.79 | 43 | 0.7267 | 0.4901 | +0.2366 | 0.1442 | 72.1% | 31/12 |
| 10 | 2.79 | 2.70 | 45 | 0.3431 | 0.0373 | +0.3059 | 0.0919 | 77.8% | 35/10 |
| 11 | 2.70 | 2.62 | 45 | 0.9004 | 0.4580 | +0.4424 | 0.0942 | 86.7% | 39/6 |
| 12 | 2.62 | 2.55 | 44 | 0.6770 | 0.3388 | +0.3383 | 0.0713 | 75.0% | 33/11 |
| 13 | 2.55 | 2.49 | 47 | 0.8481 | 0.4591 | +0.3890 | 0.0965 | 83.0% | 39/8 |
| 14 | 2.49 | 2.43 | 43 | 0.6123 | 0.2746 | +0.3377 | 0.0938 | 72.1% | 31/12 |
| 15 | 2.43 | 2.38 | 43 | 0.9353 | 0.6016 | +0.3338 | 0.0893 | 88.4% | 38/5 |
| 16 | 2.38 | 2.33 | 44 | 1.0786 | 0.6753 | +0.4033 | 0.1630 | 72.7% | 32/12 |
| 17 | 2.32 | 2.28 | 47 | 1.1683 | 0.9192 | +0.2491 | 0.0947 | 68.1% | 32/15 |
| 18 | 2.28 | 2.24 | 48 | 1.0179 | 0.7473 | +0.2706 | 0.0916 | 75.0% | 36/12 |
| 19 | 2.24 | 2.20 | 44 | 1.2336 | 0.8132 | +0.4204 | 0.1221 | 77.3% | 34/10 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (20, 4, 7) | 3.47 | 937.3 | 355.0 | 2.6 | 125.9 | 37.9 | 4.332 | -0.915 | +5.248 |
| (-20, 2, 14) | 2.98 | 16625.3 | 895.0 | 18.6 | 50.9 | 113.4 | 4.964 | 1.018 | +3.947 |
| (-21, 3, 14) | 2.86 | 291.3 | 1066.0 | 0.3 | 70.8 | 94.9 | 1.391 | 5.055 | -3.664 |
| (-30, 2, 14) | 2.35 | 1173.9 | 420.0 | 2.8 | 94.2 | 40.2 | 3.726 | 0.186 | +3.540 |
| (27, 7, 12) | 2.34 | 12200.4 | 991.0 | 12.3 | 74.3 | 117.0 | 4.699 | 1.261 | +3.438 |
| (-25, 11, 8) | 2.23 | 8381.6 | 816.0 | 10.3 | 55.3 | 90.8 | 4.885 | 1.487 | +3.399 |
| (-1, 1, 3) | 17.00 | -15.7 | 299.0 | -0.1 | 56.1 | 28.4 | 0.908 | -2.332 | +3.240 |
| (-5, 3, 4) | 8.14 | 27472.1 | 5651.0 | 4.9 | 219.4 | 242.7 | 0.957 | 4.152 | -3.195 |
| (27, 1, 2) | 3.11 | 70087.2 | 8540.0 | 8.2 | 210.5 | 268.4 | 4.769 | 1.728 | +3.041 |
| (-21, 1, 21) | 2.33 | -36.3 | 366.0 | -0.1 | 17.9 | 50.1 | 0.271 | 3.301 | -3.030 |
| (14, 2, 4) | 5.33 | 169128.4 | 15240.0 | 11.1 | 405.2 | 500.7 | 1.736 | 4.702 | -2.966 |
| (-29, 9, 3) | 2.33 | -123.3 | 279.0 | -0.4 | 62.7 | 21.8 | 3.476 | 0.521 | +2.956 |
| (21, 3, 2) | 3.79 | 46681.9 | 9564.0 | 4.9 | 249.5 | 302.4 | 1.165 | 4.092 | -2.927 |
| (0, 4, 12) | 4.33 | 75649.3 | 4144.0 | 18.3 | 257.2 | 344.0 | 1.126 | 4.018 | -2.893 |
| (-24, 2, 2) | 3.45 | 58537.8 | 7701.0 | 7.6 | 180.0 | 243.0 | 4.262 | 1.380 | +2.882 |
| (12, 8, 6) | 3.49 | 38563.8 | 4211.0 | 9.2 | 138.9 | 216.9 | 3.741 | 1.096 | +2.645 |
| (-13, 5, 7) | 4.20 | 9864.3 | 2053.0 | 4.8 | 174.8 | 107.6 | 2.576 | 0.015 | +2.561 |
| (14, 4, 0) | 5.00 | 68539.0 | 5340.0 | 12.8 | 204.0 | 271.5 | 3.272 | 0.740 | +2.533 |
| (3, 1, 0) | 22.09 | 19917.5 | 1815.0 | 11.0 | 199.0 | 194.1 | 5.400 | 7.915 | -2.515 |
| (-11, 7, 14) | 3.01 | 4032.3 | 1634.0 | 2.5 | 127.5 | 131.8 | 2.019 | 4.395 | -2.376 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `shaken` | 38.31 | 0.9549 | 35.25211804487967 | 104.57997355472382 | 1.000 |
| `recovered_i` | 21.30 | 0.9838 | 199.04809889248483 | 688.2128744615293 | 1.000 |