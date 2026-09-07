# Held-Out Log-Likelihood Comparison: `recovered_f` vs `shaken`

## 1. Executive Summary

- **Scored Test Reflections (|T|)**: 900 (held-out)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `0.6422` nats/refl
- **Model B NLL**: `0.3143` nats/refl
- **Difference (Gain $\Delta$)**: `+0.3279` nats/refl (`-0.3279` nats NLL reduction)
- **Estimated Log Bayes Factor**: `+295.09` nats
- **Uncertainty**: Bootstrap SE = `0.0206` (95% CI: `[+0.2894, +0.3691]`) | Naive SE = `0.0261` (Ratio: `0.79`x)
- **Win Fraction $P(d_h > 0)$**: `75.4%` (679 wins, 221 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `7.98e-55`
- **Wilcoxon Signed-Rank $p$-value**: `2.54e-49`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `-0.2502` | `-0.0777` | `-0.3279` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `-0.4606` | `+0.1328` | `-0.3279` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`shaken`) | Model B (`recovered_f`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0065` | `-0.0097` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `0.7667` | `0.3412` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.1245` | `-0.0269` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -1.0854 | -1.2594 | +0.1740 | 0.1586 | 67.4% | 29/14 |
| 1 | 6.10 | 4.86 | 46 | 0.3888 | 0.0412 | +0.3476 | 0.1299 | 73.9% | 34/12 |
| 2 | 4.83 | 4.21 | 45 | -0.0259 | -0.3526 | +0.3267 | 0.1067 | 82.2% | 37/8 |
| 3 | 4.20 | 3.81 | 46 | 0.5451 | 0.0417 | +0.5034 | 0.0906 | 87.0% | 40/6 |
| 4 | 3.80 | 3.53 | 46 | 0.5356 | 0.3388 | +0.1968 | 0.1099 | 65.2% | 30/16 |
| 5 | 3.53 | 3.32 | 46 | 1.0393 | 0.5481 | +0.4912 | 0.1829 | 69.6% | 32/14 |
| 6 | 3.31 | 3.16 | 44 | 0.3115 | 0.0914 | +0.2201 | 0.0646 | 72.7% | 32/12 |
| 7 | 3.14 | 3.02 | 45 | 0.9821 | 0.7296 | +0.2526 | 0.1363 | 71.1% | 32/13 |
| 8 | 3.01 | 2.89 | 46 | 0.5123 | 0.1582 | +0.3540 | 0.1189 | 82.6% | 38/8 |
| 9 | 2.88 | 2.79 | 43 | 0.7267 | 0.4974 | +0.2294 | 0.1463 | 72.1% | 31/12 |
| 10 | 2.79 | 2.70 | 45 | 0.3431 | 0.0352 | +0.3079 | 0.0917 | 77.8% | 35/10 |
| 11 | 2.70 | 2.62 | 45 | 0.9004 | 0.4575 | +0.4429 | 0.0922 | 84.4% | 38/7 |
| 12 | 2.62 | 2.55 | 44 | 0.6770 | 0.3418 | +0.3352 | 0.0712 | 75.0% | 33/11 |
| 13 | 2.55 | 2.49 | 47 | 0.8481 | 0.4640 | +0.3842 | 0.0966 | 78.7% | 37/10 |
| 14 | 2.49 | 2.43 | 43 | 0.6123 | 0.2781 | +0.3341 | 0.0943 | 72.1% | 31/12 |
| 15 | 2.43 | 2.38 | 43 | 0.9353 | 0.6142 | +0.3211 | 0.0916 | 86.0% | 37/6 |
| 16 | 2.38 | 2.33 | 44 | 1.0786 | 0.6812 | +0.3974 | 0.1643 | 72.7% | 32/12 |
| 17 | 2.32 | 2.28 | 47 | 1.1683 | 0.9164 | +0.2519 | 0.0936 | 66.0% | 31/16 |
| 18 | 2.28 | 2.24 | 48 | 1.0179 | 0.7542 | +0.2637 | 0.0906 | 75.0% | 36/12 |
| 19 | 2.24 | 2.20 | 44 | 1.2336 | 0.8206 | +0.4130 | 0.1221 | 77.3% | 34/10 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (20, 4, 7) | 3.47 | 937.3 | 355.0 | 2.6 | 125.9 | 38.0 | 4.332 | -0.912 | +5.244 |
| (-20, 2, 14) | 2.98 | 16625.3 | 895.0 | 18.6 | 50.9 | 112.4 | 4.964 | 1.068 | +3.897 |
| (-21, 3, 14) | 2.86 | 291.3 | 1066.0 | 0.3 | 70.8 | 95.3 | 1.391 | 5.090 | -3.699 |
| (-30, 2, 14) | 2.35 | 1173.9 | 420.0 | 2.8 | 94.2 | 39.9 | 3.726 | 0.176 | +3.550 |
| (27, 7, 12) | 2.34 | 12200.4 | 991.0 | 12.3 | 74.3 | 116.7 | 4.699 | 1.263 | +3.436 |
| (-25, 11, 8) | 2.23 | 8381.6 | 816.0 | 10.3 | 55.3 | 90.1 | 4.885 | 1.523 | +3.362 |
| (-21, 1, 21) | 2.33 | -36.3 | 366.0 | -0.1 | 17.9 | 50.5 | 0.271 | 3.376 | -3.105 |
| (27, 1, 2) | 3.11 | 70087.2 | 8540.0 | 8.2 | 210.5 | 268.9 | 4.769 | 1.721 | +3.048 |
| (-5, 3, 4) | 8.14 | 27472.1 | 5651.0 | 4.9 | 219.4 | 241.2 | 0.957 | 3.986 | -3.029 |
| (-1, 1, 3) | 17.00 | -15.7 | 299.0 | -0.1 | 56.1 | 30.2 | 0.908 | -2.100 | +3.007 |
| (-29, 9, 3) | 2.33 | -123.3 | 279.0 | -0.4 | 62.7 | 21.7 | 3.476 | 0.509 | +2.967 |
| (-24, 2, 2) | 3.45 | 58537.8 | 7701.0 | 7.6 | 180.0 | 242.2 | 4.262 | 1.391 | +2.871 |
| (21, 3, 2) | 3.79 | 46681.9 | 9564.0 | 4.9 | 249.5 | 301.0 | 1.165 | 3.994 | -2.829 |
| (14, 2, 4) | 5.33 | 169128.4 | 15240.0 | 11.1 | 405.2 | 498.3 | 1.736 | 4.531 | -2.795 |
| (0, 4, 12) | 4.33 | 75649.3 | 4144.0 | 18.3 | 257.2 | 342.2 | 1.126 | 3.820 | -2.695 |
| (12, 8, 6) | 3.49 | 38563.8 | 4211.0 | 9.2 | 138.9 | 215.3 | 3.741 | 1.064 | +2.677 |
| (3, 1, 0) | 22.09 | 19917.5 | 1815.0 | 11.0 | 199.0 | 194.5 | 5.400 | 8.069 | -2.669 |
| (-13, 5, 7) | 4.20 | 9864.3 | 2053.0 | 4.8 | 174.8 | 107.7 | 2.576 | 0.013 | +2.563 |
| (14, 4, 0) | 5.00 | 68539.0 | 5340.0 | 12.8 | 204.0 | 270.2 | 3.272 | 0.732 | +2.540 |
| (-18, 8, 15) | 2.52 | 1245.2 | 577.0 | 2.2 | 101.5 | 59.3 | 3.362 | 0.998 | +2.365 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `shaken` | 38.31 | 0.9549 | 35.25211804487967 | 104.57997355472382 | 1.000 |
| `recovered_f` | 21.34 | 0.9840 | 199.05123663228872 | 685.766229474762 | 1.000 |