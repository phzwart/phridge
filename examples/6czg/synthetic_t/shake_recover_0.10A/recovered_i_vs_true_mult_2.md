# Held-Out Log-Likelihood Comparison: `recovered_i` vs `deposited`

## 1. Executive Summary

- **Scored Audit Reflections (|A|)**: 900 (audit set)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `-0.2994` nats/refl
- **Model B NLL**: `-0.0807` nats/refl
- **Difference (Gain $\Delta$)**: `-0.2187` nats/refl (`+0.2187` nats NLL reduction)
- **Estimated Log Bayes Factor**: `-196.83` nats
- **Uncertainty**: Bootstrap SE = `0.0245` (95% CI: `[-0.2669, -0.1720]`) | Naive SE = `0.0137` (Ratio: `1.79`x)
- **Win Fraction $P(d_h > 0)$**: `21.2%` (191 wins, 709 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `1.13e-70`
- **Wilcoxon Signed-Rank $p$-value**: `3.15e-62`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.3123` | `-0.0936` | `+0.2187` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.1700` | `+0.0487` | `+0.2187` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`deposited`) | Model B (`recovered_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0016` | `-0.0063` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `-0.1959` | `-0.0727` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.1036` | `-0.0080` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -1.4326 | -1.1230 | -0.3096 | 0.0293 | 4.7% | 2/41 |
| 1 | 6.10 | 4.86 | 46 | 0.0249 | 0.4493 | -0.4244 | 0.0603 | 4.3% | 2/44 |
| 2 | 4.83 | 4.21 | 45 | -0.6048 | -0.2625 | -0.3423 | 0.0428 | 2.2% | 1/44 |
| 3 | 4.20 | 3.81 | 46 | -0.1605 | 0.1178 | -0.2783 | 0.0640 | 17.4% | 8/38 |
| 4 | 3.80 | 3.53 | 46 | -0.1082 | 0.2905 | -0.3987 | 0.0658 | 8.7% | 4/42 |
| 5 | 3.53 | 3.32 | 46 | 0.2507 | 0.4754 | -0.2246 | 0.0664 | 26.1% | 12/34 |
| 6 | 3.31 | 3.16 | 44 | -0.4064 | -0.1146 | -0.2919 | 0.0476 | 13.6% | 6/38 |
| 7 | 3.14 | 3.02 | 45 | 0.1832 | 0.4122 | -0.2290 | 0.0854 | 26.7% | 12/33 |
| 8 | 3.01 | 2.89 | 46 | -0.4520 | -0.1733 | -0.2787 | 0.0524 | 15.2% | 7/39 |
| 9 | 2.88 | 2.79 | 43 | -0.4831 | -0.1727 | -0.3104 | 0.0532 | 11.6% | 5/38 |
| 10 | 2.79 | 2.70 | 45 | -0.6960 | -0.5166 | -0.1794 | 0.0431 | 13.3% | 6/39 |
| 11 | 2.70 | 2.62 | 45 | -0.3233 | -0.1430 | -0.1803 | 0.0489 | 26.7% | 12/33 |
| 12 | 2.62 | 2.55 | 44 | -0.3336 | -0.2299 | -0.1037 | 0.0524 | 38.6% | 17/27 |
| 13 | 2.55 | 2.49 | 47 | -0.2988 | -0.1378 | -0.1610 | 0.0557 | 25.5% | 12/35 |
| 14 | 2.49 | 2.43 | 43 | -0.6052 | -0.5264 | -0.0788 | 0.0459 | 37.2% | 16/27 |
| 15 | 2.43 | 2.38 | 43 | 0.0209 | 0.1313 | -0.1104 | 0.0716 | 27.9% | 12/31 |
| 16 | 2.38 | 2.33 | 44 | -0.2560 | -0.0738 | -0.1822 | 0.0769 | 36.4% | 16/28 |
| 17 | 2.32 | 2.28 | 47 | -0.0633 | -0.0282 | -0.0351 | 0.0731 | 29.8% | 14/33 |
| 18 | 2.28 | 2.24 | 48 | -0.2267 | -0.0749 | -0.1517 | 0.0528 | 27.1% | 13/35 |
| 19 | 2.24 | 2.20 | 44 | -0.1127 | -0.0126 | -0.1001 | 0.0733 | 31.8% | 14/30 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (2, 2, 14) | 4.11 | 118570.2 | 2282.2 | 52.0 | 400.5 | 436.4 | 5.355 | 7.630 | -2.276 |
| (30, 2, 14) | 2.32 | 10398.0 | 346.0 | 30.1 | 119.2 | 113.2 | 3.233 | 1.207 | +2.025 |
| (1, 7, 2) | 4.96 | 81943.9 | 2740.2 | 29.9 | 319.1 | 355.4 | 1.573 | 3.515 | -1.942 |
| (-12, 6, 14) | 3.11 | 10645.6 | 217.0 | 49.1 | 117.2 | 139.1 | 0.503 | 2.386 | -1.883 |
| (9, 3, 14) | 3.66 | 133223.3 | 2219.8 | 60.0 | 413.4 | 444.2 | 3.500 | 5.294 | -1.794 |
| (27, 7, 12) | 2.34 | 12356.0 | 198.2 | 62.3 | 127.1 | 117.5 | 2.612 | 0.861 | +1.751 |
| (16, 8, 17) | 2.43 | 115.4 | 96.8 | 1.2 | 12.8 | 26.6 | -1.634 | 0.067 | -1.702 |
| (20, 2, 14) | 2.94 | 7521.4 | 296.2 | 25.4 | 98.8 | 119.0 | 0.124 | 1.812 | -1.688 |
| (14, 2, 4) | 5.33 | 190733.7 | 3048.0 | 62.6 | 496.7 | 534.4 | 4.363 | 6.039 | -1.676 |
| (26, 4, 2) | 3.04 | 6688.0 | 88.6 | 75.5 | 94.2 | 112.4 | 0.243 | 1.780 | -1.537 |
| (-12, 4, 8) | 4.46 | 74425.5 | 2057.2 | 36.2 | 313.2 | 344.9 | 1.593 | 3.014 | -1.421 |
| (10, 12, 14) | 2.32 | 8878.8 | 170.4 | 52.1 | 108.8 | 102.4 | 2.264 | 0.862 | +1.402 |
| (27, 1, 2) | 3.11 | 66291.8 | 1708.0 | 38.8 | 291.3 | 291.4 | 3.697 | 2.309 | +1.388 |
| (21, 3, 2) | 3.79 | 57198.4 | 1912.8 | 29.9 | 271.5 | 297.5 | 1.459 | 2.810 | -1.351 |
| (-12, 8, 6) | 3.51 | 49710.6 | 849.8 | 58.5 | 256.6 | 278.8 | 2.809 | 4.141 | -1.332 |
| (35, 1, 9) | 2.26 | 275.5 | 71.4 | 3.9 | 22.7 | 30.3 | -0.591 | 0.734 | -1.325 |
| (-13, 1, 8) | 4.90 | 285707.9 | 3760.0 | 76.0 | 610.9 | 649.8 | 7.362 | 8.685 | -1.324 |
| (23, 3, 8) | 3.17 | 4431.1 | 128.8 | 34.4 | 76.6 | 96.4 | -0.456 | 0.857 | -1.314 |
| (-27, 1, 2) | 3.12 | 26052.7 | 800.2 | 32.6 | 183.0 | 200.8 | 1.612 | 2.913 | -1.301 |
| (-18, 4, 10) | 3.43 | 54968.0 | 1286.6 | 42.7 | 264.3 | 285.7 | 2.377 | 3.673 | -1.297 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `deposited` | 2.62 | 0.9996 | 199.18869153077665 | 744.5611865270802 | 1.000 |
| `recovered_i` | 9.32 | 0.9972 | 199.16823180421954 | 977.3151111708368 | 1.000 |