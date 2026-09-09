# Held-Out Log-Likelihood Comparison: `recovered_f` vs `deposited`

## 1. Executive Summary

- **Scored Audit Reflections (|A|)**: 900 (audit set)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `-0.2994` nats/refl
- **Model B NLL**: `-0.0815` nats/refl
- **Difference (Gain $\Delta$)**: `-0.2179` nats/refl (`+0.2179` nats NLL reduction)
- **Estimated Log Bayes Factor**: `-196.11` nats
- **Uncertainty**: Bootstrap SE = `0.0243` (95% CI: `[-0.2657, -0.1713]`) | Naive SE = `0.0137` (Ratio: `1.77`x)
- **Win Fraction $P(d_h > 0)$**: `21.2%` (191 wins, 709 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `1.13e-70`
- **Wilcoxon Signed-Rank $p$-value**: `4.15e-62`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.3108` | `-0.0929` | `+0.2179` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.1696` | `+0.0483` | `+0.2179` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`deposited`) | Model B (`recovered_f`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0016` | `-0.0064` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `-0.1959` | `-0.0736` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.1036` | `-0.0079` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -1.4326 | -1.1260 | -0.3066 | 0.0288 | 4.7% | 2/41 |
| 1 | 6.10 | 4.86 | 46 | 0.0249 | 0.4466 | -0.4216 | 0.0592 | 4.3% | 2/44 |
| 2 | 4.83 | 4.21 | 45 | -0.6048 | -0.2612 | -0.3436 | 0.0433 | 2.2% | 1/44 |
| 3 | 4.20 | 3.81 | 46 | -0.1605 | 0.1135 | -0.2740 | 0.0631 | 17.4% | 8/38 |
| 4 | 3.80 | 3.53 | 46 | -0.1082 | 0.2889 | -0.3971 | 0.0660 | 8.7% | 4/42 |
| 5 | 3.53 | 3.32 | 46 | 0.2507 | 0.4743 | -0.2236 | 0.0659 | 26.1% | 12/34 |
| 6 | 3.31 | 3.16 | 44 | -0.4064 | -0.1173 | -0.2891 | 0.0475 | 13.6% | 6/38 |
| 7 | 3.14 | 3.02 | 45 | 0.1832 | 0.4108 | -0.2276 | 0.0854 | 26.7% | 12/33 |
| 8 | 3.01 | 2.89 | 46 | -0.4520 | -0.1741 | -0.2778 | 0.0522 | 15.2% | 7/39 |
| 9 | 2.88 | 2.79 | 43 | -0.4831 | -0.1732 | -0.3099 | 0.0532 | 11.6% | 5/38 |
| 10 | 2.79 | 2.70 | 45 | -0.6960 | -0.5170 | -0.1790 | 0.0432 | 13.3% | 6/39 |
| 11 | 2.70 | 2.62 | 45 | -0.3233 | -0.1431 | -0.1802 | 0.0490 | 26.7% | 12/33 |
| 12 | 2.62 | 2.55 | 44 | -0.3336 | -0.2305 | -0.1030 | 0.0525 | 38.6% | 17/27 |
| 13 | 2.55 | 2.49 | 47 | -0.2988 | -0.1384 | -0.1604 | 0.0557 | 25.5% | 12/35 |
| 14 | 2.49 | 2.43 | 43 | -0.6052 | -0.5274 | -0.0778 | 0.0457 | 37.2% | 16/27 |
| 15 | 2.43 | 2.38 | 43 | 0.0209 | 0.1342 | -0.1133 | 0.0720 | 27.9% | 12/31 |
| 16 | 2.38 | 2.33 | 44 | -0.2560 | -0.0746 | -0.1814 | 0.0769 | 36.4% | 16/28 |
| 17 | 2.32 | 2.28 | 47 | -0.0633 | -0.0275 | -0.0358 | 0.0733 | 29.8% | 14/33 |
| 18 | 2.28 | 2.24 | 48 | -0.2267 | -0.0742 | -0.1524 | 0.0529 | 27.1% | 13/35 |
| 19 | 2.24 | 2.20 | 44 | -0.1127 | -0.0125 | -0.1002 | 0.0735 | 31.8% | 14/30 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (2, 2, 14) | 4.11 | 118570.2 | 2282.2 | 52.0 | 400.5 | 436.1 | 5.355 | 7.583 | -2.228 |
| (30, 2, 14) | 2.32 | 10398.0 | 346.0 | 30.1 | 119.2 | 113.1 | 3.233 | 1.202 | +2.030 |
| (1, 7, 2) | 4.96 | 81943.9 | 2740.2 | 29.9 | 319.1 | 355.3 | 1.573 | 3.509 | -1.936 |
| (-12, 6, 14) | 3.11 | 10645.6 | 217.0 | 49.1 | 117.2 | 138.9 | 0.503 | 2.367 | -1.863 |
| (9, 3, 14) | 3.66 | 133223.3 | 2219.8 | 60.0 | 413.4 | 444.6 | 3.500 | 5.360 | -1.860 |
| (27, 7, 12) | 2.34 | 12356.0 | 198.2 | 62.3 | 127.1 | 117.6 | 2.612 | 0.866 | +1.746 |
| (16, 8, 17) | 2.43 | 115.4 | 96.8 | 1.2 | 12.8 | 26.6 | -1.634 | 0.071 | -1.705 |
| (20, 2, 14) | 2.94 | 7521.4 | 296.2 | 25.4 | 98.8 | 118.9 | 0.124 | 1.804 | -1.681 |
| (14, 2, 4) | 5.33 | 190733.7 | 3048.0 | 62.6 | 496.7 | 533.8 | 4.363 | 5.974 | -1.611 |
| (26, 4, 2) | 3.04 | 6688.0 | 88.6 | 75.5 | 94.2 | 112.5 | 0.243 | 1.801 | -1.557 |
| (-12, 4, 8) | 4.46 | 74425.5 | 2057.2 | 36.2 | 313.2 | 345.1 | 1.593 | 3.039 | -1.446 |
| (10, 12, 14) | 2.32 | 8878.8 | 170.4 | 52.1 | 108.8 | 102.6 | 2.264 | 0.873 | +1.391 |
| (27, 1, 2) | 3.11 | 66291.8 | 1708.0 | 38.8 | 291.3 | 291.4 | 3.697 | 2.311 | +1.386 |
| (35, 1, 9) | 2.26 | 275.5 | 71.4 | 3.9 | 22.7 | 30.3 | -0.591 | 0.743 | -1.334 |
| (21, 3, 2) | 3.79 | 57198.4 | 1912.8 | 29.9 | 271.5 | 297.0 | 1.459 | 2.777 | -1.318 |
| (-12, 8, 6) | 3.51 | 49710.6 | 849.8 | 58.5 | 256.6 | 278.7 | 2.809 | 4.123 | -1.314 |
| (23, 3, 8) | 3.17 | 4431.1 | 128.8 | 34.4 | 76.6 | 96.3 | -0.456 | 0.853 | -1.309 |
| (-27, 1, 2) | 3.12 | 26052.7 | 800.2 | 32.6 | 183.0 | 200.7 | 1.612 | 2.899 | -1.287 |
| (6, 14, 10) | 2.29 | 7872.6 | 293.0 | 26.9 | 101.6 | 92.3 | 2.010 | 0.726 | +1.284 |
| (0, 4, 12) | 4.33 | 80210.9 | 828.8 | 96.8 | 322.1 | 351.2 | 1.614 | 2.890 | -1.276 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `deposited` | 2.62 | 0.9996 | 199.18869153077665 | 744.5611865270802 | 1.000 |
| `recovered_f` | 9.27 | 0.9972 | 199.16826138527713 | 976.906380046205 | 1.000 |