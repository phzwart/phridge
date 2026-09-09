# Held-Out Log-Likelihood Comparison: `recovered_i` vs `shaken`

## 1. Executive Summary

- **Scored Audit Reflections (|A|)**: 900 (audit set)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `0.2943` nats/refl
- **Model B NLL**: `0.1974` nats/refl
- **Difference (Gain $\Delta$)**: `+0.0969` nats/refl (`-0.0969` nats NLL reduction)
- **Estimated Log Bayes Factor**: `+87.21` nats
- **Uncertainty**: Bootstrap SE = `0.0418` (95% CI: `[+0.0148, +0.1748]`) | Naive SE = `0.0232` (Ratio: `1.80`x)
- **Win Fraction $P(d_h > 0)$**: `59.2%` (533 wins, 367 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `3.49e-08`
- **Wilcoxon Signed-Rank $p$-value**: `5.50e-07`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `-0.0634` | `-0.0335` | `-0.0969` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `-0.1336` | `+0.0368` | `-0.0969` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`shaken`) | Model B (`recovered_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0105` | `-0.0162` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `0.3990` | `0.2092` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.1047` | `-0.0117` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -1.3180 | -1.1466 | -0.1713 | 0.1997 | 27.9% | 12/31 |
| 1 | 6.10 | 4.86 | 46 | -0.1035 | 0.2603 | -0.3639 | 0.0977 | 23.9% | 11/35 |
| 2 | 4.83 | 4.21 | 45 | -0.4149 | -0.2553 | -0.1596 | 0.0797 | 28.9% | 13/32 |
| 3 | 4.20 | 3.81 | 46 | 0.1762 | 0.0645 | +0.1117 | 0.0834 | 50.0% | 23/23 |
| 4 | 3.80 | 3.53 | 46 | 0.2713 | 0.3664 | -0.0950 | 0.1043 | 41.3% | 19/27 |
| 5 | 3.53 | 3.32 | 46 | 0.4871 | 0.5054 | -0.0183 | 0.1362 | 58.7% | 27/19 |
| 6 | 3.31 | 3.16 | 44 | 0.1006 | 0.0335 | +0.0671 | 0.0775 | 59.1% | 26/18 |
| 7 | 3.14 | 3.02 | 45 | 0.5850 | 0.6434 | -0.0584 | 0.0915 | 44.4% | 20/25 |
| 8 | 3.01 | 2.89 | 46 | 0.1586 | 0.0432 | +0.1153 | 0.0768 | 65.2% | 30/16 |
| 9 | 2.88 | 2.79 | 43 | 0.4398 | 0.2830 | +0.1568 | 0.1157 | 67.4% | 29/14 |
| 10 | 2.79 | 2.70 | 45 | 0.0017 | -0.1728 | +0.1744 | 0.0745 | 66.7% | 30/15 |
| 11 | 2.70 | 2.62 | 45 | 0.5205 | 0.2877 | +0.2328 | 0.0979 | 71.1% | 32/13 |
| 12 | 2.62 | 2.55 | 44 | 0.2318 | 0.1893 | +0.0425 | 0.0899 | 59.1% | 26/18 |
| 13 | 2.55 | 2.49 | 47 | 0.5757 | 0.2053 | +0.3705 | 0.0958 | 76.6% | 36/11 |
| 14 | 2.49 | 2.43 | 43 | 0.2848 | -0.0512 | +0.3360 | 0.0708 | 83.7% | 36/7 |
| 15 | 2.43 | 2.38 | 43 | 0.6981 | 0.4116 | +0.2865 | 0.0994 | 72.1% | 31/12 |
| 16 | 2.38 | 2.33 | 44 | 0.6137 | 0.4230 | +0.1907 | 0.1096 | 72.7% | 32/12 |
| 17 | 2.32 | 2.28 | 47 | 0.8722 | 0.6953 | +0.1770 | 0.0581 | 63.8% | 30/17 |
| 18 | 2.28 | 2.24 | 48 | 0.7146 | 0.4931 | +0.2215 | 0.0675 | 79.2% | 38/10 |
| 19 | 2.24 | 2.20 | 44 | 0.9032 | 0.5741 | +0.3291 | 0.0849 | 72.7% | 32/12 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (3, 1, 0) | 22.09 | 19917.5 | 1815.0 | 11.0 | 187.2 | 176.4 | 6.319 | -0.291 | +6.610 |
| (-18, 8, 15) | 2.52 | 1245.2 | 577.0 | 2.2 | 79.8 | 45.6 | 3.860 | 0.151 | +3.709 |
| (14, 2, 4) | 5.33 | 169128.4 | 15240.0 | 11.1 | 465.2 | 525.6 | 2.465 | 6.008 | -3.544 |
| (20, 4, 7) | 3.47 | 937.3 | 355.0 | 2.6 | 79.4 | 36.1 | 2.338 | -1.022 | +3.360 |
| (-30, 2, 14) | 2.35 | 1173.9 | 420.0 | 2.8 | 68.7 | 41.0 | 2.993 | 0.085 | +2.908 |
| (-16, 4, 12) | 3.38 | 66572.1 | 6116.0 | 10.9 | 285.9 | 322.1 | 1.581 | 4.359 | -2.779 |
| (-14, 6, 18) | 2.62 | 16819.3 | 1930.0 | 8.7 | 144.4 | 168.2 | 1.123 | 3.818 | -2.695 |
| (4, 2, 0) | 13.57 | 11317.2 | 444.0 | 25.5 | 140.1 | 135.3 | 2.019 | -0.632 | +2.650 |
| (-21, 1, 21) | 2.33 | -36.3 | 366.0 | -0.1 | 25.2 | 41.5 | 0.454 | 3.063 | -2.609 |
| (15, 9, 12) | 2.69 | 245.4 | 318.0 | 0.8 | 54.6 | 29.7 | 2.069 | -0.381 | +2.450 |
| (0, 4, 12) | 4.33 | 75649.3 | 4144.0 | 18.3 | 291.4 | 341.9 | 0.499 | 2.932 | -2.432 |
| (-5, 9, 1) | 3.82 | 3738.7 | 915.0 | 4.1 | 110.6 | 80.7 | 2.331 | -0.026 | +2.357 |
| (14, 2, 12) | 3.73 | 4634.3 | 2974.0 | 1.6 | 153.3 | 127.5 | 4.154 | 1.832 | +2.321 |
| (13, 5, 13) | 3.29 | 386.0 | 678.0 | 0.6 | 80.5 | 54.7 | 2.545 | 0.259 | +2.286 |
| (-21, 3, 14) | 2.86 | 291.3 | 1066.0 | 0.3 | 81.5 | 90.0 | 3.752 | 5.947 | -2.195 |
| (1, 5, 19) | 2.86 | 2163.6 | 429.0 | 5.0 | 83.1 | 55.5 | 2.020 | -0.174 | +2.194 |
| (-9, 13, 10) | 2.39 | -189.1 | 223.0 | -0.8 | 39.6 | 21.7 | 3.158 | 1.001 | +2.157 |
| (-13, 1, 8) | 4.90 | 305862.6 | 18800.0 | 16.3 | 590.8 | 642.0 | 1.933 | 4.067 | -2.133 |
| (11, 3, 0) | 6.46 | 53805.3 | 3549.0 | 15.2 | 250.2 | 304.1 | -0.097 | 1.991 | -2.088 |
| (35, 1, 9) | 2.26 | -181.5 | 357.0 | -0.5 | 46.6 | 29.8 | 3.830 | 1.896 | +1.934 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `shaken` | 22.46 | 0.9831 | 199.30483080247237 | 788.7820064945549 | 1.000 |
| `recovered_i` | 15.89 | 0.9902 | 199.12194708666098 | 503.87368945984736 | 1.000 |