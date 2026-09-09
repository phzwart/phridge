# Held-Out Log-Likelihood Comparison: `recovered_i` vs `deposited`

## 1. Executive Summary

- **Scored Audit Reflections (|A|)**: 900 (audit set)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `1.8351` nats/refl
- **Model B NLL**: `1.8508` nats/refl
- **Difference (Gain $\Delta$)**: `-0.0157` nats/refl (`+0.0157` nats NLL reduction)
- **Estimated Log Bayes Factor**: `-14.12` nats
- **Uncertainty**: Bootstrap SE = `0.0048` (95% CI: `[-0.0255, -0.0071]`) | Naive SE = `0.0054` (Ratio: `0.88`x)
- **Win Fraction $P(d_h > 0)$**: `44.0%` (396 wins, 504 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `3.56e-04`
- **Wilcoxon Signed-Rank $p$-value**: `2.33e-03`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.0152` | `+0.0005` | `+0.0157` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.0136` | `+0.0021` | `+0.0157` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`deposited`) | Model B (`recovered_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0277` | `-0.0264` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `1.9787` | `1.9726` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.1437` | `-0.1218` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -0.3826 | -0.3715 | -0.0111 | 0.0301 | 46.5% | 20/23 |
| 1 | 6.10 | 4.86 | 46 | 1.4398 | 1.5215 | -0.0818 | 0.0412 | 39.1% | 18/28 |
| 2 | 4.83 | 4.21 | 45 | 0.9462 | 0.9662 | -0.0200 | 0.0282 | 35.6% | 16/29 |
| 3 | 4.20 | 3.81 | 46 | 1.8347 | 1.8149 | +0.0198 | 0.0196 | 56.5% | 26/20 |
| 4 | 3.80 | 3.53 | 46 | 1.7472 | 1.8018 | -0.0546 | 0.0229 | 37.0% | 17/29 |
| 5 | 3.53 | 3.32 | 46 | 1.9031 | 1.9279 | -0.0248 | 0.0211 | 39.1% | 18/28 |
| 6 | 3.31 | 3.16 | 44 | 1.4958 | 1.5053 | -0.0096 | 0.0161 | 43.2% | 19/25 |
| 7 | 3.14 | 3.02 | 45 | 2.2567 | 2.2761 | -0.0194 | 0.0216 | 37.8% | 17/28 |
| 8 | 3.01 | 2.89 | 46 | 1.6764 | 1.6898 | -0.0134 | 0.0259 | 41.3% | 19/27 |
| 9 | 2.88 | 2.79 | 43 | 1.8756 | 1.9064 | -0.0307 | 0.0281 | 41.9% | 18/25 |
| 10 | 2.79 | 2.70 | 45 | 1.8283 | 1.8156 | +0.0128 | 0.0278 | 51.1% | 23/22 |
| 11 | 2.70 | 2.62 | 45 | 1.8540 | 1.8638 | -0.0098 | 0.0221 | 40.0% | 18/27 |
| 12 | 2.62 | 2.55 | 44 | 1.9875 | 1.9953 | -0.0078 | 0.0227 | 40.9% | 18/26 |
| 13 | 2.55 | 2.49 | 47 | 1.8978 | 1.9131 | -0.0153 | 0.0139 | 40.4% | 19/28 |
| 14 | 2.49 | 2.43 | 43 | 2.2461 | 2.2451 | +0.0009 | 0.0252 | 41.9% | 18/25 |
| 15 | 2.43 | 2.38 | 43 | 2.3321 | 2.3375 | -0.0054 | 0.0224 | 53.5% | 23/20 |
| 16 | 2.38 | 2.33 | 44 | 2.2999 | 2.3162 | -0.0163 | 0.0234 | 50.0% | 22/22 |
| 17 | 2.32 | 2.28 | 47 | 2.3000 | 2.3098 | -0.0098 | 0.0152 | 46.8% | 22/25 |
| 18 | 2.28 | 2.24 | 48 | 2.4980 | 2.5067 | -0.0087 | 0.0151 | 56.2% | 27/21 |
| 19 | 2.24 | 2.20 | 44 | 2.5761 | 2.5829 | -0.0068 | 0.0172 | 40.9% | 18/26 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (2, 6, 0) | 5.83 | 23774.8 | 2760.0 | 8.6 | 148.2 | 134.4 | -0.452 | 0.956 | -1.408 |
| (10, 4, 5) | 5.42 | -6310.2 | 2510.0 | -2.5 | 52.4 | 68.8 | 3.484 | 4.487 | -1.003 |
| (3, 3, 11) | 4.83 | 8593.0 | 2820.0 | 3.0 | 25.7 | 46.8 | 2.501 | 1.620 | +0.880 |
| (13, 11, 0) | 2.88 | 3394.9 | 2530.0 | 1.3 | 62.3 | 82.2 | 0.327 | 1.178 | -0.851 |
| (30, 4, 4) | 2.65 | -2125.9 | 3130.0 | -0.7 | 45.1 | 62.0 | 1.688 | 2.453 | -0.766 |
| (-8, 6, 11) | 3.74 | -9112.5 | 5580.0 | -1.6 | 55.4 | 75.6 | 2.109 | 2.763 | -0.654 |
| (15, 7, 18) | 2.47 | 19932.4 | 6800.0 | 2.9 | 90.9 | 108.7 | 3.265 | 2.646 | +0.619 |
| (13, 11, 6) | 2.76 | 8140.9 | 2140.0 | 3.8 | 30.3 | 38.6 | 3.991 | 3.376 | +0.614 |
| (19, 9, 0) | 2.95 | -8204.7 | 10170.0 | -0.8 | 121.1 | 139.8 | 3.577 | 4.168 | -0.591 |
| (17, 3, 3) | 4.47 | -4442.5 | 3330.0 | -1.3 | 40.1 | 54.9 | 0.687 | 1.261 | -0.574 |
| (-14, 2, 1) | 5.72 | 775.4 | 2120.0 | 0.4 | 41.9 | 56.6 | -0.844 | -0.285 | -0.559 |
| (12, 6, 11) | 3.46 | 13414.3 | 6960.0 | 1.9 | 72.8 | 46.6 | 1.572 | 2.128 | -0.556 |
| (26, 4, 2) | 3.04 | 6063.2 | 4430.0 | 1.4 | 86.1 | 104.1 | 0.765 | 1.315 | -0.551 |
| (-20, 2, 14) | 2.98 | 24128.3 | 8950.0 | 2.7 | 130.1 | 110.7 | 1.582 | 2.120 | -0.538 |
| (-3, 13, 6) | 2.61 | 28315.0 | 6420.0 | 4.4 | 98.9 | 86.1 | 4.646 | 5.182 | -0.536 |
| (1, 5, 4) | 6.36 | -13937.4 | 7570.0 | -1.8 | 104.9 | 92.2 | 3.251 | 2.730 | +0.521 |
| (0, 8, 5) | 4.14 | 5533.8 | 3600.0 | 1.5 | 24.5 | 48.0 | 0.840 | 0.325 | +0.515 |
| (3, 7, 24) | 2.22 | -3013.7 | 3780.0 | -0.8 | 50.0 | 29.0 | 2.776 | 2.274 | +0.502 |
| (-12, 8, 5) | 3.58 | 6320.8 | 2780.0 | 2.3 | 56.9 | 41.4 | 0.095 | 0.594 | -0.499 |
| (-3, 5, 24) | 2.34 | -2034.7 | 4150.0 | -0.5 | 39.2 | 60.0 | 2.117 | 2.609 | -0.492 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `deposited` | 120.89 | 0.5751 | 7.3761627184459995 | 1.452032401210936 | 1.000 |
| `recovered_i` | 121.56 | 0.5772 | 7.030455137496274 | 1.3920065231717647 | 1.000 |