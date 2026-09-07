# Held-Out Log-Likelihood Comparison: `recovered_i` vs `recovered_f`

## 1. Executive Summary

- **Scored Test Reflections (|T|)**: 900 (held-out)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `0.1982` nats/refl
- **Model B NLL**: `0.1974` nats/refl
- **Difference (Gain $\Delta$)**: `+0.0008` nats/refl (`-0.0008` nats NLL reduction)
- **Estimated Log Bayes Factor**: `+0.68` nats
- **Uncertainty**: Bootstrap SE = `0.0027` (95% CI: `[-0.0045, +0.0058]`) | Naive SE = `0.0019` (Ratio: `1.40`x)
- **Win Fraction $P(d_h > 0)$**: `50.7%` (456 wins, 444 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `7.14e-01`
- **Wilcoxon Signed-Rank $p$-value**: `8.02e-01`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `-0.0010` | `+0.0003` | `-0.0008` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `-0.0011` | `+0.0004` | `-0.0008` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`recovered_f`) | Model B (`recovered_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0166` | `-0.0162` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `0.2112` | `0.2092` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.0130` | `-0.0117` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -1.1711 | -1.1466 | -0.0245 | 0.0074 | 25.6% | 11/32 |
| 1 | 6.10 | 4.86 | 46 | 0.2362 | 0.2603 | -0.0242 | 0.0089 | 32.6% | 15/31 |
| 2 | 4.83 | 4.21 | 45 | -0.2664 | -0.2553 | -0.0111 | 0.0071 | 35.6% | 16/29 |
| 3 | 4.20 | 3.81 | 46 | 0.0629 | 0.0645 | -0.0016 | 0.0071 | 47.8% | 22/24 |
| 4 | 3.80 | 3.53 | 46 | 0.3658 | 0.3664 | -0.0006 | 0.0089 | 43.5% | 20/26 |
| 5 | 3.53 | 3.32 | 46 | 0.5017 | 0.5054 | -0.0037 | 0.0102 | 45.7% | 21/25 |
| 6 | 3.31 | 3.16 | 44 | 0.0312 | 0.0335 | -0.0022 | 0.0080 | 43.2% | 19/25 |
| 7 | 3.14 | 3.02 | 45 | 0.6351 | 0.6434 | -0.0084 | 0.0072 | 44.4% | 20/25 |
| 8 | 3.01 | 2.89 | 46 | 0.0427 | 0.0432 | -0.0006 | 0.0072 | 63.0% | 29/17 |
| 9 | 2.88 | 2.79 | 43 | 0.3000 | 0.2830 | +0.0170 | 0.0121 | 58.1% | 25/18 |
| 10 | 2.79 | 2.70 | 45 | -0.1725 | -0.1728 | +0.0003 | 0.0072 | 64.4% | 29/16 |
| 11 | 2.70 | 2.62 | 45 | 0.2859 | 0.2877 | -0.0017 | 0.0088 | 37.8% | 17/28 |
| 12 | 2.62 | 2.55 | 44 | 0.2022 | 0.1893 | +0.0129 | 0.0070 | 65.9% | 29/15 |
| 13 | 2.55 | 2.49 | 47 | 0.2176 | 0.2053 | +0.0124 | 0.0055 | 66.0% | 31/16 |
| 14 | 2.49 | 2.43 | 43 | -0.0347 | -0.0512 | +0.0165 | 0.0076 | 76.7% | 33/10 |
| 15 | 2.43 | 2.38 | 43 | 0.4289 | 0.4116 | +0.0174 | 0.0101 | 44.2% | 19/24 |
| 16 | 2.38 | 2.33 | 44 | 0.4187 | 0.4230 | -0.0043 | 0.0076 | 61.4% | 27/17 |
| 17 | 2.32 | 2.28 | 47 | 0.6905 | 0.6953 | -0.0047 | 0.0077 | 46.8% | 22/25 |
| 18 | 2.28 | 2.24 | 48 | 0.5064 | 0.4931 | +0.0133 | 0.0082 | 64.6% | 31/17 |
| 19 | 2.24 | 2.20 | 44 | 0.5883 | 0.5741 | +0.0142 | 0.0091 | 45.5% | 20/24 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (26, 4, 8) | 2.82 | 1150.5 | 482.0 | 2.4 | 65.3 | 62.4 | 1.688 | 1.311 | +0.377 |
| (-37, 3, 6) | 2.20 | 2196.4 | 613.0 | 3.6 | 66.4 | 65.1 | 2.189 | 1.948 | +0.241 |
| (14, 2, 4) | 5.33 | 169128.4 | 15240.0 | 11.1 | 522.9 | 525.6 | 5.771 | 6.008 | -0.237 |
| (15, 9, 12) | 2.69 | 245.4 | 318.0 | 0.8 | 32.5 | 29.7 | -0.146 | -0.381 | +0.235 |
| (18, 4, 10) | 3.39 | 17385.3 | 1536.0 | 11.3 | 158.6 | 162.1 | 1.130 | 1.357 | -0.227 |
| (-11, 7, 14) | 3.01 | 4032.3 | 1634.0 | 2.5 | 125.0 | 126.4 | 4.509 | 4.726 | -0.218 |
| (-13, 9, 11) | 2.87 | 2555.7 | 447.0 | 5.7 | 74.4 | 72.3 | 0.968 | 0.756 | +0.211 |
| (33, 3, 4) | 2.47 | 6128.6 | 1152.0 | 5.3 | 67.7 | 71.2 | 0.952 | 0.744 | +0.208 |
| (-21, 5, 21) | 2.22 | -247.5 | 396.0 | -0.6 | 26.4 | 24.9 | 1.582 | 1.376 | +0.205 |
| (-23, 7, 15) | 2.40 | 7386.7 | 641.0 | 11.5 | 105.7 | 104.3 | 1.911 | 1.707 | +0.205 |
| (-16, 4, 12) | 3.38 | 66572.1 | 6116.0 | 10.9 | 320.3 | 322.1 | 4.158 | 4.359 | -0.201 |
| (29, 9, 3) | 2.33 | 998.3 | 301.0 | 3.3 | 8.9 | 7.6 | 1.386 | 1.578 | -0.191 |
| (-12, 8, 19) | 2.41 | 4025.3 | 782.0 | 5.1 | 88.7 | 87.6 | 2.465 | 2.281 | +0.184 |
| (-14, 4, 2) | 4.94 | 125597.7 | 12880.0 | 9.8 | 413.3 | 417.2 | 2.330 | 2.513 | -0.183 |
| (9, 13, 10) | 2.39 | 1874.5 | 377.0 | 5.0 | 55.2 | 53.3 | 0.566 | 0.387 | +0.179 |
| (-19, 1, 10) | 3.58 | 56417.3 | 4609.0 | 12.2 | 280.9 | 283.9 | 1.959 | 2.138 | -0.178 |
| (-2, 10, 8) | 3.18 | 36543.4 | 3584.0 | 10.2 | 231.1 | 229.1 | 2.190 | 2.017 | +0.173 |
| (0, 4, 12) | 4.33 | 75649.3 | 4144.0 | 18.3 | 338.5 | 341.9 | 2.760 | 2.932 | -0.172 |
| (15, 7, 18) | 2.47 | 6596.8 | 680.0 | 9.7 | 98.7 | 97.1 | 1.367 | 1.195 | +0.172 |
| (35, 1, 9) | 2.26 | -181.5 | 357.0 | -0.5 | 31.0 | 29.8 | 2.068 | 1.896 | +0.171 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `recovered_f` | 15.95 | 0.9903 | 199.11732842991213 | 510.6639779862454 | 1.000 |
| `recovered_i` | 15.89 | 0.9902 | 199.12194708666098 | 503.87368945984736 | 1.000 |