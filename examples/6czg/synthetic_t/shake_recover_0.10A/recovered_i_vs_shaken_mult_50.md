# Held-Out Log-Likelihood Comparison: `recovered_i` vs `shaken`

## 1. Executive Summary

- **Scored Test Reflections (|T|)**: 900 (held-out)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `1.2158` nats/refl
- **Model B NLL**: `1.1886` nats/refl
- **Difference (Gain $\Delta$)**: `+0.0272` nats/refl (`-0.0272` nats NLL reduction)
- **Estimated Log Bayes Factor**: `+24.49` nats
- **Uncertainty**: Bootstrap SE = `0.0121` (95% CI: `[+0.0038, +0.0500]`) | Naive SE = `0.0121` (Ratio: `1.00`x)
- **Win Fraction $P(d_h > 0)$**: `58.9%` (530 wins, 370 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `1.08e-07`
- **Wilcoxon Signed-Rank $p$-value**: `2.71e-04`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `-0.0280` | `+0.0008` | `-0.0272` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `-0.0297` | `+0.0025` | `-0.0272` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`shaken`) | Model B (`recovered_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0086` | `-0.0076` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `1.3823` | `1.3215` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.1664` | `-0.1329` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -0.8499 | -0.7930 | -0.0570 | 0.0728 | 53.5% | 23/20 |
| 1 | 6.10 | 4.86 | 46 | 0.7701 | 0.7015 | +0.0686 | 0.0534 | 65.2% | 30/16 |
| 2 | 4.83 | 4.21 | 45 | 0.5742 | 0.5487 | +0.0255 | 0.0870 | 51.1% | 23/22 |
| 3 | 4.20 | 3.81 | 46 | 0.9607 | 0.9530 | +0.0077 | 0.0483 | 43.5% | 20/26 |
| 4 | 3.80 | 3.53 | 46 | 1.2389 | 1.3230 | -0.0841 | 0.0546 | 52.2% | 24/22 |
| 5 | 3.53 | 3.32 | 46 | 1.3326 | 1.3853 | -0.0528 | 0.0541 | 50.0% | 23/23 |
| 6 | 3.31 | 3.16 | 44 | 1.1478 | 1.1958 | -0.0480 | 0.0645 | 61.4% | 27/17 |
| 7 | 3.14 | 3.02 | 45 | 1.4344 | 1.4367 | -0.0023 | 0.0445 | 48.9% | 22/23 |
| 8 | 3.01 | 2.89 | 46 | 0.9248 | 0.8228 | +0.1020 | 0.0683 | 58.7% | 27/19 |
| 9 | 2.88 | 2.79 | 43 | 1.0973 | 1.0525 | +0.0448 | 0.0459 | 55.8% | 24/19 |
| 10 | 2.79 | 2.70 | 45 | 0.9189 | 0.8166 | +0.1022 | 0.0442 | 66.7% | 30/15 |
| 11 | 2.70 | 2.62 | 45 | 1.5230 | 1.4677 | +0.0553 | 0.0577 | 57.8% | 26/19 |
| 12 | 2.62 | 2.55 | 44 | 1.3693 | 1.3205 | +0.0488 | 0.0647 | 63.6% | 28/16 |
| 13 | 2.55 | 2.49 | 47 | 1.5087 | 1.4238 | +0.0849 | 0.0308 | 70.2% | 33/14 |
| 14 | 2.49 | 2.43 | 43 | 1.2312 | 1.1675 | +0.0637 | 0.0431 | 65.1% | 28/15 |
| 15 | 2.43 | 2.38 | 43 | 1.8390 | 1.7906 | +0.0483 | 0.0433 | 55.8% | 24/19 |
| 16 | 2.38 | 2.33 | 44 | 1.4978 | 1.4547 | +0.0431 | 0.0464 | 65.9% | 29/15 |
| 17 | 2.32 | 2.28 | 47 | 1.8252 | 1.8118 | +0.0134 | 0.0233 | 59.6% | 28/19 |
| 18 | 2.28 | 2.24 | 48 | 1.9025 | 1.8623 | +0.0401 | 0.0378 | 60.4% | 29/19 |
| 19 | 2.24 | 2.20 | 44 | 1.9589 | 1.9208 | +0.0381 | 0.0230 | 72.7% | 32/12 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (-3, 5, 11) | 4.26 | 10860.2 | 1340.0 | 8.1 | 38.1 | 56.3 | 6.585 | 4.376 | +2.209 |
| (-14, 8, 7) | 3.30 | 8930.1 | 2630.0 | 3.4 | 68.3 | 42.0 | 0.667 | 2.413 | -1.746 |
| (-6, 2, 2) | 10.39 | 21050.1 | 765.0 | 27.5 | 160.4 | 166.3 | -0.356 | 1.355 | -1.711 |
| (-13, 7, 14) | 2.92 | 12108.8 | 2125.0 | 5.7 | 76.5 | 97.2 | 2.318 | 0.620 | +1.698 |
| (-3, 3, 13) | 4.23 | 56569.5 | 9890.0 | 5.7 | 179.5 | 210.3 | 2.761 | 1.176 | +1.585 |
| (-14, 2, 12) | 3.78 | 28046.8 | 19035.0 | 1.5 | 250.5 | 284.5 | 2.647 | 4.206 | -1.558 |
| (-11, 5, 4) | 4.93 | 17104.2 | 4310.0 | 4.0 | 83.3 | 118.0 | 1.691 | 0.171 | +1.520 |
| (-13, 1, 10) | 4.40 | -8632.9 | 10290.0 | -0.8 | 84.4 | 131.6 | 1.521 | 3.019 | -1.498 |
| (-20, 2, 14) | 2.98 | 21187.8 | 4475.0 | 4.7 | 90.5 | 110.7 | 3.552 | 2.174 | +1.378 |
| (-10, 10, 13) | 2.66 | -51.7 | 1855.0 | -0.0 | 39.1 | 65.9 | 0.772 | 2.116 | -1.345 |
| (-1, 5, 4) | 6.37 | 22235.6 | 3590.0 | 6.2 | 172.7 | 183.3 | 0.199 | 1.467 | -1.269 |
| (13, 5, 13) | 3.29 | -4547.0 | 3390.0 | -1.3 | 74.2 | 48.7 | 3.077 | 1.824 | +1.254 |
| (-30, 2, 14) | 2.35 | -315.3 | 2100.0 | -0.2 | 63.6 | 38.6 | 2.479 | 1.241 | +1.238 |
| (-26, 8, 2) | 2.62 | 5215.5 | 1475.0 | 3.5 | 37.4 | 25.1 | 2.199 | 3.413 | -1.215 |
| (-10, 4, 21) | 2.58 | 10569.9 | 2745.0 | 3.9 | 72.8 | 94.9 | 2.144 | 0.940 | +1.205 |
| (1, 5, 19) | 2.86 | 1062.1 | 2145.0 | 0.5 | 76.8 | 50.7 | 1.668 | 0.534 | +1.134 |
| (-21, 3, 16) | 2.69 | -3624.8 | 2010.0 | -1.8 | 60.4 | 38.5 | 4.098 | 2.968 | +1.130 |
| (-13, 5, 16) | 2.96 | 13006.9 | 2680.0 | 4.9 | 79.1 | 93.3 | 2.146 | 1.098 | +1.048 |
| (-25, 5, 6) | 2.94 | 5437.0 | 1985.0 | 2.7 | 100.3 | 81.7 | 1.150 | 0.110 | +1.041 |
| (-7, 1, 22) | 2.64 | 1277.5 | 3895.0 | 0.3 | 101.0 | 76.4 | 2.716 | 1.693 | +1.023 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `shaken` | 74.71 | 0.6983 | 8.019453132674217 | 2.2685711711715126 | 1.000 |
| `recovered_i` | 74.50 | 0.7084 | 9.06805272058194 | 2.732101238260012 | 1.000 |