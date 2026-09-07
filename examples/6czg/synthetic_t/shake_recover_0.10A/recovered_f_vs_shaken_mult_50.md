# Held-Out Log-Likelihood Comparison: `recovered_f` vs `shaken`

## 1. Executive Summary

- **Scored Test Reflections (|T|)**: 900 (held-out)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `1.2158` nats/refl
- **Model B NLL**: `1.1822` nats/refl
- **Difference (Gain $\Delta$)**: `+0.0336` nats/refl (`-0.0336` nats NLL reduction)
- **Estimated Log Bayes Factor**: `+30.25` nats
- **Uncertainty**: Bootstrap SE = `0.0106` (95% CI: `[+0.0122, +0.0535]`) | Naive SE = `0.0114` (Ratio: `0.93`x)
- **Win Fraction $P(d_h > 0)$**: `59.9%` (539 wins, 361 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `3.24e-09`
- **Wilcoxon Signed-Rank $p$-value**: `4.55e-05`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `-0.0338` | `+0.0002` | `-0.0336` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `-0.0364` | `+0.0028` | `-0.0336` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`shaken`) | Model B (`recovered_f`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0086` | `-0.0080` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `1.3823` | `1.3369` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.1664` | `-0.1547` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -0.8499 | -0.8686 | +0.0187 | 0.0578 | 48.8% | 21/22 |
| 1 | 6.10 | 4.86 | 46 | 0.7701 | 0.6768 | +0.0933 | 0.0490 | 65.2% | 30/16 |
| 2 | 4.83 | 4.21 | 45 | 0.5742 | 0.5536 | +0.0206 | 0.0794 | 57.8% | 26/19 |
| 3 | 4.20 | 3.81 | 46 | 0.9607 | 0.9499 | +0.0108 | 0.0455 | 41.3% | 19/27 |
| 4 | 3.80 | 3.53 | 46 | 1.2389 | 1.3087 | -0.0698 | 0.0449 | 47.8% | 22/24 |
| 5 | 3.53 | 3.32 | 46 | 1.3326 | 1.3653 | -0.0327 | 0.0505 | 52.2% | 24/22 |
| 6 | 3.31 | 3.16 | 44 | 1.1478 | 1.1754 | -0.0276 | 0.0618 | 56.8% | 25/19 |
| 7 | 3.14 | 3.02 | 45 | 1.4344 | 1.4418 | -0.0073 | 0.0393 | 51.1% | 23/22 |
| 8 | 3.01 | 2.89 | 46 | 0.9248 | 0.8193 | +0.1055 | 0.0671 | 60.9% | 28/18 |
| 9 | 2.88 | 2.79 | 43 | 1.0973 | 1.0478 | +0.0495 | 0.0412 | 58.1% | 25/18 |
| 10 | 2.79 | 2.70 | 45 | 0.9189 | 0.8279 | +0.0909 | 0.0444 | 71.1% | 32/13 |
| 11 | 2.70 | 2.62 | 45 | 1.5230 | 1.4665 | +0.0565 | 0.0610 | 57.8% | 26/19 |
| 12 | 2.62 | 2.55 | 44 | 1.3693 | 1.3176 | +0.0518 | 0.0598 | 68.2% | 30/14 |
| 13 | 2.55 | 2.49 | 47 | 1.5087 | 1.4114 | +0.0973 | 0.0354 | 68.1% | 32/15 |
| 14 | 2.49 | 2.43 | 43 | 1.2312 | 1.1733 | +0.0579 | 0.0435 | 67.4% | 29/14 |
| 15 | 2.43 | 2.38 | 43 | 1.8390 | 1.8072 | +0.0317 | 0.0406 | 60.5% | 26/17 |
| 16 | 2.38 | 2.33 | 44 | 1.4978 | 1.4651 | +0.0326 | 0.0485 | 65.9% | 29/15 |
| 17 | 2.32 | 2.28 | 47 | 1.8252 | 1.8080 | +0.0172 | 0.0208 | 55.3% | 26/21 |
| 18 | 2.28 | 2.24 | 48 | 1.9025 | 1.8716 | +0.0309 | 0.0412 | 64.6% | 31/17 |
| 19 | 2.24 | 2.20 | 44 | 1.9589 | 1.9145 | +0.0444 | 0.0243 | 79.5% | 35/9 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (-3, 5, 11) | 4.26 | 10860.2 | 1340.0 | 8.1 | 38.1 | 55.7 | 6.585 | 4.653 | +1.932 |
| (-10, 10, 13) | 2.66 | -51.7 | 1855.0 | -0.0 | 39.1 | 69.6 | 0.772 | 2.381 | -1.610 |
| (-13, 7, 14) | 2.92 | 12108.8 | 2125.0 | 5.7 | 76.5 | 95.6 | 2.318 | 0.713 | +1.604 |
| (-11, 5, 4) | 4.93 | 17104.2 | 4310.0 | 4.0 | 83.3 | 117.5 | 1.691 | 0.177 | +1.514 |
| (-14, 8, 7) | 3.30 | 8930.1 | 2630.0 | 3.4 | 68.3 | 47.3 | 0.667 | 2.095 | -1.428 |
| (-3, 3, 13) | 4.23 | 56569.5 | 9890.0 | 5.7 | 179.5 | 205.3 | 2.761 | 1.399 | +1.363 |
| (-13, 1, 10) | 4.40 | -8632.9 | 10290.0 | -0.8 | 84.4 | 126.9 | 1.521 | 2.837 | -1.316 |
| (13, 5, 13) | 3.29 | -4547.0 | 3390.0 | -1.3 | 74.2 | 48.1 | 3.077 | 1.802 | +1.276 |
| (18, 2, 18) | 2.66 | 8491.3 | 3470.0 | 2.4 | 137.5 | 114.3 | 2.782 | 1.519 | +1.263 |
| (-10, 4, 21) | 2.58 | 10569.9 | 2745.0 | 3.9 | 72.8 | 95.2 | 2.144 | 0.921 | +1.224 |
| (-20, 2, 14) | 2.98 | 21187.8 | 4475.0 | 4.7 | 90.5 | 108.3 | 3.552 | 2.356 | +1.195 |
| (-30, 2, 14) | 2.35 | -315.3 | 2100.0 | -0.2 | 63.6 | 40.5 | 2.479 | 1.298 | +1.181 |
| (1, 5, 19) | 2.86 | 1062.1 | 2145.0 | 0.5 | 76.8 | 49.9 | 1.668 | 0.516 | +1.152 |
| (-13, 5, 16) | 2.96 | 13006.9 | 2680.0 | 4.9 | 79.1 | 94.4 | 2.146 | 1.025 | +1.121 |
| (7, 7, 11) | 3.52 | -1654.7 | 11565.0 | -0.1 | 106.7 | 149.2 | 1.919 | 3.019 | -1.101 |
| (20, 2, 14) | 2.94 | 21827.8 | 7405.0 | 2.9 | 84.6 | 116.0 | 2.816 | 1.724 | +1.091 |
| (2, 14, 12) | 2.24 | 70.6 | 1250.0 | 0.1 | 26.6 | 47.8 | 0.805 | 1.889 | -1.084 |
| (-21, 3, 16) | 2.69 | -3624.8 | 2010.0 | -1.8 | 60.4 | 39.7 | 4.098 | 3.030 | +1.069 |
| (0, 4, 17) | 3.26 | -7789.4 | 4660.0 | -1.7 | 47.3 | 72.5 | 2.375 | 3.431 | -1.057 |
| (-14, 2, 12) | 3.78 | 28046.8 | 19035.0 | 1.5 | 250.5 | 272.9 | 2.647 | 3.656 | -1.008 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `shaken` | 74.71 | 0.6983 | 8.019453132674217 | 2.2685711711715126 | 1.000 |
| `recovered_f` | 73.98 | 0.7108 | 9.387353003246394 | 2.9075257851968495 | 1.000 |