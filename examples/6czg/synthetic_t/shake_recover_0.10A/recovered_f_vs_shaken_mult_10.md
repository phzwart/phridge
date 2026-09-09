# Held-Out Log-Likelihood Comparison: `recovered_f` vs `shaken`

## 1. Executive Summary

- **Scored Audit Reflections (|A|)**: 900 (audit set)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `0.2943` nats/refl
- **Model B NLL**: `0.1982` nats/refl
- **Difference (Gain $\Delta$)**: `+0.0961` nats/refl (`-0.0961` nats NLL reduction)
- **Estimated Log Bayes Factor**: `+86.53` nats
- **Uncertainty**: Bootstrap SE = `0.0397` (95% CI: `[+0.0186, +0.1689]`) | Naive SE = `0.0230` (Ratio: `1.73`x)
- **Win Fraction $P(d_h > 0)$**: `60.1%` (541 wins, 359 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `1.42e-09`
- **Wilcoxon Signed-Rank $p$-value**: `5.86e-07`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `-0.0645` | `-0.0316` | `-0.0961` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `-0.1307` | `+0.0346` | `-0.0961` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`shaken`) | Model B (`recovered_f`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0105` | `-0.0166` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `0.3990` | `0.2112` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.1047` | `-0.0130` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -1.3180 | -1.1711 | -0.1468 | 0.1968 | 30.2% | 13/30 |
| 1 | 6.10 | 4.86 | 46 | -0.1035 | 0.2362 | -0.3397 | 0.0930 | 23.9% | 11/35 |
| 2 | 4.83 | 4.21 | 45 | -0.4149 | -0.2664 | -0.1485 | 0.0792 | 28.9% | 13/32 |
| 3 | 4.20 | 3.81 | 46 | 0.1762 | 0.0629 | +0.1134 | 0.0816 | 47.8% | 22/24 |
| 4 | 3.80 | 3.53 | 46 | 0.2713 | 0.3658 | -0.0944 | 0.1036 | 45.7% | 21/25 |
| 5 | 3.53 | 3.32 | 46 | 0.4871 | 0.5017 | -0.0146 | 0.1336 | 58.7% | 27/19 |
| 6 | 3.31 | 3.16 | 44 | 0.1006 | 0.0312 | +0.0694 | 0.0784 | 59.1% | 26/18 |
| 7 | 3.14 | 3.02 | 45 | 0.5850 | 0.6351 | -0.0501 | 0.0898 | 44.4% | 20/25 |
| 8 | 3.01 | 2.89 | 46 | 0.1586 | 0.0427 | +0.1159 | 0.0761 | 63.0% | 29/17 |
| 9 | 2.88 | 2.79 | 43 | 0.4398 | 0.3000 | +0.1399 | 0.1191 | 65.1% | 28/15 |
| 10 | 2.79 | 2.70 | 45 | 0.0017 | -0.1725 | +0.1741 | 0.0746 | 71.1% | 32/13 |
| 11 | 2.70 | 2.62 | 45 | 0.5205 | 0.2859 | +0.2345 | 0.0951 | 73.3% | 33/12 |
| 12 | 2.62 | 2.55 | 44 | 0.2318 | 0.2022 | +0.0296 | 0.0893 | 59.1% | 26/18 |
| 13 | 2.55 | 2.49 | 47 | 0.5757 | 0.2176 | +0.3581 | 0.0958 | 76.6% | 36/11 |
| 14 | 2.49 | 2.43 | 43 | 0.2848 | -0.0347 | +0.3195 | 0.0719 | 79.1% | 34/9 |
| 15 | 2.43 | 2.38 | 43 | 0.6981 | 0.4289 | +0.2691 | 0.1023 | 72.1% | 31/12 |
| 16 | 2.38 | 2.33 | 44 | 0.6137 | 0.4187 | +0.1950 | 0.1074 | 75.0% | 33/11 |
| 17 | 2.32 | 2.28 | 47 | 0.8722 | 0.6905 | +0.1817 | 0.0562 | 68.1% | 32/15 |
| 18 | 2.28 | 2.24 | 48 | 0.7146 | 0.5064 | +0.2082 | 0.0669 | 85.4% | 41/7 |
| 19 | 2.24 | 2.20 | 44 | 0.9032 | 0.5883 | +0.3149 | 0.0874 | 75.0% | 33/11 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (3, 1, 0) | 22.09 | 19917.5 | 1815.0 | 11.0 | 187.2 | 179.2 | 6.319 | -0.226 | +6.545 |
| (-18, 8, 15) | 2.52 | 1245.2 | 577.0 | 2.2 | 79.8 | 46.5 | 3.860 | 0.208 | +3.652 |
| (20, 4, 7) | 3.47 | 937.3 | 355.0 | 2.6 | 79.4 | 37.7 | 2.338 | -0.974 | +3.312 |
| (14, 2, 4) | 5.33 | 169128.4 | 15240.0 | 11.1 | 465.2 | 522.9 | 2.465 | 5.771 | -3.306 |
| (-30, 2, 14) | 2.35 | 1173.9 | 420.0 | 2.8 | 68.7 | 40.6 | 2.993 | 0.070 | +2.923 |
| (-14, 6, 18) | 2.62 | 16819.3 | 1930.0 | 8.7 | 144.4 | 168.3 | 1.123 | 3.779 | -2.656 |
| (4, 2, 0) | 13.57 | 11317.2 | 444.0 | 25.5 | 140.1 | 137.0 | 2.019 | -0.606 | +2.625 |
| (-16, 4, 12) | 3.38 | 66572.1 | 6116.0 | 10.9 | 285.9 | 320.3 | 1.581 | 4.158 | -2.577 |
| (-21, 1, 21) | 2.33 | -36.3 | 366.0 | -0.1 | 25.2 | 41.5 | 0.454 | 2.987 | -2.532 |
| (-5, 9, 1) | 3.82 | 3738.7 | 915.0 | 4.1 | 110.6 | 81.4 | 2.331 | 0.004 | +2.326 |
| (14, 2, 12) | 3.73 | 4634.3 | 2974.0 | 1.6 | 153.3 | 127.3 | 4.154 | 1.846 | +2.307 |
| (13, 5, 13) | 3.29 | 386.0 | 678.0 | 0.6 | 80.5 | 54.9 | 2.545 | 0.284 | +2.261 |
| (0, 4, 12) | 4.33 | 75649.3 | 4144.0 | 18.3 | 291.4 | 338.5 | 0.499 | 2.760 | -2.261 |
| (-21, 3, 14) | 2.86 | 291.3 | 1066.0 | 0.3 | 81.5 | 90.2 | 3.752 | 6.000 | -2.248 |
| (1, 5, 19) | 2.86 | 2163.6 | 429.0 | 5.0 | 83.1 | 54.5 | 2.020 | -0.217 | +2.238 |
| (15, 9, 12) | 2.69 | 245.4 | 318.0 | 0.8 | 54.6 | 32.5 | 2.069 | -0.146 | +2.215 |
| (-9, 13, 10) | 2.39 | -189.1 | 223.0 | -0.8 | 39.6 | 21.2 | 3.158 | 0.947 | +2.211 |
| (-13, 1, 8) | 4.90 | 305862.6 | 18800.0 | 16.3 | 590.8 | 642.6 | 1.933 | 4.118 | -2.184 |
| (11, 3, 0) | 6.46 | 53805.3 | 3549.0 | 15.2 | 250.2 | 300.0 | -0.097 | 1.871 | -1.968 |
| (-12, 8, 19) | 2.41 | 4025.3 | 782.0 | 5.1 | 64.4 | 88.7 | 0.606 | 2.465 | -1.859 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `shaken` | 22.46 | 0.9831 | 199.30483080247237 | 788.7820064945549 | 1.000 |
| `recovered_f` | 15.95 | 0.9903 | 199.11732842991213 | 510.6639779862454 | 1.000 |