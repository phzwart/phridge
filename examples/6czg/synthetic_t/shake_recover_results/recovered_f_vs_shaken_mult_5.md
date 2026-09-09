# Held-Out Log-Likelihood Comparison: `recovered_f` vs `shaken`

## 1. Executive Summary

- **Scored Audit Reflections (|A|)**: 900 (audit set)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `0.5616` nats/refl
- **Model B NLL**: `0.1743` nats/refl
- **Difference (Gain $\Delta$)**: `+0.3874` nats/refl (`-0.3874` nats NLL reduction)
- **Estimated Log Bayes Factor**: `+348.64` nats
- **Uncertainty**: Bootstrap SE = `0.0281` (95% CI: `[+0.3303, +0.4397]`) | Naive SE = `0.0293` (Ratio: `0.96`x)
- **Win Fraction $P(d_h > 0)$**: `78.0%` (702 wins, 198 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `9.41e-67`
- **Wilcoxon Signed-Rank $p$-value**: `1.34e-57`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `-0.2843` | `-0.1031` | `-0.3874` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `-0.5663` | `+0.1789` | `-0.3874` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`shaken`) | Model B (`recovered_f`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0019` | `-0.0068` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `0.6570` | `0.1749` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.0954` | `-0.0006` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -0.9420 | -1.3857 | +0.4437 | 0.2226 | 62.8% | 27/16 |
| 1 | 6.10 | 4.86 | 46 | 0.2679 | 0.1214 | +0.1466 | 0.1187 | 67.4% | 31/15 |
| 2 | 4.83 | 4.21 | 45 | -0.1075 | -0.3210 | +0.2134 | 0.1173 | 80.0% | 36/9 |
| 3 | 4.20 | 3.81 | 46 | 0.5170 | 0.0590 | +0.4581 | 0.1199 | 82.6% | 38/8 |
| 4 | 3.80 | 3.53 | 46 | 0.4641 | 0.2634 | +0.2007 | 0.1509 | 69.6% | 32/14 |
| 5 | 3.53 | 3.32 | 46 | 0.9646 | 0.3897 | +0.5749 | 0.1912 | 73.9% | 34/12 |
| 6 | 3.31 | 3.16 | 44 | 0.3040 | 0.0171 | +0.2869 | 0.0761 | 72.7% | 32/12 |
| 7 | 3.14 | 3.02 | 45 | 0.8736 | 0.4941 | +0.3795 | 0.1341 | 75.6% | 34/11 |
| 8 | 3.01 | 2.89 | 46 | 0.4705 | 0.0221 | +0.4484 | 0.0978 | 82.6% | 38/8 |
| 9 | 2.88 | 2.79 | 43 | 0.5796 | 0.1791 | +0.4006 | 0.1044 | 76.7% | 33/10 |
| 10 | 2.79 | 2.70 | 45 | 0.2922 | -0.1723 | +0.4645 | 0.0861 | 86.7% | 39/6 |
| 11 | 2.70 | 2.62 | 45 | 0.7194 | 0.1799 | +0.5395 | 0.1053 | 91.1% | 41/4 |
| 12 | 2.62 | 2.55 | 44 | 0.6266 | 0.0967 | +0.5299 | 0.0883 | 79.5% | 35/9 |
| 13 | 2.55 | 2.49 | 47 | 0.6872 | 0.3037 | +0.3836 | 0.1008 | 74.5% | 35/12 |
| 14 | 2.49 | 2.43 | 43 | 0.4932 | 0.0413 | +0.4519 | 0.0991 | 81.4% | 35/8 |
| 15 | 2.43 | 2.38 | 43 | 0.9925 | 0.6085 | +0.3840 | 0.1368 | 86.0% | 37/6 |
| 16 | 2.38 | 2.33 | 44 | 1.0715 | 0.5295 | +0.5419 | 0.1621 | 79.5% | 35/9 |
| 17 | 2.32 | 2.28 | 47 | 0.9573 | 0.5471 | +0.4102 | 0.0920 | 78.7% | 37/10 |
| 18 | 2.28 | 2.24 | 48 | 0.9283 | 0.7772 | +0.1510 | 0.1437 | 79.2% | 38/10 |
| 19 | 2.24 | 2.20 | 44 | 0.9946 | 0.6294 | +0.3653 | 0.1197 | 79.5% | 35/9 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (3, 1, 0) | 22.09 | 19735.3 | 907.5 | 21.7 | 200.8 | 196.4 | 9.659 | 1.165 | +8.494 |
| (20, 4, 7) | 3.47 | 471.7 | 177.5 | 2.7 | 127.2 | 34.9 | 5.157 | -1.074 | +6.230 |
| (-15, 9, 19) | 2.26 | -986.1 | 181.0 | -5.4 | 37.4 | 12.2 | 7.844 | 13.239 | -5.395 |
| (27, 7, 12) | 2.34 | 13603.7 | 495.5 | 27.5 | 75.1 | 118.6 | 5.793 | 1.321 | +4.471 |
| (-1, 1, 3) | 17.00 | -38.1 | 149.5 | -0.3 | 56.6 | 28.5 | 2.216 | -2.108 | +4.324 |
| (2, 2, 14) | 4.11 | 121519.8 | 5705.5 | 21.3 | 421.9 | 436.9 | 2.466 | 6.033 | -3.567 |
| (15, 7, 4) | 3.64 | 78456.1 | 3885.0 | 20.2 | 211.6 | 288.3 | 4.200 | 0.837 | +3.363 |
| (0, 4, 12) | 4.33 | 73099.5 | 2072.0 | 35.3 | 259.7 | 348.8 | 0.987 | 4.341 | -3.354 |
| (-30, 2, 14) | 2.35 | 1474.5 | 210.0 | 7.0 | 95.1 | 40.5 | 3.248 | 0.068 | +3.180 |
| (21, 3, 2) | 3.79 | 52173.5 | 4782.0 | 10.9 | 252.0 | 304.4 | 0.956 | 4.134 | -3.179 |
| (27, 1, 2) | 3.11 | 66998.6 | 4270.0 | 15.7 | 212.3 | 269.3 | 4.566 | 1.429 | +3.137 |
| (-18, 4, 10) | 3.43 | 48176.8 | 3216.5 | 15.0 | 238.7 | 287.2 | 1.356 | 4.457 | -3.101 |
| (-29, 9, 3) | 2.33 | -225.8 | 139.5 | -1.6 | 63.3 | 20.0 | 5.224 | 2.124 | +3.100 |
| (-20, 2, 14) | 2.98 | 13465.1 | 447.5 | 30.1 | 51.4 | 114.8 | 3.504 | 0.414 | +3.090 |
| (-12, 8, 19) | 2.41 | 5785.3 | 391.0 | 14.8 | 31.3 | 83.6 | 3.849 | 0.793 | +3.056 |
| (9, 3, 14) | 3.66 | 131464.6 | 5549.5 | 23.7 | 361.8 | 441.6 | 1.734 | 4.725 | -2.991 |
| (-24, 2, 2) | 3.45 | 55458.2 | 3850.5 | 14.4 | 181.8 | 244.2 | 3.953 | 1.099 | +2.854 |
| (-14, 6, 18) | 2.62 | 22310.9 | 965.0 | 23.1 | 109.0 | 167.6 | 4.088 | 1.235 | +2.853 |
| (-30, 4, 4) | 2.66 | 14537.4 | 884.0 | 16.4 | 78.9 | 129.6 | 3.713 | 0.963 | +2.750 |
| (35, 1, 10) | 2.23 | 8254.6 | 542.0 | 15.2 | 64.9 | 102.8 | 3.950 | 1.319 | +2.631 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `shaken` | 36.30 | 0.9572 | 3.6192316582631445 | 1.2079339209018956 | 1.000 |
| `recovered_f` | 17.89 | 0.9876 | 5.471534845074314 | 1.8437416562728197 | 1.000 |