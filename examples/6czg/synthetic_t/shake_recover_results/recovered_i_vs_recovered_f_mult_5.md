# Held-Out Log-Likelihood Comparison: `recovered_i` vs `recovered_f`

## 1. Executive Summary

- **Scored Audit Reflections (|A|)**: 900 (audit set)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `0.1743` nats/refl
- **Model B NLL**: `0.1749` nats/refl
- **Difference (Gain $\Delta$)**: `-0.0007` nats/refl (`+0.0007` nats NLL reduction)
- **Estimated Log Bayes Factor**: `-0.60` nats
- **Uncertainty**: Bootstrap SE = `0.0007` (95% CI: `[-0.0020, +0.0006]`) | Naive SE = `0.0006` (Ratio: `1.22`x)
- **Win Fraction $P(d_h > 0)$**: `46.4%` (418 wins, 482 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `3.57e-02`
- **Wilcoxon Signed-Rank $p$-value**: `2.37e-02`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.0005` | `+0.0002` | `+0.0007` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.0005` | `+0.0002` | `+0.0007` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`recovered_f`) | Model B (`recovered_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0068` | `-0.0066` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `0.1749` | `0.1755` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.0006` | `-0.0006` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -1.3857 | -1.3806 | -0.0051 | 0.0042 | 25.6% | 11/32 |
| 1 | 6.10 | 4.86 | 46 | 0.1214 | 0.1257 | -0.0043 | 0.0033 | 30.4% | 14/32 |
| 2 | 4.83 | 4.21 | 45 | -0.3210 | -0.3215 | +0.0005 | 0.0024 | 35.6% | 16/29 |
| 3 | 4.20 | 3.81 | 46 | 0.0590 | 0.0630 | -0.0040 | 0.0033 | 37.0% | 17/29 |
| 4 | 3.80 | 3.53 | 46 | 0.2634 | 0.2671 | -0.0038 | 0.0036 | 28.3% | 13/33 |
| 5 | 3.53 | 3.32 | 46 | 0.3897 | 0.3899 | -0.0003 | 0.0038 | 37.0% | 17/29 |
| 6 | 3.31 | 3.16 | 44 | 0.0171 | 0.0233 | -0.0062 | 0.0026 | 34.1% | 15/29 |
| 7 | 3.14 | 3.02 | 45 | 0.4941 | 0.4988 | -0.0047 | 0.0030 | 40.0% | 18/27 |
| 8 | 3.01 | 2.89 | 46 | 0.0221 | 0.0214 | +0.0008 | 0.0015 | 58.7% | 27/19 |
| 9 | 2.88 | 2.79 | 43 | 0.1791 | 0.1809 | -0.0019 | 0.0016 | 46.5% | 20/23 |
| 10 | 2.79 | 2.70 | 45 | -0.1723 | -0.1701 | -0.0022 | 0.0017 | 46.7% | 21/24 |
| 11 | 2.70 | 2.62 | 45 | 0.1799 | 0.1788 | +0.0011 | 0.0017 | 51.1% | 23/22 |
| 12 | 2.62 | 2.55 | 44 | 0.0967 | 0.0952 | +0.0015 | 0.0013 | 61.4% | 27/17 |
| 13 | 2.55 | 2.49 | 47 | 0.3037 | 0.3013 | +0.0024 | 0.0019 | 55.3% | 26/21 |
| 14 | 2.49 | 2.43 | 43 | 0.0413 | 0.0414 | -0.0001 | 0.0015 | 53.5% | 23/20 |
| 15 | 2.43 | 2.38 | 43 | 0.6085 | 0.6050 | +0.0035 | 0.0016 | 51.2% | 22/21 |
| 16 | 2.38 | 2.33 | 44 | 0.5295 | 0.5258 | +0.0037 | 0.0015 | 81.8% | 36/8 |
| 17 | 2.32 | 2.28 | 47 | 0.5471 | 0.5467 | +0.0003 | 0.0012 | 40.4% | 19/28 |
| 18 | 2.28 | 2.24 | 48 | 0.7772 | 0.7749 | +0.0023 | 0.0016 | 56.2% | 27/21 |
| 19 | 2.24 | 2.20 | 44 | 0.6294 | 0.6265 | +0.0029 | 0.0021 | 59.1% | 26/18 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (3, 1, 0) | 22.09 | 19735.3 | 907.5 | 21.7 | 196.4 | 195.2 | 1.165 | 1.020 | +0.145 |
| (-12, 8, 6) | 3.51 | 50830.6 | 2124.5 | 23.9 | 279.2 | 278.2 | 3.109 | 3.003 | +0.106 |
| (17, 5, 0) | 4.08 | 59257.2 | 3818.5 | 15.5 | 284.5 | 286.3 | 1.798 | 1.899 | -0.101 |
| (-12, 6, 14) | 3.11 | 11836.3 | 542.5 | 21.8 | 151.4 | 152.3 | 2.409 | 2.498 | -0.089 |
| (9, 3, 14) | 3.66 | 131464.6 | 5549.5 | 23.7 | 441.6 | 441.1 | 4.725 | 4.644 | +0.080 |
| (-19, 1, 10) | 3.58 | 56054.6 | 2304.5 | 24.3 | 276.4 | 277.7 | 1.587 | 1.660 | -0.073 |
| (-8, 4, 16) | 3.28 | 22932.0 | 1556.5 | 14.7 | 120.3 | 119.4 | 1.736 | 1.804 | -0.069 |
| (4, 10, 0) | 3.48 | 16164.3 | 606.5 | 26.7 | 109.0 | 110.3 | 0.939 | 0.873 | +0.066 |
| (-9, 5, 4) | 5.30 | 68282.5 | 5233.5 | 13.0 | 316.5 | 315.7 | 2.430 | 2.364 | +0.066 |
| (2, 2, 14) | 4.11 | 121519.8 | 5705.5 | 21.3 | 436.9 | 436.5 | 6.033 | 5.968 | +0.065 |
| (-14, 4, 13) | 3.40 | 9622.3 | 220.5 | 43.6 | 129.6 | 128.7 | 1.136 | 1.071 | +0.065 |
| (-13, 13, 2) | 2.50 | 1847.7 | 550.0 | 3.4 | 70.7 | 70.1 | 1.556 | 1.493 | +0.063 |
| (26, 4, 2) | 3.04 | 6863.8 | 221.5 | 31.0 | 120.1 | 120.7 | 1.889 | 1.949 | -0.061 |
| (-7, 1, 22) | 2.64 | 4453.4 | 389.5 | 11.4 | 102.2 | 101.7 | 2.335 | 2.276 | +0.059 |
| (10, 4, 2) | 5.98 | 67445.1 | 2067.5 | 32.6 | 291.6 | 292.9 | 1.034 | 1.090 | -0.056 |
| (-7, 11, 10) | 2.76 | 3473.2 | 139.5 | 24.9 | 100.0 | 100.4 | 2.526 | 2.580 | -0.054 |
| (-18, 4, 10) | 3.43 | 48176.8 | 3216.5 | 15.0 | 287.2 | 287.7 | 4.457 | 4.508 | -0.051 |
| (3, 7, 24) | 2.22 | 3210.3 | 189.0 | 17.0 | 43.1 | 43.5 | 1.796 | 1.746 | +0.050 |
| (23, 3, 8) | 3.17 | 4574.5 | 322.0 | 14.2 | 98.8 | 99.5 | 0.762 | 0.811 | -0.050 |
| (-15, 7, 4) | 3.66 | 5988.5 | 920.5 | 6.5 | 106.0 | 105.1 | 0.247 | 0.199 | +0.048 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `recovered_f` | 17.89 | 0.9876 | 5.471534845074314 | 1.8437416562728197 | 1.000 |
| `recovered_i` | 17.89 | 0.9877 | 5.477802861520732 | 1.8460569542278205 | 1.000 |