# Held-Out Log-Likelihood Comparison: `drifted_i` vs `drifted_f`

## 1. Executive Summary

- **Scored Audit Reflections (|A|)**: 900 (audit set)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `-0.1402` nats/refl
- **Model B NLL**: `-0.1394` nats/refl
- **Difference (Gain $\Delta$)**: `-0.0008` nats/refl (`+0.0008` nats NLL reduction)
- **Estimated Log Bayes Factor**: `-0.72` nats
- **Uncertainty**: Bootstrap SE = `0.0007` (95% CI: `[-0.0022, +0.0004]`) | Naive SE = `0.0004` (Ratio: `1.70`x)
- **Win Fraction $P(d_h > 0)$**: `45.8%` (412 wins, 488 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `1.24e-02`
- **Wilcoxon Signed-Rank $p$-value**: `1.52e-01`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.0008` | `-0.0000` | `+0.0008` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.0008` | `-0.0000` | `+0.0008` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`drifted_f`) | Model B (`drifted_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0019` | `-0.0019` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `-0.1172` | `-0.1159` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.0230` | `-0.0235` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -1.1867 | -1.1846 | -0.0021 | 0.0013 | 39.5% | 17/26 |
| 1 | 6.10 | 4.86 | 46 | 0.6890 | 0.6964 | -0.0074 | 0.0034 | 37.0% | 17/29 |
| 2 | 4.83 | 4.21 | 45 | -0.2898 | -0.2838 | -0.0060 | 0.0016 | 26.7% | 12/33 |
| 3 | 4.20 | 3.81 | 46 | 0.1279 | 0.1322 | -0.0043 | 0.0025 | 32.6% | 15/31 |
| 4 | 3.80 | 3.53 | 46 | 0.3634 | 0.3690 | -0.0056 | 0.0025 | 28.3% | 13/33 |
| 5 | 3.53 | 3.32 | 46 | 0.4618 | 0.4661 | -0.0043 | 0.0029 | 34.8% | 16/30 |
| 6 | 3.31 | 3.16 | 44 | -0.1321 | -0.1346 | +0.0024 | 0.0022 | 56.8% | 25/19 |
| 7 | 3.14 | 3.02 | 45 | 0.4409 | 0.4402 | +0.0007 | 0.0028 | 48.9% | 22/23 |
| 8 | 3.01 | 2.89 | 46 | -0.2584 | -0.2572 | -0.0011 | 0.0014 | 41.3% | 19/27 |
| 9 | 2.88 | 2.79 | 43 | -0.2446 | -0.2437 | -0.0009 | 0.0010 | 46.5% | 20/23 |
| 10 | 2.79 | 2.70 | 45 | -0.4607 | -0.4608 | +0.0002 | 0.0012 | 48.9% | 22/23 |
| 11 | 2.70 | 2.62 | 45 | -0.2241 | -0.2237 | -0.0005 | 0.0013 | 42.2% | 19/26 |
| 12 | 2.62 | 2.55 | 44 | -0.3518 | -0.3545 | +0.0027 | 0.0008 | 65.9% | 29/15 |
| 13 | 2.55 | 2.49 | 47 | -0.2755 | -0.2756 | +0.0000 | 0.0008 | 53.2% | 25/22 |
| 14 | 2.49 | 2.43 | 43 | -0.5768 | -0.5787 | +0.0018 | 0.0012 | 41.9% | 18/25 |
| 15 | 2.43 | 2.38 | 43 | -0.0454 | -0.0474 | +0.0021 | 0.0011 | 53.5% | 23/20 |
| 16 | 2.38 | 2.33 | 44 | -0.3314 | -0.3326 | +0.0012 | 0.0007 | 59.1% | 26/18 |
| 17 | 2.32 | 2.28 | 47 | -0.1745 | -0.1762 | +0.0017 | 0.0011 | 53.2% | 25/22 |
| 18 | 2.28 | 2.24 | 48 | -0.2414 | -0.2435 | +0.0021 | 0.0009 | 52.1% | 25/23 |
| 19 | 2.24 | 2.20 | 44 | -0.2031 | -0.2048 | +0.0017 | 0.0010 | 54.5% | 24/20 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (14, 2, 4) | 5.33 | 185493.8 | 2286.0 | 81.1 | 533.7 | 534.3 | 7.595 | 7.675 | -0.080 |
| (27, 1, 2) | 3.11 | 63525.2 | 1281.0 | 49.6 | 304.7 | 305.2 | 5.401 | 5.474 | -0.073 |
| (-16, 4, 12) | 3.38 | 80054.6 | 917.4 | 87.3 | 339.3 | 339.8 | 5.031 | 5.103 | -0.072 |
| (14, 4, 0) | 5.00 | 55763.6 | 801.0 | 69.6 | 293.1 | 292.3 | 2.749 | 2.678 | +0.071 |
| (11, 5, 4) | 4.90 | 40901.1 | 603.2 | 67.8 | 250.9 | 251.7 | 2.069 | 2.128 | -0.059 |
| (2, 10, 4) | 3.42 | 30854.4 | 510.8 | 60.4 | 217.0 | 217.5 | 2.813 | 2.871 | -0.057 |
| (-9, 3, 14) | 3.70 | 51594.7 | 639.6 | 80.7 | 271.3 | 272.0 | 2.041 | 2.097 | -0.056 |
| (-13, 1, 8) | 4.90 | 286163.9 | 2820.0 | 101.5 | 649.7 | 650.1 | 9.830 | 9.884 | -0.055 |
| (14, 6, 11) | 3.31 | 7224.1 | 326.1 | 22.2 | 107.7 | 106.9 | 0.353 | 0.300 | +0.053 |
| (2, 2, 14) | 4.11 | 124622.5 | 1711.6 | 72.8 | 426.1 | 426.5 | 5.499 | 5.550 | -0.051 |
| (-12, 10, 2) | 3.14 | 18677.3 | 558.3 | 33.5 | 169.5 | 169.1 | 2.234 | 2.187 | +0.047 |
| (-12, 8, 6) | 3.51 | 50749.5 | 637.3 | 79.6 | 269.6 | 270.0 | 3.035 | 3.082 | -0.047 |
| (9, 3, 14) | 3.66 | 132472.5 | 1664.9 | 79.6 | 429.3 | 429.8 | 4.294 | 4.341 | -0.046 |
| (-27, 1, 6) | 3.00 | 16331.4 | 168.3 | 97.0 | 150.7 | 151.2 | 1.075 | 1.121 | -0.046 |
| (-14, 4, 2) | 4.94 | 125909.8 | 1932.0 | 65.2 | 429.0 | 429.4 | 4.462 | 4.506 | -0.044 |
| (17, 5, 0) | 4.08 | 57159.6 | 1145.6 | 49.9 | 290.6 | 291.0 | 2.998 | 3.039 | -0.041 |
| (19, 9, 8) | 2.73 | 15992.1 | 376.0 | 42.5 | 150.1 | 149.8 | 1.763 | 1.723 | +0.041 |
| (-11, 1, 12) | 4.17 | 38948.5 | 809.4 | 48.1 | 238.9 | 239.4 | 1.948 | 1.988 | -0.040 |
| (-2, 4, 12) | 4.31 | 34457.8 | 316.5 | 108.9 | 228.0 | 228.7 | 1.217 | 1.254 | -0.037 |
| (-13, 5, 7) | 4.20 | 11895.3 | 308.0 | 38.6 | 133.5 | 134.2 | 0.453 | 0.490 | -0.037 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `drifted_f` | 6.27 | 0.9989 | 8.304382684723622 | 9.062916033705973 | 1.000 |
| `drifted_i` | 6.33 | 0.9988 | 8.285002301397121 | 9.027584986864827 | 1.000 |