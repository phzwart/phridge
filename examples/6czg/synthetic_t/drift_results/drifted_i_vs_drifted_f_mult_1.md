# Held-Out Log-Likelihood Comparison: `drifted_i` vs `drifted_f`

## 1. Executive Summary

- **Scored Test Reflections (|T|)**: 900 (held-out)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `-0.1658` nats/refl
- **Model B NLL**: `-0.1654` nats/refl
- **Difference (Gain $\Delta$)**: `-0.0004` nats/refl (`+0.0004` nats NLL reduction)
- **Estimated Log Bayes Factor**: `-0.32` nats
- **Uncertainty**: Bootstrap SE = `0.0006` (95% CI: `[-0.0015, +0.0006]`) | Naive SE = `0.0005` (Ratio: `1.21`x)
- **Win Fraction $P(d_h > 0)$**: `48.2%` (434 wins, 466 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `3.01e-01`
- **Wilcoxon Signed-Rank $p$-value**: `9.09e-01`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.0004` | `-0.0001` | `+0.0004` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.0004` | `-0.0001` | `+0.0004` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`drifted_f`) | Model B (`drifted_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0021` | `-0.0023` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `-0.1374` | `-0.1364` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.0284` | `-0.0291` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -1.1717 | -1.1652 | -0.0064 | 0.0025 | 39.5% | 17/26 |
| 1 | 6.10 | 4.86 | 46 | 0.7383 | 0.7409 | -0.0026 | 0.0039 | 43.5% | 20/26 |
| 2 | 4.83 | 4.21 | 45 | -0.2846 | -0.2806 | -0.0040 | 0.0024 | 42.2% | 19/26 |
| 3 | 4.20 | 3.81 | 46 | 0.1409 | 0.1375 | +0.0034 | 0.0028 | 54.3% | 25/21 |
| 4 | 3.80 | 3.53 | 46 | 0.3360 | 0.3333 | +0.0026 | 0.0026 | 52.2% | 24/22 |
| 5 | 3.53 | 3.32 | 46 | 0.5134 | 0.5155 | -0.0021 | 0.0044 | 56.5% | 26/20 |
| 6 | 3.31 | 3.16 | 44 | -0.1772 | -0.1769 | -0.0003 | 0.0020 | 38.6% | 17/27 |
| 7 | 3.14 | 3.02 | 45 | 0.3757 | 0.3802 | -0.0045 | 0.0029 | 42.2% | 19/26 |
| 8 | 3.01 | 2.89 | 46 | -0.2550 | -0.2536 | -0.0014 | 0.0009 | 43.5% | 20/26 |
| 9 | 2.88 | 2.79 | 43 | -0.2745 | -0.2734 | -0.0012 | 0.0009 | 34.9% | 15/28 |
| 10 | 2.79 | 2.70 | 45 | -0.5564 | -0.5562 | -0.0002 | 0.0006 | 53.3% | 24/21 |
| 11 | 2.70 | 2.62 | 45 | -0.2123 | -0.2136 | +0.0013 | 0.0009 | 44.4% | 20/25 |
| 12 | 2.62 | 2.55 | 44 | -0.3805 | -0.3832 | +0.0027 | 0.0008 | 68.2% | 30/14 |
| 13 | 2.55 | 2.49 | 47 | -0.2769 | -0.2768 | -0.0001 | 0.0005 | 51.1% | 24/23 |
| 14 | 2.49 | 2.43 | 43 | -0.5808 | -0.5818 | +0.0010 | 0.0007 | 53.5% | 23/20 |
| 15 | 2.43 | 2.38 | 43 | -0.0855 | -0.0871 | +0.0016 | 0.0007 | 58.1% | 25/18 |
| 16 | 2.38 | 2.33 | 44 | -0.4058 | -0.4057 | -0.0001 | 0.0006 | 43.2% | 19/25 |
| 17 | 2.32 | 2.28 | 47 | -0.2335 | -0.2345 | +0.0010 | 0.0007 | 48.9% | 23/24 |
| 18 | 2.28 | 2.24 | 48 | -0.3376 | -0.3387 | +0.0010 | 0.0006 | 52.1% | 25/23 |
| 19 | 2.24 | 2.20 | 44 | -0.2973 | -0.2982 | +0.0009 | 0.0006 | 43.2% | 19/25 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (-16, 4, 12) | 3.38 | 80193.7 | 611.6 | 131.1 | 340.6 | 341.4 | 5.415 | 5.552 | -0.136 |
| (-14, 4, 2) | 4.94 | 126721.0 | 1288.0 | 98.4 | 428.7 | 429.5 | 4.499 | 4.581 | -0.082 |
| (8, 2, 0) | 9.10 | 78201.9 | 374.4 | 208.9 | 352.7 | 354.4 | 1.740 | 1.821 | -0.081 |
| (27, 1, 2) | 3.11 | 66716.5 | 854.0 | 78.1 | 306.1 | 306.5 | 4.704 | 4.766 | -0.062 |
| (9, 3, 14) | 3.66 | 132444.0 | 1109.9 | 119.3 | 431.9 | 432.4 | 4.680 | 4.739 | -0.059 |
| (11, 1, 12) | 4.12 | 42058.7 | 403.2 | 104.3 | 245.5 | 244.8 | 1.971 | 1.913 | +0.058 |
| (-12, 4, 8) | 4.46 | 74634.5 | 1028.6 | 72.6 | 335.5 | 336.2 | 2.647 | 2.705 | -0.058 |
| (-9, 5, 4) | 5.30 | 73520.8 | 1046.7 | 70.2 | 334.0 | 333.5 | 3.235 | 3.179 | +0.056 |
| (7, 7, 11) | 3.52 | 16481.9 | 231.3 | 71.3 | 160.0 | 159.5 | 1.557 | 1.501 | +0.056 |
| (7, 5, 4) | 5.63 | 50478.1 | 634.4 | 79.6 | 274.2 | 275.0 | 1.940 | 1.993 | -0.054 |
| (9, 5, 4) | 5.27 | 126633.4 | 1439.0 | 88.0 | 441.8 | 442.3 | 5.785 | 5.838 | -0.053 |
| (11, 5, 4) | 4.90 | 41410.7 | 402.1 | 103.0 | 251.3 | 250.7 | 2.070 | 2.021 | +0.050 |
| (-17, 1, 9) | 3.99 | 32400.2 | 200.1 | 161.9 | 209.3 | 208.6 | 1.178 | 1.130 | +0.049 |
| (-12, 8, 6) | 3.51 | 50266.4 | 424.9 | 118.3 | 271.3 | 271.0 | 3.495 | 3.447 | +0.048 |
| (-3, 3, 11) | 4.86 | 24624.2 | 195.6 | 125.9 | 193.9 | 194.6 | 1.187 | 1.231 | -0.045 |
| (-27, 1, 2) | 3.12 | 25190.1 | 400.1 | 63.0 | 183.8 | 184.2 | 1.536 | 1.580 | -0.043 |
| (-25, 1, 9) | 3.03 | 13855.7 | 248.1 | 55.8 | 137.6 | 138.1 | 1.033 | 1.076 | -0.043 |
| (2, 6, 11) | 3.96 | 12719.4 | 141.5 | 89.9 | 138.5 | 137.9 | 0.672 | 0.630 | +0.042 |
| (-4, 6, 4) | 5.31 | 21428.0 | 514.3 | 41.7 | 189.7 | 189.2 | 1.408 | 1.366 | +0.042 |
| (10, 4, 2) | 5.98 | 67898.4 | 413.5 | 164.2 | 319.7 | 320.2 | 2.601 | 2.642 | -0.041 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `drifted_f` | 6.08 | 0.9990 | 199.1739897783472 | 1000.0 | 1.000 |
| `drifted_i` | 6.14 | 0.9990 | 199.17395004567555 | 1000.0 | 1.000 |