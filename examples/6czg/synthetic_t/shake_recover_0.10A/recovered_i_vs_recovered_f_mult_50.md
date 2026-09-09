# Held-Out Log-Likelihood Comparison: `recovered_i` vs `recovered_f`

## 1. Executive Summary

- **Scored Audit Reflections (|A|)**: 900 (audit set)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `1.1822` nats/refl
- **Model B NLL**: `1.1886` nats/refl
- **Difference (Gain $\Delta$)**: `-0.0064` nats/refl (`+0.0064` nats NLL reduction)
- **Estimated Log Bayes Factor**: `-5.77` nats
- **Uncertainty**: Bootstrap SE = `0.0043` (95% CI: `[-0.0157, +0.0008]`) | Naive SE = `0.0035` (Ratio: `1.22`x)
- **Win Fraction $P(d_h > 0)$**: `47.1%` (424 wins, 476 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `8.91e-02`
- **Wilcoxon Signed-Rank $p$-value**: `8.94e-02`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.0065` | `-0.0001` | `+0.0064` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.0064` | `+0.0000` | `+0.0064` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`recovered_f`) | Model B (`recovered_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0080` | `-0.0076` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `1.3369` | `1.3215` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.1547` | `-0.1329` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -0.8686 | -0.7930 | -0.0756 | 0.0449 | 44.2% | 19/24 |
| 1 | 6.10 | 4.86 | 46 | 0.6768 | 0.7015 | -0.0247 | 0.0109 | 34.8% | 16/30 |
| 2 | 4.83 | 4.21 | 45 | 0.5536 | 0.5487 | +0.0050 | 0.0136 | 42.2% | 19/26 |
| 3 | 4.20 | 3.81 | 46 | 0.9499 | 0.9530 | -0.0031 | 0.0102 | 47.8% | 22/24 |
| 4 | 3.80 | 3.53 | 46 | 1.3087 | 1.3230 | -0.0143 | 0.0189 | 52.2% | 24/22 |
| 5 | 3.53 | 3.32 | 46 | 1.3653 | 1.3853 | -0.0200 | 0.0132 | 45.7% | 21/25 |
| 6 | 3.31 | 3.16 | 44 | 1.1754 | 1.1958 | -0.0205 | 0.0120 | 40.9% | 18/26 |
| 7 | 3.14 | 3.02 | 45 | 1.4418 | 1.4367 | +0.0051 | 0.0161 | 51.1% | 23/22 |
| 8 | 3.01 | 2.89 | 46 | 0.8193 | 0.8228 | -0.0035 | 0.0123 | 45.7% | 21/25 |
| 9 | 2.88 | 2.79 | 43 | 1.0478 | 1.0525 | -0.0046 | 0.0138 | 48.8% | 21/22 |
| 10 | 2.79 | 2.70 | 45 | 0.8279 | 0.8166 | +0.0113 | 0.0111 | 42.2% | 19/26 |
| 11 | 2.70 | 2.62 | 45 | 1.4665 | 1.4677 | -0.0012 | 0.0139 | 53.3% | 24/21 |
| 12 | 2.62 | 2.55 | 44 | 1.3176 | 1.3205 | -0.0029 | 0.0118 | 47.7% | 21/23 |
| 13 | 2.55 | 2.49 | 47 | 1.4114 | 1.4238 | -0.0124 | 0.0131 | 44.7% | 21/26 |
| 14 | 2.49 | 2.43 | 43 | 1.1733 | 1.1675 | +0.0058 | 0.0115 | 48.8% | 21/22 |
| 15 | 2.43 | 2.38 | 43 | 1.8072 | 1.7906 | +0.0166 | 0.0129 | 58.1% | 25/18 |
| 16 | 2.38 | 2.33 | 44 | 1.4651 | 1.4547 | +0.0105 | 0.0115 | 61.4% | 27/17 |
| 17 | 2.32 | 2.28 | 47 | 1.8080 | 1.8118 | -0.0038 | 0.0097 | 44.7% | 21/26 |
| 18 | 2.28 | 2.24 | 48 | 1.8716 | 1.8623 | +0.0092 | 0.0103 | 47.9% | 23/25 |
| 19 | 2.24 | 2.20 | 44 | 1.9145 | 1.9208 | -0.0063 | 0.0070 | 40.9% | 18/26 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (-6, 2, 2) | 10.39 | 21050.1 | 765.0 | 27.5 | 159.2 | 166.3 | -0.354 | 1.355 | -1.709 |
| (-14, 2, 12) | 3.78 | 28046.8 | 19035.0 | 1.5 | 272.9 | 284.5 | 3.656 | 4.206 | -0.550 |
| (24, 0, 10) | 3.02 | 49740.7 | 6740.0 | 7.4 | 167.9 | 173.7 | 4.210 | 3.672 | +0.538 |
| (-1, 5, 4) | 6.37 | 22235.6 | 3590.0 | 6.2 | 179.4 | 183.3 | 1.017 | 1.467 | -0.450 |
| (-12, 8, 5) | 3.58 | 5111.6 | 1390.0 | 3.7 | 47.5 | 53.7 | 0.138 | -0.290 | +0.428 |
| (-13, 5, 20) | 2.54 | 7058.8 | 1940.0 | 3.6 | 36.1 | 29.6 | 3.547 | 3.966 | -0.419 |
| (-10, 0, 3) | 7.85 | 2277.2 | 1455.0 | 1.6 | 67.6 | 73.2 | -1.233 | -0.860 | -0.372 |
| (26, 4, 2) | 3.04 | 6607.7 | 2215.0 | 3.0 | 107.0 | 111.7 | 1.199 | 1.559 | -0.361 |
| (18, 2, 18) | 2.66 | 8491.3 | 3470.0 | 2.4 | 114.3 | 121.1 | 1.519 | 1.861 | -0.342 |
| (12, 8, 19) | 2.39 | 620.0 | 2015.0 | 0.3 | 64.5 | 58.0 | 1.959 | 1.625 | +0.334 |
| (12, 2, 19) | 2.81 | 15260.9 | 4295.0 | 3.6 | 164.1 | 159.2 | 2.608 | 2.288 | +0.320 |
| (-14, 8, 7) | 3.30 | 8930.1 | 2630.0 | 3.4 | 47.3 | 42.0 | 2.095 | 2.413 | -0.318 |
| (-10, 2, 20) | 2.79 | 7471.7 | 3165.0 | 2.4 | 59.9 | 69.0 | 1.131 | 0.820 | +0.310 |
| (6, 8, 8) | 3.66 | 5633.0 | 2005.0 | 2.8 | 36.5 | 41.2 | 0.823 | 0.525 | +0.298 |
| (-15, 7, 17) | 2.58 | 12853.4 | 2350.0 | 5.5 | 77.3 | 74.6 | 3.542 | 3.825 | -0.283 |
| (-27, 9, 1) | 2.45 | 5502.1 | 2130.0 | 2.6 | 52.7 | 47.3 | 1.402 | 1.684 | -0.282 |
| (2, 14, 12) | 2.24 | 70.6 | 1250.0 | 0.1 | 47.8 | 44.4 | 1.889 | 1.612 | +0.277 |
| (-3, 5, 11) | 4.26 | 10860.2 | 1340.0 | 8.1 | 55.7 | 56.3 | 4.653 | 4.376 | +0.277 |
| (-24, 6, 6) | 2.91 | 5355.2 | 2180.0 | 2.5 | 87.2 | 92.2 | 0.403 | 0.677 | -0.274 |
| (18, 12, 2) | 2.49 | 5451.0 | 1965.0 | 2.8 | 55.3 | 50.4 | 1.184 | 1.458 | -0.274 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `recovered_f` | 73.98 | 0.7108 | 9.387353003246394 | 2.9075257851968495 | 1.000 |
| `recovered_i` | 74.50 | 0.7084 | 9.06805272058194 | 2.732101238260012 | 1.000 |