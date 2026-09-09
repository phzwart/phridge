# Held-Out Log-Likelihood Comparison: `recovered_i` vs `shaken`

## 1. Executive Summary

- **Scored Audit Reflections (|A|)**: 900 (audit set)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `1.8739` nats/refl
- **Model B NLL**: `1.8508` nats/refl
- **Difference (Gain $\Delta$)**: `+0.0231` nats/refl (`-0.0231` nats NLL reduction)
- **Estimated Log Bayes Factor**: `+20.80` nats
- **Uncertainty**: Bootstrap SE = `0.0084` (95% CI: `[+0.0078, +0.0411]`) | Naive SE = `0.0072` (Ratio: `1.17`x)
- **Win Fraction $P(d_h > 0)$**: `52.0%` (468 wins, 432 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `2.43e-01`
- **Wilcoxon Signed-Rank $p$-value**: `3.12e-02`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `-0.0241` | `+0.0010` | `-0.0231` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `-0.0239` | `+0.0008` | `-0.0231` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`shaken`) | Model B (`recovered_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0273` | `-0.0264` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `1.9998` | `1.9726` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.1259` | `-0.1218` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -0.3625 | -0.3715 | +0.0090 | 0.0450 | 51.2% | 22/21 |
| 1 | 6.10 | 4.86 | 46 | 1.5566 | 1.5215 | +0.0351 | 0.0432 | 56.5% | 26/20 |
| 2 | 4.83 | 4.21 | 45 | 1.0070 | 0.9662 | +0.0408 | 0.0347 | 48.9% | 22/23 |
| 3 | 4.20 | 3.81 | 46 | 1.8132 | 1.8149 | -0.0017 | 0.0339 | 52.2% | 24/22 |
| 4 | 3.80 | 3.53 | 46 | 1.7713 | 1.8018 | -0.0306 | 0.0256 | 41.3% | 19/27 |
| 5 | 3.53 | 3.32 | 46 | 1.9361 | 1.9279 | +0.0082 | 0.0292 | 45.7% | 21/25 |
| 6 | 3.31 | 3.16 | 44 | 1.5288 | 1.5053 | +0.0234 | 0.0225 | 52.3% | 23/21 |
| 7 | 3.14 | 3.02 | 45 | 2.2596 | 2.2761 | -0.0165 | 0.0286 | 53.3% | 24/21 |
| 8 | 3.01 | 2.89 | 46 | 1.8125 | 1.6898 | +0.1227 | 0.0468 | 52.2% | 24/22 |
| 9 | 2.88 | 2.79 | 43 | 1.9492 | 1.9064 | +0.0429 | 0.0390 | 58.1% | 25/18 |
| 10 | 2.79 | 2.70 | 45 | 1.8767 | 1.8156 | +0.0611 | 0.0351 | 62.2% | 28/17 |
| 11 | 2.70 | 2.62 | 45 | 1.9291 | 1.8638 | +0.0653 | 0.0334 | 64.4% | 29/16 |
| 12 | 2.62 | 2.55 | 44 | 2.0614 | 1.9953 | +0.0661 | 0.0355 | 54.5% | 24/20 |
| 13 | 2.55 | 2.49 | 47 | 1.9414 | 1.9131 | +0.0283 | 0.0180 | 61.7% | 29/18 |
| 14 | 2.49 | 2.43 | 43 | 2.2774 | 2.2451 | +0.0323 | 0.0250 | 58.1% | 25/18 |
| 15 | 2.43 | 2.38 | 43 | 2.3573 | 2.3375 | +0.0199 | 0.0387 | 55.8% | 24/19 |
| 16 | 2.38 | 2.33 | 44 | 2.2771 | 2.3162 | -0.0391 | 0.0215 | 31.8% | 14/30 |
| 17 | 2.32 | 2.28 | 47 | 2.3002 | 2.3098 | -0.0097 | 0.0165 | 46.8% | 22/25 |
| 18 | 2.28 | 2.24 | 48 | 2.5202 | 2.5067 | +0.0136 | 0.0208 | 43.8% | 21/27 |
| 19 | 2.24 | 2.20 | 44 | 2.5748 | 2.5829 | -0.0081 | 0.0190 | 50.0% | 22/22 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (-13, 7, 14) | 2.92 | 15310.3 | 4250.0 | 3.6 | 75.2 | 99.0 | 2.706 | 1.393 | +1.313 |
| (2, 6, 0) | 5.83 | 23774.8 | 2760.0 | 8.6 | 147.2 | 134.4 | -0.243 | 0.956 | -1.200 |
| (6, 2, 2) | 10.31 | 10499.1 | 1030.0 | 10.2 | 102.1 | 115.3 | -2.272 | -1.214 | -1.058 |
| (28, 6, 6) | 2.59 | 17610.3 | 4650.0 | 3.8 | 95.6 | 113.0 | 2.756 | 1.851 | +0.905 |
| (20, 4, 7) | 3.47 | 5775.2 | 3550.0 | 1.6 | 70.7 | 29.1 | 0.281 | 1.145 | -0.864 |
| (-12, 8, 19) | 2.41 | 18115.2 | 7820.0 | 2.3 | 58.6 | 92.9 | 3.683 | 2.824 | +0.860 |
| (13, 11, 2) | 2.87 | 3646.0 | 1960.0 | 1.9 | 26.8 | 67.3 | 1.052 | 0.195 | +0.857 |
| (34, 0, 7) | 2.39 | 15835.7 | 5250.0 | 3.0 | 55.2 | 19.7 | 4.098 | 4.927 | -0.828 |
| (-3, 5, 11) | 4.26 | 6656.7 | 2680.0 | 2.5 | 38.8 | 55.2 | 0.594 | -0.219 | +0.813 |
| (15, 9, 7) | 3.01 | 12265.0 | 5340.0 | 2.3 | 49.2 | 75.4 | 2.235 | 1.435 | +0.800 |
| (-1, 5, 4) | 6.37 | 43715.8 | 7180.0 | 6.1 | 169.8 | 181.3 | 1.244 | 0.455 | +0.789 |
| (18, 2, 18) | 2.66 | 6044.3 | 6940.0 | 0.9 | 135.1 | 113.4 | 2.843 | 2.078 | +0.765 |
| (-13, 5, 16) | 2.96 | 30241.3 | 5360.0 | 5.6 | 77.7 | 95.3 | 6.074 | 5.341 | +0.733 |
| (-16, 8, 13) | 2.74 | 12645.1 | 5550.0 | 2.3 | 29.7 | 60.1 | 3.076 | 2.367 | +0.709 |
| (0, 2, 15) | 3.87 | -3337.5 | 9190.0 | -0.4 | 97.7 | 119.6 | 1.907 | 2.614 | -0.708 |
| (13, 11, 0) | 2.88 | 3394.9 | 2530.0 | 1.3 | 46.3 | 82.2 | 0.471 | 1.178 | -0.707 |
| (14, 4, 9) | 3.96 | 15368.9 | 4860.0 | 3.2 | 35.3 | 61.1 | 3.327 | 2.634 | +0.694 |
| (-2, 6, 21) | 2.55 | 188.3 | 4600.0 | 0.0 | 86.8 | 69.4 | 2.490 | 1.809 | +0.680 |
| (-11, 5, 4) | 4.93 | 17117.3 | 8620.0 | 2.0 | 76.1 | 110.3 | 1.422 | 0.748 | +0.674 |
| (-8, 6, 11) | 3.74 | -9112.5 | 5580.0 | -1.6 | 55.7 | 75.6 | 2.107 | 2.763 | -0.656 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `shaken` | 123.97 | 0.5611 | 6.658136633124035 | 1.2936131050297361 | 1.000 |
| `recovered_i` | 121.56 | 0.5772 | 7.030455137496274 | 1.3920065231717647 | 1.000 |