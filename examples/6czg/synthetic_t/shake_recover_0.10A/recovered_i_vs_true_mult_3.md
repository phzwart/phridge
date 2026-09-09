# Held-Out Log-Likelihood Comparison: `recovered_i` vs `deposited`

## 1. Executive Summary

- **Scored Audit Reflections (|A|)**: 900 (audit set)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `-0.2639` nats/refl
- **Model B NLL**: `-0.0498` nats/refl
- **Difference (Gain $\Delta$)**: `-0.2141` nats/refl (`+0.2141` nats NLL reduction)
- **Estimated Log Bayes Factor**: `-192.66` nats
- **Uncertainty**: Bootstrap SE = `0.0226` (95% CI: `[-0.2597, -0.1701]`) | Naive SE = `0.0136` (Ratio: `1.66`x)
- **Win Fraction $P(d_h > 0)$**: `20.9%` (188 wins, 712 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `2.15e-72`
- **Wilcoxon Signed-Rank $p$-value**: `2.53e-65`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.3039` | `-0.0898` | `+0.2141` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.1660` | `+0.0481` | `+0.2141` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`deposited`) | Model B (`recovered_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0030` | `-0.0091` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `-0.1682` | `-0.0479` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.0957` | `-0.0019` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -1.4259 | -1.1286 | -0.2972 | 0.0271 | 4.7% | 2/41 |
| 1 | 6.10 | 4.86 | 46 | 0.0724 | 0.4733 | -0.4009 | 0.0487 | 2.2% | 1/45 |
| 2 | 4.83 | 4.21 | 45 | -0.5815 | -0.2386 | -0.3429 | 0.0423 | 2.2% | 1/44 |
| 3 | 4.20 | 3.81 | 46 | -0.1444 | 0.1296 | -0.2740 | 0.0552 | 10.9% | 5/41 |
| 4 | 3.80 | 3.53 | 46 | -0.0121 | 0.3703 | -0.3824 | 0.0707 | 10.9% | 5/41 |
| 5 | 3.53 | 3.32 | 46 | 0.1315 | 0.3700 | -0.2385 | 0.0506 | 19.6% | 9/37 |
| 6 | 3.31 | 3.16 | 44 | -0.4101 | -0.1093 | -0.3008 | 0.0488 | 11.4% | 5/39 |
| 7 | 3.14 | 3.02 | 45 | 0.3277 | 0.4878 | -0.1601 | 0.1031 | 31.1% | 14/31 |
| 8 | 3.01 | 2.89 | 46 | -0.4378 | -0.1530 | -0.2847 | 0.0501 | 10.9% | 5/41 |
| 9 | 2.88 | 2.79 | 43 | -0.4125 | -0.1452 | -0.2673 | 0.0675 | 20.9% | 9/34 |
| 10 | 2.79 | 2.70 | 45 | -0.6341 | -0.4650 | -0.1690 | 0.0431 | 15.6% | 7/38 |
| 11 | 2.70 | 2.62 | 45 | -0.2527 | -0.0797 | -0.1730 | 0.0512 | 31.1% | 14/31 |
| 12 | 2.62 | 2.55 | 44 | -0.3635 | -0.2554 | -0.1081 | 0.0548 | 27.3% | 12/32 |
| 13 | 2.55 | 2.49 | 47 | -0.2721 | -0.0968 | -0.1752 | 0.0582 | 27.7% | 13/34 |
| 14 | 2.49 | 2.43 | 43 | -0.5054 | -0.3928 | -0.1126 | 0.0405 | 34.9% | 15/28 |
| 15 | 2.43 | 2.38 | 43 | 0.1747 | 0.2425 | -0.0678 | 0.0876 | 37.2% | 16/27 |
| 16 | 2.38 | 2.33 | 44 | -0.3149 | -0.1394 | -0.1755 | 0.0608 | 27.3% | 12/32 |
| 17 | 2.32 | 2.28 | 47 | -0.0469 | 0.0676 | -0.1146 | 0.0642 | 29.8% | 14/33 |
| 18 | 2.28 | 2.24 | 48 | -0.1876 | -0.0451 | -0.1425 | 0.0496 | 27.1% | 13/35 |
| 19 | 2.24 | 2.20 | 44 | -0.0691 | 0.0155 | -0.0846 | 0.0700 | 36.4% | 16/28 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (27, 1, 2) | 3.11 | 60265.8 | 2562.0 | 23.5 | 291.0 | 288.7 | 5.915 | 3.175 | +2.741 |
| (16, 8, 17) | 2.43 | -222.1 | 145.2 | -1.5 | 12.8 | 26.5 | 0.952 | 3.127 | -2.175 |
| (2, 2, 14) | 4.11 | 126040.1 | 3423.3 | 36.8 | 399.8 | 434.5 | 3.390 | 5.402 | -2.013 |
| (9, 3, 14) | 3.66 | 120355.5 | 3329.7 | 36.1 | 412.7 | 445.5 | 6.003 | 7.970 | -1.967 |
| (-12, 6, 14) | 3.11 | 10755.9 | 325.5 | 33.0 | 117.1 | 138.8 | 0.445 | 2.204 | -1.759 |
| (1, 7, 2) | 4.96 | 78832.9 | 4110.3 | 19.2 | 318.6 | 353.6 | 1.900 | 3.644 | -1.744 |
| (35, 1, 10) | 2.23 | 7399.3 | 325.2 | 22.8 | 101.0 | 97.2 | 2.956 | 1.292 | +1.664 |
| (30, 2, 14) | 2.32 | 10623.7 | 519.0 | 20.5 | 119.0 | 112.0 | 2.656 | 1.037 | +1.619 |
| (26, 4, 2) | 3.04 | 6977.4 | 132.9 | 52.5 | 94.0 | 113.2 | 0.103 | 1.628 | -1.525 |
| (7, 1, 22) | 2.63 | 2926.4 | 201.3 | 14.5 | 64.1 | 77.8 | 0.198 | 1.680 | -1.482 |
| (0, 12, 14) | 2.42 | 11538.4 | 270.9 | 42.6 | 125.0 | 121.5 | 2.767 | 1.300 | +1.467 |
| (20, 8, 0) | 3.06 | 30947.7 | 2788.8 | 11.1 | 210.6 | 205.9 | 3.297 | 1.834 | +1.463 |
| (21, 3, 2) | 3.79 | 57117.9 | 2869.2 | 19.9 | 271.0 | 298.5 | 1.397 | 2.856 | -1.458 |
| (27, 7, 12) | 2.34 | 12562.0 | 297.3 | 42.3 | 126.9 | 117.0 | 2.295 | 0.838 | +1.457 |
| (-13, 11, 2) | 2.87 | 1712.8 | 167.7 | 10.2 | 49.7 | 65.7 | -0.491 | 0.939 | -1.430 |
| (20, 2, 14) | 2.94 | 8279.1 | 444.3 | 18.6 | 98.7 | 118.8 | -0.082 | 1.322 | -1.405 |
| (-12, 4, 8) | 4.46 | 77065.1 | 3085.8 | 25.0 | 312.8 | 344.7 | 1.217 | 2.608 | -1.392 |
| (23, 3, 8) | 3.17 | 4346.3 | 193.2 | 22.5 | 76.5 | 96.6 | -0.429 | 0.919 | -1.349 |
| (0, 4, 12) | 4.33 | 81200.9 | 1243.2 | 65.3 | 321.6 | 351.4 | 1.431 | 2.760 | -1.328 |
| (14, 4, 0) | 5.00 | 55468.7 | 1602.0 | 34.6 | 265.8 | 294.7 | 1.267 | 2.581 | -1.313 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `deposited` | 4.17 | 0.9987 | 199.20962731196659 | 579.3022966013517 | 1.000 |
| `recovered_i` | 10.19 | 0.9963 | 199.19342066648593 | 792.7415793823787 | 1.000 |