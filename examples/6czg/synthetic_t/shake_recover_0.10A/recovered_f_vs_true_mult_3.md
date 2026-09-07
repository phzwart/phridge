# Held-Out Log-Likelihood Comparison: `recovered_f` vs `deposited`

## 1. Executive Summary

- **Scored Test Reflections (|T|)**: 900 (held-out)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `-0.2639` nats/refl
- **Model B NLL**: `-0.0512` nats/refl
- **Difference (Gain $\Delta$)**: `-0.2127` nats/refl (`+0.2127` nats NLL reduction)
- **Estimated Log Bayes Factor**: `-191.45` nats
- **Uncertainty**: Bootstrap SE = `0.0222` (95% CI: `[-0.2569, -0.1698]`) | Naive SE = `0.0136` (Ratio: `1.63`x)
- **Win Fraction $P(d_h > 0)$**: `20.6%` (185 wins, 715 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `3.83e-74`
- **Wilcoxon Signed-Rank $p$-value**: `7.35e-65`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.3012` | `-0.0884` | `+0.2127` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.1653` | `+0.0474` | `+0.2127` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`deposited`) | Model B (`recovered_f`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0030` | `-0.0091` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `-0.1682` | `-0.0493` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.0957` | `-0.0019` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -1.4259 | -1.1295 | -0.2964 | 0.0274 | 4.7% | 2/41 |
| 1 | 6.10 | 4.86 | 46 | 0.0724 | 0.4672 | -0.3948 | 0.0475 | 2.2% | 1/45 |
| 2 | 4.83 | 4.21 | 45 | -0.5815 | -0.2400 | -0.3415 | 0.0422 | 2.2% | 1/44 |
| 3 | 4.20 | 3.81 | 46 | -0.1444 | 0.1237 | -0.2681 | 0.0550 | 8.7% | 4/42 |
| 4 | 3.80 | 3.53 | 46 | -0.0121 | 0.3648 | -0.3769 | 0.0699 | 10.9% | 5/41 |
| 5 | 3.53 | 3.32 | 46 | 0.1315 | 0.3678 | -0.2363 | 0.0516 | 19.6% | 9/37 |
| 6 | 3.31 | 3.16 | 44 | -0.4101 | -0.1147 | -0.2954 | 0.0485 | 11.4% | 5/39 |
| 7 | 3.14 | 3.02 | 45 | 0.3277 | 0.4882 | -0.1604 | 0.1025 | 28.9% | 13/32 |
| 8 | 3.01 | 2.89 | 46 | -0.4378 | -0.1564 | -0.2814 | 0.0499 | 10.9% | 5/41 |
| 9 | 2.88 | 2.79 | 43 | -0.4125 | -0.1468 | -0.2657 | 0.0676 | 23.3% | 10/33 |
| 10 | 2.79 | 2.70 | 45 | -0.6341 | -0.4672 | -0.1668 | 0.0430 | 15.6% | 7/38 |
| 11 | 2.70 | 2.62 | 45 | -0.2527 | -0.0772 | -0.1755 | 0.0520 | 28.9% | 13/32 |
| 12 | 2.62 | 2.55 | 44 | -0.3635 | -0.2564 | -0.1071 | 0.0547 | 27.3% | 12/32 |
| 13 | 2.55 | 2.49 | 47 | -0.2721 | -0.0941 | -0.1780 | 0.0587 | 27.7% | 13/34 |
| 14 | 2.49 | 2.43 | 43 | -0.5054 | -0.3923 | -0.1131 | 0.0404 | 32.6% | 14/29 |
| 15 | 2.43 | 2.38 | 43 | 0.1747 | 0.2453 | -0.0707 | 0.0884 | 37.2% | 16/27 |
| 16 | 2.38 | 2.33 | 44 | -0.3149 | -0.1398 | -0.1750 | 0.0608 | 27.3% | 12/32 |
| 17 | 2.32 | 2.28 | 47 | -0.0469 | 0.0658 | -0.1127 | 0.0641 | 29.8% | 14/33 |
| 18 | 2.28 | 2.24 | 48 | -0.1876 | -0.0428 | -0.1448 | 0.0499 | 27.1% | 13/35 |
| 19 | 2.24 | 2.20 | 44 | -0.0691 | 0.0153 | -0.0844 | 0.0702 | 36.4% | 16/28 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (27, 1, 2) | 3.11 | 60265.8 | 2562.0 | 23.5 | 291.0 | 288.6 | 5.915 | 3.169 | +2.746 |
| (16, 8, 17) | 2.43 | -222.1 | 145.2 | -1.5 | 12.8 | 26.6 | 0.952 | 3.150 | -2.197 |
| (2, 2, 14) | 4.11 | 126040.1 | 3423.3 | 36.8 | 399.8 | 434.6 | 3.390 | 5.441 | -2.051 |
| (9, 3, 14) | 3.66 | 120355.5 | 3329.7 | 36.1 | 412.7 | 445.1 | 6.003 | 7.939 | -1.936 |
| (-12, 6, 14) | 3.11 | 10755.9 | 325.5 | 33.0 | 117.1 | 138.9 | 0.445 | 2.219 | -1.775 |
| (1, 7, 2) | 4.96 | 78832.9 | 4110.3 | 19.2 | 318.6 | 352.9 | 1.900 | 3.596 | -1.696 |
| (35, 1, 10) | 2.23 | 7399.3 | 325.2 | 22.8 | 101.0 | 97.3 | 2.956 | 1.313 | +1.644 |
| (30, 2, 14) | 2.32 | 10623.7 | 519.0 | 20.5 | 119.0 | 111.9 | 2.656 | 1.034 | +1.622 |
| (7, 1, 22) | 2.63 | 2926.4 | 201.3 | 14.5 | 64.1 | 77.9 | 0.198 | 1.707 | -1.509 |
| (26, 4, 2) | 3.04 | 6977.4 | 132.9 | 52.5 | 94.0 | 112.9 | 0.103 | 1.599 | -1.496 |
| (21, 3, 2) | 3.79 | 57117.9 | 2869.2 | 19.9 | 271.0 | 298.7 | 1.397 | 2.880 | -1.482 |
| (0, 12, 14) | 2.42 | 11538.4 | 270.9 | 42.6 | 125.0 | 121.4 | 2.767 | 1.292 | +1.475 |
| (27, 7, 12) | 2.34 | 12562.0 | 297.3 | 42.3 | 126.9 | 117.2 | 2.295 | 0.841 | +1.454 |
| (20, 8, 0) | 3.06 | 30947.7 | 2788.8 | 11.1 | 210.6 | 206.0 | 3.297 | 1.851 | +1.447 |
| (-13, 11, 2) | 2.87 | 1712.8 | 167.7 | 10.2 | 49.7 | 65.7 | -0.491 | 0.954 | -1.445 |
| (20, 2, 14) | 2.94 | 8279.1 | 444.3 | 18.6 | 98.7 | 118.8 | -0.082 | 1.325 | -1.407 |
| (-12, 4, 8) | 4.46 | 77065.1 | 3085.8 | 25.0 | 312.8 | 344.6 | 1.217 | 2.614 | -1.397 |
| (23, 3, 8) | 3.17 | 4346.3 | 193.2 | 22.5 | 76.5 | 96.5 | -0.429 | 0.914 | -1.344 |
| (0, 4, 12) | 4.33 | 81200.9 | 1243.2 | 65.3 | 321.6 | 350.9 | 1.431 | 2.726 | -1.295 |
| (14, 4, 0) | 5.00 | 55468.7 | 1602.0 | 34.6 | 265.8 | 294.3 | 1.267 | 2.560 | -1.292 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `deposited` | 4.17 | 0.9987 | 199.20962731196659 | 579.3022966013517 | 1.000 |
| `recovered_f` | 10.17 | 0.9963 | 199.19333434868048 | 792.8839187958348 | 1.000 |