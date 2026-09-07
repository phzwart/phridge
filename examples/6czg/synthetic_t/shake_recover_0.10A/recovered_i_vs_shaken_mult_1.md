# Held-Out Log-Likelihood Comparison: `recovered_i` vs `shaken`

## 1. Executive Summary

- **Scored Test Reflections (|T|)**: 900 (held-out)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `0.0559` nats/refl
- **Model B NLL**: `-0.1147` nats/refl
- **Difference (Gain $\Delta$)**: `+0.1706` nats/refl (`-0.1706` nats NLL reduction)
- **Estimated Log Bayes Factor**: `+153.50` nats
- **Uncertainty**: Bootstrap SE = `0.0727` (95% CI: `[+0.0170, +0.2976]`) | Naive SE = `0.0269` (Ratio: `2.70`x)
- **Win Fraction $P(d_h > 0)$**: `64.4%` (580 wins, 320 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `3.50e-18`
- **Wilcoxon Signed-Rank $p$-value**: `1.16e-16`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `-0.0746` | `-0.0959` | `-0.1706` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `-0.2376` | `+0.0671` | `-0.1706` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`shaken`) | Model B (`recovered_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0081` | `-0.0087` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `0.1495` | `-0.1107` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.0936` | `-0.0039` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -1.3587 | -1.1354 | -0.2233 | 0.1749 | 27.9% | 12/31 |
| 1 | 6.10 | 4.86 | 46 | -0.1755 | 0.4908 | -0.6663 | 0.1677 | 19.6% | 9/37 |
| 2 | 4.83 | 4.21 | 45 | -0.5098 | -0.2704 | -0.2395 | 0.0882 | 22.2% | 10/35 |
| 3 | 4.20 | 3.81 | 46 | 0.2042 | 0.0821 | +0.1220 | 0.1103 | 50.0% | 23/23 |
| 4 | 3.80 | 3.53 | 46 | 0.0728 | 0.3070 | -0.2343 | 0.1430 | 37.0% | 17/29 |
| 5 | 3.53 | 3.32 | 46 | 0.3293 | 0.3815 | -0.0522 | 0.1679 | 45.7% | 21/25 |
| 6 | 3.31 | 3.16 | 44 | -0.1130 | -0.1166 | +0.0035 | 0.0772 | 50.0% | 22/22 |
| 7 | 3.14 | 3.02 | 45 | 0.3806 | 0.4014 | -0.0207 | 0.0959 | 55.6% | 25/20 |
| 8 | 3.01 | 2.89 | 46 | 0.0045 | -0.1631 | +0.1676 | 0.0825 | 69.6% | 32/14 |
| 9 | 2.88 | 2.79 | 43 | 0.1188 | -0.1650 | +0.2838 | 0.0984 | 67.4% | 29/14 |
| 10 | 2.79 | 2.70 | 45 | -0.2393 | -0.5319 | +0.2926 | 0.0744 | 75.6% | 34/11 |
| 11 | 2.70 | 2.62 | 45 | 0.1678 | -0.1542 | +0.3220 | 0.1049 | 68.9% | 31/14 |
| 12 | 2.62 | 2.55 | 44 | -0.0826 | -0.3430 | +0.2604 | 0.0795 | 81.8% | 36/8 |
| 13 | 2.55 | 2.49 | 47 | 0.2537 | -0.1687 | +0.4224 | 0.0971 | 87.2% | 41/6 |
| 14 | 2.49 | 2.43 | 43 | 0.1271 | -0.5181 | +0.6452 | 0.0854 | 95.3% | 41/2 |
| 15 | 2.43 | 2.38 | 43 | 0.5296 | 0.1051 | +0.4245 | 0.1103 | 83.7% | 36/7 |
| 16 | 2.38 | 2.33 | 44 | 0.2462 | -0.2058 | +0.4520 | 0.1050 | 79.5% | 35/9 |
| 17 | 2.32 | 2.28 | 47 | 0.3589 | -0.0982 | +0.4571 | 0.0720 | 89.4% | 42/5 |
| 18 | 2.28 | 2.24 | 48 | 0.3050 | -0.1486 | +0.4537 | 0.0721 | 89.6% | 43/5 |
| 19 | 2.24 | 2.20 | 44 | 0.4257 | -0.1389 | +0.5646 | 0.0716 | 93.2% | 41/3 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (3, 1, 0) | 22.09 | 21386.8 | 181.5 | 117.8 | 192.8 | 175.1 | 5.840 | -0.355 | +6.195 |
| (20, 4, 7) | 3.47 | 512.2 | 35.5 | 14.4 | 80.5 | 31.5 | 3.701 | -1.283 | +4.984 |
| (-13, 1, 8) | 4.90 | 286007.4 | 1880.0 | 152.1 | 599.3 | 649.2 | 3.984 | 8.788 | -4.804 |
| (14, 2, 4) | 5.33 | 191022.5 | 1524.0 | 125.3 | 472.5 | 534.9 | 1.630 | 6.200 | -4.570 |
| (9, 3, 14) | 3.66 | 132444.0 | 1109.9 | 119.3 | 401.7 | 446.1 | 1.719 | 5.945 | -4.226 |
| (-18, 8, 15) | 2.52 | 1778.8 | 57.7 | 30.8 | 81.1 | 49.2 | 3.680 | -0.092 | +3.773 |
| (-30, 2, 14) | 2.35 | 1361.7 | 42.0 | 32.4 | 69.9 | 41.8 | 2.975 | -0.206 | +3.181 |
| (-5, 9, 1) | 3.82 | 3600.7 | 91.5 | 39.4 | 112.3 | 81.9 | 2.875 | 0.020 | +2.855 |
| (-12, 8, 6) | 3.51 | 50266.4 | 424.9 | 118.3 | 250.0 | 279.5 | 1.274 | 4.126 | -2.852 |
| (16, 10, 13) | 2.46 | 68.7 | 27.2 | 2.5 | 38.5 | 15.0 | 1.105 | -1.610 | +2.714 |
| (1, 7, 2) | 4.96 | 78873.8 | 1370.1 | 57.6 | 314.7 | 354.7 | 1.400 | 4.088 | -2.688 |
| (-29, 9, 3) | 2.33 | 10.5 | 27.9 | 0.4 | 31.3 | 11.1 | 1.263 | -1.398 | +2.661 |
| (0, 4, 12) | 4.33 | 80152.9 | 414.4 | 193.4 | 296.0 | 352.3 | 0.425 | 3.074 | -2.649 |
| (-7, 1, 22) | 2.64 | 4700.8 | 77.9 | 60.3 | 111.0 | 88.5 | 3.861 | 1.288 | +2.574 |
| (-18, 4, 10) | 3.43 | 54166.1 | 643.3 | 84.2 | 262.4 | 285.6 | 1.536 | 3.971 | -2.435 |
| (14, 2, 12) | 3.73 | 10230.1 | 297.4 | 34.4 | 155.6 | 127.9 | 2.578 | 0.308 | +2.270 |
| (7, 9, 19) | 2.39 | 1777.9 | 59.4 | 29.9 | 69.4 | 45.5 | 2.133 | -0.129 | +2.262 |
| (-20, 4, 15) | 2.77 | -1.2 | 37.4 | -0.0 | 34.5 | 5.7 | 0.343 | -1.867 | +2.210 |
| (9, 5, 4) | 5.27 | 126633.4 | 1439.0 | 88.0 | 381.5 | 424.0 | 1.100 | 3.300 | -2.200 |
| (16, 8, 11) | 2.86 | 283.8 | 29.7 | 9.6 | 48.1 | 23.3 | 0.885 | -1.306 | +2.191 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `shaken` | 18.05 | 0.9885 | 199.02133531797284 | 1000.0 | 1.000 |
| `recovered_i` | 9.02 | 0.9975 | 199.16555110781067 | 1000.0 | 1.000 |