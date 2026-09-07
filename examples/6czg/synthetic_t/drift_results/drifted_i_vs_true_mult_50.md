# Held-Out Log-Likelihood Comparison: `drifted_i` vs `true_model`

## 1. Executive Summary

- **Scored Test Reflections (|T|)**: 900 (held-out)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `1.1555` nats/refl
- **Model B NLL**: `1.1765` nats/refl
- **Difference (Gain $\Delta$)**: `-0.0210` nats/refl (`+0.0210` nats NLL reduction)
- **Estimated Log Bayes Factor**: `-18.93` nats
- **Uncertainty**: Bootstrap SE = `0.0095` (95% CI: `[-0.0407, -0.0033]`) | Naive SE = `0.0072` (Ratio: `1.32`x)
- **Win Fraction $P(d_h > 0)$**: `41.9%` (377 wins, 523 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `1.28e-06`
- **Wilcoxon Signed-Rank $p$-value**: `2.14e-06`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.0272` | `-0.0061` | `+0.0210` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.0217` | `-0.0007` | `+0.0210` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`true_model`) | Model B (`drifted_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0069` | `-0.0085` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `1.3184` | `1.3033` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.1629` | `-0.1268` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -0.8849 | -0.7429 | -0.1420 | 0.0615 | 18.6% | 8/35 |
| 1 | 6.10 | 4.86 | 46 | 0.6455 | 0.7293 | -0.0837 | 0.0254 | 32.6% | 15/31 |
| 2 | 4.83 | 4.21 | 45 | 0.4586 | 0.4576 | +0.0010 | 0.0689 | 31.1% | 14/31 |
| 3 | 4.20 | 3.81 | 46 | 0.9819 | 0.9911 | -0.0092 | 0.0237 | 39.1% | 18/28 |
| 4 | 3.80 | 3.53 | 46 | 1.2523 | 1.3095 | -0.0573 | 0.0325 | 34.8% | 16/30 |
| 5 | 3.53 | 3.32 | 46 | 1.3688 | 1.4046 | -0.0358 | 0.0215 | 43.5% | 20/26 |
| 6 | 3.31 | 3.16 | 44 | 1.1069 | 1.0993 | +0.0076 | 0.0248 | 43.2% | 19/25 |
| 7 | 3.14 | 3.02 | 45 | 1.3792 | 1.4227 | -0.0436 | 0.0237 | 46.7% | 21/24 |
| 8 | 3.01 | 2.89 | 46 | 0.7427 | 0.7985 | -0.0559 | 0.0390 | 37.0% | 17/29 |
| 9 | 2.88 | 2.79 | 43 | 0.9986 | 1.0535 | -0.0550 | 0.0274 | 44.2% | 19/24 |
| 10 | 2.79 | 2.70 | 45 | 0.8386 | 0.8171 | +0.0215 | 0.0230 | 42.2% | 19/26 |
| 11 | 2.70 | 2.62 | 45 | 1.4649 | 1.4524 | +0.0125 | 0.0233 | 53.3% | 24/21 |
| 12 | 2.62 | 2.55 | 44 | 1.3107 | 1.3248 | -0.0141 | 0.0280 | 38.6% | 17/27 |
| 13 | 2.55 | 2.49 | 47 | 1.4068 | 1.4092 | -0.0024 | 0.0189 | 40.4% | 19/28 |
| 14 | 2.49 | 2.43 | 43 | 1.1857 | 1.1615 | +0.0242 | 0.0221 | 53.5% | 23/20 |
| 15 | 2.43 | 2.38 | 43 | 1.7405 | 1.7622 | -0.0217 | 0.0254 | 48.8% | 21/22 |
| 16 | 2.38 | 2.33 | 44 | 1.4157 | 1.4167 | -0.0010 | 0.0284 | 31.8% | 14/30 |
| 17 | 2.32 | 2.28 | 47 | 1.8088 | 1.8256 | -0.0169 | 0.0201 | 48.9% | 23/24 |
| 18 | 2.28 | 2.24 | 48 | 1.8633 | 1.8168 | +0.0464 | 0.0293 | 54.2% | 26/22 |
| 19 | 2.24 | 2.20 | 44 | 1.9094 | 1.9108 | -0.0013 | 0.0138 | 54.5% | 24/20 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (-3, 5, 11) | 4.26 | 10860.2 | 1340.0 | 8.1 | 59.3 | 63.6 | 5.953 | 3.080 | +2.873 |
| (-6, 2, 2) | 10.39 | 21050.1 | 765.0 | 27.5 | 154.3 | 165.0 | -0.896 | 1.044 | -1.940 |
| (8, 12, 16) | 2.25 | -1066.5 | 1855.0 | -0.6 | 64.9 | 54.3 | 4.214 | 2.899 | +1.315 |
| (11, 7, 14) | 2.98 | 16005.2 | 2775.0 | 5.8 | 120.3 | 106.3 | 0.264 | 1.193 | -0.929 |
| (-8, 6, 11) | 3.74 | -1596.9 | 2790.0 | -0.6 | 53.3 | 68.0 | 0.606 | 1.398 | -0.791 |
| (6, 2, 2) | 10.31 | 11568.0 | 515.0 | 22.5 | 111.1 | 116.7 | -2.249 | -1.459 | -0.790 |
| (6, 0, 7) | 7.24 | 3275.8 | 1265.0 | 2.6 | 22.1 | 30.3 | -0.236 | -1.016 | +0.779 |
| (-11, 7, 8) | 3.69 | -1111.2 | 6545.0 | -0.2 | 92.0 | 110.5 | 1.316 | 2.043 | -0.727 |
| (-13, 5, 16) | 2.96 | 13006.9 | 2680.0 | 4.9 | 100.3 | 90.1 | 0.639 | 1.339 | -0.700 |
| (-26, 8, 2) | 2.62 | 5215.5 | 1475.0 | 3.5 | 22.9 | 28.4 | 3.826 | 3.157 | +0.668 |
| (27, 7, 12) | 2.34 | 5610.5 | 4955.0 | 1.1 | 117.0 | 106.7 | 2.967 | 2.310 | +0.657 |
| (-9, 11, 8) | 2.82 | -1321.8 | 8560.0 | -0.2 | 131.7 | 145.3 | 3.582 | 4.208 | -0.627 |
| (-10, 10, 13) | 2.66 | -51.7 | 1855.0 | -0.0 | 56.5 | 64.7 | 1.605 | 2.225 | -0.620 |
| (-15, 7, 18) | 2.50 | 4465.8 | 2290.0 | 2.0 | 88.8 | 77.0 | 1.652 | 1.037 | +0.615 |
| (8, 8, 4) | 3.93 | 15309.2 | 7985.0 | 1.9 | 171.5 | 183.2 | 2.000 | 2.608 | -0.608 |
| (8, 4, 16) | 3.25 | 8413.7 | 12420.0 | 0.7 | 178.9 | 166.4 | 2.902 | 2.321 | +0.581 |
| (24, 6, 6) | 2.89 | -5949.6 | 5975.0 | -1.0 | 89.6 | 75.0 | 3.432 | 2.858 | +0.574 |
| (-32, 4, 8) | 2.42 | 4536.4 | 1825.0 | 2.5 | 51.0 | 40.1 | 1.306 | 1.876 | -0.570 |
| (30, 4, 4) | 2.65 | 4345.9 | 1565.0 | 2.8 | 45.0 | 55.9 | 1.127 | 0.564 | +0.563 |
| (-2, 2, 5) | 9.64 | 5606.6 | 1410.0 | 4.0 | 51.5 | 54.7 | -0.467 | -1.022 | +0.556 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `true_model` | 73.06 | 0.7118 | 8.316639414194936 | 2.2151127829374033 | 1.000 |
| `drifted_i` | 74.50 | 0.7062 | 8.882128474088304 | 2.549883001187861 | 1.000 |