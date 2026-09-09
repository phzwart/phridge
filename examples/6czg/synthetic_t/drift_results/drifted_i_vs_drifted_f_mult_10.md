# Held-Out Log-Likelihood Comparison: `drifted_i` vs `drifted_f`

## 1. Executive Summary

- **Scored Audit Reflections (|A|)**: 900 (audit set)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `0.1586` nats/refl
- **Model B NLL**: `0.1584` nats/refl
- **Difference (Gain $\Delta$)**: `+0.0002` nats/refl (`-0.0002` nats NLL reduction)
- **Estimated Log Bayes Factor**: `+0.15` nats
- **Uncertainty**: Bootstrap SE = `0.0041` (95% CI: `[-0.0082, +0.0079]`) | Naive SE = `0.0030` (Ratio: `1.38`x)
- **Win Fraction $P(d_h > 0)$**: `46.0%` (414 wins, 486 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `1.79e-02`
- **Wilcoxon Signed-Rank $p$-value**: `1.12e-01`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.0004` | `-0.0006` | `-0.0002` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.0004` | `-0.0006` | `-0.0002` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`drifted_f`) | Model B (`drifted_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0110` | `-0.0117` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `0.1789` | `0.1764` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.0203` | `-0.0180` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -1.1287 | -1.1387 | +0.0100 | 0.0090 | 62.8% | 27/16 |
| 1 | 6.10 | 4.86 | 46 | 0.3694 | 0.4033 | -0.0339 | 0.0102 | 23.9% | 11/35 |
| 2 | 4.83 | 4.21 | 45 | -0.2652 | -0.2334 | -0.0318 | 0.0134 | 42.2% | 19/26 |
| 3 | 4.20 | 3.81 | 46 | 0.1064 | 0.1093 | -0.0029 | 0.0125 | 41.3% | 19/27 |
| 4 | 3.80 | 3.53 | 46 | 0.3735 | 0.3978 | -0.0243 | 0.0143 | 34.8% | 16/30 |
| 5 | 3.53 | 3.32 | 46 | 0.5448 | 0.5181 | +0.0267 | 0.0144 | 56.5% | 26/20 |
| 6 | 3.31 | 3.16 | 44 | -0.0268 | -0.0331 | +0.0063 | 0.0127 | 59.1% | 26/18 |
| 7 | 3.14 | 3.02 | 45 | 0.5434 | 0.5463 | -0.0029 | 0.0112 | 46.7% | 21/24 |
| 8 | 3.01 | 2.89 | 46 | -0.0830 | -0.0765 | -0.0065 | 0.0119 | 54.3% | 25/21 |
| 9 | 2.88 | 2.79 | 43 | 0.2528 | 0.2297 | +0.0230 | 0.0169 | 44.2% | 19/24 |
| 10 | 2.79 | 2.70 | 45 | -0.2448 | -0.2312 | -0.0135 | 0.0113 | 42.2% | 19/26 |
| 11 | 2.70 | 2.62 | 45 | 0.2999 | 0.2989 | +0.0010 | 0.0147 | 42.2% | 19/26 |
| 12 | 2.62 | 2.55 | 44 | 0.1182 | 0.0993 | +0.0189 | 0.0114 | 56.8% | 25/19 |
| 13 | 2.55 | 2.49 | 47 | 0.1506 | 0.1349 | +0.0157 | 0.0105 | 48.9% | 23/24 |
| 14 | 2.49 | 2.43 | 43 | -0.1014 | -0.1172 | +0.0159 | 0.0179 | 53.5% | 23/20 |
| 15 | 2.43 | 2.38 | 43 | 0.3157 | 0.3187 | -0.0030 | 0.0121 | 27.9% | 12/31 |
| 16 | 2.38 | 2.33 | 44 | 0.2916 | 0.3127 | -0.0210 | 0.0125 | 43.2% | 19/25 |
| 17 | 2.32 | 2.28 | 47 | 0.5840 | 0.5938 | -0.0098 | 0.0120 | 34.0% | 16/31 |
| 18 | 2.28 | 2.24 | 48 | 0.3936 | 0.3749 | +0.0188 | 0.0139 | 58.3% | 28/20 |
| 19 | 2.24 | 2.20 | 44 | 0.5809 | 0.5623 | +0.0186 | 0.0142 | 47.7% | 21/23 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (2, 14, 12) | 2.24 | 630.8 | 250.0 | 2.5 | 42.8 | 40.7 | 2.046 | 1.538 | +0.509 |
| (33, 3, 4) | 2.47 | 6128.6 | 1152.0 | 5.3 | 62.4 | 68.3 | 1.388 | 0.883 | +0.505 |
| (26, 4, 8) | 2.82 | 1150.5 | 482.0 | 2.4 | 57.2 | 52.7 | 0.847 | 0.356 | +0.491 |
| (15, 7, 18) | 2.47 | 6596.8 | 680.0 | 9.7 | 97.8 | 94.2 | 1.410 | 0.963 | +0.448 |
| (-27, 11, 4) | 2.22 | 1793.3 | 497.0 | 3.6 | 64.4 | 62.8 | 2.950 | 2.537 | +0.412 |
| (-19, 1, 10) | 3.58 | 56417.3 | 4609.0 | 12.2 | 275.6 | 281.5 | 1.668 | 2.074 | -0.406 |
| (24, 6, 6) | 2.89 | 5402.6 | 1195.0 | 4.5 | 95.8 | 99.7 | 1.078 | 1.438 | -0.361 |
| (-30, 4, 2) | 2.69 | 10696.4 | 1011.0 | 10.6 | 128.9 | 126.5 | 2.125 | 1.775 | +0.350 |
| (0, 4, 12) | 4.33 | 75649.3 | 4144.0 | 18.3 | 334.6 | 338.9 | 2.491 | 2.834 | -0.343 |
| (15, 3, 6) | 4.51 | 41374.5 | 5314.0 | 7.8 | 238.9 | 247.0 | 0.820 | 1.154 | -0.334 |
| (13, 13, 2) | 2.50 | 674.7 | 317.0 | 2.1 | 48.5 | 46.8 | 1.509 | 1.195 | +0.313 |
| (-12, 8, 19) | 2.41 | 4025.3 | 782.0 | 5.1 | 85.8 | 84.1 | 2.216 | 1.908 | +0.308 |
| (-30, 2, 15) | 2.30 | 2565.5 | 769.0 | 3.3 | 68.5 | 66.3 | 1.637 | 1.329 | +0.308 |
| (11, 1, 12) | 4.12 | 43373.1 | 4032.0 | 10.8 | 241.2 | 246.6 | 1.376 | 1.676 | -0.300 |
| (-17, 13, 4) | 2.36 | 3016.9 | 677.0 | 4.5 | 79.5 | 80.7 | 3.010 | 3.307 | -0.297 |
| (8, 8, 4) | 3.93 | 27921.8 | 1597.0 | 17.5 | 201.9 | 198.0 | 1.528 | 1.244 | +0.283 |
| (11, 9, 4) | 3.40 | 11179.2 | 3979.0 | 2.8 | 169.3 | 167.0 | 3.871 | 3.595 | +0.276 |
| (15, 9, 12) | 2.69 | 245.4 | 318.0 | 0.8 | 28.1 | 24.4 | -0.499 | -0.758 | +0.259 |
| (24, 4, 7) | 3.05 | 1277.5 | 396.0 | 3.2 | 52.4 | 48.6 | -0.097 | -0.354 | +0.257 |
| (-23, 3, 13) | 2.81 | -455.1 | 508.0 | -0.9 | 35.5 | 33.3 | 1.638 | 1.381 | +0.257 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `drifted_f` | 14.56 | 0.9917 | 199.15490824739788 | 432.57052806047244 | 1.000 |
| `drifted_i` | 14.52 | 0.9916 | 199.15699829036046 | 430.76400237912554 | 1.000 |