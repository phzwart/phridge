# Held-Out Log-Likelihood Comparison: `recovered_f` vs `deposited`

## 1. Executive Summary

- **Scored Test Reflections (|T|)**: 900 (held-out)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `1.8351` nats/refl
- **Model B NLL**: `1.8455` nats/refl
- **Difference (Gain $\Delta$)**: `-0.0104` nats/refl (`+0.0104` nats NLL reduction)
- **Estimated Log Bayes Factor**: `-9.36` nats
- **Uncertainty**: Bootstrap SE = `0.0048` (95% CI: `[-0.0203, -0.0014]`) | Naive SE = `0.0055` (Ratio: `0.87`x)
- **Win Fraction $P(d_h > 0)$**: `48.0%` (432 wins, 468 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `2.43e-01`
- **Wilcoxon Signed-Rank $p$-value**: `2.27e-01`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.0104` | `-0.0000` | `+0.0104` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.0093` | `+0.0011` | `+0.0104` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`deposited`) | Model B (`recovered_f`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0277` | `-0.0274` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `1.9787` | `1.9824` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.1437` | `-0.1369` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -0.3826 | -0.3752 | -0.0074 | 0.0283 | 51.2% | 22/21 |
| 1 | 6.10 | 4.86 | 46 | 1.4398 | 1.5142 | -0.0744 | 0.0328 | 34.8% | 16/30 |
| 2 | 4.83 | 4.21 | 45 | 0.9462 | 0.9570 | -0.0107 | 0.0313 | 53.3% | 24/21 |
| 3 | 4.20 | 3.81 | 46 | 1.8347 | 1.8121 | +0.0226 | 0.0219 | 52.2% | 24/22 |
| 4 | 3.80 | 3.53 | 46 | 1.7472 | 1.7924 | -0.0452 | 0.0264 | 43.5% | 20/26 |
| 5 | 3.53 | 3.32 | 46 | 1.9031 | 1.9192 | -0.0161 | 0.0202 | 45.7% | 21/25 |
| 6 | 3.31 | 3.16 | 44 | 1.4958 | 1.5062 | -0.0104 | 0.0180 | 54.5% | 24/20 |
| 7 | 3.14 | 3.02 | 45 | 2.2567 | 2.2610 | -0.0042 | 0.0179 | 46.7% | 21/24 |
| 8 | 3.01 | 2.89 | 46 | 1.6764 | 1.6949 | -0.0185 | 0.0309 | 45.7% | 21/25 |
| 9 | 2.88 | 2.79 | 43 | 1.8756 | 1.8932 | -0.0176 | 0.0278 | 44.2% | 19/24 |
| 10 | 2.79 | 2.70 | 45 | 1.8283 | 1.8092 | +0.0192 | 0.0295 | 48.9% | 22/23 |
| 11 | 2.70 | 2.62 | 45 | 1.8540 | 1.8606 | -0.0066 | 0.0230 | 46.7% | 21/24 |
| 12 | 2.62 | 2.55 | 44 | 1.9875 | 1.9980 | -0.0104 | 0.0285 | 45.5% | 20/24 |
| 13 | 2.55 | 2.49 | 47 | 1.8978 | 1.8930 | +0.0048 | 0.0177 | 48.9% | 23/24 |
| 14 | 2.49 | 2.43 | 43 | 2.2461 | 2.2267 | +0.0193 | 0.0268 | 55.8% | 24/19 |
| 15 | 2.43 | 2.38 | 43 | 2.3321 | 2.3517 | -0.0196 | 0.0224 | 46.5% | 20/23 |
| 16 | 2.38 | 2.33 | 44 | 2.2999 | 2.3124 | -0.0124 | 0.0224 | 59.1% | 26/18 |
| 17 | 2.32 | 2.28 | 47 | 2.3000 | 2.3083 | -0.0083 | 0.0147 | 44.7% | 21/26 |
| 18 | 2.28 | 2.24 | 48 | 2.4980 | 2.5016 | -0.0037 | 0.0170 | 52.1% | 25/23 |
| 19 | 2.24 | 2.20 | 44 | 2.5761 | 2.5833 | -0.0071 | 0.0208 | 40.9% | 18/26 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (3, 3, 11) | 4.83 | 8593.0 | 2820.0 | 3.0 | 25.7 | 51.1 | 2.501 | 1.395 | +1.105 |
| (19, 9, 0) | 2.95 | -8204.7 | 10170.0 | -0.8 | 121.1 | 143.9 | 3.577 | 4.454 | -0.876 |
| (10, 4, 5) | 5.42 | -6310.2 | 2510.0 | -2.5 | 52.4 | 66.6 | 3.484 | 4.347 | -0.863 |
| (13, 11, 0) | 2.88 | 3394.9 | 2530.0 | 1.3 | 62.3 | 81.7 | 0.327 | 1.142 | -0.815 |
| (2, 6, 0) | 5.83 | 23774.8 | 2760.0 | 8.6 | 148.2 | 139.4 | -0.452 | 0.349 | -0.801 |
| (-3, 13, 6) | 2.61 | 28315.0 | 6420.0 | 4.4 | 98.9 | 81.9 | 4.646 | 5.357 | -0.711 |
| (15, 7, 18) | 2.47 | 19932.4 | 6800.0 | 2.9 | 90.9 | 110.0 | 3.265 | 2.559 | +0.706 |
| (30, 4, 4) | 2.65 | -2125.9 | 3130.0 | -0.7 | 45.1 | 60.5 | 1.688 | 2.374 | -0.686 |
| (-12, 8, 5) | 3.58 | 6320.8 | 2780.0 | 2.3 | 56.9 | 40.5 | 0.095 | 0.774 | -0.680 |
| (13, 11, 6) | 2.76 | 8140.9 | 2140.0 | 3.8 | 30.3 | 42.9 | 3.991 | 3.327 | +0.664 |
| (-8, 6, 11) | 3.74 | -9112.5 | 5580.0 | -1.6 | 55.4 | 73.9 | 2.109 | 2.742 | -0.633 |
| (-20, 2, 14) | 2.98 | 24128.3 | 8950.0 | 2.7 | 130.1 | 108.0 | 1.582 | 2.204 | -0.622 |
| (6, 2, 2) | 10.31 | 10499.1 | 1030.0 | 10.2 | 110.9 | 116.6 | -1.640 | -1.053 | -0.587 |
| (3, 7, 24) | 2.22 | -3013.7 | 3780.0 | -0.8 | 50.0 | 23.4 | 2.776 | 2.197 | +0.579 |
| (-14, 2, 1) | 5.72 | 775.4 | 2120.0 | 0.4 | 41.9 | 56.3 | -0.844 | -0.298 | -0.546 |
| (12, 2, 7) | 5.16 | -3937.8 | 2510.0 | -1.6 | 23.9 | 39.0 | 0.807 | 1.349 | -0.543 |
| (-15, 7, 17) | 2.58 | 12098.9 | 4700.0 | 2.6 | 82.6 | 68.8 | 1.921 | 2.460 | -0.539 |
| (10, 2, 1) | 7.58 | -425.3 | 1610.0 | -0.3 | 29.1 | 41.0 | -1.805 | -1.273 | -0.532 |
| (-27, 7, 12) | 2.36 | 9013.9 | 5230.0 | 1.7 | 66.9 | 43.7 | 2.279 | 2.806 | -0.526 |
| (24, 4, 11) | 2.79 | 6477.3 | 3380.0 | 1.9 | 38.4 | 54.6 | 1.626 | 1.103 | +0.523 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `deposited` | 120.89 | 0.5751 | 7.3761627184459995 | 1.452032401210936 | 1.000 |
| `recovered_f` | 121.12 | 0.5774 | 7.097634274753213 | 1.4165830643757904 | 1.000 |