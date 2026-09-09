# Held-Out Log-Likelihood Comparison: `drifted_f` vs `true_model`

## 1. Executive Summary

- **Scored Audit Reflections (|A|)**: 900 (audit set)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `1.8351` nats/refl
- **Model B NLL**: `1.8436` nats/refl
- **Difference (Gain $\Delta$)**: `-0.0085` nats/refl (`+0.0085` nats NLL reduction)
- **Estimated Log Bayes Factor**: `-7.69` nats
- **Uncertainty**: Bootstrap SE = `0.0043` (95% CI: `[-0.0169, -0.0002]`) | Naive SE = `0.0043` (Ratio: `0.98`x)
- **Win Fraction $P(d_h > 0)$**: `47.8%` (430 wins, 470 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `1.94e-01`
- **Wilcoxon Signed-Rank $p$-value**: `6.01e-02`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.0081` | `+0.0005` | `+0.0085` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.0070` | `+0.0016` | `+0.0085` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`true_model`) | Model B (`drifted_f`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0277` | `-0.0269` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `1.9787` | `1.9779` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.1437` | `-0.1343` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -0.3826 | -0.3596 | -0.0230 | 0.0224 | 32.6% | 14/29 |
| 1 | 6.10 | 4.86 | 46 | 1.4398 | 1.4936 | -0.0539 | 0.0385 | 45.7% | 21/25 |
| 2 | 4.83 | 4.21 | 45 | 0.9462 | 0.9648 | -0.0186 | 0.0138 | 35.6% | 16/29 |
| 3 | 4.20 | 3.81 | 46 | 1.8347 | 1.8211 | +0.0137 | 0.0164 | 54.3% | 25/21 |
| 4 | 3.80 | 3.53 | 46 | 1.7472 | 1.7710 | -0.0238 | 0.0181 | 52.2% | 24/22 |
| 5 | 3.53 | 3.32 | 46 | 1.9031 | 1.9081 | -0.0050 | 0.0119 | 52.2% | 24/22 |
| 6 | 3.31 | 3.16 | 44 | 1.4958 | 1.5015 | -0.0057 | 0.0113 | 59.1% | 26/18 |
| 7 | 3.14 | 3.02 | 45 | 2.2567 | 2.2549 | +0.0019 | 0.0159 | 44.4% | 20/25 |
| 8 | 3.01 | 2.89 | 46 | 1.6764 | 1.6925 | -0.0161 | 0.0246 | 45.7% | 21/25 |
| 9 | 2.88 | 2.79 | 43 | 1.8756 | 1.9027 | -0.0271 | 0.0171 | 44.2% | 19/24 |
| 10 | 2.79 | 2.70 | 45 | 1.8283 | 1.7929 | +0.0354 | 0.0211 | 62.2% | 28/17 |
| 11 | 2.70 | 2.62 | 45 | 1.8540 | 1.8441 | +0.0099 | 0.0190 | 48.9% | 22/23 |
| 12 | 2.62 | 2.55 | 44 | 1.9875 | 2.0009 | -0.0134 | 0.0204 | 40.9% | 18/26 |
| 13 | 2.55 | 2.49 | 47 | 1.8978 | 1.9017 | -0.0039 | 0.0148 | 51.1% | 24/23 |
| 14 | 2.49 | 2.43 | 43 | 2.2461 | 2.2314 | +0.0146 | 0.0187 | 48.8% | 21/22 |
| 15 | 2.43 | 2.38 | 43 | 2.3321 | 2.3500 | -0.0179 | 0.0210 | 48.8% | 21/22 |
| 16 | 2.38 | 2.33 | 44 | 2.2999 | 2.3162 | -0.0163 | 0.0174 | 50.0% | 22/22 |
| 17 | 2.32 | 2.28 | 47 | 2.3000 | 2.3012 | -0.0012 | 0.0094 | 46.8% | 22/25 |
| 18 | 2.28 | 2.24 | 48 | 2.4980 | 2.5001 | -0.0021 | 0.0173 | 47.9% | 23/25 |
| 19 | 2.24 | 2.20 | 44 | 2.5761 | 2.5959 | -0.0198 | 0.0160 | 43.2% | 19/25 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (2, 6, 0) | 5.83 | 23774.8 | 2760.0 | 8.6 | 148.2 | 134.2 | -0.452 | 1.002 | -1.454 |
| (19, 9, 0) | 2.95 | -8204.7 | 10170.0 | -0.8 | 121.1 | 139.3 | 3.577 | 4.264 | -0.687 |
| (13, 11, 6) | 2.76 | 8140.9 | 2140.0 | 3.8 | 30.3 | 42.5 | 3.991 | 3.350 | +0.640 |
| (1, 5, 4) | 6.36 | -13937.4 | 7570.0 | -1.8 | 104.9 | 90.4 | 3.251 | 2.678 | +0.573 |
| (15, 7, 18) | 2.47 | 19932.4 | 6800.0 | 2.9 | 90.9 | 106.1 | 3.265 | 2.698 | +0.567 |
| (9, 11, 2) | 3.02 | -6753.3 | 5990.0 | -1.1 | 65.0 | 43.3 | 2.600 | 2.070 | +0.530 |
| (-31, 7, 10) | 2.24 | -9400.7 | 5160.0 | -1.8 | 59.4 | 43.4 | 4.769 | 4.291 | +0.478 |
| (23, 1, 4) | 3.55 | -9524.5 | 7980.0 | -1.2 | 64.5 | 82.9 | 1.932 | 2.410 | -0.478 |
| (18, 2, 18) | 2.66 | 6044.3 | 6940.0 | 0.9 | 117.4 | 99.8 | 2.219 | 1.744 | +0.475 |
| (-10, 4, 5) | 5.47 | 28144.8 | 8570.0 | 3.3 | 105.3 | 117.2 | 2.253 | 1.803 | +0.450 |
| (-15, 7, 17) | 2.58 | 12098.9 | 4700.0 | 2.6 | 82.6 | 71.2 | 1.921 | 2.366 | -0.446 |
| (14, 8, 18) | 2.42 | 17592.2 | 7360.0 | 2.4 | 82.3 | 66.1 | 3.070 | 3.513 | -0.443 |
| (8, 8, 4) | 3.93 | -14527.2 | 15970.0 | -0.9 | 173.2 | 186.4 | 4.473 | 4.915 | -0.442 |
| (30, 4, 4) | 2.65 | -2125.9 | 3130.0 | -0.7 | 45.1 | 55.7 | 1.688 | 2.129 | -0.441 |
| (3, 7, 24) | 2.22 | -3013.7 | 3780.0 | -0.8 | 50.0 | 33.1 | 2.776 | 2.340 | +0.437 |
| (-8, 6, 11) | 3.74 | -9112.5 | 5580.0 | -1.6 | 55.4 | 67.9 | 2.109 | 2.516 | -0.407 |
| (-12, 2, 7) | 5.23 | 2913.8 | 3710.0 | 0.8 | 76.7 | 86.2 | 0.011 | 0.417 | -0.406 |
| (6, 2, 2) | 10.31 | 10499.1 | 1030.0 | 10.2 | 110.9 | 115.7 | -1.640 | -1.234 | -0.406 |
| (24, 4, 11) | 2.79 | 6477.3 | 3380.0 | 1.9 | 38.4 | 50.9 | 1.626 | 1.225 | +0.401 |
| (-3, 3, 13) | 4.23 | 90922.6 | 19780.0 | 4.6 | 203.8 | 191.0 | 3.568 | 3.950 | -0.382 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `true_model` | 120.89 | 0.5751 | 7.3761627184459995 | 1.452032401210936 | 1.000 |
| `drifted_f` | 120.58 | 0.5792 | 7.372580833507016 | 1.4775665297253937 | 1.000 |