# Held-Out Log-Likelihood Comparison: `recovered_i` vs `recovered_f`

## 1. Executive Summary

- **Scored Test Reflections (|T|)**: 900 (held-out)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `1.8455` nats/refl
- **Model B NLL**: `1.8508` nats/refl
- **Difference (Gain $\Delta$)**: `-0.0053` nats/refl (`+0.0053` nats NLL reduction)
- **Estimated Log Bayes Factor**: `-4.76` nats
- **Uncertainty**: Bootstrap SE = `0.0018` (95% CI: `[-0.0088, -0.0019]`) | Naive SE = `0.0024` (Ratio: `0.76`x)
- **Win Fraction $P(d_h > 0)$**: `42.9%` (386 wins, 514 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `2.24e-05`
- **Wilcoxon Signed-Rank $p$-value**: `6.16e-04`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.0047` | `+0.0006` | `+0.0053` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.0047` | `+0.0006` | `+0.0053` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`recovered_f`) | Model B (`recovered_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0274` | `-0.0264` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `1.9824` | `1.9726` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.1369` | `-0.1218` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -0.3752 | -0.3715 | -0.0037 | 0.0102 | 44.2% | 19/24 |
| 1 | 6.10 | 4.86 | 46 | 1.5142 | 1.5215 | -0.0073 | 0.0202 | 43.5% | 20/26 |
| 2 | 4.83 | 4.21 | 45 | 0.9570 | 0.9662 | -0.0093 | 0.0089 | 42.2% | 19/26 |
| 3 | 4.20 | 3.81 | 46 | 1.8121 | 1.8149 | -0.0028 | 0.0099 | 39.1% | 18/28 |
| 4 | 3.80 | 3.53 | 46 | 1.7924 | 1.8018 | -0.0094 | 0.0097 | 32.6% | 15/31 |
| 5 | 3.53 | 3.32 | 46 | 1.9192 | 1.9279 | -0.0087 | 0.0080 | 37.0% | 17/29 |
| 6 | 3.31 | 3.16 | 44 | 1.5062 | 1.5053 | +0.0009 | 0.0089 | 38.6% | 17/27 |
| 7 | 3.14 | 3.02 | 45 | 2.2610 | 2.2761 | -0.0151 | 0.0109 | 53.3% | 24/21 |
| 8 | 3.01 | 2.89 | 46 | 1.6949 | 1.6898 | +0.0050 | 0.0128 | 41.3% | 19/27 |
| 9 | 2.88 | 2.79 | 43 | 1.8932 | 1.9064 | -0.0131 | 0.0100 | 55.8% | 24/19 |
| 10 | 2.79 | 2.70 | 45 | 1.8092 | 1.8156 | -0.0064 | 0.0123 | 40.0% | 18/27 |
| 11 | 2.70 | 2.62 | 45 | 1.8606 | 1.8638 | -0.0032 | 0.0097 | 51.1% | 23/22 |
| 12 | 2.62 | 2.55 | 44 | 1.9980 | 1.9953 | +0.0026 | 0.0115 | 43.2% | 19/25 |
| 13 | 2.55 | 2.49 | 47 | 1.8930 | 1.9131 | -0.0201 | 0.0115 | 27.7% | 13/34 |
| 14 | 2.49 | 2.43 | 43 | 2.2267 | 2.2451 | -0.0184 | 0.0086 | 37.2% | 16/27 |
| 15 | 2.43 | 2.38 | 43 | 2.3517 | 2.3375 | +0.0142 | 0.0071 | 62.8% | 27/16 |
| 16 | 2.38 | 2.33 | 44 | 2.3124 | 2.3162 | -0.0038 | 0.0113 | 38.6% | 17/27 |
| 17 | 2.32 | 2.28 | 47 | 2.3083 | 2.3098 | -0.0016 | 0.0075 | 46.8% | 22/25 |
| 18 | 2.28 | 2.24 | 48 | 2.5016 | 2.5067 | -0.0050 | 0.0057 | 41.7% | 20/28 |
| 19 | 2.24 | 2.20 | 44 | 2.5833 | 2.5829 | +0.0004 | 0.0074 | 43.2% | 19/25 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (2, 6, 0) | 5.83 | 23774.8 | 2760.0 | 8.6 | 139.4 | 134.4 | 0.349 | 0.956 | -0.607 |
| (26, 4, 2) | 3.04 | 6063.2 | 4430.0 | 1.4 | 92.9 | 104.1 | 0.902 | 1.315 | -0.413 |
| (-11, 3, 7) | 5.17 | -4361.4 | 3580.0 | -1.2 | 64.3 | 71.7 | 1.977 | 2.389 | -0.411 |
| (21, 11, 1) | 2.51 | -8080.3 | 2980.0 | -2.7 | 40.1 | 48.4 | 4.622 | 4.976 | -0.353 |
| (-15, 7, 17) | 2.58 | 12098.9 | 4700.0 | 2.6 | 68.8 | 77.6 | 2.460 | 2.112 | +0.348 |
| (12, 2, 7) | 5.16 | -3937.8 | 2510.0 | -1.6 | 39.0 | 30.3 | 1.349 | 1.009 | +0.340 |
| (-9, 11, 8) | 2.82 | -15445.9 | 17120.0 | -0.9 | 122.9 | 137.5 | 3.777 | 4.092 | -0.315 |
| (11, 7, 21) | 2.35 | -4692.1 | 3340.0 | -1.4 | 41.8 | 32.2 | 3.173 | 2.882 | +0.291 |
| (19, 9, 0) | 2.95 | -8204.7 | 10170.0 | -0.8 | 143.9 | 139.8 | 4.454 | 4.168 | +0.285 |
| (15, 9, 7) | 3.01 | 12265.0 | 5340.0 | 2.3 | 68.8 | 75.4 | 1.674 | 1.435 | +0.239 |
| (24, 4, 11) | 2.79 | 6477.3 | 3380.0 | 1.9 | 54.6 | 46.1 | 1.103 | 1.332 | -0.229 |
| (-10, 0, 21) | 2.70 | -2817.2 | 6280.0 | -0.4 | 88.1 | 84.0 | 2.658 | 2.431 | +0.227 |
| (3, 3, 11) | 4.83 | 8593.0 | 2820.0 | 3.0 | 51.1 | 46.8 | 1.395 | 1.620 | -0.225 |
| (11, 5, 4) | 4.90 | 118132.2 | 40210.0 | 2.9 | 206.5 | 221.5 | 3.739 | 3.515 | +0.224 |
| (-14, 4, 16) | 3.00 | 2692.4 | 4560.0 | 0.6 | 93.9 | 90.3 | 1.432 | 1.220 | +0.212 |
| (-6, 2, 2) | 10.39 | 21953.4 | 1530.0 | 14.3 | 155.7 | 158.1 | -1.420 | -1.210 | -0.210 |
| (-13, 7, 14) | 2.92 | 15310.3 | 4250.0 | 3.6 | 95.7 | 99.0 | 1.598 | 1.393 | +0.206 |
| (23, 1, 4) | 3.55 | -9524.5 | 7980.0 | -1.2 | 82.5 | 75.7 | 2.399 | 2.193 | +0.206 |
| (1, 7, 10) | 3.84 | 4217.7 | 3680.0 | 1.1 | 84.7 | 77.8 | 0.390 | 0.187 | +0.203 |
| (-27, 9, 1) | 2.45 | -2652.1 | 4260.0 | -0.6 | 45.0 | 53.3 | 2.073 | 2.274 | -0.201 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `recovered_f` | 121.12 | 0.5774 | 7.097634274753213 | 1.4165830643757904 | 1.000 |
| `recovered_i` | 121.56 | 0.5772 | 7.030455137496274 | 1.3920065231717647 | 1.000 |