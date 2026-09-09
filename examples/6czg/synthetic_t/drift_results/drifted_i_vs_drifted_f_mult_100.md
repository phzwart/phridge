# Held-Out Log-Likelihood Comparison: `drifted_i` vs `drifted_f`

## 1. Executive Summary

- **Scored Audit Reflections (|A|)**: 900 (audit set)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `1.8436` nats/refl
- **Model B NLL**: `1.8476` nats/refl
- **Difference (Gain $\Delta$)**: `-0.0040` nats/refl (`+0.0040` nats NLL reduction)
- **Estimated Log Bayes Factor**: `-3.57` nats
- **Uncertainty**: Bootstrap SE = `0.0018` (95% CI: `[-0.0073, -0.0004]`) | Naive SE = `0.0022` (Ratio: `0.80`x)
- **Win Fraction $P(d_h > 0)$**: `43.9%` (395 wins, 505 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `2.75e-04`
- **Wilcoxon Signed-Rank $p$-value**: `1.38e-02`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.0033` | `+0.0006` | `+0.0040` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.0033` | `+0.0007` | `+0.0040` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`drifted_f`) | Model B (`drifted_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0269` | `-0.0262` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `1.9779` | `1.9670` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.1343` | `-0.1194` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -0.3596 | -0.3502 | -0.0094 | 0.0128 | 30.2% | 13/30 |
| 1 | 6.10 | 4.86 | 46 | 1.4936 | 1.4912 | +0.0025 | 0.0189 | 58.7% | 27/19 |
| 2 | 4.83 | 4.21 | 45 | 0.9648 | 0.9716 | -0.0068 | 0.0081 | 46.7% | 21/24 |
| 3 | 4.20 | 3.81 | 46 | 1.8211 | 1.8203 | +0.0008 | 0.0095 | 50.0% | 23/23 |
| 4 | 3.80 | 3.53 | 46 | 1.7710 | 1.7834 | -0.0124 | 0.0099 | 41.3% | 19/27 |
| 5 | 3.53 | 3.32 | 46 | 1.9081 | 1.9221 | -0.0140 | 0.0091 | 32.6% | 15/31 |
| 6 | 3.31 | 3.16 | 44 | 1.5015 | 1.5016 | -0.0001 | 0.0066 | 43.2% | 19/25 |
| 7 | 3.14 | 3.02 | 45 | 2.2549 | 2.2574 | -0.0026 | 0.0068 | 51.1% | 23/22 |
| 8 | 3.01 | 2.89 | 46 | 1.6925 | 1.6851 | +0.0074 | 0.0102 | 50.0% | 23/23 |
| 9 | 2.88 | 2.79 | 43 | 1.9027 | 1.9133 | -0.0105 | 0.0101 | 51.2% | 22/21 |
| 10 | 2.79 | 2.70 | 45 | 1.7929 | 1.8085 | -0.0156 | 0.0107 | 37.8% | 17/28 |
| 11 | 2.70 | 2.62 | 45 | 1.8441 | 1.8473 | -0.0032 | 0.0109 | 44.4% | 20/25 |
| 12 | 2.62 | 2.55 | 44 | 2.0009 | 1.9980 | +0.0029 | 0.0114 | 43.2% | 19/25 |
| 13 | 2.55 | 2.49 | 47 | 1.9017 | 1.9150 | -0.0133 | 0.0099 | 34.0% | 16/31 |
| 14 | 2.49 | 2.43 | 43 | 2.2314 | 2.2419 | -0.0105 | 0.0088 | 41.9% | 18/25 |
| 15 | 2.43 | 2.38 | 43 | 2.3500 | 2.3341 | +0.0160 | 0.0075 | 55.8% | 24/19 |
| 16 | 2.38 | 2.33 | 44 | 2.3162 | 2.3168 | -0.0006 | 0.0112 | 40.9% | 18/26 |
| 17 | 2.32 | 2.28 | 47 | 2.3012 | 2.3042 | -0.0030 | 0.0069 | 46.8% | 22/25 |
| 18 | 2.28 | 2.24 | 48 | 2.5001 | 2.5073 | -0.0072 | 0.0056 | 33.3% | 16/32 |
| 19 | 2.24 | 2.20 | 44 | 2.5959 | 2.5947 | +0.0012 | 0.0075 | 45.5% | 20/24 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (2, 6, 0) | 5.83 | 23774.8 | 2760.0 | 8.6 | 134.2 | 129.5 | 1.002 | 1.550 | -0.548 |
| (8, 0, 9) | 5.58 | 6316.0 | 3240.0 | 1.9 | 97.1 | 104.3 | -0.040 | 0.355 | -0.395 |
| (11, 7, 21) | 2.35 | -4692.1 | 3340.0 | -1.4 | 46.1 | 36.6 | 3.337 | 2.999 | +0.337 |
| (-9, 11, 8) | 2.82 | -15445.9 | 17120.0 | -0.9 | 133.2 | 147.1 | 3.999 | 4.328 | -0.328 |
| (-15, 7, 17) | 2.58 | 12098.9 | 4700.0 | 2.6 | 71.2 | 79.2 | 2.366 | 2.051 | +0.316 |
| (5, 5, 4) | 5.96 | 23450.1 | 4860.0 | 4.8 | 127.9 | 132.5 | 0.848 | 0.560 | +0.288 |
| (21, 11, 1) | 2.51 | -8080.3 | 2980.0 | -2.7 | 44.7 | 50.7 | 4.849 | 5.112 | -0.263 |
| (-27, 9, 1) | 2.45 | -2652.1 | 4260.0 | -0.6 | 47.9 | 56.9 | 2.143 | 2.388 | -0.245 |
| (-1, 5, 4) | 6.37 | 43715.8 | 7180.0 | 6.1 | 180.8 | 184.6 | 0.520 | 0.285 | +0.235 |
| (19, 9, 0) | 2.95 | -8204.7 | 10170.0 | -0.8 | 139.3 | 134.9 | 4.264 | 4.033 | +0.231 |
| (-31, 3, 2) | 2.66 | -2690.0 | 3230.0 | -0.8 | 43.6 | 36.0 | 1.813 | 1.591 | +0.222 |
| (-30, 4, 2) | 2.69 | 5864.7 | 10110.0 | 0.6 | 116.9 | 126.8 | 2.229 | 2.444 | -0.215 |
| (-2, 2, 14) | 4.12 | 61643.7 | 11950.0 | 5.2 | 85.1 | 93.4 | 6.697 | 6.493 | +0.204 |
| (-10, 0, 21) | 2.70 | -2817.2 | 6280.0 | -0.4 | 92.2 | 88.3 | 2.816 | 2.615 | +0.202 |
| (11, 5, 4) | 4.90 | 118132.2 | 40210.0 | 2.9 | 216.0 | 229.0 | 3.600 | 3.400 | +0.199 |
| (4, 2, 0) | 13.57 | 11556.0 | 4440.0 | 2.6 | 119.4 | 125.1 | -1.068 | -0.871 | -0.197 |
| (-10, 10, 12) | 2.73 | -3406.9 | 3470.0 | -1.0 | 34.7 | 41.8 | 1.567 | 1.763 | -0.196 |
| (9, 13, 10) | 2.39 | 6579.1 | 3770.0 | 1.7 | 40.8 | 48.1 | 2.240 | 2.044 | +0.195 |
| (31, 3, 2) | 2.65 | -6152.8 | 4550.0 | -1.4 | 42.6 | 49.4 | 2.608 | 2.801 | -0.193 |
| (24, 4, 11) | 2.79 | 6477.3 | 3380.0 | 1.9 | 50.9 | 44.1 | 1.225 | 1.416 | -0.191 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `drifted_f` | 120.58 | 0.5792 | 7.372580833507016 | 1.4775665297253937 | 1.000 |
| `drifted_i` | 120.89 | 0.5806 | 7.238209339780019 | 1.4404308387578 | 1.000 |