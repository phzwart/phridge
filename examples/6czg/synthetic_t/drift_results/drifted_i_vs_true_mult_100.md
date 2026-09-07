# Held-Out Log-Likelihood Comparison: `drifted_i` vs `true_model`

## 1. Executive Summary

- **Scored Test Reflections (|T|)**: 900 (held-out)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `1.8351` nats/refl
- **Model B NLL**: `1.8476` nats/refl
- **Difference (Gain $\Delta$)**: `-0.0125` nats/refl (`+0.0125` nats NLL reduction)
- **Estimated Log Bayes Factor**: `-11.27` nats
- **Uncertainty**: Bootstrap SE = `0.0040` (95% CI: `[-0.0199, -0.0048]`) | Naive SE = `0.0045` (Ratio: `0.87`x)
- **Win Fraction $P(d_h > 0)$**: `45.4%` (409 wins, 491 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `6.90e-03`
- **Wilcoxon Signed-Rank $p$-value**: `8.64e-03`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.0120` | `+0.0005` | `+0.0125` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.0099` | `+0.0027` | `+0.0125` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`true_model`) | Model B (`drifted_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0277` | `-0.0262` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `1.9787` | `1.9670` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.1437` | `-0.1194` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -0.3826 | -0.3502 | -0.0324 | 0.0281 | 41.9% | 18/25 |
| 1 | 6.10 | 4.86 | 46 | 1.4398 | 1.4912 | -0.0514 | 0.0497 | 52.2% | 24/22 |
| 2 | 4.83 | 4.21 | 45 | 0.9462 | 0.9716 | -0.0254 | 0.0161 | 42.2% | 19/26 |
| 3 | 4.20 | 3.81 | 46 | 1.8347 | 1.8203 | +0.0144 | 0.0140 | 65.2% | 30/16 |
| 4 | 3.80 | 3.53 | 46 | 1.7472 | 1.7834 | -0.0362 | 0.0148 | 37.0% | 17/29 |
| 5 | 3.53 | 3.32 | 46 | 1.9031 | 1.9221 | -0.0189 | 0.0116 | 43.5% | 20/26 |
| 6 | 3.31 | 3.16 | 44 | 1.4958 | 1.5016 | -0.0058 | 0.0127 | 52.3% | 23/21 |
| 7 | 3.14 | 3.02 | 45 | 2.2567 | 2.2574 | -0.0007 | 0.0159 | 37.8% | 17/28 |
| 8 | 3.01 | 2.89 | 46 | 1.6764 | 1.6851 | -0.0087 | 0.0216 | 45.7% | 21/25 |
| 9 | 2.88 | 2.79 | 43 | 1.8756 | 1.9133 | -0.0377 | 0.0201 | 41.9% | 18/25 |
| 10 | 2.79 | 2.70 | 45 | 1.8283 | 1.8085 | +0.0198 | 0.0195 | 53.3% | 24/21 |
| 11 | 2.70 | 2.62 | 45 | 1.8540 | 1.8473 | +0.0067 | 0.0179 | 48.9% | 22/23 |
| 12 | 2.62 | 2.55 | 44 | 1.9875 | 1.9980 | -0.0105 | 0.0170 | 38.6% | 17/27 |
| 13 | 2.55 | 2.49 | 47 | 1.8978 | 1.9150 | -0.0172 | 0.0138 | 42.6% | 20/27 |
| 14 | 2.49 | 2.43 | 43 | 2.2461 | 2.2419 | +0.0041 | 0.0196 | 44.2% | 19/24 |
| 15 | 2.43 | 2.38 | 43 | 2.3321 | 2.3341 | -0.0020 | 0.0217 | 51.2% | 22/21 |
| 16 | 2.38 | 2.33 | 44 | 2.2999 | 2.3168 | -0.0169 | 0.0162 | 45.5% | 20/24 |
| 17 | 2.32 | 2.28 | 47 | 2.3000 | 2.3042 | -0.0042 | 0.0111 | 42.6% | 20/27 |
| 18 | 2.28 | 2.24 | 48 | 2.4980 | 2.5073 | -0.0093 | 0.0173 | 45.8% | 22/26 |
| 19 | 2.24 | 2.20 | 44 | 2.5761 | 2.5947 | -0.0186 | 0.0126 | 36.4% | 16/28 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (2, 6, 0) | 5.83 | 23774.8 | 2760.0 | 8.6 | 148.2 | 129.5 | -0.452 | 1.550 | -2.002 |
| (8, 0, 9) | 5.58 | 6316.0 | 3240.0 | 1.9 | 91.0 | 104.3 | -0.309 | 0.355 | -0.664 |
| (15, 7, 18) | 2.47 | 19932.4 | 6800.0 | 2.9 | 90.9 | 108.3 | 3.265 | 2.637 | +0.628 |
| (13, 11, 6) | 2.76 | 8140.9 | 2140.0 | 3.8 | 30.3 | 40.0 | 3.991 | 3.401 | +0.590 |
| (1, 5, 4) | 6.36 | -13937.4 | 7570.0 | -1.8 | 104.9 | 90.8 | 3.251 | 2.680 | +0.572 |
| (6, 2, 2) | 10.31 | 10499.1 | 1030.0 | 10.2 | 110.9 | 117.7 | -1.640 | -1.075 | -0.565 |
| (9, 11, 2) | 3.02 | -6753.3 | 5990.0 | -1.1 | 65.0 | 43.0 | 2.600 | 2.065 | +0.535 |
| (30, 4, 4) | 2.65 | -2125.9 | 3130.0 | -0.7 | 45.1 | 56.9 | 1.688 | 2.181 | -0.493 |
| (-8, 6, 11) | 3.74 | -9112.5 | 5580.0 | -1.6 | 55.4 | 70.1 | 2.109 | 2.580 | -0.471 |
| (19, 9, 0) | 2.95 | -8204.7 | 10170.0 | -0.8 | 121.1 | 134.9 | 3.577 | 4.033 | -0.456 |
| (-10, 4, 5) | 5.47 | 28144.8 | 8570.0 | 3.3 | 105.3 | 116.9 | 2.253 | 1.809 | +0.444 |
| (13, 11, 2) | 2.87 | 3646.0 | 1960.0 | 1.9 | 53.4 | 75.0 | 0.157 | 0.598 | -0.441 |
| (-6, 2, 2) | 10.39 | 21953.4 | 1530.0 | 14.3 | 154.5 | 157.6 | -1.633 | -1.214 | -0.419 |
| (-22, 8, 12) | 2.52 | -1199.3 | 3340.0 | -0.4 | 40.6 | 54.8 | 1.425 | 1.844 | -0.419 |
| (-3, 3, 13) | 4.23 | 90922.6 | 19780.0 | 4.6 | 203.8 | 189.3 | 3.568 | 3.986 | -0.417 |
| (-31, 7, 10) | 2.24 | -9400.7 | 5160.0 | -1.8 | 59.4 | 46.2 | 4.769 | 4.357 | +0.413 |
| (14, 8, 18) | 2.42 | 17592.2 | 7360.0 | 2.4 | 82.3 | 67.4 | 3.070 | 3.478 | -0.408 |
| (-22, 2, 17) | 2.59 | -5348.6 | 4480.0 | -1.2 | 48.9 | 61.2 | 2.616 | 3.019 | -0.402 |
| (8, 8, 4) | 3.93 | -14527.2 | 15970.0 | -0.9 | 173.2 | 185.8 | 4.473 | 4.873 | -0.400 |
| (11, 7, 14) | 2.98 | 16134.5 | 5550.0 | 2.9 | 120.6 | 105.6 | 0.800 | 1.198 | -0.399 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `true_model` | 120.89 | 0.5751 | 7.3761627184459995 | 1.452032401210936 | 1.000 |
| `drifted_i` | 120.89 | 0.5806 | 7.238209339780019 | 1.4404308387578 | 1.000 |