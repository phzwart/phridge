# Held-Out Log-Likelihood Comparison: `drifted_i` vs `drifted_f`

## 1. Executive Summary

- **Scored Audit Reflections (|A|)**: 900 (audit set)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `1.1672` nats/refl
- **Model B NLL**: `1.1765` nats/refl
- **Difference (Gain $\Delta$)**: `-0.0093` nats/refl (`+0.0093` nats NLL reduction)
- **Estimated Log Bayes Factor**: `-8.35` nats
- **Uncertainty**: Bootstrap SE = `0.0046` (95% CI: `[-0.0192, -0.0013]`) | Naive SE = `0.0037` (Ratio: `1.24`x)
- **Win Fraction $P(d_h > 0)$**: `46.6%` (419 wins, 481 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `4.20e-02`
- **Wilcoxon Signed-Rank $p$-value**: `1.30e-02`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.0089` | `+0.0003` | `+0.0093` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.0089` | `+0.0004` | `+0.0093` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`drifted_f`) | Model B (`drifted_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0098` | `-0.0085` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `1.3204` | `1.3033` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.1532` | `-0.1268` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -0.8279 | -0.7429 | -0.0850 | 0.0445 | 34.9% | 15/28 |
| 1 | 6.10 | 4.86 | 46 | 0.6968 | 0.7293 | -0.0325 | 0.0160 | 34.8% | 16/30 |
| 2 | 4.83 | 4.21 | 45 | 0.4582 | 0.4576 | +0.0006 | 0.0159 | 46.7% | 21/24 |
| 3 | 4.20 | 3.81 | 46 | 0.9989 | 0.9911 | +0.0078 | 0.0155 | 45.7% | 21/25 |
| 4 | 3.80 | 3.53 | 46 | 1.2784 | 1.3095 | -0.0311 | 0.0184 | 41.3% | 19/27 |
| 5 | 3.53 | 3.32 | 46 | 1.3896 | 1.4046 | -0.0150 | 0.0149 | 45.7% | 21/25 |
| 6 | 3.31 | 3.16 | 44 | 1.0832 | 1.0993 | -0.0160 | 0.0110 | 38.6% | 17/27 |
| 7 | 3.14 | 3.02 | 45 | 1.4203 | 1.4227 | -0.0024 | 0.0158 | 48.9% | 22/23 |
| 8 | 3.01 | 2.89 | 46 | 0.7962 | 0.7985 | -0.0023 | 0.0117 | 45.7% | 21/25 |
| 9 | 2.88 | 2.79 | 43 | 1.0529 | 1.0535 | -0.0006 | 0.0204 | 46.5% | 20/23 |
| 10 | 2.79 | 2.70 | 45 | 0.8186 | 0.8171 | +0.0015 | 0.0121 | 40.0% | 18/27 |
| 11 | 2.70 | 2.62 | 45 | 1.4472 | 1.4524 | -0.0052 | 0.0134 | 53.3% | 24/21 |
| 12 | 2.62 | 2.55 | 44 | 1.3211 | 1.3248 | -0.0037 | 0.0127 | 54.5% | 24/20 |
| 13 | 2.55 | 2.49 | 47 | 1.3951 | 1.4092 | -0.0141 | 0.0147 | 53.2% | 25/22 |
| 14 | 2.49 | 2.43 | 43 | 1.1691 | 1.1615 | +0.0076 | 0.0100 | 58.1% | 25/18 |
| 15 | 2.43 | 2.38 | 43 | 1.7745 | 1.7622 | +0.0123 | 0.0138 | 53.5% | 23/20 |
| 16 | 2.38 | 2.33 | 44 | 1.4172 | 1.4167 | +0.0005 | 0.0107 | 54.5% | 24/20 |
| 17 | 2.32 | 2.28 | 47 | 1.8251 | 1.8256 | -0.0005 | 0.0117 | 51.1% | 24/23 |
| 18 | 2.28 | 2.24 | 48 | 1.8195 | 1.8168 | +0.0027 | 0.0099 | 41.7% | 20/28 |
| 19 | 2.24 | 2.20 | 44 | 1.8991 | 1.9108 | -0.0116 | 0.0089 | 43.2% | 19/25 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (-6, 2, 2) | 10.39 | 21050.1 | 765.0 | 27.5 | 158.6 | 165.0 | -0.477 | 1.044 | -1.521 |
| (-1, 5, 4) | 6.37 | 22235.6 | 3590.0 | 6.2 | 180.6 | 185.9 | 1.156 | 1.852 | -0.696 |
| (-14, 2, 12) | 3.78 | 28046.8 | 19035.0 | 1.5 | 281.1 | 292.8 | 4.057 | 4.644 | -0.588 |
| (-13, 5, 20) | 2.54 | 7058.8 | 1940.0 | 3.6 | 41.4 | 34.0 | 3.359 | 3.876 | -0.517 |
| (13, 11, 2) | 2.87 | 2913.6 | 980.0 | 3.0 | 71.5 | 65.4 | 0.674 | 0.179 | +0.496 |
| (24, 0, 10) | 3.02 | 49740.7 | 6740.0 | 7.4 | 179.1 | 185.4 | 3.255 | 2.766 | +0.489 |
| (-2, 2, 5) | 9.64 | 5606.6 | 1410.0 | 4.0 | 50.4 | 54.7 | -0.586 | -1.022 | +0.436 |
| (12, 2, 19) | 2.81 | 15260.9 | 4295.0 | 3.6 | 156.9 | 151.7 | 2.390 | 2.029 | +0.361 |
| (-10, 0, 3) | 7.85 | 2277.2 | 1455.0 | 1.6 | 61.7 | 68.4 | -1.541 | -1.188 | -0.353 |
| (-3, 5, 11) | 4.26 | 10860.2 | 1340.0 | 8.1 | 61.9 | 63.6 | 3.425 | 3.080 | +0.344 |
| (26, 4, 2) | 3.04 | 6607.7 | 2215.0 | 3.0 | 93.0 | 99.3 | 0.488 | 0.825 | -0.337 |
| (-14, 8, 7) | 3.30 | 8930.1 | 2630.0 | 3.4 | 58.5 | 53.5 | 1.326 | 1.661 | -0.335 |
| (4, 2, 0) | 13.57 | 12792.6 | 2220.0 | 5.8 | 120.0 | 124.7 | -1.522 | -1.188 | -0.334 |
| (12, 8, 19) | 2.39 | 620.0 | 2015.0 | 0.3 | 59.1 | 52.4 | 1.738 | 1.406 | +0.332 |
| (26, 4, 16) | 2.35 | 283.7 | 2170.0 | 0.1 | 59.7 | 64.7 | 1.883 | 2.215 | -0.332 |
| (-10, 2, 20) | 2.79 | 7471.7 | 3165.0 | 2.4 | 62.3 | 73.0 | 1.042 | 0.712 | +0.329 |
| (-8, 6, 11) | 3.74 | -1596.9 | 2790.0 | -0.6 | 62.6 | 68.0 | 1.068 | 1.398 | -0.329 |
| (-11, 1, 12) | 4.17 | 95969.3 | 26980.0 | 3.6 | 202.1 | 216.2 | 3.722 | 3.396 | +0.326 |
| (1, 5, 4) | 6.36 | 16328.0 | 3785.0 | 4.3 | 91.4 | 95.9 | 0.670 | 0.348 | +0.322 |
| (20, 4, 7) | 3.47 | 4104.7 | 1775.0 | 2.3 | 31.9 | 25.0 | 0.636 | 0.950 | -0.314 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `drifted_f` | 73.82 | 0.7100 | 9.144780111604435 | 2.6785037744361655 | 1.000 |
| `drifted_i` | 74.50 | 0.7062 | 8.882128474088304 | 2.549883001187861 | 1.000 |