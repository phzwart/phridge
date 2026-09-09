# Held-Out Log-Likelihood Comparison: `drifted_i` vs `drifted_f`

## 1. Executive Summary

- **Scored Audit Reflections (|A|)**: 900 (audit set)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `-0.0244` nats/refl
- **Model B NLL**: `-0.0241` nats/refl
- **Difference (Gain $\Delta$)**: `-0.0003` nats/refl (`+0.0003` nats NLL reduction)
- **Estimated Log Bayes Factor**: `-0.25` nats
- **Uncertainty**: Bootstrap SE = `0.0021` (95% CI: `[-0.0047, +0.0037]`) | Naive SE = `0.0016` (Ratio: `1.37`x)
- **Win Fraction $P(d_h > 0)$**: `47.8%` (430 wins, 470 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `1.94e-01`
- **Wilcoxon Signed-Rank $p$-value**: `6.17e-01`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.0001` | `+0.0002` | `+0.0003` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.0001` | `+0.0002` | `+0.0003` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`drifted_f`) | Model B (`drifted_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `+0.0064` | `+0.0067` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `-0.0138` | `-0.0135` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.0106` | `-0.0106` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -1.1236 | -1.1259 | +0.0023 | 0.0040 | 46.5% | 20/23 |
| 1 | 6.10 | 4.86 | 46 | 0.6592 | 0.6839 | -0.0247 | 0.0112 | 34.8% | 16/30 |
| 2 | 4.83 | 4.21 | 45 | -0.2257 | -0.2092 | -0.0165 | 0.0051 | 31.1% | 14/31 |
| 3 | 4.20 | 3.81 | 46 | 0.1242 | 0.1290 | -0.0048 | 0.0088 | 50.0% | 23/23 |
| 4 | 3.80 | 3.53 | 46 | 0.3018 | 0.3052 | -0.0033 | 0.0076 | 39.1% | 18/28 |
| 5 | 3.53 | 3.32 | 46 | 0.5394 | 0.5440 | -0.0046 | 0.0110 | 56.5% | 26/20 |
| 6 | 3.31 | 3.16 | 44 | -0.1638 | -0.1753 | +0.0115 | 0.0066 | 50.0% | 22/22 |
| 7 | 3.14 | 3.02 | 45 | 0.4003 | 0.3866 | +0.0137 | 0.0091 | 62.2% | 28/17 |
| 8 | 3.01 | 2.89 | 46 | -0.1908 | -0.1849 | -0.0058 | 0.0050 | 47.8% | 22/24 |
| 9 | 2.88 | 2.79 | 43 | -0.1691 | -0.1694 | +0.0003 | 0.0053 | 44.2% | 19/24 |
| 10 | 2.79 | 2.70 | 45 | -0.4673 | -0.4700 | +0.0027 | 0.0041 | 51.1% | 23/22 |
| 11 | 2.70 | 2.62 | 45 | -0.0467 | -0.0532 | +0.0065 | 0.0065 | 57.8% | 26/19 |
| 12 | 2.62 | 2.55 | 44 | -0.1701 | -0.1814 | +0.0113 | 0.0051 | 56.8% | 25/19 |
| 13 | 2.55 | 2.49 | 47 | -0.1021 | -0.0959 | -0.0062 | 0.0044 | 42.6% | 20/27 |
| 14 | 2.49 | 2.43 | 43 | -0.4996 | -0.5051 | +0.0055 | 0.0045 | 46.5% | 20/23 |
| 15 | 2.43 | 2.38 | 43 | 0.0774 | 0.0784 | -0.0009 | 0.0048 | 41.9% | 18/25 |
| 16 | 2.38 | 2.33 | 44 | -0.0422 | -0.0392 | -0.0030 | 0.0049 | 47.7% | 21/23 |
| 17 | 2.32 | 2.28 | 47 | 0.0045 | 0.0017 | +0.0029 | 0.0055 | 51.1% | 24/23 |
| 18 | 2.28 | 2.24 | 48 | 0.3483 | 0.3460 | +0.0022 | 0.0071 | 43.8% | 21/27 |
| 19 | 2.24 | 2.20 | 44 | 0.1246 | 0.1172 | +0.0074 | 0.0059 | 54.5% | 24/20 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (-13, 1, 8) | 4.90 | 298524.3 | 9400.0 | 31.8 | 648.6 | 650.6 | 7.126 | 7.387 | -0.261 |
| (14, 2, 4) | 5.33 | 184945.9 | 7620.0 | 24.3 | 530.5 | 532.4 | 6.644 | 6.873 | -0.229 |
| (2, 10, 4) | 3.42 | 31824.9 | 1702.5 | 18.7 | 212.8 | 215.1 | 2.028 | 2.256 | -0.228 |
| (14, 4, 0) | 5.00 | 54687.4 | 2670.0 | 20.5 | 294.7 | 292.4 | 3.064 | 2.861 | +0.203 |
| (-18, 4, 10) | 3.43 | 48176.8 | 3216.5 | 15.0 | 278.1 | 279.5 | 4.786 | 4.983 | -0.197 |
| (-12, 8, 6) | 3.51 | 50830.6 | 2124.5 | 23.9 | 268.9 | 270.5 | 2.897 | 3.094 | -0.197 |
| (32, 4, 12) | 2.25 | 3101.1 | 551.5 | 5.6 | 67.7 | 66.4 | 1.226 | 1.046 | +0.180 |
| (-17, 13, 6) | 2.32 | 4717.0 | 482.0 | 9.8 | 85.9 | 85.1 | 2.033 | 1.855 | +0.178 |
| (-9, 13, 13) | 2.27 | 2069.4 | 178.5 | 11.6 | 55.1 | 53.8 | 0.660 | 0.482 | +0.178 |
| (-8, 8, 4) | 3.94 | 47303.8 | 1647.5 | 28.7 | 256.4 | 258.5 | 1.975 | 2.152 | -0.177 |
| (-16, 2, 11) | 3.73 | 10121.4 | 1407.5 | 7.2 | 144.0 | 141.9 | 1.434 | 1.260 | +0.174 |
| (2, 2, 14) | 4.11 | 121519.8 | 5705.5 | 21.3 | 424.2 | 425.6 | 5.374 | 5.546 | -0.173 |
| (23, 11, 10) | 2.24 | 782.8 | 154.0 | 5.1 | 33.7 | 31.7 | -0.083 | -0.251 | +0.168 |
| (21, 3, 16) | 2.65 | 5957.6 | 608.5 | 9.8 | 102.2 | 103.1 | 2.206 | 2.363 | -0.157 |
| (17, 5, 0) | 4.08 | 59257.2 | 3818.5 | 15.5 | 286.2 | 288.2 | 2.109 | 2.266 | -0.157 |
| (9, 5, 4) | 5.27 | 118950.6 | 7195.0 | 16.5 | 435.2 | 436.7 | 5.409 | 5.565 | -0.156 |
| (27, 1, 2) | 3.11 | 66998.6 | 4270.0 | 15.7 | 303.1 | 304.4 | 3.474 | 3.630 | -0.155 |
| (-2, 6, 15) | 3.28 | 27685.5 | 1596.5 | 17.3 | 204.7 | 203.3 | 2.045 | 1.904 | +0.141 |
| (17, 3, 10) | 3.61 | 9525.8 | 1543.0 | 6.2 | 128.7 | 126.4 | 0.663 | 0.524 | +0.138 |
| (-26, 4, 2) | 3.05 | 16668.3 | 1434.5 | 11.6 | 152.9 | 151.3 | 1.374 | 1.239 | +0.136 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `drifted_f` | 9.77 | 0.9943 | 7.780059205817361 | 2.6242456572623 | 1.000 |
| `drifted_i` | 9.77 | 0.9945 | 7.771953734155296 | 2.617201349742824 | 1.000 |