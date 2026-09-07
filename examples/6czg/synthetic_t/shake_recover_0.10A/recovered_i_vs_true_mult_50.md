# Held-Out Log-Likelihood Comparison: `recovered_i` vs `deposited`

## 1. Executive Summary

- **Scored Test Reflections (|T|)**: 900 (held-out)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `1.1555` nats/refl
- **Model B NLL**: `1.1886` nats/refl
- **Difference (Gain $\Delta$)**: `-0.0331` nats/refl (`+0.0331` nats NLL reduction)
- **Estimated Log Bayes Factor**: `-29.82` nats
- **Uncertainty**: Bootstrap SE = `0.0087` (95% CI: `[-0.0502, -0.0160]`) | Naive SE = `0.0087` (Ratio: `1.00`x)
- **Win Fraction $P(d_h > 0)$**: `39.3%` (354 wins, 546 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `1.65e-10`
- **Wilcoxon Signed-Rank $p$-value**: `1.49e-08`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.0433` | `-0.0101` | `+0.0331` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.0322` | `+0.0010` | `+0.0331` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`deposited`) | Model B (`recovered_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0069` | `-0.0076` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `1.3184` | `1.3215` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.1629` | `-0.1329` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -0.8849 | -0.7930 | -0.0920 | 0.0813 | 37.2% | 16/27 |
| 1 | 6.10 | 4.86 | 46 | 0.6455 | 0.7015 | -0.0560 | 0.0328 | 37.0% | 17/29 |
| 2 | 4.83 | 4.21 | 45 | 0.4586 | 0.5487 | -0.0900 | 0.0555 | 24.4% | 11/34 |
| 3 | 4.20 | 3.81 | 46 | 0.9819 | 0.9530 | +0.0289 | 0.0337 | 41.3% | 19/27 |
| 4 | 3.80 | 3.53 | 46 | 1.2523 | 1.3230 | -0.0707 | 0.0412 | 32.6% | 15/31 |
| 5 | 3.53 | 3.32 | 46 | 1.3688 | 1.3853 | -0.0166 | 0.0347 | 39.1% | 18/28 |
| 6 | 3.31 | 3.16 | 44 | 1.1069 | 1.1958 | -0.0889 | 0.0346 | 40.9% | 18/26 |
| 7 | 3.14 | 3.02 | 45 | 1.3792 | 1.4367 | -0.0576 | 0.0508 | 46.7% | 21/24 |
| 8 | 3.01 | 2.89 | 46 | 0.7427 | 0.8228 | -0.0802 | 0.0469 | 34.8% | 16/30 |
| 9 | 2.88 | 2.79 | 43 | 0.9986 | 1.0525 | -0.0539 | 0.0316 | 37.2% | 16/27 |
| 10 | 2.79 | 2.70 | 45 | 0.8386 | 0.8166 | +0.0220 | 0.0421 | 42.2% | 19/26 |
| 11 | 2.70 | 2.62 | 45 | 1.4649 | 1.4677 | -0.0028 | 0.0289 | 40.0% | 18/27 |
| 12 | 2.62 | 2.55 | 44 | 1.3107 | 1.3205 | -0.0098 | 0.0299 | 45.5% | 20/24 |
| 13 | 2.55 | 2.49 | 47 | 1.4068 | 1.4238 | -0.0169 | 0.0299 | 36.2% | 17/30 |
| 14 | 2.49 | 2.43 | 43 | 1.1857 | 1.1675 | +0.0182 | 0.0281 | 46.5% | 20/23 |
| 15 | 2.43 | 2.38 | 43 | 1.7405 | 1.7906 | -0.0501 | 0.0310 | 39.5% | 17/26 |
| 16 | 2.38 | 2.33 | 44 | 1.4157 | 1.4547 | -0.0389 | 0.0337 | 27.3% | 12/32 |
| 17 | 2.32 | 2.28 | 47 | 1.8088 | 1.8118 | -0.0030 | 0.0290 | 46.8% | 22/25 |
| 18 | 2.28 | 2.24 | 48 | 1.8633 | 1.8623 | +0.0009 | 0.0284 | 47.9% | 23/25 |
| 19 | 2.24 | 2.20 | 44 | 1.9094 | 1.9208 | -0.0113 | 0.0163 | 43.2% | 19/25 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (-6, 2, 2) | 10.39 | 21050.1 | 765.0 | 27.5 | 154.3 | 166.3 | -0.896 | 1.355 | -2.251 |
| (-3, 5, 11) | 4.26 | 10860.2 | 1340.0 | 8.1 | 59.3 | 56.3 | 5.953 | 4.376 | +1.577 |
| (26, 4, 2) | 3.04 | 6607.7 | 2215.0 | 3.0 | 87.0 | 111.7 | 0.277 | 1.559 | -1.282 |
| (24, 0, 10) | 3.02 | 49740.7 | 6740.0 | 7.4 | 192.8 | 173.7 | 2.407 | 3.672 | -1.265 |
| (-20, 2, 14) | 2.98 | 21187.8 | 4475.0 | 4.7 | 129.8 | 110.7 | 1.005 | 2.174 | -1.169 |
| (-2, 2, 5) | 9.64 | 5606.6 | 1410.0 | 4.0 | 51.5 | 62.5 | -0.467 | -1.629 | +1.162 |
| (13, 5, 7) | 4.16 | 14029.5 | 2660.0 | 5.3 | 80.5 | 89.2 | 2.178 | 1.089 | +1.090 |
| (8, 0, 12) | 4.47 | 9414.3 | 3940.0 | 2.4 | 102.6 | 130.2 | -0.490 | 0.541 | -1.030 |
| (6, 0, 7) | 7.24 | 3275.8 | 1265.0 | 2.6 | 22.1 | 34.9 | -0.236 | -1.259 | +1.022 |
| (17, 3, 3) | 4.47 | -400.1 | 1665.0 | -0.2 | 44.3 | 58.6 | -0.478 | 0.474 | -0.951 |
| (6, 2, 2) | 10.31 | 11568.0 | 515.0 | 22.5 | 111.1 | 117.8 | -2.249 | -1.305 | -0.944 |
| (-7, 11, 10) | 2.76 | 2373.7 | 1395.0 | 1.7 | 60.7 | 73.1 | 0.110 | 1.046 | -0.936 |
| (8, 12, 16) | 2.25 | -1066.5 | 1855.0 | -0.6 | 64.9 | 58.9 | 4.214 | 3.290 | +0.924 |
| (-30, 2, 4) | 2.75 | 33.8 | 1710.0 | 0.0 | 49.7 | 22.7 | 0.843 | -0.047 | +0.891 |
| (14, 6, 7) | 3.77 | -4678.0 | 2005.0 | -2.3 | 21.3 | 38.7 | 1.902 | 2.782 | -0.880 |
| (-3, 5, 24) | 2.34 | -578.6 | 2075.0 | -0.3 | 39.1 | 56.3 | 1.400 | 2.265 | -0.865 |
| (-11, 7, 8) | 3.69 | -1111.2 | 6545.0 | -0.2 | 92.0 | 113.0 | 1.316 | 2.162 | -0.846 |
| (8, 6, 11) | 3.72 | 4015.8 | 1905.0 | 2.1 | 27.2 | 44.3 | 0.360 | -0.475 | +0.836 |
| (2, 14, 11) | 2.28 | -1238.3 | 1375.0 | -0.9 | 46.2 | 36.2 | 3.027 | 2.199 | +0.828 |
| (13, 11, 0) | 2.88 | 3678.5 | 1265.0 | 2.9 | 62.1 | 79.8 | -0.085 | 0.740 | -0.824 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `deposited` | 73.06 | 0.7118 | 8.316639414194936 | 2.2151127829374033 | 1.000 |
| `recovered_i` | 74.50 | 0.7084 | 9.06805272058194 | 2.732101238260012 | 1.000 |