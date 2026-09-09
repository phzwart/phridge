# Held-Out Log-Likelihood Comparison: `recovered_i` vs `shaken`

## 1. Executive Summary

- **Scored Audit Reflections (|A|)**: 900 (audit set)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `0.5616` nats/refl
- **Model B NLL**: `0.1749` nats/refl
- **Difference (Gain $\Delta$)**: `+0.3867` nats/refl (`-0.3867` nats NLL reduction)
- **Estimated Log Bayes Factor**: `+348.04` nats
- **Uncertainty**: Bootstrap SE = `0.0283` (95% CI: `[+0.3289, +0.4400]`) | Naive SE = `0.0293` (Ratio: `0.97`x)
- **Win Fraction $P(d_h > 0)$**: `78.1%` (703 wins, 197 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `2.64e-67`
- **Wilcoxon Signed-Rank $p$-value**: `2.28e-57`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `-0.2843` | `-0.1024` | `-0.3867` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `-0.5655` | `+0.1787` | `-0.3867` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`shaken`) | Model B (`recovered_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0019` | `-0.0066` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `0.6570` | `0.1755` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.0954` | `-0.0006` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -0.9420 | -1.3806 | +0.4386 | 0.2250 | 62.8% | 27/16 |
| 1 | 6.10 | 4.86 | 46 | 0.2679 | 0.1257 | +0.1422 | 0.1177 | 67.4% | 31/15 |
| 2 | 4.83 | 4.21 | 45 | -0.1075 | -0.3215 | +0.2139 | 0.1164 | 80.0% | 36/9 |
| 3 | 4.20 | 3.81 | 46 | 0.5170 | 0.0630 | +0.4540 | 0.1187 | 82.6% | 38/8 |
| 4 | 3.80 | 3.53 | 46 | 0.4641 | 0.2671 | +0.1970 | 0.1509 | 69.6% | 32/14 |
| 5 | 3.53 | 3.32 | 46 | 0.9646 | 0.3899 | +0.5747 | 0.1910 | 73.9% | 34/12 |
| 6 | 3.31 | 3.16 | 44 | 0.3040 | 0.0233 | +0.2807 | 0.0762 | 75.0% | 33/11 |
| 7 | 3.14 | 3.02 | 45 | 0.8736 | 0.4988 | +0.3748 | 0.1349 | 75.6% | 34/11 |
| 8 | 3.01 | 2.89 | 46 | 0.4705 | 0.0214 | +0.4491 | 0.0976 | 80.4% | 37/9 |
| 9 | 2.88 | 2.79 | 43 | 0.5796 | 0.1809 | +0.3987 | 0.1044 | 76.7% | 33/10 |
| 10 | 2.79 | 2.70 | 45 | 0.2922 | -0.1701 | +0.4623 | 0.0868 | 86.7% | 39/6 |
| 11 | 2.70 | 2.62 | 45 | 0.7194 | 0.1788 | +0.5406 | 0.1055 | 91.1% | 41/4 |
| 12 | 2.62 | 2.55 | 44 | 0.6266 | 0.0952 | +0.5314 | 0.0882 | 79.5% | 35/9 |
| 13 | 2.55 | 2.49 | 47 | 0.6872 | 0.3013 | +0.3859 | 0.1006 | 74.5% | 35/12 |
| 14 | 2.49 | 2.43 | 43 | 0.4932 | 0.0414 | +0.4518 | 0.0989 | 81.4% | 35/8 |
| 15 | 2.43 | 2.38 | 43 | 0.9925 | 0.6050 | +0.3875 | 0.1361 | 88.4% | 38/5 |
| 16 | 2.38 | 2.33 | 44 | 1.0715 | 0.5258 | +0.5457 | 0.1621 | 79.5% | 35/9 |
| 17 | 2.32 | 2.28 | 47 | 0.9573 | 0.5467 | +0.4105 | 0.0923 | 76.6% | 36/11 |
| 18 | 2.28 | 2.24 | 48 | 0.9283 | 0.7749 | +0.1534 | 0.1442 | 79.2% | 38/10 |
| 19 | 2.24 | 2.20 | 44 | 0.9946 | 0.6265 | +0.3682 | 0.1192 | 81.8% | 36/8 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (3, 1, 0) | 22.09 | 19735.3 | 907.5 | 21.7 | 200.8 | 195.2 | 9.659 | 1.020 | +8.639 |
| (20, 4, 7) | 3.47 | 471.7 | 177.5 | 2.7 | 127.2 | 35.0 | 5.157 | -1.069 | +6.225 |
| (-15, 9, 19) | 2.26 | -986.1 | 181.0 | -5.4 | 37.4 | 12.2 | 7.844 | 13.262 | -5.419 |
| (27, 7, 12) | 2.34 | 13603.7 | 495.5 | 27.5 | 75.1 | 118.4 | 5.793 | 1.328 | +4.465 |
| (-1, 1, 3) | 17.00 | -38.1 | 149.5 | -0.3 | 56.6 | 28.5 | 2.216 | -2.095 | +4.311 |
| (2, 2, 14) | 4.11 | 121519.8 | 5705.5 | 21.3 | 421.9 | 436.5 | 2.466 | 5.968 | -3.502 |
| (15, 7, 4) | 3.64 | 78456.1 | 3885.0 | 20.2 | 211.6 | 288.1 | 4.200 | 0.839 | +3.361 |
| (0, 4, 12) | 4.33 | 73099.5 | 2072.0 | 35.3 | 259.7 | 348.9 | 0.987 | 4.313 | -3.325 |
| (21, 3, 2) | 3.79 | 52173.5 | 4782.0 | 10.9 | 252.0 | 305.0 | 0.956 | 4.178 | -3.222 |
| (-30, 2, 14) | 2.35 | 1474.5 | 210.0 | 7.0 | 95.1 | 40.8 | 3.248 | 0.070 | +3.178 |
| (-18, 4, 10) | 3.43 | 48176.8 | 3216.5 | 15.0 | 238.7 | 287.7 | 1.356 | 4.508 | -3.151 |
| (27, 1, 2) | 3.11 | 66998.6 | 4270.0 | 15.7 | 212.3 | 269.9 | 4.566 | 1.427 | +3.139 |
| (-29, 9, 3) | 2.33 | -225.8 | 139.5 | -1.6 | 63.3 | 20.0 | 5.224 | 2.123 | +3.102 |
| (-20, 2, 14) | 2.98 | 13465.1 | 447.5 | 30.1 | 51.4 | 115.2 | 3.504 | 0.408 | +3.096 |
| (-12, 8, 19) | 2.41 | 5785.3 | 391.0 | 14.8 | 31.3 | 83.5 | 3.849 | 0.792 | +3.057 |
| (9, 3, 14) | 3.66 | 131464.6 | 5549.5 | 23.7 | 361.8 | 441.1 | 1.734 | 4.644 | -2.911 |
| (-24, 2, 2) | 3.45 | 55458.2 | 3850.5 | 14.4 | 181.8 | 244.7 | 3.953 | 1.100 | +2.853 |
| (-14, 6, 18) | 2.62 | 22310.9 | 965.0 | 23.1 | 109.0 | 167.7 | 4.088 | 1.240 | +2.849 |
| (-30, 4, 4) | 2.66 | 14537.4 | 884.0 | 16.4 | 78.9 | 129.6 | 3.713 | 0.965 | +2.749 |
| (35, 1, 10) | 2.23 | 8254.6 | 542.0 | 15.2 | 64.9 | 102.8 | 3.950 | 1.319 | +2.631 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `shaken` | 36.30 | 0.9572 | 3.6192316582631445 | 1.2079339209018956 | 1.000 |
| `recovered_i` | 17.89 | 0.9877 | 5.477802861520732 | 1.8460569542278205 | 1.000 |