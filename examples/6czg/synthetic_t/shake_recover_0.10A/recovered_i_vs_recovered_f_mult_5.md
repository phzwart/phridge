# Held-Out Log-Likelihood Comparison: `recovered_i` vs `recovered_f`

## 1. Executive Summary

- **Scored Audit Reflections (|A|)**: 900 (audit set)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `0.0272` nats/refl
- **Model B NLL**: `0.0271` nats/refl
- **Difference (Gain $\Delta$)**: `+0.0001` nats/refl (`-0.0001` nats NLL reduction)
- **Estimated Log Bayes Factor**: `+0.10` nats
- **Uncertainty**: Bootstrap SE = `0.0010` (95% CI: `[-0.0019, +0.0020]`) | Naive SE = `0.0010` (Ratio: `1.05`x)
- **Win Fraction $P(d_h > 0)$**: `51.1%` (460 wins, 440 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `5.27e-01`
- **Wilcoxon Signed-Rank $p$-value**: `4.25e-01`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `-0.0001` | `+0.0000` | `-0.0001` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `-0.0001` | `+0.0000` | `-0.0001` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`recovered_f`) | Model B (`recovered_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `+0.0026` | `+0.0027` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `0.0152` | `0.0157` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `+0.0120` | `+0.0114` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -1.1116 | -1.1097 | -0.0018 | 0.0032 | 44.2% | 19/24 |
| 1 | 6.10 | 4.86 | 46 | 0.4356 | 0.4426 | -0.0070 | 0.0063 | 41.3% | 19/27 |
| 2 | 4.83 | 4.21 | 45 | -0.2124 | -0.2134 | +0.0010 | 0.0037 | 53.3% | 24/21 |
| 3 | 4.20 | 3.81 | 46 | 0.0843 | 0.0869 | -0.0026 | 0.0060 | 50.0% | 23/23 |
| 4 | 3.80 | 3.53 | 46 | 0.2902 | 0.2907 | -0.0005 | 0.0078 | 47.8% | 22/24 |
| 5 | 3.53 | 3.32 | 46 | 0.4276 | 0.4380 | -0.0105 | 0.0056 | 39.1% | 18/28 |
| 6 | 3.31 | 3.16 | 44 | -0.0985 | -0.1058 | +0.0073 | 0.0043 | 59.1% | 26/18 |
| 7 | 3.14 | 3.02 | 45 | 0.4505 | 0.4543 | -0.0038 | 0.0048 | 42.2% | 19/26 |
| 8 | 3.01 | 2.89 | 46 | -0.1021 | -0.0964 | -0.0056 | 0.0031 | 41.3% | 19/27 |
| 9 | 2.88 | 2.79 | 43 | -0.0817 | -0.0830 | +0.0012 | 0.0032 | 44.2% | 19/24 |
| 10 | 2.79 | 2.70 | 45 | -0.4282 | -0.4268 | -0.0014 | 0.0022 | 46.7% | 21/24 |
| 11 | 2.70 | 2.62 | 45 | -0.0193 | -0.0221 | +0.0029 | 0.0030 | 55.6% | 25/20 |
| 12 | 2.62 | 2.55 | 44 | -0.1177 | -0.1240 | +0.0063 | 0.0034 | 61.4% | 27/17 |
| 13 | 2.55 | 2.49 | 47 | 0.0007 | -0.0011 | +0.0018 | 0.0025 | 57.4% | 27/20 |
| 14 | 2.49 | 2.43 | 43 | -0.3932 | -0.3935 | +0.0002 | 0.0027 | 53.5% | 23/20 |
| 15 | 2.43 | 2.38 | 43 | 0.2795 | 0.2715 | +0.0079 | 0.0036 | 65.1% | 28/15 |
| 16 | 2.38 | 2.33 | 44 | 0.1678 | 0.1668 | +0.0010 | 0.0036 | 47.7% | 21/23 |
| 17 | 2.32 | 2.28 | 47 | 0.1472 | 0.1475 | -0.0004 | 0.0023 | 59.6% | 28/19 |
| 18 | 2.28 | 2.24 | 48 | 0.4684 | 0.4649 | +0.0035 | 0.0034 | 62.5% | 30/18 |
| 19 | 2.24 | 2.20 | 44 | 0.2401 | 0.2364 | +0.0037 | 0.0043 | 50.0% | 22/22 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (-19, 1, 10) | 3.58 | 56054.6 | 2304.5 | 24.3 | 285.0 | 287.6 | 2.292 | 2.515 | -0.223 |
| (-16, 4, 12) | 3.38 | 73827.9 | 3058.0 | 24.1 | 326.5 | 327.6 | 4.005 | 4.154 | -0.149 |
| (9, 5, 4) | 5.27 | 118950.6 | 7195.0 | 16.5 | 420.3 | 422.0 | 3.488 | 3.624 | -0.136 |
| (-9, 5, 4) | 5.30 | 68282.5 | 5233.5 | 13.0 | 331.5 | 329.7 | 3.093 | 2.965 | +0.129 |
| (-23, 1, 4) | 3.58 | 6622.4 | 1522.5 | 4.3 | 115.0 | 117.1 | 0.658 | 0.785 | -0.128 |
| (5, 1, 15) | 3.83 | 33939.8 | 1473.0 | 23.0 | 215.1 | 217.1 | 1.265 | 1.387 | -0.122 |
| (-11, 1, 12) | 4.17 | 38259.8 | 2698.0 | 14.2 | 242.1 | 243.7 | 2.061 | 2.182 | -0.120 |
| (-37, 3, 6) | 2.20 | 1872.0 | 306.5 | 6.1 | 64.3 | 63.9 | 3.159 | 3.044 | +0.115 |
| (17, 9, 12) | 2.61 | 2375.6 | 471.5 | 5.0 | 68.1 | 67.1 | 0.870 | 0.756 | +0.114 |
| (8, 4, 14) | 3.59 | 8708.4 | 1320.5 | 6.6 | 119.5 | 117.1 | 0.341 | 0.233 | +0.108 |
| (-15, 7, 4) | 3.66 | 5988.5 | 920.5 | 6.5 | 108.0 | 106.0 | 0.372 | 0.265 | +0.107 |
| (-27, 1, 2) | 3.12 | 27714.6 | 2000.5 | 13.9 | 199.8 | 200.8 | 2.088 | 2.193 | -0.105 |
| (3, 7, 24) | 2.22 | 3210.3 | 189.0 | 17.0 | 47.1 | 47.7 | 1.403 | 1.301 | +0.102 |
| (20, 6, 0) | 3.44 | 37398.5 | 4035.5 | 9.3 | 229.1 | 230.6 | 1.836 | 1.937 | -0.101 |
| (19, 1, 10) | 3.53 | 14493.7 | 830.0 | 17.5 | 150.0 | 148.6 | 1.169 | 1.071 | +0.098 |
| (2, 2, 14) | 4.11 | 121519.8 | 5705.5 | 21.3 | 432.6 | 431.8 | 5.671 | 5.575 | +0.096 |
| (-25, 11, 8) | 2.23 | 6437.4 | 408.0 | 15.8 | 91.2 | 91.9 | 1.270 | 1.359 | -0.089 |
| (16, 8, 17) | 2.43 | 56.3 | 242.0 | 0.2 | 27.4 | 26.8 | 0.350 | 0.261 | +0.089 |
| (-12, 10, 2) | 3.14 | 17876.0 | 1861.0 | 9.6 | 167.4 | 166.5 | 1.978 | 1.891 | +0.086 |
| (-9, 1, 17) | 3.29 | 39775.2 | 1326.0 | 30.0 | 240.4 | 239.5 | 2.187 | 2.100 | +0.086 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `recovered_f` | 11.90 | 0.9934 | 6.864132184791917 | 2.3333425651996667 | 1.000 |
| `recovered_i` | 11.83 | 0.9935 | 6.877894534466397 | 2.337291788101025 | 1.000 |