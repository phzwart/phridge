# Held-Out Log-Likelihood Comparison: `rerefined_deposited_i` vs `rerefined_deposited_f`

## 1. Executive Summary

- **Scored Audit Reflections (|A|)**: 900 (audit set)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `0.5155` nats/refl
- **Model B NLL**: `0.5155` nats/refl
- **Difference (Gain $\Delta$)**: `-0.0000` nats/refl (`+0.0000` nats NLL reduction)
- **Estimated Log Bayes Factor**: `-0.02` nats
- **Uncertainty**: Bootstrap SE = `0.0002` (95% CI: `[-0.0005, +0.0004]`) | Naive SE = `0.0003` (Ratio: `0.85`x)
- **Win Fraction $P(d_h > 0)$**: `49.6%` (446 wins, 454 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `8.16e-01`
- **Wilcoxon Signed-Rank $p$-value**: `2.07e-01`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.0000` | `-0.0000` | `+0.0000` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.0000` | `-0.0000` | `+0.0000` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`rerefined_deposited_f`) | Model B (`rerefined_deposited_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0057` | `-0.0057` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `0.4771` | `0.4771` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `+0.0383` | `+0.0384` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -0.6523 | -0.6518 | -0.0006 | 0.0013 | 46.5% | 20/23 |
| 1 | 6.10 | 4.86 | 46 | 0.5556 | 0.5578 | -0.0022 | 0.0024 | 43.5% | 20/26 |
| 2 | 4.83 | 4.21 | 45 | 0.1317 | 0.1311 | +0.0006 | 0.0017 | 40.0% | 18/27 |
| 3 | 4.20 | 3.81 | 46 | 0.4899 | 0.4895 | +0.0004 | 0.0012 | 39.1% | 18/28 |
| 4 | 3.80 | 3.53 | 46 | 0.7420 | 0.7417 | +0.0003 | 0.0017 | 37.0% | 17/29 |
| 5 | 3.53 | 3.32 | 46 | 0.9457 | 0.9456 | +0.0001 | 0.0017 | 65.2% | 30/16 |
| 6 | 3.31 | 3.16 | 44 | 0.4563 | 0.4538 | +0.0025 | 0.0013 | 59.1% | 26/18 |
| 7 | 3.14 | 3.02 | 45 | 1.2442 | 1.2456 | -0.0014 | 0.0013 | 42.2% | 19/26 |
| 8 | 3.01 | 2.89 | 46 | 0.3849 | 0.3844 | +0.0005 | 0.0011 | 56.5% | 26/20 |
| 9 | 2.88 | 2.79 | 43 | 0.6415 | 0.6411 | +0.0005 | 0.0009 | 62.8% | 27/16 |
| 10 | 2.79 | 2.70 | 45 | 0.2172 | 0.2178 | -0.0007 | 0.0005 | 31.1% | 14/31 |
| 11 | 2.70 | 2.62 | 45 | 0.5425 | 0.5430 | -0.0005 | 0.0006 | 57.8% | 26/19 |
| 12 | 2.62 | 2.55 | 44 | 0.3504 | 0.3494 | +0.0010 | 0.0006 | 56.8% | 25/19 |
| 13 | 2.55 | 2.49 | 47 | 0.5131 | 0.5134 | -0.0003 | 0.0006 | 57.4% | 27/20 |
| 14 | 2.49 | 2.43 | 43 | 0.2441 | 0.2453 | -0.0011 | 0.0006 | 37.2% | 16/27 |
| 15 | 2.43 | 2.38 | 43 | 0.7122 | 0.7117 | +0.0005 | 0.0004 | 53.5% | 23/20 |
| 16 | 2.38 | 2.33 | 44 | 0.7995 | 0.8002 | -0.0007 | 0.0008 | 40.9% | 18/26 |
| 17 | 2.32 | 2.28 | 47 | 0.9155 | 0.9154 | +0.0002 | 0.0004 | 61.7% | 29/18 |
| 18 | 2.28 | 2.24 | 48 | 0.4109 | 0.4099 | +0.0010 | 0.0007 | 54.2% | 26/22 |
| 19 | 2.24 | 2.20 | 44 | 0.5948 | 0.5952 | -0.0004 | 0.0006 | 47.7% | 21/23 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (2, 6, 0) | 5.83 | 973.3 | 27.6 | 35.3 | 141.9 | 140.6 | 1.900 | 1.841 | +0.059 |
| (20, 6, 0) | 3.44 | 65812.5 | 807.1 | 81.5 | 216.0 | 216.9 | 4.334 | 4.280 | +0.054 |
| (-14, 2, 12) | 3.78 | 28213.2 | 380.7 | 74.1 | 274.1 | 273.3 | 3.299 | 3.249 | +0.050 |
| (-27, 1, 6) | 3.00 | 6769.9 | 112.2 | 60.3 | 141.8 | 142.5 | 2.184 | 2.225 | -0.041 |
| (0, 4, 12) | 4.33 | 31219.3 | 414.4 | 75.3 | 310.8 | 310.2 | 3.941 | 3.905 | +0.036 |
| (-10, 0, 12) | 4.31 | 4234.8 | 122.6 | 34.5 | 207.0 | 206.0 | 2.167 | 2.131 | +0.036 |
| (14, 2, 4) | 5.33 | 126810.0 | 1524.0 | 83.2 | 508.0 | 508.9 | 3.526 | 3.561 | -0.035 |
| (-19, 7, 4) | 3.27 | 65816.4 | 797.1 | 82.6 | 205.3 | 205.8 | 4.198 | 4.164 | +0.034 |
| (11, 3, 0) | 6.46 | 27797.6 | 354.9 | 78.3 | 273.8 | 272.7 | 1.684 | 1.653 | +0.031 |
| (5, 5, 4) | 5.96 | 3031.3 | 48.6 | 62.4 | 142.5 | 143.4 | 1.269 | 1.300 | -0.031 |
| (24, 0, 10) | 3.02 | 5069.1 | 134.8 | 37.6 | 160.9 | 160.4 | 2.612 | 2.582 | +0.030 |
| (14, 4, 2) | 4.92 | 7084.6 | 113.4 | 62.5 | 163.1 | 164.0 | 1.457 | 1.486 | -0.029 |
| (-2, 4, 12) | 4.31 | 14627.8 | 211.0 | 69.3 | 208.7 | 209.6 | 1.824 | 1.853 | -0.029 |
| (17, 3, 10) | 3.61 | 22327.3 | 308.6 | 72.4 | 98.5 | 99.2 | 2.044 | 2.016 | +0.028 |
| (-12, 10, 2) | 3.14 | 30117.1 | 372.2 | 80.9 | 149.2 | 148.7 | 2.757 | 2.785 | -0.028 |
| (-14, 4, 13) | 3.40 | 1031.7 | 44.1 | 23.4 | 105.2 | 104.7 | 1.886 | 1.860 | +0.027 |
| (-15, 1, 4) | 5.26 | 8623.6 | 133.3 | 64.7 | 171.3 | 170.4 | 1.389 | 1.363 | +0.026 |
| (-7, 5, 4) | 5.66 | 15796.7 | 208.9 | 75.6 | 181.2 | 182.7 | 0.961 | 0.986 | -0.025 |
| (-9, 3, 14) | 3.70 | 30998.8 | 426.4 | 72.7 | 253.7 | 254.4 | 2.068 | 2.093 | -0.025 |
| (-5, 5, 4) | 5.99 | 17588.0 | 233.2 | 75.4 | 76.5 | 75.3 | 1.395 | 1.420 | -0.025 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `rerefined_deposited_f` | 42.98 | 0.8875 | 5.035406244011455 | 6.798548242855606 | 1.000 |
| `rerefined_deposited_i` | 42.98 | 0.8876 | 5.033316793443273 | 6.792880528577561 | 1.000 |