# Held-Out Log-Likelihood Comparison: `recovered_i` vs `deposited`

## 1. Executive Summary

- **Scored Audit Reflections (|A|)**: 900 (audit set)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `-0.3427` nats/refl
- **Model B NLL**: `-0.1147` nats/refl
- **Difference (Gain $\Delta$)**: `-0.2280` nats/refl (`+0.2280` nats NLL reduction)
- **Estimated Log Bayes Factor**: `-205.21` nats
- **Uncertainty**: Bootstrap SE = `0.0232` (95% CI: `[-0.2755, -0.1846]`) | Naive SE = `0.0137` (Ratio: `1.69`x)
- **Win Fraction $P(d_h > 0)$**: `19.8%` (178 wins, 722 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `2.51e-78`
- **Wilcoxon Signed-Rank $p$-value**: `1.67e-68`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.3245` | `-0.0965` | `+0.2280` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.1747` | `+0.0533` | `+0.2280` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`deposited`) | Model B (`recovered_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0037` | `-0.0087` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `-0.2392` | `-0.1107` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.1035` | `-0.0039` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -1.4315 | -1.1354 | -0.2961 | 0.0297 | 4.7% | 2/41 |
| 1 | 6.10 | 4.86 | 46 | 0.0711 | 0.4908 | -0.4197 | 0.0606 | 4.3% | 2/44 |
| 2 | 4.83 | 4.21 | 45 | -0.6230 | -0.2704 | -0.3526 | 0.0467 | 2.2% | 1/44 |
| 3 | 4.20 | 3.81 | 46 | -0.1966 | 0.0821 | -0.2788 | 0.0595 | 13.0% | 6/40 |
| 4 | 3.80 | 3.53 | 46 | -0.1032 | 0.3070 | -0.4102 | 0.0735 | 6.5% | 3/43 |
| 5 | 3.53 | 3.32 | 46 | 0.1441 | 0.3815 | -0.2374 | 0.0644 | 26.1% | 12/34 |
| 6 | 3.31 | 3.16 | 44 | -0.4083 | -0.1166 | -0.2917 | 0.0513 | 18.2% | 8/36 |
| 7 | 3.14 | 3.02 | 45 | 0.1935 | 0.4014 | -0.2079 | 0.0867 | 22.2% | 10/35 |
| 8 | 3.01 | 2.89 | 46 | -0.4553 | -0.1631 | -0.2922 | 0.0575 | 17.4% | 8/38 |
| 9 | 2.88 | 2.79 | 43 | -0.4898 | -0.1650 | -0.3248 | 0.0576 | 9.3% | 4/39 |
| 10 | 2.79 | 2.70 | 45 | -0.7168 | -0.5319 | -0.1849 | 0.0432 | 13.3% | 6/39 |
| 11 | 2.70 | 2.62 | 45 | -0.3674 | -0.1542 | -0.2132 | 0.0496 | 24.4% | 11/34 |
| 12 | 2.62 | 2.55 | 44 | -0.4573 | -0.3430 | -0.1143 | 0.0510 | 38.6% | 17/27 |
| 13 | 2.55 | 2.49 | 47 | -0.3809 | -0.1687 | -0.2122 | 0.0478 | 21.3% | 10/37 |
| 14 | 2.49 | 2.43 | 43 | -0.6402 | -0.5181 | -0.1221 | 0.0418 | 32.6% | 14/29 |
| 15 | 2.43 | 2.38 | 43 | -0.0180 | 0.1051 | -0.1231 | 0.0683 | 30.2% | 13/30 |
| 16 | 2.38 | 2.33 | 44 | -0.3516 | -0.2058 | -0.1458 | 0.0767 | 31.8% | 14/30 |
| 17 | 2.32 | 2.28 | 47 | -0.1726 | -0.0982 | -0.0744 | 0.0688 | 27.7% | 13/34 |
| 18 | 2.28 | 2.24 | 48 | -0.3085 | -0.1486 | -0.1599 | 0.0548 | 22.9% | 11/37 |
| 19 | 2.24 | 2.20 | 44 | -0.2307 | -0.1389 | -0.0918 | 0.0721 | 29.5% | 13/31 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (9, 3, 14) | 3.66 | 132444.0 | 1109.9 | 119.3 | 413.5 | 446.1 | 3.779 | 5.945 | -2.167 |
| (2, 2, 14) | 4.11 | 122539.2 | 1141.1 | 107.4 | 400.6 | 434.9 | 4.566 | 6.648 | -2.082 |
| (1, 7, 2) | 4.96 | 78873.8 | 1370.1 | 57.6 | 319.2 | 354.7 | 2.108 | 4.088 | -1.980 |
| (20, 2, 14) | 2.94 | 7282.9 | 148.1 | 49.2 | 98.9 | 119.6 | 0.223 | 2.124 | -1.900 |
| (-12, 6, 14) | 3.11 | 10869.7 | 108.5 | 100.2 | 117.3 | 138.3 | 0.429 | 2.185 | -1.756 |
| (27, 7, 12) | 2.34 | 12520.4 | 99.1 | 126.3 | 127.1 | 118.1 | 2.595 | 0.846 | +1.749 |
| (14, 2, 4) | 5.33 | 191022.5 | 1524.0 | 125.3 | 496.8 | 534.9 | 4.500 | 6.200 | -1.700 |
| (30, 2, 14) | 2.32 | 10789.6 | 173.0 | 62.4 | 119.2 | 113.5 | 2.741 | 1.067 | +1.675 |
| (-12, 4, 8) | 4.46 | 74634.5 | 1028.6 | 72.6 | 313.4 | 346.0 | 1.598 | 3.190 | -1.592 |
| (27, 1, 2) | 3.11 | 66716.5 | 854.0 | 78.1 | 291.6 | 290.9 | 3.791 | 2.225 | +1.567 |
| (16, 8, 17) | 2.43 | 164.0 | 48.4 | 3.4 | 12.8 | 26.0 | -1.779 | -0.276 | -1.503 |
| (21, 3, 2) | 3.79 | 56287.7 | 956.4 | 58.9 | 271.5 | 297.9 | 1.661 | 3.147 | -1.486 |
| (26, 4, 2) | 3.04 | 6856.0 | 44.3 | 154.8 | 94.2 | 112.1 | 0.169 | 1.652 | -1.483 |
| (0, 4, 12) | 4.33 | 80152.9 | 414.4 | 193.4 | 322.3 | 352.3 | 1.643 | 3.074 | -1.431 |
| (-12, 8, 6) | 3.51 | 50266.4 | 424.9 | 118.3 | 256.7 | 279.5 | 2.703 | 4.126 | -1.422 |
| (-13, 11, 2) | 2.87 | 1897.8 | 55.9 | 33.9 | 49.8 | 65.1 | -0.612 | 0.703 | -1.315 |
| (7, 1, 22) | 2.63 | 3182.2 | 67.1 | 47.4 | 64.2 | 76.7 | -0.016 | 1.268 | -1.284 |
| (-7, 11, 10) | 2.76 | 3379.5 | 27.9 | 121.1 | 65.9 | 80.3 | -0.317 | 0.956 | -1.273 |
| (23, 3, 8) | 3.17 | 4655.6 | 64.4 | 72.3 | 76.7 | 96.2 | -0.522 | 0.731 | -1.253 |
| (20, 8, 0) | 3.06 | 34357.3 | 929.6 | 37.0 | 211.0 | 204.5 | 2.417 | 1.176 | +1.241 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `deposited` | 1.51 | 0.9998 | 199.18862156908295 | 1000.0 | 1.000 |
| `recovered_i` | 9.02 | 0.9975 | 199.16555110781067 | 1000.0 | 1.000 |